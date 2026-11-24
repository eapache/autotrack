"""Local SoulX-Singer inference, with content-addressed phrase caching."""
import hashlib
import json
import os
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd
from .score import VOWELS
from . import default_threads

SR = 44100
MODEL_FILES = Path(__file__).with_name('soulx')
EFFORT_STEPS = {'low': 16, 'standard': 32, 'high': 64, 'max': 128}


def select_device(requested='auto'):
    import torch
    if requested == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable in this PyTorch installation')
    return requested


def digest_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_audio(path, samples, subtype='FLOAT'):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(suffix='.wav', dir=path.parent)
    os.close(fd)
    try:
        sf.write(temporary, samples, SR, subtype=subtype)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def english_phone(phone):
    if phone == 'ax':
        return 'AH0'
    return phone.upper() + ('1' if phone in VOWELS else '')


def check_note_duration(group, part):
    # 20 ms per frame. Each note carries its phones and BOW/SEP/EOW markers.
    for start, end, _ in group['notes']:
        if (end - start) * 50 + 1e-7 < len(group['phonemes']) + 3:
            raise ValueError(f'{part}: note too short for its SoulX phonemes')


def metadata(groups):
    """Convert timed syllables to SoulX note tokens; preserve melismas and rests."""
    start = groups[0]['notes'][0][0]
    cursor = start
    rows = []
    for group in groups:
        token = 'en_' + '-'.join(english_phone(p) for p in group['phonemes'])
        for i, (a, b, midi) in enumerate(group['notes']):
            if a - cursor > 1e-7:
                rows.append((a - cursor, '<SP>', 0, 1))
            rows.append((b - a, token, midi, 2 if i == 0 else 3))
            cursor = b
    return {'index': 'phrase', 'language': 'English', 'time': [0, (cursor-start)*1000],
            'duration': ' '.join(str(r[0]) for r in rows),
            'phoneme': ' '.join(r[1] for r in rows),
            'note_pitch': ' '.join(str(r[2]) for r in rows),
            'note_type': ' '.join(str(r[3]) for r in rows)}


class Renderer:
    def __init__(self, voice, cache, threads=None, effort='standard', seed=42,
                 phrase_seconds=13., device='auto'):
        import torch
        device = select_device(device)
        if not np.isfinite(phrase_seconds) or not 1 <= phrase_seconds <= 60:
            raise ValueError('phrase-seconds must be between 1 and 60')
        threads = default_threads() if threads is None else threads
        if threads < 1:
            raise ValueError('threads must be positive')
        self.root = Path(voice)
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.phrase_seconds = phrase_seconds
        self.settings = dict(version=3, effort=effort, steps=EFFORT_STEPS[effort], seed=seed,
                             device=device, torch=torch.__version__, threads=threads,
                             phrase_seconds=phrase_seconds)
        if device == 'cuda':
            self.settings['gpu'] = torch.cuda.get_device_name()
        self.fingerprint = {}
        for name in ('model.pt', 'en_prompt.wav', 'en_prompt.json'):
            path = self.root / name
            if not path.is_file():
                raise ValueError(f'Missing {path}. Run autotrack download-voice or set --voice.')
            self.fingerprint[name] = digest_file(path)
        self.phones = set(json.loads((MODEL_FILES / 'phone_set.json').read_text()))
        self.model = None
        self.hits = self.misses = 0

    def check_phones(self, score):
        for groups in score['parts'].values():
            for g in groups:
                for p in g['phonemes']:
                    if 'en_' + english_phone(p) not in self.phones:
                        raise ValueError(f'Unsupported SoulX phoneme: {p}')

    def sessions(self):
        if self.model is not None:
            return
        import torch
        from .soulx.model import load_model
        from .soulx.processor import DataProcessor

        device = self.settings['device']
        torch.set_num_threads(self.settings['threads'])
        self.processor = DataProcessor(hop_size=480, sample_rate=24000, device=device,
                                       phoneset_path=MODEL_FILES / 'phone_set.json')
        self.model = load_model(self.root / 'model.pt', device)
        prompt = json.loads((self.root / 'en_prompt.json').read_text())
        if not prompt:
            raise ValueError('Prompt metadata is empty')
        self.prompt = self.processor.process(prompt[0], str(self.root / 'en_prompt.wav'))

    def chunk(self, groups):
        key = hashlib.sha256(json.dumps([self.fingerprint, self.settings, groups], sort_keys=True).encode()).hexdigest()
        path = self.cache / (key + '.wav')
        start, end = groups[0]['notes'][0][0], groups[-1]['notes'][-1][1]
        length = round(end * SR) - round(start * SR)
        if path.exists():
            try:
                wave, rate = sf.read(path)
                if rate == SR and wave.ndim == 1 and len(wave) == length and np.isfinite(wave).all():
                    self.hits += 1
                    return start, wave
            except (RuntimeError, sf.LibsndfileError):
                pass
        self.sessions()
        import torch
        torch.manual_seed(self.settings['seed'])
        target = self.processor.process(metadata(groups), None)
        with torch.inference_mode():
            wave = self.model(self.prompt, target, self.settings['steps']).flatten().cpu().numpy()
        rate = self.processor.sample_rate
        divisor = gcd(SR, rate)
        wave = resample_poly(wave, SR // divisor, rate // divisor)
        if not len(wave) or not np.isfinite(wave).all():
            raise ValueError('Invalid SoulX model output')
        wave = np.pad(wave[:length], (0, max(0, length-len(wave))))
        atomic_audio(path, wave)
        self.misses += 1
        return start, wave

    def part(self, groups, length, label):
        chunks, current = [], []
        for group in groups:
            if group['notes'][-1][1] - group['notes'][0][0] > 30:
                raise ValueError(f'{label}: syllable exceeds 30 seconds; split it into shorter phrases')
            if current and (group['notes'][-1][1] - current[0]['notes'][0][0] > self.phrase_seconds or group['notes'][0][0] - current[-1]['notes'][-1][1] > .4):
                chunks.append(current)
                current = []
            current.append(group)
        if current:
            chunks.append(current)
        output = np.zeros(length)
        for index, chunk in enumerate(chunks, 1):
            start, wave = self.chunk(chunk)
            offset = round(start * SR)
            output[offset:offset + len(wave)] += wave
            print(f'{label}: phrase {index}/{len(chunks)}', flush=True)
        rms = np.sqrt(np.mean(output ** 2))
        peak = np.max(np.abs(output))
        if rms < 1e-7:
            raise ValueError(f'{label}: rendered silence')
        output *= min(.12 / rms, .85 / peak)
        return output
