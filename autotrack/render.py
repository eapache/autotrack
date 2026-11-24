"""Score-conditioned SoulX-Singer inference, with content-addressed phrase caching."""
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from .score import VOWELS
from . import default_threads

SR = 44100
MODEL_SR = 24000
FRAME_SECONDS = .02


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


def target_metadata(groups):
    """SoulX uses one space-separated entry per note, including internal rests."""
    cursor = groups[0]['notes'][0][0]
    notes = []
    for group in groups:
        if group['notes'][-1][1] - group['notes'][0][0] > 30:
            raise ValueError('Syllable exceeds 30 seconds')
        phones = group['phonemes']
        token = 'en_' + '-'.join(map(english_phone, phones))
        for index, (start, end, midi) in enumerate(group['notes']):
            if start - cursor > 1e-7:
                notes.append(('<SP>', start - cursor, 0, 1))
            # The upstream aligner inserts BOW, SEP and EOW around the phones.
            if end - start + 1e-9 < (len(phones) + 3) * FRAME_SECONDS:
                raise ValueError(f"Note too short for SoulX phoneme alignment: {group.get('lyric', token)}")
            notes.append((token, end - start, midi, 3 if index else 2))
            cursor = end
    fields = ('phoneme', 'duration', 'note_pitch', 'note_type')
    return {name: ' '.join(map(str, values)) for name, values in zip(fields, zip(*notes))}


class Renderer:
    def __init__(self, root, cache, threads=None, *, python=None, checkpoint=None,
                 prompt_audio=None, prompt_metadata=None, steps=32, device='cpu',
                 seed=42, phrase_seconds=13.):
        if not re.fullmatch(r'cpu|cuda(?::[0-9]+)?', device):
            raise ValueError('device must be cpu, cuda or cuda:N')
        self.root = Path(root).resolve()
        self.cache = Path(cache)
        # Existing installs can still use their dedicated SoulX interpreter.
        legacy = self.root / '.venv/bin/python'
        if python is None and importlib.util.find_spec('torch') is None:
            if not legacy.is_file():
                raise ValueError("Install Autotrack with the 'singer' extra to render audio.")
            python = legacy
        self.python = shutil.which(str(python or sys.executable))
        if not self.python:
            raise ValueError('Cannot find the requested SoulX Python interpreter.')
        self.python = os.path.abspath(self.python)
        if bool(prompt_audio) != bool(prompt_metadata):
            raise ValueError('Supply --prompt-audio and --prompt-metadata together')
        files = {
            'checkpoint': Path(checkpoint or self.root / 'pretrained_models/SoulX-Singer/model.pt'),
            'prompt_audio': Path(prompt_audio or self.root / 'example/audio/en_prompt.mp3'),
            'prompt_metadata': Path(prompt_metadata or self.root / 'example/audio/en_prompt.json'),
            'config': self.root / 'soulxsinger/config/soulxsinger.yaml',
            'phoneset': self.root / 'soulxsinger/utils/phoneme/phone_set.json',
        }
        for path in [self.root / 'cli/inference.py', *files.values()]:
            if not path.is_file():
                raise ValueError(f'Missing SoulX asset: {path}. Run autotrack download-voice or check the supplied path.')
        self.files = {k: str(v.resolve()) for k, v in files.items()}
        self.phones = set(json.loads(files['phoneset'].read_text()))
        threads = default_threads() if threads is None else threads
        if threads < 1 or steps < 1 or not 1 <= phrase_seconds <= 60:
            raise ValueError('threads and steps must be positive and phrase-seconds must be between 1 and 60')
        self.phrase_seconds = phrase_seconds
        self.settings = dict(engine='SoulX-Singer', version=1, steps=steps, device=device,
                             seed=seed, threads=threads, phrase_seconds=phrase_seconds)
        self.fingerprint = {k: digest_file(v) for k, v in files.items()}
        code = sorted((self.root / 'soulxsinger').rglob('*.py')) + [self.root / 'cli/inference.py']
        self.fingerprint['source'] = hashlib.sha256(''.join(digest_file(p) for p in code).encode()).hexdigest()
        self.settings['torch'] = subprocess.check_output(
            [self.python, '-c', 'from importlib.metadata import version; print(version("torch"))'],
            text=True).strip()
        self.cache.mkdir(parents=True, exist_ok=True)
        self.worker = None
        self.hits = self.misses = 0

    def close(self):
        if self.worker:
            self.worker.stdin.close()
            self.worker.wait()
            self.worker.stdout.close()

    def check_phones(self, score):
        unknown = {p for groups in score['parts'].values() for g in groups for p in g['phonemes']
                   if 'en_' + english_phone(p) not in self.phones}
        if unknown:
            raise ValueError(f'Unsupported SoulX phonemes: {", ".join(sorted(unknown))}')

    def generate(self, meta, path):
        if self.worker is None:
            self.worker = subprocess.Popen(
                [self.python, str(Path(__file__).with_name('_soulx_worker.py').resolve())],
                cwd=self.root, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            self.worker.stdin.write(json.dumps({**self.files, **self.settings}) + '\n')
        self.worker.stdin.write(json.dumps([meta, str(path)]) + '\n')
        self.worker.stdin.flush()
        if self.worker.stdout.readline().strip() != 'done':
            raise ValueError('SoulX inference failed. See the worker diagnostic above.')

    def chunk(self, groups):
        meta = target_metadata(groups)
        start, end = groups[0]['notes'][0][0], groups[-1]['notes'][-1][1]
        length = round(end * SR) - round(start * SR)
        key = hashlib.sha256(json.dumps([self.fingerprint, self.settings, meta, length], sort_keys=True).encode()).hexdigest()
        path = self.cache / (key + '.wav')
        if path.exists():
            try:
                wave, rate = sf.read(path)
                if rate == SR and wave.shape == (length,) and np.isfinite(wave).all():
                    self.hits += 1
                    return start, wave
            except sf.LibsndfileError:
                pass
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / 'phrase.wav'
            self.generate(meta, raw)
            wave, rate = sf.read(raw)
        if rate != MODEL_SR or wave.ndim != 1 or not len(wave) or not np.isfinite(wave).all():
            raise ValueError('Invalid SoulX audio')
        wave = resample_poly(wave, 147, 80)
        if abs(len(wave) - length) > SR * FRAME_SECONDS + 2:
            raise ValueError('SoulX output duration differs from the score by more than one frame')
        wave = np.pad(wave[:length], (0, max(0, length - len(wave))))
        fade = min(round(.012 * SR), length // 3)
        if fade:
            wave[:fade] *= np.linspace(0, 1, fade)
            wave[-fade:] *= np.linspace(1, 0, fade)
        atomic_audio(path, wave)
        self.misses += 1
        return start, wave

    def part(self, groups, length, label):
        chunks, current = [], []
        for group in groups:
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
