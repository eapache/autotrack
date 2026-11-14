"""Local TIGER DiffSinger inference, with content-addressed phrase caching."""
import hashlib
import json
import os
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.ndimage import gaussian_filter1d
from .score import VOWELS
from . import default_threads

SR = 44100
HOP = 512
DT = HOP / SR
RENDER_VERSION = 1
EFFORT_SPEEDUPS = {'low': 40, 'standard': 20, 'high': 10, 'max': 5}


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


def frame_durations(phones, total):
    """Exact frame budget; every phoneme gets at least one frame."""
    if total < len(phones):
        raise ValueError(f'Syllable too short for {len(phones)} phonemes ({total} audio frames)')
    vowels = [i for i, p in enumerate(phones) if p in VOWELS]
    if not vowels:
        raise ValueError('A sung syllable needs at least one vowel')
    weights = np.array([0 if p in VOWELS else .065 if p in ('s', 'sh', 'f', 'th', 'ch') else .045 for p in phones], dtype=float)
    weights *= min(1., total * DT * .4 / max(weights.sum(), .001))
    remaining = total * DT - weights.sum()
    for j, i in enumerate(vowels):
        weights[i] = remaining * ((.3 if j == 0 else .7) if len(vowels) == 2 else 1 / len(vowels))
    # Apportion the frames remaining after the one-frame minimum.
    exact = weights / weights.sum() * (total - len(phones))
    result = np.floor(exact).astype(np.int64) + 1
    for i in np.argsort(-(exact - np.floor(exact)), kind='stable')[:total - int(result.sum())]:
        result[i] += 1
    return result.tolist()


class Renderer:
    def __init__(self, voice, cache, threads=None, effort='standard', style='fresh', seed=42, phrase_seconds=13., gender=0., velocity=1.):
        import onnxruntime as ort
        self.ort = ort
        threads = default_threads() if threads is None else threads
        if not np.isfinite(phrase_seconds) or not 1 <= phrase_seconds <= 60:
            raise ValueError('phrase-seconds must be between 1 and 60')
        if not np.isfinite(gender) or not -5 <= gender <= 5:
            raise ValueError('gender must be between -5 and 5 (timbre shift, not note transposition)')
        if not np.isfinite(velocity) or not .5 <= velocity <= 2:
            raise ValueError('velocity must be between 0.5 and 2')
        self.phrase_seconds = phrase_seconds
        self.root = Path(voice)
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        if effort not in EFFORT_SPEEDUPS:
            raise ValueError('effort must be low, standard, high or max')
        speedup = EFFORT_SPEEDUPS[effort]
        if threads < 1:
            raise ValueError('threads must be positive')
        if style not in ('fresh', 'disco', 'electric', 'vinyl', 'cetera', 'mystic', 'falsetto'):
            raise ValueError('Unknown TIGER voice style')
        files = ['dsacoustic/phonemes.txt', 'dsacoustic/tiger_acou.onnx',
                 f'dsacoustic/tiger_{style}.emb', 'dsvocoder/tgm_hifigan.onnx']
        for name in files:
            if not (self.root / name).is_file():
                raise ValueError(f'Missing voice asset: {self.root / name}. Run download-voice or set --voice.')
        self.fingerprint = {name: digest_file(self.root / name) for name in files}
        self.settings = {'version': RENDER_VERSION, 'effort': effort, 'speedup': speedup, 'style': style, 'seed': seed,
                         'onnxruntime': ort.__version__, 'threads': threads, 'device': 'cpu',
                         'phrase_seconds': phrase_seconds, 'gender': gender, 'velocity': velocity}
        self.phones = {p.strip(): i for i, p in enumerate((self.root / files[0]).read_text().splitlines())}
        self.embedding = np.fromfile(self.root / files[2], dtype=np.float32)
        if self.embedding.shape != (256,):
            raise ValueError('Expected TIGER 256-element speaker embedding')
        self.options = ort.SessionOptions()
        self.options.intra_op_num_threads = threads
        self.options.inter_op_num_threads = 1
        ort.disable_telemetry_events()
        self.acoustic = self.vocoder = None
        self.hits = self.misses = 0

    def check_phones(self, score):
        unknown = sorted({p for groups in score['parts'].values() for g in groups for p in g['phonemes']} - self.phones.keys())
        if unknown:
            raise ValueError(f'Voice does not support phonemes: {", ".join(unknown)}')

    def sessions(self):
        if self.acoustic is None:
            self.acoustic = self.ort.InferenceSession(str(self.root / 'dsacoustic/tiger_acou.onnx'), self.options, providers=['CPUExecutionProvider'])
            self.vocoder = self.ort.InferenceSession(str(self.root / 'dsvocoder/tgm_hifigan.onnx'), self.options, providers=['CPUExecutionProvider'])


    def chunk(self, groups):
        self.sessions()
        key = hashlib.sha256(json.dumps([self.fingerprint, self.settings, groups], sort_keys=True).encode()).hexdigest()
        path = self.cache / (key + '.wav')
        start, end = groups[0]['notes'][0][0], groups[-1]['notes'][-1][1]
        length = round(end * SR) - round(start * SR)
        if path.exists():
            try:
                wave, sample_rate = sf.read(path)
                if sample_rate == SR and wave.ndim == 1 and len(wave) == length and np.isfinite(wave).all():
                    self.hits += 1
                    return start, wave
            except (RuntimeError, sf.LibsndfileError):
                pass
        self.ort.set_seed(self.settings['seed'])
        pad = .18
        n = round((end - start + 2 * pad) / DT)
        tt = np.arange(n) * DT
        pitch = np.full(n, -1.)
        tokens, durations = [], []
        cursor = 0
        def append(phone, frames):
            nonlocal cursor
            if frames:
                tokens.append(self.phones[phone])
                durations.append(frames)
                cursor += frames
        append('SP', round(pad / DT))
        for group in groups:
            a = round((group['notes'][0][0] - start + pad) / DT)
            b = round((group['notes'][-1][1] - start + pad) / DT)
            if a < cursor:
                raise ValueError('Overlapping phoneme timing after quantization')
            append('SP', a - cursor)
            for phone, frames in zip(group['phonemes'], frame_durations(group['phonemes'], b - a)):
                append(phone, frames)
            for na, nb, midi in group['notes']:
                pitch[(tt >= na - start + pad) & (tt < nb - start + pad)] = midi
        append('SP', n - cursor)
        if sum(durations) != n:
            raise ValueError('Internal phoneme timing mismatch')
        voiced = pitch >= 0
        indices = np.flatnonzero(voiced)
        filled = np.interp(np.arange(n), indices, pitch[indices])
        smooth = gaussian_filter1d(filled, .022 / DT)
        vibrato = .10 * np.sin(2 * np.pi * 5.2 * tt + .3) * np.minimum(tt / .4, 1)
        f0 = (440 * 2 ** ((smooth + vibrato - 69) / 12)).astype(np.float32)
        f0[~voiced] = 0
        inputs = {'tokens': np.array([tokens], np.int64), 'durations': np.array([durations], np.int64),
                  'f0': f0[None, :], 'gender': np.full((1, n), self.settings['gender'], np.float32), 'velocity': np.full((1, n), self.settings['velocity'], np.float32),
                  'spk_embed': np.broadcast_to(self.embedding, (1, n, 256)).copy(),
                  'depth': np.array(400, np.int64), 'speedup': np.array(self.settings['speedup'], np.int64)}
        mel = self.acoustic.run(None, inputs)[0]
        wave = self.vocoder.run(None, {'mel': mel, 'f0': f0[None, :]})[0][0]
        trim = round(pad * SR)
        wave = wave[trim:trim + length]
        wave = np.pad(wave, (0, max(0, length - len(wave))))
        fade = min(round(.012 * SR), len(wave) // 3)
        wave[:fade] *= np.linspace(0, 1, fade)
        wave[-fade:] *= np.linspace(1, 0, fade)
        if not np.isfinite(wave).all():
            raise ValueError('Non-finite model output')
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
