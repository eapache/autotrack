"""Four aligned mono stems into four verifiable split-stereo learning mixes."""
import shutil
import subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
from .score import PARTS, Timeline
from .render import SR, atomic_audio


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def split_mix(tracks, target, dominance_db=3., target_side='left', active=None):
    if set(tracks) != set(PARTS) or target not in PARTS:
        raise ValueError('Need all four stems and a valid target part')
    if target_side not in ('left', 'right') or not np.isfinite(dominance_db) or not -20 <= dominance_db <= 20:
        raise ValueError('target_side must be left/right and dominance must be between -20 and 20 dB')
    shapes = {x.shape for x in tracks.values()}
    if len(shapes) != 1 or any(x.ndim != 1 or not np.isfinite(x).all() for x in tracks.values()):
        raise ValueError('Stems must be finite mono arrays of identical length')
    active = active if active is not None else slice(None)
    selected = tracks[target].copy()
    backing = sum(x for name, x in tracks.items() if name != target)
    foreground_rms, backing_rms = rms(selected[active]), rms(backing[active])
    if min(foreground_rms, backing_rms) < 1e-9:
        raise ValueError('Cannot balance a silent selected part or silent backing mix')
    backing *= foreground_rms / backing_rms * 10 ** (-dominance_db / 20)
    channels = (selected, backing) if target_side == 'left' else (backing, selected)
    output = np.column_stack(channels)
    output *= 10 ** (-1.5 / 20) / np.max(np.abs(output))
    return output


def pitch_tone(midi, seconds=3.):
    """Centered, gently enveloped tonic with pitch-pipe-like harmonics."""
    if not isinstance(midi, int) or not 0 <= midi <= 127:
        raise ValueError('Pitch must be an integer MIDI note from 0 to 127')
    if not np.isfinite(seconds) or not 0 <= seconds <= 10:
        raise ValueError('pitch-seconds must be between 0 and 10')
    t = np.arange(round(seconds * SR)) / SR
    frequency = 440 * 2 ** ((midi - 69) / 12)
    tone = sum(np.sin(2 * np.pi * frequency * h * t) / h
               for h in (1, 2, 3) if frequency * h < SR / 2)
    envelope = np.minimum(t / .05, 1) * np.minimum((seconds - t) / .1, 1)
    return .12 * tone * envelope


def export(stems_dir, out_dir, score, dominance_db=3., target_side='left',
           pitch_midi=None, pitch_seconds=3., wav_dir=None):
    if not shutil.which('ffmpeg'):
        raise ValueError('ffmpeg is required for MP3 export')
    pitch_midi = pitch_midi if pitch_midi is not None else score.get('tonic_midi')
    if pitch_midi is None and pitch_seconds:
        raise ValueError('No unambiguous initial key: supply --pitch (e.g. Bb3)')
    tone = pitch_tone(pitch_midi if pitch_midi is not None else 60, pitch_seconds)
    prefix = np.zeros((len(tone) + (round(.5 * SR) if len(tone) else 0), 2))
    prefix[:len(tone)] = tone[:, None]
    tracks = {}
    for part in PARTS:
        x, sr = sf.read(Path(stems_dir) / f'{part}.wav')
        if sr != SR:
            raise ValueError(f'{part}: expected {SR} Hz, got {sr}')
        tracks[part] = x
    timeline = Timeline(score)
    active = slice(round(timeline.count_in * SR), round(timeline.seconds(score['duration_beats']) * SR))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {}
    import tempfile
    from contextlib import nullcontext
    context = tempfile.TemporaryDirectory() if wav_dir is None else nullcontext(wav_dir)
    with context as temp:
        for part in (*PARTS, 'full_mix'):
            # split_mix validates the stems before the full mix is reached.
            if part == 'full_mix':
                combined = sum(tracks.values())
                peak = np.max(np.abs(combined))
                if peak < 1e-9:
                    raise ValueError('Full mix is silent')
                y = np.column_stack((combined, combined)) * (10 ** (-1.5 / 20) / peak)
            else:
                y = split_mix(tracks, part, dominance_db, target_side, active)
            for beat in range(int(score.get('count_in_beats', 0))):
                t = np.arange(round(.055 * SR)) / SR
                click = .06 * np.sin(2 * np.pi * (1000 if beat == 0 else 750) * t) * np.exp(-75 * t)
                at = round(beat * 60 / score['tempo'][0]['bpm'] * SR)
                stop = min(len(y), at + len(click))
                y[at:stop] += click[:stop-at, None]
            y = np.concatenate((prefix, y))
            wav = Path(temp) / f'{part}.wav'
            atomic_audio(wav, y, 'PCM_24')
            actual, sr = sf.read(wav)
            peak = np.max(np.abs(actual))
            if not np.isfinite(actual).all() or peak >= .99:
                raise ValueError(f'{part}: export verification failed')
            metrics[part] = {'frames': len(actual), 'channels': 2, 'sample_rate': sr,
                             'duration_seconds': len(actual) / sr, 'peak_dbfs': float(20 * np.log10(peak)),
                             'pitch_midi': pitch_midi, 'pitch_seconds': pitch_seconds}
            if part != 'full_mix':
                music = actual[len(prefix) + active.start:len(prefix) + active.stop]
                side = 0 if target_side == 'left' else 1
                db = 20 * np.log10(rms(music[:, side]) / rms(music[:, 1-side]))
                if abs(db - dominance_db) > .02:
                    raise ValueError(f'{part}: balance verification failed')
                metrics[part].update(dominance_db=float(db), target_side=target_side)
            destination = out_dir / f'{part}.mp3'
            subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(wav),
                            '-codec:a', 'libmp3lame', '-b:a', '192k', '-joint_stereo', '0',
                            '-metadata', f'title={score["title"]} - {part.replace("_", " ").title()}',
                            str(destination)], check=True)
            print(f'Wrote {destination.name}', flush=True)
    return metrics
