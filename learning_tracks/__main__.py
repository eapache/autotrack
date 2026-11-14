"""Command-line entry point. No LLM, service credentials, or inference API."""
import argparse
import json
import shutil
import sys
import time
import os
import subprocess
import tempfile
import re
import math
from pathlib import Path
from . import __version__, default_threads
from .score import PARTS, Timeline, load

DATA = Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local/share')) / 'autotrack'


def import_options(parser):
    parser.add_argument('--map', help='Explicit role=part:voice pairs from inspect')
    parser.add_argument('--pronunciations', type=Path, help='JSON mapping lyric syllables to ARPAbet phonemes')
    parser.add_argument('--tempo', type=float, help='Override all imported tempo marks (quarter notes/minute)')
    parser.add_argument('--swing', type=float, default=1., help='Long:short eighth-note ratio; 1 straight, 2 swing. Tuplets stay explicit.')
    parser.add_argument('--count-in', type=int, default=3, help='Count-in quarter-note beats for MusicXML import')
    parser.add_argument('--missing-lyric', help='Explicit fallback syllable, e.g. doo, where no preceding lyric exists')
    parser.add_argument('--no-expand-repeats', action='store_true')


def read_input(args):
    if args.score.suffix.lower() == '.json':
        return load(args.score)
    from .musicxml import import_score
    if not math.isfinite(args.swing) or args.swing <= 0 or args.count_in < 0 or (args.tempo is not None and (not math.isfinite(args.tempo) or args.tempo <= 0)):
        raise ValueError('Tempo/swing must be positive and count-in nonnegative')
    pronunciations = json.loads(args.pronunciations.read_text()) if args.pronunciations else None
    result = import_score(args.score, args.map, pronunciations, args.tempo, args.swing,
                          args.count_in, args.missing_lyric, not args.no_expand_repeats)
    for warning in result.get('warnings', []):
        print('NOTE: ' + warning, file=sys.stderr)
    return result


def mixing_options(parser):
    parser.add_argument('--dominance-db', type=float, default=3., help='How much louder your part is, in dB (default: 3)')
    parser.add_argument('--target-side', choices=('left', 'right'), default='left', help='Ear for your part (default: left)')
    parser.add_argument('--pitch', help='Pitch-pipe note, e.g. Bb3 or C4; defaults to the explicit initial tonic')
    parser.add_argument('--pitch-seconds', type=float, default=3., help='Pitch duration, 0 to disable (default: 3)')


def record_failure(folder, error):
    if folder is not None:
        try:
            path = folder / 'run.json'
            request = json.loads(path.read_text()) if path.exists() else {}
            request.update(status='failed', error=str(error))
            path.write_text(json.dumps(request, indent=2) + '\n')
        except (OSError, ValueError):
            pass  # Preserve the original error if the working disk is unavailable.
        print(f'Working files saved: {folder}', file=sys.stderr)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Offline MusicXML → four sung learning MP3s and a full mix')
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    sub = parser.add_subparsers(dest='command', required=True)
    inspect = sub.add_parser('inspect', help='List MusicXML parts/voices and pitch ranges')
    inspect.add_argument('score', type=Path)
    imp = sub.add_parser('import-xml', help='Import MusicXML into editable, validated score JSON')
    imp.add_argument('score', type=Path)
    imp.add_argument('--out', required=True, type=Path)
    import_options(imp)
    check = sub.add_parser('validate', help='Validate structured score JSON without loading a voice model')
    check.add_argument('score', type=Path)
    run = sub.add_parser('run', help='Render five MP3s with a tonic pitch and count-in')
    run.add_argument('score', type=Path)
    run.add_argument('--out', type=Path, help='Folder for finished tracks (default: output/SONG)')
    advanced = run.add_argument_group('Advanced settings')
    advanced.add_argument('--voice', type=Path, default=DATA / 'tiger')
    advanced.add_argument('--work-dir', type=Path, help='Working folder (default: work/OUTPUT-NAME in the current folder)')
    advanced.add_argument('--cache', type=Path, help='Override the phrase cache inside the working folder')
    advanced.add_argument('--threads', type=int, default=default_threads(), help='CPU inference threads (default: up to 8)')
    run.add_argument('--effort', choices=('low', 'standard', 'high', 'max'), default='standard',
                     help='Time spent refining the singing; higher takes longer (default: standard)')
    run.add_argument('--style', default='fresh', choices=('fresh', 'disco', 'electric', 'vinyl', 'cetera', 'mystic', 'falsetto'))
    advanced.add_argument('--seed', type=int, default=42)
    advanced.add_argument('--phrase-seconds', type=float, default=13., help='Maximum phrase grouping span, 1–60 seconds; a syllable is never split')
    advanced.add_argument('--gender', type=float, default=0., help='Model timbre shift, -5 to 5; does not transpose notes')
    advanced.add_argument('--velocity', type=float, default=1., help='Model phoneme speed conditioning, 0.5 to 2; preserves score timing')
    import_options(run.add_argument_group('Score settings'))
    mixing_options(run)
    voice = sub.add_parser('download-voice', help='Install the pinned TIGER voice model (one-time network access)')
    voice.add_argument('destination', type=Path, nargs='?', default=DATA / 'tiger')
    voice.add_argument('--archive', type=Path, help='Use an already downloaded official zip')
    sub.add_parser('doctor', help='Show runtime and FFmpeg availability')
    args = parser.parse_args(argv)
    if args.command == 'run' and args.out is None:
        args.out = Path('output') / args.score.stem
    working_run = None
    try:
        if args.command == 'doctor':
            import onnxruntime as ort
            print(json.dumps({'onnxruntime': ort.__version__, 'provider': 'CPUExecutionProvider', 'ffmpeg': shutil.which('ffmpeg')}, indent=2))
        elif args.command == 'inspect':
            from .musicxml import read, describe
            from music21.pitch import Pitch
            print(f'{"Selector":<12} {"Part":<24} {"Notes":>6}  Range')
            for row in describe(read(args.score)):
                pitches = row['midi_range']
                extent = '–'.join(Pitch(p).nameWithOctave for p in pitches) if pitches else '—'
                print(f'{row["selector"]:<12} {row["name"]:<24} {row["notes"]:>6}  {extent}')
        elif args.command == 'import-xml':
            score = read_input(args)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(score, indent=2) + '\n')
            print(f'Saved {args.out}')
        elif args.command == 'validate':
            score = load(args.score)
            print(f'Valid: {score["title"]}, four parts, {score["duration_beats"]:g} quarter-note beats')
        elif args.command == 'download-voice':
            from .voice import install
            install(args.destination, args.archive)
        elif args.command == 'run':
            from .render import Renderer, SR, atomic_audio
            from .mix import export
            if not shutil.which('ffmpeg'):
                raise ValueError('Install ffmpeg for MP3 export')
            if args.out.exists():
                raise ValueError(f'Output already exists: {args.out}. Choose a new directory.')
            if not -20 <= args.dominance_db <= 20:
                raise ValueError('dominance-db must be between -20 and 20')
            score = read_input(args)
            from .mix import pitch_tone
            from music21.pitch import Pitch
            pitch_midi = score.get('tonic_midi')
            if args.pitch:
                try:
                    if not re.fullmatch(r'[A-Ga-g](?:#|b|-)?[0-9]', args.pitch):
                        raise ValueError('Use a note and octave, e.g. Bb3 or F#4')
                    pitch = Pitch(args.pitch[0].upper() + args.pitch[1:].replace('b', '-'))
                    pitch_midi = int(pitch.ps)
                except Exception as exc:
                    raise ValueError(f'Invalid pitch: {args.pitch}: {exc}') from exc
            if pitch_midi is None and args.pitch_seconds:
                raise ValueError('No unambiguous initial key: supply --pitch (e.g. Bb3) or --pitch-seconds 0')
            pitch_tone(pitch_midi if pitch_midi is not None else 60, args.pitch_seconds)
            work_root = args.work_dir or Path('work') / args.out.name
            out_resolved, work_resolved = args.out.resolve(), work_root.resolve()
            if out_resolved == work_resolved or out_resolved in work_resolved.parents or work_resolved in out_resolved.parents:
                raise ValueError('The output and working folders must be separate, not inside each other')
            if args.cache is not None:
                cache_resolved = args.cache.resolve()
                if cache_resolved == out_resolved or out_resolved in cache_resolved.parents:
                    raise ValueError('The cache must be outside the finished output folder')
            work_root.mkdir(parents=True, exist_ok=True)
            working_run = Path(tempfile.mkdtemp(prefix='run-', dir=work_root))
            (working_run / 'score.json').write_text(json.dumps(score, indent=2) + '\n')
            request = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
            request['status'] = 'in_progress'
            (working_run / 'run.json').write_text(json.dumps(request, indent=2) + '\n')
            renderer = Renderer(args.voice, args.cache or work_root / 'cache', args.threads, args.effort, args.style, args.seed,
                                phrase_seconds=args.phrase_seconds, gender=args.gender, velocity=args.velocity)
            renderer.check_phones(score)
            timeline = Timeline(score)
            length = round((timeline.seconds(score['duration_beats']) + 1) * SR)
            # Preflight every syllable before rendering anything.
            from .render import DT, frame_durations
            prepared = {part: timeline.groups(score['parts'][part]) for part in PARTS}
            for part, groups in prepared.items():
                for g in groups:
                    duration = g['notes'][-1][1] - g['notes'][0][0]
                    if duration > 30:
                        raise ValueError(f'{part}: syllable exceeds 30 seconds')
                    frame_durations(g['phonemes'], max(0, int(duration / DT) - 1))
            args.out.parent.mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            # Keep intermediates and partial results after success or failure.
            for part in PARTS:
                samples = renderer.part(prepared[part], length, part)
                atomic_audio(working_run / 'mono' / f'{part}.wav', samples, 'PCM_24')
            metrics = export(working_run / 'mono', working_run / 'mp3', score,
                             args.dominance_db, args.target_side, pitch_midi, args.pitch_seconds,
                             wav_dir=working_run / 'mixes')
            manifest = {'autotrack_version': __version__, 'title': score['title'],
                        'voice_files_sha256': renderer.fingerprint,
                        'render_settings': renderer.settings, 'cache_hits': renderer.hits,
                        'cache_misses': renderer.misses,
                        'elapsed_seconds': round(time.monotonic() - started, 2), 'mixes': metrics}
            (working_run / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
            (working_run / 'CREDITS.txt').write_text(
                'Synthetic singing: TIGER DS v102 by tigermeat / neutrogic.\n'
                'https://github.com/spicytigermeat/tiger_diffsinger\n'
                'DiffSinger and TIGER HiFi-GAN via ONNX Runtime.\n'
                'Commercial use of this voice requires permission from the rights holder.\n'
                'Composition and arrangement rights remain with their owners.\n')
            # Publish only complete deliverables, even with a work folder on another disk.
            with tempfile.TemporaryDirectory(prefix='.autotrack-', dir=args.out.parent) as temp:
                result = Path(temp) / 'result'
                shutil.copytree(working_run / 'mp3', result)
                shutil.copy2(working_run / 'CREDITS.txt', result / 'CREDITS.txt')
                result.rename(args.out)
            request['status'] = 'complete'
            (working_run / 'run.json').write_text(json.dumps(request, indent=2) + '\n')
            print(f'Tracks ready: {args.out}\nWorking files: {working_run}', flush=True)
    except KeyboardInterrupt as exc:
        record_failure(working_run, 'Interrupted')
        parser.exit(130, 'Interrupted; working files have been kept.\n')
    except ImportError as exc:
        record_failure(working_run, exc)
        parser.exit(2, f'Missing dependency: {exc}. Install the package dependencies with pip install .\n')
    except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        record_failure(working_run, exc)
        parser.exit(2, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
