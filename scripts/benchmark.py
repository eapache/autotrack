"""Reproducible uncached CPU benchmark using an original synthetic melody."""
import argparse
import gc
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from learning_tracks.render import Renderer, EFFORT_SPEEDUPS


def passage(seconds):
    return [{'phonemes': ['d', 'uw'], 'notes': [(float(i), float(min(i + 1, seconds)),
             (60, 62, 64, 67, 65, 64, 62, 60)[i % 8])]} for i in range(seconds)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--voice', required=True, type=Path)
    p.add_argument('--threads', type=int, nargs='+', default=[1, 2, 4, 8])
    p.add_argument('--seconds', type=int, nargs='+', default=[4, 12, 30, 60])
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--effort', choices=EFFORT_SPEEDUPS, default='standard')
    p.add_argument('--out', type=Path, default=Path('output/benchmark.json'))
    a = p.parse_args()
    if a.repeats < 1 or min(a.seconds) < 1 or min(a.threads) < 1:
        p.error('repeats, seconds and threads must be positive')
    report = {'platform': platform.platform(), 'cpu_count': os.cpu_count(),
              'effort': a.effort, 'repeats': a.repeats, 'mode': 'single uncached chunk', 'results': []}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    def save():
        a.out.write_text(json.dumps(report, indent=2) + '\n')
    for threads in a.threads:
        with tempfile.TemporaryDirectory() as temp:
            renderer = None
            try:
                renderer = Renderer(a.voice, temp, threads=threads, effort=a.effort)
                report['voice_files_sha256'] = renderer.fingerprint
                start = time.perf_counter()
                renderer.sessions()
                init = time.perf_counter() - start
                for seconds in a.seconds:
                    row = {'device': 'cpu', 'threads': threads, 'audio_seconds': seconds,
                           'session_seconds': init, 'settings': renderer.settings.copy()}
                    times = []
                    for attempt in range(a.repeats + 1):
                        for f in Path(temp).glob('*.wav'):
                            f.unlink()
                        start = time.perf_counter()
                        renderer.chunk(passage(seconds))
                        elapsed = time.perf_counter() - start
                        if attempt:
                            times.append(elapsed)
                    row.update(samples_seconds=times, median_seconds=statistics.median(times))
                    row['realtime_factor'] = row['median_seconds'] / seconds
                    report['results'].append(row)
                    save()
                    print(json.dumps(row), flush=True)
            except Exception as exc:
                report['results'].append({'device': 'cpu', 'threads': threads, 'error': str(exc)})
                save()
                print(f'{threads} threads: {exc}', flush=True)
            finally:
                del renderer
                gc.collect()

    return 1 if any('error' in row for row in report['results']) else 0


if __name__ == '__main__':
    sys.exit(main())
