"""CLI failure semantics without downloading or loading a model."""
import json
from contextlib import chdir
import shutil
import numpy as np
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from learning_tracks.__main__ import main
from test_pipeline import simple_score


class CLITests(unittest.TestCase):
    def test_failed_render_keeps_recoverable_work_without_publishing_output(self):
        with tempfile.TemporaryDirectory() as temp, chdir(temp):
            root = Path(temp)
            score = root / 'song.json'
            score.write_text(json.dumps(simple_score()))
            out = root / 'output'
            with patch('learning_tracks.render.Renderer') as renderer, patch('shutil.which', return_value='/ffmpeg'):
                renderer.return_value.part.side_effect = ValueError('injected failure')
                with self.assertRaises(SystemExit) as error:
                    main(['run', str(score), '--pitch', 'Bb3', '--out', str(out)])
            self.assertEqual(error.exception.code, 2)
            self.assertFalse(out.exists())
            runs = list((root / 'work/output').glob('run-*'))
            self.assertEqual(len(runs), 1)
            self.assertEqual(json.loads((runs[0] / 'score.json').read_text()), simple_score())
            self.assertEqual(json.loads((runs[0] / 'run.json').read_text())['status'], 'failed')
            self.assertFalse(list(root.glob('.autotrack-*')))

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp, chdir(temp):
            out = Path(temp) / 'output'
            out.mkdir()
            sentinel = out / 'keep.txt'
            sentinel.write_text('keep')
            with patch('shutil.which', return_value='/ffmpeg'), patch('learning_tracks.render.Renderer') as renderer:
                with self.assertRaises(SystemExit):
                    main(['run', '/nonexistent', '--out', str(out)])
                renderer.assert_not_called()
            self.assertEqual(sentinel.read_text(), 'keep')

    def test_ambiguous_pitch_and_invalid_settings_fail_before_model(self):
        with tempfile.TemporaryDirectory() as temp, chdir(temp):
            score = Path(temp) / 'song.json'
            score.write_text(json.dumps(simple_score()))
            for options in ([], ['--pitch', 'C'], ['--pitch', 'A9'],
                            ['--pitch', 'C4', '--pitch-seconds', 'nan'],
                            ['--pitch', 'C4', '--dominance-db', 'nan']):
                with self.subTest(options=options), patch('shutil.which', return_value='/ffmpeg'), patch('learning_tracks.render.Renderer') as renderer:
                    with self.assertRaises(SystemExit):
                        main(['run', str(score), '--out', str(Path(temp) / 'out'), *options])
                    renderer.assert_not_called()

    def test_effort_and_work_folder_reuse_after_failure(self):
        with tempfile.TemporaryDirectory() as temp, chdir(temp):
            root = Path(temp)
            score = root / 'song.json'
            score.write_text(json.dumps(simple_score()))
            work = root / 'scratch'
            for effort in ('low', 'standard', 'high', 'max'):
                with patch('learning_tracks.render.Renderer') as renderer, patch('shutil.which', return_value='/ffmpeg'):
                    renderer.return_value.part.side_effect = ValueError('injected failure')
                    with self.assertRaises(SystemExit):
                        main(['run', str(score), '--pitch', 'C4', '--out', str(root / 'out'),
                              '--work-dir', str(work), '--effort', effort])
                    self.assertEqual(renderer.call_args.args[3], effort)
                    self.assertEqual(renderer.call_args.args[1], work / 'cache')
            self.assertEqual(len(list(work.glob('run-*'))), 4)

    def test_overlapping_output_and_work_folders_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp, chdir(temp):
            root = Path(temp)
            score = root / 'song.json'
            score.write_text(json.dumps(simple_score()))
            for work in (root, root / 'out', root / 'out/work'):
                with patch('shutil.which', return_value='/ffmpeg'), patch('learning_tracks.render.Renderer') as renderer:
                    with self.assertRaises(SystemExit):
                        main(['run', str(score), '--pitch', 'C4', '--out', str(root / 'out'), '--work-dir', str(work)])
                    renderer.assert_not_called()

    @unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg required')
    def test_success_publishes_only_deliverables_and_keeps_all_intermediates(self):
        with tempfile.TemporaryDirectory() as temp, chdir(temp):
            root = Path(temp)
            score = root / 'song.json'
            score.write_text(json.dumps(simple_score()))
            with patch('learning_tracks.render.Renderer') as renderer:
                instance = renderer.return_value
                instance.part.side_effect = lambda groups, length, part: .1 * np.sin(2 * np.pi * 220 * np.arange(length) / 44100)
                instance.fingerprint, instance.settings = {}, {'effort': 'standard'}
                instance.hits, instance.misses = 0, 4
                main(['run', str(score), '--pitch', 'C4', '--out', str(root / 'out')])
            self.assertEqual({p.name for p in (root / 'out').iterdir()},
                             {'tenor.mp3', 'lead.mp3', 'baritone.mp3', 'bass.mp3', 'full_mix.mp3', 'CREDITS.txt'})
            run, = (root / 'work/out').glob('run-*')
            self.assertEqual(len(list((run / 'mono').glob('*.wav'))), 4)
            self.assertEqual(len(list((run / 'mixes').glob('*.wav'))), 5)
            self.assertEqual(len(list((run / 'mp3').glob('*.mp3'))), 5)
            self.assertEqual(json.loads((run / 'run.json').read_text())['status'], 'complete')
            self.assertTrue((run / 'score.json').exists())
            self.assertTrue((run / 'manifest.json').exists())

    def test_partial_stems_survive_failure(self):
        with tempfile.TemporaryDirectory() as temp, chdir(temp):
            root = Path(temp)
            score = root / 'song.json'
            score.write_text(json.dumps(simple_score()))
            with patch('learning_tracks.render.Renderer') as renderer, patch('shutil.which', return_value='/ffmpeg'):
                renderer.return_value.part.side_effect = [np.zeros(44100), ValueError('second part failed')]
                with self.assertRaises(SystemExit):
                    main(['run', str(score), '--pitch', 'C4', '--out', str(root / 'out')])
            run, = (root / 'work/out').glob('run-*')
            self.assertTrue((run / 'mono/tenor.wav').exists())
            self.assertFalse((root / 'out').exists())
            self.assertEqual(json.loads((run / 'run.json').read_text())['status'], 'failed')
