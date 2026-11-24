from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from autotrack.render import Renderer, SR


class RenderTests(unittest.TestCase):
    def test_phrase_conversion_alignment_and_cache_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for path in ('cli/inference.py', 'soulxsinger/config/soulxsinger.yaml',
                         'pretrained_models/SoulX-Singer/model.pt', 'example/audio/en_prompt.mp3',
                         'example/audio/en_prompt.json', 'soulxsinger/utils/phoneme/phone_set.json'):
                asset = root / path
                asset.parent.mkdir(parents=True, exist_ok=True)
                asset.write_text('[]')
            with patch('subprocess.check_output', return_value='test'):
                renderer = Renderer(root, root / 'cache', python=sys.executable)
            groups = [{'phonemes': ['d', 'uw'], 'notes': [(1., 1.51, 60)]}]

            def generate(meta, path):
                self.assertEqual(meta['note_pitch'], '60')
                # The model rounds the duration down to a whole 20 ms frame.
                sf.write(path, .1 * np.sin(np.arange(12000) / 10), 24000)

            with patch.object(renderer, 'generate', side_effect=generate) as generate_phrase:
                stem = renderer.part(groups, SR * 2, 'tenor')
                self.assertEqual(stem.shape, (SR * 2,))
                self.assertFalse(stem[:SR].any())
                self.assertTrue(stem[SR:round(1.51 * SR)].any())
                self.assertFalse(stem[round(1.51 * SR):].any())
                np.testing.assert_allclose(stem, renderer.part(groups, len(stem), 'tenor'), atol=1e-7)
                generate_phrase.assert_called_once()
                cached, = renderer.cache.glob('*.wav')
                cached.write_bytes(b'incomplete')
                renderer.part(groups, len(stem), 'tenor')
                self.assertEqual((renderer.hits, renderer.misses), (1, 2))
