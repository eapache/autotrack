import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import soundfile as sf

from autotrack.voice import install
from autotrack.render import digest_file


class VoiceInstallTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg required')
    def test_installs_verified_assets_and_crops_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'model.pt').write_bytes(b'checkpoint')
            (root / 'en_prompt.json').write_text(json.dumps([{'time': [500, 1500]}]))
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                            'sine=frequency=400:duration=2', str(root / 'en_prompt.mp3')], check=True)
            assets = {p.name: (root.as_uri(), digest_file(p)) for p in root.iterdir()}
            dest = root / 'installed'
            with patch('autotrack.voice.ASSETS', assets):
                install(dest)
                with self.assertRaises(ValueError):
                    install(dest)
            self.assertEqual({p.name for p in dest.iterdir()}, {'model.pt', 'en_prompt.wav', 'en_prompt.json'})
            info = sf.info(dest / 'en_prompt.wav')
            self.assertEqual((info.samplerate, info.channels, info.frames), (24000, 1, 24000))

    def test_failed_checksum_leaves_no_partial_install(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / 'model.pt'
            checkpoint.write_bytes(b'bad')
            with patch('autotrack.voice.ASSETS', {'model.pt': (root.as_uri(), 'wrong')}):
                with self.assertRaisesRegex(ValueError, 'Checksum'):
                    install(root / 'installed')
            self.assertEqual(list(root.iterdir()), [checkpoint])
