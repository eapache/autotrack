import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from learning_tracks.voice import install
from learning_tracks.render import digest_file


class VoiceInstallTests(unittest.TestCase):
    def archive(self, root, unsafe=False):
        inner = io.BytesIO()
        with zipfile.ZipFile(inner, 'w') as z:
            z.writestr('dsacoustic/phonemes.txt', 'SP\nah\n')
            z.writestr('extra/LICENSE.md', 'test license')
            z.writestr('plugin.py', 'not installed')
            if unsafe:
                z.writestr('../escape', 'unsafe')
        archive = root / 'voice.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('TIGER_DS_v102_PACK/Voice Library/TIGER_DS_v102.zip', inner.getvalue())
        return archive

    def test_installs_only_assets_and_license(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = self.archive(root)
            with patch('learning_tracks.voice.SHA256', digest_file(archive)):
                install(root / 'model', archive)
            self.assertTrue((root / 'model/extra/LICENSE.md').is_file())
            self.assertTrue((root / 'model/dsacoustic/phonemes.txt').is_file())
            self.assertFalse((root / 'model/plugin.py').exists())

    def test_checksum_and_unsafe_path_leave_no_model(self):
        for unsafe in (False, True):
            with self.subTest(unsafe=unsafe), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                archive = self.archive(root, unsafe)
                checksum = digest_file(archive) if unsafe else 'bad hash'
                with patch('learning_tracks.voice.SHA256', checksum):
                    with self.assertRaises(ValueError):
                        install(root / 'model', archive)
                self.assertEqual(list(root.iterdir()), [archive])
