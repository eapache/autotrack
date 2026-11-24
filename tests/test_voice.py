import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from autotrack import voice


class InstallTests(unittest.TestCase):
    def test_bad_checkpoint_is_never_published_and_setup_can_be_retried(self):
        def git(args, check):
            if 'init' in args:
                Path(args[-1]).mkdir()
            elif 'checkout' in args:
                source = Path(args[2])
                for name in ('cli/inference.py', 'example/audio/en_prompt.mp3', 'example/audio/en_prompt.json',
                             'soulxsinger/config/soulxsinger.yaml', 'soulxsinger/utils/phoneme/phone_set.json'):
                    asset = source / name
                    asset.parent.mkdir(parents=True, exist_ok=True)
                    asset.touch()

        checksum = hashlib.sha256(b'checkpoint').hexdigest()
        with tempfile.TemporaryDirectory() as tmp, patch('subprocess.run', side_effect=git), \
                patch.object(voice, 'MODEL_SHA256', checksum), \
                patch.object(voice, 'urlopen', side_effect=[io.BytesIO(b'partial'), io.BytesIO(b'checkpoint')]) as fetch:
            root = Path(tmp) / 'SoulX-Singer'
            with self.assertRaisesRegex(ValueError, 'checksum'):
                voice.install(root)
            self.assertEqual(list(Path(tmp).iterdir()), [])
            voice.install(root)
            voice.install(root)
            self.assertEqual(fetch.call_count, 2)
            self.assertEqual((root / voice.MODEL_PATH).read_bytes(), b'checkpoint')
