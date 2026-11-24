"""Install the upstream inference code, English prompt and checkpoint together."""
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
from urllib.request import urlopen

SOURCE_URL = 'https://github.com/Soul-AILab/SoulX-Singer.git'
SOURCE_REVISION = '81aeb3ae772c70093c3de74dc23c92d983801ae4'
MODEL_URL = ('https://huggingface.co/Soul-AILab/SoulX-Singer/resolve/'
             '40493ad90286056c7a9095035164434a79daa8c9/model.pt')
MODEL_PATH = Path('pretrained_models/SoulX-Singer/model.pt')
MODEL_SHA256 = '447eaf41f91a6b6659d55e9ec3c9b809221724fb8592aebaec35a23751a5b500'


def verify_checkpoint(path):
    with path.open('rb') as model:
        digest = hashlib.file_digest(model, 'sha256').hexdigest()
    if digest != MODEL_SHA256:
        raise ValueError(f'Checkpoint checksum mismatch at {path}')


def install(root):
    if root.exists():
        assets = (MODEL_PATH, 'example/audio/en_prompt.mp3', 'example/audio/en_prompt.json',
                  'cli/inference.py', 'soulxsinger/config/soulxsinger.yaml',
                  'soulxsinger/utils/phoneme/phone_set.json')
        if not all((root / path).is_file() for path in assets):
            raise ValueError(f'Incomplete SoulX-Singer installation at {root}. Move it aside and retry.')
        verify_checkpoint(root / MODEL_PATH)
        print(f'Voice already installed at {root}')
        return

    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root.parent) as tmp:
        source = Path(tmp) / 'source'
        subprocess.run(['git', 'init', '-q', str(source)], check=True)
        subprocess.run(['git', '-C', str(source), 'fetch', '--depth=1', SOURCE_URL, SOURCE_REVISION], check=True)
        subprocess.run(['git', '-C', str(source), 'checkout', '--detach', '-q', 'FETCH_HEAD'], check=True)
        model = source / MODEL_PATH
        model.parent.mkdir(parents=True, exist_ok=True)
        print('Downloading the SoulX-Singer checkpoint (2.8 GB)...', flush=True)
        with urlopen(MODEL_URL, timeout=120) as response, model.open('wb') as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        verify_checkpoint(model)
        source.rename(root)
    print(f'Voice installed at {root}')
