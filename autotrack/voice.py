"""One-time installation of the SoulX checkpoint and its aligned singing prompt."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import urllib.request

from .render import digest_file

MODEL = 'https://huggingface.co/Soul-AILab/SoulX-Singer/resolve/40493ad90286056c7a9095035164434a79daa8c9'
PROMPT = 'https://raw.githubusercontent.com/Soul-AILab/SoulX-Singer/81aeb3ae772c70093c3de74dc23c92d983801ae4/example/audio'
ASSETS = {
    'model.pt': (MODEL, '447eaf41f91a6b6659d55e9ec3c9b809221724fb8592aebaec35a23751a5b500'),
    'en_prompt.mp3': (PROMPT, 'fa7eed4baa89ad7f6ea5a87de3dc6497525ba0bd009148a9c7c6d6e9b12a27dc'),
    'en_prompt.json': (PROMPT, 'ef119ae77c4f22794c546197da94d7059ce7321a6c155fc6e5028e36afc00b93'),
}


def install(destination):
    destination = Path(destination)
    if destination.exists():
        raise ValueError(f'{destination} already exists; choose a new directory')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.autotrack-', dir=destination.parent) as temp:
        assets = Path(temp) / 'voice'
        assets.mkdir()
        for name, (base, checksum) in ASSETS.items():
            print(f'Downloading {name}...', flush=True)
            path = assets / name
            with urllib.request.urlopen(f'{base}/{name}', timeout=120) as src, path.open('wb') as out:
                shutil.copyfileobj(src, out, length=1024 * 1024)
            if digest_file(path) != checksum:
                raise ValueError(f'Checksum mismatch for {name}')
        start, end = json.loads((assets / 'en_prompt.json').read_text())[0]['time']
        subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(start / 1000),
                        '-i', str(assets / 'en_prompt.mp3'), '-t', str((end - start) / 1000),
                        '-ac', '1', '-ar', '24000', str(assets / 'en_prompt.wav')], check=True)
        (assets / 'en_prompt.mp3').unlink()
        assets.rename(destination)
    print(f'Installed voice at {destination}', flush=True)
