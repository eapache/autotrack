"""One-time, checksum-verified download; no network during score rendering."""
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from .render import digest_file

URL = 'https://github.com/spicytigermeat/tiger_diffsinger/releases/download/v102/TIGER_DS_v102_PACK.zip'
SHA256 = '925ca2ad44287757d26628608b2744381e75748042d0e6fae356ee8cdae19f4e'


def install(destination, archive=None):
    destination = Path(destination)
    if destination.exists():
        raise ValueError(f'{destination} already exists; choose a new directory')
    with tempfile.TemporaryDirectory() as temp:
        package = Path(archive) if archive else Path(temp) / 'tiger.zip'
        if archive is None:
            print('Downloading TIGER DS v102 (540 MiB)...', flush=True)
            with urllib.request.urlopen(URL, timeout=60) as response, open(package, 'wb') as out:
                shutil.copyfileobj(response, out)
        if digest_file(package) != SHA256:
            raise ValueError('Voice download checksum mismatch')
        with zipfile.ZipFile(package) as outer:
            inner = Path(temp) / 'voice.zip'
            with outer.open('TIGER_DS_v102_PACK/Voice Library/TIGER_DS_v102.zip') as src, inner.open('wb') as out:
                shutil.copyfileobj(src, out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.autotrack-model-', dir=destination.parent) as staging:
            staged = Path(staging) / 'voice'
            staged.mkdir()
            with zipfile.ZipFile(inner) as z:
                # Install only the acoustic/vocoder assets and their documentation.
                for entry in z.infolist():
                    name = Path(entry.filename)
                    if name.is_absolute() or '..' in name.parts:
                        raise ValueError('Unsafe archive path')
                    if entry.is_dir() or not (entry.filename.startswith(('dsacoustic/', 'dsvocoder/')) or entry.filename in ('dsconfig.yaml', 'extra/LICENSE.md')):
                        continue
                    target = staged / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(entry) as src, open(target, 'wb') as out:
                        shutil.copyfileobj(src, out)
            staged.rename(destination)
    print(f'Installed voice at {destination}', flush=True)
