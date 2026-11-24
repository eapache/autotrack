# Autotrack

Turn sheet music into sung barbershop learning tracks: **tenor, lead, baritone,
bass, and a full mix**. Everything runs on your computer.

## Set up

You need Python 3.11 or newer and FFmpeg. On Ubuntu or Debian, run these commands
from this project’s folder:

```sh
sudo apt install ffmpeg python3-venv
python3 -m venv .venv
. .venv/bin/activate
pip install .
autotrack download-voice
```

The voice download is about 2.8 GB and only needs to be done once. After that,
you can make tracks offline. In a new terminal, run `. .venv/bin/activate` again
before using Autotrack.

## Make tracks

Export your song as **MusicXML** (`.musicxml` or `.mxl`) from MuseScore or another
notation editor. If you have a PDF, first transcribe it or use music-scanning
software, then check the notes and lyrics in your editor.

Conventional two-staff barbershop music is recognized automatically, with
tenor/lead on the treble staff and baritone/bass on the bass staff. Keep each
singing part in its own notation voice, including where voices cross.

```sh
autotrack run my-song.mxl
```

Your five MP3s will be in `output/my-song/`, along with a short credits file.
Listen through before sharing the tracks. If Autotrack needs help identifying the
parts, key, or a pronunciation, see [preparing your score](docs/advanced.md#preparing-your-score).

To try the included practice exercise:

```sh
autotrack run examples/four_part_demo.musicxml \
  --pronunciations examples/demo_pronunciations.json
```

## Adjust the result

| Option | What it does |
| --- | --- |
| `--out tracks/my-song` | Choose where to save the MP3s |
| `--effort high` | Spend more time refining the singing: `low`, `standard` (default), `high`, or `max` |
| `--device cpu` | Force CPU rendering. The default uses CUDA when PyTorch detects a compatible GPU. |
| `--pitch Bb3` | Set the opening pitch yourself |
| `--target-side right` | Put your part in the right ear |
| `--dominance-db 5` | Make your part more prominent; the default is 3 |
| `--tempo 75` | Set the song’s speed in quarter-note beats per minute |

For another version, choose a new output folder:

```sh
autotrack run my-song.mxl --effort high --out output/my-song-high
```

## Working files and retries

Autotrack keeps its working files separately in the
`work/my-song/` folder. This includes the separate voices, WAV mixes, an editable
score, and settings.

To retry a failed run, use the same command. Completed phrases are reused to save
time. You can move the working folder with `--work-dir PATH`, or delete it when
you no longer need the intermediates.

## Development

Use an editable install so the `autotrack` command picks up source changes:

```sh
python -m pip install -e .
python -m unittest discover -s tests -v
```

Tests need FFmpeg but do not download a voice. For pinned dependencies, install
`requirements.txt`, then `pip install --no-deps -e .`.

- [Advanced options, score preparation, and working files](docs/advanced.md)
- [Editable score format](docs/score-format.md)

Autotrack is MIT-licensed. SoulX-Singer supplies the singing model and English
reference recording. Its inference code and weights use the Apache 2.0 license.
See the [SoulX-Singer project](https://github.com/Soul-AILab/SoulX-Singer) and
[bundled license](autotrack/soulx/LICENSE). Use sheet music and arrangements you
have permission to use.
