# Autotrack

Turn sheet music into sung barbershop learning tracks: **tenor, lead, baritone,
bass, and a full mix**. Everything runs on your computer.

## Set up

Install Autotrack with Python 3.11 or newer and FFmpeg. From this checkout:

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
ffmpeg -version
```

SoulX-Singer uses a separate Python environment. The following Linux CPU setup
requires Python 3.11, Git, curl and about 2.8 GB for the checkpoint, plus the
Python packages. The upstream checkout includes an English reference singer.

```sh
soulx_dir="${XDG_DATA_HOME:-$HOME/.local/share}/autotrack/SoulX-Singer"
git clone https://github.com/Soul-AILab/SoulX-Singer.git "$soulx_dir"
git -C "$soulx_dir" checkout 81aeb3ae772c70093c3de74dc23c92d983801ae4
python3.11 -m venv "$soulx_dir/.venv"
"$soulx_dir/.venv/bin/pip" install --index-url https://download.pytorch.org/whl/cpu torch==2.2.0 torchaudio==2.2.0
"$soulx_dir/.venv/bin/pip" install -r requirements-soulx.txt
mkdir -p "$soulx_dir/pretrained_models/SoulX-Singer"
curl -fL --retry 3 https://huggingface.co/Soul-AILab/SoulX-Singer/resolve/40493ad90286056c7a9095035164434a79daa8c9/model.pt -o "$soulx_dir/pretrained_models/SoulX-Singer/model.pt"
```

For GPU inference, install CUDA-compatible PyTorch and torchaudio builds in
the SoulX environment and pass `--device cuda`. After setup, rendering runs
offline. Activate Autotrack’s environment when starting a new terminal.

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
| `--steps 16` | Use fewer diffusion steps for a quicker render (default 32) |
| `--prompt-audio reference.wav --prompt-metadata reference.json` | Use the singer from an aligned SoulX reference recording |
| `--pitch Bb3` | Set the opening pitch yourself |
| `--target-side right` | Put your part in the right ear |
| `--dominance-db 5` | Make your part more prominent; the default is 3 |
| `--tempo 75` | Set the song’s speed in quarter-note beats per minute |

For another version, choose a new output folder:

```sh
autotrack run my-song.mxl --steps 16 --out output/my-song-preview
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

Autotrack is MIT-licensed. Soul-AILab releases the SoulX-Singer model under
Apache-2.0. Use recordings and arrangements you have permission to use.
See the [model card](https://huggingface.co/Soul-AILab/SoulX-Singer).
