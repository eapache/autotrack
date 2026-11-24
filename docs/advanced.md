# Advanced usage and development notes

## Preparing your score

### Parts

Separate parts named Tenor, Lead, Baritone/Bari, and Bass are recognized
automatically. Conventional two-staff barbershop scores are also recognized:
two explicit voices on an upper treble staff (including octave-down treble),
and two on a lower bass staff. These may be separate MusicXML parts or two
staves within one part.

The predominantly up-stem voice is tenor on the upper staff and baritone on
the lower; the down-stem voices are lead and bass. When stems are omitted,
conventional voice numbers 1/2 (or 3/4 on the lower staff) are used in that
order. Part assignments follow the whole MusicXML voice through crossings,
never the pitch order of individual notes. Printed crossing marks are not
interpreted; the MusicXML must already encode the correct voice identities.

Shared lyrics are copied into completely lyric-free voices when note onsets,
durations, and ties match a texted voice. Text on the same staff is preferred,
then the lead and other voices. Existing lyrics are preserved. If rhythms
differ, attach lyrics explicitly in your notation editor (or use
`--missing-lyric doo` for an untexted voice).

For unconventional or ambiguous layouts, or to override the automatic
assignment, list the available voices:

```sh
autotrack inspect song.mxl
```

Then assign the printed selectors to the four singing parts:

```sh
autotrack run song.mxl --map tenor=1:1,lead=1:2,baritone=2:1,bass=2:2
```

Selectors refer to explicit notation voices, not pitch order. Four distinct,
monophonic voices are required; extra instrumental voices can be left unmapped.

### Opening pitch

An explicit initial key supplies the tonic in octave 4. A key signature alone
cannot distinguish relative major and minor, so an ambiguous key needs an
explicit note: `--pitch Bb3` or `--pitch F#4`. This is the pitch-pipe note, not
each part’s first sung note. `--pitch-seconds 0` disables it; otherwise its
length can be 0–10 seconds (default 3), followed by a half-second pause.

### Pronunciation

English whole words use the CMU pronunciation dictionary. Syllabified words and
scat syllables sometimes need overrides. If a syllable is unknown, add it to a
JSON pronunciation file and pass `--pronunciations FILE`:

```json
{"doo": "d uw", "bah": "b aa"}
```

Tokens are lowercase, stress-free ARPAbet. Unknown syllables are reported before
synthesis. `--missing-lyric doo` supplies a fallback for phrases with no lyrics.

### Verses and repeats

The importer unfolds the performance and selects successive lyric verses as a
passage repeats.
A passage with only one lyric line, such as a shared chorus or a part singing
“doo,” keeps that line on every pass. Ties and melismas continue the selected verse.

When no measures repeat and multiple verses are present, the full score is sung
once per verse, with its tempo map restarting on each pass. Lyrics must be
attached to notes in MusicXML; free text printed beneath a score is not imported.

The importer checks that every written lyric entry is reached. If the repeat
structure does not reach all verses, it asks for the repeat count to be corrected
or the song to be written out in performance order. It does not silently omit
the remaining verses. Review complex D.C./D.S./coda instructions and endings in
your notation editor. `--no-expand-repeats` deliberately imports just the written
order and first applicable lyric line for debugging or already-unfolded scores.

### Timing and notation

Numeric tempo changes are preserved. `--tempo 75` replaces them with one tempo.
`--swing 2` applies a 2:1 eighth-note ratio; the default is straight. Explicit
tuplets keep their timing. `--count-in 3` sets the number of count-in quarter beats.
These options affect MusicXML import, not already-imported JSON scores.

Chords within one vocal lane, overlapping notes, microtones, and un-timed grace
notes are rejected. Dynamics, breath marks, fermata lengths, articulations and
text such as “rit.” are not performed automatically. Each voice is normalized
independently, so original inter-part dynamic balance is not retained.

## Working files

For `--out output/song`, the default working root is `work/song` in the current
directory. The `work/` directory is gitignored.
`--work-dir PATH` overrides it. Each attempt creates a new `run-*` directory:

```text
work/song/
  cache/                 reusable phrase audio
  run-.../
    score.json           expanded, editable score
    run.json             options and completion/failure status
    mono/                four aligned vocal WAVs
    mixes/               five stereo WAVs, including pitch cue and count-in
    mp3/                 encoded mixes, including partial exports after a failure
    manifest.json        completed render settings, model hashes and mix checks
    CREDITS.txt
```

The output folder receives only the five MP3s and credits, and is published only
when the exports are complete. Existing output folders are protected. Working
files remain after failure, and previous attempts are never replaced. An
interrupted process may leave `run.json` marked `in_progress`; its files are still
recoverable. A manifest is written after all mixes export successfully.

Retry with the same command to reuse the working root’s phrase cache. For a
variant in another output folder, reuse `--work-dir` to share its cache. Changes
to pitch cue or balance reuse vocal phrases; changes to notes, pronunciation,
reference recording, diffusion steps or runtime can invalidate them. `--cache PATH` overrides
the cache location. Corrupt cache files are regenerated.

You can edit a saved score and render it without importing MusicXML again:

```sh
autotrack validate work/song/run-EXAMPLE/score.json
autotrack run work/song/run-EXAMPLE/score.json --out output/revised
```

Stems and stereo mixes are 44.1 kHz, 24-bit PCM; phrase cache WAVs use floating
point samples. Final MP3s are 192 kbps. Stereo balance is measured over the music,
excluding the pitch cue and count-in. See [score format](score-format.md).

## Synthesis settings

SoulX-Singer runs in score mode with automatic transposition disabled. The
renderer maps each syllable to an English phone sequence and carries it through
melismas. Version 1 scores omit lexical stress, so the renderer uses primary
stress for vowels and maps schwa (`ax`) to unstressed `AH0`. The model generates
24 kHz mono audio. Autotrack resamples each phrase to 44.1 kHz and places it
at its score offset before mixing.

- `--steps N` sets the diffusion step count (default 32).
- `--device cpu` is the default. `--device cuda` selects the default NVIDIA GPU.
  Use `--device cuda:1` for the second visible GPU.
- `--threads N` controls PyTorch CPU threads (default up to eight).
- `--phrase-seconds N` sets the grouping span (1–60 seconds, default 13).
  Gaps over 0.4 seconds end a phrase. A single syllable stays together.
- `--seed N` sets the PyTorch seed for each phrase (default 42).

## Reference singer and installation

The default reference is the upstream English audio and its aligned metadata
in `example/audio/`. To use a different singer, pass `--prompt-audio FILE`
and `--prompt-metadata FILE` together. Supply the original recording that the
metadata timestamps refer to. Autotrack extracts the first annotated segment
using its start and end in milliseconds. To align a new recording, follow
the [upstream preprocessing instructions](https://github.com/Soul-AILab/SoulX-Singer#readme).

The `singer` extra installs inference dependencies alongside Autotrack.
On Linux, for smaller CPU-only wheels, run this in the same environment
before installing Autotrack:

```sh
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.0 torchaudio==2.2.0
```

For an NVIDIA GPU, use a compatible CUDA build of those packages and select
the GPU with `--device`. The one-time `autotrack download-voice` command installs
the pinned upstream code, reference and verified checkpoint in the user's data
directory. Re-running it checks the installed assets.

For an existing installation, `--soulx-root PATH` selects the upstream code and
`--checkpoint PATH` selects the model file. Inference uses Autotrack's Python
interpreter by default. Older installations that already have a dedicated
interpreter under the SoulX checkout can continue using it when PyTorch is
absent from Autotrack's environment. `--soulx-python PATH` explicitly selects
an interpreter. `autotrack doctor` shows Autotrack's interpreter and data path.
