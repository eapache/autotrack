# Structured score format

Version 1 JSON contains a nonempty `title`, `duration_beats`, `count_in_beats`,
`tempo`, and exactly four `parts`: tenor, lead, baritone, bass. Beats are
quarter-note units; beat zero is the start of the music, excluding the count-in
and pitch cue. Optional `tonic_midi` specifies the pitch-pipe note (integer
0–127). The pitch cue is added during mixing, not to the score timeline.

Each part is an ordered list of syllable groups. Each group contains `lyric`,
`phonemes` (an array of lowercase stress-free ARPAbet tokens), and `notes`.
Each note has `beat`, `duration`, and an integer sounding `midi` pitch (0–127).
Notes within a syllable must be contiguous; gaps between syllables are rests.
Notes cannot overlap within a part or extend beyond `duration_beats`.
A single syllable may last at most 30 seconds at rendering time.

Tempo entries contain `beat`, positive `bpm`, and optional `curve` (`step`, the
default, or `linear` to the next point). They must be strictly ordered and start
at beat zero. Linear BPM ramps are integrated analytically. The initial tempo
determines the count-in. MusicXML import emits step changes; edit the JSON to
add a gradual ritardando.

Use `autotrack import-xml` on the included example to obtain a complete editable
score. `autotrack validate` checks structure and note timing; rendering additionally
checks model phonemes, frame budgets, and maximum syllable duration.
