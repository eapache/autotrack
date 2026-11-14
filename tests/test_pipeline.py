import copy
import json
import math
import tempfile
import unittest
import shutil
import subprocess
from pathlib import Path
import numpy as np
from learning_tracks.score import PARTS, Timeline, validate
from learning_tracks.render import frame_durations
from learning_tracks.mix import split_mix, export
from learning_tracks.musicxml import read, describe, import_score


def simple_score():
    return {'version': 1, 'title': 'Test', 'duration_beats': 4, 'count_in_beats': 0,
            'tempo': [{'beat': 0, 'bpm': 60}],
            'parts': {p: [{'lyric': 'doo', 'phonemes': ['d', 'uw'], 'notes': [{'beat': 0, 'duration': 4, 'midi': 60}]}] for p in PARTS}}


class ScoreTests(unittest.TestCase):
    def test_tempo_ramp_is_integrated_not_sampled(self):
        s = simple_score()
        s['tempo'] = [{'beat': 0, 'bpm': 60, 'curve': 'linear'}, {'beat': 4, 'bpm': 120}]
        self.assertAlmostEqual(Timeline(s).seconds(4), 4 * math.log(2), places=12)

    def test_overlaps_and_melisma_across_rest_are_rejected(self):
        for start in (2, 5):
            s = simple_score()
            s['duration_beats'] = 8
            s['parts']['lead'][0]['notes'].append({'beat': start, 'duration': 1, 'midi': 62})
            with self.assertRaises(ValueError):
                validate(s)

    def test_rest_between_syllables_allowed(self):
        s = simple_score()
        s['duration_beats'] = 8
        s['parts']['lead'].append({'lyric': 'doo', 'phonemes': ['d', 'uw'], 'notes': [{'beat': 6, 'duration': 1, 'midi': 62}]})
        validate(s)

    def test_short_syllable_fails_instead_of_extending_timing(self):
        with self.assertRaises(ValueError):
            frame_durations(['s', 't', 'r', 'eh', 'ng', 'th', 's'], 5)
        self.assertEqual(frame_durations(['ow'], 20), [20])
        durations = frame_durations(['k', 'r', 'ih', 's'], 24)
        self.assertEqual(sum(durations), 24)
        self.assertGreaterEqual(min(durations), 1)
        self.assertGreater(durations[2], durations[0])


class MixTests(unittest.TestCase):
    def setUp(self):
        t = np.arange(44100) / 44100
        self.tracks = {p: .1 * np.sin(2 * np.pi * f * t) for p, f in zip(PARTS, (220, 330, 440, 550))}

    def test_each_target_is_isolated_with_requested_balance(self):
        for target in PARTS:
            for side in ('left', 'right'):
                y = split_mix(self.tracks, target, 4.5, side)
                col = 0 if side == 'left' else 1
                x = self.tracks[target]
                gain = np.dot(y[:, col], x) / np.dot(x, x)
                np.testing.assert_allclose(y[:, col], gain * x, atol=1e-12)
                other = sum(x for p, x in self.tracks.items() if p != target)
                gain = np.dot(y[:, 1-col], other) / np.dot(other, other)
                np.testing.assert_allclose(y[:, 1-col], gain * other, atol=1e-12)
                rms = np.sqrt(np.mean(y * y, axis=0))
                self.assertAlmostEqual(20 * np.log10(rms[col] / rms[1-col]), 4.5)
                self.assertLess(np.max(np.abs(y)), 1.)

    @unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg required')
    def test_five_mp3s_with_centered_tonic_and_preserved_balance(self):
        import soundfile as sf
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            stems, mixes = folder / 'mono', folder / 'mixes'
            stems.mkdir()
            for part, x in self.tracks.items():
                sf.write(stems / f'{part}.wav', x, 44100, subtype='PCM_24')
            score = simple_score()
            score['duration_beats'] = 1
            result = export(stems, mixes, score, 5., 'right', pitch_midi=58)
            self.assertEqual({f.name for f in mixes.iterdir()},
                             {f'{p}.mp3' for p in (*PARTS, 'full_mix')})
            self.assertEqual({m['frames'] for m in result.values()}, {round(4.5 * 44100)})
            for part in PARTS:
                self.assertAlmostEqual(result[part]['dominance_db'], 5, places=3)
            decoded = []
            for part in (*PARTS, 'full_mix'):
                raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(mixes / f'{part}.mp3'),
                                               '-f', 'f32le', '-acodec', 'pcm_f32le', '-'])
                audio = np.frombuffer(raw, dtype='<f4').reshape(-1, 2)
                self.assertEqual(len(audio), round(4.5 * 44100))
                self.assertTrue(np.isfinite(audio).all())
                self.assertLess(np.max(np.abs(audio)), 1.)
                np.testing.assert_allclose(audio[:44100, 0], audio[:44100, 1], atol=1e-6)
                spectrum = np.abs(np.fft.rfft(audio[44100:88200, 0]))
                self.assertAlmostEqual(np.argmax(spectrum), 440 * 2 ** ((58-69)/12), delta=1)
                self.assertLess(np.max(np.abs(audio[round(3.1*44100):round(3.4*44100)])), 1e-5)
                decoded.append(audio)
            np.testing.assert_allclose(decoded[-1][:, 0], decoded[-1][:, 1], atol=1e-6)

    def test_missing_key_requires_explicit_pitch(self):
        with self.assertRaisesRegex(ValueError, 'unambiguous'):
            export('/unused', '/unused', simple_score())

    def test_pitch_validation(self):
        from learning_tracks.mix import pitch_tone
        for seconds in (-1, float('nan'), 11):
            with self.assertRaises(ValueError):
                pitch_tone(60, seconds)
        self.assertEqual(len(pitch_tone(60, 0)), 0)

    def test_length_mismatch_and_silent_backing_fail(self):
        tracks = dict(self.tracks)
        tracks['lead'] = tracks['lead'][:-1]
        with self.assertRaises(ValueError):
            split_mix(tracks, 'lead')
        tracks = {p: x if p == 'lead' else np.zeros_like(x) for p, x in self.tracks.items()}
        with self.assertRaises(ValueError):
            split_mix(tracks, 'lead')


class MusicXMLTests(unittest.TestCase):
    def fixture(self, folder, shared=False, compressed=False, unknown=False):
        from music21 import stream, note, meter, tempo, tie
        s = stream.Score()
        for i in range(2 if shared else 4):
            p = stream.Part(id=f'P{i+1}')
            p.partName = ('Upper' if i == 0 else 'Lower') if shared else PARTS[i]
            m = stream.Measure(number=1)
            m.append(meter.TimeSignature('4/4'))
            if i == 0:
                m.insert(0, tempo.MetronomeMark(number=75))
            for v in range(2 if shared else 1):
                container = stream.Voice(id=str(v + 1)) if shared else m
                first = note.Note(60 + i * 2 + v, quarterLength=1)
                first.lyric = 'nonsenselyric' if unknown else 'time'
                first.tie = tie.Tie('start')
                second = note.Note(first.pitch, quarterLength=1)
                second.tie = tie.Tie('stop')
                container.append(first)
                container.append(second)
                container.append(note.Rest(quarterLength=1))
                last = note.Note(62 + i * 2 + v, quarterLength=1)
                last.lyric = 'here'
                container.append(last)
                if shared:
                    m.insert(0, container)
            p.append(m)
            s.append(p)
        path = Path(folder) / ('fixture.mxl' if compressed else 'fixture.musicxml')
        s.write('musicxml', fp=path, compress=compressed)
        return path

    def test_four_parts_import_ties_and_preserve_rests(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.fixture(temp)
            out = import_score(path)
            self.assertEqual(out['tempo'][0]['bpm'], 75)
            self.assertEqual(len(out['parts']['lead']), 2)
            self.assertEqual(len(out['parts']['lead'][0]['notes']), 2)
            self.assertEqual(out['parts']['lead'][1]['notes'][0]['beat'], 3)

    def test_shared_staff_voices_require_explicit_mapping(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.fixture(temp, shared=True)
            rows = describe(read(path))
            self.assertEqual({r['selector'] for r in rows}, {'1:1', '1:2', '2:1', '2:2'})
            with self.assertRaises(ValueError):
                import_score(path)
            out = import_score(path, mapping='tenor=1:1,lead=1:2,baritone=2:1,bass=2:2')
            self.assertEqual(out['parts']['lead'][0]['notes'][0]['midi'], 61)

    def test_initial_major_minor_and_ambiguous_key(self):
        from music21 import converter, key
        with tempfile.TemporaryDirectory() as temp:
            path = self.fixture(temp)
            for signature, expected in [(key.Key('B-', 'major'), 70),
                                        (key.Key('g', 'minor'), 67), (key.KeySignature(-2), None)]:
                score = converter.parse(path, forceSource=True)
                for part in score.parts:
                    for old in list(part.recurse().getElementsByClass(key.KeySignature)):
                        old.activeSite.remove(old)
                    part.measure(1).insert(0, copy.deepcopy(signature))
                score.write('musicxml', fp=path)
                self.assertEqual(import_score(path).get('tonic_midi'), expected)

    def test_repeat_is_unfolded(self):
        from music21 import converter, bar, stream
        with tempfile.TemporaryDirectory() as temp:
            path = self.fixture(temp)
            score = converter.parse(path)
            for part in score.parts:
                part.getElementsByClass(stream.Measure)[-1].rightBarline = bar.Repeat(direction='end', times=2)
            score.write('musicxml', fp=path)
            out = import_score(path)
            self.assertEqual(out['duration_beats'], 8)
            self.assertEqual(len(out['parts']['lead']), 4)

    def test_all_verses_with_and_without_repeats(self):
        from music21 import converter, bar, stream
        for repeated in (False, True):
            with self.subTest(repeated=repeated), tempfile.TemporaryDirectory() as temp:
                path = self.fixture(temp)
                score = converter.parse(path)
                for part in score.parts:
                    sung = [n for n in part.recurse().notes if n.lyrics]
                    sung[0].addLyric('day', lyricNumber=2)
                    sung[1].addLyric('gone', lyricNumber=2)
                    if repeated:
                        part.getElementsByClass(stream.Measure)[-1].rightBarline = bar.Repeat(direction='end', times=2)
                score.write('musicxml', fp=path)
                out = import_score(path)
                self.assertEqual(out['duration_beats'], 8)
                for groups in out['parts'].values():
                    self.assertEqual([g['lyric'] for g in groups], ['time', 'here', 'day', 'gone'])
                    self.assertEqual([g['notes'][0]['beat'] for g in groups], [0, 3, 4, 7])

    def test_three_verses_without_repeats_and_tempo_reset(self):
        from music21 import converter
        with tempfile.TemporaryDirectory() as temp:
            path = self.fixture(temp)
            score = converter.parse(path)
            for part in score.parts:
                for n in part.recurse().notes:
                    if n.lyrics:
                        n.addLyric('day', lyricNumber=2)
                        n.addLyric('night', lyricNumber=3)
            score.write('musicxml', fp=path)
            out = import_score(path)
            self.assertEqual(out['duration_beats'], 12)
            self.assertEqual([p['beat'] for p in out['tempo']], [0, 4, 8])
            self.assertEqual([g['lyric'] for g in out['parts']['lead']],
                             ['time', 'here', 'day', 'day', 'night', 'night'])

    def test_shared_staff_unchanged_words_repeat_alongside_new_verses(self):
        from music21 import converter, stream
        with tempfile.TemporaryDirectory() as temp:
            path = self.fixture(temp, shared=True)
            score = converter.parse(path)
            voice = score.parts[0].measure(1).getElementsByClass(stream.Voice)[1]
            for n in voice.notes:
                if n.lyrics:
                    n.addLyric('day', lyricNumber=2)
            score.write('musicxml', fp=path)
            out = import_score(path, mapping='tenor=1:1,lead=1:2,baritone=2:1,bass=2:2')
            self.assertEqual([g['lyric'] for g in out['parts']['lead']], ['time', 'here', 'day', 'day'])
            self.assertEqual([g['lyric'] for g in out['parts']['tenor']], ['time', 'here', 'time', 'here'])

    def test_unreached_verse_is_reported_instead_of_silently_omitted(self):
        from music21 import converter, bar, stream
        with tempfile.TemporaryDirectory() as temp:
            path = self.fixture(temp)
            score = converter.parse(path)
            for part in score.parts:
                for n in part.recurse().notes:
                    if n.lyrics:
                        n.addLyric('day', lyricNumber=2)
                        n.addLyric('night', lyricNumber=3)
                part.getElementsByClass(stream.Measure)[-1].rightBarline = bar.Repeat(direction='end', times=2)
            score.write('musicxml', fp=path)
            with self.assertRaisesRegex(ValueError, 'Some written verses'):
                import_score(path)

    def test_verse_change_with_first_and_second_endings(self):
        from music21 import stream, note, bar, spanner, tempo
        with tempfile.TemporaryDirectory() as temp:
            score = stream.Score()
            for name in PARTS:
                part = stream.Part()
                part.partName = name
                measures = []
                for number, text in enumerate(('time', 'here', 'gone'), 1):
                    measure = stream.Measure(number=number)
                    n = note.Note('C4', quarterLength=4)
                    n.addLyric(text, lyricNumber=2 if number == 3 else 1)
                    if number == 1:
                        n.addLyric('day', lyricNumber=2)
                        measure.insert(0, tempo.MetronomeMark(number=75))
                        measure.leftBarline = bar.Repeat(direction='start')
                    if number == 2:
                        measure.rightBarline = bar.Repeat(direction='end', times=2)
                    measure.insert(0, n)
                    part.append(measure)
                    measures.append(measure)
                part.insert(0, spanner.RepeatBracket(measures[1], number=1))
                part.insert(0, spanner.RepeatBracket(measures[2], number=2))
                score.insert(0, part)
            path = Path(temp) / 'endings.musicxml'
            score.write('musicxml', fp=path)
            out = import_score(path)
            self.assertEqual(out['duration_beats'], 16)
            for groups in out['parts'].values():
                self.assertEqual([g['lyric'] for g in groups], ['time', 'here', 'day', 'gone'])

    def test_compressed_xml_and_unknown_pronunciation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.fixture(temp, compressed=True, unknown=True)
            with self.assertRaisesRegex(ValueError, 'nonsenselyric'):
                import_score(path)
            out = import_score(path, pronunciations={'nonsenselyric': 'd uw'})
            self.assertEqual(out['parts']['lead'][0]['phonemes'], ['d', 'uw'])


if __name__ == '__main__':
    unittest.main()
