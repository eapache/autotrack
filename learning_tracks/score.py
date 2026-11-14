"""Validated, engine-independent score format and exact tempo integration."""
import json
import math
from pathlib import Path

PARTS = ('tenor', 'lead', 'baritone', 'bass')
VOWELS = set('aa ae ah ao aw ax ay eh er ey ih iy ow oy uh uw'.split())


def positive(value, label, allow_zero=False):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f'{label} must be a finite {"nonnegative" if allow_zero else "positive"} number')


def validate(score):
    if not isinstance(score, dict):
        raise ValueError('Score must be a JSON object')
    if not isinstance(score.get('title'), str) or not score['title'].strip():
        raise ValueError('Score must have a nonempty title')
    if 'tonic_midi' in score and (type(score['tonic_midi']) is not int or not 0 <= score['tonic_midi'] <= 127):
        raise ValueError('tonic_midi must be an integer MIDI note from 0 to 127')
    if score.get('version') != 1:
        raise ValueError('Expected score version 1')
    if set(score.get('parts', {})) != set(PARTS):
        raise ValueError(f'Score must contain exactly these parts: {", ".join(PARTS)}')
    positive(score.get('count_in_beats', 0), 'count_in_beats', True)
    positive(score.get('duration_beats'), 'duration_beats')
    previous = -1
    for point in score.get('tempo', []):
        positive(point['beat'], 'tempo beat', True)
        positive(point['bpm'], 'tempo bpm')
        if point['beat'] <= previous:
            raise ValueError('Tempo points must be strictly increasing')
        if point.get('curve', 'step') not in ('step', 'linear'):
            raise ValueError('Tempo curve must be step or linear')
        previous = point['beat']
    if not score.get('tempo') or score['tempo'][0]['beat'] != 0:
        raise ValueError('Tempo map must begin at beat 0')
    for part, groups in score['parts'].items():
        if not groups:
            raise ValueError(f'{part}: no sung notes')
        end = 0
        for i, group in enumerate(groups):
            phones = group.get('phonemes', [])
            if not phones or not all(isinstance(p, str) for p in phones) or not any(p in VOWELS for p in phones):
                raise ValueError(f'{part} syllable {i + 1}: phonemes must include a vowel')
            if not group.get('notes'):
                raise ValueError(f'{part} syllable {i + 1}: no notes')
            last = None
            for note in group['notes']:
                start, duration, pitch = note['beat'], note['duration'], note['midi']
                positive(start, 'note beat', True)
                positive(duration, 'note duration')
                if not isinstance(pitch, int) or not 0 <= pitch <= 127:
                    raise ValueError('MIDI pitch must be an integer from 0 to 127')
                if start < end - 1e-7:
                    raise ValueError(f'{part}: overlapping notes at beat {start}')
                if last is not None and abs(start - last) > 1e-7:
                    raise ValueError(f'{part}: a syllable cannot extend across a rest')
                end = start + duration
                last = end
            if end > score['duration_beats'] + 1e-7:
                raise ValueError(f'{part}: note extends past score duration')
    return score


def load(path):
    return validate(json.loads(Path(path).read_text()))


class Timeline:
    def __init__(self, score):
        self.points = score['tempo']
        self.count_in = score.get('count_in_beats', 0) * 60 / self.points[0]['bpm']

    def seconds(self, beat):
        result = self.count_in
        for i, p in enumerate(self.points):
            if beat <= p['beat']:
                break
            nxt = self.points[i + 1] if i + 1 < len(self.points) else None
            span = min(beat, nxt['beat']) - p['beat'] if nxt else beat - p['beat']
            slope = (nxt['bpm'] - p['bpm']) / (nxt['beat'] - p['beat']) if nxt and p.get('curve') == 'linear' else 0
            result += 60 * math.log1p(slope * span / p['bpm']) / slope if abs(slope) > 1e-12 else 60 * span / p['bpm']
            if not nxt or beat <= nxt['beat']:
                break
        return result

    def groups(self, groups):
        return [{**g, 'notes': [(self.seconds(n['beat']), self.seconds(n['beat'] + n['duration']), n['midi']) for n in g['notes']]} for g in groups]
