"""Import explicit MusicXML voices; never infer parts from pitch ordering."""
import re
import copy
from collections import Counter
from pathlib import Path
from music21 import converter, stream, note, chord, tempo, expressions, key
from .score import PARTS, validate


def read(path, expand=True):
    if Path(path).suffix.lower() not in ('.xml', '.musicxml', '.mxl'):
        raise ValueError('Export .musicxml or .mxl from MuseScore first')
    try:
        score = converter.parse(str(Path(path).resolve()), forceSource=True)
    except Exception as exc:
        raise ValueError(f'Could not read MusicXML: {exc}') from exc
    if not isinstance(score, stream.Score):
        raise ValueError('Expected a MusicXML score')
    # Keep source identities through repeat expansion, including alternate endings.
    expected = set()
    for part_index, part in enumerate(score.parts):
        for measure_index, measure in enumerate(part.getElementsByClass(stream.Measure)):
            measure._autotrack_origin = (part_index, measure_index)
            ids = set()
            lane_ids = {}
            for note_index, n in enumerate(measure.recurse().notes):
                n._autotrack_origin = (part_index, measure_index, note_index)
                voice = n.getContextByClass(stream.Voice)
                lane = str(voice.id) if voice is not None else '0'
                n._autotrack_lane = lane
                lane_ids.setdefault(lane, set()).update(lyric.number for lyric in n.lyrics)
                for lyric in n.lyrics:
                    ids.add(lyric.number)
                    expected.add((n._autotrack_origin, lyric.number))
            measure._autotrack_verses = sorted(ids)
            for n in measure.recurse().notes:
                n._autotrack_verse_ids = sorted(lane_ids[n._autotrack_lane])
    if expand:
        try:
            score = score.expandRepeats()
        except Exception as exc:
            raise ValueError(f'Could not expand repeats; unfold them in MuseScore first: {exc}') from exc
        measures = [m for part in score.parts for m in part.getElementsByClass(stream.Measure)]
        occurrences = Counter(m._autotrack_origin for m in measures)
        verse_count = max((len(n._autotrack_verse_ids) for m in measures for n in m.recurse().notes), default=1)
        if max(occurrences.values(), default=1) == 1 and verse_count > 1:
            # Strophic notation without repeats: sing each verse through the score.
            span = float(score.highestTime)
            for part in score.parts:
                original = list(part)
                for verse_pass in range(1, verse_count):
                    for element in original:
                        part.insert(verse_pass * span + float(element.offset), copy.deepcopy(element))
    visits = Counter()
    sung = set()
    for part in score.parts:
        for measure in part.getElementsByClass(stream.Measure):
            origin = measure._autotrack_origin
            verse_pass = visits[origin]
            visits[origin] += 1
            for n in measure.recurse().notes:
                ids = n._autotrack_verse_ids
                verse = ids[min(verse_pass, len(ids) - 1)] if ids else None
                selected = [lyric for lyric in n.lyrics if lyric.number == verse]
                for lyric in selected:
                    sung.add((n._autotrack_origin, lyric.number))
                n.lyrics = selected
    if expand and expected - sung:
        raise ValueError('Some written verses are not reached by the repeat markings. '
                         'Adjust the repeat count or write out the song in performance order in your notation editor.')
    return score.toSoundingPitch(inPlace=False)


def lanes(score):
    result = {}
    def walk(container, offset, voice, part_index, name):
        for el in container:
            at = offset + float(el.offset)
            if isinstance(el, stream.Voice):
                walk(el, at, str(el.id), part_index, name)
            elif isinstance(el, stream.Stream):
                walk(el, at, voice, part_index, name)
            elif isinstance(el, (note.Note, chord.Chord)):
                key = f'{part_index}:{voice}'
                result.setdefault(key, {'name': name, 'notes': []})['notes'].append((at, el))
    for i, part in enumerate(score.parts, 1):
        walk(part, float(part.offset), '0', i, part.partName or str(part.id))
    for lane in result.values():
        lane['notes'].sort(key=lambda pair: pair[0])
    return result


def describe(score):
    rows = []
    for selector, lane in lanes(score).items():
        pitches = [n.pitch.midi for _, n in lane['notes'] if isinstance(n, note.Note)]
        rows.append({'selector': selector, 'name': lane['name'], 'notes': len(lane['notes']),
                     'midi_range': [min(pitches), max(pitches)] if pitches else None})
    return rows


def resolve_mapping(available, mapping):
    if mapping:
        result = dict(item.split('=', 1) for item in mapping.split(','))
    else:
        result = {}
        for key, lane in available.items():
            name = lane['name'].lower().strip()
            aliases = {'tenor': 'tenor', 'lead': 'lead', 'baritone': 'baritone', 'bari': 'baritone', 'bass': 'bass'}
            if name in aliases:
                if aliases[name] in result:
                    raise ValueError(f'Ambiguous {aliases[name]} voices; supply an explicit --map from inspect')
                result[aliases[name]] = key
    if set(result) != set(PARTS) or len(set(result.values())) != 4:
        raise ValueError('Map four distinct voices with --map tenor=1:1,lead=1:2,baritone=2:1,bass=2:2 (use inspect for actual selectors)')
    if any(key not in available for key in result.values()):
        raise ValueError(f'Mapping refers to absent voice; available: {", ".join(available)}')
    return result


def import_score(source, mapping=None, pronunciations=None, bpm=None, swing=1., count_in=3, missing_lyric=None, expand=True):
    import cmudict
    parsed = read(source, expand)
    available = lanes(parsed)
    mapped = resolve_mapping(available, mapping)
    dictionary = cmudict.dict()
    overrides = {k.lower(): v.split() if isinstance(v, str) else v for k, v in (pronunciations or {}).items()}
    unknown = set()
    warnings = []
    def phonemes(text):
        word = text.lower().strip('.,!?;:').replace('’', "'")
        if word in overrides:
            return overrides[word]
        if word in dictionary:
            return [re.sub(r'\d', '', p).lower() for p in dictionary[word][0]]
        unknown.add(text)
        return ['ah']  # Never saved: unknown pronunciations cause an error below.
    def warp(b):
        whole = int(b)
        frac = b - whole
        split = swing / (1 + swing)
        return whole + (frac * split * 2 if frac <= .5 else split + (frac - .5) * (1 - split) * 2)
    marks = {}
    for mark in parsed.recurse().getElementsByClass(tempo.MetronomeMark):
        speed = mark.getQuarterBPM()
        if speed is not None:
            at = float(mark.getOffsetInHierarchy(parsed))
            if at in marks and abs(marks[at] - speed) > 1e-5:
                raise ValueError(f'Conflicting tempo marks at beat {at}')
            marks[at] = float(speed)
    if bpm is not None:
        marks = {0.: bpm}
    if 0. not in marks:
        raise ValueError('No numeric initial tempo; provide --tempo (quarter notes per minute)')
    for text in parsed.recurse().getElementsByClass(expressions.TextExpression):
        if re.search(r'rit\.?|rall\.?|accel\.?|swing', text.content, re.I):
            warnings.append(f'Expressive text "{text.content}" is not interpreted; use --swing or edit the JSON tempo map.')
    out = {'version': 1, 'title': parsed.metadata.bestTitle if parsed.metadata and parsed.metadata.bestTitle else Path(source).stem,
           'count_in_beats': count_in, 'duration_beats': float(parsed.highestTime),
           'tempo': [{'beat': at, 'bpm': speed, 'curve': 'step'} for at, speed in sorted(marks.items())], 'parts': {},
           'source': Path(source).name, 'warnings': sorted(set(warnings))}
    # A key signature alone cannot distinguish relative major/minor.
    initial_keys = [k for k in parsed.recurse().getElementsByClass(key.KeySignature)
                    if float(k.getOffsetInHierarchy(parsed)) == 0]
    tonics = {k.tonic.pitchClass for k in initial_keys if isinstance(k, key.Key)}
    if initial_keys and all(isinstance(k, key.Key) for k in initial_keys) and len(tonics) == 1:
        out['tonic_midi'] = 60 + tonics.pop()
    for part, selector in mapped.items():
        groups = []
        for at, n in available[selector]['notes']:
            if isinstance(n, chord.Chord):
                raise ValueError(f'{part}, beat {at}: chord in a vocal line; separate voices in MuseScore')
            duration = float(n.duration.quarterLength)
            if duration <= 0 or n.duration.isGrace:
                raise ValueError(f'{part}, beat {at}: grace notes need explicit performed durations')
            if abs(n.pitch.ps - round(n.pitch.ps)) > 1e-7:
                raise ValueError('Microtonal notes are not supported')
            # Tuplets keep their explicitly notated timing.
            a, b = (at, at + duration) if n.duration.tuplets else (warp(at), warp(at + duration))
            event = {'beat': a, 'duration': b - a, 'midi': round(n.pitch.midi)}
            lyric = n.lyrics[0].text if n.lyrics else None  # Selected during unfolding.
            contiguous = bool(groups) and abs(groups[-1]['notes'][-1]['beat'] + groups[-1]['notes'][-1]['duration'] - a) < 1e-6
            continuation = n.tie is not None and n.tie.type in ('stop', 'continue')
            continuation |= lyric in ('~', '_')
            # MusicXML omits lyric text on melisma continuation notes.
            continuation |= not lyric and contiguous
            if continuation:
                if not contiguous:
                    raise ValueError(f'{part}, beat {at}: lyric continuation without a preceding syllable')
                groups[-1]['notes'].append(event)
            else:
                lyric = lyric or missing_lyric
                if not lyric:
                    raise ValueError(f'{part}, beat {at}: missing lyric; add lyrics or pass --missing-lyric doo')
                groups.append({'lyric': lyric, 'phonemes': phonemes(lyric), 'notes': [event]})
        out['parts'][part] = groups
    if unknown:
        raise ValueError('Add these syllables to --pronunciations JSON (ARPAbet): ' + ', '.join(sorted(unknown)))
    return validate(out)
