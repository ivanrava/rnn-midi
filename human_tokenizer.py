import os
from typing import List, Tuple

import music21.converter
from music21.chord import Chord
from music21.note import Note, Rest
from music21.pitch import Pitch
from music21.stream import Score, Part
import tqdm

from miditoken import write_to_file


def concatenate_pitches(pitches: Tuple[Pitch, ...]):
    return '|'.join(sorted([p.nameWithOctave for p in pitches]))

def encode_pitch(pitch: Pitch):
    return pitch.octave * 12 + pitch.pitchClass

def lowest_pitch(pitches: Tuple[Pitch, ...]):
    minimum = 100
    for pitch in pitches:
        if encode_pitch(pitch) < minimum:
            minimum = encode_pitch(pitch)
    return minimum

def part_to_human(part: Part):
    tokens = []
    minimum = 100
    for event in part.flatten().notesAndRests:
        if type(event) == Note:
            minimum = min(minimum, lowest_pitch(event.pitches))
            tokens.append(concatenate_pitches(event.pitches) + "|" + str(event.duration.quarterLength))
        elif type(event) == Chord:
            minimum = min(minimum, lowest_pitch(event.pitches))
            tokens.append(concatenate_pitches(event.pitches) + "|" + str(event.duration.quarterLength))
        elif type(event) == Rest:
            tokens.append('r|' + str(event.duration.quarterLength))
    return tokens, minimum

def score_to_human(score: Score) -> List[List[str]]:
    t, min_treb = part_to_human(score.parts[0])
    b, min_bass = part_to_human(score.parts[1])
    return [t, b] if min_treb > min_bass else [b, t]

if __name__ == '__main__':
    # trebles = []
    # basses = []

    for root, dirs, files in tqdm.tqdm(os.walk('datasets')):
        if root.startswith('datasets/merged') and root != 'datasets/merged':
            genre = root.split('/')[-1]
            for file in tqdm.tqdm(files, leave=False):
                if file.endswith('.midi'):
                    continue
                s = music21.converter.parse(os.path.join(root, file))
                if len(s.parts) != 2:
                    continue
                treble, bass = score_to_human(s)

                for w in treble:
                    if '/' in w:
                        break
                else:
                    write_to_file(f'human/{genre}/{file.rstrip(".mid")}.humanwise.treble', treble)

                for w in bass:
                    if '/' in w:
                        break
                else:
                    write_to_file(f'human/{genre}/{file.rstrip(".mid")}.humanwise.bass', bass)

    # print(set(sorted([note for treble in trebles for note in treble])))
    # print(set(sorted([note for bass in basses for note in bass])))
