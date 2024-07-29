import os
import pickle

import music21
from music21.chord import Chord
from music21.instrument import Piano, Instrument
from music21.interval import Interval
from music21.key import KeySignature, Key
from music21.meter import TimeSignature
from music21.note import Note
from music21.pitch import Pitch
from music21.stream import Part, Score
import numpy as np
from pprint import pprint
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pretty_midi as pm


def sample_events(stream, sampling_frequency=12):
    """
    Sample the events from a Music21 stream
    :param stream: The Music21 stream
    :param sampling_frequency: How many samples should be collected for each quarter note
    """
    total_number_of_sampled_events = np.floor(stream.duration.quarterLength * sampling_frequency) + 1
    print(total_number_of_sampled_events)

def print_part(part: Part, number_in_score = 0):
    elements = part.flatten().elements
    print(f'Part #{number_in_score}:')
    print(part.getInstruments().flatten().elements)
    #print(f'\tInstrument: {elements[0].instrument}')

def count_part_notes(part: Part):
    note_count = np.zeros((8,12), dtype=int)
    def add_pitch(p: Pitch):
        note_count[p.octave, p.pitchClass] += 1

    for note in part.flatten().notes:
        if type(note) == Note:
            add_pitch(note.pitch)
        elif type(note) == Chord:
            for pitch in note.pitches:
                add_pitch(pitch)
    return note_count

    #for what in part.flatten().elements:
    #    if type(what) == Note:
    #        print(f'{what.duration} - {what.pitch.name}', end=', ')
    #    else:
    #        print(what, end=', ')
    #print()

def score_instruments(score: Score|Piano):
    return [
        {
            'notes_count': count_part_notes(part),
            **extract_instrument_metadata(part.flatten().elements[0]),
        } for part in score.parts
    ]


def extract_instrument_metadata(instrument: Instrument):
    return {
        'music21_instrument_class': instrument.__class__.__name__,
        'part_id': instrument.partId,
        'part_name': instrument.partName,
        'part_abbreviation': instrument.partAbbreviation,
        'instrument_id': instrument.instrumentId,
        'instrument_name': instrument.instrumentName,
        'instrument_abbreviation': instrument.instrumentAbbreviation,
        'midi_program': instrument.midiProgram,
        'midi_channel': instrument.midiChannel,
        'lowest_note': pitch_to_string(instrument.lowestNote),
        'highest_note': pitch_to_string(instrument.highestNote),
        'transposition': interval_to_string(instrument.transposition),
        'percussion_map': instrument.inGMPercMap
    }

def pitch_to_string(pitch: Pitch|None):
    if pitch is None:
        return None
    return pitch.nameWithOctave

def interval_to_string(interval: Interval):
    if interval is None:
        return None
    return interval.name


def parse_parts(score: Score):
    for score_part in score.parts:
        count_part_notes(score_part)

def time_signature_to_string(time_signature: TimeSignature):
    if time_signature is None:
        return None
    return time_signature.ratioString


def key_to_string(key: Key):
    return key.name if key is not None else None


def analyze_keys(score: Score):
    results = score.analyze('key')
    return [
        {'key': key_to_string(results), 'correlation': results.correlationCoefficient},
    ] + [
        {'key': key_to_string(alternate), 'correlation': alternate.correlationCoefficient} for alternate in results.alternateInterpretations
    ]


def read_midi(filename):
    midi = pm.PrettyMIDI(filename)
    score = music21.converter.parse(filename)

    return {
        'tempo_changes': midi.get_tempo_changes(),
        'tempo_estimates': midi.estimate_tempi(),
        'length': midi.get_end_time(),
        'filename': filename,
        'duration': score.duration.quarterLength,
        'time_signature': time_signature_to_string(score[music21.meter.TimeSignature][0]),
        'key': key_to_string(score[music21.key.KeySignature][0]),
        'parts': len(score.parts),
        'music21_instruments': score_instruments(score),
        'pm_instruments': [{'program': i.program, 'is_drum': i.is_drum, 'name': i.name} for i in midi.instruments],
        'analyzed_keys': analyze_keys(score),
        'key_signature_changes': [{'key_number': k.key_number, 'time': k.time} for k in midi.key_signature_changes],
        'time_signature_changes': [{'numerator': t.numerator, 'denominator': t.denominator, 'time': t.time} for t in midi.time_signature_changes]
    }
    # print(f'How many events should we sample: {sample_events(score)}')


def note_distribution(note_map):
    sns.heatmap(
        note_map,
        xticklabels=['C','C#\nDb','D','D#\nEb','E','F','F#\nGb','G','G#\nAb','A','A\nBb','B'],
        annot=True
    )
    plt.show()


def piano_roll(filename):
    midi = pm.PrettyMIDI(filename)
    sns.heatmap(midi.get_piano_roll())
    plt.show()

def analyze_midi_files(directory):
    arr = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.midi') or file.endswith('.mid'):
                descriptor = read_midi(os.path.join(root, file))
                arr.append(descriptor)
    return arr

def analyze_folder_and_save(directory, name):
    arr = analyze_midi_files(directory)
    pickle.dump(arr, open(name + '.pickle', 'wb'))

if __name__ == '__main__':
    analyze_folder_and_save("datasets/examples", "examples")
    # note_distribution(obj['music21_instruments'][0]['notes_count'] + obj['music21_instruments'][1]['notes_count'])