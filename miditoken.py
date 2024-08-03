import typing

import music21
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from music21.pitch import Pitch
from music21.stream import Score
import tqdm


def score_to_chordwise(s: Score, note_range=12 * 8, sample_freq=12, num_instruments=1):
    """
    :param s: The score to tokenize
    :param note_range: The range of the notes, by default is 8 octaves
                       (therefore a 96-keys piano, pretty much the absolute maximum a piano can have)
    :param sample_freq: Sampling frequency (# of samples per quarter note)
    :param num_instruments: How many instruments has the score
    :return: "Chordwise" representation (list of strings)
    """
    # Calculate number of samples according to given sampling frequency
    piece_samples = int(np.floor(s.duration.quarterLength * sample_freq) + 1)
    # Create "piano roll" matrix to hold chords
    pr_matrix = np.zeros((piece_samples, num_instruments, note_range))

    # Extract all note events from the score
    note_events = []
    for n in s.flatten().notes:
        note_events += NoteEvent.from_pitches(n.pitches, n.offset, n.duration.quarterLength, n.volume, sampling_freq=sample_freq)

    for n in note_events:
        pr_matrix[n.timestep, n.instrument, n.pitch] = 1  # Strike note
        pr_matrix[n.timestep + 1:n.timestep + n.duration, n.instrument, n.pitch] = 2  # Continue holding note

    chordwise_list = []
    for timestep in pr_matrix:
        for instr_idx in list(reversed(range(len(timestep)))):
            # Cast single notes to string and build the chord string
            chordwise_list.append('p' + ''.join([str(int(note)) for note in timestep[instr_idx]]))

    return chordwise_list


def augment_with_modulations(chordwise_repr: typing.List) -> typing.List:
    modulations = []
    note_range = len(chordwise_repr[1:])
    for i in range(0, 12):
        modulation = []
        for chord in chordwise_repr:
            prefix = chord[0]
            padded = '000000' + chord[1:] + '000000'
            modulation.append(prefix + padded[i:i + note_range])
        modulations.append(modulation)
    return modulations


def chordwise_to_notewise(chordwise_repr, sample_freq=12, max_samples_wait=None):
    if max_samples_wait is None:
        max_samples_wait = sample_freq * 2

    def find_next_chord(curr_prefix, curr_chord_idx) -> str|None:
        for k in range(curr_chord_idx + 1, len(chordwise_repr)):
            if chordwise_repr[k][0] == curr_prefix:
                return chordwise_repr[k][1:]
        return None

    text_tokens = []
    for chord_index in range(len(chordwise_repr)):
        chord = chordwise_repr[chord_index]
        prefix, chord = chord[0], chord[1:]
        next_chord = find_next_chord(prefix, chord_index)
        for i in range(len(chord)):
            # Rest! Ignore...
            if chord[i] == "0":
                continue
            note = prefix + str(i)
            # New note! We are pressing the key (1)
            if chord[i] == "1":
                text_tokens.append(note)
            # Piece ended OR we raise the key (2). Either way the note ends!
            if next_chord is None or next_chord[i] == "0":
                text_tokens.append("end" + note)

        # We always append the wait - so we can keep track of the measures in the next lines of code
        text_tokens.append("wait")

    i = 0
    translated_string = ""
    while i < len(text_tokens):
        wait_count = 1
        if text_tokens[i] == 'wait':
            # Aggregate the waits (until 'max_samples_wait')
            while wait_count <= max_samples_wait and i + wait_count < len(text_tokens) and text_tokens[i + wait_count] == 'wait':
                wait_count += 1
            text_tokens[i] = 'wait' + str(wait_count)
        # Concatenate the tokens while we're at it
        translated_string += text_tokens[i] + " "
        i += wait_count

    return translated_string


class NoteEvent:
    def __init__(self, pitch: Pitch, offset, duration, volume, instrument=0, sampling_freq=12):
        self.pitch = pitch.octave * 12 + pitch.pitchClass
        self.timestep = int(np.floor(offset * sampling_freq))
        self.duration = int(np.floor(duration * sampling_freq))
        self.instrument = instrument
        self.volume = np.round(volume.velocity / 127.0, 2)

        self.sampling_freq = sampling_freq

    @classmethod
    def from_pitches(cls, pitches: tuple[Pitch, ...], offset, duration, volume, instrument=0, sampling_freq=12):
        return [NoteEvent(p, offset, duration, volume, instrument, sampling_freq) for p in pitches]


def chordwise_pianoroll(chordwise_array):
    piano_roll = np.zeros((12*8, len(chordwise_array)))
    for i,ch in enumerate(chordwise_array):
        piano_roll[:, i] = [int(c) for c in ch.lstrip("p")]

    plt.figure(figsize=(20,8))
    ax = sns.heatmap(piano_roll)
    ax.invert_yaxis()
    plt.show()


def write_to_file(filename, contents):
    with open(filename, "w") as f:
        f.write(contents)


def score_to_representations(s: Score, filename: str, augment=True, debug=False, sample_freq=12):
    chordwise_repr = score_to_chordwise(s, sample_freq=sample_freq)
    chordwise_reprs = [chordwise_repr]

    if debug:
        chordwise_pianoroll(chordwise_repr)

    if augment:
        modulations = augment_with_modulations(chordwise_repr)
        chordwise_reprs = modulations

    for i,r in enumerate(chordwise_reprs):
        write_to_file(f'{filename}-{i}.chordwise', ' '.join(r))
        notewise_repr = chordwise_to_notewise(r, sample_freq=sample_freq)
        write_to_file(f'{filename}-{i}.notewise', notewise_repr)


def read_directories(sample_freq=12):
    import os

    directory = "datasets"

    for root, dirs, files in tqdm.tqdm(os.walk(directory)):
        if root.startswith('datasets/merged') and root != 'datasets/merged':
            genre = root.split('/')[-1]
            for file in tqdm.tqdm(files, leave=False):
                score = music21.converter.parse(os.path.join(root, file))
                score_to_representations(score, "texts/" + genre + "/" + file.strip('.midi').strip('.mid'), sample_freq=sample_freq)


def vocabulary(extension: str):
    import os

    voc = {}

    for root, _, files in tqdm.tqdm(os.walk("texts")):
        for file in tqdm.tqdm(files, leave=False):
            if file.endswith(extension):
                with open(os.path.join(root, file), 'r') as f:
                    contents = f.read()
                    for word in contents.split(" "):
                        if word not in voc:
                            voc[word] = 0
                        else:
                            voc[word] += 1

    print(len(voc))


if __name__ == '__main__':
    read_directories()
    vocabulary(".notewise")
    vocabulary(".chordwise")
