import pickle
from pprint import pprint
from typing import List, Dict

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def read_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def filter_dict_keys(el, keys_to_keep):
    return {k:el[k] for k in el if k in keys_to_keep}


def filter_dicts_array_to_df(data, keys_to_keep):
    return pd.DataFrame([filter_dict_keys(el, keys_to_keep) for el in data])


def adl_parse_genres(data: pd.DataFrame):
    working_copy = data.copy(deep=True)
    working_copy[['genre', 'subgenre', 'author', 'title']] = working_copy['filename'].str.lstrip('datasets/adl-piano-midi/').str.rstrip('.mid').str.split(pat='/', expand=True)

    return working_copy


def keep_most_probable_key(data: List):
    def filter_el(piece: Dict):
        return {**piece, 'key': piece['analyzed_keys'][0]['key']}

    return [filter_el(piece) for piece in data]


def keep_most_probable_tempo(data: List):
    def filter_el(piece: Dict):
        tempos = piece['tempo_estimates'][0]
        if len(tempos) > 0:
            return {**piece, 'tempo': tempos[0]}
        else:
            return {**piece, 'tempo': None}

    return [filter_el(piece) for piece in data]


def check_which_pieces_change_key_time_signature(data: List):
    def filter_el(piece: Dict):
        return {**piece,
                'changes_key': str(len(piece['key_signature_changes']) > 1),
                'changes_time_signature': str(len(piece['time_signature_changes']) > 1)}

    return [filter_el(piece) for piece in data]


def histogram_lengths(data: pd.DataFrame, dataset: str, xlim = None):
    plt.grid(True)
    sns.histplot(data, x='length')
    plt.title(f'{dataset} - Total length: {np.round(data["length"].sum() / 60, 2)} hours')
    plt.xlabel('Length [minutes]')
    if xlim is not None:
        plt.xlim(xlim)
    plt.show()


def histogram_tempos(data: pd.DataFrame, dataset: str, ylim = None):
    plt.grid(True)
    sns.histplot(data, x='tempo', y='length')
    plt.title(f'{dataset} - Total length: {np.round(data["length"].sum() / 60, 2)} hours')
    plt.xlabel('Tempo [BPM]')
    plt.ylabel('Length [minutes]')
    if ylim is not None:
        plt.ylim(ylim)
    plt.show()


def violin(data: pd.DataFrame, dataset: str, group_by: str):
    plt.figure(figsize=(8, 8))
    plt.title(f'{dataset} - lengths by {group_by}')
    sns.violinplot(data[data['length'] <= 10], hue=group_by, x='length', y=group_by)
    plt.show()


def columns(data: pd.DataFrame, dataset: str, group_by: str):
    working_copy = data.copy(deep=True)
    working_copy['length'] /= 60

    plt.title(f'{dataset} - lengths according to {group_by}')
    plt.grid(True)
    sns.barplot(working_copy, y=group_by, x='length', hue='are_all_keyboards', estimator=sum, errorbar=None)
    plt.xlabel('Length [hours]')
    plt.show()


def note_distribution(note_map, dataset: str):
    import matplotlib.ticker as tkr

    formatter = tkr.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))

    ax = sns.heatmap(
        np.where(note_map==0, np.nan, note_map),
        xticklabels=['C','C#\nDb','D','D#\nEb','E','F','F#\nGb','G','G#\nAb','A','A\nBb','B'],
        cbar_kws={"format": formatter}
    )
    ax.invert_yaxis()
    plt.title(f"{dataset} note distribution - C4 is Middle C")
    plt.show()


def note_distribution_3d(note_map):
    X = np.arange(0, 12)
    Y = np.arange(0, 8)
    X, Y = np.meshgrid(X, Y)
    X.ravel()
    Y.ravel()
    note_map.ravel()

    plt.figure()
    ax = plt.axes(projection='3d')
    # x geograficamente "di lato"
    # y geograficamente "profondità"
    # z sono le "altezze" / numero di elementi / conteggio
    ax.bar3d(X.ravel(), Y.ravel(), 0, 0.5,0.5, note_map.ravel())
    plt.show()


if __name__ == '__main__':
    filename = 'vgmusic.pickle'
    dataset_human_name = 'VGMusic'

    keyboard_instruments = [
        'Piano','ElectricPiano','Harpsichord','Clavichord',
        'Celesta','Organ','Accordion',
        'Vibraphone','Sampler','Glockenspiel','Harp','Instrument'
    ]
    keyboard_pairs = []
    for k1 in keyboard_instruments:
        for k2 in keyboard_instruments:
            keyboard_pairs.append(f'{k1},{k2}')

    collected_data = read_pickle(filename)
    collected_data = keep_most_probable_key(collected_data)
    collected_data = keep_most_probable_tempo(collected_data)
    collected_data = check_which_pieces_change_key_time_signature(collected_data)

    if 'vgm' in filename:
        collected_data = [el for el in collected_data if len(el['music21_instruments']) < 8 and el['length'] < 25*60]

    pprint(collected_data[0])

    res = filter_dicts_array_to_df(collected_data, [
        'duration', 'length', 'filename', 'key', 'changes_key', 'changes_time_signature',
        'music21_instruments', 'tempo'
    ])

    res['instruments_number'] = res['music21_instruments'].str.len()
    res['instruments_classes'] = res['music21_instruments'].apply(lambda x: [i['music21_instrument_class'] for i in x])

    print(res['instruments_classes'].str.join(',').value_counts()[:40])
    res['is_two_keyboards'] = res['instruments_classes'].apply(lambda instrs: str(len(instrs) <= 2 and ','.join(instrs) in keyboard_pairs + keyboard_instruments))
    res['are_all_keyboards'] = res['instruments_classes'].apply(lambda instrs: str(all([i in keyboard_instruments for i in instrs])))

    if 'vgm' in filename:
        res = res[res['are_all_keyboards'] == 'True']

    if 'adl' in filename:
        res = adl_parse_genres(res)
    res['length'] /= 60

    histogram_lengths(res, dataset_human_name, [0, 10])
    if 'adl' in filename:
        violin(res, dataset_human_name, 'genre')
        columns(res, dataset_human_name, 'genre')

    pprint(res['tempo'])
    columns(res, dataset_human_name, 'key')
    columns(res, dataset_human_name, 'changes_key')
    columns(res, dataset_human_name, 'changes_time_signature')
    columns(res, dataset_human_name, 'is_two_keyboards')
    columns(res, dataset_human_name, 'are_all_keyboards')

    histogram_tempos(res, dataset_human_name)

    keyboard_notes = sum([instr['notes_count'] for instrs in res[res['are_all_keyboards'] == 'True']['music21_instruments'] for instr in instrs])

    note_distribution_3d(keyboard_notes)
    note_distribution(keyboard_notes, dataset_human_name)

