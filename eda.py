import pickle
from typing import List, Dict

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def read_pickle_raw(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def filter_dict_keys(el, keys_to_keep):
    return {k:el[k] for k in el if k in keys_to_keep}


def filter_raw_array_to_df(data, keys_to_keep):
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


def check_which_pieces_change_key_or_time_signature(data: List):
    def filter_el(piece: Dict):
        return {**piece,
                'changes_key': str(len(piece['key_signature_changes']) > 1),
                'changes_time_signature': str(len(piece['time_signature_changes']) > 1)}

    return [filter_el(piece) for piece in data]


def plot_lengths(df: pd.DataFrame):
    plt.grid(True)
    sns.histplot(data=df, x='length', hue='dataset')
    plt.title(f'Total length: {np.round(df["length"].sum() / 60, 2)} hours')
    plt.xlabel('Length [minutes]')
    plt.show()

    sns.barplot(data=df, x='dataset', y='length', hue='dataset', estimator=sum, errorbar=None)
    plt.title('Dataset lengths')
    plt.grid(True)
    plt.ylabel('Length [minutes]')
    plt.xlabel('')
    plt.show()


def plot_tempos(df: pd.DataFrame):
    sns.histplot(data=df, x='tempo', y='length', hue='dataset')
    plt.grid(True)
    plt.title(f'Total length: {np.round(df["length"].sum() / 60, 2)} hours')
    plt.xlabel('Tempo [BPM]')
    plt.ylabel('Length [minutes]')
    plt.show()


def plot_notes(df: pd.DataFrame, dataset_name: str):
    note_map = sum([instr['notes_count'] for instrs in df['music21_instruments'] for instr in instrs])
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
    plt.title(f"{dataset_name} - Note distribution (C4 is Middle C)")
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


def plot_velocities(df: pd.DataFrame, dataset_name: str):
    velocity_data = sum([instr['velocities'] for instrs in df['music21_instruments'] for instr in instrs])
    velocity_data = [sum(velocity_data[i:i + 2]) for i in range(0, len(velocity_data), 2)]
    df = pd.DataFrame({
        'x': [i*2 for i in range(len(velocity_data))],  # gli indici rappresentano le ascisse
        'y': velocity_data  # i valori rappresentano le ordinate
    })
    sns.barplot(x='x', y='y', data=df)
    plt.xticks(ticks=range(0, len(velocity_data), 10))
    plt.title(f'{dataset_name} - Velocities')
    plt.grid(True)
    plt.show()


def preprocess_raw_array(data):
    collected_data = keep_most_probable_key(data)
    collected_data = keep_most_probable_tempo(collected_data)
    collected_data = check_which_pieces_change_key_or_time_signature(collected_data)
    return collected_data

def filter_raw_array(data, max_instruments: int = 8, max_minutes: int = 15):
    collected_data = [el for el in data if
                      len(el['music21_instruments']) <= max_instruments
                      and el['length'] < max_minutes * 60]
    return collected_data


def read_pickle_into_df(
        filename: str = 'pijama.pickle',
        dataset_human_name: str = 'PiJAMA',
        genre: str = None
):
    collected_data = read_pickle_raw(filename)

    collected_data = preprocess_raw_array(collected_data)
    collected_data = filter_raw_array(collected_data)

    df = filter_raw_array_to_df(collected_data, [
        'duration', 'length', 'filename', 'key',
        'changes_key', 'changes_time_signature',
        'music21_instruments', 'tempo'
    ])
    df['dataset'] = dataset_human_name
    if genre is not None:
        df['genre'] = genre
    return df


def merge_datasets_and_clean(dfs):
    df_merged = pd.concat(dfs, axis=0)
    df_merged = df_merged.drop(columns=['subgenre', 'author', 'title'])
    df_merged = df_merged.dropna()
    df_merged['length'] /= 60
    return df_merged


def add_columns(df):
    keyboard_instruments = [
        'Piano','ElectricPiano','Harpsichord','Clavichord','Celesta','Sampler',
        'Organ','PipeOrgan','ElectricOrgan','ReedOrgan','Accordion',
        'Vibraphone','Marimba','Glockenspiel','Xylophone','Harp',
    ]
    keyboard_pairs = []
    for k1 in keyboard_instruments:
        for k2 in keyboard_instruments:
            keyboard_pairs.append(f'{k1},{k2}')

    df['instruments_number'] = df['music21_instruments'].str.len()
    df['instruments_classes'] = df['music21_instruments'].apply(lambda x: [i['music21_instrument_class'] for i in x])
    df['is_two_keyboards'] = df['instruments_classes'].apply(lambda instrs: len(instrs) <= 2 and ','.join(instrs) in keyboard_pairs + keyboard_instruments)
    df['are_all_keyboards'] = df['instruments_classes'].apply(lambda instrs: all([i in keyboard_instruments for i in instrs]))
    return df


def plot_genres(df):
    plt.figure(figsize=(8, 8))
    plt.title('Genres distribution')
    sns.violinplot(data=df, hue='genre', x='length', y='genre')
    plt.show()

    working_copy = df.copy(deep=True)
    working_copy['length'] /= 60

    plt.title(f'Genres distribution')
    plt.grid(True)
    sns.barplot(working_copy, y='genre', x='length', hue='dataset', estimator=sum, errorbar=None)
    plt.xlabel('Length [hours]')
    plt.show()


def plot_keys_time_signatures(df):
    working_copy = df.copy(deep=True)
    working_copy['length'] /= 60

    plt.title(f'Key distribution')
    plt.grid(True)
    sns.barplot(df, y='key', x='length', hue='dataset', estimator=sum, errorbar=None)
    plt.xlabel('Length [hours]')
    plt.show()


def plots(df: pd.DataFrame):
    plot_lengths(df)
    plot_genres(df)
    plot_keys_time_signatures(df)
    plot_tempos(df)
    for dataset in df['dataset'].unique():
        plot_notes(df[df['dataset'] == dataset], dataset)
        plot_velocities(df[df['dataset'] == dataset], dataset)


if __name__ == '__main__':
    df_maestro = read_pickle_into_df('maestro.pickle', 'MAESTRO', 'Classical')
    df_pijama = read_pickle_into_df('pijama.pickle', 'PiJAMA', 'Jazz')
    df_vgmusic = read_pickle_into_df('vgmusic.pickle', 'VGMusic', 'Soundtracks')
    df_adl = read_pickle_into_df('adl.pickle', 'ADL Piano')
    df_adl = adl_parse_genres(df_adl)

    df_merged = merge_datasets_and_clean([df_maestro, df_adl, df_vgmusic, df_pijama])
    df_merged = add_columns(df_merged)

    df_piano = df_merged[df_merged['are_all_keyboards']]
    df_not_piano = df_merged[~df_merged['are_all_keyboards']]
    print(len(df_piano))
    print(len(df_not_piano))
    assert len(df_piano) + len(df_not_piano) == len(df_merged)

    plots(df_piano)
