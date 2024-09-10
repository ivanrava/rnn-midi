import music21
import glob
import tqdm
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def literature_metrics(score: music21.stream.Score):
    """
    Compute some established metrics
    :param score: Music21 Score
    :return: % of empty bars, (mean) unique pitch classes per bar, tone distance, % of >= 16th, % of >= 32nd, % of > semibreve, % of > breve
    """
    def store_pitch(pitch_map, pitch: music21.pitch.Pitch, curr_min, curr_max):
        if pitch not in pitch_map:
            pitch_map[pitch.name] = 1
        else:
            pitch_map[pitch.name] += 1

        abstone = pitch.octave * 12 + pitch.pitchClass
        return min(abstone, curr_min), max(abstone, curr_max)

    eb = 0
    upc = 0
    qn16 = 0
    qn32 = 0
    morethansb = 0
    morethanbreve = 0
    measures = score.parts[0].getElementsByClass('Measure')
    measures_with_notes = 0
    notes = 0
    min_tone = 200
    max_tone = -1
    for m in measures:
        notes_in_measure = m.flatten().notes
        if len(notes_in_measure) == 0:
            eb += 1
            continue
        measures_with_notes += 1
        pitches = {}
        for n in notes_in_measure:
            notes += 1
            if type(n) is music21.chord.Chord:
                for p in n.pitches:
                    min_tone, max_tone = store_pitch(pitches, p, min_tone, max_tone)
            else:
                min_tone, max_tone = store_pitch(pitches, n.pitch, min_tone, max_tone)
            if n.duration.quarterLength > 0.1:
                qn32 += 1
                if n.duration.quarterLength > 0.2:
                    qn16 += 1
                    if n.duration.quarterLength > 4:
                        morethansb += 1
                        if n.duration.quarterLength > 8:
                            morethanbreve += 1

        upc += len(pitches)


    return (
        eb / len(measures),
        upc / measures_with_notes,
        max_tone - min_tone,
        qn32 / notes,
        qn16 / notes,
        morethansb / notes,
        morethanbreve / notes
    )


def aggregate_stats(filename):
    s = music21.converter.parse(filename)
    metrics = literature_metrics(s)
    key = s.analyze('key')
    return {'eb': metrics[0],
            'upc': metrics[1],
            'td': metrics[2],
            'qn16': metrics[3],
            'qn32': metrics[4],
            'qnsb': metrics[5],
            'qnb': metrics[6],
            'keycoeff': key.correlationCoefficient}

def compute_metrics(folder):
    filenames = glob.glob(f'{folder}/*.mid')
    data = []
    for f in tqdm.tqdm(filenames):
        data.append(aggregate_stats(f))
    return pd.DataFrame(data)

if __name__ == '__main__':
    df = compute_metrics('')
    df.to_csv('.csv')