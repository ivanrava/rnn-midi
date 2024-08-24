import music21
import numpy as np
import torch
from tqdm import tqdm

import dataloader
import miditoken
import render_notewise
from rnn_timedistributed import RNNTD
from utils import log


def complete_midi(filename: str, sample_freq: int = 12):
    score = music21.converter.parse(filename)

    chordwise_repr = miditoken.score_to_chordwise(score, sample_freq=sample_freq)
    notewise_repr = miditoken.chordwise_to_notewise(chordwise_repr, sample_freq=sample_freq)

    chordwise_repr = ' '.join(chordwise_repr)
    return chordwise_repr, notewise_repr


def load_rnn_plain(modelfile: str, vocab_length):
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)

    embedding_size = 256
    hidden_size = 512
    dropout_rate = .1
    lstm_layers = 2

    model = RNNTD(
        device,
        device_str,
        input_vocab_size=vocab_length,
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        dropout_rate=dropout_rate,
        nl=lstm_layers
    ).to(device)
    model.load_state_dict(torch.load(modelfile))
    return model


def encode_sequence(model: RNNTD, sequence: list):
    preds = model.generate_new_note(sequence)
    return preds


def pick_note_from_preds(preds):
    log_preds = preds[-1, :]
    probs = np.exp(log_preds)
    probs /= np.sum(probs)
    return np.random.choice(np.arange(len(probs)), p=probs)



if __name__ == '__main__':
    dataloader.set_seed()

    quarter_notes_to_generate = 48
    sampling_freq = 12
    samples_to_generate = quarter_notes_to_generate * sampling_freq

    probability_to_pick_max = 1

    filename = 'datasets/examples/moonlite.mid'
    modelfile = 'best_model.pt'
    sequence_length = 100
    to_guess = 1

    folder = f'texts-{sampling_freq}'
    extension = '.notewise'
    limit_genres = ['Classical']
    max_docs_per_genre = 0
    docs = dataloader.read_documents(folder, extension, limit_genres=limit_genres, max_docs_per_genre=max_docs_per_genre)
    vocabulary, vocab_size = dataloader.build_vocab(docs, augment=12)
    vocab_reverse = {num:word for word, num in vocabulary.items()}

    chordwise, notewise = complete_midi(filename)
    representation = notewise if extension == '.notewise' else chordwise
    whole_piece = [w for w in representation.split(' ') if w in vocabulary]
    log(f'Starting from {len(whole_piece)} samples')

    model = load_rnn_plain(modelfile, vocab_size)

    for _ in tqdm(range(quarter_notes_to_generate)):
        sequence = whole_piece[-(sequence_length-to_guess+1):]
        sequence = [vocabulary[word] for word in sequence if word in vocabulary]

        preds = encode_sequence(model, sequence)

        whole_piece += [vocab_reverse[pick_note_from_preds(p)] for p in preds]

    log(f'Ended at {len(whole_piece)} samples')
    render_notewise.render_notewise(whole_piece, 'midi_completion.mid')