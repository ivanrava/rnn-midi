import music21
import numpy as np
import torch
from tqdm import tqdm

import dataloader
import miditoken
import render_notewise
from rnn_timedistributed import RNNTD
from utils import log


def generate_representations(filename: str, sample_freq: int = 12):
    score = music21.converter.parse(filename)

    chordwise_repr = miditoken.score_to_chordwise(score, sample_freq=sample_freq)
    notewise_repr = miditoken.chordwise_to_notewise(chordwise_repr, sample_freq=sample_freq)

    chordwise_repr = ' '.join(chordwise_repr)
    return chordwise_repr, notewise_repr


def load_model(modelfile: str, embedding_size: int, hidden_size: int, lstm_layers: int, vocab_length):
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)

    m = RNNTD(
        device,
        device_str,
        input_vocab_size=vocab_length,
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        nl=lstm_layers
    ).to(device)
    m.load_state_dict(torch.load(modelfile))
    return m


def encode_sequence(model: RNNTD, sequence: list):
    preds = model.generate_new_note(sequence)
    return preds


def pick_note_from_preds(preds, temperature = 1.0):
    log_preds = preds[-1, :]
    probs = np.exp(log_preds / temperature)
    probs /= np.sum(probs)
    return np.random.choice(np.arange(len(probs)), p=probs)


def make_temperature_schedule(events_to_generate: int):
    samples_turnaround = 100
    temperature_schedule = np.cos(np.linspace(-np.pi, np.pi, samples_turnaround))
    temp_low = 0.8
    temp_high = 1
    temperature_schedule = temp_low + ((temperature_schedule + 1) / 2) * (temp_high - temp_low)
    temperature_schedule = np.tile(temperature_schedule, int(events_to_generate / samples_turnaround))
    return temperature_schedule


def build_vocabs(augment: int, docfolder: str):
    limit_genres = ['Classical']
    docs = dataloader.read_documents(docfolder, '.notewise', limit_genres=limit_genres)
    vocabulary, vocab_size = dataloader.build_vocab(docs, augment=augment)
    vocab_reverse = {num:word for word, num in vocabulary.items()}
    return vocabulary, vocab_size, vocab_reverse


if __name__ == '__main__':
    dataloader.set_seed()

    sequence_length = 256
    events_to_generate = 200
    to_guess = 1

    vocabulary, vocab_size, vocab_reverse = build_vocabs(augment=20, docfolder='texts-12')
    model = load_model('best_model.pt', embedding_size=256, hidden_size=512, lstm_layers=1, vocab_length=vocab_size)

    _, notewise = generate_representations("datasets/examples/moonlite.mid")
    whole_piece = [w for w in notewise.split(' ') if w in vocabulary]
    log(f'Starting from {len(whole_piece)} samples')

    for t in tqdm(make_temperature_schedule(events_to_generate=events_to_generate)):
        sequence = whole_piece[-(sequence_length-to_guess+1):]
        sequence = [vocabulary[word] for word in sequence if word in vocabulary]

        preds = encode_sequence(model, sequence)

        whole_piece += [vocab_reverse[pick_note_from_preds(p, temperature=t)] for p in preds]

    log(f'Ended at {len(whole_piece)} samples')
    render_notewise.render_notewise(whole_piece, 'midi_completion.mid')