import music21
import torch

import miditoken
import dataloader
from rnn_plain import RNNPlain


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
    lstm_layers = 4

    model = RNNPlain(
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


def encode_sequence(model: RNNPlain, sequence: list):
    preds = model.generate_new_note(sequence)
    return preds


if __name__ == '__main__':
    filename = 'datasets/examples/moonlite.mid'
    modelfile = 'best_model.pt'
    sequence_length = 10
    to_guess = 1

    folder = 'texts-12'
    extension = '.notewise'
    limit_genres = None
    max_docs_per_genre = 0
    docs = dataloader.read_documents(folder, extension, limit_genres=limit_genres, max_docs_per_genre=max_docs_per_genre)
    vocabulary, vocab_size = dataloader.build_vocab(docs)

    chordwise, notewise = complete_midi(filename)
    representation = notewise if extension == '.notewise' else chordwise
    sequence = representation.split(' ')[:-(sequence_length-to_guess)]
    sequence = [vocabulary[word] for word in sequence]

    model = load_rnn_plain(modelfile, vocab_size)

    preds = encode_sequence(model, sequence)
    print(preds)