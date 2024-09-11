import os

from tqdm import tqdm

import dataloader
from complete_midi import complete_midi, load_model, build_vocabs

if __name__ == '__main__':
    dataloader.set_seed()

    events_to_generate = 1000

    sequence_length = 256
    modelfile = 'best_model.pt'
    augment = 12
    docfolder_all = 'texts-12'
    docfolder_test = 'texts-12-test'

    vocabulary, vocab_size, vocab_reverse = build_vocabs(augment=augment, docfolder=docfolder_all)
    model = load_model(modelfile, embedding_size=256, hidden_size=512, lstm_layers=1, vocab_length=vocab_size)

    for _, _, file in os.walk(docfolder_all):
        for f in tqdm(file):
            complete_midi(f, f'eval_out/{f}.mid', model, vocabulary, vocab_reverse, sequence_length, events=events_to_generate, cut_tail=False)