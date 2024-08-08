import datetime
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import tqdm


def set_seed(seed = 1337):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

class NotewiseDataset(Dataset):
    def __init__(self, midi_files: dict, window_len: int, notes_to_guess: int = 1):
        self._window_len = window_len
        self._notes_to_guess = notes_to_guess
        # TODO: add genre
        log("Reading words...")
        word_list = []
        for genre_docs_list in midi_files.values():
            for doc in genre_docs_list:
                for word in doc.split(' '):
                    if word == '':
                        continue
                    word_list.append(word)

        unique_words = set(word_list)
        log(f"Finished reading words: {len(word_list)} (unique {len(unique_words)})")

        min_wait = min([int(w.lstrip('wait')) for w in unique_words if w.startswith('wait')])
        max_wait = max([int(w.lstrip('wait')) for w in unique_words if w.startswith('wait')])
        min_note = min([int(w.lstrip('p')) for w in unique_words if w.startswith('p')])
        max_note = max([int(w.lstrip('p')) for w in unique_words if w.startswith('p')])
        log(f"Min wait: {min_wait}")
        log(f"Max wait: {max_wait}")
        log(f"Min note: {min_note}")
        log(f"Max note: {max_note}")

        # FIXME: convert to numbers
        word_list = torch.tensor(word_list, dtype=torch.int64, device=torch.device('cpu'))
        self.tensors = word_list.unfold(0, self._window_len, 1)

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        designed_window = self.tensors[idx]
        return designed_window[:self._window_len-self._notes_to_guess], designed_window[self._window_len-self._notes_to_guess:]

    @staticmethod
    def convert_word(word: str):
        if word.startswith('p'):
            return int(word.lstrip('p'))
        elif word.startswith('endp'):
            return int(word.lstrip('endp'))
        elif word == 'wait':
            return 1
        elif word.startswith('wait'):
            return int(word.lstrip('wait'))


def build_dataloader(dataset: NotewiseDataset, batch_size=16) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size)

def log(message):
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]: {message}")

def read_documents(texts_folder: str, extension: str):
    documents = {}
    for root, _, files in tqdm.tqdm(os.walk(texts_folder)):
        genre = root.split('/')[-1]
        for file in tqdm.tqdm(files, leave=False):
            if file.endswith(extension):
                try:
                    with open(os.path.join(root, file), 'r') as f:
                        if genre in documents:
                            documents[genre].append(f.read())
                        else:
                            documents[genre] = [f.read()]
                except Exception as e:
                    print(e)
    return documents

if __name__ == '__main__':
    set_seed()

    log("Reading MIDI documents...")
    docs = read_documents("texts-4", '.notewise')
    log(f"Documents read: {sum([len(docs[g]) for g in docs])} totalling {sum([sum([sys.getsizeof(d) for d in docs[g]]) for g in docs]) / 10**6 } MB")
    log(f'Genres: {len(docs)}')

    log("Building dataloader...")
    dataloader = build_dataloader(NotewiseDataset(docs, 10), batch_size=16)
    log(f"Dataloader built: {len(dataloader)} batches")