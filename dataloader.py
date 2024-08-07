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
    def __init__(self, midi_files: dict, window_len, notes_to_guess=1):
        self._window_len = window_len
        self._notes_to_guess = notes_to_guess
        # TODO: add genre
        word_list = []
        for genre_docs_list in midi_files.values():
            for doc in genre_docs_list:
                for word in doc:
                    word_list.append(self.convert_word(word))

        word_list = torch.tensor(word_list, dtype=torch.int64, device=torch.device('cpu'))
        self.tensors = word_list.unfold(0, self._window_len, 1)

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        designed_window = self.tensors[idx]
        return designed_window[:self._window_len], designed_window[self._window_len:]

    @staticmethod
    def convert_word(word: str):
        if word.startswith('p'):
            return int(word.lstrip('p'))
        elif word.startswith('endp'):
            return int(word.lstrip('endp'))
        else:
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
    log("Dataloader built")