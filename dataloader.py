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
        self.word_list = []
        for genre_docs_list in midi_files.values():
            for doc in genre_docs_list:
                for word in doc.split(' '):
                    if word == '':
                        continue
                    self.word_list.append(word)

        unique_words = set(self.word_list)
        log(f"Finished reading words: {len(self.word_list)} (unique {len(unique_words)})")

        self.min_wait = min([int(w.lstrip('wait')) for w in unique_words if w.startswith('wait')])
        self.max_wait = max([int(w.lstrip('wait')) for w in unique_words if w.startswith('wait')])
        self.min_note = min([int(w.lstrip('p')) for w in unique_words if w.startswith('p')])
        self.max_note = max([int(w.lstrip('p')) for w in unique_words if w.startswith('p')])
        log(f"Min wait: {self.min_wait}")
        log(f"Max wait: {self.max_wait}")
        log(f"Min note: {self.min_note}")
        log(f"Max note: {self.max_note}")

        """
        log("Converting words into numbers...")
        word_list = [self.convert_word(word, max_wait, max_note, min_wait, min_note) for word in word_list]
        word_list = torch.tensor(word_list, dtype=torch.int64, device=torch.device('cpu'))
        log("Tensors created!")
        log(f"Unfolding into {self._window_len} words windows...")
        self.tensors = word_list.unfold(0, self._window_len, 1)
        log("Unfolded / windowed!")
        """

    def __len__(self):
        return len(self.word_list) - self._window_len + 1

    def __getitem__(self, idx):
        designed_window = self.word_list[idx:idx + self._window_len]
        designed_window = np.array([self.convert_word(word) for word in designed_window])

        example = designed_window[:-self._notes_to_guess]
        label = designed_window[self._window_len-self._notes_to_guess:]

        example = torch.tensor(example, dtype=torch.int64, device=torch.device('cpu'))
        label = torch.tensor(label, dtype=torch.int64, device=torch.device('cpu'))

        return example, label

    def convert_word(self, word: str):
        wait_slice = np.zeros(self.max_wait - self.min_wait + 1)
        note_slice = np.zeros(self.max_note - self.min_note + 1)
        if word.startswith('p'):
            note_index = int(word.lstrip('p'))
            note_slice[note_index - self.min_note] = 1
        elif word.startswith('endp'):
            note_index = int(word.lstrip('endp'))
            note_slice[note_index - self.min_note] = -1
        elif word == 'wait':
            raise Exception("Unexpected word")
        elif word.startswith('wait'):
            wait_index = int(word.lstrip('wait'))
            wait_slice[wait_index - self.min_wait] = 1
        return np.concatenate((wait_slice, note_slice), axis=0)


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
    docs = read_documents("texts-12", '.notewise')
    log(f"Documents read: {sum([len(docs[g]) for g in docs])} totalling {sum([sum([sys.getsizeof(d) for d in docs[g]]) for g in docs]) / 10**6 } MB")
    log(f'Genres: {len(docs)}')

    log("Building dataloader...")
    dataloader = build_dataloader(NotewiseDataset(docs, 10), batch_size=16)
    log(f"Dataloader built: {len(dataloader)} batches")