import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tqdm

from utils import log


def set_seed(seed = 1337):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

class NotewiseDataset(Dataset):
    def __init__(self, midi_files: dict, vocab: dict, vocab_size: int, window_len: int = 10, notes_to_guess: int = 1):
        self._window_len = window_len
        self._notes_to_guess = notes_to_guess
        self.vocab_size = vocab_size

        self.reset_memoized_window_indexes()

        # TODO: add genre
        log("Encoding documents with vocabulary...")
        self.docs = []
        for genre_docs_list in midi_files.values():
            for doc in genre_docs_list:
                word_list = []
                for word in doc.split(' '):
                    if word in vocab:
                        word_list.append(vocab[word])
                self.docs.append(word_list)
        log(f"Documents encoded: {len(self.docs)} with {sum([len(doc) for doc in self.docs])} words")

        """
        self.min_wait = min([int(w.lstrip('wait')) for w in unique_words if w.startswith('wait')])
        self.max_wait = max([int(w.lstrip('wait')) for w in unique_words if w.startswith('wait')])
        self.min_note = min([int(w.lstrip('p')) for w in unique_words if w.startswith('p')])
        self.max_note = max([int(w.lstrip('p')) for w in unique_words if w.startswith('p')])
        log(f"Min wait: {self.min_wait}")
        log(f"Max wait: {self.max_wait}")
        log(f"Min note: {self.min_note}")
        log(f"Max note: {self.max_note}")
        """

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
        return sum([self.num_windows(doc) for doc in self.docs])

    def num_windows(self, doc):
        return len(doc) - self._window_len

    def reset_memoized_window_indexes(self):
        self.current_windows = 0
        self.current_doc_idx = 0

    def item_idx_to_local_window(self, item_idx: int):
        windows = self.current_windows
        starting_doc_idx = self.current_doc_idx
        for doc in self.docs[starting_doc_idx:]:
            local_windows = self.num_windows(doc)
            if item_idx <= windows + local_windows:
                local_idx = item_idx - windows
                return doc[local_idx:local_idx + self._window_len]
            else:
                windows += local_windows
                self.current_windows = windows
                self.current_doc_idx += 1

        self.reset_memoized_window_indexes()
        return self.item_idx_to_local_window(item_idx)

    def __getitem__(self, idx):
        designed_window = self.item_idx_to_local_window(idx)

        example = designed_window[:-self._notes_to_guess]
        label = designed_window[self._window_len-self._notes_to_guess:]

        example = torch.tensor(example, dtype=torch.int64, device=torch.device('cpu'))
        label = torch.tensor(label, dtype=torch.int64, device=torch.device('cpu'))

        return example, label

    """
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
    """

class MidiDataLoader(DataLoader):
    def __iter__(self):
        self.dataset.reset_memoized_window_indexes()
        return super().__iter__()

def build_dataloader(dataset: NotewiseDataset, batch_size=16) -> DataLoader:
    log("Building dataloader...")
    loader = MidiDataLoader(dataset, batch_size=batch_size)
    log(f"Dataloader built: {len(loader)} batches")
    return loader

def read_documents(texts_folder: str, extension: str) -> dict:
    log("Reading MIDI documents...")

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

    log(f"Documents read: {sum([len(documents[g]) for g in documents])} totalling {sum([sum([sys.getsizeof(d) for d in documents[g]]) for g in documents]) / 10**6 } MB")
    log(f'Genres: {len(documents)}')

    return documents

def split_list(lst: list, p1: float, p2: float, _p3: float):
    random.shuffle(lst)
    len1 = int(len(lst) * p1)
    len2 = int(len(lst) * p2)

    sublist1 = lst[:len1]
    sublist2 = lst[len1:len1+len2]
    sublist3 = lst[len1+len2:]

    return sublist1, sublist2, sublist3


def split_documents_dict(documents: dict, p1: float, p2: float, _p3: float):
    d1 = {}
    d2 = {}
    d3 = {}
    for genre in documents:
        d1[genre], d2[genre], d3[genre] = split_list(documents[genre], p1, p2, _p3)
    return d1, d2, d3


def build_vocab(docs_dict: dict):
    log("Building vocabulary...")
    vocab = {}
    vocab_size = 0
    for genre_docs_list in docs_dict.values():
        for doc in genre_docs_list:
            for word in doc.split(' '):
                if word == '':
                    continue
                if word not in vocab:
                    vocab[word] = vocab_size
                    vocab_size += 1
    log(f"Vocabolary built: {vocab_size} words")
    return vocab, vocab_size


def build_split_loaders(
        folder: str, extension: str,
        train_perc=0.8, val_perc=0.1, test_perc=0.1,
        window_len: int = 10, to_guess: int = 1, batch_size: int = 16
    ):
    docs = read_documents(folder, extension)
    vocab, vocab_size = build_vocab(docs)
    train_docs, val_docs, test_docs = split_documents_dict(docs, train_perc, val_perc, test_perc)

    train_loader = build_dataloader(NotewiseDataset(train_docs, vocab, vocab_size, window_len, to_guess), batch_size=batch_size)
    val_loader = build_dataloader(NotewiseDataset(val_docs, vocab, vocab_size, window_len, to_guess), batch_size=batch_size)
    test_loader = build_dataloader(NotewiseDataset(test_docs, vocab, vocab_size, window_len, to_guess), batch_size=batch_size)

    return train_loader, val_loader, test_loader, vocab_size

if __name__ == '__main__':
    set_seed()

    docs = read_documents("texts-12", '.notewise')
    dataloader = build_dataloader(NotewiseDataset(docs, 10), batch_size=16)