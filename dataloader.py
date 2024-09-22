import os
import random
import sys

import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset, DataLoader

from utils import log

PAD_TOKEN = '<PAD>'
PAD_IDX = 0


def set_seed(seed = 1337):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def augment_document(word_list, vocab, transposition, correction: int = 0):
    augmented_doc = []
    offset = transposition - correction
    if offset == 0:
        return [vocab[w] for w in word_list]
    invalid = False
    for word in tqdm.tqdm(word_list, leave=False):
        if word.startswith('wait'):
            augmented_doc.append(vocab[word])
            continue
        note_value = int(word[1 if word.startswith("p") else 4:])
        note_value += offset
        if note_value < 0 or note_value > 119: # 95
            invalid = True
            break
        augmented_doc.append(vocab[f'{word[0] if word.startswith("p") else "endp"}{note_value}'])

    return augmented_doc if not invalid else None


class NotewiseNonOverlappingDataset(Dataset):
    def __init__(self,
                 midi_files: dict,
                 vocab: dict,
                 vocab_size: int,
                 window_len: int = 10,
                 notes_to_guess: int = 1,
                 augment: int = 0,
                 window_dodge: int = 1
                 ):
        self._window_len = window_len
        self._notes_to_guess = notes_to_guess
        self.vocab_size = vocab_size
        self.augment_semitones = augment
        self.window_dodge = window_dodge

        self.reset_memoized_window_indexes()

        # TODO: add genre
        log("Augmenting documents...")
        self.docs = []
        for genre_docs_list in tqdm.tqdm(midi_files.values()):
            for doc in tqdm.tqdm(genre_docs_list, leave=False):
                self.docs += self.augment(doc, vocab)
        log(f"Documents encoded!")

    def augment(self, doc, vocab):
        word_list = [w for w in doc.split(' ') if w != '']
        augmented_docs = []
        for transposition in range(0, self.augment_semitones) if self.augment_semitones > 0 else [0]:
            augmented = augment_document(word_list, vocab, transposition, correction=int(self.augment_semitones / 2))
            if augmented is not None:
                augmented_docs.append(augmented)
        return augmented_docs

    def __len__(self):
        return sum([self.num_windows(doc) for doc in self.docs])

    def num_windows(self, doc):
        return int(np.floor((len(doc) - self._window_len) / self.window_dodge)) + 1

    def reset_memoized_window_indexes(self):
        self.current_windows = 0
        self.current_doc_idx = 0

    def item_idx_to_local_windows(self, item_idx: int):
        windows = self.current_windows
        starting_doc_idx = self.current_doc_idx
        for doc in self.docs[starting_doc_idx:]:
            local_windows = self.num_windows(doc)
            if item_idx < windows + local_windows:
                local_idx = item_idx - windows
                local_idx *= self.window_dodge
                return doc[local_idx:local_idx + self._window_len], doc[local_idx+self._notes_to_guess:local_idx + self._window_len+self._notes_to_guess]
            else:
                windows += local_windows
                self.current_windows = windows
                self.current_doc_idx += 1

        self.reset_memoized_window_indexes()
        return self.item_idx_to_local_windows(item_idx)

    def __getitem__(self, idx):
        designed_window, designed_label = self.item_idx_to_local_windows(idx)
        if len(designed_window) < self._window_len:
            designed_window[-1] = PAD_IDX

        example = designed_window + [PAD_IDX] * (self._window_len - len(designed_window))
        label = designed_label + [PAD_IDX] * (self._window_len - len(designed_label))

        example = torch.tensor(example, dtype=torch.int64, device=torch.device('cpu'))
        label = torch.tensor(label, dtype=torch.int64, device=torch.device('cpu'))

        return example, label

class MidiDataLoader(DataLoader):
    def __iter__(self):
        self.dataset.reset_memoized_window_indexes()
        return super().__iter__()

def build_dataloader(dataset: Dataset, batch_size=16) -> DataLoader:
    log("Building dataloader...")
    loader = MidiDataLoader(dataset, batch_size=batch_size)
    log(f"Dataloader built: {len(loader)} batches")
    return loader

def complex_midi(full_filename: str):
    return os.path.exists(f'datasets/merged/{full_filename}.midi')

def read_documents(texts_folder: str, extension: str, limit_genres: list = None, max_docs_per_genre: int = 0) -> dict:
    log("Reading MIDI documents...")

    documents = {}
    for root, _, files in tqdm.tqdm(os.walk(texts_folder)):
        genre = root.split('/')[-1]
        if limit_genres is not None and genre not in limit_genres:
            continue
        for file in tqdm.tqdm(files if max_docs_per_genre == 0 else files[:max_docs_per_genre], leave=False):
            if file.endswith(extension) and not complex_midi(os.path.join(root, file).replace(texts_folder, "").strip(extension)):
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


def build_vocab(docs_dict: dict, augment: int = 12):
    log("Building vocabulary...")
    vocab = {PAD_TOKEN: PAD_IDX}
    vocab_size = len(vocab)
    for genre_docs_list in tqdm.tqdm(docs_dict.values()):
        for doc in tqdm.tqdm(genre_docs_list, leave=False):
            for word in tqdm.tqdm(doc.split(' '), leave=False):
                if word == '':
                    continue
                if augment and not word.startswith('wait'):
                    for t in tqdm.tqdm(range(0,augment) if augment > 0 else [0], leave=False):
                        offset = int(t-augment/2)
                        note_value = int(word[1 if word.startswith('p') else 4:])
                        note_value += offset
                        if 0 <= note_value <= 119: # 95
                            w = f'{word[0] if word.startswith("p") else "endp"}{note_value}'
                            if w not in vocab:
                                vocab[w] = vocab_size
                                vocab_size += 1
                        else:
                            continue
                else:
                    if word not in vocab:
                        vocab[word] = vocab_size
                        vocab_size += 1
    log(f"Vocabolary built: {vocab_size} words")
    return vocab, vocab_size


def build_split_loaders(
        folder: str, extension: str,
        train_perc=0.8, val_perc=0.1, test_perc=0.1,
        window_len: int = 10, to_guess: int = 1, batch_size: int = 16,
        max_docs_per_genre: int = 0, limit_genres: list = None,
        augment: int = 12,
        window_dodge: int = 1
    ):
    from encoder_decoder import CompletionEvalDataset
    docs = read_documents(folder, extension, limit_genres=limit_genres, max_docs_per_genre=max_docs_per_genre)
    vocab, vocab_size = build_vocab(docs, augment=augment)
    train_docs, val_docs, test_docs = split_documents_dict(docs, train_perc, val_perc, test_perc)

    train_loader = build_dataloader(NotewiseNonOverlappingDataset(train_docs, vocab, vocab_size, window_len, to_guess, augment=augment, window_dodge=window_dodge), batch_size=batch_size)
    val_loader = build_dataloader(NotewiseNonOverlappingDataset(val_docs, vocab, vocab_size, window_len, to_guess, window_dodge=1), batch_size=batch_size)
    test_loader = build_dataloader(NotewiseNonOverlappingDataset(test_docs, vocab, vocab_size, window_len, to_guess, window_dodge=1), batch_size=batch_size)

    flattened_docs = []
    for d in sum(val_docs.values(), []):
        new_d = []
        for w in d.split(" "):
            if w in vocab:
                new_d.append(vocab[w])
        flattened_docs.append(new_d)
    comp_eval_loader = DataLoader(CompletionEvalDataset(flattened_docs, window_len-to_guess), batch_size=batch_size)

    return train_loader, val_loader, test_loader, comp_eval_loader, vocab
