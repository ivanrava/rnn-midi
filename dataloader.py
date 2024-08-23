import glob
import os
import random
import sys

import numpy as np
import pretty_midi
import torch
from torch.utils.data import Dataset, DataLoader
import tqdm

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
    for word in word_list:
        if word.startswith('wait'):
            augmented_doc.append(vocab[word])
            continue
        note_value = int(word[1 if word.startswith("p") else 4:])
        note_value += offset
        if note_value < 0 or note_value > 95:
            invalid = True
            break
        augmented_doc.append(vocab[f'{word[0] if word.startswith("p") else "endp"}{note_value}'])

    return augmented_doc if not invalid else None


class NotewiseNonOverlappingDataset(Dataset):
    def __init__(self, midi_files: dict, vocab: dict, vocab_size: int, window_len: int = 10, notes_to_guess: int = 1, augment: int = 12):
        self._window_len = window_len
        self._notes_to_guess = notes_to_guess
        self.vocab_size = vocab_size
        self.augment_semitones = augment

        self.reset_memoized_window_indexes()

        # TODO: add genre
        log("Augmenting documents...")
        self.docs = []
        for genre_docs_list in midi_files.values():
            for doc in genre_docs_list:
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
        return np.ceil(len(doc) / self._window_len)

    def reset_memoized_window_indexes(self):
        self.current_windows = 0
        self.current_doc_idx = 0

    def item_idx_to_local_windows(self, item_idx: int):
        windows = self.current_windows
        starting_doc_idx = self.current_doc_idx
        for doc in self.docs[starting_doc_idx:]:
            local_windows = self.num_windows(doc)
            if item_idx <= windows + local_windows:
                local_idx = (item_idx - windows) * self._window_len
                return doc[local_idx:local_idx + self._window_len], doc[local_idx+self._notes_to_guess:local_idx + self._window_len+self._notes_to_guess]
            else:
                windows += local_windows
                self.current_windows = windows
                self.current_doc_idx += 1

        self.reset_memoized_window_indexes()
        return self.item_idx_to_local_windows(item_idx)

    def __getitem__(self, idx):
        designed_window, designed_label = self.item_idx_to_local_windows(idx)

        example = designed_window + [PAD_IDX] * (self._window_len - len(designed_window))
        label = designed_label + [PAD_IDX] * (self._window_len - len(designed_label))

        example = torch.tensor(example, dtype=torch.int64, device=torch.device('cpu'))
        label = torch.tensor(label, dtype=torch.int64, device=torch.device('cpu'))

        return example, label

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

class TripletDataset:
    def __init__(self, pms, window_len: int = 32, notes_to_guess: int = 1):
        self.midis = []
        self._window_len = window_len
        self._notes_to_guess = notes_to_guess

        log("Expanding note data through PM...")
        for pm in tqdm.tqdm(pms):
            self.midis.append(self.midi_to_notes(pm))
        log("Notes expanded!")

    def midi_to_notes(self, pm: pretty_midi.PrettyMIDI):
        parsed_notes = []
        unsorted_notes = []
        for instr in pm.instruments:
            unsorted_notes += instr.notes
        sorted_notes = sorted(unsorted_notes, key=lambda n: n.start)
        prev_start = sorted_notes[0].start

        for note in sorted_notes:
            start = note.start
            end = note.end
            parsed_notes.append([
                note.pitch, # Pitch
                start - prev_start, # Step
                end - start # Duration
            ])
            prev_start = start

        return parsed_notes

    def num_windows(self, midi):
        return len(midi) - self._window_len

    def __len__(self):
        return sum([self.num_windows(midi) for midi in self.midis])

    def reset_memoized_window_indexes(self):
        self.current_windows = 0
        self.current_doc_idx = 0

    def item_idx_to_local_window(self, item_idx: int):
        windows = self.current_windows
        starting_doc_idx = self.current_doc_idx
        for doc in self.midis[starting_doc_idx:]:
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

        example = torch.tensor(example, device=torch.device('cpu'))
        label = torch.tensor(label, device=torch.device('cpu'))

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

def read_documents(texts_folder: str, extension: str, limit_genres: list = None, max_docs_per_genre: int = 0) -> dict:
    log("Reading MIDI documents...")

    documents = {}
    for root, _, files in tqdm.tqdm(os.walk(texts_folder)):
        genre = root.split('/')[-1]
        if limit_genres is not None and genre not in limit_genres:
            continue
        for file in tqdm.tqdm(files if max_docs_per_genre == 0 else files[:max_docs_per_genre], leave=False):
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


def build_vocab(docs_dict: dict, augment: int = 12):
    log("Building vocabulary...")
    vocab = {PAD_TOKEN: PAD_IDX}
    vocab_size = len(vocab)
    for genre_docs_list in docs_dict.values():
        for doc in genre_docs_list:
            for word in doc.split(' '):
                if word == '':
                    continue
                if augment and not word.startswith('wait'):
                    for t in range(0,augment):
                        offset = t-augment/2
                        note_value = int(word[1:])
                        note_value += offset
                        if 0 <= note_value <= 95:
                            word = f'{word[0]}{note_value}'
                            if word not in vocab:
                                vocab[word] = vocab_size
                                vocab_size += 1
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
        augment: int = 12
    ):
    docs = read_documents(folder, extension, limit_genres=limit_genres, max_docs_per_genre=max_docs_per_genre)
    vocab, vocab_size = build_vocab(docs, augment=augment)
    train_docs, val_docs, test_docs = split_documents_dict(docs, train_perc, val_perc, test_perc)

    train_loader = build_dataloader(NotewiseNonOverlappingDataset(train_docs, vocab, vocab_size, window_len, to_guess, augment=augment), batch_size=batch_size)
    val_loader = build_dataloader(NotewiseNonOverlappingDataset(val_docs, vocab, vocab_size, window_len, to_guess), batch_size=batch_size)
    test_loader = build_dataloader(NotewiseNonOverlappingDataset(test_docs, vocab, vocab_size, window_len, to_guess), batch_size=batch_size)

    return train_loader, val_loader, test_loader, vocab_size

def build_split_loaders_triplet(train_perc=0.8, val_perc=0.1, test_perc=0.1,
        window_len: int = 10, to_guess: int = 1, batch_size: int = 16):
    log("Reading MIDI files...")
    filenames = glob.glob(f'datasets/merged/**/*.mid*')
    pms = []
    total = 0
    valid = 0
    for filename in tqdm.tqdm(filenames):
        pm = pretty_midi.PrettyMIDI(filename)
        total += 1
        for i, instr in enumerate(pm.instruments):
            if i >= 2:
                break
            if instr.program > 8:
                break
        else:
            pms.append(pm)
            valid += 1
    log(f"MIDI files read: {valid} valid out of {total} ({valid / total * 100}%)")
    train_pms, val_pms, test_pms = split_list(pms, train_perc, val_perc, test_perc)

    train_loader = build_dataloader(TripletDataset(train_pms, window_len, to_guess), batch_size=batch_size)
    val_loader = build_dataloader(TripletDataset(val_pms, window_len, to_guess), batch_size=batch_size)
    test_loader = build_dataloader(TripletDataset(test_pms, window_len, to_guess), batch_size=batch_size)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    set_seed()

    docs = read_documents("texts-12", '.notewise')
    dataloader = build_dataloader(NotewiseDataset(docs, 10), batch_size=16)