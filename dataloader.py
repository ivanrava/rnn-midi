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

class MidiDataset(Dataset):
    def __init__(self, midi_files, window_len):
        self._midi_files = midi_files
        self._window_len = window_len

    def __len__(self):
        return len(self._midi_files)

    def __getitem__(self, idx):
        ...

def build_dataloader(dataset: MidiDataset, batch_size=16) -> DataLoader:
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