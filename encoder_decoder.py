import copy
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast, GradScaler
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import wandb

from dataloader import set_seed
from utils import log

PAD_IDX = 0
PAD_TOKEN = '<PAD>'


class EncoderWords(nn.Module):
    def __init__(self, input_vocab_size: int, embedding_size=1024, hidden_size=1024, dropout_rate=0.2, nl=2, *args, **kwargs):
        super(EncoderWords, self).__init__(*args, **kwargs)

        self.embedding = nn.Embedding(input_vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True, num_layers=nl, dropout=dropout_rate)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.lstm(embedded)
        return output, hidden

class DecoderWords(nn.Module):
    def __init__(self, num_guesses, device, output_size, embedding_size=1024, hidden_size=1024, dropout_rate=0.1, nl=2, *args, **kwargs):
        super(DecoderWords, self).__init__(*args, **kwargs)

        self.embedding = nn.Embedding(output_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True, num_layers=nl, dropout=dropout_rate)
        self.expansion = nn.Linear(hidden_size, output_size)

        self.num_guesses = num_guesses
        self.device = device

    def forward_step(self, decoder_input, decoder_hidden):
        output = self.embedding(decoder_input)
        output, hidden = self.lstm(output, decoder_hidden)
        output = self.expansion(output)

        return output, hidden

    def forward(self, encoder_outputs, encoder_hidden, label_tensor=None):
        batch_size = encoder_outputs.size(0)

        decoder_input = torch.tensor([[0]]*batch_size, dtype=torch.long, device=self.device)
        decoder_hidden = encoder_hidden

        decoder_outputs = []
        for i in range(self.num_guesses):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if label_tensor is not None:
                # Teacher forcing
                decoder_input = label_tensor[:, i].unsqueeze(1)
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1)

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=2)

        return decoder_outputs, decoder_hidden, None

class EncDecWords(nn.Module):
    def __init__(self, encoder: EncoderWords, decoder: DecoderWords, device, device_str: str, *args, **kwargs):
        super(EncDecWords, self).__init__(*args, **kwargs)

        self.encoder = encoder
        self.decoder = decoder

        self.device = device
        self.device_str = device_str

    def forward(self, input_tensor, label_tensor):
        encoder_outputs, encoder_hidden = self.encoder(input_tensor)
        decoder_outputs, _, _  = self.decoder(encoder_outputs, encoder_hidden, label_tensor)

        return decoder_outputs

    def train_model(self, train_batches, val_batches, epochs: int, optimizer, criterion, scaler):
        train_losses = []
        accuracies = []
        val_losses = []
        best_val_loss = np.inf
        best_ep = 0
        best_model = copy.deepcopy(self.state_dict())

        for ep in tqdm(range(epochs)):
            self.train()
            ep_loss = 0.0
            num_correct = 0
            num_total = 0

            for batch in tqdm(train_batches, leave=False):
                optimizer.zero_grad()
                input_tensor, label_tensor = [x.to(self.device) for x in batch]

                with autocast(self.device_str):
                    output_tensor = self(input_tensor, label_tensor)
                    loss = criterion(output_tensor.view(-1, output_tensor.size(-1)), label_tensor.view(-1))

                ep_loss += loss.item()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                corr, tot = self.accuracy_values(output_tensor, label_tensor)
                num_correct += corr
                num_total += tot

            train_losses.append(ep_loss / len(train_batches))
            log(f"Train loss: {train_losses[-1]}")
            log(f"Train accuracy: {num_correct / num_total} ({num_correct}/{num_total})")

            val_loss, accuracy = self.eval_model(val_batches, criterion)
            log(f"Val loss: {val_loss}")
            log(f"Val accuracy: {accuracy}")
            accuracies.append(accuracy)
            val_losses.append(val_loss)

            wandb.log({
                "train_loss": train_losses[-1],
                "train_accuracy": num_correct / num_total,
                "val_loss": val_loss,
                "val_accuracy": accuracy
            })

            if val_loss < best_val_loss:
                log(f"New best epoch with {val_loss} val loss")
                best_val_loss = val_loss
                best_model = copy.deepcopy(self.state_dict())
                best_ep = ep+1

            torch.save(best_model, "best_model.pt")

        return train_losses, accuracies, val_losses, best_ep, best_val_loss

    def eval_model(self, val_batches, criterion):
        self.eval()

        val_loss = 0.0
        num_correct = 0
        num_total = 0

        with torch.no_grad():
            for batch in tqdm(val_batches, leave=False):
                input_tensor, label_tensor = [x.to(self.device) for x in batch]

                with autocast(self.device_str):
                    output_tensor = self(input_tensor, label_tensor)
                    loss = criterion(output_tensor.view(-1, output_tensor.size(-1)), label_tensor.view(-1))

                val_loss += loss.item()
                corr, tot = self.accuracy_values(output_tensor, label_tensor)
                num_correct += corr
                num_total += tot

        log(f"Accuracy count: {num_correct} / {num_total}")
        return val_loss/len(val_batches), num_correct / num_total

    def accuracy_values(self, outputs, labels):
        predictions = outputs.argmax(dim=-1).cpu().numpy().flatten()
        labels = labels.cpu().numpy().flatten()
        num_correct = np.sum(predictions == labels)
        num_total = len(labels)
        return num_correct, num_total

    def generate(self, test_batches):
        self.eval()

        test_preds = []
        test_labels = []

        with torch.no_grad():
            for batch in test_batches:
                input_tensor, target_tensor = [b.to(self.device) for b in batch]

                with autocast(self.device_str):
                    output_tensor = self(input_tensor, target_tensor)
                    predictions = output_tensor.argmax(dim=-1)

                test_preds.extend(predictions.cpu().numpy().flatten())
                test_labels.extend(target_tensor.cpu().numpy().flatten())

        return test_preds, test_labels


class NFollowingDataset(Dataset):
    def __init__(self,
                 docs: list,
                 window_len: int = 10,
                 notes_to_guess: int = 1,
                 window_dodge: int = 1):
        self.docs = docs
        self._window_len = window_len
        self._notes_to_guess = notes_to_guess
        self._window_dodge = window_dodge

    def __len__(self):
        return sum([self.num_windows(doc) for doc in self.docs])

    def num_windows(self, doc):
        return int(np.ceil((len(doc) - self._window_len) / self._window_dodge))

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
                local_idx *= self._window_dodge
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
            designed_window += [PAD_IDX] * (self._window_len - len(designed_window))

        example = designed_window[:-self._notes_to_guess]
        label = designed_window[-self._notes_to_guess:]

        example = torch.tensor(example, dtype=torch.int64, device=torch.device('cpu'))
        label = torch.tensor(label, dtype=torch.int64, device=torch.device('cpu'))

        return example, label

def read_documents(texts_folder: str, limit_genres: list = None, max_docs_per_genre: int = 0) -> list:
    log("Reading MIDI documents...")

    documents = {}
    for root, _, files in tqdm(os.walk(texts_folder)):
        genre = root.split('/')[-1]
        if limit_genres is not None and genre not in limit_genres:
            continue
        for file in tqdm(files if max_docs_per_genre == 0 else files[:max_docs_per_genre], leave=False):
            if file.endswith('.notewise'):
                try:
                    with open(os.path.join(root, file), 'r') as f:
                        if genre in documents:
                            documents[genre].append(f.read().split(" "))
                        else:
                            documents[genre] = [f.read().split(" ")]
                except Exception as e:
                    print(e)

    log(f"Documents read: {sum([len(documents[g]) for g in documents])} totalling {sum([sum([sys.getsizeof(d) for d in documents[g]]) for g in documents]) / 10**6 } MB")
    log(f'Genres: {len(documents)}')

    return [doc for genre in documents for doc in documents[genre]]

def build_vocab(docs: list):
    log("Building vocabulary...")
    vocab = {PAD_TOKEN: PAD_IDX}
    vocab_size = len(vocab)
    for doc in tqdm(docs):
        for word in tqdm(doc, leave=False):
            if word == '':
                continue
            if word not in vocab:
                vocab[word] = vocab_size
                vocab_size += 1
    log(f"Vocabolary built: {vocab_size} words")
    return vocab, vocab_size

def split_list(lst: list, p1: float, p2: float, _p3: float):
    random.shuffle(lst)
    len1 = int(len(lst) * p1)
    len2 = int(len(lst) * p2)

    sublist1 = lst[:len1]
    sublist2 = lst[len1:len1+len2]
    sublist3 = lst[len1+len2:]

    return sublist1, sublist2, sublist3

def augment_docs(docs: list, augment: int = 12, max_note: int = 119) -> list:
    augmented_docs = []
    for doc in tqdm(docs):
        for t in tqdm(range(0, augment), leave=False):
            augmented_doc = []
            offset = int(t - augment / 2)
            for word in tqdm(doc, leave=False):
                if word == '':
                    continue
                if word.startswith('wait'):
                    augmented_doc.append(word)
                else:
                    sep_index = 1 if word.startswith('p') else 4
                    prefix, note_value = word[:sep_index], int(word[sep_index:])
                    note_value += offset
                    if 0 <= note_value <= max_note:
                        augmented_doc.append(f'{prefix}{note_value}')
            augmented_docs.append(augmented_doc)
    return augmented_docs

class MidiDataLoader(DataLoader):
    def __iter__(self):
        self.dataset.reset_memoized_window_indexes()
        return super().__iter__()

def build_dataloader(dataset: Dataset, batch_size=64) -> DataLoader:
    log("Building dataloader...")
    loader = MidiDataLoader(dataset, batch_size=batch_size)
    log(f"Dataloader built: {len(loader)} batches")
    return loader

def build_split_loaders(
        folder: str,
        train_perc=0.8, val_perc=0.1, test_perc=0.1,
        window_len: int = 10, to_guess: int = 1, batch_size: int = 16,
        max_docs_per_genre: int = 0, limit_genres: list = None,
        augment: int = 12,
        window_dodge: int = 1
):
    docs = read_documents(folder, limit_genres, max_docs_per_genre)
    train_docs, val_docs, test_docs = split_list(docs, train_perc, val_perc, test_perc)
    if augment > 0:
        train_docs = augment_docs(train_docs, augment)
        random.shuffle(train_docs)
    vocab, vocab_size = build_vocab(train_docs + val_docs + test_docs)
    train_docs = [[vocab[w] for w in doc if w != ''] for doc in train_docs]
    val_docs = [[vocab[w] for w in doc if w != ''] for doc in val_docs]
    test_docs = [[vocab[w] for w in doc if w != ''] for doc in test_docs]

    train_loader = build_dataloader(NFollowingDataset(train_docs, window_len, to_guess, window_dodge), batch_size=batch_size)
    val_loader = build_dataloader(NFollowingDataset(val_docs, window_len, to_guess, window_dodge=1), batch_size=batch_size)
    test_loader = build_dataloader(NFollowingDataset(test_docs, window_len, to_guess, window_dodge=1), batch_size=batch_size)

    return train_loader, val_loader, test_loader, vocab_size

def train_encdec():
    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    index_padding = 0
    criterion = nn.NLLLoss(ignore_index=index_padding)

    train_loader, val_loader, test_loader, vocab_size = build_split_loaders(
        wandb.config.dataset,
        train_perc=wandb.config.train_ratio,
        val_perc=wandb.config.val_ratio,
        test_perc=wandb.config.test_ratio,
        window_len=wandb.config.whole_sequence_length,
        to_guess=wandb.config.to_guess,
        batch_size=wandb.config.batch_size,
        limit_genres=wandb.config.limit_genres,
        max_docs_per_genre=wandb.config.max_docs_per_genre,
        augment=wandb.config.augment,
        window_dodge=wandb.config.window_dodge
    )

    encoder = EncoderWords(
        input_vocab_size=vocab_size,
        embedding_size=wandb.config.embedding_size,
        hidden_size=wandb.config.hidden_size,
        dropout_rate=wandb.config.dropout_rate,
        nl=wandb.config.lstm_layers
    )
    decoder = DecoderWords(
        wandb.config.to_guess,
        device,
        vocab_size,
        embedding_size=wandb.config.embedding_size,
        hidden_size=wandb.config.hidden_size,
        dropout_rate=wandb.config.dropout_rate,
        nl=wandb.config.lstm_layers
    )

    model = EncDecWords(encoder, decoder, device, 'cuda' if torch.cuda.is_available() else 'cpu').to(device)
    optimizer = Adam(model.parameters(), lr=wandb.config.lr)

    train_losses, accuracies, val_losses, best_epoch, best_val_loss = model.train_model(
        train_loader, val_loader, wandb.config.epochs, optimizer, criterion, scaler
    )


if __name__ == '__main__':
    wandb.init(
        project='rnn-midi',
        config={
            "model": "encoder-decoder",
            "embedding_size": 1024,
            "hidden_size": 1024,
            "dropout_rate": .2,
            "learning_rate": 1e-3,
            "epochs": 10,
            "lstm_layers": 2,
            "whole_sequence_length": 128,
            "to_guess": 1,
            "train_perc": 0.8,
            "val_perc": 0.1,
            "test_perc": 0.1,
            "batch_size": 64,
            "dataset": "texts-12",
            "limit_genres": None,
            "max_docs_per_genre": 0,
            'augment': 12
        }
    )
    train_encdec()
