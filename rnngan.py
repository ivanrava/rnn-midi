import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import tqdm

from dataloader import set_seed
from utils import log

noise_size = 256
embedding_size = 1024
hidden_size_d = hidden_size_g = 1024
sequence_length = 256
output_size = 1
batch_size = 64
learning_rate = 0.0002
augment = 12
nl = 1
dropout = 0.1

PAD_TOKEN = '<PAD>'
PAD_IDX = 0

class GANDataset(Dataset):
    def __init__(self, docs: list, maxlen: int = 100):
        self.docs = docs
        self.maxlen = maxlen

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        doc = self.docs[idx]
        if len(doc) > self.maxlen:
            doc = doc[:self.maxlen]
        else:
            doc += [PAD_IDX] * (self.maxlen - len(doc))
        return torch.tensor(doc, dtype=torch.int64, device=torch.device('cpu'))

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, nl=1, dropout_rate=0.1):
        super(Generator, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=nl, dropout=dropout_rate)
        self.expansion = TimeDistributed(nn.Linear(hidden_size, output_size))

    def forward(self, z):
        # Z = random noise (batch_size, sequence_length, input_size)
        # h_0 = torch.zeros(1, z.size(0), self.hidden_size).to(z.device)
        # c_0 = torch.zeros(1, z.size(0), self.hidden_size).to(z.device)
        output, hidden = self.lstm(z)
        output = self.expansion(output)
        output = F.log_softmax(output, dim=-1)
        return output


class Discriminator(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, nl=1, dropout_rate=0.1):
        super(Discriminator, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=nl, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # X is the data to be discriminated (batch_size, sequence_length, input_size)
        #h_0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        #c_0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.lstm(embedded)
        out = torch.sigmoid(self.fc(output[:, -1, :]))
        return out


def train_gan(generator, discriminator, g_optimizer, d_optimizer, criterion, num_epochs, train_batches):
    for epoch in tqdm.tqdm(range(num_epochs)):
        for batch in tqdm.tqdm(train_batches, leave=False):
            real_data = batch.to(device)
            # Generate "real" labels: all 1s
            real_labels = torch.ones(batch_size, 1).to(device)

            # Real data to discriminator, update
            d_optimizer.zero_grad()
            real_output = discriminator(real_data)
            d_loss_real = criterion(real_output, real_labels)
            d_loss_real.backward()

            # Generate fake data
            noise = torch.randn(batch_size, sequence_length, noise_size).to(device)
            fake_data = generator(noise)
            # Generate "fake" labels: all 0s
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Fake data to discriminator, update
            fake_output = discriminator(fake_data.detach())
            d_loss_fake = criterion(fake_output, fake_labels)
            d_loss_fake.backward()

            d_optimizer.step()

            # Generator update
            g_optimizer.zero_grad()
            fake_output = discriminator(fake_data)
            g_loss = criterion(fake_output, real_labels)
            g_loss.backward()

            g_optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], D Loss: {d_loss_real + d_loss_fake}, G Loss: {g_loss}")


def read_documents(texts_folder: str) -> list:
    log("Reading MIDI documents...")

    documents = []
    for root, _, files in os.walk(texts_folder):
        for file in tqdm.tqdm(files):
            if file.endswith(".notewise"):
                try:
                    with open(os.path.join(root, file), 'r') as f:
                        documents.append(f.read().split(" "))
                except Exception as e:
                    print(e)

    log(f"Documents read: {len(documents)}")

    return documents

def augment_docs(docs: list, augment: int = 12, max_note: int = 95) -> list:
    augmented_docs = []
    for doc in tqdm.tqdm(docs):
        for t in tqdm.tqdm(range(0, augment), leave=False):
            augmented_doc = []
            offset = int(t - augment / 2)
            for word in tqdm.tqdm(doc, leave=False):
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

def build_vocab(docs: list, augment: int = 12, max_note: int = 95):
    log("Building vocabulary...")
    vocab = {PAD_TOKEN: PAD_IDX}
    vocab_size = len(vocab)
    for doc in tqdm.tqdm(docs):
        for word in tqdm.tqdm(doc, leave=False):
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

def build_split_loaders(
        folder: str,
        train_perc: float, val_perc: float, test_perc: float,
        batch_size: int, augment: int, maxlen: int
):
    docs = read_documents(folder)
    train_docs, val_docs, test_docs = split_list(docs, train_perc, val_perc, test_perc)
    if augment > 0:
        train_docs = augment_docs(train_docs, augment)
        random.shuffle(train_docs)
    vocab, vocab_size = build_vocab(train_docs + val_docs + test_docs, augment=augment)
    train_docs = [vocab[w] for w in train_docs]
    val_docs = [vocab[w] for w in val_docs]
    test_docs = [vocab[w] for w in test_docs]

    train_loader = DataLoader(GANDataset(train_docs, maxlen), batch_size=batch_size)
    val_loader = DataLoader(GANDataset(val_docs, maxlen), batch_size=batch_size)
    test_loader = DataLoader(GANDataset(test_docs, maxlen), batch_size=batch_size)

    return train_loader, val_loader, test_loader, vocab_size


if __name__ == '__main__':
    set_seed()
    train_loader, val_loader, test_loader, vocab_size = build_split_loaders(
        'texts-12', train_perc=0.8, val_perc=0.1, test_perc=0.1,
        batch_size=batch_size, augment=augment, maxlen=sequence_length,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator(noise_size, hidden_size_g, vocab_size, nl=nl, dropout_rate=dropout).to(device)
    discriminator = Discriminator(vocab_size, embedding_size, hidden_size_d, nl=nl, dropout_rate=dropout).to(device)

    g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    train_gan(generator, discriminator, g_optimizer, d_optimizer, criterion,
              num_epochs=50, train_batches=train_loader)
