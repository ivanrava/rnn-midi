import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from utils import log


class EncoderWords(nn.Module):
    def __init__(self, input_vocab_size: int, embedding_size=128, hidden_size=256, dropout_rate=0.1, *args, **kwargs):
        super(EncoderWords, self).__init__(*args, **kwargs)

        self.embedding = nn.Embedding(input_vocab_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.gru(embedded)
        return output, hidden

class DecoderWords(nn.Module):
    def __init__(self, num_guesses, device, output_size, embedding_size=128, hidden_size=256, dropout_rate=0.1, *args, **kwargs):
        super(DecoderWords, self).__init__(*args, **kwargs)

        self.embedding = nn.Embedding(output_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True)
        self.expansion = nn.Linear(hidden_size, output_size)

        self.num_guesses = num_guesses
        self.device = device

    def forward_step(self, decoder_input, decoder_hidden):
        output = self.embedding(decoder_input)
        output, hidden = self.gru(output, decoder_hidden)
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

            for batch in tqdm(train_batches, leave=False):
                optimizer.zero_grad()
                input_tensor, label_tensor = [x.to(self.device) for x in batch]

                with autocast(self.device_str):
                    output_tensor = self(input_tensor, label_tensor)
                    loss = criterion(output_tensor.view(-1), label_tensor.view(-1))

                ep_loss += loss.item()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            train_losses.append(ep_loss / len(train_batches))
            log(f"Train loss: {ep_loss/len(train_batches)}")

            val_loss, accuracy = self.eval_model(val_batches, criterion)
            log(f"Val loss: {val_loss}")
            log(f"Val accuracy: {accuracy}")
            accuracies.append(accuracy)
            val_losses.append(val_loss)

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
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in val_batches:
                input_tensor, label_tensor = [x.to(self.device) for x in batch]

                with autocast(self.device_str):
                    output_tensor = self(input_tensor, label_tensor)
                    loss = criterion(output_tensor.view(-1), label_tensor.view(-1))

                    predictions = output_tensor.argmax(dim=-1)

                val_loss += loss.item()
                val_preds.extend(predictions.cpu().numpy().flatten())
                val_labels.extend(label_tensor.cpu().numpy().flatten())

        return val_loss/len(val_batches), accuracy_score(val_labels, val_preds)

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