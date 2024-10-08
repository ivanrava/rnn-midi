import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast
from tqdm import tqdm
import wandb

from utils import log


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


class RNNTD(nn.Module):
    def __init__(self, device, device_str, input_vocab_size: int, embedding_size=128, hidden_size=256, dropout_rate=0.1, nl=4, *args, **kwargs):
        super(RNNTD, self).__init__(*args, **kwargs)

        self.embedding = nn.Embedding(input_vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True, num_layers=nl)
        self.expansion = TimeDistributed(nn.Linear(hidden_size, input_vocab_size))

        self.dropout = nn.Dropout(dropout_rate)
        self.device = device
        self.device_str = device_str

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.lstm(embedded)
        output = self.expansion(output)
        output = F.log_softmax(output, dim=-1)
        return output

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
                    output_tensor = self(input_tensor)
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
                    output_tensor = self(input_tensor)
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

    def test_model(self, test_batches):
        self.eval()

        test_preds = []
        test_labels = []

        with torch.no_grad():
            for batch in test_batches:
                input_tensor, target_tensor = [b.to(self.device) for b in batch]

                with autocast(self.device_str):
                    output_tensor = self(input_tensor)
                    predictions = output_tensor.argmax(dim=-1)

                test_preds.extend(predictions.cpu().numpy().flatten())
                test_labels.extend(target_tensor.cpu().numpy().flatten())

        return test_preds, test_labels

    def generate_new_note(self, sequence):
        self.eval()

        with torch.no_grad():
            batched_input = torch.tensor(sequence).unsqueeze(0).to(self.device)
            with autocast(self.device_str):
                output_tensor = self(batched_input)
                predictions = output_tensor

        return predictions.cpu().numpy()