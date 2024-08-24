import torch
import wandb
from torch import nn, GradScaler
from torch.optim import Adam

from dataloader import set_seed, build_split_loaders
from rnn_timedistributed import RNNTD

count = 10
sweep_configuration = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_acc"},
    "parameters": {
        "embedding_size": {"values": [128,256,512]},
        "hidden_size": {"values": [128,256,512]},
        "dropout_rate": {"max": 0.8, "min": 0.01},
        "learning_rate": {"values": [0.0001, 0.0003, 0.001, 0.003, 0.01]},
        "lstm_layers": {"values": [1, 2, 3, 4]},
        "whole_sequence_length": {"values": [16, 32, 64, 128, 256]},
        "batch_size": {"values": [4, 32, 64]},
        'augment': {"values": [0, 4, 6, 8, 12]},
        'window_overlap': {"values": [1, 2, 4, 8, 16]}
    },
}
dataset_sampling_frequency = 'texts-12'
dataset_format = '.notewise'
limit_genres = ['Classical']
train_ratio = 0.8
val_ratio = test_ratio = 0.1
to_guess = 1
max_docs_per_genre = 0

def train_model():
    wandb.init(project="rnn-midi")
    set_seed()
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    train_loader, val_loader, test_loader, vocab_size = build_split_loaders(
        dataset_sampling_frequency, dataset_format,
        train_perc=train_ratio, val_perc=val_ratio, test_perc=test_ratio,
        window_len=wandb.config.whole_sequence_length, to_guess=to_guess, batch_size=wandb.config.batch_size,
        limit_genres=limit_genres, max_docs_per_genre=max_docs_per_genre,
        augment=wandb.config.augment, window_overlap=wandb.config.window_overlap
    )
    model = RNNTD(
        device,
        device_str,
        input_vocab_size=vocab_size,
        embedding_size=wandb.config.embedding_size,
        hidden_size=wandb.config.hidden_size,
        dropout_rate=wandb.config.dropout_rate,
        nl=wandb.config.lstm_layers
    ).to(device)
    optimizer = Adam(model.parameters(), lr=wandb.config.learning_rate)
    wandb.watch(model, log_freq=10000)
    index_padding = 0
    criterion = nn.NLLLoss(ignore_index=index_padding)
    scaler = GradScaler(device_str)
    train_losses, accuracies, val_losses, best_epoch, best_val_loss = model.train_model(
        train_loader, val_loader, 5, optimizer, criterion, scaler
    )

if __name__ == '__main__':
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="rnn-midi")
    wandb.agent(sweep_id, function=train_model, count=count)