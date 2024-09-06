import torch
from torch import nn, GradScaler
from torch.optim import Adam
import wandb

from dataloader import set_seed, build_split_loaders, build_split_loaders_triplet
from encoder_decoder import EncDecWords, EncoderWords, DecoderWords
from rnn_plain import RNNPlain
from rnn_timedistributed import RNNTD
from rnn_triplet import RNNTriplet
from utils import log

if __name__ == '__main__':
    set_seed()

    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    log(f"Performing training on {device}")

    #model_str = "encoder-decoder"
    # model_str = "rnn-plain"
    # model_str = "rnn-triplet"
    model_str = "rnn-timedistributed"

    embedding_size = 512
    hidden_size = 512
    dropout_rate = .2
    lr = 1e-3
    epochs = 5
    lstm_layers = 2
    batch_size = 64

    augment = 12

    dataset_sampling_frequency = 'texts-12'
    dataset_format = '.notewise'
    limit_genres = ['Classical']
    max_docs_per_genre = 0

    index_padding = 0
    criterion = nn.NLLLoss(ignore_index=index_padding)
    scaler = GradScaler(device_str)

    whole_sequence_length = 128
    window_dodge = 32
    to_guess = 1

    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    wandb.init(
        project='rnn-midi',
        config={
            "model": model_str,
            "torch_device": device_str,
            "embedding_size": embedding_size,
            "hidden_size": hidden_size,
            "dropout_rate": dropout_rate,
            "learning_rate": lr,
            "epochs": epochs,
            "lstm_layers": lstm_layers,
            "criterion": "NLLLoss",
            "whole_sequence_length": whole_sequence_length,
            "to_guess": to_guess,
            "train_perc": train_ratio,
            "val_perc": val_ratio,
            "test_perc": test_ratio,
            "batch_size": batch_size,
            "dataset_sampling_frequency": dataset_sampling_frequency,
            "dataset_format": dataset_format,
            "limit_genres": limit_genres,
            "max_docs_per_genre": max_docs_per_genre,
            'augment': augment
        }
    )

    log(f'Embedding size: {embedding_size}')
    log(f'Hidden size: {hidden_size}')
    log(f'Dropout rate: {dropout_rate}')
    log(f'Learning rate: {lr}')
    log(f'Epochs: {epochs}')
    log(f'LSTM layers: {lstm_layers}')

    log(f'Sequence length: {whole_sequence_length} (to guess: {to_guess})')

    log(f'Dataset percentages: {train_ratio}:{val_ratio}:{test_ratio}')

    log(f'Batch size: {batch_size}')

    if model_str == 'rnn-plain' or model_str == 'rnn-timedistributed':
        train_loader, val_loader, test_loader, vocab_size = build_split_loaders(
            dataset_sampling_frequency, dataset_format,
            train_perc=train_ratio, val_perc=val_ratio, test_perc=test_ratio,
            window_len=whole_sequence_length, to_guess=to_guess, batch_size=batch_size,
            limit_genres=limit_genres, max_docs_per_genre=max_docs_per_genre,
            augment=augment, window_dodge=window_dodge
        )
    elif model_str == 'rnn-triplet':
        train_loader, val_loader, test_loader, vocab_size = build_split_loaders_triplet(
            train_perc=train_ratio, val_perc=val_ratio, test_perc=test_ratio,
            window_len=whole_sequence_length, to_guess=to_guess, batch_size=batch_size,
        )

    model = None
    optimizer = None
    if model_str == 'encoder-decoder':
        encoder = EncoderWords(
            input_vocab_size=vocab_size,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            nl=lstm_layers
        )
        decoder = DecoderWords(
            to_guess,
            device,
            vocab_size,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            nl = lstm_layers
        )

        model = EncDecWords(encoder, decoder, device, device_str).to(device)
        optimizer = Adam(model.parameters(), lr=lr)
    elif model_str == 'rnn-plain':
        model = RNNPlain(
            device,
            device_str,
            input_vocab_size=vocab_size,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            nl=lstm_layers
        ).to(device)
        optimizer = Adam(model.parameters(), lr=lr)
    elif model_str == 'rnn-triplet':
        model = RNNTriplet(
            device,
            device_str,
            input_vocab_size=96,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            nl=lstm_layers
        ).to(device)
        optimizer = Adam(model.parameters(), lr=lr)
    elif model_str == 'rnn-timedistributed':
        model = RNNTD(
            device,
            device_str,
            input_vocab_size=vocab_size,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            nl=lstm_layers
        ).to(device)
        optimizer = Adam(model.parameters(), lr=lr)

    log(f"Starting training for {epochs} epochs...")
    wandb.watch(model, log_freq=10000)
    train_losses, accuracies, val_losses, best_epoch, best_val_loss = model.train_model(
        train_loader, val_loader, epochs, optimizer, criterion, scaler
    )
    log(f"Training finished! Best: epoch {best_epoch} with {best_val_loss}")

    wandb.finish()
