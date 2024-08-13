import torch
from torch import nn, GradScaler
from torch.optim import Adam
import wandb

from dataloader import set_seed, build_split_loaders
from encoder_decoder import EncDecWords, EncoderWords, DecoderWords
from rnn_plain import RNNPlain
from utils import log

if __name__ == '__main__':
    set_seed()

    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    log(f"Performing training on {device}")

    #model_str = "encoder-decoder"
    model_str = "rnn-plain"

    embedding_size = 256
    hidden_size = 512
    dropout_rate = .1
    lr = 1e-3
    epochs = 10
    lstm_layers = 4
    batch_size = 32

    dataset_sampling_frequency = 'texts-12'
    dataset_format = '.notewise'
    limit_genres = None
    max_docs_per_genre = 0

    criterion = nn.NLLLoss()
    scaler = GradScaler(device_str)

    whole_sequence_length = 10
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

    train_loader, val_loader, test_loader, vocab_size = build_split_loaders(
        dataset_sampling_frequency, dataset_format,
        train_perc=train_ratio, val_perc=val_ratio, test_perc=test_ratio,
        window_len=whole_sequence_length, to_guess=to_guess, batch_size=batch_size,
        limit_genres=limit_genres, max_docs_per_genre=max_docs_per_genre
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
