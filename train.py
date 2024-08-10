import torch
from torch import nn, GradScaler
from torch.optim import Adam

from dataloader import set_seed, build_split_loaders
from model import EncDecWords, EncoderWords, DecoderWords
from utils import log

if __name__ == '__main__':
    set_seed()

    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    log(f"Performing training on {device}")

    embedding_size = 128
    hidden_size = 256
    dropout_rate = .1
    lr = 1e-3
    epochs = 10

    log(f'Embedding size: {embedding_size}')
    log(f'Hidden size: {hidden_size}')
    log(f'Dropout rate: {dropout_rate}')
    log(f'Learning rate: {lr}')
    log(f'Epochs: {epochs}')

    criterion = nn.NLLLoss()
    scaler = GradScaler(device_str)

    whole_sequence_length = 10
    to_guess = 1
    log(f'Sequence length: {whole_sequence_length} (to guess: {to_guess})')

    train_perc = 0.8
    val_perc = 0.1
    test_perc = 0.1
    log(f'Dataset percentages: {train_perc}:{val_perc}:{test_perc}')

    batch_size = 64
    log(f'Batch size: {batch_size}')

    train_loader, val_loader, test_loader, vocab_size = build_split_loaders(
        "texts-12", ".notewise",
        train_perc=train_perc, val_perc=val_perc, test_perc=test_perc,
        window_len=whole_sequence_length, to_guess=to_guess, batch_size=batch_size
    )

    encoder = EncoderWords(
        input_vocab_size=vocab_size,
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        dropout_rate=dropout_rate
    )
    decoder = DecoderWords(
        to_guess,
        device,
        vocab_size,
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        dropout_rate=dropout_rate
    )

    model = EncDecWords(encoder, decoder, device, device_str).to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    log(f"Starting training for {epochs} epochs...")
    train_losses, accuracies, val_losses, best_epoch, best_val_loss = model.train_model(
        train_loader, val_loader, epochs, optimizer, criterion, scaler
    )
    log(f"Training finished! Best: epoch {best_epoch} with {best_val_loss}")
