from dataset import FMDataset
from encoder import FMEncoder
from fm_ddsp_batch import fm_renderer_batch, make_mod_matrix_batch
from loss_batch import (compute_spectrogram_cqt_batched, 
                        cqt_spectrogram_loss_batched,
                        make_fundamental_weight_batched)

import torch
from torch.utils.data import DataLoader
from nnAudio.features import CQT2010v2
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description='Trains existing dataset from generate_dataset')
parser.add_argument('--data_dir', type=str)
parser.add_argument('--Fs', type=int, default=16000)
parser.add_argument('--duration', type=float, default=1.0)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--n_epochs', type=int, default=25)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--start_epoch', type=int, default=0)


def make_fundamental_weight(n_bins, fundamental_bin, bins_per_octave,
                             suppression=0.1, width=2.0):
    bins = torch.arange(n_bins, dtype=torch.float32)
    gaussian = torch.exp(
        -0.5 * ((bins - fundamental_bin) / (width * bins_per_octave / 12)) ** 2
    )
    freq_weights = 1.0 - (1.0 - suppression) * gaussian
    return freq_weights


def train_stage1(args):
    # set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Cuda? {torch.cuda.is_available()}")

    # create dataset + dataloader
    dataset = FMDataset(save_dir=args.data_dir)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=0)

    # create encoder + optimizer + scheduler
    encoder = FMEncoder(n_bins=224).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=True
    )

    # CQT transform
    Fs = args.Fs
    duration = args.duration
    hop_size = 512
    bins_per_octave = 32
    n_octaves = 7
    n_bins = bins_per_octave * n_octaves
    fmin = 32.7  # CQT2010v2 default

    cqt_transform = CQT2010v2(
        sr=Fs,
        hop_length=hop_size,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave
    ).to(device)

    # frequency weights — suppress fundamental contribution
    # stage 1 has fixed f0=440 so one weight vector suffices, broadcast across batch
    f0_stage1 = 440.0
    fundamental_bin = int(bins_per_octave * np.log2(f0_stage1 / fmin))
    freq_weights = make_fundamental_weight(
        n_bins, fundamental_bin, bins_per_octave,
        suppression=0.2, width=1.2
    ).to(device)
    freq_weights = freq_weights.unsqueeze(0)  # [1, n_bins] — broadcasts across batch

    # output dir for checkpoints
    output_dir = os.path.join(args.data_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    # optionally resume from checkpoint
    if args.resume:
        encoder.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed from {args.resume}")

    # debug — track weight change per epoch
    initial_weight = encoder.fc1.weight.data.clone()

    for epoch in range(args.start_epoch, args.n_epochs):
        epoch_loss = 0.0
        print(f"Epoch {epoch}/{args.n_epochs}:")

        for batch in dataloader:
            params, spec = batch
            spec = spec.float().to(device)
            f0_batch = params['f0'].float().to(device)

            optimizer.zero_grad()

            # encoder forward pass — full batch at once
            predicted = encoder(spec)

            # batched FM render — no Python item loop
            mod_matrices = make_mod_matrix_batch(predicted['mod_values'])
            audio_batch = fm_renderer_batch(
                f0_batch,
                predicted['ratios'],
                predicted['levels'],
                mod_matrices,
                predicted['carrier_weights'],
                Fs, duration
            )

            # check for NaN audio before computing loss
            if torch.isnan(audio_batch).any():
                print("NaN in audio_batch, skipping batch")
                optimizer.zero_grad()
                continue

            # batched CQT spectrogram
            pred_specs = compute_spectrogram_cqt_batched(audio_batch, cqt_transform)

            # batched loss with fundamental suppression
            spectral_loss = cqt_spectrogram_loss_batched(pred_specs, spec, freq_weights)
            lambda_sparse = 0.001
            sparsity_loss = predicted['levels'].sum(dim=1).mean() * lambda_sparse
            loss = spectral_loss + sparsity_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=0.5)
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss_avg = epoch_loss / len(dataloader)
        print(f"Average epoch loss: {epoch_loss_avg}")

        weight_change = (encoder.fc1.weight.data - initial_weight).abs().mean()
        print(f"Average weight change: {weight_change.item()}")
        initial_weight = encoder.fc1.weight.data.clone()

        scheduler.step(epoch_loss_avg)

        # only save checkpoint if weights are valid
        if not torch.isnan(encoder.fc1.weight).any():
            torch.save(encoder.state_dict(),
                       os.path.join(output_dir, f'encoder_epoch_{epoch}.pt'))
        else:
            print(f"NaN weights detected at epoch {epoch} — stopping training")
            break


def train(args):
    """General training function — no stage-specific constraints."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Cuda? {torch.cuda.is_available()}")

    dataset = FMDataset(save_dir=args.data_dir)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=0)

    encoder = FMEncoder(n_bins=224).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5, verbose=True
    )

    Fs = args.Fs
    duration = args.duration
    hop_size = 512
    bins_per_octave = 32
    n_octaves = 7
    n_bins = bins_per_octave * n_octaves
    fmin = 32.7

    cqt_transform = CQT2010v2(
        sr=Fs,
        hop_length=hop_size,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave
    ).to(device)

    output_dir = os.path.join(args.data_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    if args.resume:
        encoder.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed from {args.resume}")

    initial_weight = encoder.fc1.weight.data.clone()

    for epoch in range(args.start_epoch, args.n_epochs):
        epoch_loss = 0.0
        print(f"Epoch {epoch}/{args.n_epochs}:")

        for batch in dataloader:
            params, spec = batch
            spec = spec.float().to(device)
            f0_batch = params['f0'].float().to(device)

            # compute per-example fundamental bins for freq weighting
            fundamental_bins = (bins_per_octave * torch.log2(
                f0_batch / fmin)).int().to(device)
            freq_weights = make_fundamental_weight_batched(
                n_bins, fundamental_bins, bins_per_octave,
                suppression=0.2, width=1.2, device=device
            )

            optimizer.zero_grad()
            predicted = encoder(spec)

            mod_matrices = make_mod_matrix_batched(predicted['mod_values'])
            audio_batch = fm_renderer_batched(
                f0_batch,
                predicted['ratios'],
                predicted['levels'],
                mod_matrices,
                predicted['carrier_weights'],
                Fs, duration
            )

            if torch.isnan(audio_batch).any():
                print("NaN in audio_batch, skipping batch")
                optimizer.zero_grad()
                continue

            pred_specs = compute_spectrogram_cqt_batched(audio_batch, cqt_transform)
            loss = cqt_spectrogram_loss_batched(pred_specs, spec, freq_weights)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=0.5)
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss_avg = epoch_loss / len(dataloader)
        print(f"Average epoch loss: {epoch_loss_avg}")

        weight_change = (encoder.fc1.weight.data - initial_weight).abs().mean()
        print(f"Average weight change: {weight_change.item()}")
        initial_weight = encoder.fc1.weight.data.clone()

        scheduler.step(epoch_loss_avg)

        if not torch.isnan(encoder.fc1.weight).any():
            torch.save(encoder.state_dict(),
                       os.path.join(output_dir, f'encoder_epoch_{epoch}.pt'))
        else:
            print(f"NaN weights detected at epoch {epoch} — stopping training")
            break


if __name__ == '__main__':
    args = parser.parse_args()
    train_stage1(args)