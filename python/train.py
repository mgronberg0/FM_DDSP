from dataset import FMDataset
from encoder import FMEncoder, compute_spectrogram_cqt
from fm_ddsp import fm_renderer, make_mod_matrix
from loss import cqt_spectrogram_loss_enhanced2

import torch
from torch.utils.data import DataLoader
import torchaudio
import nnAudio as nn
from nnAudio.features import CQT2010v2
import numpy as np
import sys
import os
import random
import argparse

parser = argparse.ArgumentParser(description = 'Trains existing dataset from generate_dataset')

parser.add_argument('--data_dir',type=str)
parser.add_argument('--Fs', type=int, default = 16000)
parser.add_argument('--duration', type = float, default = 1.0)
parser.add_argument('--lr', type=float, default = 0.0001)
parser.add_argument('--n_epochs', type=int, default = 25)
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--start_epoch', type=int, default=0)


def train(args):
    # set up device
    device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Cuda? {torch.cuda.is_available()}")
    # create dataset
    dataset = FMDataset(save_dir = args.data_dir)
    # create dataloader
    dataloader = DataLoader(dataset, 
                        batch_size=64, 
                        shuffle=True, 
                        num_workers=1)
    # create encoder
    encoder = FMEncoder(n_bins = 224).to(device)
    # create optimizer
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr )
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        factor=0.5,
        verbose=True
    )
    # create cqt transform
    Fs = args.Fs
    duration = args.duration
    hop_size = 512
    bins_per_octave = 32
    n_octaves=7
    n_bins = bins_per_octave*n_octaves
    cqt_transform = CQT2010v2(sr = Fs,
                              hop_length = hop_size,
                              n_bins = n_bins,
                              bins_per_octave = bins_per_octave).to(device)

    # create output dir for checkpoints
    output_dir = os.path.join(args.data_dir,'output')
    os.makedirs(output_dir, exist_ok=True)
    # debug
    initial_weight = encoder.fc1.weight.data.clone()
    for epoch in range(args.n_epochs):
        epoch_loss = 0.0
        print(f"Epoch {epoch}/{args.n_epochs}:")
        for batch in dataloader:
            # get data from batch
            params, spec = batch
            spec = spec.float().to(device)
            optimizer.zero_grad()
            predicted = encoder(spec)
            batch_size = spec.shape[0]
            batch_loss = 0.0
            for i in range(batch_size):
                # create audio
                f0 = params['f0'][i].item()
                audio_i = fm_renderer(
                    f0,
                    predicted['ratios'][i],
                    predicted['levels'][i],
                    make_mod_matrix(predicted['mod_values'][i]),
                    predicted['carrier_weights'][i],
                    Fs, duration)
                pred_spec = compute_spectrogram_cqt(audio_i,cqt_transform)
                batch_loss += cqt_spectrogram_loss_enhanced1(pred_spec, spec[i])
            # calculate loss
            loss = batch_loss / batch_size
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            optimizer.step()
            # calculate epoch loss
            epoch_loss+=loss.item()
        epoch_loss_avg = epoch_loss / len(dataloader)
        print(f"Average epoch loss: {epoch_loss_avg}")
        weight_change = (encoder.fc1.weight.data - initial_weight).abs().mean()
        print(f"Average weight change: {weight_change.item()}")
        initial_weight = encoder.fc1.weight.data.clone()
        # Save checkpoint
        torch.save(encoder.state_dict(), 
                   os.path.join(output_dir, f'encoder_epoch_{epoch}.pt'))

def train_stage1(args):
    # set up device
    device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Cuda? {torch.cuda.is_available()}")
    # create dataset
    dataset = FMDataset(save_dir = args.data_dir)
    # create dataloader
    dataloader = DataLoader(dataset, 
                        batch_size=32, 
                        shuffle=True, 
                        num_workers=1)
    # create encoder
    encoder = FMEncoder(n_bins = 224).to(device)
    # create optimizer
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr )
    # create cqt transform
    Fs = args.Fs
    duration = args.duration
    hop_size = 512
    bins_per_octave = 32
    n_octaves=7
    n_bins = bins_per_octave*n_octaves
    cqt_transform = CQT2010v2(sr = Fs,
                              hop_length = hop_size,
                              n_bins = n_bins,
                              bins_per_octave = bins_per_octave).to(device)
    f0 =  440.0
    fmin = 32.7 # default of cqt_transform
    # create frequency weights and reduce fundamental
    freq_weights = torch.ones(n_bins)
    fundamental_bin = int(bins_per_octave * np.log2(f0 / fmin))
    freq_weights = make_fundamental_weight(n_bins, fundamental_bin, bins_per_octave, 
                                           suppression = .2, 
                                           width = 1.2).to(device)
    # create output dir for checkpoints
    output_dir = os.path.join(args.data_dir,'output')
    os.makedirs(output_dir, exist_ok=True)
    # debug
    initial_weight = encoder.fc1.weight.data.clone()
    # resume function
    if args.resume:
        encoder.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed from {args.resume}")
    for epoch in range(args.start_epoch, args.n_epochs):
        epoch_loss = 0.0
        print(f"Epoch {epoch}/{args.n_epochs}:")
        for batch in dataloader:
            # get data from batch
            params, spec = batch
            spec = spec.float().to(device)
            optimizer.zero_grad()
            predicted = encoder(spec)
            batch_size = spec.shape[0]
            batch_loss = 0.0
            for i in range(batch_size):
                # create audio
                f0 = params['f0'][i].item()
                audio_i = fm_renderer(
                    f0,
                    predicted['ratios'][i],
                    predicted['levels'][i],
                    make_mod_matrix(predicted['mod_values'][i]),
                    predicted['carrier_weights'][i],
                    Fs, duration)
                if torch.isnan(audio_i).any():
                    print(f"NaN in audio_i!")
                    print(f"f0: {params['f0'][i].item()}")
                    print(f"ratios: {predicted['ratios'][i]}")
                    print(f"levels: {predicted['levels'][i]}")
                    print(f"mod_values: {predicted['mod_values'][i]}")
                    print(f"carrier_weights: {predicted['carrier_weights'][i]}")
                    break
                pred_spec = compute_spectrogram_cqt(audio_i,cqt_transform)
                # print(f"pred_spec: min={pred_spec.min():.4f}, max={pred_spec.max():.4f}, has_nan={torch.isnan(pred_spec).any()}")
                # print(f"spec: min={spec.min():.4f}, max={spec.max():.4f}, has_nan={torch.isnan(spec).any()}")
                # apply frequency weighting to subdue loss contribution from fundamental frequency
                batch_loss += cqt_spectrogram_loss_enhanced2(pred_spec*freq_weights, 
                                                             spec[i]*freq_weights)
            # calculate loss
            loss = batch_loss / batch_size
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            optimizer.step()
            # calculate epoch loss
            epoch_loss+=loss.item()
        epoch_loss_avg = epoch_loss / len(dataloader)
        print(f"Average epoch loss: {epoch_loss_avg}")
        weight_change = (encoder.fc1.weight.data - initial_weight).abs().mean()
        print(f"Average weight change: {weight_change.item()}")
        initial_weight = encoder.fc1.weight.data.clone()
        # Save checkpoint
        torch.save(encoder.state_dict(), 
                   os.path.join(output_dir, f'encoder_epoch_{epoch}.pt'))
                
def make_fundamental_weight(n_bins, fundamental_bin, bins_per_octave, suppression = 0.1, width = 2.0):
    bins = torch.arange(n_bins, dtype=torch.float32)
    gaussian = torch.exp(-0.5 * ((bins - fundamental_bin) / (width*bins_per_octave / 12)) ** 2)
    freq_weights = 1.0 - (1.0 - suppression) * gaussian
    return freq_weights                
                
            

if __name__ == '__main__':
    args = parser.parse_args()
    train_stage1(args)
