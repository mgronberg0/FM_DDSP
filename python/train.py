from dataset import FMDataset
from encoder import FMEncoder, compute_spectrogram_cqt
from fm_ddsp import fm_renderer, make_mod_matrix
from loss import cqt_spectrogram_loss_enhanced

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


def train(args):
    # set up device
    device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # create dataset
    dataset = FMDataset(save_dir = args.data_dir)
    # create dataloader
    dataloader = DataLoader(dataset, 
                        batch_size=32, 
                        shuffle=True, 
                        num_workers=4)
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
                audio_i = fm_renderer(
                    params['f0'][i].item(),
                    predicted['ratios'][i],
                    predicted['levels'][i],
                    make_mod_matrix(predicted['mod_values'][i]),
                    predicted['carrier_weights'][i],
                    Fs, duration)
                pred_spec = compute_spectrogram_cqt(audio_i,cqt_transform)
                batch_loss += cqt_spectrogram_loss_enhanced(pred_spec, spec[i])
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
                
                
                
            

if __name__ == '__main__':
    args = parser.parse_args()
    train(args)
