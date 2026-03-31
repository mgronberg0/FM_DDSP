from dataset import FMDataset
from encoder import FMEncoder
from fm_ddsp import fm_renderer, make_mod_matrix
from loss import cqt_spectrogram_loss

import torch
from torch.utils.data import DataLoader
import torchaudio
import nnAudio as nn
from nnAudio.features import CQT2010v2
import numpy as np
import sys
import os
import random

# set up device
device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# create dataset
dataset = FMDataset(data_set_dir).to(device)
# create dataloader
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4).to(device)
# create encoder
encoder = FMEncoder().to(device)
# create optimizer
optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr )
# create cqt transform
Fs = args.Fs
hop_size = 512
bins_per_octave = 32
n_octaves=7
n_bins = bins_per_octave*n_octaves
cqt_transform = CQT2010v2(sr = Fs,
                          hop_length = hop_size,
                          n_bins = n_bins,
                          bins_per_octave = bins_per_octave)
