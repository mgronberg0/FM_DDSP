import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from nnAudio.features import CQT2010v2

class FMEncoder(nn.Module):
    def __init__(self, n_bins=224 ): #n_bins comes from 7 octaves of 32 bins per octave
        super().__init__()
        num_features = 128
        self.fc1 = nn.Linear(n_bins, 2*n_bins)
        self.fc2 = nn.Linear(2*n_bins, n_bins)
        self.fc3 = nn.Linear(n_bins, num_features)
        self.levels_head = nn.Linear(num_features, 4)
        self.mod_values_head = nn.Linear(num_features, 7)
        self.ratios_head = nn.Linear(num_features,4)
        self.carrier_weights_head = nn.Linear(num_features, 4)

        # # CCN
        # self.conv_block1 = nn.Sequential(
        #     nn.Conv2d(1, 32, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2)
        # )
        # self.conv_block2 = nn.Sequential(
        #     nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2)
        # )
        

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # pass through each output head with it's activation
        levels = torch.sigmoid(self.levels_head(x))
        mod_values = torch.sigmoid(self.mod_values_head(x))
        ratios = F.softplus(self.ratios_head(x)) + 0.25
        cw = self.carrier_weights_head(x)
        carrier_weights = cw / (cw.sum(dim=1, keepdim=True) + 1e-8)

        return {
            'levels': levels,
            'mod_values': mod_values,
            'ratios': ratios,
            'carrier_weights': carrier_weights
        }

def compute_spectrogram_mel(audio, Fs = 16000, n_fft = 4096, n_mels = 256):

    mel_transform = T.MelSpectrogram(
        sample_rate = Fs,
        n_fft = n_fft,
        hop_length = n_fft // 4,
        n_mels = n_mels
    )
    spectrogram = mel_transform(audio.unsqueeze(0))
    spectrogram = torch.log1p(spectrogram)
    spectrogram = spectrogram.mean(dim=2)

    # output is expects [channels, n_mels]
    return spectrogram

def compute_spectrogram_cqt(audio, cqt_transform):
    # Fs = 16000, hop_size = 512, bins_per_octave = 32, n_octaves=7
    # cqt_transform = CQT2010v2(sr = Fs,
    #                           hop_length = hop_size,
    #                           n_bins = bins_per_octave*n_octaves,
    #                           bins_per_octave = bins_per_octave)
    spec = cqt_transform(audio.unsqueeze(0))
    spec = spec.abs()
    spec = torch.log1p(spec)
    spec = spec.mean(dim=2)
    spec = spec.squeeze()
    # output is expects [n_bins]
    return spec
