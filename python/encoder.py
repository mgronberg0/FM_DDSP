import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

class FMEncoder(nn.Module):
    def __init__(self, n_mels=256 ): #n_channels=1
        super().__init__()
        num_features = 128
        self.fc1 = nn.Linear(n_mels, 2*n_mels)
        self.fc2 = nn.Linear(2*n_mels, n_mels)
        self.fc3 = nn.Linear(n_mels, num_features)
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

def compute_spectrogram(audio, Fs = 16000, n_fft = 4096, n_mels = 256):

    mel_transform = T.MelSpectrogram(
        sample_rate = Fs,
        n_fft = n_fft,
        hop_length = n_fft // 4,
        n_mels = n_mels
    )
    spectrogram = mel_transform(audio.unsqueeze(0))
    spectrogram = torch.log1p(spectrogram)
    spectrogram = spectrogram.mean(dim=2)
    print(spectrogram.shape)

    # output is expects [channels, n_mels]
    return spectrogram
