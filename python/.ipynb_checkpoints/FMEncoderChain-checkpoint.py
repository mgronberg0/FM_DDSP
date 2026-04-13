import torch
import torch.nn as nn
import torch.nn.funtional as F
import torchaudio.transforms as T
from nnAudio.features import CQT2010v2

class FMEncoderChain(nn.Module):
    def __init__(self, n_bins = 224):
        super().__init__()
        num_features = 128 # TODO: experiment with lower values
        self.fc1 = nn.Linear(n_bins, 2*n_bins)
        self.bn1 = nn.BatchNorm1d(2*n_bins)
        self.fc2 = nn.Linear(2*n_bins, n_bins)
        self.bn2 = nn.BatchNorm1d(n_bins)
        self.fc3 = nn.Linear(n_bins, num_features)
        self.bn3 = nn.BatchNorm1d(num_features)

        self.ratios_head = nn.Linear(num_features, 2)
        self.levels_head = nn.Linear(num_features, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))

        # integer-like results from 1.0 to 8.0
        ratios = torch.sigmoid(self.ratios_head(x)) * 7.0 + 1.0

        # levels allow for [0.0, 4.0] to allow overmodulation
        levels = torch.sigmoid(self.levels_head(x)) * 4.0 # TODO: explore alternatives

        return {'ratios': ratios, 'levels': levels}