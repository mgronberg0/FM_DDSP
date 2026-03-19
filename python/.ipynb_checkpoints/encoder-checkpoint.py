import torch
import torch.nn as nn
import torch.nn.functional as F

class FMEncoder(nn.Module):
    def __init__(self, n_mels=256, n_channels=1):
        super().__init__()
        self.fc1 = nn.Linear(2049, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.levels_head = nn.Linear(128, 4)
        self.mod_values_head = nn.Linear(128, 7)
        self.ratios_head = nn.Linear(128,4)
        self.carrier_weights_head = nn.Linear(128, 4)

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

