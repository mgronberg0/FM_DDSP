import torch
import torchaudio
import numpy as np
import sys
import os
import random
sys.path.append('..')
from fm_ddsp import fm_renderer, make_mod_matrix



class FMDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 n_examples = 10000, 
                 Fs = 16000, 
                 duration = 1.0, 
                 n_fft=4096):
        super().__init__()
        self.data = []
        self.n_examples = n_examples
        self.Fs = Fs
        self.duration = duration
        self.n_fft = n_fft
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.Fs,
            n_fft=self.n_fft,
            n_mels = 256,
            hop_length = self.n_fft//4
        )
        self._generate()

    def __len__ (self):
        return self.n_examples

    def __getitem__(self, idx):
        return self.data[idx]
        
    def _generate(self):
        ratio_choices = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        for i in range(self.n_examples):
            if i%1000 == 0 or i==self.n_examples-1:
                print(f"Generating example {i+1}/{self.n_examples}")
            # Midi Note Ranges
            midi_note = random.randint(36, 72)
            f0 = 440.0 * 2 ** ((midi_note - 69)/12.0)
            # Algorithm Ranges
            alg = random.choice(ALGORITHMS)
            # Mod Matrix, Carrier wieghts ranges
            mod_values = torch.rand(7) * torch.tensor(alg['mod_mask'], dtype=torch.float32)
            carrier_weights = torch.rand(4) * torch.tensor(alg['carrier_mask'], dtype=torch.float32)
            # Ratio and operator level ranges
            ratios = torch.tensor([random.choice(ratio_choices) for _ in range(4)])
            levels = torch.rand(4) * 0.9 + 0.1
            # form into mod matrix
            mod_matrix = make_mod_matrix(mod_values)
            # create audio
            with torch.no_grad():
                audio = fm_renderer(f0, ratios, levels, mod_matrix, carrier_weights, self.Fs, self.duration)
            mel_spec = self.mel_transform(audio)
            self.data.append({
                'spectrogram': torch.log1p(mel_spec).mean(dim=1),
                'f0':f0,
                'algorithm':alg['name'],
                'mod_values':mod_values,
                'ratios':ratios,
                'levels':levels,
                'carrier_weights':carrier_weights
            })

            

            

        


