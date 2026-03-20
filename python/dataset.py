import torch
import torchaudio
import numpy as np
import sys
import random
sys.path.append('..')
from fm_ddsp import fm_renderer, make_mod_matrix

# Agorithms based on digitone's algos https://support.elektron.se/support/solutions/articles/43000566579-algorithms
# mod_mask = [mm[0][0], mm[1][0], mm[2][0], mm[2][1], mm[3][0], mm[3][1], mm[3][2]]
# * indicates carrier, [fb] = feedback
ALGORITHMS = [
    {
        # b2->b1*, a[fb]->c*, b1->c*
        # a->op0, b1->op1, b2->op2, c->op3
        'name': 'algo_1',
        'mod_mask': [1, 0, 1, 0, 1, 1, 0],
        'carrier_mask' : [0, 1, 1, 1],
    },
    {
        #a->c*, b2[fb]->b1*
        #b2->op0, b1->op1, a->op2, c->op3
        'name': 'algo_2',
        'mod_mask': [1, 1, 0, 0, 0, 0, 1],
        'carrier_mask': [0, 1, 0, 1],
    },
    {
        # a[fb]->c*, a[fb]->b2*, a[fb]->b1*
        # a->op0, b1->op1, b2->op2, c->op3
        'name': 'algo_3',
        'mod_mask':     [1, 1, 1, 0, 1, 0, 0],
        'carrier_mask': [0, 1, 1, 1]
    },
    {
        # b2[fb]->b1->a*->c*
        # b2->op0, b1->op1, a->op2, c->op3
        'name': 'algo_4',
        'mod_mask':     [1, 1, 0, 1, 0, 0, 1],
        'carrier_mask': [0, 0, 1, 1]
    },
    {
        # b2[fb]->b1, b2[fb]->a*, b1->a*, a->c*
        # b2->op0, b1->op1, a->op2, c->op3
        'name': 'algo_5',
        'mod_mask':     [1, 1, 1, 1, 0, 0, 1],
        'carrier_mask': [0, 0, 1, 1]
    },
    {
        # a[fb]->c*, a[fb]->b2*, b2->c*, b2->b1*
        # a->op0, b2->op1, b1->op2, c->op3
        'name': 'algo_6',
        'mod_mask':     [1, 1, 0, 0, 1, 1, 0],
        'carrier_mask': [0, 0, 1, 1]
    },
    {
        # a[fb]->c*, b2->b1*
        # a->op0, b2->op1, b1->op2, c->op3
        'name': 'algo_7',
        'mod_mask':     [1, 0, 0, 1, 1, 0, 0],
        'carrier_mask': [0, 0, 1, 1]
    },
    {
        # a->c*, b2*, b1[fb]*
        # b1->op0, a->op1, b2->op2, c->op3
        'name': 'algo_8',
        'mod_mask':     [1, 0, 0, 0, 0, 1, 0],
        'carrier_mask': [1, 0, 1, 1]
    },
]

class FMDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 n_examples = 10000, 
                 Fs = 16000, 
                 duration = 1.0, 
                 n_fft=4096):
        super().__init__()
        self.n_examples = n_examples
        self.Fs = Fs
        self.duration = duration
        self.n_fft = n_fft
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.Fs,
            n_fft=self.n_fft,
            n_mels = 256,
            hop_length = self.n_fft//4)
        )
        self._generate()

    def __len__ (self):
        return self.n_examples

    def __getitem__(self, idx):
        return self.data[idx]
        
    def _generate(self):
        ratio_choices = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        for i in range(self.n_examples):
            if i%1000 == 0:
                print(f"Generating example {i}/{self.n_examples}")
            midi_note = random.randint(36, 72)
            f0 = 440.0 * 2 ** ((midi_note - 69)/12.0)
            alg = random.choice(ALGORITHMS)
            mod_values = torch.rand(7) * torch.tensor(alg['mod_mask'], dtype=torch.float32)
            carrier_weights = torch.rand(4) * torch.tensor(alg['carrier_mask'], dtype=torch.float32)
            ratios = torch.tensor([random.choice(ratio_choices) for _ in range(4)])
            levels = torch.rand(4) * 0.9 + 0.1
            mod_matrix = make_mod_matrix(mod_values)
            with torch.no_grad()
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

            

            

        


