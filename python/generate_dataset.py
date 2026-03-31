import argparse
import json
import torch
import torchaudio
import nnAudio as nn
import numpy as np
import sys
import os
import random
sys.path.append("..")
import fm_ddsp
from encoder import compute_spectrogram_cqt
from nnAudio.features import CQT2010v2

parser = argparse.ArgumentParser(description='Generates FM synthesis audio, parameter, and spectrograph files for use as training data')
parser.add_argument('--n_examples', type=int, default=100000)
parser.add_argument('--save_dir', type=str, default='./data/synthetics')
parser.add_argument('--Fs', type=int, default = 16000)
parser.add_argument('--duration', type=float, default= 1.0)
parser.add_argument('--overwrite', type=bool, default=True)
parser.add_argument('--seed', type = int, default = 217)

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

def generate_dataset(args):
    # set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    # create save dir
    os.makedirs(args.save_dir, exist_ok=True)
    # create transform
    Fs = args.Fs
    hop_size = 512
    bins_per_octave = 32
    n_octaves=7
    n_bins = bins_per_octave*n_octaves
    num_digits = int(np.log10(args.n_examples)+1) # number of digits in num examples

    cqt_transform = CQT2010v2(sr = Fs,
                              hop_length = hop_size,
                              n_bins = n_bins,
                              bins_per_octave = bins_per_octave)
    # create manifest data
    manifest_data = []
    manifest_path = os.path.join(args.save_dir,'manifest.json')
    if os.path.exists(manifest_path) and not args.overwrite:
        print("Dataset already exists. Use --overwrite to regen")
        exit()
    # loop over n_examples
    print_interval = max(1, args.n_examples // 10)
    for i in range(args.n_examples):
        # create parameters for FM synthesis
        if i % print_interval == 0 or i+1 == args.n_examples:
            print(f"Generating example {i:0{num_digits}d}/{args.n_examples:0{num_digits}d}")
        parameters = create_parameters()
        mod_matrix = fm_ddsp.make_mod_matrix(parameters['mod_values'])
        # render audio
        with torch.no_grad():
            audio = fm_ddsp.fm_renderer(
                parameters['f0'], 
                parameters['ratios'], 
                parameters['levels'], 
                mod_matrix, 
                parameters['carrier_weights'], 
                args.Fs, args.duration)
        # compute spectrogram
        cqt_spec = compute_spectrogram_cqt(audio, cqt_transform)
        # save spec_{}.pt
        spec_file = f'spec_{i:0{num_digits}d}.pt'
        spec_file_path = os.path.join(args.save_dir,spec_file)
        torch.save(cqt_spec, spec_file_path)
        # save params_{}.json
        params_file = f'params_{i:0{num_digits}d}.json'
        params_file_path = os.path.join(args.save_dir,params_file)
        parameters_dict = {
            'f0': parameters['f0'],
            'algorithm': parameters['algorithm'],
            'mod_values': parameters['mod_values'].tolist(),
            'ratios': parameters['ratios'].tolist(),
            'levels': parameters['levels'].tolist(),
            'carrier_weights': parameters['carrier_weights'].tolist()
        }
        with open(params_file_path, 'w') as f:
            json.dump(parameters_dict, f, indent=2)
        manifest_data.append({'index':i,
                              'parameter_file':params_file,
                              'spectrogram_file':spec_file,
                              'algorithm': parameters['algorithm']
                             })    
    with open(manifest_path, 'w') as f:
        manifest = {
            'n_examples': args.n_examples,
            'Fs': args.Fs,
            'duration': args.duration,
            'seed': args.seed,
            'n_bins': n_bins,
            'bins_per_ocatave':bins_per_octave
        }
        manifest['examples'] = manifest_data
        json.dump(manifest, f, indent=2)
            

def create_parameters():
    ratio_choices = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        # for i in range(self.n_examples):
        #     if i%1000 == 0 or i==self.n_examples-1:
        #         print(f"Generating example {i+1}/{self.n_examples}")
    midi_note = random.randint(36, 72)
    f0 = 440.0 * 2 ** ((midi_note - 69)/12.0)
    alg = random.choice(ALGORITHMS)
    mod_values = torch.rand(7) * torch.tensor(alg['mod_mask'], dtype=torch.float32)
    carrier_weights = torch.rand(4) * torch.tensor(alg['carrier_mask'], dtype=torch.float32)
    ratios = torch.tensor([random.choice(ratio_choices) for _ in range(4)])
    levels = torch.rand(4) * 0.9 + 0.1
    return {
        'f0':f0,
        'algorithm':alg['name'],
        'mod_values':mod_values,
        'ratios':ratios,
        'levels':levels,
        'carrier_weights':carrier_weights
    }


if __name__ == '__main__':
    args = parser.parse_args()
    generate_dataset(args)
    

