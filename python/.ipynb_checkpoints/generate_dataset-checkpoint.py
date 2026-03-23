import argparse
import json
import torch
import torchaudio
import numpy as np
import sys
import os
import random

parser = argparse.ArgumentParser(description='Generates FM synthesis audio, parameter, and spectrograph files for use as training data')
parser.add_argument('n_examples', type=int, default=100000)
parser.add_argument('--save_dir', type=str, default='./data/synthetics')
parser.add_argument('--Fs', type=int, default = 16000)
parser.add_argument('--duration', type=float, default= 1.0)
parser.add_argument('--seed', type = int, default = 217)



def generate_dataset(n_examples, sav_dir, Fs, duration, seed)
    # create save dir
    # loop over n_examples
        # create parameters for FM synthesis
        # render audio
        # compute spectrogram
        # save spec_{}.pt
        # save params_{}.json
    # write manifest

    pass

if __name__ == '__main__':
    args = parser.parse_args()
    generate_dataset(args)
    

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