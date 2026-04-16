import sys
import os
import json
import torch
sys.path.append('..')

class FMDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 save_dir,
                manifest_file_name = "manifest.json"):
        super().__init__()
        self.save_dir = save_dir
        self.manifest_path = os.path.join(self.save_dir, manifest_file_name)
        with open(self.manifest_path, 'r') as f:
            self.manifest = json.load(f)
            self.n_examples = self.manifest['n_examples']        

    def __len__ (self):
        return self.n_examples

    def __getitem__(self, idx):
        entry = self.manifest['examples'][idx]
        param_file = os.path.join(self.save_dir,entry['parameter_file'])
        spec_file = os.path.join(self.save_dir,entry['spectrogram_file'])
        with open(param_file, 'r') as f:
            parameters = json.load(f)
            params = {
                'f0': parameters['f0'],
                'ratios': torch.tensor(parameters['ratios']),
                'levels': torch.tensor(parameters['levels']),

            }
        # load .pt spec file
        spec = torch.load(spec_file, weights_only = False).detach().clone()
        return params, spec
    