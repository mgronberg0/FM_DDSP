import torch
import torchaudio
import numpy as np

def sin_wav(freq, Fs, duration):
    t = torch.linspace(0,duration,int(Fs*duration))
    return torch.sin(t*2*np.pi*freq)

def make_phase(freq, Fs, duration):
    # returns the phase ramp tensor before sin is applied
    t = torch.linspace(0, duration, int(Fs*duration))
    return t * (2*np.pi) * freq
    
def apply_phase_mod(phase, modulation):
    # adds modulation to phase and applies sin, returns audio signal
    modulated_phase = phase+modulation
    return torch.sin(modulated_phase)

def operator(freq, Fs, duration, level, modulation=None):
    # returns operator output tensor [n_samples]
    self_phase = make_phase(freq, Fs, duration)
    if modulation is None:
        modulation = torch.zeros_like(self_phase)
    return apply_phase_mod(self_phase,modulation) * level

def make_mod_matrix(values):
    mod_matrix = torch.zeros(4,4, device = values.device)
    mod_matrix[0][0] = values[0]
    mod_matrix[1][0] = values[1]
    mod_matrix[2][0] = values[2]
    mod_matrix[2][1] = values[3]
    mod_matrix[3][0] = values[4]
    mod_matrix[3][1] = values[5]
    mod_matrix[3][2] = values[6]
    return mod_matrix

def fm_renderer(f0, ratios, levels, mod_matrix, carrier_weights, Fs, duration):
    num_samples = int(Fs * duration)
    num_ops = 4
    # # apply mask to zero cyclic modulation
    # mask = torch.zeros(num_ops, num_ops)
    # mask[1][0] = 1.0
    # mask[2][0] = 1.0
    # mask[2][1] = 1.0
    # mask[3][0] = 1.0
    # mask[3][1] = 1.0
    # mask[3][2] = 1.0
    # mask[0][0] = 1.0 # op0 feedback only
    # mod_matrix = mod_matrix * mask
    t = torch.linspace(0, duration, num_samples, device=ratios.device)
    freqs = f0 * ratios
    phase = torch.remainder(2 * torch.pi * freqs.unsqueeze(1) * t.unsqueeze(0),
                            2* torch.pi)
    # compute raw operator signals
    raw = torch.sin(phase) * levels.unsqueeze(1)  # [4, num_samples]
    # compute feedback
    feedback = torch.roll(raw, 1, dims=1) * torch.diag(mod_matrix).unsqueeze(1)
    feedback[:, 0] = 0.0
    # compute modulated phase via matmul
    phase_mod = torch.matmul(mod_matrix, raw)
    # fix diagonal — use feedback instead of raw for self-modulation
    diag = torch.diag(mod_matrix)
    phase_mod -= diag.unsqueeze(1) * raw
    phase_mod += diag.unsqueeze(1) * feedback
    # recompute operator outputs with modulation
    op_out = torch.sin(phase + phase_mod) * levels.unsqueeze(1)
    # normalize carrier weights and compute audio
    carrier_weights = carrier_weights / (carrier_weights.sum() + 1e-8)
    audio_out = (op_out * carrier_weights.unsqueeze(1)).sum(dim=0)
    return audio_out