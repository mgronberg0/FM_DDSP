import torch
import numpy as np

#----------------------------------------
# fm_chain.py
# A simplified fm engine that uses one algorithm: op0->op1->op2
# No mod matrix, no carrier weights. 
# Parameters:
# f0, ratios, levels

def fm_renderer(f0, ratios, levels, Fs, duration):
    # f0: [batch]
    # ratios; [batch, 2] - only op0 and 0p1 get ratio controls, hold carrier constant
    # levels: [batch, 3]
    # returns: [batch, n_samples]
    # Fs:              int — sample rate
    # duration:        float — duration in seconds
    batch_size = f0.shape[0]
    num_samples = int(Fs * duration)

    # time axis
    t = torch.linspace(0, duration, num_samples, device=ratios.device).unsqueeze(0)
    # 0:1 indexing preserves dim:
    # ratios[:, 0] shape = [batch]
    # ratios[:, 0:1] shape = [batch, 1]
    f0_op0 = f0.unsqueeze(1) * ratios[:, 0:1] # [batch, 1]
    f0_op1 = f0.unsqueeze(1) * ratios[:, 1:2] 

    phase_op0 = 2 * torch.pi * f0_op0 * t # shape [batch, n_samples]
    op0_out = torch.sin(phase_op0) * levels[:, 0:1]

    phase_op1 = 2 * torch.pi * f0_op1 * t + op0_out
    op1_out = torch.sin(phase_op1) * levels[:, 1:2]

    # carrier
    phase_op2 = 2 * torch.pi * f0.unsqueeze(1) * t + op1_out
    audio = torch.sin(phase_op2)

    return audio # shape [batch, n_samples]

    

    