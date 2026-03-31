import torch
import torchaudio
import numpy as np
import torch.nn.functional as F

def multiscale_stft_loss(predicted, target, fft_sizes=[2048]):
    total_loss = 0.0
    for size in fft_sizes:
        pred_mag = torch.stft(predicted, 
                               n_fft=size, 
                               hop_length = int(size/4), 
                               window = torch.hann_window(size),
                               return_complex=True).abs()
        targ_mag = torch.stft(target, 
                               n_fft=size, 
                               hop_length = int(size/4), 
                               window = torch.hann_window(size),
                               return_complex=True).abs()
        l1_loss_log = torch.nn.functional.l1_loss(torch.log1p(pred_mag), 
                                                  torch.log1p(targ_mag))
        l1_loss_lin = torch.nn.functional.l1_loss(pred_mag, 
                                                  targ_mag)
        total_loss += l1_loss_log + l1_loss_lin
    return total_loss / len(fft_sizes)

def cqt_spectrogram_loss(pred_spec, target_spec):
    l1_loss_log = torch.nn.functional.l1_loss(torch.log1p(pred_spec),
                                              torch.log1p(target_spec))
    l1_loss_lin = torch.nn.functional.l1_loss(pred_spec, 
                                               target_spec)

    total_loss = l1_loss_log + l1_loss_lin
    return total_loss