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

def cqt_spectrogram_loss_enhanced(predicted_spec, target_spec, verbose = False):
    # bias spectrograms
    pred_spec = (predicted_spec - predicted_spec.min()) + 1e-8
    targ_spec = (target_spec - target_spec.min()) + 1e-8
    # normalize spectrogram
    pred_norm = pred_spec / (pred_spec.max() + 1e-8)
    targ_norm = targ_spec / (targ_spec.max() + 1e-8)

    # bi-directional weighted loss - emphasise spectral convergence of harmonic peaks
    targ_weights = targ_norm / (targ_norm.sum() + 1e-8)
    targ_loss = (targ_weights * (torch.log1p(pred_norm) - torch.log1p(targ_norm)).abs()).sum()
    pred_weights = pred_norm / (pred_norm.sum() + 1e-8)
    pred_loss = (pred_weights * (torch.log1p(pred_norm) - torch.log1p(targ_norm)).abs()).sum()
    

    total_loss = targ_loss + pred_loss
    return total_loss

def cqt_spectrogram_loss_enhanced2(predicted_spec, target_spec,  freq_weights = None, verbose = False):
    
    # bias spectrograms
    pred_spec = (predicted_spec - predicted_spec.min()) + 1e-8
    targ_spec = (target_spec - target_spec.min()) + 1e-8
    # normalize spectrogram
    pred_norm = pred_spec / (pred_spec.max() + 1e-8)
    targ_norm = targ_spec / (targ_spec.max() + 1e-8)

    # bi-directional weighted loss - emphasise spectral convergence of harmonic peaks
    targ_weights = targ_norm / (targ_norm.sum() + 1e-8)
    targ_loss = (targ_weights * (torch.log1p(pred_norm) - torch.log1p(targ_norm)).abs()).sum()
    pred_weights = pred_norm / (pred_norm.sum() + 1e-8)
    pred_loss = (pred_weights * (torch.log1p(pred_norm) - torch.log1p(targ_norm)).abs()).sum()
    

    total_loss = targ_loss + pred_loss
    return total_loss