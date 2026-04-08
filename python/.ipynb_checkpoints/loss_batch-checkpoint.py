import torch
import numpy as np
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# loss_batch.py
# Batched versions of loss.py functions.
# All functions accept a leading batch dimension.
#
# Shape conventions:
#   audio:     [batch, n_samples]
#   spec:      [batch, n_bins]
#   freq_weights: [batch, n_bins] or None
# ─────────────────────────────────────────────────────────────────────────────


def compute_spectrogram_cqt_batched(audio, cqt_transform):
    """
    Computes CQT spectrogram for a batch of audio signals.

    Args:
        audio:         [batch, n_samples]
        cqt_transform: CQT2010v2 instance (already on correct device)

    Returns:
        spec: [batch, n_bins] — log magnitude, time-averaged
    """
    # CQT2010v2 expects [batch, n_samples] — audio is already that shape
    spec = cqt_transform(audio)             # [batch, n_bins, time_frames]
    spec = spec.abs()
    spec = torch.clamp(spec, min=0.0)
    spec = torch.log1p(spec)
    spec = spec.mean(dim=2)                 # average across time -> [batch, n_bins]
    return spec


def make_fundamental_weight_batched(n_bins, fundamental_bins, bins_per_octave,
                                     suppression=0.1, width=2.0, device='cpu'):
    """
    Computes a Gaussian frequency weight tensor for a batch of fundamental bins.
    Suppresses the contribution of the fundamental frequency in the loss.

    Args:
        n_bins:           int — number of CQT bins
        fundamental_bins: [batch] int tensor — bin index of each example's fundamental
        bins_per_octave:  int
        suppression:      float — minimum weight at the fundamental bin (0=remove, 1=no effect)
        width:            float — width of suppression window in semitones

    Returns:
        freq_weights: [batch, n_bins]
    """
    batch_size = fundamental_bins.shape[0]
    bins = torch.arange(n_bins, dtype=torch.float32, device=device)  # [n_bins]
    bins = bins.unsqueeze(0).expand(batch_size, -1)                   # [batch, n_bins]
    centers = fundamental_bins.float().unsqueeze(1)                    # [batch, 1]

    sigma = width * bins_per_octave / 12.0
    gaussian = torch.exp(-0.5 * ((bins - centers) / sigma) ** 2)     # [batch, n_bins]

    freq_weights = 1.0 - (1.0 - suppression) * gaussian              # [batch, n_bins]
    return freq_weights


def cqt_spectrogram_loss_batched(pred_spec, target_spec, freq_weights=None, verbose=False, threshold = 0.05):
    """
    Bidirectional weighted CQT spectrogram loss operating on batches.

    - target_weighted: rewards getting target harmonic peaks right
    - pred_weighted:   punishes spurious energy in wrong places
    - freq_weights:    optional [batch, n_bins] to suppress fundamental contribution

    Args:
        pred_spec:    [batch, n_bins]
        target_spec:  [batch, n_bins]
        freq_weights: [batch, n_bins] or None
        verbose:      bool — print spectral convergence if True

    Returns:
        loss: scalar tensor
    """
    # normalise each example to the target
    ref = target_spec.max(dim = 1, keepdim = True).values + 1e-8
    pred_norm = pred_spec / ref
    targ_norm = target_spec / ref

    # apply frequency weights if provided
    if freq_weights is not None:
        pred_norm = pred_norm * freq_weights
        targ_norm = targ_norm * freq_weights
        
    # per-bin log magnitude error [batch, n_bins]
    log_error = (torch.log1p(pred_norm) - torch.log1p(targ_norm)).abs()

    # target-weighted loss — emphasise target harmonic peaks and erase non-peaks via thresholding
    targ_mask = (targ_norm > threshold).float()
    targ_weights = targ_mask*(targ_norm / (targ_norm.sum(dim=1, keepdim=True) + 1e-8))
    targ_loss = (targ_weights * log_error).sum(dim=1).mean()

    # pred-weighted loss — punish spurious energy
    pred_mask = (pred_norm > threshold).float()
    pred_weights = pred_mask*(pred_norm / (pred_norm.sum(dim=1, keepdim=True) + 1e-8))
    pred_loss = (pred_weights * log_error).sum(dim=1, keepdim=False).mean()

    total_loss = targ_loss + pred_loss

    if verbose:
        with torch.no_grad():
            sc = torch.norm(targ_norm - pred_norm, dim=1) / (torch.norm(targ_norm, dim=1) + 1e-8)
            print(f"Spectral Convergence (mean): {sc.mean().item():.4f}")

    return total_loss