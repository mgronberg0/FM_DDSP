import torch
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# fm_ddsp_batch.py
# Batched versions of fm_ddsp.py functions.
# All functions accept an extra leading batch dimension.
#
# Shape conventions:
#   f0:              [batch]
#   ratios:          [batch, 4]
#   levels:          [batch, 4]
#   mod_matrix:      [batch, 4, 4]
#   carrier_weights: [batch, 4]
#   returns audio:   [batch, n_samples]
#
# Use make_mod_matrix_batched to construct mod_matrix from the encoder's
# mod_values output of shape [batch, 7].
# ─────────────────────────────────────────────────────────────────────────────

def make_mod_matrix_batched(values):
    """
    Constructs a batched 4x4 modulation matrix from 7 learnable values.

    Args:
        values: [batch, 7] tensor

    Returns:
        mod_matrix: [batch, 4, 4] tensor

    Matrix layout (mod_matrix[target][source]):
        [v0,  0,   0,   0 ]   <- op0 feedback
        [v1,  0,   0,   0 ]
        [v2,  v3,  0,   0 ]
        [v4,  v5,  v6,  0 ]
    """
    batch_size = values.shape[0]
    mod_matrix = torch.zeros(batch_size, 4, 4, device=values.device)
    mod_matrix[:, 0, 0] = values[:, 0]
    mod_matrix[:, 1, 0] = values[:, 1]
    mod_matrix[:, 2, 0] = values[:, 2]
    mod_matrix[:, 2, 1] = values[:, 3]
    mod_matrix[:, 3, 0] = values[:, 4]
    mod_matrix[:, 3, 1] = values[:, 5]
    mod_matrix[:, 3, 2] = values[:, 6]
    return mod_matrix


def fm_renderer_batched(f0, ratios, levels, mod_matrix, carrier_weights, Fs, duration):
    """
    Batched differentiable 4-operator FM renderer.

    Args:
        f0:              [batch] — fundamental frequency in Hz
        ratios:          [batch, 4] — operator frequency ratios
        levels:          [batch, 4] — operator output levels [0, 1]
        mod_matrix:      [batch, 4, 4] — modulation routing matrix
        carrier_weights: [batch, 4] — carrier blend weights
        Fs:              int — sample rate
        duration:        float — duration in seconds

    Returns:
        audio: [batch, n_samples]
    """
    batch_size = f0.shape[0]
    num_samples = int(Fs * duration)

    # time axis — shared across batch [n_samples]
    t = torch.linspace(0, duration, num_samples, device=ratios.device)

    # operator frequencies [batch, 4]
    freqs = f0.unsqueeze(1) * ratios

    # phase ramps [batch, 4, n_samples]
    # freqs.unsqueeze(2) = [batch, 4, 1]
    # t.unsqueeze(0).unsqueeze(0) = [1, 1, n_samples]
    phase = torch.remainder(
        2 * torch.pi * freqs.unsqueeze(2) * t.unsqueeze(0).unsqueeze(0),
        2 * torch.pi
    )

    # raw operator signals (no modulation) [batch, 4, n_samples]
    # levels.unsqueeze(2) = [batch, 4, 1]
    raw = torch.sin(phase) * levels.unsqueeze(2)

    # feedback — one sample delay on diagonal (op self-modulation)
    # torch.diagonal returns [batch, 4] for a [batch, 4, 4] tensor
    diag = torch.diagonal(mod_matrix, dim1=1, dim2=2)  # [batch, 4]

    # roll along sample dimension (dim=2)
    feedback = torch.roll(raw, 1, dims=2) * diag.unsqueeze(2)  # [batch, 4, n_samples]
    feedback[:, :, 0] = 0.0  # zero out wrap-around sample

    # modulated phase via batch matrix multiply
    # torch.bmm: [batch, 4, 4] @ [batch, 4, n_samples] = [batch, 4, n_samples]
    phase_mod = torch.bmm(mod_matrix, raw)

    # replace diagonal contribution with feedback
    # diag.unsqueeze(2) = [batch, 4, 1] broadcasts across n_samples
    phase_mod -= diag.unsqueeze(2) * raw
    phase_mod += diag.unsqueeze(2) * feedback

    # recompute operator outputs with modulation [batch, 4, n_samples]
    op_out = torch.sin(phase + phase_mod) * levels.unsqueeze(2)

    # normalise carrier weights and compute audio output
    # carrier_weights: [batch, 4] -> normalise across operator dim
    carrier_weights = carrier_weights / (carrier_weights.sum(dim=1, keepdim=True) + 1e-8)

    # weighted sum over operator dimension
    # carrier_weights.unsqueeze(2) = [batch, 4, 1]
    # sum over dim=1 (operators) -> [batch, n_samples]
    audio_out = (op_out * carrier_weights.unsqueeze(2)).sum(dim=1)

    return audio_out