"""
evaluate_chain.py
Evaluates a trained FMEncoderChain checkpoint against a dataset.
Organises results by spectral loss and parameter error.
"""

import torch
import numpy as np
import json
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from nnAudio.features import CQT2010v2

import sys
from FMEncoderChain import FMEncoderChain
from fm_chain import fm_renderer
from loss_batch import compute_spectrogram_cqt_batched, cqt_spectrogram_loss_batched
from dataset import FMDataset

# ─────────────────────────────────────────────────────────────────────────────

def evaluate(data_dir, checkpoint_path, n_samples=200, Fs=16000, duration=1.0,
             f0=110.0, device_str='cpu'):

    device = torch.device(device_str)
    
    # load dataset
    dataset = FMDataset(save_dir=data_dir)
    n_samples = min(n_samples, len(dataset))
    print(f"Evaluating {n_samples} examples from {data_dir}")

    # load encoder
    encoder = FMEncoderChain(n_bins=224).to(device)
    encoder.load_state_dict(torch.load(checkpoint_path, map_location=device))
    encoder.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")

    # CQT transform
    cqt_transform = CQT2010v2(
        sr=Fs, hop_length=512, n_bins=224, bins_per_octave=32
    ).to(device)

    results = []

    with torch.no_grad():
        for idx in range(n_samples):
            params, spec = dataset[idx]
            spec = spec.float().unsqueeze(0).to(device)

            # encoder forward pass
            predicted = encoder(spec)

            # render predicted audio
            f0_t = torch.tensor([f0], dtype=torch.float32).to(device)
            audio_pred = fm_renderer(
                f0_t,
                predicted['ratios'],
                predicted['levels'],
                Fs, duration
            )

            # compute predicted spectrogram
            pred_spec = compute_spectrogram_cqt_batched(audio_pred, cqt_transform)

            # spectral loss
            spectral_loss = cqt_spectrogram_loss_batched(pred_spec, spec).item()

            # ground truth params
            gt_ratios = params['ratios'].float()
            gt_levels = params['levels'].float()
            pred_ratios = predicted['ratios'][0].cpu()
            pred_levels = predicted['levels'][0].cpu()

            # parameter errors
            ratio_error = (pred_ratios - gt_ratios).abs().mean().item()
            level_error = (pred_levels - gt_levels).abs().mean().item()
            param_error = ratio_error + level_error

            results.append({
                'idx': idx,
                'spectral_loss': spectral_loss,
                'param_error': param_error,
                'ratio_error': ratio_error,
                'level_error': level_error,
                'gt_r0': gt_ratios[0].item(),
                'gt_r1': gt_ratios[1].item(),
                'gt_l0': gt_levels[0].item(),
                'gt_l1': gt_levels[1].item(),
                'pred_r0': pred_ratios[0].item(),
                'pred_r1': pred_ratios[1].item(),
                'pred_l0': pred_levels[0].item(),
                'pred_l1': pred_levels[1].item(),
                'spec': spec[0].cpu().numpy(),
                'pred_spec': pred_spec[0].cpu().numpy(),
            })

    # sort by spectral loss
    results_by_spectral = sorted(results, key=lambda x: x['spectral_loss'])
    # sort by parameter error
    results_by_param = sorted(results, key=lambda x: x['param_error'])

    return results, results_by_spectral, results_by_param


def plot_summary(results, results_by_spectral, results_by_param, n_show=5):
    """Plot overview of results and best/worst examples."""

    spectral_losses = [r['spectral_loss'] for r in results]
    param_errors = [r['param_error'] for r in results]
    ratio_errors = [r['ratio_error'] for r in results]
    level_errors = [r['level_error'] for r in results]

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('FMEncoderChain Evaluation', fontsize=14, fontweight='bold')

    # ── row 1: distributions ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(3, 4, 1)
    ax1.hist(spectral_losses, bins=40, color='#378ADD', alpha=0.8)
    ax1.set_title('spectral loss distribution')
    ax1.set_xlabel('spectral loss')
    ax1.axvline(np.median(spectral_losses), color='red', linestyle='--',
                label=f'median={np.median(spectral_losses):.4f}')
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(3, 4, 2)
    ax2.hist(param_errors, bins=40, color='#D85A30', alpha=0.8)
    ax2.set_title('parameter error distribution')
    ax2.set_xlabel('mean abs error')
    ax2.axvline(np.median(param_errors), color='red', linestyle='--',
                label=f'median={np.median(param_errors):.4f}')
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(3, 4, 3)
    ax3.scatter(
        [r['gt_r0'] for r in results],
        [r['pred_r0'] for r in results],
        alpha=0.3, s=10, color='#534AB7', label='op0'
    )
    ax3.scatter(
        [r['gt_r1'] for r in results],
        [r['pred_r1'] for r in results],
        alpha=0.3, s=10, color='#1D9E75', label='op1'
    )
    lims = [1, 8]
    ax3.plot(lims, lims, 'k--', alpha=0.5, linewidth=0.8)
    ax3.set_xlabel('GT ratio'); ax3.set_ylabel('pred ratio')
    ax3.set_title('ratio prediction (GT vs pred)')
    ax3.legend(fontsize=8)

    ax4 = fig.add_subplot(3, 4, 4)
    ax4.scatter(
        [r['gt_l0'] for r in results],
        [r['pred_l0'] for r in results],
        alpha=0.3, s=10, color='#534AB7', label='op0'
    )
    ax4.scatter(
        [r['gt_l1'] for r in results],
        [r['pred_l1'] for r in results],
        alpha=0.3, s=10, color='#1D9E75', label='op1'
    )
    lims = [0, 1]
    ax4.plot(lims, lims, 'k--', alpha=0.5, linewidth=0.8)
    ax4.set_xlabel('GT level'); ax4.set_ylabel('pred level')
    ax4.set_title('level prediction (GT vs pred)')
    ax4.legend(fontsize=8)

    # ── row 2: best spectral examples ────────────────────────────────────────
    for i, r in enumerate(results_by_spectral[:n_show]):
        ax = fig.add_subplot(3, n_show, n_show + i + 1)
        ax.plot(r['spec'], color='#378ADD', linewidth=0.8, label='GT')
        ax.plot(r['pred_spec'], color='#D85A30', linewidth=0.8,
                linestyle='--', label='pred')
        ax.set_title(
            f'best #{i+1} spec\n'
            f'sl={r["spectral_loss"]:.4f}\n'
            f'r=[{r["gt_r0"]:.0f},{r["gt_r1"]:.0f}]→[{r["pred_r0"]:.1f},{r["pred_r1"]:.1f}]\n'
            f'l=[{r["gt_l0"]:.2f},{r["gt_l1"]:.2f}]→[{r["pred_l0"]:.2f},{r["pred_l1"]:.2f}]',
            fontsize=7
        )
        if i == 0:
            ax.legend(fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])

    # ── row 3: worst spectral examples ───────────────────────────────────────
    for i, r in enumerate(results_by_spectral[-n_show:]):
        ax = fig.add_subplot(3, n_show, 2*n_show + i + 1)
        ax.plot(r['spec'], color='#378ADD', linewidth=0.8, label='GT')
        ax.plot(r['pred_spec'], color='#D85A30', linewidth=0.8,
                linestyle='--', label='pred')
        ax.set_title(
            f'worst #{i+1} spec\n'
            f'sl={r["spectral_loss"]:.4f}\n'
            f'r=[{r["gt_r0"]:.0f},{r["gt_r1"]:.0f}]→[{r["pred_r0"]:.1f},{r["pred_r1"]:.1f}]\n'
            f'l=[{r["gt_l0"]:.2f},{r["gt_l1"]:.2f}]→[{r["pred_l0"]:.2f},{r["pred_l1"]:.2f}]',
            fontsize=7
        )
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.show()


def print_summary_table(results_by_spectral, results_by_param, n_show=10):
    """Print sorted tables of best and worst examples."""

    print("\n" + "="*80)
    print(f"{'BEST by spectral loss':^80}")
    print("="*80)
    print(f"{'idx':>5} {'spec_loss':>10} {'param_err':>10} "
          f"{'GT r0':>6} {'PR r0':>6} {'GT r1':>6} {'PR r1':>6} "
          f"{'GT l0':>6} {'PR l0':>6} {'GT l1':>6} {'PR l1':>6}")
    print("-"*80)
    for r in results_by_spectral[:n_show]:
        print(f"{r['idx']:>5} {r['spectral_loss']:>10.4f} {r['param_error']:>10.4f} "
              f"{r['gt_r0']:>6.1f} {r['pred_r0']:>6.2f} "
              f"{r['gt_r1']:>6.1f} {r['pred_r1']:>6.2f} "
              f"{r['gt_l0']:>6.2f} {r['pred_l0']:>6.2f} "
              f"{r['gt_l1']:>6.2f} {r['pred_l1']:>6.2f}")

    print("\n" + "="*80)
    print(f"{'WORST by spectral loss':^80}")
    print("="*80)
    print(f"{'idx':>5} {'spec_loss':>10} {'param_err':>10} "
          f"{'GT r0':>6} {'PR r0':>6} {'GT r1':>6} {'PR r1':>6} "
          f"{'GT l0':>6} {'PR l0':>6} {'GT l1':>6} {'PR l1':>6}")
    print("-"*80)
    for r in results_by_spectral[-n_show:]:
        print(f"{r['idx']:>5} {r['spectral_loss']:>10.4f} {r['param_error']:>10.4f} "
              f"{r['gt_r0']:>6.1f} {r['pred_r0']:>6.2f} "
              f"{r['gt_r1']:>6.1f} {r['pred_r1']:>6.2f} "
              f"{r['gt_l0']:>6.2f} {r['pred_l0']:>6.2f} "
              f"{r['gt_l1']:>6.2f} {r['pred_l1']:>6.2f}")

    print("\n" + "="*80)
    print(f"{'WORST by parameter error':^80}")
    print("="*80)
    print(f"{'idx':>5} {'spec_loss':>10} {'param_err':>10} "
          f"{'GT r0':>6} {'PR r0':>6} {'GT r1':>6} {'PR r1':>6} "
          f"{'GT l0':>6} {'PR l0':>6} {'GT l1':>6} {'PR l1':>6}")
    print("-"*80)
    for r in results_by_param[-n_show:]:
        print(f"{r['idx']:>5} {r['spectral_loss']:>10.4f} {r['param_error']:>10.4f} "
              f"{r['gt_r0']:>6.1f} {r['pred_r0']:>6.2f} "
              f"{r['gt_r1']:>6.1f} {r['pred_r1']:>6.2f} "
              f"{r['gt_l0']:>6.2f} {r['pred_l0']:>6.2f} "
              f"{r['gt_l1']:>6.2f} {r['pred_l1']:>6.2f}")

    # aggregate stats
    print("\n" + "="*80)
    print(f"{'AGGREGATE STATISTICS':^80}")
    print("="*80)
    print(f"{'metric':<30} {'mean':>10} {'median':>10} {'std':>10}")
    print("-"*80)
    for key, label in [
        ('spectral_loss', 'spectral loss'),
        ('param_error',   'parameter error'),
        ('ratio_error',   'ratio error'),
        ('level_error',   'level error'),
    ]:
        vals = [r[key] for r in results_by_spectral]
        print(f"{label:<30} {np.mean(vals):>10.4f} {np.median(vals):>10.4f} {np.std(vals):>10.4f}")

    # per-parameter ratio accuracy
    print("\nratio prediction accuracy (within 0.5 of GT):")
    r0_acc = np.mean([abs(r['pred_r0'] - r['gt_r0']) < 0.5 for r in results_by_spectral])
    r1_acc = np.mean([abs(r['pred_r1'] - r['gt_r1']) < 0.5 for r in results_by_spectral])
    print(f"  op0 ratio: {r0_acc*100:.1f}%")
    print(f"  op1 ratio: {r1_acc*100:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--n_samples', type=int, default=200)
    parser.add_argument('--f0', type=float, default=110.0)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    results, by_spectral, by_param = evaluate(
        args.data_dir, args.checkpoint,
        n_samples=args.n_samples, f0=args.f0, device_str=args.device
    )
    print_summary_table(by_spectral, by_param)
    plot_summary(results, by_spectral, by_param)