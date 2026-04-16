from dataset_chain import FMDataset
from FMEncoderChain import FMEncoderChain
from fm_chain import fm_renderer
from loss_batch import (compute_spectrogram_cqt_batched, 
                        cqt_spectrogram_loss_batched,
                        make_fundamental_weight_batched)

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from nnAudio.features import CQT2010v2
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Trains existing dataset from generate_dataset')
parser.add_argument('--data_dir', type=str)
parser.add_argument('--Fs', type=int, default=16000)
parser.add_argument('--duration', type=float, default=1.0)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--n_epochs', type=int, default=25)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--f0', type = float, default = 440)
parser.add_argument('--output_subdir', type = str, default = "output")

def stage1_loss(pred_params, gt_params, 
                pred_spec, gt_spec, 
                device,
                lambda_direct):
    # spectral loss
    spectral_loss = cqt_spectrogram_loss_batched(pred_spec, gt_spec)
    # direct supervision of parameters loss with per-example adaptive weighting based on spectral loss performance
    ref = gt_spec.max(dim=1, keepdim=True).values + 1e-8
    pred_norm = pred_spec / ref
    targ_norm = gt_spec / ref
    sc = torch.norm(targ_norm - pred_norm, dim=1) / (torch.norm(targ_norm, dim=1) + 1e-8)
    adaptive_lambda = lambda_direct * sc.detach()

    gt_levels = gt_params['levels'].float().to(device)
    gt_ratios = gt_params['ratios'].float().to(device)

    # store masks for each operator biased towards 1 that quickly snap to 0
    op0_activity = torch.sigmoid((gt_levels[:,0].detach() - 0.05) * 20)
    op1_activity = torch.sigmoid((gt_levels[:,1].detach() - 0.05) * 20)
    
    level_supervision = (pred_params['levels'] - gt_levels).abs().mean(dim=1)

    ratio_sup = (pred_params['ratios'] - gt_ratios).abs()  # [batch, 2]
    ratio_sup[:, 0] = ratio_sup[:, 0] * op0_activity
    ratio_sup[:, 1] = ratio_sup[:, 1] * op1_activity
    ratio_supervision = ratio_sup.mean(dim=1)  # [batch]

    supervision_loss = (adaptive_lambda * (level_supervision + ratio_supervision)).mean()

    supervision_loss = (adaptive_lambda * (level_supervision + ratio_supervision)).mean()
    # try to enforce sparceness in contribution from operators 1 and 2
    #sparsity_loss = pred_params['levels'][:, :2].sum(dim=1).mean() * lambda_sparse
    loss = spectral_loss + supervision_loss #+ sparsity_loss
    return loss, spectral_loss, supervision_loss, adaptive_lambda.mean()

def stage3_loss(pred_params, gt_params, 
                pred_spec, gt_spec, 
                device,
                lambda_direct):
    # spectral loss
    spectral_loss = cqt_spectrogram_loss_batched(pred_spec, gt_spec)
    # direct supervision of parameters loss with per-example adaptive weighting based on spectral loss performance
    ref = gt_spec.max(dim=1, keepdim=True).values + 1e-8
    pred_norm = pred_spec / ref
    targ_norm = gt_spec / ref
    sc = torch.norm(targ_norm - pred_norm, dim=1) / (torch.norm(targ_norm, dim=1) + 1e-8)
    adaptive_lambda = lambda_direct * sc.detach()

    gt_levels = gt_params['levels'].float().to(device)
    gt_ratios = gt_params['ratios'].float().to(device)

    # store masks for each operator biased towards 1 that quickly snap to 0
    op0_activity = 1 #torch.sigmoid((gt_levels[:,0].detach() - 0.05) * 20)
    op1_activity = 1 #torch.sigmoid((gt_levels[:,1].detach() - 0.05) * 20)
    
    level_supervision = (pred_params['levels'] - gt_levels).abs().mean(dim=1)

    ratio_sup = (pred_params['ratios'] - gt_ratios).abs()  # [batch, 2]
    ratio_sup[:, 0] = ratio_sup[:, 0] * op0_activity
    ratio_sup[:, 1] = ratio_sup[:, 1] * op1_activity
    ratio_supervision = ratio_sup.mean(dim=1)  # [batch]

    supervision_loss = (adaptive_lambda * (level_supervision + ratio_supervision)).mean()

    supervision_loss = (adaptive_lambda * (level_supervision + ratio_supervision)).mean()
    # try to enforce sparceness in contribution from operators 1 and 2
    #sparsity_loss = pred_params['levels'][:, :2].sum(dim=1).mean() * lambda_sparse
    loss = spectral_loss + supervision_loss #+ sparsity_loss
    return loss, spectral_loss, supervision_loss, adaptive_lambda.mean()
    
    

def train(args):
    # set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Cuda? {torch.cuda.is_available()}")

    # create dataset + dataloader
    dataset = FMDataset(save_dir=args.data_dir)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=4)

    # choose an example to use as our per-epoch plot
    fixed_spec = dataset[6][1].float().unsqueeze(0).to(device)
    fixed_params = dataset[6][0]

    # create encoder + optimizer + scheduler
    encoder = FMEncoderChain(n_bins=224).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    # CQT transform
    Fs = args.Fs
    duration = args.duration
    hop_size = 512
    bins_per_octave = 32
    n_octaves = 7
    n_bins = bins_per_octave * n_octaves
    fmin = 32.7  # CQT2010v2 default

    cqt_transform = CQT2010v2(
        sr=Fs,
        hop_length=hop_size,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave
    ).to(device)

    # frequency weights — suppress fundamental contribution
    # stage 1 has fixed f0=440 so one weight vector suffices, broadcast across batch
    f0_stage1 = args.f0

    # output dir for checkpoints
    output_dir = os.path.join(args.data_dir, args.output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    # optionally resume from checkpoint
    if args.resume:
        encoder.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed from {args.resume}")

    # debug — track weight change per epoch
    initial_weight = encoder.fc1.weight.data.clone()

    for epoch in range(args.start_epoch, args.n_epochs):
        epoch_loss = 0.0
        epoch_spectral_loss = 0.0
        epoch_adaptive_lambda = 0.0
        print(f"Epoch {epoch}/{args.n_epochs}:")
        # epoch number influcenes supervised parameter losses
        supervision_fadeout_percent = 0.5 # once we're at 70% of our epochs, the supervisied losses go to zero
        lambda_direct = 1 * max(0.0, 1.0 - epoch / (args.n_epochs * supervision_fadeout_percent))
        for batch in dataloader:
            params, spec = batch
            spec = spec.float().to(device)
            f0_batch = params['f0'].float().to(device)

            optimizer.zero_grad()

            # encoder forward pass — full batch at once
            predicted = encoder(spec)

            # batched FM render — no Python item loop
            audio_batch = fm_renderer(
                f0_batch,
                predicted['ratios'],
                predicted['levels'],
                Fs, duration
            )

            # check for NaN audio before computing loss
            if torch.isnan(audio_batch).any():
                print("NaN in audio_batch, skipping batch")
                optimizer.zero_grad()
                continue

            # batched CQT spectrogram
            pred_spec = compute_spectrogram_cqt_batched(audio_batch, cqt_transform)

            loss, spectral_loss, supervision_loss, adaptive_lambda_mean = stage3_loss(predicted, params, 
                pred_spec, spec, 
                device,
                lambda_direct)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=0.5)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_spectral_loss += spectral_loss.item()
            epoch_adaptive_lambda += adaptive_lambda_mean.item()
        # end of batch loop
        epoch_loss_avg = epoch_loss / len(dataloader)
        print(f"Average epoch loss: {epoch_loss_avg}")

        weight_change = (encoder.fc1.weight.data - initial_weight).abs().mean()
        print(f"Average weight change: {weight_change.item()}")
        initial_weight = encoder.fc1.weight.data.clone()

        spectral_loss_avg = epoch_spectral_loss / len(dataloader)
        scheduler.step(spectral_loss_avg)
        adaptive_lambda_avg = epoch_adaptive_lambda / len(dataloader)
        print(f"Spectral loss avg: {spectral_loss_avg}")
        print(f"Adaptive lambda avg: {adaptive_lambda_avg}")
        # plot results of epoch
        encoder.eval()
        with torch.no_grad():
            fixed_predicted = encoder(fixed_spec)
            fixed_audio = fm_renderer(
                torch.tensor([fixed_params['f0']]).float().to(device),
                fixed_predicted['ratios'],
                fixed_predicted['levels'],
                Fs, duration
            )
            fixed_pred_spec = compute_spectrogram_cqt_batched(fixed_audio, cqt_transform)
        encoder.train()
        
        plt.figure(figsize=(12, 3))
        plt.subplot(1, 2, 1)
        plt.plot(fixed_spec.squeeze().cpu().numpy(), label='target')
        plt.plot(fixed_pred_spec.squeeze().cpu().numpy(), label='predicted')
        plt.title(f'Epoch {epoch} — Spectrogram')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.bar(range(2), fixed_params['levels'], alpha=0.5, label='GT levels')
        plt.bar(range(2), fixed_predicted['levels'][0].cpu().numpy(), alpha=0.5, label='Pred levels')
        plt.title(f'Epoch {epoch} — Levels')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        print(f"--- Epoch {epoch} Evaluation ---")
        print(f"GT levels:      {[round(x,2) for x in fixed_params['levels'].tolist()]}")
        print(f"Pred levels:    {[round(x,2) for x in fixed_predicted['levels'][0].cpu().tolist()]}")
        print(f"GT ratios:      {[round(x,2) for x in fixed_params['ratios'].tolist()]}")
        print(f"Pred ratios:    {[round(x,2) for x in fixed_predicted['ratios'][0].cpu().tolist()]}")
        print(f"lambda_direct:  {lambda_direct:.4f}")
        print(f"--------------------------------")

        # only save checkpoint if weights are valid
        if not torch.isnan(encoder.fc1.weight).any():
            torch.save(encoder.state_dict(),
                       os.path.join(output_dir, f'encoder_epoch_{epoch}.pt'))
        else:
            print(f"NaN weights detected at epoch {epoch} — stopping training")
            break




if __name__ == '__main__':
    args = parser.parse_args()
    train(args)