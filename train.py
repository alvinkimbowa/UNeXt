import argparse
import os
import json
from collections import OrderedDict
from glob import glob

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
from albumentations import RandomRotate90,Resize,Flip
from albumentations.augmentations.geometric.transforms import Affine
import torch.nn.functional as F

import archs
import losses
from dataset import Dataset, nnUNetDataset
from metrics import iou_score
from utils import AverageMeter, str2bool
from archs import UNext
from monounet.MonogenicNets import center_crop
from TinyUNet import TinyUNet
from monounet import MonogenicNets, MonoUNets
from monounet.mono_layer import Mono2D

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')
MONO_ARCH_NAMES = MonogenicNets.__all__
MONOUNET_ARCH_NAMES = MonoUNets.__all__

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    
    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='UNext')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--data_augmentation', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=1, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=2, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=256, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=256, type=int,
                        help='image height')
    parser.add_argument('--cascade_refiner', default=False, type=str2bool,
                        help='train a refiner model with a frozen base model')
    parser.add_argument('--base_arch', default=None,
                        help='base model architecture for cascade')
    parser.add_argument('--base_ckpt', default=None,
                        help='checkpoint path for the frozen base model')
    parser.add_argument('--mask_threshold', default=0.5, type=float,
                        help='threshold for converting base logits to mask')
    parser.add_argument('--mask_class', default=1, type=int,
                        help='class index to use for mask when base has >1 channel')
    parser.add_argument('--mask_dropout', default=0.0, type=float,
                        help='dropout probability for the base mask during refiner training')
    parser.add_argument('--mask_dropout_foreground_only', default=False, type=str2bool,
                        help='apply mask dropout only on foreground pixels')
    parser.add_argument('--save_mask_debug', default=False, type=str2bool,
                        help='save debug images of masked inputs during training')
    parser.add_argument('--mask_debug_samples', default=4, type=int,
                        help='number of debug samples to save per epoch')
    parser.add_argument('--mask_debug_every', default=1, type=int,
                        help='save debug masks every N epochs')
    parser.add_argument('--mask_patch_prob', default=0.0, type=float,
                        help='probability to apply patch masking on the base mask')
    parser.add_argument('--mask_patch_empty_prob', default=0.0, type=float,
                        help='probability to replace the base mask with all zeros')
    parser.add_argument('--mask_patch_bands', default=4, type=int,
                        help='number of vertical bands for patch masking')
    parser.add_argument('--mask_patch_min_bands', default=1, type=int,
                        help='min number of bands to mask')
    parser.add_argument('--mask_patch_max_bands', default=2, type=int,
                        help='max number of bands to mask')
    parser.add_argument('--mask_foreground_prob', default=0.0, type=float,
                        help='probability to add random foreground blobs to the base mask')
    parser.add_argument('--mask_foreground_blobs_min', default=1, type=int,
                        help='min number of random foreground blobs')
    parser.add_argument('--mask_foreground_blobs_max', default=3, type=int,
                        help='max number of random foreground blobs')
    parser.add_argument('--mask_foreground_radius_min', default=6, type=int,
                        help='min radius (px) for random foreground blobs')
    parser.add_argument('--mask_foreground_radius_max', default=24, type=int,
                        help='max radius (px) for random foreground blobs')
    parser.add_argument('--mask_shift_prob', default=0.0, type=float,
                        help='probability to shift the base mask')
    parser.add_argument('--mask_shift_max', default=16, type=int,
                        help='max pixel shift for mask translation')
    parser.add_argument('--mask_rotate_prob', default=0.0, type=float,
                        help='probability to rotate the base mask')
    parser.add_argument('--mask_rotate_max_deg', default=10.0, type=float,
                        help='max degrees for mask rotation')
    parser.add_argument('--model_dir_suffix', default='',
                        help='optional suffix for model directory naming')
    
    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: BCEDiceLoss)')
    
    # dataset
    parser.add_argument('--dataset', default='isic',
                        help='dataset name')
    parser.add_argument('--img_ext', default='.png',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')
    parser.add_argument('--split', default='Tr',
                        help='split (Tr or Ts)')
    parser.add_argument('--fold', default=0,
                        help='fold index (0-4) or "all" to combine all folds')

    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD', 'AdamW'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD', 'AdamW']) +
                        ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR', 'PolyLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    parser.add_argument('--resume', default=None, type=str,
                        help='path to checkpoint to resume training')
    parser.add_argument('--save_every', default=10, type=int,
                        help='save full model checkpoint every N epochs (default: 10)')

    parser.add_argument('--num_workers', default=4, type=int)

    config = parser.parse_args()

    return config


class CascadedSegModel(nn.Module):
    def __init__(self, base_model, refiner_model, mask_threshold=0.5, mask_class=1,
                 mask_dropout=0.0, mask_dropout_foreground_only=False, mask_patch_prob=0.0,
                 mask_patch_empty_prob=0.0, mask_patch_bands=4, mask_patch_min_bands=1,
                 mask_patch_max_bands=2, mask_foreground_prob=0.0,
                 mask_foreground_blobs_min=1, mask_foreground_blobs_max=3,
                 mask_foreground_radius_min=6, mask_foreground_radius_max=24,
                 mask_shift_prob=0.0, mask_shift_max=16,
                 mask_rotate_prob=0.0, mask_rotate_max_deg=10.0):
        super().__init__()
        self.base_model = base_model
        self.refiner_model = refiner_model
        self.mask_threshold = mask_threshold
        self.mask_class = mask_class
        self.mask_dropout = mask_dropout
        self.mask_dropout_foreground_only = mask_dropout_foreground_only
        self.mask_patch_prob = mask_patch_prob
        self.mask_patch_empty_prob = mask_patch_empty_prob
        self.mask_patch_bands = mask_patch_bands
        self.mask_patch_min_bands = mask_patch_min_bands
        self.mask_patch_max_bands = mask_patch_max_bands
        self.mask_foreground_prob = mask_foreground_prob
        self.mask_foreground_blobs_min = mask_foreground_blobs_min
        self.mask_foreground_blobs_max = mask_foreground_blobs_max
        self.mask_foreground_radius_min = mask_foreground_radius_min
        self.mask_foreground_radius_max = mask_foreground_radius_max
        self.mask_shift_prob = mask_shift_prob
        self.mask_shift_max = mask_shift_max
        self.mask_rotate_prob = mask_rotate_prob
        self.mask_rotate_max_deg = mask_rotate_max_deg
        for p in self.base_model.parameters():
            p.requires_grad = False
        self.base_model.eval()

    def train(self, mode=True):
        super().train(mode)
        # Keep the base model frozen regardless of training mode.
        self.base_model.eval()
        return self

    def _base_to_mask(self, base_output, target_size):
        if isinstance(base_output, (list, tuple)):
            base_output = base_output[-1]
        if base_output.dim() != 4:
            raise ValueError("Base model output must be a 4D tensor")
        if base_output.size(1) > 1:
            probs = torch.softmax(base_output, dim=1)
            pred = torch.argmax(probs, dim=1, keepdim=True)
            mask = (pred == self.mask_class).float()
        else:
            probs = torch.sigmoid(base_output)
            mask = (probs >= self.mask_threshold).float()
        if mask.size(2) != target_size[0] or mask.size(3) != target_size[1]:
            mask = F.interpolate(mask, size=target_size, mode='nearest')
        return mask

    def _shift_mask(self, mask, dy, dx):
        b, c, h, w = mask.shape
        out = torch.zeros_like(mask)
        y1 = max(0, dy)
        y2 = min(h, h + dy)
        x1 = max(0, dx)
        x2 = min(w, w + dx)
        src_y1 = max(0, -dy)
        src_y2 = src_y1 + (y2 - y1)
        src_x1 = max(0, -dx)
        src_x2 = src_x1 + (x2 - x1)
        if y2 > y1 and x2 > x1:
            out[:, :, y1:y2, x1:x2] = mask[:, :, src_y1:src_y2, src_x1:src_x2]
        return out

    def _rotate_mask(self, mask, angles_deg):
        b, c, h, w = mask.shape
        angles = angles_deg * (torch.pi / 180.0)
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)
        theta = torch.zeros(b, 2, 3, device=mask.device, dtype=mask.dtype)
        theta[:, 0, 0] = cos_a
        theta[:, 0, 1] = -sin_a
        theta[:, 1, 0] = sin_a
        theta[:, 1, 1] = cos_a
        grid = F.affine_grid(theta, mask.size(), align_corners=False)
        return F.grid_sample(mask, grid, mode='nearest', padding_mode='zeros', align_corners=False)

    def _apply_mask_augment(self, mask):
        if self.training and self.mask_shift_prob > 0 and self.mask_shift_max > 0:
            apply_shift = (torch.rand(mask.size(0), device=mask.device) < self.mask_shift_prob)
            if apply_shift.any():
                max_shift = int(self.mask_shift_max)
                for idx in torch.nonzero(apply_shift, as_tuple=False).flatten():
                    dy = int(torch.randint(-max_shift, max_shift + 1, (1,), device=mask.device).item())
                    dx = int(torch.randint(-max_shift, max_shift + 1, (1,), device=mask.device).item())
                    mask[idx:idx + 1] = self._shift_mask(mask[idx:idx + 1], dy, dx)
        if self.training and self.mask_rotate_prob > 0 and self.mask_rotate_max_deg > 0:
            apply_rot = (torch.rand(mask.size(0), device=mask.device) < self.mask_rotate_prob)
            if apply_rot.any():
                max_deg = float(self.mask_rotate_max_deg)
                angles = torch.zeros(mask.size(0), device=mask.device, dtype=mask.dtype)
                angles[apply_rot] = (torch.rand(int(apply_rot.sum()), device=mask.device) * 2 - 1) * max_deg
                mask = self._rotate_mask(mask, angles)
        if self.training and self.mask_patch_empty_prob > 0:
            empty_keep = (torch.rand(mask.size(0), 1, 1, 1, device=mask.device) >= self.mask_patch_empty_prob).float()
            mask = mask * empty_keep
        if self.training and self.mask_patch_prob > 0 and self.mask_patch_bands > 1:
            apply_patch = (torch.rand(mask.size(0), device=mask.device) < self.mask_patch_prob)
            if apply_patch.any():
                _, _, h, w = mask.shape
                bands = self.mask_patch_bands
                band_width = w // bands
                for idx in torch.nonzero(apply_patch, as_tuple=False).flatten():
                    k_min = max(1, min(self.mask_patch_min_bands, bands))
                    k_max = max(k_min, min(self.mask_patch_max_bands, bands))
                    k = int(torch.randint(k_min, k_max + 1, (1,), device=mask.device).item())
                    band_indices = torch.randperm(bands, device=mask.device)[:k]
                    for b in band_indices:
                        start = int(b.item() * band_width)
                        end = int((b.item() + 1) * band_width) if b.item() < bands - 1 else w
                        mask[idx, :, :, start:end] = 0
        if self.training and self.mask_foreground_prob > 0:
            apply_fg = (torch.rand(mask.size(0), device=mask.device) < self.mask_foreground_prob)
            if apply_fg.any():
                _, _, h, w = mask.shape
                for idx in torch.nonzero(apply_fg, as_tuple=False).flatten():
                    rmin = max(1, int(self.mask_foreground_radius_min))
                    rmax = max(rmin, int(self.mask_foreground_radius_max))
                    nmin = max(1, int(self.mask_foreground_blobs_min))
                    nmax = max(nmin, int(self.mask_foreground_blobs_max))
                    n = int(torch.randint(nmin, nmax + 1, (1,), device=mask.device).item())
                    for _ in range(n):
                        k = int(torch.randint(rmin, rmax + 1, (1,), device=mask.device).item())
                        k = max(3, k | 1)  # odd kernel size >= 3
                        k = min(k, h if h % 2 == 1 else h - 1, w if w % 2 == 1 else w - 1)
                        noise = torch.rand(1, 1, h, w, device=mask.device)
                        smooth = F.avg_pool2d(noise, kernel_size=k, stride=1, padding=k // 2)
                        q = 0.97 + 0.02 * torch.rand(1, device=mask.device)
                        thresh = torch.quantile(smooth.view(-1), q)
                        blob = (smooth > thresh).float()
                        blob = F.max_pool2d(blob, kernel_size=3, stride=1, padding=1)
                        mask[idx, 0] = torch.clamp(mask[idx, 0] + blob[0, 0], 0, 1)
        if self.training and self.mask_dropout > 0:
            if self.mask_dropout_foreground_only:
                keep = (torch.rand_like(mask) >= self.mask_dropout).float()
                mask = mask * keep
            else:
                keep = (torch.rand(mask.size(0), 1, 1, 1, device=mask.device) >= self.mask_dropout).float()
                mask = mask * keep
        return mask

    def make_mask(self, x, apply_dropout=True):
        with torch.no_grad():
            base_output = self.base_model(x)
        mask = self._base_to_mask(base_output, x.shape[2:])
        if apply_dropout:
            mask = self._apply_mask_augment(mask)
        return mask

    def forward(self, x):
        mask = self.make_mask(x, apply_dropout=True)
        refined_input = torch.cat([x, mask], dim=1)
        return self.refiner_model(refined_input)

def plot_training_curves(log_data, output_path):
    """
    Plot training curves with loss and dice metrics.
    
    Args:
        log_data: Dictionary with keys 'epoch', 'loss', 'val_loss', 'dice', 'val_dice'
        output_path: Path to save the plot
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    epochs = log_data['epoch']
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:blue')
    line1 = ax1.plot(epochs, log_data['loss'], 'b-', label='Train Loss', alpha=0.7)
    line2 = ax1.plot(epochs, log_data['val_loss'], 'b--', label='Val Loss', alpha=0.7)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, axis='x', alpha=0.3)  # Vertical grid lines for epochs
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Dice', color='tab:red')
    line3 = ax2.plot(epochs, log_data['dice'], 'r-', label='Train Dice', alpha=0.7)
    line4 = ax2.plot(epochs, log_data['val_dice'], 'r--', label='Val Dice', alpha=0.7)
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.grid(True, axis='y', alpha=0.3)  # Horizontal grid lines for dice
    
    # Combine legends
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(0, 1.12), ncols=2, frameon=False)
    
    plt.title('Training Curves')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def _flatten_list(values):
    flat = []
    for v in values:
        if isinstance(v, (list, tuple)):
            flat.extend(_flatten_list(v))
        else:
            flat.append(float(v))
    return flat


def log_mono_params(model, mono_history, epoch):
    """
    Collect Mono2D* params (rescaled via get_params) and accumulate in memory.
    """
    for name, module in model.named_modules():
        if not isinstance(module, Mono2D):
            continue
        params = module.get_params()
        if name not in mono_history:
            mono_history[name] = {
                "epochs": [],
                "wls": [],
                "wls_x": [],
                "wls_y": [],
                "sigmaonf": [],
            }
        entry = mono_history[name]
        entry["epochs"].append(epoch)
        # Store flattened values to keep arrays JSON-friendly
        entry["wls"].append(_flatten_list(params.get("wls", [])) if params.get("wls", None) is not None else None)
        entry["wls_x"].append(_flatten_list(params.get("wls_x", [])) if params.get("wls_x", None) is not None else None)
        entry["wls_y"].append(_flatten_list(params.get("wls_y", [])) if params.get("wls_y", None) is not None else None)
        entry["sigmaonf"].append(_flatten_list(params.get("sigmaonf", [])) if params.get("sigmaonf", None) is not None else None)


def save_mono_param_logs(mono_history, model_dir):
    """
    Save a single JSON of all epochs and plot trajectories for wls/wls_x/wls_y/sigmaonf.
    Call this every epoch to keep plots/files updated like loss curves.
    """
    if not mono_history:
        return

    out_dir = os.path.join(model_dir, "mono_params")
    os.makedirs(out_dir, exist_ok=True)

    # Save one JSON with full history (overwrites each call)
    json_path = os.path.join(out_dir, "mono_params_all_epochs.json")
    with open(json_path, "w") as f:
        json.dump(mono_history, f, indent=2)

    # Plot trajectories per module (overwrites each call)
    for name, hist in mono_history.items():
        epochs = hist.get("epochs", [])
        if not epochs:
            continue

        to_plot = []
        for key in ["wls", "wls_x", "wls_y", "sigmaonf"]:
            series = hist.get(key, [])
            # Skip if empty or all None
            if not series or all(v is None for v in series):
                continue
            to_plot.append((key, series))

        if not to_plot:
            continue

        fig, axes = plt.subplots(len(to_plot), 1, figsize=(8, 3 * len(to_plot)), squeeze=False)
        axes = axes[:, 0]
        for ax, (label, series) in zip(axes, to_plot):
            max_len = max(len(v) for v in series if v is not None)
            for idx in range(max_len):
                xs, ys = [], []
                for ep, values in zip(epochs, series):
                    if values is None or idx >= len(values):
                        continue
                    xs.append(ep)
                    ys.append(values[idx])
                if not xs:
                    continue
                ax.plot(xs, ys, label=f"{label}[{idx}]", linewidth=1)
            ax.set_title(f"{name} - {label} over epochs")
            ax.set_xlabel("epoch")
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, ncol=2)

        plt.tight_layout()
        fig_path = os.path.join(out_dir, f"{name.replace('.', '_')}_trajectories.png")
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)


# args = parser.parse_args()
def train(config, train_loader, model, criterion, optimizer, epoch):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for batch_idx, (input, target, _) in enumerate(train_loader):
        input = input.cuda()
        target = target.cuda()

        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                # Center crop target to match output size
                target_cropped = center_crop(target, output.size()[2:])
                loss += criterion(output, target_cropped)
            loss /= len(outputs)
            # Center crop target for metrics on final output
            target_cropped = center_crop(target, outputs[-1].size()[2:])
            iou,dice = iou_score(outputs[-1], target_cropped)
        else:
            output = model(input)
            # Center crop target to match output size
            target_cropped = center_crop(target, output.size()[2:])
            loss = criterion(output, target_cropped)
            iou,dice = iou_score(output, target_cropped)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        avg_meters['dice'].update(dice, input.size(0))

        if (config.get('save_mask_debug') and config.get('cascade_refiner') and
                hasattr(model, 'make_mask') and batch_idx == 0 and
                (epoch % max(config.get('mask_debug_every', 1), 1) == 0)):
            debug_dir = os.path.join(config['model_dir'], 'debug')
            os.makedirs(debug_dir, exist_ok=True)
            with torch.no_grad():
                mask = model.make_mask(input, apply_dropout=True).cpu()
            max_samples = min(config.get('mask_debug_samples', 4), input.size(0))
            for i in range(max_samples):
                fig, axes = plt.subplots(1, 2, figsize=(6, 3))
                img = input[i, 0].detach().cpu().numpy()
                axes[0].imshow(img, cmap='gray')
                axes[0].set_title('input')
                axes[0].axis('off')
                axes[1].imshow(mask[i, 0].numpy(), cmap='gray')
                axes[1].set_title('mask')
                axes[1].axis('off')
                fig.tight_layout()
                out_path = os.path.join(debug_dir, f'sample_{i}.png')
                plt.savefig(out_path, dpi=150, bbox_inches='tight')
                plt.close(fig)

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)])


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                   'dice': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    # Center crop target to match output size
                    target_cropped = center_crop(target, output.size()[2:])
                    loss += criterion(output, target_cropped)
                loss /= len(outputs)
                # Center crop target for metrics on final output
                target_cropped = center_crop(target, outputs[-1].size()[2:])
                iou,dice = iou_score(outputs[-1], target_cropped)
            else:
                output = model(input)
                # Center crop target to match output size
                target_cropped = center_crop(target, output.size()[2:])
                loss = criterion(output, target_cropped)
                iou,dice = iou_score(output, target_cropped)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)])


def main():
    config = vars(parse_args())

    fold_str = str(config['fold'])
    arch_name = config['arch']
    if config['deep_supervision']:
        arch_name += 'DS'
    if config['data_augmentation']:
        arch_name += 'DA'
    if config.get('model_dir_suffix'):
        arch_name += f"_{config['model_dir_suffix']}"
    model_dir = f"models/{arch_name}/{config['dataset']}/fold_{fold_str}"
    print("model dir:", model_dir)
    os.makedirs(model_dir, exist_ok=True)
    config['model_dir'] = model_dir
    
    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open(f'{model_dir}/config.yml', 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    cudnn.benchmark = True

    def build_model(arch_name, input_channels):
        if arch_name == "UNext" or arch_name == "UNext_S":
            return archs.__dict__[arch_name](config['num_classes'],
                                             input_channels,
                                             config['deep_supervision'])
        if arch_name == "TinyUNet":
            return TinyUNet(input_channels, config['num_classes'])
        if arch_name in MONO_ARCH_NAMES:
            return MonogenicNets.__dict__[arch_name](input_channels,
                                                     config['num_classes'],
                                                     img_size=(config['input_h'], config['input_w']),
                                                     deep_supervision=config['deep_supervision'])
        if arch_name in MONOUNET_ARCH_NAMES:
            return MonoUNets.__dict__[arch_name](in_channels=input_channels,
                                                 num_classes=config['num_classes'],
                                                 img_size=(config['input_h'], config['input_w']),
                                                 deep_supervision=config['deep_supervision'])
        raise NotImplementedError

    # create model
    if config['cascade_refiner']:
        if not config['base_ckpt']:
            raise ValueError("cascade_refiner requires --base_ckpt for the frozen base model")
        base_arch = config['base_arch'] or config['arch']
        base_model = build_model(base_arch, config['input_channels'])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_ckpt = torch.load(config['base_ckpt'], weights_only=False, map_location=device)
        base_state = base_ckpt.get('model_state_dict', base_ckpt.get('state_dict', base_ckpt))
        base_model.load_state_dict(base_state)
        refiner_in_channels = config['input_channels'] + 1
        refiner_model = build_model(config['arch'], refiner_in_channels)
        model = CascadedSegModel(
            base_model=base_model,
            refiner_model=refiner_model,
            mask_threshold=config['mask_threshold'],
            mask_class=config['mask_class'],
            mask_dropout=config.get('mask_dropout', 0.0),
            mask_dropout_foreground_only=config.get('mask_dropout_foreground_only', False),
            mask_patch_prob=config.get('mask_patch_prob', 0.0),
            mask_patch_empty_prob=config.get('mask_patch_empty_prob', 0.0),
            mask_patch_bands=config.get('mask_patch_bands', 4),
            mask_patch_min_bands=config.get('mask_patch_min_bands', 1),
            mask_patch_max_bands=config.get('mask_patch_max_bands', 2),
            mask_foreground_prob=config.get('mask_foreground_prob', 0.0),
            mask_foreground_blobs_min=config.get('mask_foreground_blobs_min', 1),
            mask_foreground_blobs_max=config.get('mask_foreground_blobs_max', 3),
            mask_foreground_radius_min=config.get('mask_foreground_radius_min', 6),
            mask_foreground_radius_max=config.get('mask_foreground_radius_max', 24),
            mask_shift_prob=config.get('mask_shift_prob', 0.0),
            mask_shift_max=config.get('mask_shift_max', 16),
            mask_rotate_prob=config.get('mask_rotate_prob', 0.0),
            mask_rotate_max_deg=config.get('mask_rotate_max_deg', 10.0),
        )
    else:
        model = build_model(config['arch'], config['input_channels'])

    print(model)
    model = model.cuda()

    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(params, lr=config['lr'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    elif config['scheduler'] == 'PolyLR':
        scheduler = lr_scheduler.PolynomialLR(optimizer, total_iters=config['epochs'], power=config.get('power', 0.9))
    else:
        raise NotImplementedError
    
    if config['data_augmentation']:
        # The XTinyMonoV2 models only need scaling (and rotation!!?).
        # No need for color, intensity, contrast, brightness, etc. augmentations.
        # The monogenic layer makes these obselete as it is already invariant to these.
        train_transform = Compose([
                Affine(rotate=(-15, 15), scale=(0.8, 1.2), p=0.8),
                Resize(config['input_h'], config['input_w']),
                transforms.Normalize(),
            ])
    else:
        train_transform = Compose([
            Resize(config['input_h'], config['input_w']),
            transforms.Normalize(),
        ])

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    train_dataset = nnUNetDataset(
        dataset_name=config['dataset'],
        input_channels=config['input_channels'],
        split=config['split'],
        fold=config['fold'],
        split_type='train',
        transform=train_transform)
    
    val_dataset = nnUNetDataset(
        dataset_name=config['dataset'],
        input_channels=config['input_channels'],
        split=config['split'],
        fold=config['fold'],
        split_type='val',
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('dice', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
    ])
    mono_history = {}

    best_dice = 0
    trigger = 0
    start_epoch = 0

    device = next(model.parameters()).device
    ckpt_path = config.get('resume') or os.path.join(model_dir, "model_latest.pth")
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, weights_only=False, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict'] and scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_dice = checkpoint.get('best_dice', checkpoint.get('best_iou', best_dice))
        start_epoch = checkpoint['epoch'] + 1
        if 'log' in checkpoint and isinstance(checkpoint['log'], OrderedDict):
            log = checkpoint['log']
        if 'mono_history' in checkpoint and isinstance(checkpoint['mono_history'], dict):
            mono_history = checkpoint['mono_history']
        trigger = checkpoint.get('trigger', trigger)
        print(f"Loaded checkpoint from epoch {start_epoch}")

    for epoch in range(start_epoch, config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])
        elif config['scheduler'] == 'PolyLR':
            scheduler.step()

        print('loss %.4f - dice %.4f - val_loss %.4f - val_dice %.4f'
              % (train_log['loss'], train_log['dice'], val_log['loss'], val_log['dice']))

        log['epoch'].append(epoch)
        log['lr'].append(optimizer.param_groups[0]['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['dice'].append(train_log['dice'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])

        # Log Mono2D parameter snapshots (rescaled values) per epoch
        log_mono_params(model, mono_history, epoch)
        save_mono_param_logs(mono_history, model_dir)

        pd.DataFrame(log).to_csv(f'{model_dir}/log.csv', index=False)
        
        # Plot training curves
        plot_training_curves(log, f'{model_dir}/loss_curves.png')

        trigger += 1

        if val_log['dice'] > best_dice:
            torch.save(model.state_dict(), f'{model_dir}/model_best.pth')
            best_dice = val_log['dice']
            print("=> saved best model")
            trigger = 0

        # Save full checkpoint for resuming every save_every epochs
        if (epoch + 1) % config['save_every'] == 0:
            checkpoint_payload = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'best_dice': best_dice,
                'trigger': trigger,
                'log': log,
                'mono_history': mono_history,
                'config': config,
                # legacy aliases for backward compatibility
                'state_dict': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
                'best_iou': best_dice,
            }
            torch.save(checkpoint_payload, ckpt_path)

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()
    
    # Save final checkpoint with full training state
    final_checkpoint = {
        'epoch': config['epochs'] - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'best_dice': best_dice,
        'trigger': trigger,
        'log': log,
        'mono_history': mono_history,
        'config': config,
        # legacy aliases for backward compatibility
        'state_dict': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
        'best_iou': best_dice,
    }
    torch.save(final_checkpoint, f'{model_dir}/model_final.pth')
    print("=> saved final model")
    save_mono_param_logs(mono_history, model_dir)
    print("Training completed")


if __name__ == '__main__':
    main()
