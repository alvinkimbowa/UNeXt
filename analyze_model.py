#!/usr/bin/env python3
"""
Analyze model parameters and FLOPs
Callable from command line or from run_train.sh
"""
import argparse
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from os.path import join

from thop import profile, clever_format

import archs
from TinyUNet import TinyUNet
from monounet import MonogenicNets, MonoUNets
from utils import str2bool

ARCH_NAMES = archs.__all__
MONO_ARCH_NAMES = MonogenicNets.__all__
MONOUNET_ARCH_NAMES = MonoUNets.__all__


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

    def forward(self, x):
        with torch.no_grad():
            base_output = self.base_model(x)
        mask = self._base_to_mask(base_output, x.shape[2:])
        refined_input = torch.cat([x, mask], dim=1)
        return self.refiner_model(refined_input)


def load_model(arch, input_channels, num_classes, input_h, input_w, deep_supervision=False):
    """Load model the same way train.py does"""
    if arch == "UNext" or arch == "UNext_S" or arch == "CMUNeXt-S":
        model = archs.__dict__[arch](num_classes, input_channels, deep_supervision)
    elif arch == "TinyUNet":
        model = TinyUNet(input_channels, num_classes)
    elif arch in MONO_ARCH_NAMES:
        model = MonogenicNets.__dict__[arch](
            input_channels,
            num_classes,
            img_size=(input_h, input_w),
            deep_supervision=deep_supervision
        )
    elif arch in MONOUNET_ARCH_NAMES:
        model = MonoUNets.__dict__[arch](
            in_channels=input_channels,
            num_classes=num_classes,
            img_size=(input_h, input_w),
            deep_supervision=deep_supervision
        )
    else:
        raise NotImplementedError(f"Architecture {arch} not supported")
    
    return model


def load_cascaded_model(arch, base_arch, input_channels, num_classes, input_h, input_w,
                        deep_supervision=False, mask_threshold=0.5, mask_class=1, base_ckpt=None,
                        mask_dropout=0.0, mask_dropout_foreground_only=False, mask_patch_prob=0.0,
                        mask_patch_empty_prob=0.0, mask_patch_bands=4, mask_patch_min_bands=1,
                        mask_patch_max_bands=2, mask_foreground_prob=0.0,
                        mask_foreground_blobs_min=1, mask_foreground_blobs_max=3,
                        mask_foreground_radius_min=6, mask_foreground_radius_max=24,
                        mask_shift_prob=0.0, mask_shift_max=16,
                        mask_rotate_prob=0.0, mask_rotate_max_deg=10.0):
    base_model = load_model(base_arch, input_channels, num_classes, input_h, input_w, deep_supervision)
    if base_ckpt:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(base_ckpt, weights_only=False, map_location=device)
        state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
        base_model.load_state_dict(state)
    refiner_model = load_model(arch, input_channels + 1, num_classes, input_h, input_w, deep_supervision)
    return CascadedSegModel(
        base_model=base_model,
        refiner_model=refiner_model,
        mask_threshold=mask_threshold,
        mask_class=mask_class,
        mask_dropout=mask_dropout,
        mask_dropout_foreground_only=mask_dropout_foreground_only,
        mask_patch_prob=mask_patch_prob,
        mask_patch_empty_prob=mask_patch_empty_prob,
        mask_patch_bands=mask_patch_bands,
        mask_patch_min_bands=mask_patch_min_bands,
        mask_patch_max_bands=mask_patch_max_bands,
        mask_foreground_prob=mask_foreground_prob,
        mask_foreground_blobs_min=mask_foreground_blobs_min,
        mask_foreground_blobs_max=mask_foreground_blobs_max,
        mask_foreground_radius_min=mask_foreground_radius_min,
        mask_foreground_radius_max=mask_foreground_radius_max,
        mask_shift_prob=mask_shift_prob,
        mask_shift_max=mask_shift_max,
        mask_rotate_prob=mask_rotate_prob,
        mask_rotate_max_deg=mask_rotate_max_deg,
    )


def count_parameters(model):
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def count_flops(model, input_tensor):
    """Count FLOPs using thop"""
    macs, params = profile(model, inputs=(input_tensor,), verbose=False)
    flops = macs * 2  # MACs to FLOPS (multiply-add operations)
    return macs, flops, params


def analyze_model(arch, input_channels=1, num_classes=2, input_h=256, input_w=256, 
                  deep_supervision=False, gpu=0, save_path=None, cascade_refiner=False,
                  base_arch=None, mask_threshold=0.5, mask_class=1, base_ckpt=None,
                  mask_dropout=0.0, mask_dropout_foreground_only=False, mask_patch_prob=0.0,
                  mask_patch_empty_prob=0.0, mask_patch_bands=4, mask_patch_min_bands=1,
                  mask_patch_max_bands=2, mask_foreground_prob=0.0,
                  mask_foreground_blobs_min=1, mask_foreground_blobs_max=3,
                  mask_foreground_radius_min=6, mask_foreground_radius_max=24,
                  mask_shift_prob=0.0, mask_shift_max=16,
                  mask_rotate_prob=0.0, mask_rotate_max_deg=10.0):
    """Analyze model parameters and FLOPs"""
    
    # Set device
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*60}")
    print(f"MODEL ANALYSIS")
    print(f"{'='*60}")
    print(f"Architecture: {arch}")
    print(f"Input channels: {input_channels}")
    print(f"Number of classes: {num_classes}")
    print(f"Input size: {input_h}x{input_w}")
    print(f"Deep supervision: {deep_supervision}")
    print(f"Device: {device}")
    print(f"{'='*60}")
    
    # Load model
    if cascade_refiner:
        base_arch = base_arch or arch
        model = load_cascaded_model(
            arch,
            base_arch,
            input_channels,
            num_classes,
            input_h,
            input_w,
            deep_supervision,
            mask_threshold,
            mask_class,
            base_ckpt,
            mask_dropout,
            mask_dropout_foreground_only,
            mask_patch_prob,
            mask_patch_empty_prob,
            mask_patch_bands,
            mask_patch_min_bands,
            mask_patch_max_bands,
            mask_foreground_prob,
            mask_foreground_blobs_min,
            mask_foreground_blobs_max,
            mask_foreground_radius_min,
            mask_foreground_radius_max,
            mask_shift_prob,
            mask_shift_max,
            mask_rotate_prob,
            mask_rotate_max_deg,
        )
    else:
        model = load_model(arch, input_channels, num_classes, input_h, input_w, deep_supervision)
    model = model.to(device)
    model.eval()
    print("✓ Model loaded successfully")
    
    # Create input tensor
    input_tensor = torch.randn(1, input_channels, input_h, input_w).to(device)
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    
    # Count FLOPs
    macs, flops, thop_params = count_flops(model, input_tensor)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"PARAMETERS:")
    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"  Non-trainable parameters: {total_params - trainable_params:,}")
    
    print(f"\nCOMPUTATIONAL COMPLEXITY:")
    print(f"  MACs (Multiply-Accumulates): {macs:,} ({macs/1e9:.2f}G)")
    print(f"  FLOPS (Floating Point Operations): {flops:,} ({flops/1e9:.2f}G)")
    print(f"  thop parameter count: {thop_params:,}")
    
    print(f"{'='*60}\n")
    
    # Prepare metrics dictionary
    metrics = {
        "architecture": arch,
        "cascade_refiner": bool(cascade_refiner),
        "base_arch": base_arch if cascade_refiner else None,
        "input_channels": input_channels,
        "mask_dropout": float(mask_dropout) if cascade_refiner else None,
        "mask_dropout_foreground_only": bool(mask_dropout_foreground_only) if cascade_refiner else None,
        "mask_patch_prob": float(mask_patch_prob) if cascade_refiner else None,
        "mask_patch_empty_prob": float(mask_patch_empty_prob) if cascade_refiner else None,
        "mask_patch_bands": int(mask_patch_bands) if cascade_refiner else None,
        "mask_patch_min_bands": int(mask_patch_min_bands) if cascade_refiner else None,
        "mask_patch_max_bands": int(mask_patch_max_bands) if cascade_refiner else None,
        "mask_foreground_prob": float(mask_foreground_prob) if cascade_refiner else None,
        "mask_foreground_blobs_min": int(mask_foreground_blobs_min) if cascade_refiner else None,
        "mask_foreground_blobs_max": int(mask_foreground_blobs_max) if cascade_refiner else None,
        "mask_foreground_radius_min": int(mask_foreground_radius_min) if cascade_refiner else None,
        "mask_foreground_radius_max": int(mask_foreground_radius_max) if cascade_refiner else None,
        "mask_shift_prob": float(mask_shift_prob) if cascade_refiner else None,
        "mask_shift_max": int(mask_shift_max) if cascade_refiner else None,
        "mask_rotate_prob": float(mask_rotate_prob) if cascade_refiner else None,
        "mask_rotate_max_deg": float(mask_rotate_max_deg) if cascade_refiner else None,
        "num_classes": num_classes,
        "input_size": [input_h, input_w],
        "deep_supervision": deep_supervision,
        "device": str(device),
        "parameters": {
            "total": int(total_params),
            "trainable": int(trainable_params),
            "non_trainable": int(total_params - trainable_params),
            "total_millions": round(total_params/1e6, 2),
            "trainable_millions": round(trainable_params/1e6, 2)
        }
    }
    
    metrics["macs"] = {
        "total": int(macs),
        "giga": round(macs/1e9, 2)
    }
    metrics["flops"] = {
        "total": int(flops),
        "giga": round(flops/1e9, 2)
    }
    
    # Save metrics to JSON
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {save_path}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Analyze model parameters and FLOPs')
    parser.add_argument('--arch', '-a', type=str, required=True,
                        help='Model architecture name')
    parser.add_argument('--input_channels', type=int, default=1,
                        help='Number of input channels (default: 1)')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of output classes (default: 2)')
    parser.add_argument('--input_h', type=int, default=256,
                        help='Input height (default: 256)')
    parser.add_argument('--input_w', type=int, default=256,
                        help='Input width (default: 256)')
    parser.add_argument('--deep_supervision', default=False, type=str2bool,
                        help='Use deep supervision')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID (default: 0, use -1 for CPU)')
    parser.add_argument('--save', type=str, default=None,
                        help='Path to save metrics JSON (optional)')
    parser.add_argument('--cascade_refiner', default=False, type=str2bool,
                        help='analyze a refiner model with a frozen base model')
    parser.add_argument('--base_arch', default=None,
                        help='base model architecture for cascade analysis')
    parser.add_argument('--base_ckpt', default=None,
                        help='checkpoint path for the frozen base model')
    parser.add_argument('--mask_threshold', default=0.5, type=float,
                        help='threshold for converting base logits to mask')
    parser.add_argument('--mask_class', default=1, type=int,
                        help='class index to use for mask when base has >1 channel')
    parser.add_argument('--mask_dropout', default=0.0, type=float,
                        help='dropout probability for the base mask during refiner analysis')
    parser.add_argument('--mask_dropout_foreground_only', default=False, type=str2bool,
                        help='apply mask dropout only on foreground pixels')
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
    
    args = parser.parse_args()
    print("save path: ", args.save)
    analyze_model(
        arch=args.arch,
        input_channels=args.input_channels,
        num_classes=args.num_classes,
        input_h=args.input_h,
        input_w=args.input_w,
        deep_supervision=args.deep_supervision,
        gpu=args.gpu,
        save_path=args.save,
        cascade_refiner=args.cascade_refiner,
        base_arch=args.base_arch,
        mask_threshold=args.mask_threshold,
        mask_class=args.mask_class,
        base_ckpt=args.base_ckpt,
        mask_dropout=args.mask_dropout,
        mask_dropout_foreground_only=args.mask_dropout_foreground_only,
        mask_patch_prob=args.mask_patch_prob,
        mask_patch_empty_prob=args.mask_patch_empty_prob,
        mask_patch_bands=args.mask_patch_bands,
        mask_patch_min_bands=args.mask_patch_min_bands,
        mask_patch_max_bands=args.mask_patch_max_bands,
        mask_foreground_prob=args.mask_foreground_prob,
        mask_foreground_blobs_min=args.mask_foreground_blobs_min,
        mask_foreground_blobs_max=args.mask_foreground_blobs_max,
        mask_foreground_radius_min=args.mask_foreground_radius_min,
        mask_foreground_radius_max=args.mask_foreground_radius_max,
        mask_shift_prob=args.mask_shift_prob,
        mask_shift_max=args.mask_shift_max,
        mask_rotate_prob=args.mask_rotate_prob,
        mask_rotate_max_deg=args.mask_rotate_max_deg,
    )


if __name__ == '__main__':
    main()
