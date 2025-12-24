import argparse
import os
from glob import glob

import cv2
import csv
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import label

import archs
from dataset import nnUNetDataset
from metrics import iou_score
from utils import AverageMeter, str2bool
from albumentations import RandomRotate90,Resize
import time
from archs import UNext
import numpy as np
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric
from TinyUNet import TinyUNet
from monounet import MonogenicNets, MonoUNets
from monounet.MonogenicNets import center_crop

MONO_ARCH_NAMES = MonogenicNets.__all__
MONOUNET_ARCH_NAMES = MonoUNets.__all__

class CascadedSegModel(torch.nn.Module):
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


def build_model(arch_name, input_channels, num_classes, input_h, input_w, deep_supervision=False):
    if arch_name == "UNext" or arch_name == "UNext_S":
        return archs.__dict__[arch_name](num_classes, input_channels, deep_supervision)
    if arch_name == "TinyUNet":
        return TinyUNet(input_channels, num_classes)
    if arch_name in MONO_ARCH_NAMES:
        return MonogenicNets.__dict__[arch_name](
            input_channels,
            num_classes,
            img_size=(input_h, input_w),
            deep_supervision=deep_supervision,
        )
    if arch_name in MONOUNET_ARCH_NAMES:
        return MonoUNets.__dict__[arch_name](
            in_channels=input_channels,
            num_classes=num_classes,
            img_size=(input_h, input_w),
            deep_supervision=deep_supervision,
        )
    raise NotImplementedError

def visualize_prediction(img, pred, gt=None, dice=None, masd=None, hd95=None, save_path=None):
    """
    Visualize prediction overlaid on input image with optional ground truth and metrics.
    
    Args:
        img: Input image as numpy array (H, W) for grayscale or (H, W, 3) for RGB
        pred: Prediction mask as numpy array (H, W) with values 0 or 1
        gt: Optional ground truth mask as numpy array (H, W) with values 0 or 1
        dice: Optional dice score (0-100)
        masd: Optional MASD score
        hd95: Optional HD95 score
        save_path: Path to save the visualized image
    """
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap='gray')
    
    if gt is not None and gt.max() > 0:
        plt.contour(gt, colors='#90EE90', linewidths=1)
    
    # Overlay prediction (red)
    plt.contour(pred, colors='#FF0000', linewidths=1)
    
    # Add metrics text at bottom left
    metrics_text = []
    if dice is not None:
        metrics_text.append(f'Dice: {dice*100:.2f}%')
    if masd is not None:
        metrics_text.append(f'MASD: {masd:.2f}')
    if hd95 is not None:
        metrics_text.append(f'HD95: {hd95:.2f}')
        
    text_str = '\n'.join(metrics_text)
    img_height, img_width = img.shape[:2]
        
    plt.text(img_width * 0.02, img_height * 0.98, text_str,
            fontsize=10, color='white',
            verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='black'))
    
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        print("Saving prediction to: ", save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', required=True,
                        help='model name')
    parser.add_argument('--train_dataset', default='Dataset073_GE_LE',
                        help='train dataset name')
    parser.add_argument('--train_fold', type=str, required=True,
                        help='train fold index (0-4) or "all" to combine all folds')
    parser.add_argument('--test_dataset', type=str, required=True,
                        help='test dataset name')
    parser.add_argument('--test_split', type=str, required=True,
                        help='test split (Ts or Te)')
    parser.add_argument('--save_preds', type=str2bool, default=False,
                        help='save predictions (True or False)')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='model_best.pth')
    parser.add_argument('--deep_supervision', default=False, type=str2bool,
                        help='use deep supervision (affects model directory path)')
    parser.add_argument('--data_augmentation', default=False, type=str2bool,
                        help='use data augmentation (affects model directory path)')
    parser.add_argument('--overlay', type=str2bool, default=False,
                        help='overlay predictions on images and save visualized images (True or False)')
    parser.add_argument('--largest_component', type=str2bool, default=False,
                        help='keep only the largest connected component (True or False)')
    parser.add_argument('--model_dir_suffix', default='',
                        help='optional suffix for model directory naming')
    args = parser.parse_args()

    return args

def find_largest_component_per_class(segmentation, num_classes):
    output = np.zeros_like(segmentation)
    for i in range(segmentation.shape[0]):
        for cls in range(1, num_classes):  # skip background class 0
            binary_mask = (segmentation[i] == cls).astype(np.uint8)
            labeled_array, num_features = label(binary_mask)
            if num_features == 0:
                continue
            largest_label = max(range(1, num_features + 1), key=lambda x: np.sum(labeled_array == x))
            output[i][labeled_array == largest_label] = cls
    return output

def main():
    args = parse_args()

    arch_name = args.name
    if args.deep_supervision:
        arch_name += 'DS'
    if args.data_augmentation:
        arch_name += 'DA'
    if args.model_dir_suffix:
        arch_name += f"_{args.model_dir_suffix}"
    model_dir = f"models/{arch_name}/{args.train_dataset}/fold_{args.train_fold}"
    with open(f'{model_dir}/config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    print("=> creating model %s" % config['arch'])
    if config.get('cascade_refiner'):
        base_arch = config.get('base_arch') or config['arch']
        base_model = build_model(
            base_arch,
            config['input_channels'],
            config['num_classes'],
            config['input_h'],
            config['input_w'],
            deep_supervision=False,
        )
        if config.get('base_ckpt'):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            base_ckpt = torch.load(config['base_ckpt'], weights_only=False, map_location=device)
            base_state = base_ckpt.get('model_state_dict', base_ckpt.get('state_dict', base_ckpt))
            base_model.load_state_dict(base_state)
        refiner_model = build_model(
            config['arch'],
            config['input_channels'] + 1,
            config['num_classes'],
            config['input_h'],
            config['input_w'],
            deep_supervision=False,
        )
        model = CascadedSegModel(
            base_model=base_model,
            refiner_model=refiner_model,
            mask_threshold=config.get('mask_threshold', 0.5),
            mask_class=config.get('mask_class', 1),
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
        model = build_model(
            config['arch'],
            config['input_channels'],
            config['num_classes'],
            config['input_h'],
            config['input_w'],
            deep_supervision=False,
        )
    
    model = model.cuda()

    ckpt = torch.load(os.path.join(model_dir, args.ckpt), weights_only=False, map_location="cuda")
    state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    model.load_state_dict(state_dict)
    model.eval()

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    if args.test_dataset == args.train_dataset:
        assert args.train_fold != 'all', "TODO: Implement validation for all folds"
        val_dataset = nnUNetDataset(
            dataset_name=args.test_dataset,
            input_channels=config['input_channels'],
            split=args.test_split,
            fold=args.train_fold,
            split_type='val',
            transform=val_transform,
            eval=False)
    else:
        val_dataset = nnUNetDataset(
            dataset_name=args.test_dataset,
            input_channels=config['input_channels'],
            split=args.test_split,
            transform=val_transform,
            eval=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    gput = AverageMeter()
    cput = AverageMeter()

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd95_metric = HausdorffDistanceMetric(include_background=False, reduction="mean", percentile=95)
    surface_dice_metric = SurfaceDistanceMetric(include_background=False, reduction="mean")

    if args.save_preds:
        save_dir = os.path.join(model_dir, 'test', args.test_dataset, 'preds')
        os.makedirs(save_dir, exist_ok=True)
    
    if args.overlay and args.save_preds:
        overlay_dir = os.path.join(model_dir, 'test', args.test_dataset, 'overlays')
        os.makedirs(overlay_dir, exist_ok=True)
    
    # Collect image IDs to match with metrics later
    image_ids = []        
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            model = model.cuda()

            # check if target is 0,1.. if it is 0,255 then convert it to 0,1
            if target.max() == 255:
                target = (target / 255).long()

            # compute output
            output = model(input).cpu()

            output = torch.sigmoid(output)
            output[output>=0.5]=1
            output[output<0.5]=0

            if args.largest_component:
                output = output.cpu().numpy()
                output = find_largest_component_per_class(output, config['num_classes'] + 1)
                output = torch.from_numpy(output)

                target = target.cpu().numpy()
                target = find_largest_component_per_class(target, config['num_classes'] + 1)
                target = torch.from_numpy(target)

            dice = dice_metric(output, center_crop(target, output.shape[2:]))
            hd95 = hd95_metric(output, center_crop(target, output.shape[2:]))
            masd = surface_dice_metric(output, center_crop(target, output.shape[2:]))

            output = output.numpy()
            
            # Collect image IDs
            for i in range(len(output)):
                image_ids.append(meta['img_id'][i])

            if args.save_preds:
                for i in range(len(output)):
                    for c in range(config['num_classes']):
                        if args.overlay:
                            save_path = os.path.join(overlay_dir, meta['img_id'][i] + '.png')
                            visualize_prediction(
                                img=input[i, 0, :, :].cpu().numpy(),
                                gt=target[i, c].cpu().numpy(),
                                pred=output[i, c],
                                dice=dice[i].item(),
                                masd=masd[i].item(),
                                hd95=hd95[i].item(),
                                save_path=save_path
                            )
                        else:
                            cv2.imwrite(os.path.join(save_dir, meta['img_id'][i] + '.png'),
                                    (output[i, c] * 255).astype('uint8'))
            

    torch.cuda.empty_cache()

    if args.save_preds and args.overlay:
        print("Overlay mode is enabled. Metrics will not be calculated.")
        return
    
    # Calculate metrics
    dice_score = dice_metric.aggregate().item() * 100
    dice_std = dice_metric.get_buffer().std().item() * 100
    hd95_score = hd95_metric.aggregate().item()
    hd95_std = hd95_metric.get_buffer().std().item()
    masd_score = surface_dice_metric.aggregate().item()
    masd_std = surface_dice_metric.get_buffer().std().item()

    # Print metrics
    print("\n")
    print(f"Dice: {dice_score:.2f}% ± {dice_std:.2f}%")
    print(f"HD95: {hd95_score:.2f} ± {hd95_std:.2f}")
    print(f"MASD: {masd_score:.2f} ± {masd_std:.2f}")

    # Save aggregated metrics to CSV
    results_csv_path = os.path.join(model_dir, 'test', f'results{"_largest_component" if args.largest_component else ""}.csv')
    os.makedirs(os.path.dirname(results_csv_path), exist_ok=True)
    csv_exists = os.path.exists(results_csv_path) and os.path.getsize(results_csv_path) > 0
    with open(results_csv_path, 'a', newline='') as csvfile:
        fieldnames = ['test_dataset_name', 'dice', 'dice_std', 'hd95', 'hd95_std', 'masd', 'masd_std']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header if file doesn't exist
        if not csv_exists:
            writer.writeheader()
        
        # Write results
        writer.writerow({
            'test_dataset_name': args.test_dataset,
            'dice': f"{dice_score:.2f}",
            'dice_std': f"{dice_std:.2f}",
            'hd95': f"{hd95_score:.2f}",
            'hd95_std': f"{hd95_std:.2f}",
            'masd': f"{masd_score:.2f}",
            'masd_std': f"{masd_std:.2f}"
        })
    
    print(f"Results saved to: {results_csv_path}\n")
    
    # Save image-wise metrics to CSV (extract from metric buffers)
    if args.test_dataset == args.train_dataset:
        image_wise_csv_path = os.path.join(os.path.dirname(model_dir), f'image_wise_results{"_largest_component" if args.largest_component else ""}_{args.test_dataset}.csv')
    else:
        image_wise_csv_path = os.path.join(model_dir, 'test', f'image_wise_results{"_largest_component" if args.largest_component else ""}_{args.test_dataset}.csv')
    
    os.makedirs(os.path.dirname(image_wise_csv_path), exist_ok=True)
    
    # Extract per-image metrics from metric buffers
    dice_buffer = dice_metric.get_buffer() * 100  # Convert to percentage
    hd95_buffer = hd95_metric.get_buffer()
    masd_buffer = surface_dice_metric.get_buffer()
    
    csv_exists = os.path.exists(image_wise_csv_path) and os.path.getsize(image_wise_csv_path) > 0
    with open(image_wise_csv_path, 'a', newline='') as csvfile:
        fieldnames = ['image_id', 'dice', 'hd95', 'masd']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header only if file doesn't exist or is empty
        if not csv_exists:
            writer.writeheader()
        
        # Write image-wise results
        for i, image_id in enumerate(image_ids):
            writer.writerow({
                'image_id': image_id,
                'dice': f"{dice_buffer[i].item():.2f}",
                'hd95': f"{hd95_buffer[i].item():.2f}",
                'masd': f"{masd_buffer[i].item():.2f}"
            })
    
    print(f"Image-wise results saved to: {image_wise_csv_path}\n")


if __name__ == '__main__':
    main()
