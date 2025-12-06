import argparse
import os
from glob import glob

import cv2
import csv
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm

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
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    model_dir = f"models/{args.train_dataset}/{args.name}/fold_{args.train_fold}"
    with open(f'{model_dir}/config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    print("=> creating model %s" % config['arch'])
    if config['arch'] == "UNext" or config['arch'] == "UNext_S":
        model = archs.__dict__[config['arch']](config['num_classes'],
                                               config['input_channels'],
                                               config['deep_supervision'])
    elif config['arch'] == "TinyUNet":
        model = TinyUNet(config['input_channels'],
                         config['num_classes'])

    model = model.cuda()

    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth')))
    model.eval()

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    if args.test_dataset == args.train_dataset:
        assert args.train_fold != 'all', "TODO: Implement validation for all folds"
        val_dataset = nnUNetDataset(
            dataset_name=args.test_dataset,
            split=args.test_split,
            fold=args.train_fold,
            split_type='val',
            transform=val_transform,
            eval=False)
    else:
        val_dataset = nnUNetDataset(
            dataset_name=args.test_dataset,
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
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            model = model.cuda()
            # compute output
            output = model(input).cpu()

            iou,dice = iou_score(output, target)
            iou_avg_meter.update(iou, input.size(0))
            dice_avg_meter.update(dice, input.size(0))

            output = torch.sigmoid(output)
            output[output>=0.5]=1
            output[output<0.5]=0

            dice_metric(output, target)
            hd95_metric(output, target)
            surface_dice_metric(output, target)

            output = output.numpy()

            if args.save_preds:
                for i in range(len(output)):
                    for c in range(config['num_classes']):
                        cv2.imwrite(os.path.join(save_dir, meta['img_id'][i] + '.png'),
                                (output[i, c] * 255).astype('uint8'))

    torch.cuda.empty_cache()
    
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

    # Save to CSV
    results_csv_path = os.path.join(model_dir, 'test', 'results.csv')
    os.makedirs(os.path.dirname(results_csv_path), exist_ok=True)
    csv_exists = os.path.exists(results_csv_path)
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




if __name__ == '__main__':
    main()
