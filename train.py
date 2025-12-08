import argparse
import os
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
from TinyUNet import TinyUNet
import monounet.MonogenicNets

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')
MONO_ARCH_NAMES = monounet.MonogenicNets.__all__


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
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=256, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=256, type=int,
                        help='image height')
    
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

    parser.add_argument('--num_workers', default=4, type=int)

    config = parser.parse_args()

    return config

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


# args = parser.parse_args()
def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.cuda()
        target = target.cuda()

        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, F.interpolate(target, output.size()[2:], mode='nearest'))
            loss /= len(outputs)
            iou,dice = iou_score(outputs[-1], target)
        else:
            output = model(input)
            loss = criterion(output, target)
            iou,dice = iou_score(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        avg_meters['dice'].update(dice, input.size(0))

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
                    loss += criterion(output, F.interpolate(target, output.size()[2:], mode='nearest'))
                loss /= len(outputs)
                iou,dice = iou_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou,dice = iou_score(output, target)

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
    model_dir = f"models/{arch_name}/{config['dataset']}/fold_{fold_str}"
    os.makedirs(model_dir, exist_ok=True)
    
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

    # create model
    if config['arch'] == "UNext" or config['arch'] == "UNext_S":
        model = archs.__dict__[config['arch']](config['num_classes'],
                                               config['input_channels'],
                                               config['deep_supervision'])
    elif config['arch'] == "TinyUNet":
        model = TinyUNet(config['input_channels'],
                         config['num_classes'])
    elif config['arch'] in MONO_ARCH_NAMES:
        model = monounet.MonogenicNets.__dict__[config['arch']](config['input_channels'],
                                                                config['num_classes'],
                                                                img_size=(config['input_h'], config['input_w']),
                                                                deep_supervision=config['deep_supervision'])
    else:
        raise NotImplementedError

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
    
    if config['arch'] == "TinyUNet":
        train_transform = Compose([
            Resize(config['input_h'], config['input_w']),
            transforms.Normalize(),
        ])
    elif config['arch'] in MONO_ARCH_NAMES:
        # For XTiny models, add scaling and rotation if data_augmentation is enabled
        if config['data_augmentation'] and config['arch'].startswith('XTiny'):
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
    else:
        train_transform = Compose([
            RandomRotate90(),
            Flip(),
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

    best_iou = 0
    trigger = 0
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
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

        pd.DataFrame(log).to_csv(f'{model_dir}/log.csv', index=False)
        
        # Plot training curves
        plot_training_curves(log, f'{model_dir}/loss_curves.png')

        trigger += 1

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), f'{model_dir}/model_best.pth')
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()
    
    torch.save(model.state_dict(), f'{model_dir}/model_final.pth')
    print("=> saved final model")
    print("Training completed")


if __name__ == '__main__':
    main()
