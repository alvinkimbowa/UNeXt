import os
import json

import cv2
import numpy as np
import torch
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        
        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))

        mask = []
        for i in range(self.num_classes):
            mask.append(cv2.imread(os.path.join(self.mask_dir, str(i),
                        img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        mask = np.dstack(mask)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)
        
        return img, mask, {'img_id': img_id}


class nnUNetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, split, fold, split_type='train', num_classes=1, transform=None):
        nnunet_raw = os.environ['nnUNet_raw']
        nnunet_preprocessed = os.environ['nnUNet_preprocessed']
        
        with open(os.path.join(f'{nnunet_preprocessed}/{dataset_name}/dataset.json'), 'r') as f:
            dataset_info = json.load(f)
        img_ext = dataset_info['file_ending']
        
        with open(os.path.join(f'{nnunet_preprocessed}/{dataset_name}/splits_final.json'), 'r') as f:
            splits = json.load(f)
        
        if fold == 'all':
            img_ids = []
            for split_dict in splits:
                img_ids.extend(split_dict[split_type])
            img_ids = list(set(img_ids))
        else:
            img_ids = splits[int(fold)][split_type]
        
        self.img_ids = img_ids
        self.img_dir = os.path.join(f'{nnunet_raw}/{dataset_name}/images{split}')
        self.label_dir = os.path.join(f'{nnunet_raw}/{dataset_name}/labels{split}')
        self.img_ext = img_ext
        self.num_classes = num_classes
        self.transform = transform
        self.dataset_name = dataset_name
    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_filename = f'{self.img_dir}/{img_id}_0000{self.img_ext}'
        label_filename = f'{self.label_dir}/{img_id}{self.img_ext}'
        
        img = cv2.imread(os.path.join(self.img_dir, img_filename))
        mask = cv2.imread(os.path.join(self.label_dir, label_filename), cv2.IMREAD_GRAYSCALE)[..., None]
        
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32')
        mask = mask.transpose(2, 0, 1)   
        
        return img, mask, {'img_id': img_id}
