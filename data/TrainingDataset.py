import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path


class TrainingDataset(Dataset):

    def __init__(self, data_path, split_size=256, patch_size=384, transform=None, load_data=True, merge_image=True):
        super(TrainingDataset, self).__init__()
        self.imgs = list(Path(data_path).rglob(f'imgs_{patch_size}/*'))
        self.gt_imgs = [img_path.parent.parent / ('gt_' + img_path.parent.name) / img_path.name for img_path in self.imgs]

        self.load_data = load_data
        if self.load_data:
            self.imgs = [Image.open(img_path).convert("RGB") for img_path in self.imgs]
            self.gt_imgs = [Image.open(gt_img_path).convert("L") for gt_img_path in self.gt_imgs]

        self.split_size = split_size
        self.transform = transform
        self.merge_image = merge_image

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index, merge_image=None):
        if self.merge_image and merge_image is None:
            merge_image = self.merge_image

        if self.load_data:
            sample = self.imgs[index]
            gt_sample = self.gt_imgs[index]
        else:
            sample = Image.open(self.imgs[index]).convert("RGB")
            gt_sample = Image.open(self.gt_imgs[index]).convert("L")

        if self.transform:
            transform = self.transform({'image': sample, 'gt': gt_sample})
            sample = transform['image']
            gt_sample = transform['gt']

        # Merge two images
        if merge_image:
            random_index = random.randint(0, len(self.imgs) - 1)
            random_sample, random_gt_sample = self.__getitem__(index=random_index, merge_image=False)

            sample = np.minimum(sample, random_sample)
            gt_sample = np.minimum(gt_sample, random_gt_sample)

        gt_sample = gt_sample.float()
        return sample, gt_sample
