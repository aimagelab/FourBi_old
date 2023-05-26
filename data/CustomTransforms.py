import random

from torchvision import transforms
from torchvision.transforms import functional


class ToTensor(transforms.ToTensor):

    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']
        image = super().__call__(image)
        gt = super().__call__(gt)
        return {'image': image, 'gt': gt}


class ThresholdMask:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']
        gt = (gt > self.threshold).float()
        return {'image': image, 'gt': gt}


class ColorJitter(transforms.ColorJitter):

    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']
        image = super().__call__(image)
        return {'image': image, 'gt': gt}


class GaussianBlur(transforms.GaussianBlur):
    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']
        image = functional.gaussian_blur(img=image, kernel_size=self.kernel_size, sigma=self.sigma)
        # gt = super().forward(gt)
        return {'image': image, 'gt': gt}


class RandomCrop(transforms.RandomCrop):

    def __init__(self, size):
        super(RandomCrop, self).__init__(size=size)
        self.size = size

    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']
        i, j, h, w = self.get_params(image, output_size=(self.size, self.size))
        image = functional.crop(image, i, j, h, w)
        gt = functional.crop(gt, i, j, h, w)
        return {'image': image, 'gt': gt}


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):

    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']
        if random.random() < self.p:
            image = functional.hflip(image)
            gt = functional.hflip(gt)
        return {'image': image, 'gt': gt}


class RandomVerticalFlip(transforms.RandomVerticalFlip):

    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']
        if random.random() < self.p:
            image = functional.vflip(image)
            gt = functional.vflip(gt)
        return {'image': image, 'gt': gt}


class RandomRotation(transforms.RandomRotation):

    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']
        angle = self.get_params(self.degrees)

        image = functional.rotate(image, angle, fill=[255, 255, 255])

        gt = functional.invert(gt)
        gt = functional.rotate(gt, angle)
        gt = functional.invert(gt)
        return {'image': image, 'gt': gt}


class CenterCrop(transforms.CenterCrop):

    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']
        image = super().forward(image)
        gt = super().forward(gt)
        return {'image': image, 'gt': gt}


class RandomAutoContrast(transforms.RandomAutocontrast):
    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']
        if random.random() < self.p:
            image = functional.autocontrast(image)
            gt = functional.autocontrast(gt)
        return {'image': image, 'gt': gt}


class RandomAdjustSharpness(transforms.RandomAdjustSharpness):
    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']
        if random.random() < self.p:
            image = functional.adjust_contrast(image, self.sharpness_factor)
            gt = functional.adjust_contrast(gt, self.sharpness_factor)
        return {'image': image, 'gt': gt}


class RandomEqualize(transforms.RandomEqualize):
    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']
        if random.random() < self.p:
            image = functional.equalize(image)
            gt = functional.equalize(gt)
        return {'image': image, 'gt': gt}
