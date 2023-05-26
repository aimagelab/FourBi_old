import math
import os

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms, functional
from torchvision.utils import make_grid

import data.CustomTransforms as CustomTransform


def get_path(root: str, paths: list, index: int):
    assert index < len(paths)
    return os.path.join(root, paths[index])


def get_transform(transform_variant: str, output_size: int):
    transform_list = []
    if transform_variant == 'gaussian':
        transform_list.append(CustomTransform.GaussianBlur(kernel_size=(3, 5), sigma=(0.3, 1.5)))
    elif transform_variant == 'equalize_contrast':
        transform_list.append(CustomTransform.RandomEqualize())

    if transform_variant != 'no_color_jitter':
        transform_list.append(CustomTransform.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5, saturation=0.5))
    transform_list.append(CustomTransform.RandomRotation((0, 360)))
    transform_list.append(CustomTransform.RandomHorizontalFlip())
    transform_list.append(CustomTransform.RandomVerticalFlip())
    transform_list.append(CustomTransform.RandomCrop(output_size))
    transform_list.append(CustomTransform.ToTensor())

    if transform_variant == 'threshold_mask':
        transform_list.append(CustomTransform.ThresholdMask(threshold=0.9))

    if transform_variant == 'latin':
        transform_list = [
            CustomTransform.RandomRotation((-10, 10)),
            CustomTransform.RandomCrop(output_size),
            CustomTransform.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5, saturation=0.5),
            CustomTransform.ToTensor(),
        ]

    transform = transforms.Compose(transform_list)
    return transform


def get_patches(image_source: Image, patch_size: int, stride: int):
    image = np.asarray(image_source)
    image_patches = []

    h = ((image.shape[0] // patch_size) + 1) * patch_size
    w = ((image.shape[1] // patch_size) + 1) * patch_size

    padding_image = np.ones((h, w, 3)) if len(image.shape) == 3 else np.ones((h, w))
    padding_image = padding_image * 255.0
    padding_image[:image.shape[0], :image.shape[1]] = image

    for j in range(0, w - patch_size + 1, stride):
        for i in range(0, h - patch_size + 1, stride):
            image_patches.append(padding_image[i:i + patch_size, j:j + patch_size])

    num_rows = math.floor((padding_image.shape[0] - patch_size) / stride) + 1
    num_cols = math.floor((padding_image.shape[1] - patch_size) / stride) + 1

    return np.array(image_patches), num_rows, num_cols


def reconstruct_ground_truth(patches, original, num_rows, config):
    channels = 1
    batch_size = 1
    patch_size = config['test_patch_size']
    stride = config['test_stride']

    _, _, height, width = original.shape
    width, height = original.shape[-1], original.shape[-2]
    tmp_patches = patches.view(batch_size, channels, -1, num_rows, patch_size, patch_size)
    patch_width, patch_height = tmp_patches.shape[-1], tmp_patches.shape[-2]
    tensor_padded_width = patch_width * tmp_patches.shape[-3] - (patch_width - stride) * (tmp_patches.shape[-3] - 1)
    tensor_padded_height = patch_height * tmp_patches.shape[-4] - (patch_height - stride) * (tmp_patches.shape[-4] - 1)

    padding_up = 0
    padding_left = 0

    if stride == (patch_size // 2):
        patches = tmp_patches

        x_steps = [x + (stride // 2) for x in range(0, tensor_padded_height, stride)]
        x_steps[0], x_steps[-1] = 0, tensor_padded_height
        y_steps = [y + (stride // 2) for y in range(0, tensor_padded_width, stride)]
        y_steps[0], y_steps[-1] = 0, tensor_padded_width

        canvas = torch.zeros(batch_size, channels, tensor_padded_height, tensor_padded_width)
        for j in range(len(x_steps) - 1):
            for i in range(len(y_steps) - 1):
                patch = patches[0, :, j, i, :, :]
                x1_abs, x2_abs = x_steps[j], x_steps[j + 1]
                y1_abs, y2_abs = y_steps[i], y_steps[i + 1]
                x1_rel, x2_rel = x1_abs - (j * stride), x2_abs - (j * stride)
                y1_rel, y2_rel = y1_abs - (i * stride), y2_abs - (i * stride)
                canvas[0, :, x1_abs:x2_abs, y1_abs:y2_abs] = patch[:, x1_rel:x2_rel, y1_rel:y2_rel]
        canvas = functional.crop(canvas, top=padding_up, left=padding_left, height=height, width=width)
        canvas = canvas.to(original.device)
    else:
        tensor = make_grid(patches, nrow=num_rows, padding=0, value_range=(0, 1))
        tensor = functional.rgb_to_grayscale(tensor)
        _, _, height, width = original.shape
        canvas = functional.crop(tensor, top=padding_up, left=padding_left, height=height, width=width)
        canvas = canvas.unsqueeze(0)

    return canvas
