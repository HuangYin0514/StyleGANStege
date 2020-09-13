import os
from functools import partial

import torch
from torchvision import datasets, transforms

from .Celeba import Dataset
from .data_tools import cycle


__dataset_factory = {
    'Celeba': Dataset,
}


# ---------------------- Global settings ----------------------
def getDataLoader(dataset, batch_size, dataset_path, shuffle=True, augment=True, transparent=False):
    # check ------------------------------------------------------------
    avai_dataset = list(__dataset_factory.keys())
    if dataset not in avai_dataset:
        raise KeyError(
            'Unknown model: {}. Must be one of {}'.format(name, avai_dataset)
        )

    # transform ------------------------------------------------------------
    def convert_rgb_to_transparent(image):
        if image.mode == 'RGB':
            return image.convert('RGBA')
        return image

    def convert_transparent_to_rgb(image):
        if image.mode == 'RGBA':
            return image.convert('RGB')
        return image

    def expand_greyscale(num_channels):
        def inner(tensor):
            return tensor.expand(num_channels, -1, -1)
        return inner

    def resize_to_minimum_size(min_size, image):
        if max(*image.size) < min_size:
            return torchvision.transforms.functional.resize(image, min_size)
        return image

    img_size=32
    convert_image_fn = convert_transparent_to_rgb if not transparent else convert_rgb_to_transparent
    num_channels = 3 if not transparent else 4
    transform = transforms.Compose([
        transforms.Lambda(convert_image_fn),
        transforms.Lambda(partial(resize_to_minimum_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Lambda(expand_greyscale(num_channels))
    ])

    # dataset ------------------------------------------------------------
    image_dataset = __dataset_factory[dataset](dataset_path, transform=transform)
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4,
                                             drop_last=True, pin_memory=True)
    dataloader = cycle(dataloader)
    return dataloader


def check_data(images, img_save_path):
    """
    check data of image
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torchvision.utils as vutils

    # [weight, hight]
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(np.transpose(vutils.make_grid(images,
                                             padding=2,
                                             normalize=True),
                            (1, 2, 0)))
    plt.savefig(img_save_path)
