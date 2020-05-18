import os
from functools import partial

import torch
from torchvision import datasets, transforms

from .Celeba import Dataset


__dataset_factory = {
    'Celeba': Dataset,
}

def cycle(iterable):
    while True:
        for i in iterable:
            yield i

# ---------------------- Global settings ----------------------
def getDataLoader(dataset, batch_size, dataset_path, image_size, shuffle=True, augment=True, transparent=False, **kwargs):
    # check ------------------------------------------------------------
    avai_dataset = list(__dataset_factory.keys())
    if dataset not in avai_dataset:
        raise KeyError(
            'Unknown model: {}. Must be one of {}'.format(name, avai_dataset)
        )

    # dataset ------------------------------------------------------------
    image_dataset = __dataset_factory[dataset](dataset_path, image_size=image_size)
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
