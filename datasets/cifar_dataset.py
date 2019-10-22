import torchvision as tv
import torch

import json
import os
import numpy
from PIL import ImageFont, ImageDraw


def construct_cifar_dataset(data_root):
    # Data transforms
    mean = [0.5071, 0.4867, 0.4408]
    stdv = [0.2675, 0.2565, 0.2761]
    train_transforms = tv.transforms.Compose([
        tv.transforms.RandomCrop(32, padding=4),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=mean, std=stdv),
    ])
    test_transforms = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=mean, std=stdv),
    ])

    train_set = tv.datasets.CIFAR100(
        data_root, train=True, transform=train_transforms, download=True)
    valid_set = tv.datasets.CIFAR100(
        data_root, train=True, transform=test_transforms, download=False)
    test_set = tv.datasets.CIFAR100(
        data_root, train=False, transform=test_transforms, download=False)

    return train_set, valid_set, test_set
