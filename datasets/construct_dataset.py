import torch.cuda
from torch.utils.data import DataLoader
import torch.utils.data.sampler as Sampler
import torchvision

from .imagenet_dataset import get_imagenet_train_folder, ImagenetValidationImagefolder
from .cifar_dataset import construct_cifar_dataset

import os


class SubsetSampler(Sampler.Sampler):
    """Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def construct_train_dataloaders(args):
    train_sampler = None
    val_sampler = None
    shuffle = True
    if args.num_classes == 1000 or args.num_classes == 1001:
        train_folder = get_imagenet_train_folder(
            os.path.join(args.directory, 'train'))
        val_folder = ImagenetValidationImagefolder(
            os.path.join(args.directory, 'val'))
    elif args.num_classes == 100:
        train_folder, val_folder, _ = construct_cifar_dataset(args.directory)
        # hold 5k data
        train_sampler = Sampler.SubsetRandomSampler(list(range(0, 45000)))
        val_sampler = SubsetSampler(list(range(45000, 50000)))
        # sampler is mutually exclusive with shuffle
        shuffle = False
    elif args.num_classes == 10:
        train_folder = torchvision.datasets.MNIST(args.directory, train=True, download=True,
                                                  transform=torchvision.transforms.Compose([
                                                      torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize(
                                                          (0.1307,), (0.3081,))
                                                  ]))
        val_folder = torchvision.datasets.MNIST(args.directory, train=True, download=True,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(
                                                        (0.1307,), (0.3081,))
                                                ]))

    if torch.cuda.is_available():
        train_data_loader = DataLoader(train_folder, args.batch_size, shuffle=shuffle, sampler=train_sampler,
                                       num_workers=args.num_workers, pin_memory=True, drop_last=True)
        val_data_loader = DataLoader(val_folder, args.batch_size, shuffle=False, sampler=val_sampler,
                                     num_workers=args.num_workers, pin_memory=True, drop_last=False)
    else:
        train_data_loader = DataLoader(train_folder, args.batch_size, shuffle=shuffle, sampler=train_sampler,
                                       num_workers=args.num_workers, pin_memory=False)
        val_data_loader = DataLoader(val_folder, args.batch_size, shuffle=False, sampler=val_sampler,
                                     num_workers=args.num_workers, pin_memory=False)

    return train_data_loader, val_data_loader


def construct_test_dataloaders(args):
    test_folder = None

    if args.num_classes == 100:
        _, _, test_folder = construct_cifar_dataset(args.directory)
        pin_memory = False
        if torch.cuda.is_available():
            pin_memory = True
        test_dataloader = DataLoader(test_folder, args.batch_size, shuffle=False, num_workers=args.num_workers,
                                     pin_memory=pin_memory)
    elif args.num_classes == 1000:
        _, test_dataloader = construct_train_dataloaders(args)

    elif args.num_classes == 10:
        test_folder = torchvision.datasets.MNIST(args.directory, train=False, download=True,
                                                 transform=torchvision.transforms.Compose([
                                                     torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Normalize((0.1307,),
                                                                                      (0.3081,))
                                                 ]))
        if torch.cuda.is_available():
            pin_memory = True
        test_dataloader = DataLoader(test_folder, args.batch_size, shuffle=True, num_workers=args.num_workers,
                                     pin_memory=pin_memory)

    return test_dataloader
