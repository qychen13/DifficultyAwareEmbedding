import torchvision.transforms as Transforms
import torch.utils.data as data
from torchvision.datasets.folder import ImageFolder, find_classes, IMG_EXTENSIONS

import os
import PIL.Image as Image
import json

# default_transforms are designed for validating ImageNet data
train_transforms = Transforms.Compose([Transforms.Resize(256),
                                       Transforms.RandomCrop(224),
                                       Transforms.RandomHorizontalFlip(),
                                       Transforms.ToTensor(),
                                       Transforms.Normalize(
                                           mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
val_transforms = Transforms.Compose([Transforms.Resize(256),
                                     Transforms.CenterCrop(224),
                                     Transforms.ToTensor(),
                                     Transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def get_label2index_index2classname(savefile=None):
    """
    This is a helper function for the ImageNet classification model.
    The model index is arranged w.r.t. sorted parent folder names.
    The label is the annotation provided by ImageNet official website.
    """
    fdir = os.path.dirname(__file__)
    with open(os.path.join(fdir, 'map_clsloc.txt')) as f:
        lines = [line.split() for line in sorted(list(f))]
    label2index = {line[1]: i for i, line in enumerate(lines)}
    index2classname = {i: line[2] for i, line in enumerate(lines)}

    if savefile is not None:
        with open(savefile, 'w') as f:
            json.dump((label2index, index2classname), f)

    return label2index, index2classname


def default_loader(path):
    return Image.open(path).convert('RGB')


def get_imagenet_train_folder(path):
    return ImageFolder(path, transform=train_transforms, loader=default_loader)


class ImagenetValidationImagefolder(data.Dataset):
    def __init__(self, dir, transform=val_transforms, loader=default_loader):
        # load validation file names and labels
        cdir = os.path.dirname(__file__)
        f = open(os.path.join(cdir, 'imagenet_validation_pairs.json'), 'r')
        imgs = json.load(f)
        f.close()

        f = open(os.path.join(
            cdir, 'imagenet_maps_label2index_index2classname.json'), 'r')
        label2index, index2classname = json.load(f)
        f.close()

        self.dir = dir
        self.imgs = imgs
        self.transform = transform
        self.loader = loader
        self.label2index = label2index
        self.index2classname = index2classname

    def get_classname(self, index):
        return self.index2classname[index]

    def __getitem__(self, index):
        # load image file
        filename, label = self.imgs[index]
        path = os.path.join(self.dir, filename)
        img = self.loader(path)

        # transform image file
        img = self.transform(img)

        # transform image label to index
        target_indx = self.label2index[str(label)]

        return img, target_indx

    def __len__(self):
        return len(self.imgs)
