from .vision import *
from ..utils.config import configs
from .vision.transform import *
import torchvision
import tensorflow_datasets as tfds
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

__all__ = ['build_dataset']

class CIFAR10CorruptDataset(Dataset):
    """Corrupted CIFAR10C dataset."""

    def __init__(self, corruption_category_name, split=['test'], transform=None):
        """
        Arguments:
            corruption_category_name (string): tfds name for the corruption to be downloaded.
            split (array of strings): dictates whether this is a train or test split
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if split == "train":
            real_split = ["test[:60%]"]
        elif split == "val":
            real_split = ["test[60%:]"]
        else:
            real_split = split

        self.ds = list(tfds.load(configs.data_provider.dataset, split=real_split, shuffle_files=True)[0])
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            raise NotImplementedError

        item = self.ds[idx]
        img = Image.fromarray(np.array(item["image"]))
        label = torch.tensor(item["label"].numpy())

        if self.transform:
            img = self.transform(img)

        return [img, label]

def build_dataset():
    if configs.data_provider.dataset == 'image_folder':
        dataset = ImageFolder(
            root=configs.data_provider.root,
            transforms=ImageTransform(),
        )
    elif configs.data_provider.dataset == 'imagenet':
        dataset = ImageNet(root=configs.data_provider.root,
                       transforms=ImageTransform(), )
    elif configs.data_provider.dataset == 'cifar10':
        dataset = {
            'train': torchvision.datasets.CIFAR10(configs.data_provider.root, train=True,
                                                  transform=ImageTransform()['train'], download=True),
            'val': torchvision.datasets.CIFAR10(configs.data_provider.root, train=False,
                                                transform=ImageTransform()['val'], download=True),
        }
    # modifications for CIFAR10-C
    elif configs.data_provider.dataset.startswith('cifar10_corrupted'):
        dataset = {
            'train': CIFAR10CorruptDataset(configs.data_provider.dataset, split="train", 
                                                    transform=ImageTransform()["train"]),
            'val': CIFAR10CorruptDataset(configs.data_provider.dataset, split="val", 
                                                    transform=ImageTransform()["val"]),
        }
    elif configs.data_provider.dataset == 'cifar100':
        dataset = {
            'train': torchvision.datasets.CIFAR100(configs.data_provider.root, train=True,
                                                   transform=ImageTransform()['train'], download=True),
            'val': torchvision.datasets.CIFAR100(configs.data_provider.root, train=False,
                                                 transform=ImageTransform()['val'], download=True),
        }
    elif configs.data_provider.dataset == 'imagehog':
        dataset = ImageHog(
            root=configs.data_provider.root,
            transforms=ImageTransform(),
        )
    else:
        raise NotImplementedError(configs.data_provider.dataset)

    return dataset
