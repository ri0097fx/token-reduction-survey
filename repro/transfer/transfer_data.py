import glob
import os

import torch
from PIL import Image
from timm.data import create_transform
from torchvision import datasets, transforms

DATASETS = os.environ["VTR_DATASETS"]
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
FOLDERS = {"cub200": "CUB_200_2011/images", "cars": "stanford-cars/car_data/car_data", "pets": "oxford-iiit-pet/images",
           "flowers": "flowers_dataset/102_flower/flowers", "indoor67": "indoor67/indoorCVPR_09"}
NUM_CLASSES = {"cub200": 200, "cars": 196, "pets": 37, "flowers": 102, "indoor67": 67, "cifar10": 10, "cifar100": 100}
NAMES = ["cub200", "cars", "pets", "flowers", "indoor67", "cifar10", "cifar100"]


def train_transform(aa="rand-m9-mstd0.5-inc1", reprob=0.25):
    return create_transform(input_size=224, is_training=True, color_jitter=0.4, auto_augment=aa or None,
                            interpolation="bicubic", re_prob=reprob, re_mode="pixel", re_count=1, mean=MEAN, std=STD)


def eval_transform():
    return transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                               transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return self.transform(Image.open(path).convert("RGB")), label


def flowers_official(train, transform):
    from scipy.io import loadmat
    root = os.path.join(DATASETS, "flowers_dataset")
    ids = loadmat(os.path.join(root, "official_split", "setid.mat"))
    labels = loadmat(os.path.join(root, "official_split", "imagelabels.mat"))["labels"][0]
    paths = {int(os.path.basename(p)[6:11]): p for p in glob.glob(os.path.join(DATASETS, FOLDERS["flowers"], "*", "*", "*.jpg"))}
    sel = [int(i) for i in (list(ids["trnid"][0]) + list(ids["valid"][0]) if train else list(ids["tstid"][0]))]
    items = [(paths[i], int(labels[i - 1]) - 1) for i in sel]
    assert len(items) == (2040 if train else 6149) and len({l for _, l in items}) == 102
    return ListDataset(items, transform)


def get_dataset(name, train, transform):
    if name == "flowers":
        return flowers_official(train, transform)
    if name == "cifar10":
        return datasets.CIFAR10(DATASETS, train=train, transform=transform, download=False)
    if name == "cifar100":
        return datasets.CIFAR100(DATASETS, train=train, transform=transform, download=False)
    root = os.path.join(DATASETS, FOLDERS[name], "train" if train else "test")
    ds = datasets.ImageFolder(root, transform=transform)
    assert len(ds.classes) == NUM_CLASSES[name], (name, len(ds.classes))
    return ds
