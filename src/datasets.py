"""
PyTorch Dataset definitions.

- HAM10000ClassifierDataset: real (+ optionally synthetic) images for the
  DenseNet-121 classifier.
- SingleClassImageDataset: real training images for one class only, used to
  train that class's VAE.
"""
import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import CLASS_TO_IDX, CLF_IMAGE_SIZE, VAE_IMAGE_SIZE

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_classifier_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(CLF_IMAGE_SIZE, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(CLF_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_vae_transforms(train: bool = True):
    """
    train=True: adds flip + small rotation augmentation. Skin lesions have
    no canonical orientation, so this is legitimate augmentation of real
    images (not synthetic data) -- it meaningfully increases effective
    training diversity, which matters most for the smallest classes
    (df, vasc) where the VAE otherwise sees very few unique images.
    train=False (e.g. reconstruction-quality checks): no augmentation.
    """
    ops = [transforms.Resize((VAE_IMAGE_SIZE, VAE_IMAGE_SIZE))]
    if train:
        ops += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(20),
        ]
    ops += [transforms.ToTensor()]  # VAE decoder ends in sigmoid -> expects [0,1]
    return transforms.Compose(ops)


class HAM10000ClassifierDataset(Dataset):
    """
    real_df: dataframe with columns [file_path, dx] for one partition
             (train / val / test), already filtered to that partition.
    synthetic_df: optional dataframe with columns [file_path, dx] for
                  VAE-generated training images (train partition only --
                  never pass this for val/test).
    """

    def __init__(self, real_df: pd.DataFrame, train: bool, synthetic_df: pd.DataFrame = None):
        frames = [real_df[["file_path", "dx"]].copy()]
        if synthetic_df is not None and len(synthetic_df) > 0:
            frames.append(synthetic_df[["file_path", "dx"]].copy())
        self.df = pd.concat(frames, ignore_index=True)
        self.transform = build_classifier_transforms(train=train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["file_path"]).convert("RGB")
        img = self.transform(img)
        label = CLASS_TO_IDX[row["dx"]]
        return img, torch.tensor(label, dtype=torch.long)


class SingleClassImageDataset(Dataset):
    """Training-partition images for exactly one class, for VAE training."""

    def __init__(self, file_paths, train: bool = True):
        self.file_paths = list(file_paths)
        self.transform = build_vae_transforms(train=train)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = Image.open(self.file_paths[idx]).convert("RGB")
        img = self.transform(img)
        return img
