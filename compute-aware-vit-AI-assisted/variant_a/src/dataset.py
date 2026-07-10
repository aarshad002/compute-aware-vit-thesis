"""CIFAR-100 dataset loading with 224x224 resizing for ViT models."""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transforms(image_size: int, split: str) -> transforms.Compose:
    """Return train or val transforms with ImageNet normalisation."""
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    if split == "train":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(image_size, padding=int(image_size * 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


def get_dataloaders(
    data_dir: str,
    image_size: int,
    batch_size: int,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """Return train and val DataLoaders for CIFAR-100.

    Downloads the dataset to data_dir if not already present.
    """
    train_dataset = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=get_transforms(image_size, "train"),
    )
    val_dataset = datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=True,
        transform=get_transforms(image_size, "val"),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
