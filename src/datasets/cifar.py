import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset


def build_dataloaders(config):
    dataset_name = config["data"]["dataset_name"].lower()
    data_dir = config["data"].get("data_dir", "./data")
    image_size = config["data"].get("image_size", 224)
    batch_size = config["training"]["batch_size"]
    num_workers = config["data"].get("num_workers", 0)

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if dataset_name == "cifar100":
        train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
        val_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=val_transform)
    elif dataset_name == "cifar10":
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
        val_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=val_transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    debug_subset = config["data"].get("debug_subset", None)
    if debug_subset is not None:
        train_dataset = Subset(train_dataset, range(min(debug_subset, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(debug_subset, len(val_dataset))))

    use_pin_memory = torch.cuda.is_available()

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    return train_loader, val_loader