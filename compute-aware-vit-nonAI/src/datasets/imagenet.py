import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label, idx


def build_imagenet_val_loader(
    root="data/imagenet/val",
    image_size=224,
    batch_size=32,
    num_workers=4,
    debug_subset=None,
):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = datasets.ImageFolder(root=root, transform=transform)
    dataset = IndexedDataset(dataset)

    if debug_subset is not None:
        dataset = Subset(dataset, range(min(debug_subset, len(dataset))))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return loader
