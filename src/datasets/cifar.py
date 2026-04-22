import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
import json

class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label, idx

class BudgetLabeledDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, budget_label_path):
        self.dataset = dataset

        with open(budget_label_path, "r") as f:
            raw_labels = json.load(f)

        self.budget_labels = {int(k): int(v) for k, v in raw_labels.items()}

        # Keep only samples whose original index exists in the label file
        self.valid_indices = sorted(self.budget_labels.keys())

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        original_idx = self.valid_indices[idx]
        image, label, _ = self.dataset[original_idx]
        budget_target = self.budget_labels[original_idx]
        return image, label, original_idx, budget_target
    
def build_dataloaders(config):
    dataset_name = config["data"]["dataset_name"].lower()
    data_dir     = config["data"].get("data_dir", "./data")
    image_size   = config["data"].get("image_size", 224)
    batch_size   = config["training"]["batch_size"]
    num_workers  = config["data"].get("num_workers", 0)

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
        train_dataset = datasets.CIFAR100(root=data_dir, train=True,  download=True, transform=train_transform)
        val_dataset   = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=val_transform)
    elif dataset_name == "cifar10":
        train_dataset = datasets.CIFAR10(root=data_dir, train=True,  download=True, transform=train_transform)
        val_dataset   = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=val_transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_dataset = IndexedDataset(train_dataset)
    val_dataset   = IndexedDataset(val_dataset)

    controller_cfg      = config.get("controller", {})
    supervised_training = controller_cfg.get("supervised_training", False)
    use_sampler         = False

    if supervised_training:
        train_budget_label_path = controller_cfg["budget_label_path"]
        val_budget_label_path   = controller_cfg.get("val_budget_label_path")

        train_dataset = BudgetLabeledDataset(train_dataset, train_budget_label_path)

        if val_budget_label_path is not None:
            val_dataset = BudgetLabeledDataset(val_dataset, val_budget_label_path)

        use_sampler = True

    debug_subset = config["data"].get("debug_subset", None)
    if debug_subset is not None:
        train_dataset = Subset(train_dataset, range(min(debug_subset, len(train_dataset))))
        val_dataset   = Subset(val_dataset,   range(min(debug_subset, len(val_dataset))))
        use_sampler   = False  # sampler not compatible with Subset here

    use_pin_memory = torch.cuda.is_available()

    # ── build train loader ───────────────────────────────────────────
    if use_sampler:
        # WeightedRandomSampler: each budget appears equally in every batch
        budget_targets = train_dataset.get_budget_targets()
        class_counts   = Counter(budget_targets)
        total          = len(budget_targets)
        num_classes    = len(class_counts)

        # weight per class = total / (num_classes × count)
        class_weight = {
            c: total / (num_classes * count)
            for c, count in class_counts.items()
        }
        sample_weights = [class_weight[t] for t in budget_targets]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

        print("WeightedRandomSampler enabled — budget class counts:", dict(sorted(class_counts.items())))
        print("Effective class weights:", {k: round(v, 2) for k, v in sorted(class_weight.items())})

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,        # replaces shuffle=True
            num_workers=num_workers,
            pin_memory=use_pin_memory,
        )
    else:
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