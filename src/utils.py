# ──────────────────────────────────────────────────────────────────────
# src/utils.py
# Helper functions:  model loading, dataloaders, seeds
# ──────────────────────────────────────────────────────────────────────
import random
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# --------------------- reproducibility --------------------------------
def seed_everything(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------ datasets ------------------------------------
_DATA_ROOT = os.getenv("DATA_ROOT", "./data")


def _default_transform(img_size=224):
    return transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ]
    )


def get_dataset(name: str, split: str):
    """
    Minimal examples using torchvision datasets.  Extend with your own
    datasets as needed. `split` is 'train' or 'test'.
    """
    root = f"{_DATA_ROOT}/{name}"
    if name in ("cifar10", "cifar100"):
        cls = datasets.CIFAR10 if name == "cifar10" else datasets.CIFAR100
        train = split == "train"
        return cls(root, train=train, download=True, transform=_default_transform(32))

    if name in ("svhn",):
        train = split == "train"
        return datasets.SVHN(
            root,
            split="train" if train else "test",
            download=True,
            transform=_default_transform(32),
        )

    #  Add custom dataset classes here  -------------------------------
    raise ValueError(f"Dataset '{name}' not implemented.")


def get_dataloader(name: str, split: str, batch_size: int = 128, num_workers: int = 4):
    ds = get_dataset(name, split)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


# ------------------------- models -------------------------------------
def get_model(name: str):
    """Loads a model and moves it to CUDA if available."""
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    elif name == "vit_b_16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
    else:
        raise ValueError(f"Model '{name}' not supported.")

    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    return model
