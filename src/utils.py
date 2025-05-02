# ──────────────────────────────────────────────────────────────────────
# src/utils.py
# Helper functions:  model loading, dataloaders, seeds
# ──────────────────────────────────────────────────────────────────────
import random

# import os
import torch

# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms, models
import glob
from pathlib import Path
from typing import Sequence, Union, Optional

import pandas as pd


# --------------------- reproducibility --------------------------------
def seed_everything(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------------------------------------------
#  Benchmark results I/O
# ------------------------------------------------------------------
def load_benchmark(
    path_or_glob: Union[str, Path] = "results/*.parquet",
    *,
    flatten_json: Optional[Sequence[str]] = ("init_params", "fit_params"),
) -> pd.DataFrame:
    """Load all benchmark Parquet files into a single tidy DataFrame.

    Args:
        path_or_glob: Directory or glob pattern that locates the Parquet
            files (e.g. `"results/*.parquet"` or `"my_runs/"`).
        flatten_json: Column names whose values are JSON/dict objects.
            Each such column is expanded into real columns prefixed with
            `"<col>."`. Use `None` to disable flattening.

    Returns:
        A concatenated :class:`~pandas.DataFrame` containing one row per
        *ID dataset x OOD dataset* pair, plus any expanded hyper-parameter
        columns.

    Raises:
        FileNotFoundError: If no Parquet files match *path_or_glob*.
    """
    # resolve files
    if "*" in str(path_or_glob):
        parquet_files = glob.glob(str(path_or_glob))
    else:
        parquet_files = list(Path(path_or_glob).glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found at {path_or_glob}")

    # read & concat
    df = pd.concat(map(pd.read_parquet, parquet_files), ignore_index=True)

    # expand dict‑style columns
    if flatten_json:
        for col in flatten_json:
            if col in df.columns:
                df = df.join(
                    df[col]
                    .apply(lambda x: pd.Series(x if isinstance(x, dict) else {}))
                    .add_prefix(f"{col}.")
                )

    # specify react, scale, ash methods
    df["method"] = df.apply(
        lambda x: (
            x["method"] + "+react"
            if x["init_params.use_react"] is True
            else x["method"]
        ),
        axis=1,
    )
    df["method"] = df.apply(
        lambda x: (
            x["method"] + "+scale"
            if x["init_params.use_scale"] is True
            else x["method"]
        ),
        axis=1,
    )
    df["method"] = df.apply(
        lambda x: (
            x["method"] + "+ash" if x["init_params.use_ash"] is True else x["method"]
        ),
        axis=1,
    )
    return df


# # ------------------------ datasets ------------------------------------
# _DATA_ROOT = os.getenv("DATA_ROOT", "./data")


# def _default_transform(img_size=224):
#     return transforms.Compose(
#         [
#             transforms.Resize(img_size),
#             transforms.CenterCrop(img_size),
#             transforms.ToTensor(),
#         ]
#     )


# def get_dataset(name: str, split: str):
#     """
#     Minimal examples using torchvision datasets.  Extend with your own
#     datasets as needed. `split` is 'train' or 'test'.
#     """
#     root = f"{_DATA_ROOT}/{name}"
#     if name in ("cifar10", "cifar100"):
#         cls = datasets.CIFAR10 if name == "cifar10" else datasets.CIFAR100
#         train = split == "train"
#         return cls(root, train=train, download=True, transform=_default_transform(32))

#     if name in ("svhn",):
#         train = split == "train"
#         return datasets.SVHN(
#             root,
#             split="train" if train else "test",
#             download=True,
#             transform=_default_transform(32),
#         )

#     #  Add custom dataset classes here  -------------------------------
#     raise ValueError(f"Dataset '{name}' not implemented.")


# def get_dataloader(name: str, split: str, batch_size: int = 128, num_workers: int = 4):
#     ds = get_dataset(name, split)
#     return DataLoader(
#         ds,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=True,
#     )


# # ------------------------- models -------------------------------------
# def get_model(name: str):
#     """Loads a model and moves it to CUDA if available."""
#     if name == "resnet18":
#         model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
#     elif name == "resnet50":
#         model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
#     elif name == "vit_b_16":
#         model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
#     else:
#         raise ValueError(f"Model '{name}' not supported.")

#     model.eval()
#     if torch.cuda.is_available():
#         model.cuda()
#     return model
