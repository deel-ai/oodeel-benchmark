# ──────────────────────────────────────────────────────────────────────
# src/utils.py
# Helper functions:  seeds, CUDA memory tracking, benchmark results loading
# ──────────────────────────────────────────────────────────────────────
import random

# import os
import gc
import time
from contextlib import contextmanager
import glob
from pathlib import Path
from typing import Sequence, Union, Optional

import torch

# from torch.utils.data import DataLoader
from torchvision import models
from .openood_networks.utils import get_network

import pandas as pd


# --------------------- reproducibility --------------------------------
def seed_everything(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------- CUDA memory tracking ----------------------------
def _to_mib(b):
    return b / (1024**2)  # bytes → MiB


@contextmanager
def cuda_tracker(console, tag="", enabled: bool = True):
    """Context manager to track CUDA memory usage.
    Args:
        console: Console object for logging.
        tag: Tag to identify the context.
        enabled: Whether to track memory usage or not.
    """
    if enabled:
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.memory_allocated()
        t0 = time.time()
        yield
        end = torch.cuda.memory_allocated()
        peak = torch.cuda.max_memory_allocated()
        dt = time.time() - t0
        console.log(
            (
                f"[mem] {tag:20s} "
                f"Δ={_to_mib(end-start):7.1f} MiB  "
                f"peak={_to_mib(peak):7.1f} MiB  "
                f"time={dt:5.1f}s"
            ),
            markup=False,
        )
        gc.collect()
        torch.cuda.empty_cache()
    else:
        yield


# ------------------- benchmark results loading --------------------------
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
    # df["method"] = df.apply(
    #     lambda x: (
    #         x["method"] + "+react"
    #         if x["init_params.use_react"] is True
    #         else x["method"]
    #     ),
    #     axis=1,
    # )
    # df["method"] = df.apply(
    #     lambda x: (
    #         x["method"] + "+scale"
    #         if x["init_params.use_scale"] is True
    #         else x["method"]
    #     ),
    #     axis=1,
    # )
    # df["method"] = df.apply(
    #     lambda x: (
    #         x["method"] + "+ash" if x["init_params.use_ash"] is True else x["method"]
    #     ),
    #     axis=1,
    # )
    return df


# ------------------------- models -------------------------------------
def get_model(dataset_name, model_name, device=None, source=None):
    """Loads a model and moves it to CUDA if available."""
    # load from torchvision
    if source == "torchvision":
        if model_name == "vit_b_16":
            assert dataset_name == "imagenet", "vit_b_16 only supports imagenet"
            model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        elif model_name == "vit_b_16_swag_linear":
            assert (
                dataset_name == "imagenet"
            ), "vit_b_16_swag_linear only supports imagenet"
            model = models.vit_b_16(
                weights=models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
            )
        elif model_name == "swin_t":
            assert dataset_name == "imagenet", "swin_t only supports imagenet"
            model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        elif model_name == "mobilenet_v2":
            assert dataset_name == "imagenet", "mobilenet_v2 only supports imagenet"
            model = models.mobilenet_v2(
                weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
            )
        elif model_name == "mobilenet_v3_large":
            assert (
                dataset_name == "imagenet"
            ), "mobilenet_v3_large only supports imagenet"
            model = models.mobilenet_v3_large(
                weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
            )
        elif model_name == "regnet_y_16gf":
            assert dataset_name == "imagenet", "regnet_y_16gf only supports imagenet"
            model = models.regnet_y_16gf(
                weights=models.RegNet_Y_16GF_Weights.IMAGENET1K_V1
            )
    # load cifar models
    elif model_name.startswith("cifar"):
        # load from https://github.com/chenyaofo/pytorch-cifar-models
        model = torch.hub.load(
            repo_or_dir="chenyaofo/pytorch-cifar-models",
            model=model_name,
            pretrained=True,
            verbose=False,
        ).to(device)
    else:
        # load from openood_networks
        model = get_network(
            dataset_name=dataset_name,
            model_name=model_name,
            device=device,
        )

    model.eval()
    if device is not None:
        model.to(device)
    return model


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


# def get_dataloader(
#     name: str, split: str, batch_size: int = 128, num_workers: int = 4
# ):
#     ds = get_dataset(name, split)
#     return DataLoader(
#         ds,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=True,
#     )
