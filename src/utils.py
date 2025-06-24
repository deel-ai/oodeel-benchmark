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
import yaml
from .openood_networks.utils import get_network

CFG_ROOT = Path(__file__).parent.parent / "configs"


def _resolve_attr(obj, path: str):
    for attr in path.split("."):
        obj = getattr(obj, attr)
    return obj


def _load_model_cfg(model_name: str):
    with open(CFG_ROOT / "models" / f"{model_name}.yaml") as f:
        return yaml.safe_load(f)


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
    return df


# ------------------------- models -------------------------------------
def get_model(dataset_name, model_name, device=None, source=None):
    """Load a model according to its configuration."""
    cfg = _load_model_cfg(model_name)
    model_source = source or cfg.get("source", "openood")

    if model_source == "torchvision":
        # torchvision models
        arch = cfg.get("architecture", model_name)
        weights_str = cfg.get("weights")
        weights = _resolve_attr(models, weights_str) if weights_str else None
        model = getattr(models, arch)(weights=weights)
    elif model_source == "torchhub":
        # torch hub models
        repo = cfg.get("repo", "chenyaofo/pytorch-cifar-models")
        model = torch.hub.load(
            repo_or_dir=repo, model=model_name, pretrained=True, verbose=False
        ).to(device)
    else:
        # openood models
        model = get_network(
            dataset_name=dataset_name, model_name=model_name, device=device
        )

    model.eval()
    if device is not None:
        model.to(device)
    return model
