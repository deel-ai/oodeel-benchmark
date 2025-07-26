"""Profile OODeel detectors on ImageNet/ResNet-50.

Usage: ``python -m src.profile_efficiency``.
"""

import argparse
import itertools
import time
import gc
from pathlib import Path
import yaml
import pandas as pd
import torch
from rich.console import Console

from .utils import get_model
from .dataset import get_dataloader

try:
    import oodeel.methods as oodeel_methods
except Exception:
    oodeel_methods = None

try:
    import openood.methods as openood_methods
except Exception:
    openood_methods = None

CFG_ROOT = Path(__file__).parent.parent / "configs"
PROFILE_DIR = Path(__file__).parent.parent / "profile_results"
PROFILE_DIR.mkdir(exist_ok=True, parents=True)


def load_yaml(folder: str, name: str):
    path = CFG_ROOT / folder / f"{name}.yaml" if folder else CFG_ROOT / f"{name}.yaml"
    return yaml.safe_load(open(path, "r"))


def first_from_grid(spec: dict) -> dict:
    """Return the first combination from a parameter grid."""
    if not spec:
        return {}
    keys, values = zip(*spec.items())
    combo = next(iter(itertools.product(*values)))
    return dict(zip(keys, combo))


def build_params(method_cfg: dict, model_cfg: dict):
    """Extract default init and fit parameters for a method."""
    # init params
    if "modes" in method_cfg:
        base = first_from_grid(method_cfg.get("base", {}))
        mode = (
            first_from_grid(method_cfg["modes"][0]) if method_cfg.get("modes") else {}
        )
        init_params = {**base, **mode}
    else:
        init_params = first_from_grid(method_cfg.get("init_grid", {}))

    # fit params
    fit_grid = method_cfg.get("fit_grid", {})
    layer_pack = fit_grid.get("layer_packs", ["full"])[0]
    other = first_from_grid({k: v for k, v in fit_grid.items() if k != "layer_packs"})
    layers = model_cfg["layer_packs"].get(layer_pack, [])
    fit_params = {**other, "feature_layers_id": layers}
    return init_params, fit_params


def profile_phase(console: Console, tag: str, func, *args, **kwargs):
    """Run a function and measure elapsed time and CUDA memory."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated()
    else:
        start_mem = 0
    t0 = time.time()
    result = func(*args, **kwargs)
    dt = time.time() - t0
    if torch.cuda.is_available():
        end_mem = torch.cuda.memory_allocated()
        peak = torch.cuda.max_memory_allocated()
        delta = (end_mem - start_mem) / (1024**2)
        peak_mem = peak / (1024**2)
    else:
        delta = peak_mem = 0.0
    console.log(
        f"[mem] {tag:20s} Δ={delta:7.1f} MiB  peak={peak_mem:7.1f} MiB  time={dt:5.1f}s",
        markup=False,
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, dt, delta, peak_mem


def profile_method(
    method_name: str,
    console: Console,
    model,
    id_fit,
    id_test,
    ood_loader,
    implementation: str,
):
    """Profile a single detector implementation."""
    meth_cfg = load_yaml("methods", method_name)
    model_cfg = load_yaml("models", "resnet50")
    DetectorClass = None
    if implementation == "oodeel" and oodeel_methods:
        DetectorClass = getattr(
            oodeel_methods, meth_cfg.get("class", method_name.upper())
        )
    elif implementation == "openood" and openood_methods:
        DetectorClass = getattr(
            openood_methods, meth_cfg.get("class", method_name.upper()), None
        )
    if DetectorClass is None:
        return []
    init_params, fit_params = build_params(meth_cfg, model_cfg)
    detector = DetectorClass(**init_params)

    records = []
    _, dt, delta, peak = profile_phase(
        console,
        f"{method_name} fit",
        detector.fit,
        model,
        fit_dataset=id_fit,
        **fit_params,
    )
    records.append(
        dict(
            method=method_name,
            implementation=implementation,
            phase="fit",
            time_s=dt,
            delta_mem_MiB=delta,
            peak_mem_MiB=peak,
        )
    )

    _, dt, delta, peak = profile_phase(
        console, f"{method_name} id_score", detector.score, id_test
    )
    records.append(
        dict(
            method=method_name,
            implementation=implementation,
            phase="id_score",
            time_s=dt,
            delta_mem_MiB=delta,
            peak_mem_MiB=peak,
        )
    )

    _, dt, delta, peak = profile_phase(
        console, f"{method_name} ood_score", detector.score, ood_loader
    )
    records.append(
        dict(
            method=method_name,
            implementation=implementation,
            phase="ood_score",
            time_s=dt,
            delta_mem_MiB=delta,
            peak_mem_MiB=peak,
        )
    )

    return records


def main(methods=None, ood_ds="ninco"):
    console = Console()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(
        f"Using {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}"
    )

    model = get_model("imagenet", "resnet50", device, source="openood")

    ds_cfg = load_yaml("datasets", "imagenet")
    id_fit = get_dataloader(
        "imagenet",
        "train",
        "imagenet",
        batch_size=ds_cfg.get("batch_size", 64),
        num_workers=ds_cfg.get("num_workers", 8),
        fit_subset_cfg=ds_cfg.get("fit_subset"),
    )
    id_test = get_dataloader(
        "imagenet",
        "test",
        "imagenet",
        batch_size=ds_cfg.get("batch_size", 64),
        num_workers=ds_cfg.get("num_workers", 8),
    )
    ood_loader = get_dataloader(
        ood_ds,
        "test",
        "imagenet",
        batch_size=ds_cfg.get("batch_size", 64),
        num_workers=ds_cfg.get("num_workers", 8),
    )

    if methods is None:
        methods = [p.stem for p in (CFG_ROOT / "methods").glob("*.yaml")]

    records = []
    for meth in methods:
        console.print(f"\n[bold cyan]Profiling {meth} (OODeel)[/]")
        records.extend(
            profile_method(meth, console, model, id_fit, id_test, ood_loader, "oodeel")
        )
        if openood_methods:
            console.print(f"[bold cyan]Profiling {meth} (OpenOOD)[/]")
            records.extend(
                profile_method(
                    meth, console, model, id_fit, id_test, ood_loader, "openood"
                )
            )

    df = pd.DataFrame(records)
    out_file = PROFILE_DIR / "imagenet_resnet50.parquet"
    df.to_parquet(out_file, index=False)
    console.print(f"\n[green]Results saved to {out_file}")
    console.print(
        df.groupby(["method", "implementation", "phase"])[
            ["time_s", "delta_mem_MiB"]
        ].mean()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="*", help="Subset of methods to profile")
    parser.add_argument("--ood", default="ninco", help="OOD dataset name")
    args = parser.parse_args()
    main(args.methods, args.ood)
