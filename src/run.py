# ────────────────────────────────────────────────────────────────
# src/run.py        •  launch with:   python -m src.run
# ────────────────────────────────────────────────────────────────
import os

os.environ["WANDB_CONSOLE"] = "off"
os.environ["WANDB_SILENT"] = "true"  # suppress banners & warnings
printed_wandb_banner = False

import argparse
import json
import hashlib
import itertools
from pathlib import Path

import yaml
import pandas as pd
import torch
import numpy as np
from rich.console import Console
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
    SpinnerColumn,
)
from sklearn.metrics import average_precision_score

import oodeel.methods as oodeel_methods
import oodeel.aggregator as oodeel_aggregator
from oodeel.eval.metrics import bench_metrics

from .dataset import get_dataloader  # your helpers
from .utils import seed_everything, cuda_tracker, get_model


# ─── paths ───────────────────────────────────────────────────────
CFG_ROOT = Path(__file__).parent.parent / "configs"
RESULT_DIR = Path(__file__).parent.parent / "results"
RESULT_DIR.mkdir(exist_ok=True, parents=True)

# ─── device info ────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")


# ─── helpers ────────────────────────────────────────────────────
def load_yaml(folder, name):
    return yaml.safe_load(open(CFG_ROOT / folder / f"{name}.yaml"))


def grid(d):
    if not d:
        yield {}
    else:
        k, v = zip(*d.items())
        for combo in itertools.product(*v):
            yield dict(zip(k, combo))


def run_is_complete(p: Path, expected_rows: int) -> bool:
    if not p.exists():
        return False
    try:
        return len(pd.read_parquet(p)) == expected_rows
    except Exception:
        return False


def wblog():
    try:
        import wandb

        return wandb
    except ModuleNotFoundError:

        class _Stub:
            def init(*a, **k):
                return _Stub()

            def log(*a, **k):
                pass

            def save(*a, **k):
                pass

            def finish(*a, **k):
                pass

        return _Stub()


wandb = wblog()


# ─── pretty progress text ───────────────────────────────────────
def short_descr(run):
    # mode (e.g. react, scale, ash)
    mode_txt = ""
    for name in ["react", "scale", "ash"]:
        if run["init"].get(f"use_{name}", False):
            mode_txt = f"({name})"
            break
    # aggregator
    agg = run["init"].get("aggregator")
    agg_txt = f" ({agg})" if agg else ""
    # description
    return (
        f"[green]{run['id_ds']}[/] / "
        f"[cyan]{run['model']}[/] / "
        f"[magenta]{run['method']}{mode_txt}:{run['layer_pack']}[/]{agg_txt}"
    )


def show_phase(prog, task, run, phase):
    prog.tasks[task].description = f"{short_descr(run)} → {phase}"


# ─── sweep builder ──────────────────────────────────────────────
def build_runs():
    bench_cfg = load_yaml("", "benchmark")
    datasets = {d: load_yaml("datasets", d) for d in bench_cfg["datasets"]}
    methods_cfg = {m: load_yaml("methods", m) for m in bench_cfg["methods"]}

    model_cache, runs = {}, []
    # iterate over all id datasets
    for id_ds, ds_cfg in datasets.items():
        # iterate over all models
        for model_name in ds_cfg["models"]:
            if model_name not in model_cache:
                model_cache[model_name] = load_yaml("models", model_name)
            model_spec = model_cache[model_name]

            # iterate over all methods
            for meth_name, meth_cfg in methods_cfg.items():
                meth_class = meth_cfg.get("class", meth_name.upper())

                # build init grid
                if "modes" in meth_cfg:  # manage react, ash, scale modes (e.g. ODIN)
                    base_spec = meth_cfg.get("base", {})
                    mode_specs = meth_cfg["modes"]
                    init_grid = []
                    for mode in mode_specs:
                        for base_combo in grid(base_spec):
                            for mode_combo in grid(mode):
                                merged = {**base_combo, **mode_combo}
                                init_grid.append(merged)
                else:  # no modes
                    init_grid = list(grid(meth_cfg.get("init_grid", {})))

                # build fit grid (e.g. feature_layers_id)
                fit_grid = meth_cfg.get("fit_grid", {})
                layer_packs = fit_grid.get("layer_packs", ["full"])

                other_fit_grid = list(
                    grid({k: v for k, v in fit_grid.items() if k != "layer_packs"})
                )

                # iterate over all combinations
                # fit params
                for pack in layer_packs:
                    layers = model_spec["layer_packs"][pack]
                    # init params
                    for init in init_grid:
                        for fit_extra in other_fit_grid:
                            fit = {**fit_extra, "feature_layers_id": layers}
                            uid = hashlib.md5(
                                json.dumps(
                                    [id_ds, model_name, pack, meth_name, init, fit],
                                    sort_keys=True,
                                ).encode()
                            ).hexdigest()[:8]
                            runs.append(
                                dict(
                                    uid=uid,
                                    id_ds=id_ds,
                                    model=model_name,
                                    source=model_spec.get("source", None),
                                    layer_pack=pack,
                                    layers=layers,
                                    ood_lists=ds_cfg["ood"],
                                    method=meth_name,
                                    method_class=meth_class,
                                    init=init,
                                    fit=fit,
                                    batch_size=bench_cfg["batch_size"],
                                    num_workers=bench_cfg["num_workers"],
                                )
                            )
    return runs


# ─── argparse ─────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--shard-index", type=int, default=0, help="Which shard am I (0-based)?"
    )
    p.add_argument("--num-shards", type=int, default=1, help="Total number of shards.")
    return p.parse_args()


# ─── main ────────────────────────────────────────────────────────
def main():
    cli = parse_args()

    runs = build_runs()
    console = Console()
    console.print(f"[bold cyan]👉  total runs to execute: {len(runs)}[/]")
    runs.sort(key=lambda r: r["uid"])

    # keep only the runs that belong to *this* shard
    runs = [r for i, r in enumerate(runs) if i % cli.num_shards == cli.shard_index]

    console.print(
        f"[bold yellow]Shard {cli.shard_index + 1}/{cli.num_shards}"
        f" → {len(runs)} runs[/]"
    )

    total_ood_pairs = sum(sum(len(v) for v in r["ood_lists"].values()) for r in runs)

    console.print(f"[bold green]Total OOD pairs to score: {total_ood_pairs}[/]")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.completed}/{task.total}",
        TimeRemainingColumn(),
        SpinnerColumn(speed=0.2),
        console=console,
    ) as progress:

        task = progress.add_task("Starting…", total=total_ood_pairs)

        for run in runs:
            out_file = RESULT_DIR / f"{run['uid']}.parquet"
            expected_rows = sum(len(v) for v in run["ood_lists"].values())

            if run_is_complete(out_file, expected_rows):
                progress.advance(task, expected_rows)
                continue
            if out_file.exists():  # incomplete
                console.log(f"[yellow]Re-running incomplete uid {run['uid']}[/]")
                out_file.unlink(missing_ok=True)

            progress.tasks[task].description = short_descr(run)
            seed_everything()

            wb = wandb.init(
                project="oodeel-bench",
                name=run["uid"],
                config=run,
                resume="allow",
                reinit=True,
                settings=wandb.Settings(console="off"),  # <- no spam
            )

            # print once the W&B banner
            global printed_wandb_banner
            if not printed_wandb_banner:
                proj_url = getattr(wb, "project_url", "https://wandb.ai")
                console.print(
                    "\n"
                    f"[bold blue]W&B:[/] logging to [link={proj_url}]{proj_url}[/link]"
                    f"\n(local dir: {wb.dir})\n"
                )
                printed_wandb_banner = True

            scatter_tables = {}  # one per benchmark

            # 1) model --------------------------------------------------
            model = get_model(
                run["id_ds"], run["model"], device, run.get("source", None)
            )
            # HOTFIX: SwinT model has a different resize size
            # TODO: get kwargs from the model config
            if run["model"] == "swin_t":
                kwargs = {"pre_size": 232}
            else:
                kwargs = {}

            # 2) data ---------------------------------------------------
            id_fit_loader = get_dataloader(
                run["id_ds"],
                "train",
                run["id_ds"],
                batch_size=run["batch_size"],
                num_workers=run["num_workers"],
                fit_subset_cfg=load_yaml("datasets", run["id_ds"]).get("fit_subset"),
                **kwargs,
            )
            id_test_loader = get_dataloader(
                run["id_ds"],
                "test",
                run["id_ds"],
                batch_size=run["batch_size"],
                num_workers=run["num_workers"],
            )

            # 3) detector ----------------------------------------------
            Detector = getattr(oodeel_methods, run["method_class"])
            init_kw = run["init"].copy()
            fit_kw = run["fit"].copy()

            # aggregator
            if "aggregator" in init_kw:
                agg_name = init_kw.pop("aggregator")
                init_kw["aggregator"] = getattr(oodeel_aggregator, agg_name)()

            # vit: hardcode the postproc_fns
            if run["model"].startswith("vit_"):
                num_layers = len(fit_kw["feature_layers_id"])
                if num_layers >= 1:
                    fit_kw["postproc_fns"] = [lambda x: torch.mean(x, dim=1)] * (
                        num_layers - 1
                    ) + [lambda x: x[:, 0]]
            if run["model"] == "swin_t":
                from torchvision.ops.misc import Permute
                import torch.nn as nn
                from torchvision.transforms import Compose

                num_layers = len(fit_kw["feature_layers_id"])
                if num_layers >= 1:
                    fit_kw["postproc_fns"] = [
                        Compose(
                            [
                                Permute([0, 3, 1, 2]),
                                nn.AdaptiveAvgPool2d(1),
                                nn.Flatten(1),
                            ]
                        )
                    ] * num_layers

            console.log(
                r"Running \[uid=",
                f"{run['uid']}" + r"] → " + short_descr(run),
            )

            # init detector
            detector = Detector(**init_kw)

            show_phase(progress, task, run, "[yellow]Fitting[/]")
            with cuda_tracker(console, "Fit", True):
                # fit the detector
                detector.fit(model, fit_dataset=id_fit_loader, **run["fit"])

            show_phase(progress, task, run, "[blue]Scoring ID[/]")
            with cuda_tracker(console, "ID score", True):
                # score the ID dataset
                id_scores = detector.score(id_test_loader)[0].tolist()

            # 4) loop OOD ----------------------------------------------
            records, done, total_oods = [], 0, expected_rows
            run_metrics = []
            for grp, ood_names in run["ood_lists"].items():
                for ood_name in ood_names:
                    done += 1
                    show_phase(
                        progress,
                        task,
                        run,
                        f"[bold]{ood_name}[/] ({done}/{total_oods})",
                    )

                    ood_loader = get_dataloader(
                        ood_name,
                        "test",
                        run["id_ds"],
                        batch_size=run["batch_size"],
                        num_workers=run["num_workers"],
                    )
                    with cuda_tracker(console, f"OOD {ood_name}", True):
                        ood_scores = detector.score(ood_loader)[0].tolist()

                    # metrics
                    split_val_test = False  # HARDCODED, TODO: make it configurable

                    if split_val_test:
                        # split the scores into val and test
                        id_scores_val = id_scores[: len(id_scores) // 2]
                        id_scores_test = id_scores[len(id_scores) // 2 :]
                        ood_scores_val = ood_scores[: len(ood_scores) // 2]
                        ood_scores_test = ood_scores[len(ood_scores) // 2 :]

                        # compute metrics for val
                        m_val = bench_metrics(
                            (np.array(id_scores_val), np.array(ood_scores_val)),
                            metrics=["auroc", "tpr5fpr", "fpr95tpr"],
                        )
                        auroc_val, tpr5_val, fpr95_val = (
                            m_val["auroc"],
                            m_val["tpr5fpr"].item(),
                            m_val["fpr95tpr"].item(),
                        )
                        ap_val = average_precision_score(
                            np.concatenate(
                                [
                                    np.zeros(len(id_scores_val)),
                                    np.ones(len(ood_scores_val)),
                                ]
                            ),
                            np.concatenate([id_scores_val, ood_scores_val]),
                        ).item()
                    else:
                        # no split, use the whole scores
                        id_scores_test = id_scores
                        ood_scores_test = ood_scores
                        auroc_val, tpr5_val, fpr95_val, ap_val = None, None, None, None

                    # compute metrics for test
                    m_test = bench_metrics(
                        (np.array(id_scores_test), np.array(ood_scores_test)),
                        metrics=["auroc", "tpr5fpr", "fpr95tpr"],
                    )
                    auroc, tpr5, fpr95 = (
                        m_test["auroc"],
                        m_test["tpr5fpr"].item(),
                        m_test["fpr95tpr"].item(),
                    )
                    ap = average_precision_score(
                        np.concatenate(
                            [
                                np.zeros(len(id_scores_test)),
                                np.ones(len(ood_scores_test)),
                            ]
                        ),
                        np.concatenate([id_scores_test, ood_scores_test]),
                    ).item()
                    run_metrics.append(
                        {
                            "group": grp,
                            "auroc_val": auroc_val,
                            "tpr5fpr_val": tpr5_val,
                            "fpr95tpr_val": fpr95_val,
                            "ap_score_val": ap_val,
                            "auroc": auroc,
                            "tpr5fpr": tpr5,
                            "fpr95tpr": fpr95,
                            "ap_score": ap,
                        }
                    )

                    # parquet row
                    if not run["init"]:
                        run["init"] = {"null": "null"}
                    records.append(
                        dict(
                            uid=run["uid"],
                            id_dataset=run["id_ds"],
                            ood_group=grp,
                            ood_dataset=ood_name,
                            model=run["model"],
                            layer_pack=run["layer_pack"],
                            method=run["method"],
                            init_params=run["init"],
                            fit_params=run["fit"],
                            id_scores=id_scores,
                            ood_scores=ood_scores,
                            auroc_val=auroc_val,
                            tpr5fpr_val=tpr5_val,
                            fpr95tpr_val=fpr95_val,
                            ap_score_val=ap_val,
                            auroc=auroc,
                            tpr5fpr=tpr5,
                            fpr95tpr=fpr95,
                            ap_score=ap,
                        )
                    )
                    pd.DataFrame(records).to_parquet(out_file, index=False)

                    # scatter plot build / log
                    bench_name = f"{run['id_ds']}_vs_{ood_name}"
                    if bench_name not in scatter_tables:
                        scatter_tables[bench_name] = wandb.Table(
                            columns=[
                                "AUROC",
                                "TPR5FPR",
                                "method",
                                "model",
                                "layer_pack",
                                "uid",
                            ]
                        )

                    scatter_tables[bench_name].add_data(
                        auroc,
                        tpr5,
                        run["method"],
                        run["model"],
                        run["layer_pack"],
                        run["uid"],
                    )
                    wandb.log(
                        {
                            f"scatter/{bench_name}": wandb.plot.scatter(
                                scatter_tables[bench_name],
                                "AUROC",
                                "TPR5FPR",
                                title=bench_name,
                            )
                        }
                    )

                    progress.advance(task)

            # after the loop – summarise the whole run -----------------------------
            avg = pd.DataFrame(run_metrics).groupby("group").mean()
            near, far = avg.loc["near", "auroc"], avg.loc["far", "auroc"]
            harmonic = 2 / (1 / near + 1 / far)
            console.log(
                f"[green]✓[/] [green]{run['id_ds']}[/] | [cyan]{run['model']}[/] | "
                f"[magenta]{run['method']}:{run['layer_pack']}[/] → "
                f"AUROC-near={near:.3f}  AUROC-far={far:.3f}  HM={harmonic:.3f}"
            )
            console.print("")  # empty line
            wb.save(str(out_file))
            wb.finish()

            # free memory ----------------------------------------------
            del detector, model, id_fit_loader, id_test_loader
            torch.cuda.empty_cache()

            if run["method"] == "dknn":  # faiss-gpu needs to be released
                import faiss

                faiss.StandardGpuResources().noTempMemory()  # dumps its temp buffers


if __name__ == "__main__":
    main()
