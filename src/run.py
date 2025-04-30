# ────────────────────────────────────────────────────────────────
# src/run.py        •  launch with:   python -m src.run
# ────────────────────────────────────────────────────────────────
import json
import hashlib
import itertools
from pathlib import Path
from time import perf_counter

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

import oodeel.methods as oodeel_methods
import oodeel.aggregator as oodeel_aggregator
from oodeel.eval.metrics import bench_metrics

from dataset import get_dataloader  # your helpers
from openood_networks import get_network
from utils import seed_everything

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
    agg = run["init"].get("aggregator")
    agg_txt = f" ({agg})" if agg else ""
    return (
        f"[green]{run['id_ds']}[/] / "
        f"[cyan]{run['model']}[/] / "
        f"[magenta]{run['method']}:{run['layer_pack']}[/]{agg_txt}"
    )


def show_phase(prog, task, run, phase):
    prog.update(task, description=f"{short_descr(run)} → {phase}", advance=0)


# ─── sweep builder ──────────────────────────────────────────────
def build_runs():
    bench_cfg = load_yaml("", "benchmark")
    datasets = {d: load_yaml("datasets", d) for d in bench_cfg["datasets"]}
    methods_cfg = {m: load_yaml("methods", m) for m in bench_cfg["methods"]}

    model_cache, runs = {}, []
    for id_ds, ds_cfg in datasets.items():
        for model_name in ds_cfg["models"]:
            if model_name not in model_cache:
                model_cache[model_name] = load_yaml("models", model_name)
            model_spec = model_cache[model_name]

            for meth_name, meth_cfg in methods_cfg.items():
                meth_class = meth_cfg.get("class", meth_name.upper())
                init_grid = list(grid(meth_cfg.get("init_grid", {})))
                fit_grid = meth_cfg.get("fit_grid", {})
                layer_packs = fit_grid.get("layer_packs", ["full"])

                other_fit_grid = list(
                    grid({k: v for k, v in fit_grid.items() if k != "layer_packs"})
                )

                for pack in layer_packs:
                    layers = model_spec["layer_packs"][pack]
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


# ─── main ────────────────────────────────────────────────────────
def main():
    runs = build_runs()
    console = Console()
    console.print(f"[bold cyan]👉  total runs to execute: {len(runs)}[/]")
    runs.sort(key=lambda r: r["uid"])

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.completed}/{task.total}",
        TimeRemainingColumn(),
        SpinnerColumn(speed=0.2),
        console=console,
    ) as progress:
        task = progress.add_task("Starting…", total=len(runs))

        for run in runs:
            out_file = RESULT_DIR / f"{run['uid']}.parquet"
            expected_rows = sum(len(v) for v in run["ood_lists"].values())

            if run_is_complete(out_file, expected_rows):
                progress.advance(task)
                continue
            if out_file.exists():  # incomplete
                console.log(f"[yellow]Re-running incomplete uid {run['uid']}[/]")
                out_file.unlink(missing_ok=True)

            progress.update(task, description=short_descr(run))
            seed_everything()

            wb = wandb.init(
                project="oodeel-bench",
                name=run["uid"],
                config=run,
                resume="allow",
                reinit=True,
            )
            scatter_tables = {}  # one per benchmark

            # 1) model --------------------------------------------------
            model = get_network(run["id_ds"], run["model"]).to(device).eval()

            # 2) data ---------------------------------------------------
            id_fit_loader = get_dataloader(
                run["id_ds"],
                "train",
                run["id_ds"],
                batch_size=run["batch_size"],
                num_workers=run["num_workers"],
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
            if "aggregator" in init_kw:
                agg_name = init_kw.pop("aggregator")
                init_kw["aggregator"] = getattr(oodeel_aggregator, agg_name)()
            detector = Detector(**init_kw)

            show_phase(progress, task, run, "[yellow]Fitting[/]")
            t0 = perf_counter()
            detector.fit(model, fit_dataset=id_fit_loader, **run["fit"])
            console.log(f"Fit done in {perf_counter()-t0:.1f}s  uid={run['uid']}")

            show_phase(progress, task, run, "[blue]Scoring ID[/]")
            id_scores = detector.score(id_test_loader)[0].tolist()

            # 4) loop OOD ----------------------------------------------
            records, done, total_oods = [], 0, expected_rows
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
                    ood_scores = detector.score(ood_loader)[0].tolist()

                    # metrics
                    m = bench_metrics(
                        (np.array(id_scores), np.array(ood_scores)),
                        metrics=["auroc", "tpr5fpr"],
                    )
                    auroc, tpr5 = m["auroc"], m["tpr5fpr"]

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
                            auroc=auroc,
                            tpr5fpr=tpr5,
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
            wb.save(str(out_file))
            wb.finish()


if __name__ == "__main__":
    main()
