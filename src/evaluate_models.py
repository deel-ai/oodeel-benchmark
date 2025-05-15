# ────────────────────────────────────────────────────────────────
# src/evaluate_models.py • launch with: python -m src.evaluate_models
# ────────────────────────────────────────────────────────────────
import os
from pathlib import Path
import yaml
import pandas as pd
import torch
from tqdm import tqdm

from .dataset import get_dataloader
from .utils import get_model

# ─── paths ─────────────────────────────────────────────────────
CFG_ROOT = Path(__file__).parent.parent / "configs"
RESULT_DIR = Path(__file__).parent.parent / "evaluate_models"
RESULT_DIR.mkdir(exist_ok=True, parents=True)

# ─── device ───────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")


# ─── helpers ──────────────────────────────────────────────────
def load_yaml(folder: str, name: str):
    path = CFG_ROOT / folder / f"{name}.yaml" if folder else CFG_ROOT / f"{name}.yaml"
    return yaml.safe_load(open(path, "r"))


# ─── main ─────────────────────────────────────────────────────
def main():
    # load benchmark configuration
    bench_cfg = load_yaml("", "benchmark")
    datasets = bench_cfg.get("datasets", [])

    for id_ds in datasets:
        ds_cfg = load_yaml("datasets", id_ds)
        models = ds_cfg.get("models", [])
        records = []
        print(f"Evaluating dataset '{id_ds}' with models: {models}")

        for model_name in models:
            print(f"  - Loading model '{model_name}'...")
            model_cfg = load_yaml("models", model_name)
            source = model_cfg.get("source", "openood")
            model = get_model(id_ds, model_name, device, source=source)
            model.eval()

            loader = get_dataloader(
                id_ds,
                split="test",
                preprocessor_dataset_name=id_ds,
                batch_size=ds_cfg.get("batch_size", 64),
                num_workers=ds_cfg.get("num_workers", 4),
            )

            correct1 = 0
            correct5 = 0
            total = 0
            top5 = id_ds in ("imagenet", "imagenet200")

            with torch.no_grad():
                for images, targets in tqdm(loader, desc=f"Evaluating {model_name}"):
                    images = images.to(device)
                    targets = targets.to(device)
                    outputs = model(images)

                    # top-1
                    _, pred1 = outputs.topk(1, dim=1, largest=True, sorted=True)
                    pred1 = pred1.view(-1)
                    correct1 += (pred1 == targets).sum().item()

                    # top-5 if applicable
                    if top5:
                        _, pred5 = outputs.topk(5, dim=1, largest=True, sorted=True)
                        # compare each row
                        for i in range(targets.size(0)):
                            if targets[i].item() in pred5[i].tolist():
                                correct5 += 1

                    total += targets.size(0)

            acc1 = correct1 / total if total > 0 else 0
            record = {"dataset": id_ds, "model": model_name, "accuracy": acc1}
            if top5:
                acc5 = correct5 / total if total > 0 else 0
                record["accuracy@5"] = acc5

            records.append(record)

        # save results for this dataset
        df = pd.DataFrame(records)
        out_file = RESULT_DIR / f"{id_ds}.parquet"
        df.to_parquet(out_file, index=False)
        print(f"Results saved to {out_file}\n")


if __name__ == "__main__":
    main()