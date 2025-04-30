# oodeel-benchmark

Minimal, reproducible harness to benchmark **[OODeel](https://github.com/deel-ai/oodeel)** detectors on any mix of  
_ID datasets × OOD datasets × models × feature-layer-packs × detector hparams grids_.

```
configs/                # YAML knobs (no code)
└─ datasets/            # 1 file per ID dataset
└─ models/              # 1 file per architecture (feature layer packs)
└─ methods/             # 1 file per detector (hyper-params)
src/                    # code
└─ dataset/             # dataset loaders (ID + OOD)
└─ openood_networks/    # model loaders (ID)
└─ utils.py             # utils (seed, etc.)
└─ run.py               # launch everything (crash-safe, resumable)
results/                # one .parquet per (ID, model, detector, …)
```

---

## Quick start

```bash
git clone git@github.com:y-prudent/oodeel-benchmark.git
cd oodeel-benchmark
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # torch, oodeel[torch], rich, wandb…
CUDA_VISIBLE_DEVICES=0 python -m src.run    # single GPU
```

_Parquet files and W&B dashboards appear as the sweep progresses._

---

## Multi-GPU / multi-machine

```bash
# 2 GPUs on the same box
CUDA_VISIBLE_DEVICES=0 python src/run.py --shard-index 0 --num-shards 2 &
CUDA_VISIBLE_DEVICES=1 python -m src/run.py --shard-index 1 --num-shards 2 &
```

Each process grabs its slice of the sweep; they meet only in the shared `results/` folder. Restarting is instant—completed files are skipped.

---

## Customising the sweep

1. **Add / edit YAMLs** under `configs/` to declare new datasets, models,
   layer packs or detector grids (see the existing examples).
2. **Optional**: limit huge training splits by inserting

   ```yaml
   fit_subset:
     per_class: 50 # ≤50 imgs / class
     max_samples: 50000
   ```

   in a dataset YAML.

---

## Metrics & plots

- Per OOD pair we save raw scores **and** `auroc`, `tpr5fpr` in Parquet.
- Live **AUROC × TPR** scatter plots are logged to Weights-and-Biases
  (`project=oodeel-bench`) — filter by method, model or layer pack.

---

### TL;DR

_One command runs the whole grid, auto-resumes, and streams metrics to W&B — all configs stay in plain YAML._
