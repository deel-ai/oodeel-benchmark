import glob

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
from PIL import Image

from src.dataset import DATASETS_INFO, get_dataset
from src.utils import load_benchmark


# ──────────────────────────────────────────────────────────────────────
# 0) Cache the raw bench load once
# ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_raw_results():
    return load_benchmark("/data/corentin.friedrich/results/*.parquet")


@st.cache_data
def load_evals():
    # Load ID‐accuracy parquets (preserves id_dataset and model)
    acc_files = glob.glob("evaluate_models/*.parquet")
    df_acc = pd.concat([pd.read_parquet(f) for f in acc_files], ignore_index=True)
    return df_acc


ood_ds = {
    "imagenet": "ninco",
    "imagenet200": "ninco",
    "cifar10": "cifar100",
    "cifar100": "cifar10",
}


@st.cache_data
def get_9_id_images(id_ds_name, ood_ds):
    """Get 9 random images from the ID and OOD datasets."""
    ds = get_dataset(id_ds_name, "test", id_ds_name)
    np.random.seed(42)  # for reproducibility
    indices = np.random.choice(len(ds), size=9, replace=False)
    img_paths = ["/datasets/openood/" + DATASETS_INFO[id_ds_name]["data_dir"] + ds[i]["image_name"] for i in indices]
    imgs = [Image.open(p) for p in img_paths]

    # Same for OOD dataset
    ood_ds_name = ood_ds[id_ds_name]
    ood_ds = get_dataset(ood_ds_name, "test", id_ds_name)
    np.random.seed(42)
    ood_indices = np.random.choice(len(ood_ds), size=9, replace=False)
    ood_img_paths = [
        "/datasets/openood/" + DATASETS_INFO[ood_ds_name]["data_dir"] + ood_ds[i]["image_name"] for i in ood_indices
    ]
    ood_imgs = [Image.open(p) for p in ood_img_paths]

    return indices, imgs, ood_indices, ood_imgs


@st.cache_data
def get_9_ood_images(ood_id_ds_name):
    """Get 9 random images from the ID dataset."""
    ds = get_dataset(id_ds_name, "test", id_ds_name)
    np.random.seed(42)  # for reproducibility
    indices = np.random.choice(len(ds), size=9, replace=False)
    img_paths = ["/datasets/openood/" + DATASETS_INFO[id_ds_name]["data_dir"] + ds[i]["image_name"] for i in indices]
    imgs = [Image.open(p) for p in img_paths]
    return indices, imgs


# ──────────────────────────────────────────────────────────────────────
# 1) Build *all* leaderboards once, cache them
# ──────────────────────────────────────────────────────────────────────
@st.cache_data
def build_all_leaderboards(sort_by="near", best_only=True):
    """Build leaderboards for all ID datasets.

    Parameters
    ----------
    sort_by: str
        Column used to rank the runs (default is "near").
    best_only: bool
        If True, keep only the run with the best ``sort_by`` score for each
        (model, method_label, layer_pack) combination. If False, keep all runs.
    """

    raw = load_raw_results()  # cached raw results
    tables = {}
    for id_ds in sorted(raw["id_dataset"].unique()):
        df = raw[raw["id_dataset"] == id_ds].copy()

        # vectorized hyper_mode label
        modes = [
            np.where(df["init_params.use_react"] == True, "react", ""),
            np.where(df["init_params.use_ash"] == True, "ash", ""),
            np.where(df["init_params.use_scale"] == True, "scale", ""),
        ]
        df["hyper_mode"] = [
            "+ " + " + ".join([f for f in flags if f]) if any(flags) else "none" for flags in zip(*modes)
        ]
        df["method_label"] = df["method"] + df["hyper_mode"].apply(lambda s: "" if s == "none" else f" ({s[2:]})")

        df2 = (
            df[df["ood_dataset"] == ood_ds[id_ds]]
            # .sort_values(by="auroc", ascending=False)
            # .drop_duplicates(subset=["model", "method_label", "layer_pack"], keep="first")
        )

        uids = df2["uid"].unique()

        tbl = df[df["uid"].isin(uids)]
        tables[id_ds] = tbl

    return tables


# ──────────────────────────────────────────────────────────────────────
# Streamlit layout
# ──────────────────────────────────────────────────────────────────────

st.set_page_config(layout="centered")
st.title("OODeel Demo")

tables = build_all_leaderboards(sort_by="near", best_only=True)

id_datasets = tables.keys()
id_ds = st.selectbox("In-Distribution (ID) dataset", id_datasets)

df = tables[id_ds]

id_indices, id_images, ood_indices, ood_images = get_9_id_images(id_ds, ood_ds)

available_models = df["model"].unique()
model = st.selectbox("Model", available_models)

available_methods = df[df["model"] == model]["method"].unique()
ood_method = st.selectbox("OOD Method", available_methods)

df = df[df["model"] == model][df["method"] == ood_method]

id_scores = df.iloc[0]["id_scores"]
ood_scores = df[df["ood_dataset"] == ood_ds[id_ds]].iloc[0]["ood_scores"]

threshold = st.slider(
    "Select OOD Detection Threshold",
    min_value=float(id_scores.min()),
    max_value=float(id_scores.max()),
    value=float(np.percentile(id_scores, 95)),
    step=(float(id_scores.max() - id_scores.min())) / 100,
)

st.subheader("ID/OOD Scores Histogram")

checkbox = st.checkbox("Show OOD score histogram")


fig = go.Figure()
if checkbox:
    fig.add_trace(
        go.Histogram(
            x=ood_scores, name="OOD scores", opacity=0.6, marker_color="tomato", histnorm="probability density"
        )
    )
fig.add_trace(
    go.Histogram(x=id_scores, name="ID scores", opacity=0.6, marker_color="steelblue", histnorm="probability density")
)

# Overlay both histograms
fig.update_layout(barmode="overlay")

# Show the selected threshold on the plotly chart
fig.add_vline(
    x=threshold,
    line_color="red",
    line_dash="dash",
    annotation_text=f"Threshold: {threshold:.2f}",
    annotation_position="top left",
)
st.plotly_chart(fig, use_container_width=True)

# Plot 9 random images from the ID dataset
st.subheader("ID Images")
cols = st.columns(3)
for i, (ind, img) in enumerate(zip(id_indices, id_images)):
    with cols[i % 3]:
        if id_scores[ind] < threshold:
            st.image(img, caption=f"Score: {id_scores[ind]:.3f} -> :green[**ID**] ", use_container_width=True)
        else:
            st.image(img, caption=f"Score: {id_scores[ind]:.3f} -> :red[**OOD**] ", use_container_width=True)

# Plot 9 random images from the OOD dataset
st.subheader("OOD Images")
cols = st.columns(3)
for i, (ind, img) in enumerate(zip(ood_indices, ood_images)):
    with cols[i % 3]:
        if ood_scores[ind] < threshold:
            st.image(img, caption=f"Score: {ood_scores[ind]:.3f} -> :red[**ID**] ", use_container_width=True)
        else:
            st.image(img, caption=f"Score: {ood_scores[ind]:.3f} -> :green[**OOD**] ", use_container_width=True)
