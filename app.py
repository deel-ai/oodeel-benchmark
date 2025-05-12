import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt

from src.utils import load_benchmark


# ──────────────────────────────────────────────────────────────────────
# 0) Cache the raw bench load once
# ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_raw_results():
    return load_benchmark()


# ──────────────────────────────────────────────────────────────────────
# 1) Build *all* leaderboards once, cache them
# ──────────────────────────────────────────────────────────────────────
@st.cache_data
def build_all_leaderboards(sort_by="near"):
    raw = load_raw_results()  # now uses the cached raw
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
            "+ " + " + ".join([f for f in flags if f]) if any(flags) else "none"
            for flags in zip(*modes)
        ]
        df["method_label"] = df["method"] + df["hyper_mode"].apply(
            lambda s: "" if s == "none" else f" ({s[2:]})"
        )

        tbl = (
            df.groupby(["uid", "model", "method_label", "layer_pack", "ood_group"])[
                "auroc"
            ]
            .mean()
            .reset_index()
            .pivot_table(
                index=["uid", "model", "method_label", "layer_pack"],
                columns="ood_group",
                values="auroc",
            )
            .reset_index()
        )
        tbl = tbl.sort_values(sort_by, ascending=False)
        tbl = tbl.drop_duplicates(
            subset=["model", "method_label", "layer_pack"], keep="first"
        ).reset_index(drop=True)

        tbl = tbl[["method_label", "near", "far", "model", "layer_pack", "uid"]]
        tbl.index.name = "rank"
        tables[id_ds] = tbl

    return tables


@st.cache_data
def filter_leaderboard(df, models, methods, packs, search):
    mask = (
        df["model"].isin(models)
        & df["method_label"].isin(methods)
        & df["layer_pack"].isin(packs)
    )
    out = df[mask]
    if search:
        s = search.lower()
        txt = (
            out[["model", "method_label", "layer_pack"]]
            .astype(str)
            .agg(" ".join, axis=1)
            .str.lower()
        )
        out = out[txt.str.contains(s)]
    return out


# ──────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────
def plot_scatter_pareto(df, id_ds):
    fig = px.scatter(
        df,
        x="near",
        y="far",
        color="method_label",
        symbol="model",
        hover_data=["uid", "layer_pack"],
        labels={"near": "Near AUROC", "far": "Far AUROC"},
        title=f"Near vs Far AUROC — {id_ds}",
    )
    # compute Pareto front
    pts = df[["near", "far"]].drop_duplicates().sort_values("near", ascending=False)
    max_far = -np.inf
    pareto = []
    for _, r in pts.iterrows():
        if r.far > max_far:
            pareto.append((r.near, r.far))
            max_far = r.far
    pareto = pd.DataFrame(pareto, columns=["near", "far"])[::-1]
    fig.add_trace(
        go.Scatter(
            x=pareto.near,
            y=pareto.far,
            mode="lines",
            line=dict(dash="dash", color="black"),
            name="Pareto front",
        )
    )
    return fig


def plot_box(df):
    melt = df.melt(
        id_vars=["method_label"],
        value_vars=["near", "far"],
        var_name="OOD-type",
        value_name="AUROC",
    )
    return px.box(
        melt,
        x="method_label",
        y="AUROC",
        color="OOD-type",
        title="Distribution of Near vs Far AUROC by Method",
    )


def plot_heatmap(df, id_ds):
    heat = df.assign(avg=df[["near", "far"]].mean(axis=1)).pivot_table(
        index="method_label", columns="model", values="avg"
    )
    fig, ax = plt.subplots(figsize=(12, 10), dpi=200)
    sns.heatmap(
        heat,
        annot=True,
        fmt=".3f",
        cbar_kws={"label": "Avg (near+far) AUROC"},
        annot_kws={"fontsize": 6},
        ax=ax,
    )
    ax.set_title(f"Avg AUROC per Method x Model — {id_ds}", fontsize=16, pad=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=6)
    plt.tight_layout()
    return fig


def plot_small_multiples(df):
    return px.scatter(
        df,
        x="near",
        y="far",
        facet_col="model",
        facet_col_wrap=3,
        color="method_label",
        height=800,
        title="Near vs Far AUROC by Model (small multiples)",
    )


# ──────────────────────────────────────────────────────────────────────
# Streamlit layout
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(layout="centered")
st.title("OODeel Benchmark Dashboard")

# Sidebar
raw = load_raw_results()  # **fast** after first call
ID_OPTIONS = sorted(raw["id_dataset"].unique())
id_ds = st.sidebar.selectbox(
    "ID dataset", ID_OPTIONS, index=ID_OPTIONS.index("imagenet")
)

tables = build_all_leaderboards()
df = tables[id_ds]

st.sidebar.header("Filters")
search = st.sidebar.text_input("Search")
models = st.sidebar.multiselect("Model", df.model.unique(), default=df.model.unique())
methods = st.sidebar.multiselect(
    "Method", df.method_label.unique(), default=df.method_label.unique()
)
packs = st.sidebar.multiselect(
    "Layer pack", df.layer_pack.unique(), default=df.layer_pack.unique()
)

filtered = filter_leaderboard(df, models, methods, packs, search)

# Tabs
tab1, tab2 = st.tabs(["🏆 Leaderboard", "📊 Visualizations"])

with tab1:
    st.subheader(f"Top runs — ID: {id_ds.capitalize()}")
    st.write(f"Showing {len(filtered)} / {len(df)} runs")
    st.dataframe(filtered, height=800, use_container_width=True)

with tab2:
    st.subheader("Scatter with Pareto front")
    st.plotly_chart(plot_scatter_pareto(filtered, id_ds), use_container_width=True)

    st.subheader("Near vs Far AUROC Boxplot")
    st.plotly_chart(plot_box(filtered), use_container_width=True)

    st.subheader("Method x Model Avg AUROC Heatmap")
    st.pyplot(plot_heatmap(filtered, id_ds))

    st.subheader("Small Multiples by Model")
    st.plotly_chart(plot_small_multiples(filtered), use_container_width=True)
