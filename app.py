import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import yaml

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
        color_discrete_sequence=px.colors.qualitative.Plotly,
        title="Distribution of Near vs Far AUROC by Method",
    )


def plot_heatmap_plotly(df, id_ds):
    # build pivoted avg table
    heat = df.assign(avg=df[["near", "far"]].mean(axis=1)).pivot_table(
        index="method_label",
        columns="model",
        values="avg",
        aggfunc="mean",  # ← handle duplicates
    )
    # use plotly express for a nice interactive heatmap
    fig = px.imshow(
        heat,
        text_auto=".3f",
        labels={"x": "Model", "y": "Method", "color": "Avg AUROC"},
        aspect="auto",
        color_continuous_scale="Magma",
        # color_continuous_scale="YlOrRd",
        # color_continuous_scale="RdYlGn",
    )
    fig.update_layout(
        title=f"Avg AUROC per Method x Model — {id_ds}",
        xaxis_tickangle=-45,
        margin=dict(l=150, t=50, b=50),  # make room on the left
        height=max(600, 24 * heat.shape[0]),  # grow height per method
    )
    fig.update_yaxes(automargin=True, tickfont=dict(size=8))  # ensure no clipping
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
    # description above the leaderboard
    st.markdown(
        "_One row per (ID dataset × model × method × layer_pack)._  "
        "For each, we tested multiple hyper-parameter configurations and "
        "**selected the best** according to the _near_-OOD AUROC."
    )
    # Leaderboard table
    st.subheader(f"Top runs — ID: {id_ds.capitalize()}")
    st.write(f"Showing {len(filtered)} / {len(df)} runs")
    styled = filtered.style.background_gradient(
        subset=["near", "far"],
        cmap="YlOrRd_r",
    )
    styled.format(
        {
            "near": "{:.3f}",
            "far": "{:.3f}",
        }
    )
    st.dataframe(styled, height=800, use_container_width=True)

    def sanitize(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, dict):
            return {k: sanitize(v) for k, v in x.items()}
        if isinstance(x, list):
            return [sanitize(v) for v in x]
        return x

    # Run‐config viewer
    st.markdown("**Run Configuration**")
    sel = st.selectbox("Select UID", filtered["uid"])
    run_row = raw[raw["uid"] == sel].iloc[0]
    config = {
        "id_dataset": run_row.id_dataset,
        "model": run_row.model,
        "method": run_row.method,
        "init_params": run_row.init_params,
        "fit_params": run_row.fit_params,
    }
    st.code(yaml.dump(sanitize(config), sort_keys=False), language="yaml")

with tab2:
    # explanation of what the visualizations show
    st.markdown(
        "_Interactive visualizations of the selected runs:_  \n"
        "- **Scatter & Pareto:** each point is a run (near vs far AUROC), with the "
        "Pareto front highlighted.  \n"
        "- **Boxplot:** distribution of near/far AUROC across methods.  \n"
        "- **Heatmap:** average AUROC (near+far) per method × model.  \n"
        "- **Small multiples:** per-model scatter facets."
    )
    st.subheader("Scatter with Pareto front")
    st.plotly_chart(plot_scatter_pareto(filtered, id_ds), use_container_width=True)

    st.subheader("Near vs Far AUROC Boxplot")
    st.plotly_chart(plot_box(filtered), use_container_width=True)

    st.subheader("Method x Model Avg AUROC Heatmap")
    st.plotly_chart(plot_heatmap_plotly(filtered, id_ds), use_container_width=True)

    st.subheader("Small Multiples by Model")
    st.plotly_chart(plot_small_multiples(filtered), use_container_width=True)
