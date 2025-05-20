import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from plotly.colors import qualitative
import yaml
import math
import glob
import copy

from src.utils import load_benchmark


# ──────────────────────────────────────────────────────────────────────
# 0) Cache the raw bench load once
# ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_raw_results():
    return load_benchmark("results/*.parquet")


@st.cache_data
def load_evals():
    # Load ID‐accuracy parquets (preserves id_dataset and model)
    acc_files = glob.glob("evaluate_models/*.parquet")
    df_acc = pd.concat([pd.read_parquet(f) for f in acc_files], ignore_index=True)
    return df_acc


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


def sanitize(x):
    """
    Sanitize a Python object for YAML serialization.
    """
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {k: sanitize(v) for k, v in x.items() if v is not None}
    if isinstance(x, list):
        return [sanitize(v) for v in x if v is not None]
    return x


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
        color_discrete_sequence=px.colors.qualitative.Plotly,
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
    best = df.sort_values("near", ascending=False).drop_duplicates(
        subset=["model", "method_label"], keep="first"
    )
    melt = best.melt(
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
    best = df.sort_values("near", ascending=False).drop_duplicates(
        subset=["model", "method_label"], keep="first"
    )

    heat = best.assign(avg=df[["near", "far"]].mean(axis=1)).pivot_table(
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
    num_rows = math.ceil(len(df.model.unique()) / 3)
    fig = px.scatter(
        df,
        x="near",
        y="far",
        facet_col="model",
        facet_col_wrap=3,
        color="method_label",
        height=300 * num_rows,
        title="Near vs Far AUROC by Model (small multiples)",
        color_discrete_sequence=px.colors.qualitative.Plotly,
    )
    # cleanup facet titles
    for anno in fig.layout.annotations:
        anno.text = anno.text.replace("model=", "")
    return fig


def plot_model_corr_heatmap(df, id_ds):
    # 1) For each (model, method_label), pick the run with max near AUROC
    best = df.sort_values("near", ascending=False).drop_duplicates(
        subset=["model", "method_label"], keep="first"
    )

    # 2) Pivot so rows=models, cols=methods, values=near
    pivot = best.pivot(index="model", columns="method_label", values="near")

    # 3) Compute Spearman (rank) correlation between model-vectors
    #    .corr(method="spearman") works directly on the pivoted DataFrame
    corr = pivot.T.corr(method="spearman")

    # 4) Plot with Plotly
    fig = px.imshow(
        corr,
        text_auto=".2f",
        labels={"x": "Model", "y": "Model", "color": "Spearman ρ"},
        aspect="auto",
        color_continuous_scale="YlOrRd_r",
        zmin=0,
        zmax=1,
    )
    fig.update_layout(
        title=f"Model vs Model Rank-Correlation — {id_ds}",
        xaxis_tickangle=-45,
        margin=dict(l=80, t=50, b=50),
        height=500,
    )
    # make sure the heatmap is square
    fig.update_xaxes(scaleanchor="y", constrain="domain")
    fig.update_yaxes(scaleanchor="x", constrain="domain")
    return fig


def plot_method_corr_heatmap(df, id_ds):
    """Method vs Method rank-correlation heatmap."""
    # 1) For each (model, method_label), keep run with highest near AUROC
    best = df.sort_values("near", ascending=False).drop_duplicates(
        subset=["model", "method_label"], keep="first"
    )

    # 2) Pivot so rows=models, cols=methods
    pivot = best.pivot(index="model", columns="method_label", values="near")

    # 3) Compute Spearman correlation between method vectors
    corr = pivot.corr(method="spearman")

    # 4) Plot heatmap
    fig = px.imshow(
        corr,
        text_auto=".2f",
        labels={"x": "Method", "y": "Method", "color": "Spearman ρ"},
        aspect="auto",
        color_continuous_scale="YlGnBu_r",
        zmin=0,
        zmax=1,
    )
    fig.update_layout(
        title=f"Method vs Method Rank-Correlation — {id_ds}",
        xaxis_tickangle=-45,
        margin=dict(l=150, t=50, b=50),
        height=max(500, 20 * corr.shape[0]),
    )
    fig.update_yaxes(automargin=True)
    fig.update_xaxes(scaleanchor="y", constrain="domain")
    fig.update_yaxes(scaleanchor="x", constrain="domain")
    return fig


def plot_activation_shaping_boxplots(df, id_ds, ood_group="near"):
    # 1) Extract pure modes from method_label
    df2 = df.copy()
    df2["hyper_mode"] = df2["method_label"].str.extract(r"\((.*)\)$")[0].fillna("none")
    df2 = df2[df2["hyper_mode"].isin(["none", "react", "ash", "scale"])]
    df2["base_method"] = df2["method_label"].str.replace(r" \(.+\)$", "", regex=True)

    # 2) Keep only methods that support shaping
    shaped_methods = sorted(df2[df2["hyper_mode"] != "none"]["base_method"].unique())
    df2 = df2[df2["base_method"].isin(shaped_methods)]

    # 3) Pick best near‐AUROC per (base_method, hyper_mode, model)
    best = df2.sort_values("near", ascending=False).drop_duplicates(
        subset=["base_method", "hyper_mode", "model"], keep="first"
    )

    # 4) Compute Δ near = near – baseline(none) for each (base_method, model)
    baseline = (
        best[best.hyper_mode == "none"]
        .set_index(["base_method", "model"])[ood_group]
        .rename("baseline")
    )
    best = (
        best.set_index(["base_method", "model"])
        .join(baseline, how="left")
        .reset_index()
    )
    best["delta"] = best[ood_group] - best["baseline"]

    # 5) Remove the “none” baseline (all zeros) and plot
    best = best[best.hyper_mode != "none"]
    modes = ["react", "ash", "scale"]

    fig = px.box(
        best,
        x="hyper_mode",
        y="delta",
        color="hyper_mode",
        facet_col="base_method",
        # facet_col_wrap=per_row,
        category_orders={"hyper_mode": modes},
        labels={"delta": f"Δ {ood_group.capitalize()} AUROC", "hyper_mode": "Mode"},
        title=f"Activation-Shaping Impact (Δ {ood_group.capitalize()} AUROC) — {id_ds}",
        color_discrete_sequence=px.colors.qualitative.Plotly,
    )
    fig.update_layout(showlegend=True, height=300)

    # add a horizontal line at y=0
    fig.add_hline(y=0, line_color="black", line_width=1, line_dash="dash", opacity=0.3)

    # set y-axis limits to 1% and 99% quantiles
    lo, hi = best["delta"].quantile([0.01, 0.99])
    fig.update_yaxes(range=[lo, hi])

    # clean up facet titles
    for anno in fig.layout.annotations:
        anno.text = anno.text.replace("base_method=", "")

    return fig


def plot_layerpack_boxplots(df, id_ds, ood_group="near"):
    # 1) Filter for the three layer_pack modes
    packs = ["penultimate", "partial", "full"]
    df2 = df[df["layer_pack"].isin(packs)].copy()

    # 2) For each (method_label, layer_pack, model), pick best near‐AUROC
    best = df2.sort_values("near", ascending=False).drop_duplicates(
        subset=["method_label", "layer_pack", "model"], keep="first"
    )

    # 3) Compute Δ auroc = auroc – baseline(penultimate) per (method_label, model)
    baseline = (
        best[best.layer_pack == "penultimate"]
        .set_index(["method_label", "model"])[ood_group]
        .rename("baseline")
    )
    best = (
        best.set_index(["method_label", "model"])
        .join(baseline, how="left")
        .reset_index()
    )
    best["delta"] = best[ood_group] - best["baseline"]

    # 4) Faceted boxplot of Δ auroc by layer_pack
    best = best[best.layer_pack != "penultimate"]
    packs_cats = [p for p in best.layer_pack.unique() if p != "penultimate"]
    packs_cats.sort(key=lambda p: (p != "partial", p != "full"))
    fig = px.box(
        best,
        x="layer_pack",
        y="delta",
        color="layer_pack",
        facet_col="method_label",
        category_orders={"layer_pack": packs_cats},
        labels={
            "delta": f"Δ {ood_group.capitalize()} AUROC",
            "layer_pack": "Layers",
        },
        title=f"Layer-Pack Impact (Δ {ood_group.capitalize()} AUROC) — {id_ds}",
        color_discrete_sequence=px.colors.qualitative.Plotly,
    )
    fig.update_layout(showlegend=True, height=300)

    # add a horizontal line at y=0
    fig.add_hline(y=0, line_color="black", line_width=1, line_dash="dash", opacity=0.3)

    # set y-axis limits to 1% and 99% quantiles
    lo, hi = best["delta"].quantile([0.01, 0.99])
    fig.update_yaxes(range=[lo, hi])

    # clean up facet titles
    for anno in fig.layout.annotations:
        anno.text = anno.text.replace("method_label=", "")

    return fig


def plot_id_accuracy_vs_ood(df, eval_df, id_ds, ood_group="near"):
    # max OOD AUROC per model
    ood_summary = df.groupby("model", as_index=False)[ood_group].max()
    df_plot = eval_df.merge(ood_summary, on="model")
    # compute spearman correlation
    rho = df_plot["accuracy"].corr(df_plot[ood_group], method="spearman")
    # scatter
    fig = px.scatter(
        df_plot,
        x="accuracy",
        y=ood_group,
        color="model",
        text="model",
        labels={
            "accuracy": "ID Accuracy",
            ood_group: f"Max {ood_group.capitalize()} OOD AUROC",
        },
        title=f"ID vs OOD Performance ({ood_group.capitalize()}):"
        f" Spearman ρ = {rho:.2f} — {id_ds} ",
        trendline="ols",
        color_discrete_sequence=px.colors.qualitative.Plotly,
    )
    fig.update_traces(textposition="top center", showlegend=True)
    return fig


def plot_method_rank_stats_grouped(df, id_ds):
    # compute mean & std of ranks for both near and far
    stats = {}
    for grp in ["near", "far"]:
        best = df.sort_values(grp, ascending=False).drop_duplicates(
            subset=["model", "method_label"], keep="first"
        )
        # keep base methods AND energy variants with one shaping mode
        is_base = ~best["method_label"].str.contains(r"\(")
        is_energy_variant = best["method_label"].str.match(
            r"^energy \((react|ash|scale)\)$"
        )
        best_base = best[is_base | is_energy_variant].copy()
        pivot = best_base.pivot(index="model", columns="method_label", values=grp)
        ranks = pivot.rank(axis=1, method="average", ascending=False)
        stats[grp] = {
            "mean": ranks.mean(axis=0),
            "std": ranks.std(axis=0),
        }

    # sort methods by mean near‐rank ascending (best on left)
    mean_near = stats["near"]["mean"].sort_values()
    methods = mean_near.index.tolist()
    near_mean = mean_near.values
    near_std = stats["near"]["std"][methods].values
    far_mean = stats["far"]["mean"][methods].values
    far_std = stats["far"]["std"][methods].values

    # Colors: Set2 palette (colorblind friendly)
    colors = qualitative.Set2
    near_color = colors[0]  # e.g. '#66c2a5'
    far_color = colors[1]  # e.g. '#fc8d62'

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=methods,
            y=near_mean,
            name="Near mean rank",
            error_y=dict(type="data", array=near_std, thickness=1.5, width=2),
            marker=dict(color=near_color),
            width=0.35,
        )
    )
    fig.add_trace(
        go.Bar(
            x=methods,
            y=far_mean,
            name="Far mean rank",
            error_y=dict(type="data", array=far_std, thickness=1.5, width=2),
            marker=dict(color=far_color),
            width=0.35,
        )
    )

    fig.update_layout(
        title=f"Method Rank Variability — {id_ds}",
        barmode="group",
        bargap=0.2,
        bargroupgap=0.1,
        xaxis=dict(
            title="Method (sorted by near mean ↑)", tickangle=-45, automargin=True
        ),
        yaxis=dict(title="Mean rank (↓ best)"),
        # move legend above the plot, centered
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=50, r=50, t=80, b=150),  # extra top margin for legend
        # height=400 + 30 * len(methods),
    )

    return fig


# ──────────────────────────────────────────────────────────────────────
# Export figures helper
# ──────────────────────────────────────────────────────────────────────


def chart_with_download(fig, key, default_width=700, default_height=400):
    """
    Render the interactive Plotly figure, and inside an expander labeled
    “Export figure” put width/height inputs and PNG/PDF download buttons.
    The export-specific layout tweaks are applied only to the clone.
    """
    # 1) Show interactive figure unchanged
    st.plotly_chart(fig, use_container_width=True)

    # 2) Export controls inside spoiler
    with st.expander("Export figure"):
        c1, c2, c3, c4 = st.columns([1, 1, 0.6, 0.6], gap="small")

        width = c1.number_input(
            "Width (px)",
            min_value=100,
            value=default_width,
            key=f"{key}_w",
            help="Export width in pixels",
        )
        height = c2.number_input(
            "Height (px)",
            min_value=100,
            value=default_height,
            key=f"{key}_h",
            help="Export height in pixels",
        )

        # 3) Clone fig and apply NeurIPS-style layout
        fig_export = copy.deepcopy(fig)
        fig_export.update_layout(
            template="plotly_white",
            font=dict(family="serif", size=12),
            legend=dict(font=dict(size=10)),
            margin=dict(l=50, r=50, t=50, b=50),
        )
        for trace in fig_export.data:
            if hasattr(trace, "line") and trace.line is not None:
                trace.line.width = 1.5
            if hasattr(trace, "marker") and trace.marker is not None:
                ms = getattr(trace.marker, "size", None)
                if isinstance(ms, (int, float)):
                    trace.marker.size = max(ms, 8)

        # 4) Generate PNG @300 DPI
        png_bytes = fig_export.to_image(
            format="png",
            engine="kaleido",
            width=width,
            height=height,
            scale=300 / 72,  # ≈4.17 for 300 DPI
        )
        # 5) Generate vector PDF
        pdf_bytes = fig_export.to_image(
            format="pdf",
            engine="kaleido",
            width=width,
            height=height,
        )

        # 6) Download buttons
        c3.download_button(
            label="⬇️ PNG",
            data=png_bytes,
            file_name=f"{key}.png",
            mime="image/png",
            key=f"{key}_dl_png",
            help=f"Download {width}×{height}px @300 DPI",
        )
        c4.download_button(
            label="⬇️ PDF",
            data=pdf_bytes,
            file_name=f"{key}.pdf",
            mime="application/pdf",
            key=f"{key}_dl_pdf",
            help="Download vector PDF",
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
eval_df = load_evals()
eval_df = eval_df[eval_df["dataset"] == id_ds]

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
tab1, tab2, tab3 = st.tabs(
    ["🏆 Leaderboard", "📊 Visualizations", "📚 Paper Experiments"]
)

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
    st.markdown(r"#### Scatter with Pareto front")
    chart_with_download(
        plot_scatter_pareto(filtered, id_ds),
        key=f"{id_ds}_scatter_pareto",
        default_width=700,
        default_height=400,
    )

    st.markdown(r"#### Near vs Far AUROC Boxplot")
    chart_with_download(
        plot_box(filtered),
        key=f"{id_ds}_boxplot_near_far",
        default_width=700,
        default_height=400,
    )

    st.markdown(r"#### Method x Model Avg AUROC Heatmap")
    chart_with_download(
        plot_heatmap_plotly(filtered, id_ds),
        key=f"{id_ds}_heatmap_avg_auroc",
        default_width=700,
        default_height=700,
    )

    st.markdown(r"#### Small Multiples by Model")
    chart_with_download(
        plot_small_multiples(filtered),
        key=f"{id_ds}_small_multiples",
        default_width=700,
        default_height=800,
    )


with tab3:
    # exp 1: model vs model rank-correlation
    st.markdown(r"#### Exp 1: Model vs Model Rank-Correlation Heatmap")
    st.markdown(
        r"""
        For each ID dataset, define for each model $i$ the vector of best Near-OOD
         AUROCs over all methods:
        $$
        \mathbf{v}_i = \bigl[\,v_{i1},\,v_{i2},\,\dots,\,v_{iK}\bigr], 
        \quad v_{ik} = \max_{\text{layer\_pack}}\mathrm{AUROC}_{\text{near}}\bigl(\text{model}_i,
        \text{method}_k\bigr)
        $$
        Then compute Spearman’s rank‐correlation
        $$
        \rho_{ij} = \mathrm{SpearmanCorr}\bigl(\mathbf{v}_i,\mathbf{v}_j\bigr)
        $$
        and display the matrix $\{\rho_{ij}\}$ in a **Model Correlation Heatmap**.
        """
    )

    chart_with_download(
        plot_model_corr_heatmap(filtered, id_ds),
        key=f"{id_ds}_model_corr",
        default_width=700,
        default_height=700,
    )

    # exp 1b: method vs method rank-correlation
    st.markdown("---")
    st.markdown(r"#### Exp 1b: Method vs Method Rank-Correlation Heatmap")
    st.markdown(
        r"""
        For each method $k$ and model $i$, define the vector of best Near-OOD AUROCs across models:
        $$
        \mathbf{v}_k = \bigl[\,v_{1k},\,v_{2k},\,\dots\bigr], \quad
        v_{ik} = \max_{\text{layer\_pack}}\mathrm{AUROC}_{\text{near}}\bigl(\text{model}_i, \text{method}_k\bigr).
        $$
        Compute Spearman’s rank-correlation
        $$
        \rho_{k\ell} = \mathrm{SpearmanCorr}\bigl(\mathbf{v}_k,\mathbf{v}_\ell\bigr)
        $$
        and visualize the matrix $\{\rho_{k\ell}\}$ in a **Method Correlation Heatmap**.
        """
    )

    chart_with_download(
        plot_method_corr_heatmap(filtered, id_ds),
        key=f"{id_ds}_method_corr",
        default_width=700,
        default_height=700,
    )

    # exp 2: method ranking variability
    st.markdown("---")
    st.markdown("#### Exp 2: Method Ranking Variability (Near & Far)")
    st.markdown(
        r"""
    For each base method $k$ and model $i$, let
    $$
    v_{i,k} = \max_{\text{layer\_pack}} \mathrm{AUROC}(\text{model}_i, k).
    $$
    Convert each score vector $\{v_{i,k}\}_k$ into ranks $r_{i,k}$ (1 = best).
    We then plot, for each method $k$, the mean rank $\mathbb{E}_i[r_{i,k}]$ with
        error bars showing the rank’s standard deviation across models, separately
        for Near and Far.
    """
    )
    chart_with_download(
        plot_method_rank_stats_grouped(filtered, id_ds),
        key=f"{id_ds}_method_ranking_both",
        default_width=700,
        default_height=400,
    )

    # exp 3: activation shaping effect
    st.markdown("---")
    st.markdown("#### Exp 3: Activation‐Shaping Impact (Δ Near & Far AUROC)")
    st.markdown(
        r"""
    For each logit-based method $k$ and each shaping mode 
    $m \in \{\mathrm{react}, \mathrm{ash}, \mathrm{scale}\}$, define for each model $i$:

    $$
    \Delta v_{i,k,m}
    =
    \mathrm{AUROC}(\mathrm{model}_i, k, m)
    -
    \mathrm{AUROC}(\mathrm{model}_i, k, \mathrm{none}).
    $$

    We then draw boxplots of the distributions $\{\Delta v_{i,k,m}\}_i$ 
    across models, comparing each mode against the no‐shaping baseline.
    """
    )

    chart_with_download(
        plot_activation_shaping_boxplots(filtered, id_ds, "near"),
        key=f"{id_ds}_activation_shaping_near",
        default_width=700,
        default_height=400,
    )
    chart_with_download(
        plot_activation_shaping_boxplots(filtered, id_ds, "far"),
        key=f"{id_ds}_activation_shaping_far",
        default_width=700,
        default_height=400,
    )

    # exp 4: layer-pack impact
    st.markdown("---")
    st.markdown("#### Exp 4: Layer-Pack Impact (Δ Near & Far AUROC)")

    st.markdown(
        r"""
    For each feature-based method $k$ and each layer‐pack mode 
    $p \in \{\mathrm{penultimate}, \mathrm{partial}, \mathrm{full}\}$, we define:

    - **penultimate**: OOD scores computed **only** on the penultimate layer’s features.
    - **partial**: OOD scores computed on a **subset** of late feature layers, then 
                **aggregated** via a combination of p-values using Fisher test.
    - **full**: OOD scores computed on **all major block outputs**, then aggregated by
                the same p-value combination as above.

    For each model $i$, let
    $$
    \Delta v_{i,k,p}
    =
    \mathrm{AUROC}(\mathrm{model}_i, k, p)
    -
    \mathrm{AUROC}(\mathrm{model}_i, k, \mathrm{penultimate}).
    $$

    We then draw boxplots of the distributions $\{\Delta v_{i,k,p}\}_i$ across models,
                comparing **partial** and **full** against the penultimate baseline.
    """
    )
    chart_with_download(
        plot_layerpack_boxplots(filtered, id_ds, "near"),
        key=f"{id_ds}_layer_pack_near",
        default_width=700,
        default_height=400,
    )
    chart_with_download(
        plot_layerpack_boxplots(filtered, id_ds, "far"),
        key=f"{id_ds}_layer_pack_far",
        default_width=700,
        default_height=400,
    )

    st.markdown("---")
    # Exp 5: In-Distribution Accuracy vs. Max OOD AUROC
    st.markdown("#### Exp 5: In-Distribution Accuracy vs. Max OOD AUROC")
    st.markdown(
        r"""
    Define for each model $i$:
    $$
    v_i = \max_{k}\,\mathrm{AUROC}(\text{model}_i, \text{method}_k),\quad 
    \mathrm{Acc}_i = \text{ID Accuracy}(\text{model}_i).
    $$
    We then plot $ v_i $ against $ \mathrm{Acc}_i $, and report Spearman’s correlation
                $\rho$ to examine the relationship between in-distribution accuracy and
                OOD robustness.
    """
    )
    chart_with_download(
        plot_id_accuracy_vs_ood(filtered, eval_df, id_ds, "near"),
        key=f"{id_ds}_id_acc_vs_ood_near",
        default_width=700,
        default_height=400,
    )
    chart_with_download(
        plot_id_accuracy_vs_ood(filtered, eval_df, id_ds, "far"),
        key=f"{id_ds}_id_acc_vs_ood_far",
        default_width=700,
        default_height=400,
    )
