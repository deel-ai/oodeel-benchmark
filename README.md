<p align="center">
  <img src="https://github.com/user-attachments/assets/0f330118-d3b7-4b46-8674-c3a6b25a4ddf" width="40%" alt="OODeel Benchmark logo"/>
</p>

This repository contains a lightweight [Streamlit](https://streamlit.io/) application to explore the results of the OODeel out‑of‑distribution detection benchmark.

The Parquet files in `reduced_results/` and `evaluate_models/` ship with the repository and are loaded at runtime. The app builds interactive leaderboards, scatter plots, box plots and heatmaps to compare detection methods and models.

## Online demo

[https://oodeel-benchmark.streamlit.app](https://oodeel-benchmark.streamlit.app)

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app will open in your browser and load the provided Parquet files.
