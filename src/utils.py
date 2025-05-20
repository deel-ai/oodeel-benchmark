from pathlib import Path
from typing import Sequence, Union, Optional
import glob
import pandas as pd


def load_benchmark(
    path_or_glob: Union[str, Path] = "reduced_results/*.parquet",
    *,
    flatten_json: Optional[Sequence[str]] = ("init_params", "fit_params"),
) -> pd.DataFrame:
    """Load benchmark Parquet files into a single DataFrame.

    Parameters
    ----------
    path_or_glob:
        Glob pattern or directory pointing to the Parquet files.
    flatten_json:
        Column names containing JSON/dict values that should be expanded.
    """
    if "*" in str(path_or_glob):
        parquet_files = glob.glob(str(path_or_glob))
    else:
        parquet_files = list(Path(path_or_glob).glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found at {path_or_glob}")

    df = pd.concat(map(pd.read_parquet, parquet_files), ignore_index=True)

    if flatten_json:
        for col in flatten_json:
            if col in df.columns:
                df = df.join(
                    df[col]
                    .apply(lambda x: pd.Series(x if isinstance(x, dict) else {}))
                    .add_prefix(f"{col}.")
                )

    return df
