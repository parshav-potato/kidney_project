"""Data loading functions for NHIS and NHANES datasets."""

from pathlib import Path

import pandas as pd

from kidney.config import COLUMN_MAPPINGS, FILE_PATHS, DEFAULT_NHANES_FILES, NHANES_FILE_ORDER


def load_nhis() -> dict[str, pd.DataFrame]:
    """Load and standardise NHIS datasets for all configured years."""
    dataframes: dict[str, pd.DataFrame] = {}
    for year, path in FILE_PATHS.items():
        try:
            column_map = COLUMN_MAPPINGS[year]
            df = pd.read_csv(path)
            df = df.rename(columns=column_map)[list(column_map.values())]
            dataframes[year] = df
            print(f"Loaded data for {year}: {df.shape[0]} rows")
        except FileNotFoundError:
            print(f"Warning: File not found for {year} ({path})")
        except Exception as e:
            print(f"Error processing file for {year} ({path}): {e}")
    return dataframes


def load_nhanes(files: dict[str, Path] | None = None) -> pd.DataFrame:
    """Load and merge NHANES XPT files (left-join on SEQN)."""
    merged_files = {**DEFAULT_NHANES_FILES, **(files or {})}
    missing = [str(p) for p in merged_files.values() if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing NHANES files:\n - " + "\n - ".join(missing))

    df: pd.DataFrame | None = None
    for key in NHANES_FILE_ORDER:
        path = merged_files.get(key)
        if not path:
            continue
        part = pd.read_sas(path, format="xport")
        df = part if df is None else df.merge(part, on="SEQN", how="left")

    if df is None:
        raise ValueError("No NHANES files were loaded.")
    return df
