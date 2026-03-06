"""Five-step NHIS preprocessing pipeline."""

import numpy as np
import pandas as pd


def preprocess(dataframes: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Run the full preprocessing pipeline: filter adults, fill missing,
    clean invalid, calculate BMI, remove incomplete records."""
    print("Starting data preprocessing...")
    dataframes = _filter_adults(dataframes)
    print("  Filtered to adults")
    dataframes = _fill_missing_health(dataframes)
    print("  Filled missing health values")
    dataframes = _clean_invalid(dataframes)
    print("  Cleaned invalid values")
    dataframes = _calculate_bmi(dataframes)
    print("  Calculated BMI")
    dataframes = _remove_incomplete(dataframes)
    print("  Removed incomplete records")
    print("Preprocessing complete!")
    return dataframes


def _filter_adults(dfs: dict[str, pd.DataFrame], min_age: int = 18) -> dict[str, pd.DataFrame]:
    return {y: df[df["age"] >= min_age].copy() for y, df in dfs.items()}


def _fill_missing_health(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    health_cols = ("diabetes", "hypertension", "prediabetes", "historic_hypertension")
    result = {}
    for y, df in dfs.items():
        df = df.copy()
        for col in health_cols:
            if col in df.columns:
                df[col] = df[col].fillna(2)
        result[y] = df
    return result


def _clean_invalid(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    result = {}
    for y, df in dfs.items():
        df = df.copy()
        df["Height"] = df["Height"].where(df["Height"] < 96, np.nan)
        df["Weight"] = df["Weight"].where(df["Weight"] < 996, np.nan)

        for col in ("diabetes", "prediabetes", "hypertension", "historic_hypertension"):
            if col in df.columns:
                df[col] = df[col].apply(lambda x: x if x in {1, 2, 1.0, 2.0} else np.nan)

        if "poverty_ratio" in df.columns:
            df["poverty_ratio"] = df["poverty_ratio"].where(
                (df["poverty_ratio"] >= 0) & (df["poverty_ratio"] <= 11), np.nan
            )
        if "race_ethnicity" in df.columns:
            df["race_ethnicity"] = df["race_ethnicity"].where(
                df["race_ethnicity"].isin([1, 2, 3, 4]), np.nan
            )
        if "education" in df.columns:
            df["education"] = df["education"].where(
                (df["education"] >= 0) & (df["education"] <= 10), np.nan
            )
        if "survey_weight" in df.columns:
            df["survey_weight"] = pd.to_numeric(df["survey_weight"], errors="coerce")
            df["survey_weight"] = df["survey_weight"].where(df["survey_weight"] > 0, np.nan)

        df = df.dropna(how="all")
        result[y] = df
    return result


def _calculate_bmi(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    result = {}
    for y, df in dfs.items():
        df = df.copy()
        df["BMI"] = (df["Weight"] / (df["Height"] ** 2)) * 703
        result[y] = df
    return result


def _remove_incomplete(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    result = {}
    for y, df in dfs.items():
        clean = df.dropna()
        print(f"Year {y}: {clean.shape[0]} complete records "
              f"(removed {df.shape[0] - clean.shape[0]} incomplete)")
        result[y] = clean
    return result
