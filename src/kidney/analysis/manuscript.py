"""Manuscript-ready statistics with 95% confidence intervals."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from kidney.analysis.prevalence import weighted_prevalence, weighted_stats
from kidney.config import (
    ELIGIBILITY_BMI_THRESHOLD,
    EDUCATION_CATEGORIES,
    EDUCATION_LABELS,
    OBESITY_THRESHOLD,
    POVERTY_CATEGORIES,
    POVERTY_CATEGORY_LABELS,
    RACE_ETHNICITY_CODES,
)


def calculate_time_period_prevalences(
    dataframes: dict[str, pd.DataFrame],
    obesity_threshold: float = OBESITY_THRESHOLD,
) -> dict[str, dict[str, tuple[float, float, float]]]:
    """Prevalence + 95% CI for 3 time periods."""
    periods = {
        "2015-2018": ["2015", "2016", "2017", "2018"],
        "2019-2021": ["2019", "2020", "2021"],
        "2022-2024": ["2022", "2023", "2024"],
    }
    results: dict = {}
    for name, years in periods.items():
        dfs = [dataframes[y] for y in years if y in dataframes]
        if not dfs:
            continue
        df = pd.concat(dfs, ignore_index=True)
        conditions = {
            "diabetes": weighted_stats(df, df["diabetes"] == 1),
            "prediabetes": weighted_stats(df, df["prediabetes"] == 1),
            "hypertension": weighted_stats(df, df["hypertension"] == 1),
            "obesity": weighted_stats(df, df["BMI"] > obesity_threshold),
        }
        eligible = (
            (df["diabetes"] != 1)
            & (df["hypertension"] != 1)
            & (df["BMI"] < ELIGIBILITY_BMI_THRESHOLD)
        )
        conditions["eligible"] = weighted_stats(df, eligible)
        results[name] = conditions
    return results


def calculate_race_eligibility_comparison(
    dataframes: dict[str, pd.DataFrame],
) -> dict[str, tuple[float, float, float]]:
    """Eligibility by race/ethnicity, pooling 2019-2024."""
    dfs = [
        dataframes[y]
        for y in ("2019", "2020", "2021", "2022", "2023", "2024")
        if y in dataframes and "race_ethnicity" in dataframes[y].columns
    ]
    if not dfs:
        return {}
    df = pd.concat(dfs, ignore_index=True)
    results = {}
    for code, name in RACE_ETHNICITY_CODES.items():
        sub = df[df["race_ethnicity"] == code]
        if len(sub) == 0:
            continue
        eligible = (
            (sub["diabetes"] != 1)
            & (sub["hypertension"] != 1)
            & (sub["BMI"] < ELIGIBILITY_BMI_THRESHOLD)
        )
        results[name] = weighted_stats(sub, eligible)
    return results


def calculate_education_eligibility_comparison(
    dataframes: dict[str, pd.DataFrame],
) -> dict[str, tuple[float, float, float]]:
    """Eligibility by education level, pooling 2021-2024."""
    dfs = [
        dataframes[y]
        for y in ("2021", "2022", "2023", "2024")
        if y in dataframes and "education" in dataframes[y].columns
    ]
    if not dfs:
        return {}
    df = pd.concat(dfs, ignore_index=True)
    results = {}
    for (_, (lo, hi)), label in zip(EDUCATION_CATEGORIES.items(), EDUCATION_LABELS):
        sub = df[(df["education"] >= lo) & (df["education"] < hi)]
        if len(sub) == 0:
            continue
        eligible = (
            (sub["diabetes"] != 1)
            & (sub["hypertension"] != 1)
            & (sub["BMI"] < ELIGIBILITY_BMI_THRESHOLD)
        )
        results[label] = weighted_stats(sub, eligible)
    return results


def calculate_poverty_eligibility_comparison(
    dataframes: dict[str, pd.DataFrame],
) -> dict[str, tuple[float, float, float]]:
    """Eligibility by poverty level, pooling 2019-2024."""
    dfs = [
        dataframes[y]
        for y in ("2019", "2020", "2021", "2022", "2023", "2024")
        if y in dataframes and "poverty_ratio" in dataframes[y].columns
    ]
    if not dfs:
        return {}
    df = pd.concat(dfs, ignore_index=True)
    results = {}
    for (_, (lo, hi)), label in zip(POVERTY_CATEGORIES.items(), POVERTY_CATEGORY_LABELS):
        sub = df[(df["poverty_ratio"] >= lo) & (df["poverty_ratio"] < hi)]
        if len(sub) == 0:
            continue
        eligible = (
            (sub["diabetes"] != 1)
            & (sub["hypertension"] != 1)
            & (sub["BMI"] < ELIGIBILITY_BMI_THRESHOLD)
        )
        results[label] = weighted_stats(sub, eligible)
    return results


def calculate_impact_excluding_prediabetes(
    dataframes: dict[str, pd.DataFrame],
) -> dict[str, float]:
    """Compare eligibility with vs without prediabetes exclusion (2022-2024)."""
    dfs = [dataframes[y] for y in ("2022", "2023", "2024") if y in dataframes]
    if not dfs:
        return {}
    df = pd.concat(dfs, ignore_index=True)

    with_predm = (
        (df["diabetes"] != 1)
        & (df["hypertension"] != 1)
        & (df["BMI"] < ELIGIBILITY_BMI_THRESHOLD)
    )
    without_predm = with_predm & (df["prediabetes"] != 1)

    pct_with = weighted_prevalence(df, with_predm)
    pct_without = weighted_prevalence(df, without_predm)
    return {
        "with_prediabetes": pct_with,
        "without_prediabetes": pct_without,
        "difference": pct_with - pct_without,
        "percentage_drop": (pct_with - pct_without) / pct_with * 100 if pct_with else 0,
    }


def calculate_bmi_threshold_impact(
    dataframes: dict[str, pd.DataFrame],
    strict_bmi: float = 30.0,
    relaxed_bmi: float = 35.0,
) -> dict[str, float]:
    """Compare eligibility at BMI<30 vs BMI<35 (2022-2024)."""
    dfs = [dataframes[y] for y in ("2022", "2023", "2024") if y in dataframes]
    if not dfs:
        return {}
    df = pd.concat(dfs, ignore_index=True)

    base = (df["diabetes"] != 1) & (df["hypertension"] != 1)
    pct_s = weighted_prevalence(df, base & (df["BMI"] < strict_bmi))
    pct_r = weighted_prevalence(df, base & (df["BMI"] < relaxed_bmi))
    return {
        "strict_bmi_30": pct_s,
        "relaxed_bmi_35": pct_r,
        "difference": pct_r - pct_s,
        "percentage_increase": (pct_r - pct_s) / pct_s * 100 if pct_s else 0,
    }


def generate_manuscript_statistics() -> dict:
    """Load data, run all manuscript analyses, return results dict."""
    from kidney.data.loader import load_nhis
    from kidney.data.preprocessing import preprocess
    from kidney.analysis.trends import trend_p_values

    print("Loading and preprocessing data...")
    dataframes = preprocess(load_nhis())

    print("\nCalculating manuscript statistics...\n")
    stats: dict = {}

    print("- Time period prevalences")
    stats["time_periods"] = calculate_time_period_prevalences(dataframes)

    print("- Trend p-values")
    stats["p_values"] = trend_p_values(dataframes)

    print("- Race/ethnicity eligibility")
    stats["race_eligibility"] = calculate_race_eligibility_comparison(dataframes)

    print("- Poverty level eligibility")
    stats["poverty_eligibility"] = calculate_poverty_eligibility_comparison(dataframes)

    print("- Education level eligibility")
    stats["education_eligibility"] = calculate_education_eligibility_comparison(dataframes)

    print("- Impact of excluding prediabetes")
    stats["prediabetes_impact"] = calculate_impact_excluding_prediabetes(dataframes)

    print("- BMI threshold impact")
    stats["bmi_impact"] = calculate_bmi_threshold_impact(dataframes)

    return stats


def main() -> None:
    """Entry point for kidney-manuscript console script."""
    stats = generate_manuscript_statistics()

    print("\n" + "=" * 80)
    print("MANUSCRIPT STATISTICS")
    print("=" * 80)

    print("\n1. TIME PERIOD PREVALENCES (Mean [95% CI]):")
    print("-" * 80)
    for period, conditions in stats["time_periods"].items():
        print(f"\n{period}:")
        for condition, (mean, lower, upper) in conditions.items():
            print(f"  {condition.capitalize()}: {mean:.1f}% ({lower:.1f}-{upper:.1f}%)")

    print("\n\n2. TREND P-VALUES:")
    print("-" * 80)
    for condition, p_val in stats["p_values"].items():
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "NS"
        print(f"  {condition.capitalize()}: p={p_val:.4f} {sig}")

    print("\n\n3. ELIGIBILITY BY RACE/ETHNICITY (Mean [95% CI]):")
    print("-" * 80)
    for race, (mean, lower, upper) in stats["race_eligibility"].items():
        print(f"  {race}: {mean:.1f}% ({lower:.1f}-{upper:.1f}%)")

    print("\n\n4. ELIGIBILITY BY POVERTY LEVEL (Mean [95% CI]):")
    print("-" * 80)
    for pov, (mean, lower, upper) in stats["poverty_eligibility"].items():
        print(f"  {pov}: {mean:.1f}% ({lower:.1f}-{upper:.1f}%)")

    print("\n\n5. ELIGIBILITY BY EDUCATION LEVEL (Mean [95% CI]):")
    print("-" * 80)
    for edu, (mean, lower, upper) in stats["education_eligibility"].items():
        print(f"  {edu}: {mean:.1f}% ({lower:.1f}-{upper:.1f}%)")

    print("\n\n6. IMPACT OF EXCLUDING PREDIABETES:")
    print("-" * 80)
    for key, value in stats["prediabetes_impact"].items():
        print(f"  {key}: {value:.1f}%")

    print("\n\n7. IMPACT OF BMI THRESHOLD:")
    print("-" * 80)
    for key, value in stats["bmi_impact"].items():
        print(f"  {key}: {value:.1f}%")

    # Save JSON
    print("\n\nSaving statistics to manuscript_stats.json...")
    json_stats: dict = {}
    for key, value in stats.items():
        if isinstance(value, dict):
            json_stats[key] = {}
            for k, v in value.items():
                if isinstance(v, dict):
                    json_stats[key][k] = {
                        kk: list(vv) if isinstance(vv, tuple) else vv
                        for kk, vv in v.items()
                    }
                elif isinstance(v, tuple):
                    json_stats[key][k] = list(v)
                else:
                    json_stats[key][k] = v
        else:
            json_stats[key] = value

    with open("manuscript_stats.json", "w") as f:
        json.dump(json_stats, f, indent=2)
    print("Done!")


if __name__ == "__main__":
    main()
