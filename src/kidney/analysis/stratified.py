"""Generic stratified analysis framework.

Replaces 8 near-identical functions (regional, age-group, insurance,
poverty, race, education prevalences and eligibility) with a single
generic ``stratified_analysis`` driven by declarative ``Stratifier`` objects.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pandas as pd

from kidney.analysis.prevalence import weighted_prevalence
from kidney.config import (
    AGE_GROUPS,
    AGE_GROUP_LABELS,
    EDUCATION_CATEGORIES,
    EDUCATION_LABELS,
    INSURANCE_STATUS,
    POVERTY_CATEGORIES,
    POVERTY_CATEGORY_LABELS,
    RACE_ETHNICITY_CODES,
    REGIONS,
)


@dataclass(frozen=True)
class Category:
    label: str
    filter: Callable[[pd.DataFrame], pd.DataFrame]


@dataclass(frozen=True)
class Stratifier:
    name: str
    categories: Sequence[Category]
    min_year: int = 2015


def stratified_analysis(
    dataframes: dict[str, pd.DataFrame],
    stratifier: Stratifier,
    metric_fn: Callable[[pd.DataFrame], pd.Series],
) -> dict[str, list[float]]:
    """Compute weighted prevalence of *metric_fn* across years, stratified
    by the categories in *stratifier*.

    Args:
        dataframes: Year-keyed DataFrames.
        stratifier: Defines how to slice the data.
        metric_fn: Given a (sub-)DataFrame, returns a boolean mask.

    Returns:
        ``{category_label: [pct_per_year, ...]}`` for years >= stratifier.min_year.
    """
    years = sorted(y for y in dataframes if int(y) >= stratifier.min_year)
    result: dict[str, list[float]] = {cat.label: [] for cat in stratifier.categories}

    for year in years:
        df = dataframes[year]
        for cat in stratifier.categories:
            sub = cat.filter(df)
            if len(sub) == 0:
                result[cat.label].append(0.0)
            else:
                result[cat.label].append(weighted_prevalence(sub, metric_fn(sub)))
    return result


# ---------------------------------------------------------------------------
# Pre-built stratifier factories
# ---------------------------------------------------------------------------

def region_stratifier() -> Stratifier:
    return Stratifier(
        name="region",
        categories=[
            Category(name, lambda df, code=code: df[df["region"] == code])
            for code, name in REGIONS.items()
        ],
    )


def age_stratifier() -> Stratifier:
    return Stratifier(
        name="age_group",
        categories=[
            Category(
                label,
                lambda df, lo=lo, hi=hi: df[(df["age"] >= lo) & (df["age"] <= hi)],
            )
            for (lo, hi), label in zip(AGE_GROUPS, AGE_GROUP_LABELS)
        ],
    )


def insurance_stratifier() -> Stratifier:
    return Stratifier(
        name="insurance",
        categories=[
            Category(name, lambda df, code=code: df[df["insurance"] == code])
            for code, name in INSURANCE_STATUS.items()
        ],
        min_year=2019,
    )


def poverty_stratifier() -> Stratifier:
    return Stratifier(
        name="poverty",
        categories=[
            Category(
                label,
                lambda df, lo=lo, hi=hi: df[
                    (df["poverty_ratio"] >= lo) & (df["poverty_ratio"] < hi)
                ],
            )
            for (_, (lo, hi)), label in zip(
                POVERTY_CATEGORIES.items(), POVERTY_CATEGORY_LABELS
            )
        ],
        min_year=2019,
    )


def race_stratifier() -> Stratifier:
    return Stratifier(
        name="race_ethnicity",
        categories=[
            Category(name, lambda df, code=code: df[df["race_ethnicity"] == code])
            for code, name in RACE_ETHNICITY_CODES.items()
        ],
        min_year=2019,
    )


def education_stratifier() -> Stratifier:
    return Stratifier(
        name="education",
        categories=[
            Category(
                label,
                lambda df, lo=lo, hi=hi: df[
                    (df["education"] >= lo) & (df["education"] < hi)
                ],
            )
            for (_, (lo, hi)), label in zip(
                EDUCATION_CATEGORIES.items(), EDUCATION_LABELS
            )
        ],
        min_year=2021,
    )
