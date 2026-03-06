"""NHANES cross-validation: lab/exam-based condition prevalence."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
import pandas as pd

from kidney.config import NHANES_CONDITION_ORDER, NHANES_WEIGHT_COLUMNS


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConditionPrevalence:
    weighted_pct: float
    unweighted_pct: float
    n_cases: int
    weighted_population_cases: float


@dataclass(frozen=True)
class NHANESConditionSummary:
    cycle: str
    weight_column: str
    n_records: int
    weighted_population_total: float
    stats: dict[str, ConditionPrevalence]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_weight_column(columns) -> str:
    for c in NHANES_WEIGHT_COLUMNS:
        if c in columns:
            return c
    raise ValueError(f"No NHANES weight column found (tried {NHANES_WEIGHT_COLUMNS})")


def _mean_bp(df: pd.DataFrame, prefixes: Sequence[str]) -> pd.Series:
    cols = [c for c in df.columns if any(c.startswith(p) for p in prefixes)]
    if not cols:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return df[cols].mean(axis=1, skipna=True)


def _weighted_stats(flag: pd.Series, weight: pd.Series) -> ConditionPrevalence:
    if flag.empty:
        return ConditionPrevalence(0.0, 0.0, 0, 0.0)
    bf = flag.fillna(False).astype(bool)
    w = weight.fillna(0)
    wc = float((bf.astype(float) * w).sum())
    tw = float(w.sum())
    return ConditionPrevalence(
        weighted_pct=wc / tw * 100 if tw else 0.0,
        unweighted_pct=float(bf.mean() * 100) if len(bf) else 0.0,
        n_cases=int(bf.sum()),
        weighted_population_cases=wc,
    )


def _compute_condition_flags(df: pd.DataFrame, obesity_age_min: int = 18) -> pd.DataFrame:
    """Create boolean flags for each condition using NHANES clinical thresholds."""
    base = pd.Series(False, index=df.index, dtype=bool)
    flags = pd.DataFrame(index=df.index)

    # Diabetes: HbA1c >= 6.5 or fasting glucose >= 126
    dm = base.copy()
    if "LBXGH" in df.columns:
        dm = dm | (df["LBXGH"] >= 6.5)
    if "LBXGLU" in df.columns:
        dm = dm | (df["LBXGLU"] >= 126)
    flags["diabetes_flag"] = dm.fillna(False)

    # Prediabetes: HbA1c 5.7-6.4 or glucose 100-125 or self-report borderline, AND not diabetic
    predm = base.copy()
    if "LBXGH" in df.columns:
        predm = predm | ((df["LBXGH"] >= 5.7) & (df["LBXGH"] < 6.5))
    if "LBXGLU" in df.columns:
        predm = predm | ((df["LBXGLU"] >= 100) & (df["LBXGLU"] < 126))
    if "DIQ010" in df.columns:
        predm = predm | (df["DIQ010"] == 3)
    flags["prediabetes_flag"] = (~flags["diabetes_flag"] & predm).fillna(False)

    # Hypertension: mean SBP >= 140 or mean DBP >= 90
    systolic = _mean_bp(df, ("BPXSY", "BPXOSY"))
    diastolic = _mean_bp(df, ("BPXDI", "BPXODI"))
    flags["hypertension_flag"] = ((systolic >= 140) | (diastolic >= 90)).fillna(False)

    # Historic hypertension: told by doctor
    hist = base.copy()
    if "BPQ020" in df.columns:
        hist = df["BPQ020"] == 1
    flags["historic_hypertension_flag"] = hist.fillna(False)

    # Obesity: BMI >= 30
    ob = base.copy()
    if "BMXBMI" in df.columns:
        ob = df["BMXBMI"] >= 30
        if "RIDAGEYR" in df.columns:
            ob = ob & (df["RIDAGEYR"] >= obesity_age_min)
    flags["obesity_flag"] = ob.fillna(False)

    flags["any_condition_flag"] = (
        flags["diabetes_flag"]
        | flags["prediabetes_flag"]
        | flags["hypertension_flag"]
        | flags["historic_hypertension_flag"]
        | flags["obesity_flag"]
    )
    flags["mean_sbp"] = systolic
    flags["mean_dbp"] = diastolic
    return flags


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def summarize_nhanes_conditions(
    df: pd.DataFrame,
    *,
    cycle_label: str = "2021-2022",
    weight_column: str | None = None,
    age_min: int = 18,
    exclude_pregnancy: bool = True,
) -> NHANESConditionSummary:
    """Compute weighted prevalence of key conditions using NHANES data."""
    if weight_column is None:
        weight_column = _get_weight_column(df.columns)

    df = df.copy()
    if "RIDAGEYR" not in df.columns:
        raise ValueError("NHANES dataframe must include RIDAGEYR.")
    df = df[df["RIDAGEYR"] >= age_min]

    if exclude_pregnancy and "RIDEXPRG" in df.columns:
        df = df[~(df["RIDEXPRG"] == 1)]

    df["analysis_weight"] = pd.to_numeric(df[weight_column], errors="coerce")
    df = df[df["analysis_weight"].notna() & (df["analysis_weight"] > 0)]

    flags = _compute_condition_flags(df, obesity_age_min=age_min)
    df = pd.concat([df, flags], axis=1)
    weight = df["analysis_weight"]

    stats: dict[str, ConditionPrevalence] = {}
    for label, col in [
        ("Diabetes", "diabetes_flag"),
        ("Prediabetes", "prediabetes_flag"),
        ("Hypertension", "hypertension_flag"),
        ("Historic Hypertension", "historic_hypertension_flag"),
        ("Obesity", "obesity_flag"),
        ("Any Condition", "any_condition_flag"),
    ]:
        stats[label] = _weighted_stats(df[col], weight)

    return NHANESConditionSummary(
        cycle=cycle_label,
        weight_column=weight_column,
        n_records=len(df),
        weighted_population_total=float(weight.sum()),
        stats=stats,
    )


def format_condition_summary(summary: NHANESConditionSummary) -> str:
    lines = [
        f"NHANES {summary.cycle} adults (n={summary.n_records}, "
        f"{summary.weight_column} sum = {summary.weighted_population_total:,.0f})"
    ]
    for cond in NHANES_CONDITION_ORDER:
        s = summary.stats.get(cond)
        if s:
            lines.append(
                f" - {cond}: {s.weighted_pct:.2f}% weighted "
                f"({s.unweighted_pct:.2f}% unweighted, cases={s.n_cases:,})"
            )
    return "\n".join(lines)


def prepare_nhanes_for_comparison(
    df: pd.DataFrame,
    weight_column: str | None = None,
    age_min: int = 18,
) -> pd.DataFrame:
    """Map NHANES to NHIS-compatible column names and 1/2 coding."""
    df = df.copy()
    if weight_column is None:
        weight_column = _get_weight_column(df.columns)
    if "RIDAGEYR" not in df.columns:
        raise ValueError("NHANES dataframe must include RIDAGEYR.")
    df = df[df["RIDAGEYR"] >= age_min]
    if "RIDEXPRG" in df.columns:
        df = df[~(df["RIDEXPRG"] == 1)]

    df[weight_column] = pd.to_numeric(df[weight_column], errors="coerce")
    df = df[df[weight_column].notna() & (df[weight_column] > 0)]

    flags = _compute_condition_flags(df, obesity_age_min=age_min)

    out = pd.DataFrame(index=df.index)
    out["diabetes"] = flags["diabetes_flag"].map({True: 1, False: 2})
    out["prediabetes"] = flags["prediabetes_flag"].map({True: 1, False: 2})
    out["hypertension"] = flags["historic_hypertension_flag"].map({True: 1, False: 2})
    out["historic_hypertension"] = flags["historic_hypertension_flag"].map({True: 1, False: 2})
    out["BMI"] = df["BMXBMI"] if "BMXBMI" in df.columns else np.nan
    out["age"] = df["RIDAGEYR"]
    out["survey_weight"] = df[weight_column]

    if "RIDRETH3" in df.columns:
        out["race_ethnicity"] = df["RIDRETH3"].map({1: 1, 2: 1, 3: 2, 4: 3, 6: 4})
    else:
        out["race_ethnicity"] = np.nan

    if "DMDEDUC2" in df.columns:
        out["education"] = df["DMDEDUC2"].map({1: 1, 2: 1, 3: 3, 4: 5, 5: 8})
    else:
        out["education"] = np.nan

    return out.dropna(subset=["diabetes", "prediabetes", "hypertension", "BMI", "survey_weight"])
