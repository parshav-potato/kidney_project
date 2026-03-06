"""Donor category analysis: 8 marginal groups, population segments, Venn data."""

import numpy as np
import pandas as pd

from kidney.analysis.prevalence import weighted_prevalence
from kidney.analysis.trends import fit_linear_trend, project_trend

DONOR_CATEGORY_LABELS = [
    "Ideal (No HTN, No PreDM, BMI<30)",
    "BMI 30-34.9 Only",
    "Prediabetes Only",
    "Prediabetes + BMI 30-34.9",
    "HTN Only",
    "HTN + BMI 30-34.9",
    "HTN + Prediabetes",
    "HTN + PreDM + BMI 30-34.9 (Excluded)",
]


def calculate_donor_categories(
    df: pd.DataFrame, bmi_expanded: float = 35.0
) -> dict[str, dict[str, float]]:
    """Compute 8 mutually-exclusive donor categories (all require no diabetes)."""
    no_dm = df["diabetes"] != 1
    htn = df["hypertension"] == 1
    predm = df["prediabetes"] == 1
    obese_band = (df["BMI"] >= 30) & (df["BMI"] < bmi_expanded)
    bmi_under_30 = df["BMI"] < 30

    total_w = df["survey_weight"].sum() if "survey_weight" in df.columns else len(df)

    definitions = [
        ("Ideal (No HTN, No PreDM, BMI<30)", no_dm & ~htn & ~predm & bmi_under_30),
        ("BMI 30-34.9 Only", no_dm & ~htn & ~predm & obese_band),
        ("Prediabetes Only", no_dm & ~htn & predm & bmi_under_30),
        ("Prediabetes + BMI 30-34.9", no_dm & ~htn & predm & obese_band),
        ("HTN Only", no_dm & htn & ~predm & bmi_under_30),
        ("HTN + BMI 30-34.9", no_dm & htn & ~predm & obese_band),
        ("HTN + Prediabetes", no_dm & htn & predm & bmi_under_30),
        ("HTN + PreDM + BMI 30-34.9 (Excluded)", no_dm & htn & predm & obese_band),
    ]

    results = {}
    for label, mask in definitions:
        if "survey_weight" in df.columns:
            count = float(df.loc[mask, "survey_weight"].sum())
        else:
            count = float(mask.sum())
        pct = count / total_w * 100 if total_w > 0 else 0.0
        results[label] = {"pct": pct, "count": count, "n": int(mask.sum())}
    return results


def calculate_donor_category_trends(
    dataframes: dict[str, pd.DataFrame], bmi_expanded: float = 35.0
) -> dict[str, dict[str, list[float]]]:
    """Donor category percentages and counts across all years."""
    years = sorted(dataframes)
    trends = {label: {"pcts": [], "counts": []} for label in DONOR_CATEGORY_LABELS}
    for year in years:
        cats = calculate_donor_categories(dataframes[year], bmi_expanded)
        for label in DONOR_CATEGORY_LABELS:
            entry = cats.get(label, {"pct": 0, "count": 0})
            trends[label]["pcts"].append(entry["pct"])
            trends[label]["counts"].append(entry["count"])
    return trends


def calculate_ideal_donor_projections(
    dataframes: dict[str, pd.DataFrame],
    fit_start: int = 2019,
    project_to: int = 2044,
) -> dict:
    """Fit linear trend to ideal-donor eligibility and project forward."""
    years = sorted(dataframes)
    ideal_pcts = []
    expanded_pcts = []

    for year in years:
        df = dataframes[year]
        ideal = (
            (df["diabetes"] != 1)
            & (df["prediabetes"] != 1)
            & (df["hypertension"] != 1)
            & (df["BMI"] < 30)
        )
        expanded = (
            (df["diabetes"] != 1)
            & (df["prediabetes"] != 1)
            & (df["hypertension"] != 1)
            & (df["BMI"] < 35)
        )
        ideal_pcts.append(weighted_prevalence(df, ideal))
        expanded_pcts.append(weighted_prevalence(df, expanded))

    model, yrs_fit, _ = fit_linear_trend(years, ideal_pcts, fit_start)
    base_year = int(yrs_fit[0])
    slope = float(model.coef_[0])
    proj_years = sorted({2024, 2034, project_to})
    proj_values = list(project_trend(model, base_year, proj_years))

    return {
        "years": years,
        "ideal_pcts": ideal_pcts,
        "expanded_pcts": expanded_pcts,
        "model": model,
        "base_year": base_year,
        "slope_pp_per_yr": slope,
        "proj_years": proj_years,
        "proj_values": proj_values,
    }


def calculate_venn_diagram_data(
    df: pd.DataFrame, obesity_threshold: float = 30.0
) -> dict[str, float]:
    """7 exclusive regions of a 3-set Venn (Prediabetes, Hypertension, Obesity)."""
    A = df["prediabetes"] == 1
    B = df["hypertension"] == 1
    C = df["BMI"] > obesity_threshold
    return {
        "only_prediabetes": weighted_prevalence(df, A & ~B & ~C),
        "only_hypertension": weighted_prevalence(df, ~A & B & ~C),
        "only_obesity": weighted_prevalence(df, ~A & ~B & C),
        "prediabetes_and_hypertension": weighted_prevalence(df, A & B & ~C),
        "prediabetes_and_obesity": weighted_prevalence(df, A & ~B & C),
        "hypertension_and_obesity": weighted_prevalence(df, ~A & B & C),
        "all_three": weighted_prevalence(df, A & B & C),
    }


def calculate_eligibility_venn_data(
    df: pd.DataFrame, bmi_threshold: float = 30.0
) -> dict[str, float]:
    """DM/PreDM/HTN Venn regions among adults with BMI below threshold."""
    sub = df[df["BMI"] < bmi_threshold].copy()
    if "survey_weight" in df.columns and "survey_weight" in sub.columns:
        universe_pct = sub["survey_weight"].sum() / df["survey_weight"].sum() * 100
    else:
        universe_pct = len(sub) / len(df) * 100 if len(df) else 0

    A = sub["diabetes"] == 1
    B = (sub["prediabetes"] == 1) & ~A
    C = sub["hypertension"] == 1

    only_a = weighted_prevalence(sub, A & ~B & ~C)
    only_b = weighted_prevalence(sub, ~A & B & ~C)
    only_c = weighted_prevalence(sub, ~A & ~B & C)
    ab = weighted_prevalence(sub, A & B & ~C)
    ac = weighted_prevalence(sub, A & ~B & C)
    bc = weighted_prevalence(sub, ~A & B & C)
    abc = weighted_prevalence(sub, A & B & C)
    ideal = weighted_prevalence(sub, ~A & ~B & ~C)

    return {
        "only_dm": only_a,
        "only_predm": only_b,
        "only_htn": only_c,
        "dm_predm": ab,
        "dm_htn": ac,
        "predm_htn": bc,
        "all_three": abc,
        "ideal_pct": ideal,
        "total_dm": only_a + ab + ac + abc,
        "total_predm": only_b + ab + bc + abc,
        "total_htn": only_c + ac + bc + abc,
        "universe_pct": universe_pct,
    }


def calculate_population_segments(
    df: pd.DataFrame, bmi_expanded: float = 35.0
) -> dict[str, dict[str, float] | float | int]:
    """18 mutually-exclusive population segments (3 glycaemic x 2 HTN x 3 BMI)."""
    total_w = df["survey_weight"].sum() if "survey_weight" in df.columns else len(df)

    dm = df["diabetes"] == 1
    predm = (df["prediabetes"] == 1) & ~dm
    neither_glyc = ~dm & ~predm
    htn_yes = df["hypertension"] == 1
    htn_no = ~htn_yes
    bmi_lt30 = df["BMI"] < 30
    bmi_30_35 = (df["BMI"] >= 30) & (df["BMI"] < bmi_expanded)
    bmi_gte35 = df["BMI"] >= bmi_expanded

    glyc = [("DM", dm), ("PreDM", predm), ("Neither", neither_glyc)]
    htn = [("HTN", htn_yes), ("NoHTN", htn_no)]
    bmi = [("BMI<30", bmi_lt30), ("BMI30-35", bmi_30_35), ("BMI>=35", bmi_gte35)]

    def _wpct(mask):
        cnt = float(df.loc[mask, "survey_weight"].sum()) if "survey_weight" in df.columns else float(mask.sum())
        return {"pct": cnt / total_w * 100 if total_w > 0 else 0.0, "count": cnt}

    segments: dict = {}
    for g_name, g_mask in glyc:
        for h_name, h_mask in htn:
            for b_name, b_mask in bmi:
                segments[f"{g_name}_{h_name}_{b_name}"] = _wpct(g_mask & h_mask & b_mask)

    segments["total_weight"] = total_w
    segments["year_n"] = len(df)
    segments["dm_total_pct"] = _wpct(dm)["pct"]
    segments["predm_total_pct"] = _wpct(predm)["pct"]
    segments["neither_total_pct"] = _wpct(neither_glyc)["pct"]
    segments["htn_total_pct"] = _wpct(htn_yes)["pct"]
    segments["bmi_lt30_total_pct"] = _wpct(bmi_lt30)["pct"]
    segments["bmi_30_35_total_pct"] = _wpct(bmi_30_35)["pct"]
    segments["bmi_gte35_total_pct"] = _wpct(bmi_gte35)["pct"]
    return segments


def calculate_impact_of_bmi_relaxation(
    dataframes: dict[str, pd.DataFrame],
    strict_bmi: float = 30.0,
    relaxed_bmi: float = 35.0,
) -> dict[str, float]:
    """Impact of changing BMI threshold from strict to relaxed."""
    from kidney.analysis.eligibility import eligibility_metrics

    years = sorted(dataframes)
    eligible_strict, _ = eligibility_metrics(dataframes, strict_bmi)
    eligible_relaxed, _ = eligibility_metrics(dataframes, relaxed_bmi)

    model_s, yrs_s, _ = fit_linear_trend(years, eligible_strict, 2019)
    model_r, _, _ = fit_linear_trend(years, eligible_relaxed, 2019)
    base = int(yrs_s[0])

    from kidney.config import US_ADULT_POPULATION_2025

    p25_s, p25_r = project_trend(model_s, base, [2025])[0], project_trend(model_r, base, [2025])[0]
    p29_s, p29_r = project_trend(model_s, base, [2029])[0], project_trend(model_r, base, [2029])[0]

    idx_2019 = years.index("2019") if "2019" in years else 0
    pct_inc = p25_r - p25_s
    return {
        "baseline_2019": eligible_strict[idx_2019],
        "baseline_2019_relaxed": eligible_relaxed[idx_2019],
        "projected_2025_strict": float(p25_s),
        "projected_2025_relaxed": float(p25_r),
        "projected_2029_strict": float(p29_s),
        "projected_2029_relaxed": float(p29_r),
        "percentage_point_increase": float(pct_inc),
        "additional_individuals": float(pct_inc / 100 * US_ADULT_POPULATION_2025),
        "additional_millions": round(pct_inc / 100 * US_ADULT_POPULATION_2025 / 1_000_000, 1),
    }


def generate_summary_text(impact: dict[str, float]) -> str:
    """Format impact dict into a human-readable paragraph."""
    return (
        f"Using KDIGO-recommended criteria (BMI <35), we found that the percentage of the U.S. population "
        f"eligible for living kidney donation has declined steadily, from {impact['baseline_2019_relaxed']:.1f}% "
        f"in 2019 to {impact['projected_2025_relaxed']:.1f}% in 2025 (projected). "
        f"If current trends continue, eligibility is projected to fall further to approximately "
        f"{impact['projected_2029_relaxed']:.1f}% by 2029. Using a more conservative BMI criterion of <30 "
        f"would decrease the eligible population by {impact['percentage_point_increase']:.1f} percentage points "
        f"in 2025, reducing it to {impact['projected_2025_strict']:.1f}% and excluding approximately "
        f"{impact['additional_millions']} million individuals who meet the KDIGO BMI guidelines for donation."
    )
