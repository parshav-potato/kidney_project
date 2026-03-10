"""National prevalence reconciliation via constrained quadratic optimization.

Bridges CDC national marginal prevalences with NHIS/NHANES survey microdata
intersection structure to compute P(B∪D∪P∪H) at the national level using
inclusion-exclusion, solved as a convex QP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from kidney.analysis.nhanes import _mean_bp
from kidney.analysis.prevalence import weighted_prevalence
from kidney.config import OBESITY_THRESHOLD


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NationalMarginals:
    """CDC national prevalence percentages for 4 conditions."""
    B: float = 40.3   # Obesity (BMI >= 30)
    D: float = 12.0   # Diabetes
    P: float = 41.0   # Prediabetes
    H: float = 47.7   # Hypertension


@dataclass(frozen=True)
class SurveyIntersections:
    """All 15 non-empty subset intersection prevalences from survey microdata."""
    singles: dict[str, float]           # 4 values: {B, D, P, H} -> prevalence %
    pairs: dict[tuple[str, ...], float] # 6 values
    triples: dict[tuple[str, ...], float]  # 4 values
    quadruple: float                    # 1 value: P(B∩D∩P∩H)


@dataclass
class ReconciliationResult:
    """Output of the constrained QP reconciliation."""
    national_marginals: NationalMarginals
    survey_intersections: SurveyIntersections
    optimized_pairs: dict[tuple[str, ...], float] = field(default_factory=dict)
    optimized_triples: dict[tuple[str, ...], float] = field(default_factory=dict)
    optimized_quadruple: float = 0.0
    national_union_ie: float = 0.0        # P(B∪D∪P∪H) via inclusion-exclusion
    survey_direct_union: float = 0.0      # direct survey union for comparison
    optimizer_success: bool = False
    residual_norm: float = 0.0
    n_iterations: int = 0


# ---------------------------------------------------------------------------
# Condition labels and subset ordering
# ---------------------------------------------------------------------------

CONDITIONS = ("B", "D", "P", "H")

# All subsets of size >= 2, in a fixed canonical order
PAIRS = list(combinations(CONDITIONS, 2))       # 6
TRIPLES = list(combinations(CONDITIONS, 3))     # 4
QUADRUPLE = tuple(CONDITIONS)                   # 1

# Decision variable layout: 6 pairs + 4 triples + 1 quadruple = 11
VAR_SUBSETS: list[tuple[str, ...]] = list(PAIRS) + list(TRIPLES) + [QUADRUPLE]


# ---------------------------------------------------------------------------
# Mask builders
# ---------------------------------------------------------------------------

def nhis_condition_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Boolean masks for 4 conditions from preprocessed NHIS data."""
    return {
        "B": df["BMI"] > OBESITY_THRESHOLD,
        "D": df["diabetes"] == 1,
        "P": df["prediabetes"] == 1,
        "H": df["hypertension"] == 1,
    }


def nhanes_condition_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Boolean masks for 4 conditions from raw NHANES data using lab/exam thresholds.

    Uses >=130/80 hypertension threshold (ACC/AHA 2017 guideline).
    """
    base = pd.Series(False, index=df.index, dtype=bool)

    # Diabetes: HbA1c >= 6.5 or fasting glucose >= 126
    dm = base.copy()
    if "LBXGH" in df.columns:
        dm = dm | (df["LBXGH"] >= 6.5)
    if "LBXGLU" in df.columns:
        dm = dm | (df["LBXGLU"] >= 126)
    dm = dm.fillna(False)

    # Prediabetes: (HbA1c 5.7-6.4 or glucose 100-125) and not diabetic
    predm = base.copy()
    if "LBXGH" in df.columns:
        predm = predm | ((df["LBXGH"] >= 5.7) & (df["LBXGH"] < 6.5))
    if "LBXGLU" in df.columns:
        predm = predm | ((df["LBXGLU"] >= 100) & (df["LBXGLU"] < 126))
    predm = (~dm & predm).fillna(False)

    # Hypertension: mean SBP >= 130 or mean DBP >= 80 or on medication
    systolic = _mean_bp(df, ("BPXSY", "BPXOSY"))
    diastolic = _mean_bp(df, ("BPXDI", "BPXODI"))
    htn = (systolic >= 130) | (diastolic >= 80)
    if "BPQ050A" in df.columns:
        htn = htn | (df["BPQ050A"] == 1)
    htn = htn.fillna(False)

    # Obesity: BMI >= 30
    ob = base.copy()
    if "BMXBMI" in df.columns:
        ob = (df["BMXBMI"] >= 30).fillna(False)

    return {"B": ob, "D": dm, "P": predm, "H": htn}


# ---------------------------------------------------------------------------
# Survey intersection computation
# ---------------------------------------------------------------------------

def compute_survey_intersections(
    df: pd.DataFrame,
    masks: dict[str, pd.Series],
    weight_col: str = "survey_weight",
) -> SurveyIntersections:
    """Compute all 15 intersection prevalences from microdata."""
    singles: dict[str, float] = {}
    for c in CONDITIONS:
        singles[c] = weighted_prevalence(df, masks[c])

    pairs: dict[tuple[str, ...], float] = {}
    for subset in PAIRS:
        combined = masks[subset[0]]
        for c in subset[1:]:
            combined = combined & masks[c]
        pairs[subset] = weighted_prevalence(df, combined)

    triples: dict[tuple[str, ...], float] = {}
    for subset in TRIPLES:
        combined = masks[subset[0]]
        for c in subset[1:]:
            combined = combined & masks[c]
        triples[subset] = weighted_prevalence(df, combined)

    combined_all = masks["B"]
    for c in CONDITIONS[1:]:
        combined_all = combined_all & masks[c]
    quadruple = weighted_prevalence(df, combined_all)

    return SurveyIntersections(
        singles=singles,
        pairs=pairs,
        triples=triples,
        quadruple=quadruple,
    )


def _compute_direct_union(
    df: pd.DataFrame,
    masks: dict[str, pd.Series],
) -> float:
    """Compute P(B∪D∪P∪H) directly from microdata."""
    union_mask = masks["B"]
    for c in CONDITIONS[1:]:
        union_mask = union_mask | masks[c]
    return weighted_prevalence(df, union_mask)


# ---------------------------------------------------------------------------
# QP construction and solving
# ---------------------------------------------------------------------------

def _marginal_value(national: NationalMarginals, c: str) -> float:
    return getattr(national, c)


def _build_qp(
    survey: SurveyIntersections,
    national: NationalMarginals,
):
    """Build all QP components: targets, objective, gradient, bounds, constraints.

    Returns (objective_fn, gradient_fn, x0, bounds, constraints).
    """
    n_vars = len(VAR_SUBSETS)  # 11

    # --- Scaling estimates: for each variable i, for each condition c in S_i ---
    # estimates[i] = list of target values e_ic
    estimates: list[list[float]] = [[] for _ in range(n_vars)]

    def _survey_intersection(subset: tuple[str, ...]) -> float:
        size = len(subset)
        if size == 1:
            return survey.singles[subset[0]]
        if size == 2:
            return survey.pairs[subset]
        if size == 3:
            return survey.triples[subset]
        return survey.quadruple

    for i, subset in enumerate(VAR_SUBSETS):
        s_S = _survey_intersection(subset)
        for c in subset:
            s_c = survey.singles[c]
            m_c = _marginal_value(national, c)
            if s_c > 0:
                e_ic = s_S * m_c / s_c
            else:
                e_ic = 0.0
            estimates[i].append(e_ic)

    # Precompute per-variable: n_i, sum of estimates, sum of squares
    n_i = np.array([len(e) for e in estimates], dtype=float)
    sum_e = np.array([sum(e) for e in estimates], dtype=float)
    sum_e2 = np.array([sum(v**2 for v in e) for e in estimates], dtype=float)

    # --- Objective: f(x) = Σ_i Σ_c (x_i - e_ic)^2 = Σ_i [n_i*x_i^2 - 2*x_i*sum_e_i + sum_e2_i]
    def objective(x):
        return float(np.sum(n_i * x**2 - 2 * x * sum_e + sum_e2))

    # --- Gradient: ∂f/∂x_i = 2*(n_i*x_i - sum_e_i)
    def gradient(x):
        return 2.0 * (n_i * x - sum_e)

    # --- Bounds: 0 <= x_i <= min{m(c) : c in S_i}
    bounds = []
    for subset in VAR_SUBSETS:
        upper = min(_marginal_value(national, c) for c in subset)
        bounds.append((0.0, upper))

    # --- Initial guess: unconstrained optimum clipped to bounds
    x0_raw = sum_e / n_i
    x0 = np.clip(x0_raw, [b[0] for b in bounds], [b[1] for b in bounds])

    # --- Build index lookup for constraint construction
    var_index = {subset: i for i, subset in enumerate(VAR_SUBSETS)}

    # --- Linear inequality constraints for SLSQP (fun(x) >= 0) ---
    constraints = []

    # Monotonicity: pair >= triple (for each triple, each of its 3 parent pairs)
    for triple in TRIPLES:
        i_triple = var_index[triple]
        for pair in combinations(triple, 2):
            i_pair = var_index[pair]
            # x_pair - x_triple >= 0
            constraints.append({
                "type": "ineq",
                "fun": lambda x, ip=i_pair, it=i_triple: x[ip] - x[it],
                "jac": lambda x, ip=i_pair, it=i_triple: _ineq_jac(n_vars, ip, it),
            })

    # Monotonicity: triple >= quadruple
    i_quad = var_index[QUADRUPLE]
    for triple in TRIPLES:
        i_triple = var_index[triple]
        constraints.append({
            "type": "ineq",
            "fun": lambda x, it=i_triple, iq=i_quad: x[it] - x[iq],
            "jac": lambda x, it=i_triple, iq=i_quad: _ineq_jac(n_vars, it, iq),
        })

    # Valid union: 0 <= Σm - Σx_pairs + Σx_triples - x_quad <= 100
    sum_m = sum(_marginal_value(national, c) for c in CONDITIONS)
    pair_indices = [var_index[p] for p in PAIRS]
    triple_indices = [var_index[t] for t in TRIPLES]

    def _ie_value(x):
        return sum_m - sum(x[i] for i in pair_indices) + sum(x[i] for i in triple_indices) - x[i_quad]

    def _ie_jac(x):
        j = np.zeros(n_vars)
        for i in pair_indices:
            j[i] = -1.0
        for i in triple_indices:
            j[i] = 1.0
        j[i_quad] = -1.0
        return j

    # IE >= 0
    constraints.append({
        "type": "ineq",
        "fun": lambda x: _ie_value(x),
        "jac": lambda x: _ie_jac(x),
    })
    # IE <= 100  =>  100 - IE >= 0
    constraints.append({
        "type": "ineq",
        "fun": lambda x: 100.0 - _ie_value(x),
        "jac": lambda x: -_ie_jac(x),
    })

    return objective, gradient, x0, bounds, constraints


def _ineq_jac(n_vars: int, i_pos: int, i_neg: int) -> np.ndarray:
    """Jacobian for constraint x[i_pos] - x[i_neg] >= 0."""
    j = np.zeros(n_vars)
    j[i_pos] = 1.0
    j[i_neg] = -1.0
    return j


def reconcile(
    survey: SurveyIntersections,
    national: NationalMarginals,
    survey_direct_union: float,
) -> ReconciliationResult:
    """Solve the constrained QP and compute inclusion-exclusion union."""
    objective, gradient, x0, bounds, constraints = _build_qp(survey, national)

    result = minimize(
        objective, x0, jac=gradient, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-15},
    )

    x = result.x
    var_index = {subset: i for i, subset in enumerate(VAR_SUBSETS)}

    # Unpack optimized values
    opt_pairs = {p: float(x[var_index[p]]) for p in PAIRS}
    opt_triples = {t: float(x[var_index[t]]) for t in TRIPLES}
    opt_quad = float(x[var_index[QUADRUPLE]])

    # Inclusion-exclusion
    sum_m = sum(_marginal_value(national, c) for c in CONDITIONS)
    ie_union = (
        sum_m
        - sum(opt_pairs.values())
        + sum(opt_triples.values())
        - opt_quad
    )

    return ReconciliationResult(
        national_marginals=national,
        survey_intersections=survey,
        optimized_pairs=opt_pairs,
        optimized_triples=opt_triples,
        optimized_quadruple=opt_quad,
        national_union_ie=ie_union,
        survey_direct_union=survey_direct_union,
        optimizer_success=result.success,
        residual_norm=float(np.sqrt(result.fun)) if result.fun >= 0 else 0.0,
        n_iterations=result.nit,
    )


# ---------------------------------------------------------------------------
# Top-level entry points
# ---------------------------------------------------------------------------

def reconcile_national_prevalence(
    dataframes: dict[str, pd.DataFrame],
    nhanes_df: pd.DataFrame | None = None,
    national: NationalMarginals | None = None,
) -> dict[str, ReconciliationResult]:
    """Run reconciliation using NHIS (pooled 2019-2024) and optionally NHANES.

    Returns dict with keys 'nhis' and optionally 'nhanes'.
    """
    if national is None:
        national = NationalMarginals()

    results: dict[str, ReconciliationResult] = {}

    # Pool NHIS 2019-2024 into a single DataFrame
    nhis_years = [y for y in sorted(dataframes) if int(y) >= 2019]
    if nhis_years:
        pooled = pd.concat([dataframes[y] for y in nhis_years], ignore_index=True)
        masks = nhis_condition_masks(pooled)
        survey = compute_survey_intersections(pooled, masks)
        direct_union = _compute_direct_union(pooled, masks)
        results["nhis"] = reconcile(survey, national, direct_union)

    # NHANES
    if nhanes_df is not None:
        nhanes_prepped = _prepare_nhanes_for_reconciliation(nhanes_df)
        if nhanes_prepped is not None:
            masks = nhanes_condition_masks(nhanes_prepped)
            weight_col = _find_nhanes_weight(nhanes_prepped)
            if weight_col and weight_col != "survey_weight":
                nhanes_prepped = nhanes_prepped.rename(columns={weight_col: "survey_weight"})
            survey = compute_survey_intersections(nhanes_prepped, masks)
            direct_union = _compute_direct_union(nhanes_prepped, masks)
            results["nhanes"] = reconcile(survey, national, direct_union)

    return results


def _prepare_nhanes_for_reconciliation(df: pd.DataFrame) -> pd.DataFrame | None:
    """Filter NHANES to adults with valid weights, keeping raw columns for mask builders."""
    from kidney.config import NHANES_WEIGHT_COLUMNS

    df = df.copy()
    if "RIDAGEYR" not in df.columns:
        return None
    df = df[df["RIDAGEYR"] >= 18]
    if "RIDEXPRG" in df.columns:
        df = df[~(df["RIDEXPRG"] == 1)]

    weight_col = _find_nhanes_weight(df)
    if weight_col is None:
        return None
    df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce")
    df = df[df[weight_col].notna() & (df[weight_col] > 0)]

    return df if len(df) > 0 else None


def _find_nhanes_weight(df: pd.DataFrame) -> str | None:
    from kidney.config import NHANES_WEIGHT_COLUMNS
    for c in NHANES_WEIGHT_COLUMNS:
        if c in df.columns:
            return c
    return None


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------

_CONDITION_NAMES = {"B": "Obesity", "D": "Diabetes", "P": "Prediabetes", "H": "Hypertension"}


def _subset_label(subset: tuple[str, ...]) -> str:
    return " & ".join(_CONDITION_NAMES[c] for c in subset)


def format_reconciliation_result(rec: ReconciliationResult) -> str:
    """Format reconciliation result for console output."""
    lines: list[str] = []

    lines.append(f"  Optimizer: {'converged' if rec.optimizer_success else 'FAILED'} "
                 f"(iterations={rec.n_iterations}, residual={rec.residual_norm:.4f})")

    lines.append("\n  CDC marginals: "
                 f"B={rec.national_marginals.B}%, D={rec.national_marginals.D}%, "
                 f"P={rec.national_marginals.P}%, H={rec.national_marginals.H}%")

    lines.append(f"\n  Survey singles: " + ", ".join(
        f"{_CONDITION_NAMES[c]}={rec.survey_intersections.singles[c]:.2f}%"
        for c in CONDITIONS
    ))

    lines.append("\n  Optimized pairwise intersections:")
    for subset, val in rec.optimized_pairs.items():
        survey_val = rec.survey_intersections.pairs[subset]
        lines.append(f"    {_subset_label(subset):35s}  national={val:6.2f}%  survey={survey_val:6.2f}%")

    lines.append("\n  Optimized triple intersections:")
    for subset, val in rec.optimized_triples.items():
        survey_val = rec.survey_intersections.triples[subset]
        lines.append(f"    {_subset_label(subset):35s}  national={val:6.2f}%  survey={survey_val:6.2f}%")

    survey_quad = rec.survey_intersections.quadruple
    lines.append(f"\n  Optimized quadruple intersection:")
    lines.append(f"    {'All four':35s}  national={rec.optimized_quadruple:6.2f}%  survey={survey_quad:6.2f}%")

    lines.append(f"\n  Inclusion-exclusion union (national): {rec.national_union_ie:.2f}%")
    lines.append(f"  Direct union (survey):                {rec.survey_direct_union:.2f}%")
    lines.append(f"  Difference:                           {rec.national_union_ie - rec.survey_direct_union:+.2f} pp")

    return "\n".join(lines)
