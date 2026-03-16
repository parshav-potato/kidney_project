"""National prevalence reconciliation via Iterative Proportional Fitting (IPF).

Alternative to the constrained QP approach in national_reconciliation.py.
IPF adjusts a 2x2x2x2 joint distribution table (B, D, P, H) so that its
marginals match CDC national prevalences, while preserving the cross-product
ratios (odds ratios) from the original survey data.  Equivalent to minimising
KL-divergence from the survey distribution subject to marginal constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from kidney.analysis.national_reconciliation import (
    CONDITIONS,
    PAIRS,
    TRIPLES,
    QUADRUPLE,
    NationalMarginals,
    SurveyIntersections,
    ReconciliationResult,
    nhis_condition_masks,
    nhanes_condition_masks,
    compute_survey_intersections,
    _compute_direct_union,
    _marginal_value,
    _prepare_nhanes_for_reconciliation,
    _find_nhanes_weight,
)
from kidney.config import IPF_MAX_ITERATIONS, IPF_TOLERANCE, IPF_CELL_FLOOR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AXIS_MAP: dict[str, int] = {"B": 0, "D": 1, "P": 2, "H": 3}

_CONDITION_NAMES = {"B": "Obesity", "D": "Diabetes", "P": "Prediabetes", "H": "Hypertension"}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class IPFResult:
    """Output of the IPF algorithm before conversion to ReconciliationResult."""
    initial_table: np.ndarray
    converged_table: np.ndarray
    converged: bool
    n_iterations: int
    max_marginal_error: float
    iteration_history: list[dict]
    target_marginals: dict[str, float]


# ---------------------------------------------------------------------------
# Joint table computation
# ---------------------------------------------------------------------------

def compute_joint_table(
    df: pd.DataFrame,
    masks: dict[str, pd.Series],
    weight_col: str = "survey_weight",
) -> np.ndarray:
    """Compute the 2x2x2x2 joint distribution from survey microdata.

    Returns a numpy array of shape (2,2,2,2) with axes (B, D, P, H).
    Index 0 = condition absent, index 1 = condition present.
    Table sums to 1.0 (proportions).
    """
    b = masks["B"].astype(int).values
    d = masks["D"].astype(int).values
    p = masks["P"].astype(int).values
    h = masks["H"].astype(int).values

    cell_idx = b * 8 + d * 4 + p * 2 + h

    if weight_col in df.columns:
        weights = df[weight_col].values.astype(float)
    else:
        weights = np.ones(len(df), dtype=float)

    counts = np.zeros(16)
    for i in range(16):
        counts[i] = weights[cell_idx == i].sum()

    table = counts / counts.sum()
    return table.reshape(2, 2, 2, 2)


# ---------------------------------------------------------------------------
# IPF algorithm
# ---------------------------------------------------------------------------

def _marginal_sum(table: np.ndarray, axis: int, index: int) -> float:
    """Sum of all cells where the given axis has the given index value."""
    idx = [slice(None)] * 4
    idx[axis] = index
    return float(table[tuple(idx)].sum())


def ipf_fit(
    table: np.ndarray,
    target_marginals: dict[str, float],
    max_iter: int = IPF_MAX_ITERATIONS,
    tol: float = IPF_TOLERANCE,
    floor: float = IPF_CELL_FLOOR,
) -> IPFResult:
    """Run IPF on a 2x2x2x2 table to match target marginals.

    Parameters
    ----------
    table : (2,2,2,2) ndarray summing to 1
        Initial joint distribution (proportions, 0-1 scale).
    target_marginals : dict mapping condition letter to proportion (0-1 scale)
    max_iter : maximum iterations
    tol : convergence tolerance (max absolute marginal error)
    floor : minimum cell value to ensure strict positivity
    """
    initial = table.copy()
    t = table.copy()

    # Floor zero cells for convergence guarantee
    t = np.maximum(t, floor)
    t /= t.sum()

    history: list[dict] = []
    max_err = float("inf")

    for iteration in range(1, max_iter + 1):
        for c in CONDITIONS:
            axis = AXIS_MAP[c]
            target = target_marginals[c]

            current_1 = _marginal_sum(t, axis, 1)
            current_0 = _marginal_sum(t, axis, 0)

            factor_1 = target / current_1 if current_1 > 0 else 1.0
            factor_0 = (1.0 - target) / current_0 if current_0 > 0 else 1.0

            idx_1 = [slice(None)] * 4
            idx_1[axis] = 1
            t[tuple(idx_1)] *= factor_1

            idx_0 = [slice(None)] * 4
            idx_0[axis] = 0
            t[tuple(idx_0)] *= factor_0

        # Check convergence
        errors = {}
        for c in CONDITIONS:
            actual = _marginal_sum(t, AXIS_MAP[c], 1)
            errors[c] = abs(actual - target_marginals[c])

        max_err = max(errors.values())
        history.append({
            "iter": iteration,
            **{c: _marginal_sum(t, AXIS_MAP[c], 1) for c in CONDITIONS},
            "max_error": max_err,
        })

        if max_err < tol:
            break

    # Final normalisation
    t /= t.sum()

    return IPFResult(
        initial_table=initial,
        converged_table=t,
        converged=(max_err < tol),
        n_iterations=iteration,
        max_marginal_error=max_err,
        iteration_history=history,
        target_marginals=target_marginals,
    )


# ---------------------------------------------------------------------------
# Intersection extraction
# ---------------------------------------------------------------------------

def _intersection_from_table(table: np.ndarray, conditions: tuple[str, ...]) -> float:
    """Sum table cells where all specified conditions are present (=1).

    Returns value in percentage (0-100) scale.
    """
    idx = [slice(None)] * 4
    for c in conditions:
        idx[AXIS_MAP[c]] = 1
    return float(table[tuple(idx)].sum()) * 100


def extract_intersections_from_table(table: np.ndarray) -> SurveyIntersections:
    """Extract all 15 intersection prevalences from a 2x2x2x2 table."""
    singles = {c: _intersection_from_table(table, (c,)) for c in CONDITIONS}
    pairs = {p: _intersection_from_table(table, p) for p in PAIRS}
    triples = {tr: _intersection_from_table(table, tr) for tr in TRIPLES}
    quadruple = float(table[1, 1, 1, 1]) * 100
    return SurveyIntersections(
        singles=singles, pairs=pairs, triples=triples, quadruple=quadruple,
    )


# ---------------------------------------------------------------------------
# ReconciliationResult adapter
# ---------------------------------------------------------------------------

def reconcile_ipf(
    survey: SurveyIntersections,
    national: NationalMarginals,
    survey_direct_union: float,
    initial_table: np.ndarray,
    max_iter: int = IPF_MAX_ITERATIONS,
    tol: float = IPF_TOLERANCE,
) -> tuple[ReconciliationResult, IPFResult]:
    """Run IPF and return both ReconciliationResult and IPFResult."""
    target_marginals = {
        c: _marginal_value(national, c) / 100.0 for c in CONDITIONS
    }

    ipf_result = ipf_fit(initial_table, target_marginals, max_iter=max_iter, tol=tol)

    fitted = extract_intersections_from_table(ipf_result.converged_table)
    national_union = (1.0 - ipf_result.converged_table[0, 0, 0, 0]) * 100

    rec = ReconciliationResult(
        national_marginals=national,
        survey_intersections=survey,
        optimized_pairs={p: fitted.pairs[p] for p in PAIRS},
        optimized_triples={t: fitted.triples[t] for t in TRIPLES},
        optimized_quadruple=fitted.quadruple,
        national_union_ie=national_union,
        survey_direct_union=survey_direct_union,
        optimizer_success=ipf_result.converged,
        residual_norm=ipf_result.max_marginal_error,
        n_iterations=ipf_result.n_iterations,
    )
    return rec, ipf_result


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def reconcile_national_prevalence_ipf(
    dataframes: dict[str, pd.DataFrame],
    nhanes_df: pd.DataFrame | None = None,
    national: NationalMarginals | None = None,
) -> dict[str, tuple[ReconciliationResult, IPFResult]]:
    """Run IPF reconciliation for NHIS (pooled 2019-2024) and optionally NHANES.

    Returns dict with keys 'nhis' and optionally 'nhanes', each mapping to
    a (ReconciliationResult, IPFResult) tuple.
    """
    if national is None:
        national = NationalMarginals()

    results: dict[str, tuple[ReconciliationResult, IPFResult]] = {}

    # Pool NHIS 2019-2024
    nhis_years = [y for y in sorted(dataframes) if int(y) >= 2019]
    if nhis_years:
        pooled = pd.concat([dataframes[y] for y in nhis_years], ignore_index=True)
        masks = nhis_condition_masks(pooled)
        survey = compute_survey_intersections(pooled, masks)
        direct_union = _compute_direct_union(pooled, masks)
        table = compute_joint_table(pooled, masks)
        results["nhis"] = reconcile_ipf(survey, national, direct_union, table)

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
            table = compute_joint_table(nhanes_prepped, masks)
            results["nhanes"] = reconcile_ipf(survey, national, direct_union, table)

    return results


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------

def _subset_label(subset: tuple[str, ...]) -> str:
    return " & ".join(_CONDITION_NAMES[c] for c in subset)


def format_ipf_result(rec: ReconciliationResult, ipf_result: IPFResult | None = None) -> str:
    """Format IPF reconciliation result for console output."""
    lines: list[str] = []

    status = "converged" if rec.optimizer_success else "NOT converged"
    lines.append(f"  IPF: {status} "
                 f"(iterations={rec.n_iterations}, max_error={rec.residual_norm:.2e})")

    lines.append("\n  CDC marginals: "
                 f"B={rec.national_marginals.B}%, D={rec.national_marginals.D}%, "
                 f"P={rec.national_marginals.P}%, H={rec.national_marginals.H}%")

    lines.append(f"\n  Survey singles: " + ", ".join(
        f"{_CONDITION_NAMES[c]}={rec.survey_intersections.singles[c]:.2f}%"
        for c in CONDITIONS
    ))

    lines.append("\n  IPF-fitted pairwise intersections:")
    for subset, val in rec.optimized_pairs.items():
        survey_val = rec.survey_intersections.pairs[subset]
        lines.append(f"    {_subset_label(subset):35s}  ipf={val:6.2f}%  survey={survey_val:6.2f}%")

    lines.append("\n  IPF-fitted triple intersections:")
    for subset, val in rec.optimized_triples.items():
        survey_val = rec.survey_intersections.triples[subset]
        lines.append(f"    {_subset_label(subset):35s}  ipf={val:6.2f}%  survey={survey_val:6.2f}%")

    survey_quad = rec.survey_intersections.quadruple
    lines.append(f"\n  IPF-fitted quadruple intersection:")
    lines.append(f"    {'All four':35s}  ipf={rec.optimized_quadruple:6.2f}%  survey={survey_quad:6.2f}%")

    lines.append(f"\n  National union (1 - P(none)):     {rec.national_union_ie:.2f}%")
    lines.append(f"  Direct union (survey):            {rec.survey_direct_union:.2f}%")
    lines.append(f"  Difference:                       {rec.national_union_ie - rec.survey_direct_union:+.2f} pp")

    return "\n".join(lines)
