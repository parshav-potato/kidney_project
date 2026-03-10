"""National prevalence reconciliation report.

Standalone module that loads NHIS/NHANES data, runs the constrained QP
reconciliation against CDC national marginals, and prints a detailed
step-by-step breakdown of the computation:

    1. Inputs -- CDC marginals and survey marginals/intersections
    2. Scaling estimates -- per-variable target values via each condition
    3. Unconstrained optima -- closed-form means
    4. Bounds and constraints -- box bounds, monotonicity, valid IE range
    5. Optimizer output -- which variables moved and why
    6. Inclusion-exclusion -- final national union

Run via:
    kidney-reconciliation
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from kidney.data.loader import load_nhis, load_nhanes
from kidney.data.preprocessing import preprocess
from kidney.analysis.national_reconciliation import (
    CONDITIONS,
    PAIRS,
    TRIPLES,
    QUADRUPLE,
    VAR_SUBSETS,
    NationalMarginals,
    SurveyIntersections,
    ReconciliationResult,
    nhis_condition_masks,
    nhanes_condition_masks,
    compute_survey_intersections,
    reconcile,
    _compute_direct_union,
    _marginal_value,
    _build_qp,
    _prepare_nhanes_for_reconciliation,
    _find_nhanes_weight,
)

_CONDITION_NAMES = {
    "B": "Obesity",
    "D": "Diabetes",
    "P": "Prediabetes",
    "H": "Hypertension",
}

_SEP = "=" * 72
_THIN = "-" * 72


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _subset_label(subset: tuple[str, ...]) -> str:
    return " & ".join(_CONDITION_NAMES[c] for c in subset)


def _print_header(step: int, title: str) -> None:
    print(f"\n{_SEP}")
    print(f"  Step {step}: {title}")
    print(_SEP)


# ---------------------------------------------------------------------------
# Step printers
# ---------------------------------------------------------------------------

def _print_inputs(
    survey: SurveyIntersections,
    national: NationalMarginals,
    source_label: str,
) -> None:
    _print_header(1, f"Inputs ({source_label})")

    print("\n  CDC national marginals (given constants):")
    for c in CONDITIONS:
        print(f"    {_CONDITION_NAMES[c]:15s}  {_marginal_value(national, c):6.1f}%")

    print(f"\n  Survey marginals ({source_label}):")
    for c in CONDITIONS:
        print(f"    {_CONDITION_NAMES[c]:15s}  {survey.singles[c]:6.2f}%"
              f"    (CDC/survey ratio = {_marginal_value(national, c) / survey.singles[c]:.3f})"
              if survey.singles[c] > 0 else
              f"    {_CONDITION_NAMES[c]:15s}  {survey.singles[c]:6.2f}%")

    print(f"\n  Survey pairwise intersections:")
    for p in PAIRS:
        print(f"    P({_subset_label(p):40s})  = {survey.pairs[p]:6.4f}%")

    print(f"\n  Survey triple intersections:")
    for t in TRIPLES:
        print(f"    P({_subset_label(t):40s})  = {survey.triples[t]:6.4f}%")

    print(f"\n  Survey quadruple intersection:")
    print(f"    P({_subset_label(QUADRUPLE):40s})  = {survey.quadruple:6.4f}%")


def _print_scaling_estimates(
    survey: SurveyIntersections,
    national: NationalMarginals,
) -> list[list[tuple[str, float]]]:
    _print_header(2, "Scaling Estimates")
    print("\n  For each intersection x_i, compute one estimate per condition c:")
    print("    e_ic = survey_intersection x CDC(c) / survey(c)")

    all_estimates: list[list[tuple[str, float]]] = []

    for i, subset in enumerate(VAR_SUBSETS):
        size = len(subset)
        if size == 2:
            s_S = survey.pairs[subset]
        elif size == 3:
            s_S = survey.triples[subset]
        else:
            s_S = survey.quadruple

        estimates: list[tuple[str, float]] = []
        for c in subset:
            s_c = survey.singles[c]
            m_c = _marginal_value(national, c)
            e = s_S * m_c / s_c if s_c > 0 else 0.0
            estimates.append((c, e))

        all_estimates.append(estimates)
        mean_e = np.mean([e for _, e in estimates])

        print(f"\n  x_{i:2d} = P({_subset_label(subset)})")
        print(f"       survey intersection = {s_S:.4f}%")
        for c, e in estimates:
            s_c = survey.singles[c]
            m_c = _marginal_value(national, c)
            print(f"       via {_CONDITION_NAMES[c]:15s}: {s_S:.4f} x {m_c:.1f} / {s_c:.4f} = {e:.4f}")
        print(f"       unconstrained optimum (mean) = {mean_e:.4f}")

    return all_estimates


def _print_bounds_and_initial(
    survey: SurveyIntersections,
    national: NationalMarginals,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    _print_header(3, "Bounds, Constraints & Initial Guess")

    _, _, x0, bounds, constraints = _build_qp(survey, national)

    print("\n  Box bounds (0 <= x_i <= min marginal in subset):")
    for i, subset in enumerate(VAR_SUBSETS):
        lo, hi = bounds[i]
        print(f"    x_{i:2d}  P({_subset_label(subset):40s})   [{lo:.1f}, {hi:.1f}]")

    print(f"\n  Monotonicity constraints ({len(constraints) - 2} inequalities):")
    print("    pair >= each child triple,  triple >= quadruple")

    print(f"\n  Inclusion-exclusion validity:")
    print("    0 <= Sum(marginals) - Sum(pairs) + Sum(triples) - quadruple <= 100")

    print(f"\n  Initial guess x0 (unconstrained mean, clipped to bounds):")
    for i, subset in enumerate(VAR_SUBSETS):
        lo, hi = bounds[i]
        clipped = " (clipped)" if x0[i] == lo or x0[i] == hi else ""
        print(f"    x_{i:2d} = {x0[i]:.4f}{clipped}")

    return x0, bounds


def _print_optimizer_result(
    survey: SurveyIntersections,
    national: NationalMarginals,
    x0: np.ndarray,
) -> ReconciliationResult:
    _print_header(4, "Optimizer (SLSQP)")

    objective, gradient, _, bounds, constraints = _build_qp(survey, national)

    from scipy.optimize import minimize as sp_minimize
    result = sp_minimize(
        objective, x0, jac=gradient, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-15},
    )

    x = result.x
    print(f"\n  Converged: {result.success}")
    print(f"  Iterations: {result.nit}")
    print(f"  Objective at x0:       {objective(x0):.4f}")
    print(f"  Objective at solution: {result.fun:.4f}")

    print(f"\n  Variable changes:")
    any_moved = False
    for i, subset in enumerate(VAR_SUBSETS):
        delta = x[i] - x0[i]
        marker = ""
        if abs(delta) > 1e-6:
            marker = "  <- adjusted"
            any_moved = True
        print(f"    x_{i:2d}  P({_subset_label(subset):40s})   "
              f"x0={x0[i]:8.4f}  ->  x*={x[i]:8.4f}  (d={delta:+.4f}){marker}")

    if not any_moved:
        print("\n  All variables stayed at unconstrained optimum -- no constraints active.")

    # Build ReconciliationResult for return
    var_index = {s: idx for idx, s in enumerate(VAR_SUBSETS)}
    opt_pairs = {p: float(x[var_index[p]]) for p in PAIRS}
    opt_triples = {t: float(x[var_index[t]]) for t in TRIPLES}
    opt_quad = float(x[var_index[QUADRUPLE]])

    sum_m = sum(_marginal_value(national, c) for c in CONDITIONS)
    ie_union = sum_m - sum(opt_pairs.values()) + sum(opt_triples.values()) - opt_quad

    return ReconciliationResult(
        national_marginals=national,
        survey_intersections=survey,
        optimized_pairs=opt_pairs,
        optimized_triples=opt_triples,
        optimized_quadruple=opt_quad,
        national_union_ie=ie_union,
        survey_direct_union=0.0,  # filled in later
        optimizer_success=result.success,
        residual_norm=float(np.sqrt(result.fun)) if result.fun >= 0 else 0.0,
        n_iterations=result.nit,
    )


def _print_inclusion_exclusion(
    rec: ReconciliationResult,
    direct_union: float,
) -> None:
    _print_header(5, "Inclusion-Exclusion")

    national = rec.national_marginals
    sum_m = sum(_marginal_value(national, c) for c in CONDITIONS)
    sum_pairs = sum(rec.optimized_pairs.values())
    sum_triples = sum(rec.optimized_triples.values())
    quad = rec.optimized_quadruple

    print(f"\n  P(BUDUPUH) = Sum marginals - Sum pairs + Sum triples - quadruple")
    print()
    print(f"    Sum marginals  = {' + '.join(f'{_marginal_value(national, c):.1f}' for c in CONDITIONS)}")
    print(f"                 = {sum_m:.2f}")
    print()
    print(f"    - Sum pairs    = {sum_pairs:.4f}")
    for p in PAIRS:
        print(f"        P({_subset_label(p):40s}) = {rec.optimized_pairs[p]:8.4f}")
    print()
    print(f"    + Sum triples  = {sum_triples:.4f}")
    for t in TRIPLES:
        print(f"        P({_subset_label(t):40s}) = {rec.optimized_triples[t]:8.4f}")
    print()
    print(f"    - quadruple  = {quad:.4f}")

    print(f"\n  {_THIN}")
    print(f"  National union (IE):    {rec.national_union_ie:.2f}%")
    print(f"  Survey direct union:    {direct_union:.2f}%")
    print(f"  Difference:             {rec.national_union_ie - direct_union:+.2f} pp")
    print(f"  {_THIN}")


# ---------------------------------------------------------------------------
# Single-source report
# ---------------------------------------------------------------------------

def run_reconciliation_report(
    df: pd.DataFrame,
    masks: dict[str, pd.Series],
    national: NationalMarginals,
    source_label: str,
) -> ReconciliationResult:
    """Run and print full reconciliation report for one data source."""

    print(f"\n{'#' * 72}")
    print(f"#  {source_label}")
    print(f"{'#' * 72}")

    survey = compute_survey_intersections(df, masks)
    direct_union = _compute_direct_union(df, masks)

    # Step 1
    _print_inputs(survey, national, source_label)

    # Step 2
    _print_scaling_estimates(survey, national)

    # Step 3
    x0, bounds = _print_bounds_and_initial(survey, national)

    # Step 4
    rec = _print_optimizer_result(survey, national, x0)
    rec.survey_direct_union = direct_union

    # Step 5
    _print_inclusion_exclusion(rec, direct_union)

    return rec


# ---------------------------------------------------------------------------
# Result export
# ---------------------------------------------------------------------------

def _result_to_dict(rec: ReconciliationResult, source: str) -> dict:
    """Convert a ReconciliationResult to a JSON-serialisable dict."""
    national = rec.national_marginals
    return {
        "source": source,
        "cdc_marginals": {
            _CONDITION_NAMES[c]: _marginal_value(national, c) for c in CONDITIONS
        },
        "survey_singles": {
            _CONDITION_NAMES[c]: round(rec.survey_intersections.singles[c], 4)
            for c in CONDITIONS
        },
        "optimized_pairs": {
            _subset_label(k): round(v, 4)
            for k, v in rec.optimized_pairs.items()
        },
        "optimized_triples": {
            _subset_label(k): round(v, 4)
            for k, v in rec.optimized_triples.items()
        },
        "optimized_quadruple": round(rec.optimized_quadruple, 4),
        "national_union_pct": round(rec.national_union_ie, 2),
        "survey_union_pct": round(rec.survey_direct_union, 2),
        "difference_pp": round(rec.national_union_ie - rec.survey_direct_union, 2),
        "optimizer_converged": rec.optimizer_success,
        "iterations": rec.n_iterations,
        "residual_norm": round(rec.residual_norm, 4),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_reconciliation_report(
    output_dir: str = "output",
) -> dict[str, ReconciliationResult]:
    """Load data, run reconciliation for NHIS and NHANES, print full report."""

    print(_SEP)
    print("  National Prevalence Reconciliation Report")
    print("  CDC marginals x survey intersection structure -> P(BUDUPUH)")
    print(_SEP)

    national = NationalMarginals()
    results: dict[str, ReconciliationResult] = {}

    # --- Load and preprocess NHIS ---
    print("\nLoading NHIS data...")
    dataframes = preprocess(load_nhis())

    nhis_years = [y for y in sorted(dataframes) if int(y) >= 2019]
    if nhis_years:
        pooled = pd.concat([dataframes[y] for y in nhis_years], ignore_index=True)
        masks = nhis_condition_masks(pooled)
        label = f"NHIS {nhis_years[0]}-{nhis_years[-1]} pooled (n={len(pooled):,})"
        results["nhis"] = run_reconciliation_report(pooled, masks, national, label)

    # --- Load NHANES ---
    nhanes_df = None
    try:
        print("\nLoading NHANES data...")
        nhanes_df = load_nhanes()
    except (FileNotFoundError, Exception) as exc:
        print(f"  Skipping NHANES ({exc})")

    if nhanes_df is not None:
        prepped = _prepare_nhanes_for_reconciliation(nhanes_df)
        if prepped is not None:
            weight_col = _find_nhanes_weight(prepped)
            if weight_col and weight_col != "survey_weight":
                prepped = prepped.rename(columns={weight_col: "survey_weight"})
            masks = nhanes_condition_masks(prepped)
            label = f"NHANES 2021-2022 (n={len(prepped):,})"
            results["nhanes"] = run_reconciliation_report(prepped, masks, national, label)

    # --- Summary comparison ---
    if len(results) > 1:
        print(f"\n{_SEP}")
        print("  Summary Comparison")
        print(_SEP)
        print(f"\n  {'Source':<35s}  {'National Union':>15s}  {'Survey Union':>15s}  {'Difference':>12s}")
        print(f"  {_THIN}")
        for source, rec in results.items():
            diff = rec.national_union_ie - rec.survey_direct_union
            print(f"  {source.upper():<35s}  {rec.national_union_ie:>14.2f}%  "
                  f"{rec.survey_direct_union:>14.2f}%  {diff:>+11.2f} pp")

    # --- Save JSON ---
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "reconciliation_results.json")
    json_data = {source: _result_to_dict(rec, source) for source, rec in results.items()}
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\nResults saved to {json_path}")

    return results


def main() -> None:
    """Entry point for kidney-reconciliation console script."""
    generate_reconciliation_report()


if __name__ == "__main__":
    main()
