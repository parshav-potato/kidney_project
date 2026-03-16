"""IPF national prevalence reconciliation report.

Standalone module that loads NHIS/NHANES data, runs the Iterative Proportional
Fitting reconciliation against CDC national marginals, and prints a detailed
step-by-step breakdown:

    1. Inputs -- CDC marginals and survey marginals/intersections
    2. Initial joint table -- 16-cell distribution and marginal discrepancies
    3. IPF iteration log -- marginals converging per iteration
    4. Converged results -- fitted intersections vs survey originals
    5. Union and comparison -- P(B∪D∪P∪H) = 1 - P(none)

Run via:
    kidney-ipf-reconciliation
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
from kidney.analysis.ipf_reconciliation import (
    AXIS_MAP,
    IPFResult,
    compute_joint_table,
    ipf_fit,
    extract_intersections_from_table,
    reconcile_ipf,
    _marginal_sum,
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


def _format_table_2d(table: np.ndarray) -> None:
    """Print the 16-cell table in a readable B-row x (D,P,H)-column layout."""
    print("\n  Joint distribution (B rows x DPH columns):")
    print(f"  {'':6s}", end="")
    for d in range(2):
        for p in range(2):
            for h in range(2):
                label = f"D={d}P={p}H={h}"
                print(f"  {label:>10s}", end="")
    print(f"  {'Row total':>10s}")
    print(f"  {_THIN}")

    for b in range(2):
        print(f"  B={b}   ", end="")
        row_total = 0.0
        for d in range(2):
            for p in range(2):
                for h in range(2):
                    val = table[b, d, p, h]
                    row_total += val
                    print(f"  {val:10.6f}", end="")
        print(f"  {row_total:10.6f}")

    print(f"  {'Col tot':6s}", end="")
    for d in range(2):
        for p in range(2):
            for h in range(2):
                col_total = table[:, d, p, h].sum()
                print(f"  {col_total:10.6f}", end="")
    print(f"  {table.sum():10.6f}")


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
        s = survey.singles[c]
        m = _marginal_value(national, c)
        ratio_str = f"    (CDC/survey ratio = {m / s:.3f})" if s > 0 else ""
        print(f"    {_CONDITION_NAMES[c]:15s}  {s:6.2f}%{ratio_str}")

    print(f"\n  Survey pairwise intersections:")
    for p in PAIRS:
        print(f"    P({_subset_label(p):40s})  = {survey.pairs[p]:6.4f}%")

    print(f"\n  Survey triple intersections:")
    for t in TRIPLES:
        print(f"    P({_subset_label(t):40s})  = {survey.triples[t]:6.4f}%")

    print(f"\n  Survey quadruple intersection:")
    print(f"    P({_subset_label(QUADRUPLE):40s})  = {survey.quadruple:6.4f}%")


def _print_initial_table(
    table: np.ndarray,
    national: NationalMarginals,
) -> None:
    _print_header(2, "Initial Joint Distribution Table")

    _format_table_2d(table)

    print("\n  Survey marginals vs CDC targets:")
    print(f"    {'Condition':15s}  {'Survey':>10s}  {'CDC':>10s}  {'Difference':>12s}")
    print(f"    {'-' * 50}")
    for c in CONDITIONS:
        survey_m = _marginal_sum(table, AXIS_MAP[c], 1) * 100
        cdc_m = _marginal_value(national, c)
        diff = cdc_m - survey_m
        print(f"    {_CONDITION_NAMES[c]:15s}  {survey_m:9.2f}%  {cdc_m:9.1f}%  {diff:+11.2f} pp")

    n_zero = int((table == 0).sum())
    if n_zero > 0:
        print(f"\n  Note: {n_zero} cell(s) are zero and will be floored to {1e-12:.0e} for IPF.")


def _print_ipf_iterations(ipf_result: IPFResult) -> None:
    _print_header(3, "IPF Iteration Log")

    print(f"\n  Target marginals (proportions):")
    for c in CONDITIONS:
        print(f"    {_CONDITION_NAMES[c]:15s}  {ipf_result.target_marginals[c]:.4f}")

    history = ipf_result.iteration_history
    total = len(history)

    print(f"\n  {'Iter':>5s}  ", end="")
    for c in CONDITIONS:
        print(f"  {'P(' + c + '=1)':>10s}", end="")
    print(f"  {'Max Error':>12s}")
    print(f"  {'-' * 60}")

    # Show first 5, then every 10th, then last
    shown = set()
    for i in range(min(5, total)):
        shown.add(i)
    for i in range(9, total, 10):
        shown.add(i)
    shown.add(total - 1)

    for i in sorted(shown):
        row = history[i]
        print(f"  {row['iter']:5d}  ", end="")
        for c in CONDITIONS:
            print(f"  {row[c]:10.6f}", end="")
        print(f"  {row['max_error']:12.2e}")

    print(f"\n  Converged: {ipf_result.converged} after {ipf_result.n_iterations} iteration(s)")
    print(f"  Final max marginal error: {ipf_result.max_marginal_error:.2e}")


def _print_converged_results(
    ipf_result: IPFResult,
    survey: SurveyIntersections,
) -> None:
    _print_header(4, "Converged Table and Intersections")

    _format_table_2d(ipf_result.converged_table)

    fitted = extract_intersections_from_table(ipf_result.converged_table)

    print(f"\n  Fitted vs survey intersections:")
    print(f"    {'Intersection':45s}  {'IPF':>8s}  {'Survey':>8s}  {'Change':>10s}")
    print(f"    {'-' * 75}")

    for p in PAIRS:
        ipf_val = fitted.pairs[p]
        surv_val = survey.pairs[p]
        diff = ipf_val - surv_val
        print(f"    {_subset_label(p):45s}  {ipf_val:7.4f}%  {surv_val:7.4f}%  {diff:+9.4f} pp")

    for t in TRIPLES:
        ipf_val = fitted.triples[t]
        surv_val = survey.triples[t]
        diff = ipf_val - surv_val
        print(f"    {_subset_label(t):45s}  {ipf_val:7.4f}%  {surv_val:7.4f}%  {diff:+9.4f} pp")

    ipf_q = fitted.quadruple
    surv_q = survey.quadruple
    print(f"    {'All four':45s}  {ipf_q:7.4f}%  {surv_q:7.4f}%  {ipf_q - surv_q:+9.4f} pp")


def _print_union(
    rec: ReconciliationResult,
    ipf_result: IPFResult,
    direct_union: float,
) -> None:
    _print_header(5, "Union and Comparison")

    p_none = ipf_result.converged_table[0, 0, 0, 0]
    union = (1.0 - p_none) * 100

    print(f"\n  P(none) = table[0,0,0,0] = {p_none:.6f}")
    print(f"  P(BUDUPUH) = 1 - P(none) = {union:.2f}%")

    # Verify via inclusion-exclusion
    national = rec.national_marginals
    sum_m = sum(_marginal_value(national, c) for c in CONDITIONS)
    sum_pairs = sum(rec.optimized_pairs.values())
    sum_triples = sum(rec.optimized_triples.values())
    ie_union = sum_m - sum_pairs + sum_triples - rec.optimized_quadruple

    print(f"\n  Verification via inclusion-exclusion:")
    print(f"    Sum marginals  = {sum_m:.2f}")
    print(f"    - Sum pairs    = {sum_pairs:.4f}")
    print(f"    + Sum triples  = {sum_triples:.4f}")
    print(f"    - Quadruple    = {rec.optimized_quadruple:.4f}")
    print(f"    = {ie_union:.2f}%  (should match {union:.2f}%)")

    print(f"\n  {_THIN}")
    print(f"  National union (IPF):       {union:.2f}%")
    print(f"  Survey direct union:        {direct_union:.2f}%")
    print(f"  Difference:                 {union - direct_union:+.2f} pp")
    print(f"  {_THIN}")


# ---------------------------------------------------------------------------
# Single-source report
# ---------------------------------------------------------------------------

def run_ipf_report(
    df: pd.DataFrame,
    masks: dict[str, pd.Series],
    national: NationalMarginals,
    source_label: str,
) -> tuple[ReconciliationResult, IPFResult]:
    """Run and print full IPF reconciliation report for one data source."""

    print(f"\n{'#' * 72}")
    print(f"#  {source_label}")
    print(f"{'#' * 72}")

    survey = compute_survey_intersections(df, masks)
    direct_union = _compute_direct_union(df, masks)
    table = compute_joint_table(df, masks)

    # Step 1
    _print_inputs(survey, national, source_label)

    # Step 2
    _print_initial_table(table, national)

    # Step 3
    rec, ipf_result = reconcile_ipf(survey, national, direct_union, table)

    _print_ipf_iterations(ipf_result)

    # Step 4
    _print_converged_results(ipf_result, survey)

    # Step 5
    _print_union(rec, ipf_result, direct_union)

    return rec, ipf_result


# ---------------------------------------------------------------------------
# Result export
# ---------------------------------------------------------------------------

def _result_to_dict(
    rec: ReconciliationResult,
    ipf_result: IPFResult,
    source: str,
) -> dict:
    """Convert results to a JSON-serialisable dict."""
    national = rec.national_marginals
    return {
        "source": source,
        "method": "IPF",
        "cdc_marginals": {
            _CONDITION_NAMES[c]: _marginal_value(national, c) for c in CONDITIONS
        },
        "survey_singles": {
            _CONDITION_NAMES[c]: round(rec.survey_intersections.singles[c], 4)
            for c in CONDITIONS
        },
        "ipf_fitted_pairs": {
            _subset_label(k): round(v, 4)
            for k, v in rec.optimized_pairs.items()
        },
        "ipf_fitted_triples": {
            _subset_label(k): round(v, 4)
            for k, v in rec.optimized_triples.items()
        },
        "ipf_fitted_quadruple": round(rec.optimized_quadruple, 4),
        "national_union_pct": round(rec.national_union_ie, 2),
        "survey_union_pct": round(rec.survey_direct_union, 2),
        "difference_pp": round(rec.national_union_ie - rec.survey_direct_union, 2),
        "converged": ipf_result.converged,
        "iterations": ipf_result.n_iterations,
        "max_marginal_error": float(f"{ipf_result.max_marginal_error:.2e}"),
        "converged_table_flat": [round(float(v), 8) for v in ipf_result.converged_table.ravel()],
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_ipf_reconciliation_report(
    output_dir: str = "output",
) -> dict[str, ReconciliationResult]:
    """Load data, run IPF reconciliation for NHIS and NHANES, print full report."""

    print(_SEP)
    print("  National Prevalence Reconciliation Report (IPF Method)")
    print("  Iterative Proportional Fitting: survey 2^4 table -> CDC marginals")
    print(_SEP)

    national = NationalMarginals()
    results: dict[str, ReconciliationResult] = {}
    ipf_results: dict[str, IPFResult] = {}

    # --- Load and preprocess NHIS ---
    print("\nLoading NHIS data...")
    dataframes = preprocess(load_nhis())

    nhis_years = [y for y in sorted(dataframes) if int(y) >= 2019]
    if nhis_years:
        pooled = pd.concat([dataframes[y] for y in nhis_years], ignore_index=True)
        masks = nhis_condition_masks(pooled)
        label = f"NHIS {nhis_years[0]}-{nhis_years[-1]} pooled (n={len(pooled):,})"
        rec, ipf_res = run_ipf_report(pooled, masks, national, label)
        results["nhis"] = rec
        ipf_results["nhis"] = ipf_res

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
            rec, ipf_res = run_ipf_report(prepped, masks, national, label)
            results["nhanes"] = rec
            ipf_results["nhanes"] = ipf_res

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
    json_path = os.path.join(output_dir, "ipf_reconciliation_results.json")
    json_data = {
        source: _result_to_dict(results[source], ipf_results[source], source)
        for source in results
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\nResults saved to {json_path}")

    return results


def main() -> None:
    """Entry point for kidney-ipf-reconciliation console script."""
    generate_ipf_reconciliation_report()


if __name__ == "__main__":
    main()
