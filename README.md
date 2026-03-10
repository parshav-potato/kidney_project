# Kidney Donation Eligibility Analysis

Research project analyzing NHIS (National Health Interview Survey, 2015-2024) and NHANES (National Health and Nutrition Examination Survey, 2021-2022) data to assess trends in kidney donation eligibility among U.S. adults, following KDIGO guidelines.

## Setup

```bash
# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## Data

Place data files in `data/`:

```
data/
    NHIS_CSV/          # NHIS CSV files by year
        adult15csv/    # contains samadult.csv
        ...
        adult24csv/    # contains adult24.csv
    nhanes_data/       # NHANES 2021-2022 XPT files
        DEMO_L.xpt
        DIQ_L.xpt
        BPQ_L.xpt
        BPXO_L.xpt
        BMX_L.xpt
        GHB_L.xpt
        GLU_L.xpt
```

## Usage

```bash
# Run full analysis pipeline (generates 20+ PNG plots in output/)
kidney-analysis

# Generate manuscript statistics with 95% CI (saves manuscript_stats.json)
kidney-manuscript

# Run national prevalence reconciliation report (saves reconciliation_results.json)
kidney-reconciliation
```

## Architecture

```
src/kidney/
    config.py              # Constants, column mappings, thresholds, colors
    data/
        loader.py          # load_nhis(), load_nhanes()
        preprocessing.py   # 5-step pipeline: filter, fill, clean, BMI, dropna
    analysis/
        prevalence.py      # weighted_prevalence(), weighted_stats() with Kish CI
        eligibility.py     # eligible_mask(), any_condition_mask()
        stratified.py      # Generic stratified_analysis() + 6 stratifier factories
        trends.py          # fit_linear_trend(), project_trend(), trend_p_values()
        donors.py          # Donor categories, population segments, Venn data, projections
        nhanes.py          # NHANES condition flags, cross-validation
        manuscript.py      # Pooled-period stats, demographic comparisons
        national_reconciliation.py  # Constrained QP: CDC marginals + survey intersections
        reconciliation_report.py    # Standalone CLI report for the reconciliation
    visualization/
        style.py           # save_and_show(), hide_top_right()
        trends.py          # Generic plot_stratified_trends() + specialised plots
        diagrams.py        # Venn/Euler, population segments, donor bars
        comparison.py      # NHIS vs NHANES, BMI threshold Euler
    pipeline.py            # Orchestrator: runs all analyses, generates all plots
```

### Key Design Patterns

**Generic stratified analysis** (`analysis/stratified.py`): A single `stratified_analysis()` function replaces 8 near-identical analysis functions. Declarative `Stratifier` objects define how to slice data (by region, age, poverty, race, education, insurance).

**Unified eligibility masks** (`analysis/eligibility.py`): `eligible_mask()` and `any_condition_mask()` eliminate 6+ inline duplications of eligibility formulas. Parameters control variants (e.g., `exclude_prediabetes=False` for the manuscript's impact analysis).

**Generic trend plotting** (`visualization/trends.py`): `plot_stratified_trends()` replaces 8 near-identical plotting functions with one configurable function.

**National prevalence reconciliation** (`analysis/national_reconciliation.py`): Bridges CDC national marginals (obesity 40.3%, diabetes 12.0%, prediabetes 41.0%, hypertension 47.7%) with survey microdata. The survey gives the intersection structure (how conditions overlap); the CDC gives the true marginal prevalences. For each intersection, scaling estimates preserve the survey's conditional probabilities while adjusting to CDC marginals: `e_ic = survey_intersection * CDC(c) / survey(c)`. The unconstrained optimum is the mean of estimates per variable. A convex QP with monotonicity constraints is solved via SLSQP, and the 4 CDC marginals + 11 optimized intersections are plugged into inclusion-exclusion to yield national P(B U D U P U H). Run `kidney-reconciliation` for a step-by-step breakdown.

### NHANES Cross-Validation

NHANES 2021-2022 provides lab/exam-based condition flags as a cross-validation source against NHIS self-report. Key differences:

- **NHANES uses measurement thresholds**: HbA1c >= 6.5% or fasting glucose >= 126 (diabetes), HbA1c 5.7-6.4% or glucose 100-125 (prediabetes), SBP >= 130 or DBP >= 80 (hypertension, ACC/AHA 2017), BMI >= 30 (obesity).
- **NHIS uses self-report**: "Has a doctor ever told you..."
- NHANES marginals are much closer to CDC national estimates than NHIS (e.g., NHANES obesity 39.3% vs CDC 40.3%, while NHIS reports 31.1%).
- The reconciliation runs on both sources. NHANES has a much smaller gap (+3.9 pp) vs NHIS (+39.5 pp) between the reconciled national union and the survey's direct union.
- NHANES has zero overlap between diabetes and prediabetes by construction (mutually exclusive lab thresholds), so those intersection terms are 0.

### NHIS Survey Year Discontinuity

The NHIS redesigned its survey in 2019. Variable names changed:
- **2015-2018**: `samadult.csv`, variables like `AWEIGHTP`, `AGE_P`, `DIBPILL`
- **2019+**: `adult{YY}.csv`, variables like `WEIGHTLBTC_A`, `AGEP_A`, `DIBPILL_A`
- **2019+** added: insurance, poverty ratio, race/ethnicity
- **2021+** added: education level

All mappings are in `config.COLUMN_MAPPINGS`.

### Column Value Conventions

Health condition columns use NHIS coding: `1` = Yes, `2` = No, `NaN` = missing.

### Eligibility Criteria (KDIGO)

- **Ideal Donors**: No diabetes + No prediabetes + No hypertension + BMI < 30
- **Expanded Donors**: Same but BMI < 35
