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
