"""Full analysis pipeline -- orchestrates data loading, analysis, and visualisation."""

from __future__ import annotations

import os
import warnings
from typing import Any

import pandas as pd

from kidney.config import (
    ELIGIBILITY_BMI_THRESHOLD,
    OBESITY_THRESHOLD,
    RACE_ETHNICITY_COLORS,
    EDUCATION_COLORS,
    REGION_COLORS,
    NHANES_CONDITION_ORDER,
    US_ADULT_POPULATION_2025,
)
from kidney.data.loader import load_nhis, load_nhanes
from kidney.data.preprocessing import preprocess
from kidney.analysis.prevalence import weighted_prevalence
from kidney.analysis.eligibility import (
    any_condition_mask,
    eligible_mask,
    eligibility_metrics,
)
from kidney.analysis.stratified import (
    stratified_analysis,
    region_stratifier,
    age_stratifier,
    insurance_stratifier,
    poverty_stratifier,
    race_stratifier,
    education_stratifier,
)
from kidney.analysis.trends import fit_linear_trend, project_trend
from kidney.analysis.donors import (
    DONOR_CATEGORY_LABELS,
    calculate_donor_categories,
    calculate_donor_category_trends,
    calculate_ideal_donor_projections,
    calculate_venn_diagram_data,
    calculate_eligibility_venn_data,
    calculate_impact_of_bmi_relaxation,
    generate_summary_text,
    calculate_population_segments,
)
from kidney.analysis.nhanes import (
    format_condition_summary,
    summarize_nhanes_conditions,
    prepare_nhanes_for_comparison,
)
from kidney.analysis.national_reconciliation import (
    reconcile_national_prevalence,
    format_reconciliation_result,
)
from kidney.analysis.ipf_reconciliation import (
    reconcile_national_prevalence_ipf,
    format_ipf_result,
)
from kidney.visualization.trends import (
    plot_condition_prevalences,
    plot_any_condition_with_ci,
    plot_eligibility_comparison,
    plot_eligibility_by_region,
    plot_eligibility_with_projections,
    plot_donor_eligibility_trends,
    plot_stratified_trends,
)
from kidney.visualization.diagrams import (
    plot_venn_diagram,
    plot_marginal_donor_bar,
    plot_donor_venn_marginal,
    plot_population_diagram,
    plot_population_diagram_proportional,
)
from kidney.visualization.comparison import (
    plot_nhis_vs_nhanes_conditions,
    plot_nhis_vs_nhanes_donor_categories,
    plot_bmi_threshold_venn_comparison,
)

warnings.filterwarnings("ignore")


def _out(output_dir: str, name: str, save: bool) -> str | None:
    return os.path.join(output_dir, name) if save else None


def run_full_analysis(
    show_stats: bool = False,
    save_plots: bool = False,
    output_dir: str = "output",
) -> dict[str, pd.DataFrame]:
    """Run the complete analysis pipeline."""

    print("=" * 80)
    print("Kidney Donation Eligibility Analysis")
    print("=" * 80)

    if save_plots:
        os.makedirs(output_dir, exist_ok=True)

    # --- Load and preprocess ---
    print("\n1. Loading data...")
    dataframes = load_nhis()

    print("\n2. Preprocessing data...")
    dataframes = preprocess(dataframes)

    years = sorted(dataframes)
    year_to_idx = {y: i for i, y in enumerate(years)}
    nhanes_summary: Any = None
    nhanes_df: pd.DataFrame | None = None

    # --- Analysis 1: Condition prevalences ---
    print("\n3. Calculating condition prevalences...")
    cond_prevs: dict[str, list[float]] = {
        "Diabetes": [], "Prediabetes": [], "Hypertension": [],
        "Historic Hypertension": [], "Obesity": [], "Any Condition": [],
    }
    for y in years:
        df = dataframes[y]
        cond_prevs["Diabetes"].append(weighted_prevalence(df, df["diabetes"] == 1))
        cond_prevs["Prediabetes"].append(weighted_prevalence(df, df["prediabetes"] == 1))
        cond_prevs["Hypertension"].append(weighted_prevalence(df, df["hypertension"] == 1))
        cond_prevs["Historic Hypertension"].append(weighted_prevalence(df, df["historic_hypertension"] == 1))
        cond_prevs["Obesity"].append(weighted_prevalence(df, df["BMI"] > OBESITY_THRESHOLD))
        cond_prevs["Any Condition"].append(weighted_prevalence(df, any_condition_mask(df)))

    plot_condition_prevalences(years, cond_prevs,
                               save_path=_out(output_dir, "condition_prevalences.png", save_plots))

    # --- Analysis 2: Regional prevalence ---
    print("\n5. Analyzing regional trends...")
    region_data = stratified_analysis(dataframes, region_stratifier(), any_condition_mask)
    plot_stratified_trends(years, region_data, title="Health Conditions by Region (2015-2024)",
                           ylabel="Percentage with Any Condition (%)", colors=REGION_COLORS,
                           save_path=_out(output_dir, "regional_analysis.png", save_plots))

    # --- Analysis 3: Age group ---
    print("\n6. Analyzing age group trends...")
    age_data = stratified_analysis(dataframes, age_stratifier(), any_condition_mask)
    plot_stratified_trends(years, age_data, title="Health Conditions by Age Group (2015-2024)",
                           ylabel="Percentage with Any Condition (%)",
                           save_path=_out(output_dir, "age_group_analysis.png", save_plots))

    # --- Analysis 4: Insurance (2019+) ---
    print("\n7. Analyzing insurance status trends...")
    ins_data = stratified_analysis(
        dataframes, insurance_stratifier(),
        lambda df: any_condition_mask(df, include_historic_htn=False),
    )
    ins_years = [y for y in years if int(y) >= 2019]
    if ins_years:
        plot_stratified_trends(ins_years, ins_data, title="Health Conditions by Insurance Status (2019-2024)",
                               ylabel="Percentage with Any Condition (%)",
                               colors=["#66B2FF", "#FF9999"],
                               save_path=_out(output_dir, "insurance_analysis.png", save_plots))

    # --- Analysis 5: Any condition with CI ---
    print("\n8. Plotting 'Any Condition' with confidence intervals...")
    plot_any_condition_with_ci(years, cond_prevs["Any Condition"],
                               save_path=_out(output_dir, "any_condition_ci.png", save_plots))

    # --- Analysis 6: Comorbidity Venn ---
    print("\n9. Creating Venn diagram for co-morbidity (2023)...")
    if "2023" in dataframes:
        venn_data = calculate_venn_diagram_data(dataframes["2023"], OBESITY_THRESHOLD)
        plot_venn_diagram(venn_data, save_path=_out(output_dir, "venn_2023.png", save_plots))

    # --- Analysis 7: Eligibility comparison ---
    print("\n10. Calculating kidney donation eligibility...")
    elig_strict, abs_strict = eligibility_metrics(dataframes, 30.0)
    elig_relaxed, abs_relaxed = eligibility_metrics(dataframes, 35.0)

    model_s, yrs_s, _ = fit_linear_trend(years, elig_strict, 2019)
    model_r, _, _ = fit_linear_trend(years, elig_relaxed, 2019)
    base_year = int(yrs_s[0])

    plot_eligibility_comparison(years, elig_strict, elig_relaxed, model_s, model_r, base_year,
                                save_path=_out(output_dir, "eligibility_comparison.png", save_plots))

    # --- Analysis 8: Regional eligibility ---
    print("\n11. Analyzing eligibility by region...")
    reg_elig = stratified_analysis(dataframes, region_stratifier(), eligible_mask)
    plot_eligibility_by_region(years, reg_elig, elig_relaxed,
                               title="Eligible Population by Region (BMI < 35)",
                               save_path=_out(output_dir, "regional_eligibility.png", save_plots))

    # --- Analysis 9: Projections ---
    print("\n12. Creating projections (2019-2028)...")
    plot_eligibility_with_projections(years, elig_strict, abs_strict, model_s, base_year,
                                      save_path=_out(output_dir, "eligibility_projections.png", save_plots))

    # --- Analysis 10: BMI relaxation impact ---
    print("\n13. Calculating impact of BMI relaxation...")
    impact = calculate_impact_of_bmi_relaxation(dataframes)
    summary_text = generate_summary_text(impact)
    print("\n" + "=" * 80 + "\nSUMMARY\n" + "=" * 80)
    print(summary_text)
    print("=" * 80)

    if save_plots:
        with open(os.path.join(output_dir, "summary.txt"), "w") as f:
            f.write("Kidney Donation Eligibility Analysis - Summary\n")
            f.write("=" * 80 + "\n\n" + summary_text + "\n\n")
            f.write("Detailed Impact Metrics:\n" + "-" * 80 + "\n")
            for k, v in impact.items():
                f.write(f"{k}: {v}\n")

    # --- Analysis 11: Eligibility by poverty ---
    print("\n14. Analyzing eligible donors by poverty level...")
    pov_years = [y for y in years if int(y) >= 2019]
    if pov_years:
        pov_elig = stratified_analysis(dataframes, poverty_stratifier(), eligible_mask)
        plot_stratified_trends(pov_years, pov_elig,
                               title="Kidney Donation Eligibility by Poverty Level (2019+)",
                               ylabel="Eligibility Rate (%)",
                               colors=["#d73027", "#fc8d59", "#fee08b", "#91cf60"],
                               legend_title="Poverty Level (FPL)",
                               annotation="Eligibility: No prediabetes, No hypertension, BMI < 35",
                               save_path=_out(output_dir, "eligibility_by_poverty.png", save_plots))

    # --- Analysis 12: Eligibility by race/ethnicity ---
    print("\n15. Analyzing eligible donors by race/ethnicity...")
    race_years = [y for y in years if int(y) >= 2019]
    if race_years:
        race_elig = stratified_analysis(dataframes, race_stratifier(), eligible_mask)
        plot_stratified_trends(race_years, race_elig,
                               title="Kidney Donation Eligibility by Race/Ethnicity (2019-2024)",
                               ylabel="Eligibility Rate (%)",
                               colors=RACE_ETHNICITY_COLORS,
                               legend_title="Race/Ethnicity",
                               annotation="Eligibility: No prediabetes, No hypertension, BMI < 35",
                               save_path=_out(output_dir, "eligibility_by_race_ethnicity.png", save_plots))

    # --- Analysis 13: Eligibility by education ---
    print("\n16. Analyzing eligible donors by education level...")
    edu_years = [y for y in years if int(y) >= 2021]
    if edu_years:
        edu_elig = stratified_analysis(
            dataframes, education_stratifier(),
            lambda df: eligible_mask(df, exclude_prediabetes=False),
        )
        plot_stratified_trends(edu_years, edu_elig,
                               title="Kidney Donation Eligibility by Education Level (2021-2024)",
                               ylabel="Eligibility Rate (%)",
                               colors=EDUCATION_COLORS,
                               legend_title="Education Level",
                               annotation="Eligibility: No diabetes, No hypertension, BMI < 35",
                               save_path=_out(output_dir, "eligibility_by_education.png", save_plots))

    # --- Analysis 14: NHANES comparison ---
    print("\n17. Comparing NHIS vs NHANES condition prevalence...")
    nhis_vs_nhanes: dict = {}
    comparison_years = ("2021", "2022", "2023")
    try:
        nhanes_df = load_nhanes()
        nhanes_summary = summarize_nhanes_conditions(nhanes_df, cycle_label="2021-2022")
        print(format_condition_summary(nhanes_summary))

        for cond in nhanes_summary.stats:
            if cond not in cond_prevs:
                continue
            nhis_vals = [cond_prevs[cond][year_to_idx[y]] for y in comparison_years if y in year_to_idx]
            if not nhis_vals:
                continue
            nhis_avg = sum(nhis_vals) / len(nhis_vals)
            nhanes_val = nhanes_summary.stats[cond].weighted_pct
            nhis_vs_nhanes[cond] = {"nhanes": nhanes_val, "nhis": nhis_avg, "difference": nhanes_val - nhis_avg}
            print(f"   {cond}: NHANES {nhanes_val:.2f}% vs NHIS {nhis_avg:.2f}% (diff {nhanes_val - nhis_avg:+.2f} pp)")
    except FileNotFoundError as exc:
        print(f"   Skipping NHANES comparison (files missing): {exc}")
    except Exception as exc:
        print(f"   Error during NHANES comparison: {exc}")

    # --- Analysis 15: Donor categories ---
    print("\n18. Calculating donor categories...")
    latest_year = max(years)
    donor_cats = calculate_donor_categories(dataframes[latest_year])
    print(f"\n   Donor Categories ({latest_year}):")
    for label, info in donor_cats.items():
        pop_m = info["pct"] / 100 * US_ADULT_POPULATION_2025 / 1e6
        print(f"   {label}: {info['pct']:.1f}% (~{pop_m:.1f}M)")

    plot_marginal_donor_bar(donor_cats, latest_year,
                            save_path=_out(output_dir, "marginal_donors_bar.png", save_plots))
    plot_donor_venn_marginal(donor_cats, latest_year,
                             save_path=_out(output_dir, "donor_venn_marginal.png", save_plots))

    # --- Analysis 16: Ideal donor projections ---
    print("\n19. Projecting ideal donor eligibility (10-20 years)...")
    proj = calculate_ideal_donor_projections(dataframes)
    print(f"   Slope: {proj['slope_pp_per_yr']:+.3f} pp/year")
    for py, pv in zip(proj["proj_years"], proj["proj_values"]):
        print(f"   {py}: {pv:.1f}% (~{pv / 100 * US_ADULT_POPULATION_2025 / 1e6:.0f}M)")

    plot_donor_eligibility_trends(
        proj["years"], proj["ideal_pcts"], proj["expanded_pcts"],
        proj["model"], proj["base_year"], proj["proj_years"], proj["proj_values"],
        proj["slope_pp_per_yr"],
        save_path=_out(output_dir, "donor_eligibility_trends.png", save_plots))

    # --- Analysis 17: Population segments ---
    print("\n20. Creating population health segment diagrams...")
    pop_segs = calculate_population_segments(dataframes[latest_year])
    plot_population_diagram(pop_segs, latest_year,
                            save_path=_out(output_dir, "population_diagram_schematic.png", save_plots))
    plot_population_diagram_proportional(pop_segs, latest_year,
                                         save_path=_out(output_dir, "population_diagram_proportional.png", save_plots))

    # --- Analysis 18: BMI threshold Euler ---
    venn_30 = calculate_eligibility_venn_data(dataframes[latest_year], 30.0)
    venn_35 = calculate_eligibility_venn_data(dataframes[latest_year], 35.0)
    print(f"\n   BMI < 30 ideal donors: {venn_30['ideal_pct']:.1f}% of sub-pop")
    print(f"   BMI < 35 expanded donors: {venn_35['ideal_pct']:.1f}% of sub-pop")
    plot_bmi_threshold_venn_comparison(venn_30, venn_35, latest_year,
                                       save_path=_out(output_dir, "bmi_threshold_venn_comparison.png", save_plots))

    # --- Analysis 19: Donor category trends ---
    print("\n21. Computing donor category trends across all years...")
    cat_trends = calculate_donor_category_trends(dataframes)
    for yr, pct in zip(years, cat_trends["Ideal (No HTN, No PreDM, BMI<30)"]["pcts"]):
        print(f"   {yr}: {pct:.1f}%")

    # --- Analysis 20: NHIS vs NHANES visualisations ---
    if nhanes_summary is not None:
        try:
            print("\n22. NHIS vs NHANES comparison plots...")
            nhanes_comp_df = prepare_nhanes_for_comparison(nhanes_df)
            plot_nhis_vs_nhanes_conditions(
                cond_prevs, nhanes_summary, nhis_year_index=-1,
                nhis_label=f"NHIS {latest_year}", nhanes_label="NHANES 2021-22",
                save_path=_out(output_dir, "nhis_vs_nhanes_conditions.png", save_plots))

            nhanes_donor_cats = calculate_donor_categories(nhanes_comp_df)
            plot_nhis_vs_nhanes_donor_categories(
                donor_cats, nhanes_donor_cats,
                nhis_label=f"NHIS {latest_year}", nhanes_label="NHANES 2021-22",
                save_path=_out(output_dir, "nhis_vs_nhanes_donor_categories.png", save_plots))
        except Exception as exc:
            print(f"   Error during NHIS vs NHANES plots: {exc}")
    else:
        print("\n22-24. Skipping NHIS vs NHANES plots (NHANES data not loaded).")

    # --- Analysis 21: National prevalence reconciliation ---
    print("\n23. National prevalence reconciliation (CDC vs survey)...")
    try:
        recon_results = reconcile_national_prevalence(dataframes, nhanes_df)
        for source, rec in recon_results.items():
            print(f"\n--- {source.upper()} ---")
            print(format_reconciliation_result(rec))
    except Exception as exc:
        print(f"   Error during national reconciliation: {exc}")

    # --- Analysis 22: IPF national prevalence reconciliation ---
    print("\n24. National prevalence reconciliation (IPF method)...")
    try:
        ipf_results = reconcile_national_prevalence_ipf(dataframes, nhanes_df)
        for source, (rec, ipf_res) in ipf_results.items():
            print(f"\n--- {source.upper()} (IPF) ---")
            print(format_ipf_result(rec, ipf_res))
    except Exception as exc:
        print(f"   Error during IPF reconciliation: {exc}")

    # --- Export CSV ---
    csv_rows = []
    for i, yr in enumerate(years):
        row = {"year": yr}
        for cond, vals in cond_prevs.items():
            row[cond] = round(vals[i], 2)
        row["Eligible (BMI<30)"] = round(elig_strict[i], 2)
        row["Eligible (BMI<35)"] = round(elig_relaxed[i], 2)
        csv_rows.append(row)
    csv_path = os.path.join(output_dir, "comorbidity_stats.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"\nComorbidity statistics saved to: {csv_path}")
    print("\nAnalysis complete!")
    return dataframes


def main() -> None:
    """Entry point for kidney-analysis console script."""
    dataframes = run_full_analysis(save_plots=True, output_dir="output")
    print(f"\nProcessed {len(dataframes)} years of data")
    print(f"Total records: {sum(len(df) for df in dataframes.values())}")


if __name__ == "__main__":
    main()
