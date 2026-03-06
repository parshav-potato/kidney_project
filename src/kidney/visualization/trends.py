"""Generic stratified trend plots and specialised trend visualisations."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as t_dist
from sklearn.linear_model import LinearRegression

from kidney.config import PLOT_COLORS, REGION_COLORS, US_ADULT_POPULATION_2025
from kidney.visualization.style import save_and_show, hide_top_right


# ---------------------------------------------------------------------------
# Generic trend plot (replaces 8 near-identical functions)
# ---------------------------------------------------------------------------

def plot_stratified_trends(
    years: list[str],
    data: dict[str, list[float]],
    *,
    title: str = "",
    ylabel: str = "Percentage (%)",
    colors: dict[str, str] | list[str] | None = None,
    legend_title: str | None = None,
    annotation: str | None = None,
    figsize: tuple[int, int] = (12, 6),
    save_path: str | None = None,
) -> None:
    """One line per category over years. Replaces 8 near-identical functions."""
    fig, ax = plt.subplots(figsize=figsize)

    color_list: list[str] | None = None
    if isinstance(colors, dict):
        pass  # looked up per-label below
    elif isinstance(colors, list):
        color_list = colors
    else:
        color_list = PLOT_COLORS

    for i, (label, values) in enumerate(data.items()):
        if isinstance(colors, dict):
            c = colors.get(label, "#000000")
        elif color_list:
            c = color_list[i % len(color_list)]
        else:
            c = "#000000"
        ax.plot(years, values, marker="o", linewidth=2.5, label=label, color=c, markersize=8)

    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(title=legend_title, fontsize=11, loc="best")
    ax.tick_params(axis="both", which="major", labelsize=11)
    hide_top_right(ax)

    if annotation:
        ax.text(
            0.02, 0.98, annotation,
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )

    plt.xticks(rotation=45)
    plt.tight_layout()
    save_and_show(fig, save_path)


# ---------------------------------------------------------------------------
# Specialised trend plots (unique logic, not generalisable)
# ---------------------------------------------------------------------------

def plot_condition_prevalences(
    years: list[str],
    prevalences: dict[str, list[float]],
    title: str = "Prevalence of Health Conditions (2015-2024)",
    save_path: str | None = None,
) -> None:
    """Line chart of all 6 health conditions over time."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for (cond, pcts), color in zip(prevalences.items(), PLOT_COLORS):
        ax.plot(years, pcts, marker="o", linewidth=2, label=cond, color=color)
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Percentage of Population (%)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_and_show(fig, save_path)


def plot_any_condition_with_ci(
    years: list[str],
    percentages: list[float],
    fit_start_year: int = 2019,
    title: str = "Prevalence of Any Health Condition with 95% CI",
    save_path: str | None = None,
) -> None:
    """'Any Condition' with trend + 95% CI band + 5-year projection."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(years, percentages, marker="o", linewidth=2, label="Any Condition", color=PLOT_COLORS[5])

    fit_idx = [i for i, y in enumerate(years) if int(y) >= fit_start_year]
    if len(fit_idx) >= 2:
        fit_pcts = np.array([percentages[i] for i in fit_idx])
        yrs_fit = np.array([int(years[i]) for i in fit_idx])
        base = yrs_fit[0]
        X_fit = (yrs_fit - base).reshape(-1, 1)
        model = LinearRegression().fit(X_fit, fit_pcts)
        resid = fit_pcts - model.predict(X_fit)
        n = len(X_fit)
        if n > 2:
            mse = np.sum(resid**2) / (n - 2)
            x_mean = np.mean(X_fit)
            ss_dev = np.sum((X_fit - x_mean) ** 2)
            t_val = t_dist.ppf(0.975, df=n - 2)

            all_yrs = np.array([int(y) for y in years])
            plot_yrs = np.arange(all_yrs[0], all_yrs[-1] + 6)
            X_plot = (plot_yrs - base).reshape(-1, 1)
            trend = model.predict(X_plot)
            se = np.sqrt(mse * (1 / n + (X_plot - x_mean) ** 2 / ss_dev)).ravel()

            ax.plot([str(y) for y in plot_yrs], trend, ls="--", lw=2,
                    color=PLOT_COLORS[5], label="Trend (post-2019)")
            ax.fill_between([str(y) for y in plot_yrs], trend - t_val * se,
                            trend + t_val * se, color=PLOT_COLORS[5], alpha=0.2, label="95% CI")

    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_and_show(fig, save_path)


def plot_eligibility_comparison(
    years: list[str],
    eligible_strict: list[float],
    eligible_relaxed: list[float],
    model_strict: LinearRegression,
    model_relaxed: LinearRegression,
    base_year: int,
    title: str = "Temporal Trends: Eligibility for Living Kidney Donation",
    save_path: str | None = None,
) -> None:
    """BMI<30 vs BMI<35 with projection dashes."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(years, eligible_strict, marker="s", color="black", lw=1.5, label="Current BMI criteria (< 30)")
    ax.plot(years, eligible_relaxed, marker="o", color="blue", lw=1.5, label="Relaxed BMI criteria (< 35)")

    proj_yrs = np.arange(2019, 2029)
    X = (proj_yrs - base_year).reshape(-1, 1)
    ax.plot([str(y) for y in proj_yrs], model_strict.predict(X), ls="--", color="black", lw=1.5, label="Projection (BMI < 30)")
    ax.plot([str(y) for y in proj_yrs], model_relaxed.predict(X), ls="--", color="blue", lw=1.5, label="Projection (BMI < 35)")

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Percentage Eligible (%)", fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.7)
    hide_top_right(ax)
    plt.tight_layout()
    save_and_show(fig, save_path)


def plot_eligibility_by_region(
    years: list[str],
    eligible_by_region: dict[str, list[float]],
    eligible_national: list[float],
    title: str = "Eligible Population by Region",
    save_path: str | None = None,
) -> None:
    """Regional eligibility lines with national overlay."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(years, eligible_national, marker="o", lw=2, label="National", color="black")
    for region, pcts in eligible_by_region.items():
        ax.plot(years, pcts, marker="o", lw=2, label=region, color=REGION_COLORS.get(region, "#000"))
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Percentage Eligible (%)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend()
    hide_top_right(ax)
    plt.tight_layout()
    save_and_show(fig, save_path)


def plot_eligibility_with_projections(
    years: list[str],
    eligible_pct: list[float],
    absolute_eligible: list[float],
    model: LinearRegression,
    base_year: int,
    title: str = "Projected Eligible Population (2019-2028)",
    save_path: str | None = None,
) -> None:
    """Dual-axis: bar chart of absolute eligible + percentage line with projection."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(years, absolute_eligible, color="lightgray", label="Absolute Eligible")
    ax2 = ax.twinx()
    ax2.plot(years, eligible_pct, marker="o", lw=2, color="black", label="Actual %")

    proj_yrs = np.arange(2019, 2029)
    X = (proj_yrs - base_year).reshape(-1, 1)
    ax2.plot([str(y) for y in proj_yrs], model.predict(X), ls="--", lw=2, color="black", label="Trend")

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Absolute Number Eligible", fontsize=12)
    ax2.set_ylabel("Percentage Eligible (%)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.7)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper left")
    hide_top_right(ax)
    hide_top_right(ax2)
    plt.tight_layout()
    save_and_show(fig, save_path)


def plot_donor_eligibility_trends(
    years: list[str],
    ideal_pcts: list[float],
    expanded_pcts: list[float],
    model: LinearRegression,
    base_year: int,
    proj_years: list[int],
    proj_values: list[float],
    slope_pp: float,
    title: str = "Kidney Donor Eligibility: Trends & Projections",
    save_path: str | None = None,
) -> None:
    """Two-panel: historical ideal vs expanded (left), 10-20yr projection (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    year_ints = [int(y) for y in years]

    # Left panel
    ax = axes[0]
    ax.plot(year_ints, ideal_pcts, marker="o", lw=2, color="#2563eb", label="Ideal (BMI < 30)")
    ax.plot(year_ints, expanded_pcts, marker="s", lw=2, color="#16a34a", label="Expanded (BMI < 35)")
    ax.fill_between(year_ints, ideal_pcts, expanded_pcts, alpha=0.25, color="#16a34a", label="Additional Pool (BMI 30-35)")
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Eligible Population (%)", fontsize=12)
    ax.set_title("Historical Donor Eligibility", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(expanded_pcts) * 1.15)
    hide_top_right(ax)

    # Right panel
    ax2 = axes[1]
    recent_idx = [i for i, y in enumerate(years) if int(y) >= 2019]
    recent_years = [year_ints[i] for i in recent_idx]
    recent_vals = [ideal_pcts[i] for i in recent_idx]
    ax2.plot(recent_years, recent_vals, marker="o", lw=2, color="#2563eb", label="Historical (2019-2024)")

    latest_year = max(year_ints)
    latest_pct = ideal_pcts[-1]
    full_proj_x = [latest_year] + [y for y in proj_years if y > latest_year]
    full_proj_y = [latest_pct] + [proj_values[proj_years.index(y)] for y in full_proj_x[1:]]
    ax2.plot(full_proj_x, full_proj_y, ls="--", marker="x", ms=10, lw=2, color="#dc2626", label="Linear Projection")

    for px, py in zip(full_proj_x, full_proj_y):
        if px > latest_year:
            band = 1.5 * ((px - latest_year) / 10)
            ax2.fill_between([px - 0.3, px + 0.3], py - band, py + band, alpha=0.12, color="#dc2626")

    for px, py in zip(proj_years, proj_values):
        if px > latest_year:
            pop_m = py / 100 * US_ADULT_POPULATION_2025 / 1e6
            ax2.annotate(
                f"{py:.1f}% (~{pop_m:.0f}M)", xy=(px, py), xytext=(px, py + 3),
                fontsize=10, ha="center", color="#dc2626",
                arrowprops=dict(arrowstyle="->", color="#dc2626"),
            )

    ax2.set_xlabel("Year", fontsize=12)
    ax2.set_ylabel("Eligible Population (%)", fontsize=12)
    ax2.set_title(f"10-20 Year Projection (slope: {slope_pp:+.2f} pp/yr)", fontsize=13, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(2018, max(proj_years) + 2)
    hide_top_right(ax2)

    plt.tight_layout()
    save_and_show(fig, save_path)
