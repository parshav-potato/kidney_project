"""NHIS vs NHANES comparison plots and BMI threshold Euler comparison."""

from __future__ import annotations

from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.optimize import brentq

from kidney.config import US_ADULT_POPULATION_2025
from kidney.visualization.style import save_and_show, hide_top_right


# ---------------------------------------------------------------------------
# NHIS vs NHANES conditions bar chart
# ---------------------------------------------------------------------------

def plot_nhis_vs_nhanes_conditions(
    nhis_prevalences: dict[str, list[float]],
    nhanes_summary: Any,
    nhis_year_index: int = -1,
    nhis_label: str = "NHIS 2024",
    nhanes_label: str = "NHANES 2021-22",
    title: str = "Condition Prevalence: NHIS (Self-Report) vs NHANES (Lab/Exam)",
    save_path: str | None = None,
) -> None:
    conditions = [
        "Diabetes", "Prediabetes", "Hypertension",
        "Historic Hypertension", "Obesity", "Any Condition",
    ]
    nhis_vals = [nhis_prevalences[c][nhis_year_index] if c in nhis_prevalences else 0 for c in conditions]
    nhanes_vals = []
    for c in conditions:
        s = nhanes_summary.stats.get(c)
        nhanes_vals.append(s.weighted_pct if s else 0)

    x = np.arange(len(conditions))
    w = 0.35

    fig, ax = plt.subplots(figsize=(14, 7))
    bars1 = ax.bar(x - w / 2, nhis_vals, w, label=nhis_label, color="#3B82F6", edgecolor="white", lw=0.8)
    bars2 = ax.bar(x + w / 2, nhanes_vals, w, label=nhanes_label, color="#EF4444", edgecolor="white", lw=0.8)

    for bar, val in zip(bars1, nhis_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#1e40af")
    for bar, val in zip(bars2, nhanes_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#991b1b")

    for i, (nv, av) in enumerate(zip(nhis_vals, nhanes_vals)):
        diff = av - nv
        if abs(diff) > 0.5:
            ax.annotate(f"{diff:+.1f} pp", xy=(i, max(nv, av) + 3.5),
                        ha="center", fontsize=8, color="#6b7280", fontstyle="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=10, rotation=20, ha="right")
    ax.set_ylabel("Prevalence (%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    hide_top_right(ax)

    ax.text(0.5, -0.18, "NHIS: self-reported diagnoses | NHANES: lab measurements + physical exam",
            transform=ax.transAxes, ha="center", fontsize=9, color="#6b7280")
    plt.tight_layout()
    save_and_show(fig, save_path)


# ---------------------------------------------------------------------------
# NHIS vs NHANES donor categories
# ---------------------------------------------------------------------------

def plot_nhis_vs_nhanes_donor_categories(
    nhis_cats: dict[str, dict[str, float]],
    nhanes_cats: dict[str, dict[str, float]],
    nhis_label: str = "NHIS 2024",
    nhanes_label: str = "NHANES 2021-22",
    title: str = "Donor Categories: NHIS (Self-Report) vs NHANES (Lab/Exam)",
    save_path: str | None = None,
) -> None:
    from kidney.analysis.donors import DONOR_CATEGORY_LABELS as labels

    nhis_pcts = [nhis_cats.get(l, {}).get("pct", 0) for l in labels]
    nhanes_pcts = [nhanes_cats.get(l, {}).get("pct", 0) for l in labels]

    y = np.arange(len(labels))
    h = 0.35

    fig, ax = plt.subplots(figsize=(13, 7))
    bars1 = ax.barh(y - h / 2, nhis_pcts, h, label=nhis_label, color="#3B82F6", edgecolor="white", lw=0.8)
    bars2 = ax.barh(y + h / 2, nhanes_pcts, h, label=nhanes_label, color="#EF4444", edgecolor="white", lw=0.8)

    for bar, val in zip(bars1, nhis_pcts):
        if val > 0.3:
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=9, fontweight="bold", color="#1e40af")
    for bar, val in zip(bars2, nhanes_pcts):
        if val > 0.3:
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=9, fontweight="bold", color="#991b1b")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("% of Adult Population", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    hide_top_right(ax)

    ideal_nhis, ideal_nhanes = nhis_pcts[0], nhanes_pcts[0]
    diff = ideal_nhis - ideal_nhanes
    if abs(diff) > 1:
        ax.text(0.98, 0.02,
                f"Ideal donor gap: {diff:+.1f} pp\n(NHIS {ideal_nhis:.1f}% vs NHANES {ideal_nhanes:.1f}%)",
                transform=ax.transAxes, fontsize=10, va="bottom", ha="right",
                fontweight="bold", color="#dc2626",
                bbox=dict(boxstyle="round", facecolor="#fee2e2", edgecolor="#fca5a5", alpha=0.92))
    plt.tight_layout()
    save_and_show(fig, save_path)


# ---------------------------------------------------------------------------
# BMI threshold Euler comparison
# ---------------------------------------------------------------------------

def _circle_intersection_area(r1: float, r2: float, d: float) -> float:
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return np.pi * min(r1, r2) ** 2
    a1 = r1**2 * np.arccos((d**2 + r1**2 - r2**2) / (2 * d * r1))
    a2 = r2**2 * np.arccos((d**2 + r2**2 - r1**2) / (2 * d * r2))
    a3 = 0.5 * np.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
    return a1 + a2 - a3


def _distance_for_overlap(r1: float, r2: float, target_area: float) -> float:
    if target_area <= 0:
        return r1 + r2 + 0.15
    max_area = np.pi * min(r1, r2) ** 2
    if target_area >= max_area * 0.999:
        return abs(r1 - r2)
    lo = abs(r1 - r2) + 1e-9
    hi = r1 + r2 - 1e-9
    return brentq(lambda d: _circle_intersection_area(r1, r2, d) - target_area, lo, hi)


def plot_bmi_threshold_venn_comparison(
    venn_30: dict[str, float],
    venn_35: dict[str, float],
    year: str,
    title: str | None = None,
    save_path: str | None = None,
) -> None:
    """Side-by-side area-proportional Euler diagrams for BMI<30 vs BMI<35."""
    if title is None:
        title = (f"Donor Eligibility: BMI < 30 vs BMI < 35 ({year})\n"
                 "(DM / Prediabetes / HTN among BMI-Eligible Adults)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    panels = [
        (axes[0], venn_30, f'BMI < 30  (Ideal Donors)\n({venn_30["universe_pct"]:.1f}% of population)'),
        (axes[1], venn_35, f'BMI < 35  (Expanded Donors)\n({venn_35["universe_pct"]:.1f}% of population)'),
    ]

    dm_fill, predm_fill, htn_fill = "#E8453C", "#F5A623", "#5B9BD5"

    for ax, vd, subtitle in panels:
        ax.set_aspect("equal"); ax.axis("off")
        dm_pct, predm_pct, htn_pct = vd["total_dm"], vd["total_predm"], vd["total_htn"]

        scale = 3.5
        r_dm = scale * np.sqrt(max(dm_pct, 0.3) / 100)
        r_predm = scale * np.sqrt(max(predm_pct, 0.3) / 100)
        r_htn = scale * np.sqrt(max(htn_pct, 0.3) / 100)

        area_unit = np.pi * scale**2 / 100
        d_dm_htn = _distance_for_overlap(r_dm, r_htn, (vd["dm_htn"] + vd["all_three"]) * area_unit)
        d_predm_htn = _distance_for_overlap(r_predm, r_htn, (vd["predm_htn"] + vd["all_three"]) * area_unit)
        d_dm_predm = _distance_for_overlap(r_dm, r_predm, (vd["dm_predm"] + vd["all_three"]) * area_unit)

        dm_cx, dm_cy = 0.0, 0.0
        predm_cx, predm_cy = d_dm_predm, 0.0

        if d_dm_predm > 1e-6:
            htn_x = (d_dm_htn**2 + d_dm_predm**2 - d_predm_htn**2) / (2 * d_dm_predm)
            htn_y = -np.sqrt(max(d_dm_htn**2 - htn_x**2, 0))
        else:
            htn_x, htn_y = 0.0, -d_dm_htn
        htn_cx, htn_cy = htn_x, htn_y

        # Re-centre
        mid_x = (min(dm_cx, predm_cx, htn_cx) + max(dm_cx, predm_cx, htn_cx)) / 2
        mid_y = (min(dm_cy, predm_cy, htn_cy) + max(dm_cy, predm_cy, htn_cy)) / 2
        off_x, off_y = 5.0 - mid_x, 5.0 - mid_y
        dm_cx += off_x; dm_cy += off_y
        predm_cx += off_x; predm_cy += off_y
        htn_cx += off_x; htn_cy += off_y

        for center, radius, color, z in [
            ((htn_cx, htn_cy), r_htn, htn_fill, 2),
            ((dm_cx, dm_cy), r_dm, dm_fill, 3),
            ((predm_cx, predm_cy), r_predm, predm_fill, 3),
        ]:
            ax.add_patch(Circle(center, radius, facecolor=color, edgecolor="black",
                                alpha=0.55, lw=2.5, zorder=z))

        ax.text(dm_cx, dm_cy + r_dm + 0.25, "Diabetes", fontsize=12, fontweight="bold",
                color="black", ha="center", va="bottom", zorder=10)
        ax.text(predm_cx, predm_cy + r_predm + 0.25, "Prediabetes", fontsize=12, fontweight="bold",
                color="black", ha="center", va="bottom", zorder=10)
        ax.text(htn_cx, htn_cy - r_htn - 0.25, "HTN", fontsize=12, fontweight="bold",
                color="black", ha="center", va="top", zorder=10)

        def _txt(x, y, val, fs=11):
            if val < 0.05:
                return
            ax.text(x, y, f"{val:.1f}%", ha="center", va="center",
                    fontsize=fs, fontweight="bold", color="#1f2937", zorder=10)

        _away_dm = np.array([dm_cx - htn_cx, dm_cy - htn_cy])
        _away_dm /= np.linalg.norm(_away_dm) + 1e-9
        _txt(dm_cx + _away_dm[0] * r_dm * 0.35, dm_cy + _away_dm[1] * r_dm * 0.35, vd["only_dm"])

        _away_predm = np.array([predm_cx - htn_cx, predm_cy - htn_cy])
        _away_predm /= np.linalg.norm(_away_predm) + 1e-9
        _txt(predm_cx + _away_predm[0] * r_predm * 0.35, predm_cy + _away_predm[1] * r_predm * 0.35, vd["only_predm"])

        _avg = np.array([(dm_cx + predm_cx) / 2 - htn_cx, (dm_cy + predm_cy) / 2 - htn_cy])
        _avg /= np.linalg.norm(_avg) + 1e-9
        _txt(htn_cx - _avg[0] * r_htn * 0.4, htn_cy - _avg[1] * r_htn * 0.4, vd["only_htn"])

        _txt((dm_cx + htn_cx) / 2, (dm_cy + htn_cy) / 2, vd["dm_htn"])
        _txt((predm_cx + htn_cx) / 2, (predm_cy + htn_cy) / 2, vd["predm_htn"])
        _txt((dm_cx + predm_cx) / 2, (dm_cy + predm_cy) / 2, vd["dm_predm"])
        _txt((dm_cx + predm_cx + htn_cx) / 3, (dm_cy + predm_cy + htn_cy) / 3, vd["all_three"])

        pad = 0.9
        pts_x, pts_y, pts_r = [dm_cx, predm_cx, htn_cx], [dm_cy, predm_cy, htn_cy], [r_dm, r_predm, r_htn]
        ax.set_xlim(min(x - r for x, r in zip(pts_x, pts_r)) - pad,
                    max(x + r for x, r in zip(pts_x, pts_r)) + pad)
        ax.set_ylim(min(y - r for y, r in zip(pts_y, pts_r)) - pad - 0.6,
                    max(y + r for y, r in zip(pts_y, pts_r)) + pad + 0.8)

        ideal_pct = vd["ideal_pct"]
        universe_pct = vd["universe_pct"]
        ideal_of_total = ideal_pct / 100 * universe_pct
        ideal_m = ideal_of_total / 100 * US_ADULT_POPULATION_2025 / 1e6
        ax.text(0.5, 0.01, f"Ideal Donors: {ideal_of_total:.1f}%  (~{ideal_m:.0f}M)",
                ha="center", fontsize=12, fontweight="bold", color="#16a34a", transform=ax.transAxes)
        ax.set_title(subtitle, fontsize=12, fontweight="bold", pad=10)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.subplots_adjust(wspace=0.08, top=0.86, bottom=0.06, left=0.02, right=0.98)
    save_and_show(fig, save_path)
