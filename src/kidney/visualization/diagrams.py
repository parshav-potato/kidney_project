"""Complex diagram visualisations: Venn/Euler, population segments, donor bars."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, Circle
from typing import Any

from kidney.config import US_ADULT_POPULATION_2025
from kidney.visualization.style import save_and_show, hide_top_right


# ---------------------------------------------------------------------------
# Euler circle helpers
# ---------------------------------------------------------------------------

def _draw_euler_circle(ax, center, radius, fill_color, edge_color,
                       alpha=0.38, lw=2.5, zorder=2):
    shadow = Circle(center, radius, facecolor="#00000010", edgecolor="none", zorder=zorder - 0.5)
    shadow.set_path_effects([pe.withSimplePatchShadow(
        offset=(2, -2), shadow_rgbFace="#333333", alpha=0.18)])
    ax.add_patch(shadow)
    c = Circle(center, radius, facecolor=fill_color, edgecolor=edge_color,
               alpha=alpha, lw=lw, zorder=zorder)
    ax.add_patch(c)
    return c


def _place_label(ax, x, y, text, fontsize=11, bold=True, zorder=10):
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold" if bold else "normal", color="#1f2937", zorder=zorder,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#d1d5db", alpha=0.92, lw=0.8))


# ---------------------------------------------------------------------------
# Comorbidity Venn
# ---------------------------------------------------------------------------

def plot_venn_diagram(
    venn_data: dict[str, float],
    title: str = "Co-morbidity Venn Diagram (2023)",
    save_path: str | None = None,
) -> None:
    """Euler-style diagram: Prediabetes/Diabetes, Hypertension, Obesity."""
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal"); ax.axis("off")

    A_t = (venn_data["only_prediabetes"] + venn_data["prediabetes_and_hypertension"]
           + venn_data["prediabetes_and_obesity"] + venn_data["all_three"])
    B_t = (venn_data["only_hypertension"] + venn_data["prediabetes_and_hypertension"]
           + venn_data["hypertension_and_obesity"] + venn_data["all_three"])
    C_t = (venn_data["only_obesity"] + venn_data["prediabetes_and_obesity"]
           + venn_data["hypertension_and_obesity"] + venn_data["all_three"])

    scale = 3.0
    r_a, r_b, r_c = [scale * np.sqrt(t / 100) for t in (A_t, B_t, C_t)]

    cx, cy = 5, 5
    sep = max(r_a, r_b, r_c) * 0.72
    centers = {
        "A": (cx - sep * 0.55, cy + sep * 0.35),
        "B": (cx + sep * 0.55, cy + sep * 0.35),
        "C": (cx, cy - sep * 0.50),
    }

    _draw_euler_circle(ax, centers["C"], r_c, "#F59E0B", "#D97706", alpha=0.35, zorder=2)
    _draw_euler_circle(ax, centers["A"], r_a, "#E8453C", "#B5332B", alpha=0.38, zorder=3)
    _draw_euler_circle(ax, centers["B"], r_b, "#3B82F6", "#2563EB", alpha=0.38, zorder=4)

    ax.text(centers["A"][0] - r_a * 0.65, centers["A"][1] + r_a * 0.85,
            "Prediabetes /\nDiabetes", fontsize=13, fontweight="bold",
            color="#8B1A1A", ha="center", va="bottom", zorder=10)
    ax.text(centers["B"][0] + r_b * 0.65, centers["B"][1] + r_b * 0.85,
            "Hypertension", fontsize=13, fontweight="bold",
            color="#1E3A5F", ha="center", va="bottom", zorder=10)
    ax.text(centers["C"][0], centers["C"][1] - r_c * 1.05,
            "Obesity", fontsize=13, fontweight="bold",
            color="#92400E", ha="center", va="top", zorder=10)

    ac, bc, cc = centers["A"], centers["B"], centers["C"]
    _place_label(ax, ac[0] - r_a * 0.35, ac[1] + r_a * 0.15, f"{venn_data['only_prediabetes']:.1f}%")
    _place_label(ax, bc[0] + r_b * 0.35, bc[1] + r_b * 0.15, f"{venn_data['only_hypertension']:.1f}%")
    _place_label(ax, cc[0], cc[1] - r_c * 0.45, f"{venn_data['only_obesity']:.1f}%")
    _place_label(ax, (ac[0] + bc[0]) / 2, (ac[1] + bc[1]) / 2 + sep * 0.18, f"{venn_data['prediabetes_and_hypertension']:.1f}%")
    _place_label(ax, (ac[0] + cc[0]) / 2 - sep * 0.12, (ac[1] + cc[1]) / 2 - sep * 0.05, f"{venn_data['prediabetes_and_obesity']:.1f}%")
    _place_label(ax, (bc[0] + cc[0]) / 2 + sep * 0.12, (bc[1] + cc[1]) / 2 - sep * 0.05, f"{venn_data['hypertension_and_obesity']:.1f}%")
    tri_cx = (ac[0] + bc[0] + cc[0]) / 3
    tri_cy = (ac[1] + bc[1] + cc[1]) / 3
    _place_label(ax, tri_cx, tri_cy, f"{venn_data['all_three']:.1f}%", fontsize=12)

    pad = 1.5
    all_x = [c[0] for c in centers.values()]
    all_y = [c[1] for c in centers.values()]
    all_r = [r_a, r_b, r_c]
    ax.set_xlim(min(x - r for x, r in zip(all_x, all_r)) - pad,
                max(x + r for x, r in zip(all_x, all_r)) + pad)
    ax.set_ylim(min(y - r for y, r in zip(all_y, all_r)) - pad - 0.5,
                max(y + r for y, r in zip(all_y, all_r)) + pad + 1.0)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=18)
    plt.tight_layout()
    save_and_show(fig, save_path)


# ---------------------------------------------------------------------------
# Donor bar chart
# ---------------------------------------------------------------------------

def plot_marginal_donor_bar(
    categories: dict[str, dict[str, float]],
    year: str,
    title: str | None = None,
    save_path: str | None = None,
) -> None:
    """Horizontal bar chart of 8 donor categories."""
    from kidney.analysis.donors import DONOR_CATEGORY_LABELS

    if title is None:
        title = f"Potential Kidney Donor Pool by Category ({year})"

    labels = DONOR_CATEGORY_LABELS
    est_m = [categories.get(l, {"pct": 0})["pct"] / 100 * US_ADULT_POPULATION_2025 / 1e6 for l in labels]
    colors = ["#16a34a", "#f59e0b", "#ef4444", "#b91c1c", "#3b82f6", "#1d4ed8", "#8b5cf6", "#6b7280"]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(labels, est_m, color=colors, edgecolor="black", linewidth=0.5)

    for bar, val, lbl in zip(bars, est_m, labels):
        entry = categories.get(lbl, {"pct": 0})
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}M ({entry["pct"]:.1f}%)', va="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("Estimated US Adults (Millions)", fontsize=12)
    ax.set_title(title + "\n(All categories exclude Diabetes)", fontsize=14, fontweight="bold")
    ax.set_xlim(0, max(est_m) * 1.25)
    ax.grid(axis="x", alpha=0.3)
    hide_top_right(ax)

    total_pot = sum(est_m[:-1])
    ax.text(0.98, 0.04,
            f"Ideal: {est_m[0]:.1f}M\nExpanded (BMI<35): {est_m[0] + est_m[1]:.1f}M\n"
            f"Total Potential (7 categories): {total_pot:.1f}M",
            transform=ax.transAxes, fontsize=10, va="bottom", ha="right",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

    plt.tight_layout()
    save_and_show(fig, save_path)


# ---------------------------------------------------------------------------
# Donor Venn marginal
# ---------------------------------------------------------------------------

def plot_donor_venn_marginal(
    categories: dict[str, dict[str, float]],
    year: str,
    title: str | None = None,
    save_path: str | None = None,
) -> None:
    """Euler diagram of HTN x Prediabetes x Obesity(30-35) among non-diabetic adults."""
    if title is None:
        title = f"Marginal Donor Conditions ({year})\n(Non-Diabetic Adults Only)"

    only_htn = categories.get("HTN Only", {}).get("pct", 0)
    only_predm = categories.get("Prediabetes Only", {}).get("pct", 0)
    only_ob = categories.get("BMI 30-34.9 Only", {}).get("pct", 0)
    htn_predm = categories.get("HTN + Prediabetes", {}).get("pct", 0)
    htn_ob = categories.get("HTN + BMI 30-34.9", {}).get("pct", 0)
    predm_ob = categories.get("Prediabetes + BMI 30-34.9", {}).get("pct", 0)
    all_three = categories.get("HTN + PreDM + BMI 30-34.9 (Excluded)", {}).get("pct", 0)

    A_t = only_htn + htn_predm + htn_ob + all_three
    B_t = only_predm + htn_predm + predm_ob + all_three
    C_t = only_ob + htn_ob + predm_ob + all_three

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal"); ax.axis("off")

    scale = 3.2
    r_a = scale * np.sqrt(max(A_t, 0.5) / 100)
    r_b = scale * np.sqrt(max(B_t, 0.5) / 100)
    r_c = scale * np.sqrt(max(C_t, 0.5) / 100)

    cx, cy = 5, 5.2
    sep = max(r_a, r_b, r_c) * 0.72
    centers = {"A": (cx - sep * 0.55, cy + sep * 0.35),
               "B": (cx + sep * 0.55, cy + sep * 0.35),
               "C": (cx, cy - sep * 0.50)}

    _draw_euler_circle(ax, centers["C"], r_c, "#F59E0B", "#D97706", alpha=0.35, zorder=2)
    _draw_euler_circle(ax, centers["A"], r_a, "#3B82F6", "#2563EB", alpha=0.38, zorder=3)
    _draw_euler_circle(ax, centers["B"], r_b, "#E8453C", "#B5332B", alpha=0.38, zorder=4)

    ax.text(centers["A"][0] - r_a * 0.65, centers["A"][1] + r_a * 0.85,
            "HTN", fontsize=14, fontweight="bold", color="#1E3A5F", ha="center", va="bottom", zorder=10)
    ax.text(centers["B"][0] + r_b * 0.65, centers["B"][1] + r_b * 0.85,
            "Prediabetes", fontsize=14, fontweight="bold", color="#8B1A1A", ha="center", va="bottom", zorder=10)
    ax.text(centers["C"][0], centers["C"][1] - r_c * 1.05,
            "BMI 30-34.9", fontsize=14, fontweight="bold", color="#92400E", ha="center", va="top", zorder=10)

    ac, bc, cc = centers["A"], centers["B"], centers["C"]
    _place_label(ax, ac[0] - r_a * 0.35, ac[1] + r_a * 0.15, f"{only_htn:.1f}%")
    _place_label(ax, bc[0] + r_b * 0.35, bc[1] + r_b * 0.15, f"{only_predm:.1f}%")
    _place_label(ax, cc[0], cc[1] - r_c * 0.45, f"{only_ob:.1f}%")
    _place_label(ax, (ac[0] + bc[0]) / 2, (ac[1] + bc[1]) / 2 + sep * 0.18, f"{htn_predm:.1f}%")
    _place_label(ax, (ac[0] + cc[0]) / 2 - sep * 0.12, (ac[1] + cc[1]) / 2 - sep * 0.05, f"{htn_ob:.1f}%")
    _place_label(ax, (bc[0] + cc[0]) / 2 + sep * 0.12, (bc[1] + cc[1]) / 2 - sep * 0.05, f"{predm_ob:.1f}%")
    tri_cx = (ac[0] + bc[0] + cc[0]) / 3
    tri_cy = (ac[1] + bc[1] + cc[1]) / 3
    _place_label(ax, tri_cx, tri_cy, f"{all_three:.1f}%", fontsize=12)

    pad = 1.5
    all_x = [c[0] for c in centers.values()]
    all_y = [c[1] for c in centers.values()]
    all_r = [r_a, r_b, r_c]
    ax.set_xlim(min(x - r for x, r in zip(all_x, all_r)) - pad,
                max(x + r for x, r in zip(all_x, all_r)) + pad)
    ax.set_ylim(min(y - r for y, r in zip(all_y, all_r)) - pad - 1.5,
                max(y + r for y, r in zip(all_y, all_r)) + pad + 1.0)

    ideal_pct = categories.get("Ideal (No HTN, No PreDM, BMI<30)", {}).get("pct", 0)
    ideal_m = ideal_pct / 100 * US_ADULT_POPULATION_2025 / 1e6
    ax.text(0.5, 0.02,
            f"IDEAL DONORS (No HTN, No PreDM, BMI<30): {ideal_pct:.1f}% (~{ideal_m:.0f}M)",
            ha="center", fontsize=12, fontweight="bold", color="#16a34a",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#dcfce7", edgecolor="#86efac", alpha=0.92, lw=1.5))
    ax.text(0.5, -0.03, "Reference population: All adults (% of total)",
            ha="center", fontsize=9, color="#6b7280", transform=ax.transAxes)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=18)
    plt.tight_layout()
    save_and_show(fig, save_path)


# ---------------------------------------------------------------------------
# Population segment diagrams (schematic + proportional)
# ---------------------------------------------------------------------------

def plot_population_diagram(
    segments: dict[str, Any],
    year: str,
    title: str | None = None,
    save_path: str | None = None,
) -> None:
    """Two-panel: Euler circles (left) + stacked BMI bars (right)."""
    if title is None:
        title = f"Population Health Segments ({year})"

    dm_pct = segments.get("dm_total_pct", 0)
    predm_pct = segments.get("predm_total_pct", 0)
    htn_pct = segments.get("htn_total_pct", 0)
    bmi_lt30_pct = segments.get("bmi_lt30_total_pct", 0)
    bmi_30_35_pct = segments.get("bmi_30_35_total_pct", 0)
    bmi_gte35_pct = segments.get("bmi_gte35_total_pct", 0)

    bmi_keys = ["BMI<30", "BMI30-35", "BMI>=35"]

    def _region_pct(glyc, htn_key):
        return sum(segments.get(f"{glyc}_{htn_key}_{b}", {}).get("pct", 0) for b in bmi_keys)

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.97)
    ax_venn = fig.add_axes([0.02, 0.08, 0.48, 0.82])
    ax_bars = fig.add_axes([0.56, 0.12, 0.40, 0.72])

    # Left: Euler
    ax_venn.set_aspect("equal"); ax_venn.axis("off")
    scale = 3.5
    r_dm = scale * np.sqrt(dm_pct / 100)
    r_predm = scale * np.sqrt(predm_pct / 100)
    r_htn = scale * np.sqrt(htn_pct / 100)

    cx, cy = 5.0, 5.0
    sep = max(r_dm, r_predm, r_htn) * 0.72
    dm_center = [cx - sep * 0.55, cy + sep * 0.35]
    predm_center = [cx + sep * 0.55, cy + sep * 0.35]
    htn_center = [cx, cy - sep * 0.50]

    while np.hypot(predm_center[0] - dm_center[0], predm_center[1] - dm_center[1]) < r_dm + r_predm + 0.15:
        dm_center[0] -= 0.12
        predm_center[0] += 0.12

    _draw_euler_circle(ax_venn, tuple(htn_center), r_htn, "#3B82F6", "#2563EB", alpha=0.25, lw=2.5, zorder=2)
    _draw_euler_circle(ax_venn, tuple(dm_center), r_dm, "#E8453C", "#B5332B", alpha=0.35, lw=2.5, zorder=3)
    _draw_euler_circle(ax_venn, tuple(predm_center), r_predm, "#F59E0B", "#D97706", alpha=0.35, lw=2.5, zorder=3)

    ax_venn.text(dm_center[0], dm_center[1] + r_dm + 0.35,
                 f"Diabetes\n{dm_pct:.1f}%", fontsize=13, fontweight="bold",
                 color="#8B1A1A", ha="center", va="bottom", zorder=10)
    ax_venn.text(predm_center[0], predm_center[1] + r_predm + 0.35,
                 f"Prediabetes\n{predm_pct:.1f}%", fontsize=13, fontweight="bold",
                 color="#92400E", ha="center", va="bottom", zorder=10)
    ax_venn.text(htn_center[0] - r_htn - 0.3, htn_center[1],
                 f"HTN\n{htn_pct:.1f}%", fontsize=13, fontweight="bold",
                 color="#1E3A5F", ha="right", va="center", zorder=10)

    venn_regions = [
        ("DM only", "DM", "NoHTN", dm_center[0], dm_center[1] + r_dm * 0.25),
        ("DM + HTN", "DM", "HTN", (dm_center[0] + htn_center[0]) / 2 - 0.15, (dm_center[1] + htn_center[1]) / 2),
        ("PreDM only", "PreDM", "NoHTN", predm_center[0], predm_center[1] + r_predm * 0.25),
        ("PreDM + HTN", "PreDM", "HTN", (predm_center[0] + htn_center[0]) / 2 + 0.15, (predm_center[1] + htn_center[1]) / 2),
        ("HTN only", "Neither", "HTN", htn_center[0], htn_center[1] - r_htn * 0.30),
    ]
    for label, glyc, htn_key, rx, ry in venn_regions:
        total = _region_pct(glyc, htn_key)
        if total < 0.05:
            continue
        _place_label(ax_venn, rx, ry, f"{label}\n{total:.1f}%", fontsize=9, bold=True)

    no_cond = _region_pct("Neither", "NoHTN")
    nc_x = max(dm_center[0] + r_dm, predm_center[0] + r_predm) + 1.0
    ax_venn.text(nc_x, cy + 0.5, f"No conditions\n{no_cond:.1f}%",
                 fontsize=12, fontweight="bold", color="#166534", ha="left", va="center", zorder=10,
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#dcfce7", edgecolor="#86efac", alpha=0.92, lw=1.2))

    pad = 1.5
    ax_venn.set_xlim(min(dm_center[0] - r_dm, htn_center[0] - r_htn) - pad, nc_x + 4.5)
    ax_venn.set_ylim(htn_center[1] - r_htn - pad, max(dm_center[1], predm_center[1]) + max(r_dm, r_predm) + 1.5)
    ax_venn.set_title("Condition Overlaps (area-proportional)", fontsize=12, fontweight="bold", pad=10)

    cond_patches = [
        mpatches.Patch(facecolor="#E8453C", edgecolor="#B5332B", alpha=0.5, label=f"Diabetes ({dm_pct:.1f}%)"),
        mpatches.Patch(facecolor="#F59E0B", edgecolor="#D97706", alpha=0.5, label=f"Prediabetes ({predm_pct:.1f}%)"),
        mpatches.Patch(facecolor="#3B82F6", edgecolor="#2563EB", alpha=0.4, label=f"Hypertension ({htn_pct:.1f}%)"),
    ]
    ax_venn.legend(handles=cond_patches, loc="lower left", fontsize=9, framealpha=0.92, edgecolor="#d1d5db", fancybox=True)

    # Right: Stacked bars
    bar_regions = [
        ("No conditions", "Neither", "NoHTN"),
        ("HTN only", "Neither", "HTN"),
        ("PreDM only", "PreDM", "NoHTN"),
        ("DM + HTN", "DM", "HTN"),
        ("PreDM + HTN", "PreDM", "HTN"),
        ("DM only", "DM", "NoHTN"),
    ]
    bmi_colors = ["#4ade80", "#fbbf24", "#ef4444"]

    for i, (label, glyc, htn_key) in enumerate(bar_regions):
        vals = [segments.get(f"{glyc}_{htn_key}_{bk}", {}).get("pct", 0) for bk in bmi_keys]
        left = 0.0
        for v, color in zip(vals, bmi_colors):
            ax_bars.barh(i, v, left=left, height=0.6, color=color, edgecolor="white", lw=0.5)
            if v >= 1.5:
                ax_bars.text(left + v / 2, i, f"{v:.1f}%", ha="center", va="center", fontsize=8, fontweight="bold", color="#1f2937")
            left += v
        ax_bars.text(sum(vals) + 0.4, i, f"{sum(vals):.1f}%", ha="left", va="center", fontsize=9, fontweight="bold", color="#374151")

    ax_bars.set_yticks(range(len(bar_regions)))
    ax_bars.set_yticklabels([r[0] for r in bar_regions], fontsize=10, fontweight="bold")
    ax_bars.set_xlabel("% of Total U.S. Adult Population", fontsize=11, fontweight="bold")
    ax_bars.set_title("BMI Distribution by Condition Group", fontsize=12, fontweight="bold", pad=10)
    ax_bars.invert_yaxis()
    hide_top_right(ax_bars)
    ax_bars.set_xlim(0, max(_region_pct(g, h) for _, g, h in bar_regions) * 1.15)

    bmi_patches = [
        mpatches.Patch(facecolor="#4ade80", edgecolor="gray", lw=0.5, label=f"BMI < 30 ({bmi_lt30_pct:.1f}%)"),
        mpatches.Patch(facecolor="#fbbf24", edgecolor="gray", lw=0.5, label=f"BMI 30-34.9 ({bmi_30_35_pct:.1f}%)"),
        mpatches.Patch(facecolor="#ef4444", edgecolor="gray", lw=0.5, label=f"BMI \u2265 35 ({bmi_gte35_pct:.1f}%)"),
    ]
    ax_bars.legend(handles=bmi_patches, loc="lower right", fontsize=9, framealpha=0.92, edgecolor="#d1d5db", fancybox=True)

    bmi30_pct = segments.get("Neither_NoHTN_BMI<30", {}).get("pct", 0)
    pop_m = bmi30_pct / 100 * US_ADULT_POPULATION_2025 / 1e6
    fig.text(0.50, 0.02,
             f"Primary Eligible Pool  (No conditions + BMI < 30): {bmi30_pct:.1f}%  (~{pop_m:.0f}M adults)",
             ha="center", fontsize=11, fontweight="bold", color="#16a34a",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#dcfce7", edgecolor="#86efac", alpha=0.92, lw=1.5))

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_population_diagram_proportional(
    segments: dict[str, Any],
    year: str,
    title: str | None = None,
    save_path: str | None = None,
) -> None:
    """Proportional-area treemap: 3 glycaemic columns x 2 HTN rows x 3 BMI bands."""
    if title is None:
        title = f"Population Health Segments -- Proportional ({year})"

    dm_pct = segments.get("dm_total_pct", 0)
    predm_pct = segments.get("predm_total_pct", 0)
    neither_pct = segments.get("neither_total_pct", 0)
    total_pct = dm_pct + predm_pct + neither_pct
    if total_pct == 0:
        return

    total_w = 14
    w_dm = max(dm_pct / total_pct * total_w, 1.5)
    w_predm = max(predm_pct / total_pct * total_w, 1.5)
    w_neither = max(neither_pct / total_pct * total_w, 1.5)
    wsum = w_dm + w_predm + w_neither
    w_dm, w_predm, w_neither = w_dm / wsum * total_w, w_predm / wsum * total_w, w_neither / wsum * total_w

    total_h = 8
    gap = 0.3

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(-0.5, total_w + 2 * gap + 1.5)
    ax.set_ylim(-1.5, total_h + 2)
    ax.set_aspect("auto"); ax.axis("off")

    glyc_specs = [
        ("DM", w_dm, "#fecaca", "#dc2626", "#991b1b"),
        ("PreDM", w_predm, "#fed7aa", "#ea580c", "#9a3412"),
        ("Neither", w_neither, "#dcfce7", "#16a34a", "#166534"),
    ]
    bmi_colors = {"BMI<30": "#bbf7d0", "BMI30-35": "#fef9c3", "BMI>=35": "#fecaca"}
    bmi_keys = ["BMI<30", "BMI30-35", "BMI>=35"]

    x_cursor = 0.5
    for g_label, g_w, bg_color, edge_color, text_color in glyc_specs:
        htn_pct_col = sum(segments.get(f"{g_label}_HTN_{b}", {}).get("pct", 0) for b in bmi_keys)
        nohtn_pct_col = sum(segments.get(f"{g_label}_NoHTN_{b}", {}).get("pct", 0) for b in bmi_keys)
        col_total = htn_pct_col + nohtn_pct_col
        if col_total == 0:
            x_cursor += g_w + gap
            continue

        h_nohtn = max(nohtn_pct_col / col_total * total_h, 0.8)
        h_htn = max(htn_pct_col / col_total * total_h, 0.8)
        hsum = h_nohtn + h_htn
        h_nohtn, h_htn = h_nohtn / hsum * total_h, h_htn / hsum * total_h

        col_rect = FancyBboxPatch((x_cursor, 0), g_w, total_h, boxstyle="round,pad=0.1",
                                   facecolor=bg_color, edgecolor=edge_color, lw=2, alpha=0.4)
        ax.add_patch(col_rect)

        g_total = {"Neither": segments.get("neither_total_pct", 0),
                    "DM": segments.get("dm_total_pct", 0),
                    "PreDM": segments.get("predm_total_pct", 0)}.get(g_label, col_total)
        ax.text(x_cursor + g_w / 2, total_h + 0.4, f"{g_label}\n{g_total:.1f}%",
                ha="center", va="bottom", fontsize=12, fontweight="bold", color=text_color)

        # No-HTN row (top)
        y_base = h_htn
        nohtn_vals = [segments.get(f"{g_label}_NoHTN_{b}", {}).get("pct", 0) for b in bmi_keys]
        nohtn_sum = sum(nohtn_vals)
        bmi_y = y_base
        for bk, bv in zip(bmi_keys, nohtn_vals):
            if nohtn_sum == 0:
                continue
            bh = bv / nohtn_sum * h_nohtn
            if bh < 0.05:
                bmi_y += bh; continue
            rect = FancyBboxPatch((x_cursor + 0.05, bmi_y + 0.02), g_w - 0.1, bh - 0.04,
                                   boxstyle="round,pad=0.05", facecolor=bmi_colors[bk],
                                   edgecolor="gray", lw=0.5, alpha=0.7)
            ax.add_patch(rect)
            if bh > 0.35:
                ax.text(x_cursor + g_w / 2, bmi_y + bh / 2, f"{bk}\n{bv:.1f}%",
                        ha="center", va="center", fontsize=8, color="#374151")
            bmi_y += bh
        ax.text(x_cursor + g_w / 2, y_base + h_nohtn + 0.05, f"No HTN ({nohtn_sum:.1f}%)",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

        # HTN row (bottom)
        htn_vals = [segments.get(f"{g_label}_HTN_{b}", {}).get("pct", 0) for b in bmi_keys]
        htn_sum = sum(htn_vals)
        htn_bg = FancyBboxPatch((x_cursor + 0.02, 0.02), g_w - 0.04, h_htn - 0.04,
                                 boxstyle="round,pad=0.05", facecolor="#bfdbfe",
                                 edgecolor="#1d4ed8", lw=1, alpha=0.3)
        ax.add_patch(htn_bg)
        bmi_y = 0
        for bk, bv in zip(bmi_keys, htn_vals):
            if htn_sum == 0:
                continue
            bh = bv / htn_sum * h_htn
            if bh < 0.05:
                bmi_y += bh; continue
            rect = FancyBboxPatch((x_cursor + 0.08, bmi_y + 0.04), g_w - 0.16, bh - 0.08,
                                   boxstyle="round,pad=0.05", facecolor=bmi_colors[bk],
                                   edgecolor="#1d4ed8", lw=0.5, alpha=0.6)
            ax.add_patch(rect)
            if bh > 0.35:
                ax.text(x_cursor + g_w / 2, bmi_y + bh / 2, f"{bk}\n{bv:.1f}%",
                        ha="center", va="center", fontsize=8, color="#1e3a5f")
            bmi_y += bh
        ax.text(x_cursor + g_w / 2, -0.15, f"HTN ({htn_sum:.1f}%)",
                ha="center", va="top", fontsize=9, fontweight="bold", color="#1e40af")
        x_cursor += g_w + gap

    legend_patches = [
        mpatches.Patch(facecolor="#fecaca", edgecolor="#dc2626", label="Diabetes (DM)"),
        mpatches.Patch(facecolor="#fed7aa", edgecolor="#ea580c", label="Prediabetes"),
        mpatches.Patch(facecolor="#dcfce7", edgecolor="#16a34a", label="Neither DM/PreDM"),
        mpatches.Patch(facecolor="#bfdbfe", edgecolor="#1d4ed8", label="HTN zone"),
        mpatches.Patch(facecolor="#bbf7d0", edgecolor="gray", label="BMI < 30"),
        mpatches.Patch(facecolor="#fef9c3", edgecolor="gray", label="BMI 30-35"),
        mpatches.Patch(facecolor="#fecaca", edgecolor="gray", label="BMI >= 35"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9, framealpha=0.9, edgecolor="gray", ncol=2)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)
    plt.tight_layout()
    save_and_show(fig, save_path)
