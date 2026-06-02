#!/usr/bin/env python3
"""Generate the diagnostic hero figure: 2D heatmap of DFL gain vs K/N and model_dim.

x-axis: K/N ratio (0 → 1)
y-axis: model dimensionality (1 → 5000)
color:  DFL gain % (red = DFL hurts, white = neutral, blue = DFL helps)
markers: each experiment, green border = prediction correct, red border = wrong
overlay: prediction boundary curves from the diagnostic rule
"""

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_scoring():
    path = Path("results/diagnostic/scoring.json")
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_predictions():
    path = Path("results/diagnostic/blind_predictions.json")
    with open(path) as f:
        return json.load(f)


def main():
    scoring = load_scoring()
    predictions = load_predictions()

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    kn_grid = np.linspace(0.005, 1.0, 200)
    dim_grid = np.logspace(0, np.log10(5000), 200)
    KN, DIM = np.meshgrid(kn_grid, dim_grid)

    alignment = np.sqrt(2 * KN)
    gain_pred = np.zeros_like(alignment)
    gain_pred[(alignment > 0.5) & (DIM <= 1)] = -2.0
    gain_pred[(alignment > 0.7)] = np.minimum(gain_pred[(alignment > 0.7)], -2.0)
    gain_pred[(DIM <= 1)] = np.where(
        gain_pred[(DIM <= 1)] == 0, 0.0, gain_pred[(DIM <= 1)]
    )
    gain_pred[(alignment < 0.35) & (DIM >= 10)] = 5.0
    gain_pred[(alignment >= 0.35) & (alignment <= 0.5) & (DIM >= 10)] = 1.5

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "dfl_gain", ["#d73027", "#f7f7f7", "#4575b4"], N=256
    )
    norm = mcolors.TwoSlopeNorm(vmin=-5, vcenter=0, vmax=10)

    im = ax.pcolormesh(KN, DIM, gain_pred, cmap=cmap, norm=norm, shading="auto", alpha=0.4)

    boundary_kn = np.linspace(0.005, 1.0, 500)
    ax.axvline(x=0.5**2 / 2, color="gray", linestyle="--", linewidth=1, alpha=0.5,
               label=r"$\sqrt{2K/N}=0.5$")
    b_kn_05 = 0.5**2 / 2
    b_kn_035 = 0.35**2 / 2
    b_kn_07 = 0.7**2 / 2
    ax.axvline(x=b_kn_035, color="gray", linestyle=":", linewidth=1, alpha=0.5,
               label=r"$\sqrt{2K/N}=0.35$")
    ax.axvline(x=b_kn_07, color="gray", linestyle="-.", linewidth=1, alpha=0.5,
               label=r"$\sqrt{2K/N}=0.7$")
    ax.axhline(y=10, color="black", linestyle="--", linewidth=0.8, alpha=0.4,
               label=r"$d=10$")

    domain_markers = {
        "finance": "o",
        "weather": "s",
        "air_quality": "D",
    }
    domain_colors = {
        "finance": "#1f77b4",
        "weather": "#2ca02c",
        "air_quality": "#ff7f0e",
    }

    if scoring and scoring.get("scored"):
        for s in scoring["scored"]:
            kn = s["alignment"]**2 / 2
            dim = s["model_dim"]
            gain = s["dfl_gain_pct"]
            correct = s["correct"]
            domain = s["domain"]

            marker = domain_markers.get(domain, "o")
            facecolor = cmap(norm(gain))
            edgecolor = "#2ca02c" if correct else "#d62728"
            edgewidth = 2.0

            ax.scatter(kn, dim, marker=marker, s=80,
                       facecolors=facecolor, edgecolors=edgecolor,
                       linewidths=edgewidth, zorder=5)
    else:
        for case in predictions["cases"]:
            kn = case["alignment"]**2 / 2
            dim = case["model_dim"]
            domain = case["domain"]
            predicted = case["predict"]

            marker = domain_markers.get(domain, "o")
            color = domain_colors.get(domain, "gray")

            ax.scatter(kn, dim, marker=marker, s=60,
                       facecolors=color, edgecolors="black",
                       linewidths=0.8, zorder=5, alpha=0.7)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$K/N$ ratio", fontsize=13)
    ax.set_ylabel(r"Model dimensionality $d$", fontsize=13)
    ax.set_title("Gradient Alignment Diagnostic: When Does DFL Help?", fontsize=14)
    ax.set_xlim(0.003, 0.6)
    ax.set_ylim(0.8, 6000)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Predicted DFL Gain (%)", fontsize=11)

    from matplotlib.lines import Line2D
    legend_elements = []
    for domain, marker in domain_markers.items():
        legend_elements.append(
            Line2D([0], [0], marker=marker, color="w",
                   markerfacecolor=domain_colors.get(domain, "gray"),
                   markeredgecolor="black", markersize=8,
                   label=domain.replace("_", " ").title())
        )
    if scoring:
        legend_elements.append(
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="white", markeredgecolor="#2ca02c",
                   markeredgewidth=2, markersize=8, label="Correct prediction")
        )
        legend_elements.append(
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="white", markeredgecolor="#d62728",
                   markeredgewidth=2, markersize=8, label="Wrong prediction")
        )

    ax.legend(handles=legend_elements, loc="upper right", fontsize=9,
              framealpha=0.9)

    ax.text(0.006, 2000, "DFL wins\n(low K/N, rich model)",
            fontsize=9, color="#4575b4", fontweight="bold", alpha=0.7)
    ax.text(0.15, 2, "MSE sufficient\n(high K/N or simple model)",
            fontsize=9, color="#d73027", fontweight="bold", alpha=0.7)

    plt.tight_layout()
    out_dir = Path("results/diagnostic")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "diagnostic_heatmap.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved diagnostic heatmap to {out_path}")

    png_path = out_dir / "diagnostic_heatmap.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved PNG to {png_path}")
    plt.close()


if __name__ == "__main__":
    main()
