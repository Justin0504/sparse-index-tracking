#!/usr/bin/env python3
"""Generate the diagnostic hero figure: DFL gain vs alignment and effective dim.

Uses effective_dim = model_dim × het_factor (domain heterogeneity coefficient).
Two panels:
  Left:  alignment vs effective_dim, color = actual DFL gain %
  Right: alignment vs DFL gain %, grouped by effective_dim regime
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


EXPERIMENT_DATA = [
    # (label, domain, K, N, model_dim, het_factor, alignment, dfl_gain_pct)
    # Finance dim=1 (het=1.0 by convention)
    ("S&P 500\n20y crisis", "finance", 20, 100, 1, 1.0, 0.632, 0.00),
    ("S&P 500\n20y covid", "finance", 20, 100, 1, 1.0, 0.632, 0.00),
    ("S&P 500\n20y normal", "finance", 20, 100, 1, 1.0, 0.632, 0.09),
    ("Nikkei K=5", "finance", 5, 218, 1, 1.0, 0.214, 0.30),
    ("Nikkei K=10", "finance", 10, 218, 1, 1.0, 0.303, 0.01),
    ("Nikkei K=20", "finance", 20, 218, 1, 1.0, 0.428, 0.02),
    ("EuroStoxx K=5", "finance", 5, 47, 1, 1.0, 0.461, 0.11),
    ("EuroStoxx K=10", "finance", 10, 47, 1, 1.0, 0.652, 0.33),
    ("EuroStoxx K=20", "finance", 20, 47, 1, 1.0, 0.923, 0.24),
    ("ASX K=5", "finance", 5, 162, 1, 1.0, 0.248, 0.55),
    ("ASX K=10", "finance", 10, 162, 1, 1.0, 0.351, 0.51),
    ("ASX K=20", "finance", 20, 162, 1, 1.0, 0.497, 0.07),
    ("HangSeng K=5", "finance", 5, 62, 1, 1.0, 0.402, 0.95),
    ("HangSeng K=10", "finance", 10, 62, 1, 1.0, 0.568, 1.11),
    ("HangSeng K=20", "finance", 20, 62, 1, 1.0, 0.803, 0.19),
    # SP1500 dim=1 shrinkage
    ("SP1500\nK=10", "finance", 10, 1224, 1, 1.0, 0.128, 0.83),
    ("SP1500\nK=20", "finance", 20, 1224, 1, 1.0, 0.181, 0.29),
    ("SP1500\nK=50", "finance", 50, 1224, 1, 1.0, 0.286, -0.14),
    # Finance dim=12 (structured model, het=1.0)
    ("S&P 500\n20y struct", "finance_struct", 20, 100, 12, 1.0, 0.632, 0.45),
    ("SP1500\nK=10 struct", "finance_struct", 10, 1224, 12, 1.0, 0.128, 1.73),
    ("SP1500\nK=20 struct", "finance_struct", 20, 1224, 12, 1.0, 0.181, 1.93),
    ("SP1500\nK=50 struct", "finance_struct", 50, 1224, 12, 1.0, 0.286, 0.41),
    # Weather dim=10, het=0.144
    ("Weather\nN100 K=5", "weather", 5, 100, 10, 0.144, 0.316, -0.29),
    ("Weather\nN100 K=10", "weather", 10, 100, 10, 0.144, 0.447, -1.55),
    ("Weather\nN100 K=20", "weather", 20, 100, 10, 0.144, 0.632, -3.61),
    ("Weather\nN100 K=50", "weather", 50, 100, 10, 0.144, 1.000, -11.19),
    ("Weather\nN200 K=5", "weather", 5, 189, 10, 0.144, 0.230, 0.04),
    ("Weather\nN200 K=10", "weather", 10, 189, 10, 0.144, 0.325, 1.01),
    ("Weather\nN200 K=20", "weather", 20, 189, 10, 0.144, 0.460, -1.81),
    ("Weather\nN200 K=50", "weather", 50, 189, 10, 0.144, 0.727, -6.99),
    ("Weather\nN200 K=94", "weather", 94, 189, 10, 0.144, 0.997, -10.53),
    ("Weather\nN415 K=5", "weather", 5, 415, 10, 0.144, 0.155, -0.07),
    ("Weather\nN415 K=10", "weather", 10, 415, 10, 0.144, 0.220, 0.02),
    ("Weather\nN415 K=20", "weather", 20, 415, 10, 0.144, 0.310, -2.48),
    ("Weather\nN415 K=50", "weather", 50, 415, 10, 0.144, 0.491, -15.09),
    ("Weather\nN415 K=207", "weather", 207, 415, 10, 0.144, 0.999, -11.95),
    # EPA dim=10, het=0.606
    ("EPA N132\nK=10", "air_quality", 10, 132, 10, 0.606, 0.389, 5.76),
    ("EPA N132\nK=20", "air_quality", 20, 132, 10, 0.606, 0.550, 2.24),
    ("EPA N132\nK=50", "air_quality", 50, 132, 10, 0.606, 0.870, 9.41),
    ("EPA N124\nK=10", "air_quality", 10, 124, 10, 0.606, 0.402, 0.29),
    ("EPA N124\nK=20", "air_quality", 20, 124, 10, 0.606, 0.568, 0.52),
    ("EPA N124\nK=50", "air_quality", 50, 124, 10, 0.606, 0.898, 4.03),
]

DOMAIN_MARKERS = {"finance": "o", "finance_struct": "^", "weather": "s", "air_quality": "D"}
DOMAIN_LABELS = {"finance": "Finance (dim=1)", "finance_struct": "Finance (dim=12)",
                 "weather": "Weather (dim=10, het=0.14)",
                 "air_quality": "EPA PM2.5 (dim=10, het=0.61)"}


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "dfl_gain", ["#d73027", "#f7f7f7", "#4575b4"], N=256
    )
    norm = mcolors.TwoSlopeNorm(vmin=-16, vcenter=0, vmax=10)

    # Panel 1: alignment vs effective_dim, color = DFL gain
    for label, domain, K, N, dim, het, align, gain in EXPERIMENT_DATA:
        eff_dim = dim * het
        marker = DOMAIN_MARKERS[domain]
        facecolor = cmap(norm(gain))
        ax1.scatter(align, eff_dim, marker=marker, s=100,
                    facecolors=facecolor, edgecolors="black",
                    linewidths=1.0, zorder=5)

    ax1.axhline(y=1.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax1.axvline(x=0.4, color="gray", linestyle=":", linewidth=1, alpha=0.5)

    ax1.fill_between([0, 0.4], 0.5, 1.5, alpha=0.08, color="#aaaaaa",
                     label=r"$d_{\mathrm{eff}} \leq 1.5$: equivalent")
    ax1.fill_between([0.4, 1.5], 0.5, 1.5, alpha=0.08, color="#d73027",
                     label=r"$d_{\mathrm{eff}} \leq 1.5$, high align: MSE suff.")
    ax1.fill_between([0, 0.4], 4, 15, alpha=0.06, color="#4575b4",
                     label=r"$d_{\mathrm{eff}} \geq 4$, low align: DFL wins")

    ax1.set_xlabel(r"Gradient alignment $\sqrt{2K/N}$", fontsize=12)
    ax1.set_ylabel(r"Effective dim $d \times h$", fontsize=12)
    ax1.set_title("(a) DFL gain by alignment and effective model dimension", fontsize=11)
    ax1.set_xlim(0.1, 1.1)
    ax1.set_ylim(0.5, 15)
    ax1.legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax1.set_yscale("log")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax1, shrink=0.8, pad=0.02)
    cbar.set_label("DFL gain (%)", fontsize=10)

    # Panel 2: alignment vs DFL gain %, grouped by domain
    domain_groups = {}
    for label, domain, K, N, dim, het, align, gain in EXPERIMENT_DATA:
        domain_groups.setdefault(domain, []).append((align, gain))

    for domain, points in domain_groups.items():
        aligns, gains = zip(*points)
        marker = DOMAIN_MARKERS[domain]
        ax2.scatter(aligns, gains, marker=marker, s=80, label=DOMAIN_LABELS[domain],
                    edgecolors="black", linewidths=0.8, alpha=0.8, zorder=5)

    ax2.axhline(y=0, color="black", linewidth=0.5, alpha=0.5)
    ax2.axhline(y=1, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.axhline(y=-1, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.fill_between([0, 1.5], -1, 1, alpha=0.05, color="gray")

    ax2.set_xlabel(r"Gradient alignment $\sqrt{2K/N}$", fontsize=12)
    ax2.set_ylabel("DFL gain over MSE (%)", fontsize=12)
    ax2.set_title("(b) DFL gain by domain", fontsize=11)
    ax2.set_xlim(0.1, 1.4)
    ax2.legend(fontsize=9, loc="upper left", framealpha=0.9)

    ax2.annotate("DFL helps", xy=(0.15, 7), fontsize=9, color="#4575b4",
                 fontweight="bold", alpha=0.7)
    ax2.annotate("DFL hurts", xy=(0.9, -9), fontsize=9, color="#d73027",
                 fontweight="bold", alpha=0.7)

    plt.tight_layout()
    out_dir = Path("results/diagnostic")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "diagnostic_heatmap.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved to {out_path}")

    png_path = out_dir / "diagnostic_heatmap.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {png_path}")
    plt.close()


if __name__ == "__main__":
    main()
