#!/usr/bin/env python3
"""Generate blind predictions for the live diagnostic protocol.

CRITICAL: This script must be run BEFORE any new experiments.
The predictions are locked with a timestamp and should be committed to git.
"""

import json
import math
from datetime import datetime
from pathlib import Path


def compute_alignment(K: int, N: int) -> float:
    return math.sqrt(2 * K / N)


def predict(alignment: float, model_dim: int) -> str:
    """Apply the diagnostic prediction rule.

    Rules derived from existing experiments (S&P 500 N=100/451, FTSE 100):
    - alignment > 0.5 and d<=1, or alignment > 0.7 → MSE sufficient
    - d <= 1 (any alignment) → three paths equivalent
    - alignment < 0.35 and d >= 10 → DFL wins (richer models)
    - otherwise → ambiguous
    """
    if alignment > 0.5 and model_dim <= 1:
        return "mse_sufficient"
    elif alignment > 0.7:
        return "mse_sufficient"
    elif model_dim <= 1:
        return "equivalent"
    elif alignment < 0.35 and model_dim >= 10:
        return "dfl_wins"
    elif alignment >= 0.35 and model_dim >= 10:
        return "ambiguous"
    else:
        return "ambiguous"


def main():
    cases = []

    # === DIMENSION 1: Scale (S&P 1500) ===
    for K in [10, 20, 50]:
        N_est = 1300
        a = compute_alignment(K, N_est)
        cases.append({
            "id": f"SP1500_K{K}_shrink",
            "domain": "finance", "market": "SP1500",
            "N": N_est, "K": K, "model_dim": 1, "model": "shrinkage",
            "alignment": round(a, 4),
            "predict": predict(a, 1),
            "rationale": f"K/N={K/N_est:.3f}, d=1: grid search covers 1D"
        })
        cases.append({
            "id": f"SP1500_K{K}_struct",
            "domain": "finance", "market": "SP1500",
            "N": N_est, "K": K, "model_dim": 12, "model": "structured",
            "alignment": round(a, 4),
            "predict": predict(a, 12),
            "rationale": f"K/N={K/N_est:.3f}, d=12: DFL should help richer model"
        })

    # === DIMENSION 1: Multi-market ===
    markets = [
        ("Nikkei225", 200, "^N225"),
        ("EuroStoxx50", 48, "^STOXX50E"),
        ("ASX200", 180, "^AXJO"),
        ("HangSeng", 50, "^HSI"),
    ]
    for mkt, N_est, _ in markets:
        for K in [5, 10, 20]:
            if K > N_est // 2:
                continue
            a = compute_alignment(K, N_est)
            cases.append({
                "id": f"{mkt}_K{K}_shrink",
                "domain": "finance", "market": mkt,
                "N": N_est, "K": K, "model_dim": 1, "model": "shrinkage",
                "alignment": round(a, 4),
                "predict": predict(a, 1),
                "rationale": f"K/N={K/N_est:.3f}, d=1: 1-param model"
            })

    # === DIMENSION 2: Weather stations ===
    for N in [100, 200, 500]:
        for K in [5, 10, 20, 50]:
            if K > N // 2:
                continue
            a = compute_alignment(K, N)
            cases.append({
                "id": f"Weather_N{N}_K{K}",
                "domain": "weather", "market": "NOAA_ASOS",
                "N": N, "K": K, "model_dim": 10, "model": "spatial_structured",
                "alignment": round(a, 4),
                "predict": predict(a, 10),
                "rationale": f"K/N={K/N:.3f}, d=10: spatial structured model"
            })
        # High K/N case
        K_high = N // 2
        a = compute_alignment(K_high, N)
        cases.append({
            "id": f"Weather_N{N}_K{K_high}",
            "domain": "weather", "market": "NOAA_ASOS",
            "N": N, "K": K_high, "model_dim": 10, "model": "spatial_structured",
            "alignment": round(a, 4),
            "predict": predict(a, 10),
            "rationale": f"K/N={K_high/N:.3f}: high K/N, DFL should not help"
        })

    # === DIMENSION 2: EPA air quality ===
    for N in [200, 500, 800]:
        for K in [10, 20, 50]:
            if K > N // 2:
                continue
            a = compute_alignment(K, N)
            cases.append({
                "id": f"EPA_N{N}_K{K}",
                "domain": "air_quality", "market": "EPA_AQS",
                "N": N, "K": K, "model_dim": 10, "model": "spatial_structured",
                "alignment": round(a, 4),
                "predict": predict(a, 10),
                "rationale": f"K/N={K/N:.3f}, d=10: spatial structured model"
            })

    predictions = {
        "generated_at": datetime.now().isoformat(),
        "diagnostic_version": "v1.0",
        "prediction_rules": {
            "mse_sufficient": "alignment > 0.5 (d<=1) or > 0.7 → MSE-task gradients aligned",
            "equivalent": "d <= 1, any alignment → all three paths converge",
            "dfl_wins": "alignment < 0.35, d >= 10 → DFL enables richer model optimization",
            "ambiguous": "intermediate regime, outcome is context-dependent",
        },
        "scoring_threshold_pct": 1.0,
        "total_cases": len(cases),
        "cases": cases,
    }

    # Category summary
    cats = {}
    for c in cases:
        p = c["predict"]
        cats[p] = cats.get(p, 0) + 1
    predictions["category_counts"] = cats

    out = Path("results/diagnostic/blind_predictions.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"Generated {len(cases)} blind predictions:")
    for cat, cnt in sorted(cats.items()):
        print(f"  {cat}: {cnt}")
    print(f"\nSaved to {out}")
    print("IMPORTANT: Commit this file BEFORE running any new experiments.")


if __name__ == "__main__":
    main()
