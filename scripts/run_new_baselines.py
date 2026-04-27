"""Run only the new AAAI baselines (val-tuned shrinkage + SPO+).

Usage:
    .venv/bin/python scripts/run_new_baselines.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_baselines import (
    load_data,
    run_val_tuned_baseline,
    run_spo_plus_baseline,
    SPARSITY_LEVELS,
)
from src.utils import get_logger
import numpy as np

log = get_logger(__name__)


def main():
    prices, returns, tickers, folds, fb = load_data()
    log.info(f"Loaded {len(tickers)} stocks, {len(folds)} folds")

    out_dir = Path("results/baselines")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Validation-tuned shrinkage (grid search α)
    log.info("\n>>> Running Validation-Tuned Shrinkage...")
    val_tuned_results = run_val_tuned_baseline(prices, returns, folds, fb)
    with open(out_dir / "val_tuned_shrinkage_results.json", "w") as f:
        json.dump({"results": val_tuned_results}, f, indent=2)
    log.info(f"Val-tuned shrinkage results saved ({len(val_tuned_results)} entries)")

    # 2. SPO+ baseline
    log.info("\n>>> Running SPO+ Baseline...")
    spo_results = run_spo_plus_baseline(prices, returns, folds, fb)
    with open(out_dir / "spo_plus_results.json", "w") as f:
        json.dump({"results": spo_results}, f, indent=2)
    log.info(f"SPO+ results saved ({len(spo_results)} entries)")

    # Summary
    log.info("\n" + "=" * 60)
    log.info("SUMMARY: Mean Tracking Error by K")
    log.info("=" * 60)
    for name, results in [("Val-Tuned", val_tuned_results), ("SPO+", spo_results)]:
        log.info(f"\n{name}:")
        for K in SPARSITY_LEVELS:
            tes = [r["tracking_error"] for r in results if r["K"] == K]
            if tes:
                log.info(f"  K={K:3d}: TE={np.mean(tes):.4f} +/- {np.std(tes):.4f}")


if __name__ == "__main__":
    main()
