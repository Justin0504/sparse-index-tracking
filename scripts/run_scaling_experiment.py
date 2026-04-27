"""Run scaling experiment: N=100 vs N=500 to test DFL gain at larger universe.

Key hypothesis: DFL gains should persist (or increase) at N=500 since K/N is even
smaller, making the gradient misalignment more severe.

Usage:
    .venv/bin/python scripts/run_scaling_experiment.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from src.utils import get_logger

log = get_logger(__name__)

# N=500 experiments: K values that span interesting K/N ratios
# K=20 → K/N=0.04, K=50 → K/N=0.10, K=100 → K/N=0.20
K_VALUES = [20, 50, 100]
N_VALUE = 500


def main():
    root = str(Path(__file__).resolve().parent.parent)
    results = {}

    for K in K_VALUES:
        for loss in ["mse", "task"]:
            label = f"neural_{loss}_N{N_VALUE}_K{K}"
            log.info(f"\n>>> Running {label}...")
            cmd = [
                sys.executable, "-m", "src.main",
                "model.name=neural",
                f"training.loss={loss}",
                f"data.top_n_by_mcap={N_VALUE}",
                f"optimization.sparsity_levels=[{K}]",
                "training.device=cpu",
                "+run_naive_baseline=false",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=root)
            lines = result.stdout.strip().split('\n')
            for l in lines[-5:]:
                log.info(f"  {l}")
            if result.returncode != 0:
                log.error(f"  FAILED: {result.stderr[-300:]}")
            else:
                results[label] = "done"

    log.info(f"\n>>> Completed {len(results)}/{len(K_VALUES)*2} experiments")


if __name__ == "__main__":
    main()
