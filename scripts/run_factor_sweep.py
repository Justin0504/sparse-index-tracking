"""Factor number sweep: vary n_factors to study misspecification.

Runs Factor+MSE and Factor+DFL with n_factors in {3, 5, 10, 20}.
Fewer factors = more misspecified model = where DFL should help most.

Usage:
    python scripts/run_factor_sweep.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

FACTORS = [3, 5, 10, 20]
LOSSES = ["mse", "task"]


def main():
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "results" / "factor_sweep"
    results_dir.mkdir(parents=True, exist_ok=True)

    for n_factors in FACTORS:
        for loss in LOSSES:
            tag = f"factor_{loss}_nf{n_factors}"
            print(f"\n{'='*60}")
            print(f"Running: {tag}")
            print(f"{'='*60}\n")

            cmd = [
                sys.executable, "-m", "src.main",
                f"model.name=factor",
                f"model.factor.n_factors={n_factors}",
                f"training.loss={loss}",
                # Override output path via hydra
                f"hydra.run.dir=results/factor_sweep/{tag}",
            ]

            result = subprocess.run(
                cmd,
                cwd=str(project_root),
                capture_output=False,
            )

            if result.returncode != 0:
                print(f"WARNING: {tag} exited with code {result.returncode}")
            else:
                print(f"Completed: {tag}")

    print("\n" + "=" * 60)
    print("Factor sweep complete. Results in results/factor_sweep/")
    print("=" * 60)


if __name__ == "__main__":
    main()
