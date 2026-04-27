#!/usr/bin/env python3
"""Run experiments for multi-step lookahead and curriculum learning.

Compares:
  1. Baseline DFL (lookahead=1, no curriculum)
  2. Multi-step lookahead (lookahead=2,3)
  3. Curriculum learning (curriculum_epochs=5,10)
  4. Combined: lookahead + curriculum

For both K=10 and K=20, using shrinkage and conditional_shrinkage models.
"""

import subprocess
import sys

PYTHON = sys.executable

CONFIGS = [
    # ── Shrinkage model ──
    # Baseline DFL
    ("shrinkage_task_baseline_K10", [
        "model.name=shrinkage", "training.loss=task",
        "optimization.sparsity_levels=[10]",
        "training.lookahead_steps=1", "training.curriculum_epochs=0",
    ]),
    ("shrinkage_task_baseline_K20", [
        "model.name=shrinkage", "training.loss=task",
        "optimization.sparsity_levels=[20]",
        "training.lookahead_steps=1", "training.curriculum_epochs=0",
    ]),
    # Lookahead=2
    ("shrinkage_task_look2_K10", [
        "model.name=shrinkage", "training.loss=task",
        "optimization.sparsity_levels=[10]",
        "training.lookahead_steps=2", "training.curriculum_epochs=0",
    ]),
    ("shrinkage_task_look2_K20", [
        "model.name=shrinkage", "training.loss=task",
        "optimization.sparsity_levels=[20]",
        "training.lookahead_steps=2", "training.curriculum_epochs=0",
    ]),
    # Lookahead=3
    ("shrinkage_task_look3_K10", [
        "model.name=shrinkage", "training.loss=task",
        "optimization.sparsity_levels=[10]",
        "training.lookahead_steps=3", "training.curriculum_epochs=0",
    ]),
    ("shrinkage_task_look3_K20", [
        "model.name=shrinkage", "training.loss=task",
        "optimization.sparsity_levels=[20]",
        "training.lookahead_steps=3", "training.curriculum_epochs=0",
    ]),
    # Curriculum=5
    ("shrinkage_task_curric5_K10", [
        "model.name=shrinkage", "training.loss=task",
        "optimization.sparsity_levels=[10]",
        "training.lookahead_steps=1", "training.curriculum_epochs=5",
    ]),
    ("shrinkage_task_curric5_K20", [
        "model.name=shrinkage", "training.loss=task",
        "optimization.sparsity_levels=[20]",
        "training.lookahead_steps=1", "training.curriculum_epochs=5",
    ]),
    # Curriculum=10
    ("shrinkage_task_curric10_K10", [
        "model.name=shrinkage", "training.loss=task",
        "optimization.sparsity_levels=[10]",
        "training.lookahead_steps=1", "training.curriculum_epochs=10",
    ]),
    ("shrinkage_task_curric10_K20", [
        "model.name=shrinkage", "training.loss=task",
        "optimization.sparsity_levels=[20]",
        "training.lookahead_steps=1", "training.curriculum_epochs=10",
    ]),
    # Combined: lookahead=2 + curriculum=5
    ("shrinkage_task_look2_curric5_K10", [
        "model.name=shrinkage", "training.loss=task",
        "optimization.sparsity_levels=[10]",
        "training.lookahead_steps=2", "training.curriculum_epochs=5",
    ]),
    ("shrinkage_task_look2_curric5_K20", [
        "model.name=shrinkage", "training.loss=task",
        "optimization.sparsity_levels=[20]",
        "training.lookahead_steps=2", "training.curriculum_epochs=5",
    ]),

    # ── Conditional shrinkage model ──
    # Baseline (already run, but re-run for consistency)
    ("cond_task_baseline_K10", [
        "model.name=conditional_shrinkage", "training.loss=task",
        "optimization.sparsity_levels=[10]",
        "training.lookahead_steps=1", "training.curriculum_epochs=0",
    ]),
    ("cond_task_baseline_K20", [
        "model.name=conditional_shrinkage", "training.loss=task",
        "optimization.sparsity_levels=[20]",
        "training.lookahead_steps=1", "training.curriculum_epochs=0",
    ]),
    # Lookahead=2
    ("cond_task_look2_K10", [
        "model.name=conditional_shrinkage", "training.loss=task",
        "optimization.sparsity_levels=[10]",
        "training.lookahead_steps=2", "training.curriculum_epochs=0",
    ]),
    ("cond_task_look2_K20", [
        "model.name=conditional_shrinkage", "training.loss=task",
        "optimization.sparsity_levels=[20]",
        "training.lookahead_steps=2", "training.curriculum_epochs=0",
    ]),
    # Curriculum=5
    ("cond_task_curric5_K10", [
        "model.name=conditional_shrinkage", "training.loss=task",
        "optimization.sparsity_levels=[10]",
        "training.lookahead_steps=1", "training.curriculum_epochs=5",
    ]),
    ("cond_task_curric5_K20", [
        "model.name=conditional_shrinkage", "training.loss=task",
        "optimization.sparsity_levels=[20]",
        "training.lookahead_steps=1", "training.curriculum_epochs=5",
    ]),
    # Combined
    ("cond_task_look2_curric5_K10", [
        "model.name=conditional_shrinkage", "training.loss=task",
        "optimization.sparsity_levels=[10]",
        "training.lookahead_steps=2", "training.curriculum_epochs=5",
    ]),
    ("cond_task_look2_curric5_K20", [
        "model.name=conditional_shrinkage", "training.loss=task",
        "optimization.sparsity_levels=[20]",
        "training.lookahead_steps=2", "training.curriculum_epochs=5",
    ]),
]


def main():
    common = ["+run_naive_baseline=false"]
    for name, overrides in CONFIGS:
        print(f"\n>>> Running {name}...")
        cmd = [PYTHON, "-m", "src.main"] + overrides + common
        result = subprocess.run(cmd, capture_output=True, text=True)
        # Print only INFO lines with metrics
        for line in (result.stdout + result.stderr).splitlines():
            if "TE=" in line or "Results saved" in line:
                print(f"  {line}")
        if result.returncode != 0:
            print(f"  [FAILED] returncode={result.returncode}")
            # Print last few error lines
            for line in result.stderr.splitlines()[-5:]:
                print(f"  {line}")


if __name__ == "__main__":
    main()
