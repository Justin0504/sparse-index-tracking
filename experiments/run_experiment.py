"""End-to-end experiment: Neural + DFL vs. Two-Stage MSE baseline.

Usage::

    python experiments/run_experiment.py [--n-assets 50] [--T 1260] \\
        [--seed 42] [--epochs-2stage 50] [--epochs-dfl 30] \\
        [--batch-size 16] [--lr 1e-3] [--lookback 60]

The script trains both models on synthetic S&P 500-like data, evaluates
out-of-sample tracking error, runs a Diebold–Mariano significance test,
and prints a comparison table.
"""

from __future__ import annotations

import argparse
import sys
import os

# Allow running from the repository root or the experiments/ directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from src.data import create_data_splits, generate_synthetic_sp500
from src.evaluate import diebold_mariano_test, evaluate_model, print_comparison
from src.models import DFLModel, TwoStageModel
from src.train import train_dfl, train_two_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DFL vs Two-Stage for sparse index tracking"
    )
    parser.add_argument("--n-assets", type=int, default=50)
    parser.add_argument("--T", type=int, default=1260, help="Total trading days")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs-2stage", type=int, default=50)
    parser.add_argument("--epochs-dfl", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-warmstart", action="store_true",
                        help="Do not warm-start DFL from two-stage weights")
    parser.add_argument("--verbose", action="store_true", default=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    """Run the full experiment and return a results dictionary."""
    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    print(f"\n{'='*60}")
    print("  Generating synthetic S&P 500-like data …")
    print(f"  n_assets={args.n_assets}, T={args.T}, seed={args.seed}")
    print(f"{'='*60}")

    returns, index_returns = generate_synthetic_sp500(
        n_assets=args.n_assets,
        T=args.T,
        seed=args.seed,
    )

    train_ds, val_ds, test_ds = create_data_splits(
        returns,
        index_returns,
        train_ratio=0.6,
        val_ratio=0.2,
        lookback=args.lookback,
    )

    print(
        f"  Train: {len(train_ds)} samples | "
        f"Val: {len(val_ds)} samples | "
        f"Test: {len(test_ds)} samples"
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # ------------------------------------------------------------------
    # Two-stage MSE baseline
    # ------------------------------------------------------------------
    print("\n  Training two-stage MSE baseline …")
    two_stage = TwoStageModel(
        n_assets=args.n_assets,
        lookback=args.lookback,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
    )
    two_stage = train_two_stage(
        two_stage,
        train_loader,
        val_loader,
        n_epochs=args.epochs_2stage,
        lr=args.lr,
        device=device,
        verbose=args.verbose,
    )

    results_baseline = evaluate_model(two_stage, test_loader, device)

    # ------------------------------------------------------------------
    # End-to-end DFL model
    # ------------------------------------------------------------------
    print("\n  Training end-to-end DFL model …")
    dfl = DFLModel(
        n_assets=args.n_assets,
        lookback=args.lookback,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
    )
    warmstart = (
        None if args.no_warmstart else two_stage.state_dict()
    )
    dfl = train_dfl(
        dfl,
        train_loader,
        val_loader,
        n_epochs=args.epochs_dfl,
        lr=args.lr,
        device=device,
        verbose=args.verbose,
        warmstart_state=warmstart,
    )

    results_dfl = evaluate_model(dfl, test_loader, device)

    # ------------------------------------------------------------------
    # Statistical comparison
    # ------------------------------------------------------------------
    dm = diebold_mariano_test(
        results_baseline["tracking_diffs"],
        results_dfl["tracking_diffs"],
    )
    print_comparison(results_baseline, results_dfl, dm)

    return {
        "baseline": results_baseline,
        "dfl": results_dfl,
        "dm_test": dm,
    }


if __name__ == "__main__":
    args = parse_args()
    run(args)
