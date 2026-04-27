#!/usr/bin/env python3
"""Conditional shrinkage diagnostics: extract train/val curves to diagnose overfitting."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.features import FeatureBuilder
from src.data.loader import SP500DataLoader
from src.data.splitter import RollingWindowSplitter
from src.models.shrinkage_model import ConditionalShrinkageModel, StructuredShrinkageModel
from src.training import Trainer
from src.utils import set_seed, get_logger

log = get_logger(__name__)


def run_diagnostics(model_type="conditional", K=20, loss_type="task", device="cpu"):
    set_seed(42)

    fb = FeatureBuilder(vol_window=21, momentum_window=63,
                        use_sector=True, use_mcap_quintile=True)
    loader = SP500DataLoader(start_date="2006-01-01", end_date="2025-12-31", top_n=100)
    prices, index_prices, tickers = loader.load()
    n_stocks = len(tickers)

    returns = fb.compute_returns(prices)
    splitter = RollingWindowSplitter(train_years=3, val_months=6,
                                     test_years=1, step_years=1)
    folds = splitter.split(returns.index)

    min_offset = 64
    sample_date = returns.index[min_offset + 1]
    n_features = fb.build_features(returns, sample_date).shape[1]
    n_market_features = fb.build_market_features(returns, sample_date).shape[0]

    all_diagnostics = []

    for fold_info in folds:
        train_mask = (returns.index >= fold_info.train_start) & (returns.index <= fold_info.train_end)
        val_mask = (returns.index >= fold_info.val_start) & (returns.index <= fold_info.val_end)

        ret_train = returns.loc[train_mask]
        ret_val = returns.loc[val_mask]

        train_prices = prices.loc[prices.index.isin(ret_train.index)]
        last_p = train_prices.iloc[-1].values
        w_index = (last_p / last_p.sum()).astype(np.float32)

        def _get_rebal(ret_df):
            dates = [d for d in ret_df.index if d.weekday() == 4]
            return [d for d in dates if returns.index.get_loc(d) >= min_offset]

        rebal_train = _get_rebal(ret_train)
        rebal_val = _get_rebal(ret_val)

        if not rebal_train:
            continue

        if model_type == "conditional":
            model = ConditionalShrinkageModel(n_stocks=n_stocks,
                                               n_market_features=n_market_features,
                                               hidden_dim=16)
        else:
            model = StructuredShrinkageModel(n_stocks=n_stocks, n_sectors=11)

        total_params = sum(p.numel() for p in model.parameters())

        trainer = Trainer(
            model=model, feature_builder=fb, loss_type=loss_type,
            lr=1e-3, weight_decay=1e-4, epochs=50, patience=10,
            device=device, sparsity_K=K, selection="market_cap",
        )

        history = trainer.train(
            returns_train=ret_train.values,
            returns_val=ret_val.values,
            rebalance_dates_train=rebal_train,
            rebalance_dates_val=rebal_val,
            returns_df_train=returns,
            returns_df_val=returns,
            w_index=w_index,
            lookback=63,
        )

        diag = {
            "fold": fold_info.fold,
            "model_type": model_type,
            "loss": loss_type,
            "n_params": total_params,
            "train_losses": [float(x) for x in history.get("train_losses", [])],
            "val_losses": [float(x) for x in history.get("val_losses", [])],
            "best_epoch": int(history.get("best_epoch", -1)),
        }
        all_diagnostics.append(diag)

        n_epochs = len(diag["train_losses"])
        final_train = diag["train_losses"][-1] if diag["train_losses"] else float("nan")
        final_val = diag["val_losses"][-1] if diag["val_losses"] else float("nan")
        best_train = diag["train_losses"][diag["best_epoch"]] if diag["best_epoch"] >= 0 and diag["train_losses"] else float("nan")
        best_val = diag["val_losses"][diag["best_epoch"]] if diag["best_epoch"] >= 0 and diag["val_losses"] else float("nan")
        gap = best_val - best_train if not np.isnan(best_train) and not np.isnan(best_val) else float("nan")

        log.info(f"  Fold {fold_info.fold} | {model_type} {loss_type} | "
                 f"epochs={n_epochs} best={diag['best_epoch']} | "
                 f"train@best={best_train:.6f} val@best={best_val:.6f} gap={gap:.6f}")

    return all_diagnostics


def main():
    all_diags = []

    for model_type in ["conditional", "structured"]:
        for loss in ["mse", "task"]:
            log.info(f"\n{'='*60}")
            log.info(f"Model={model_type}, loss={loss}")
            log.info(f"{'='*60}")
            diags = run_diagnostics(model_type=model_type, K=20, loss_type=loss)
            all_diags.extend(diags)

    out_path = Path("results/shrinkage_diagnostics.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"diagnostics": all_diags}, f, indent=2)
    log.info(f"\nDiagnostics saved to {out_path}")

    # Summary table
    print("\n=== Shrinkage Model Diagnostics Summary ===\n")
    print(f"{'Model':>12} {'Loss':>5} {'Params':>7} {'AvgBestEp':>10} {'TrainLoss':>10} {'ValLoss':>10} {'Gap':>10} {'EarlyStopped':>13}")
    for model_type in ["conditional", "structured"]:
        for loss in ["mse", "task"]:
            folds = [d for d in all_diags if d["model_type"] == model_type and d["loss"] == loss]
            if not folds:
                continue
            gaps, best_epochs, train_at_best, val_at_best = [], [], [], []
            early_stopped = 0
            for d in folds:
                be = d["best_epoch"]
                if be >= 0 and d["train_losses"] and d["val_losses"]:
                    train_at_best.append(d["train_losses"][min(be, len(d["train_losses"])-1)])
                    val_at_best.append(d["val_losses"][min(be, len(d["val_losses"])-1)])
                    gaps.append(val_at_best[-1] - train_at_best[-1])
                    best_epochs.append(be)
                    if be < len(d["train_losses"]) - 1:
                        early_stopped += 1

            avg_gap = np.mean(gaps) if gaps else float("nan")
            avg_be = np.mean(best_epochs) if best_epochs else float("nan")
            avg_tl = np.mean(train_at_best) if train_at_best else float("nan")
            avg_vl = np.mean(val_at_best) if val_at_best else float("nan")
            print(f"{model_type:>12} {loss:>5} {folds[0]['n_params']:>7} {avg_be:>10.1f} "
                  f"{avg_tl:>10.6f} {avg_vl:>10.6f} {avg_gap:>10.6f} {early_stopped:>9}/{len(folds)}")


if __name__ == "__main__":
    main()
