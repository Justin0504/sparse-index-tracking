"""Main entry point for sparse index tracking experiments.

Usage:
    python -m src.main                          # default config
    python -m src.main model.name=neural        # override model
    python -m src.main training.loss=mse        # two-stage baseline
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from src.data import SP500DataLoader, FeatureBuilder, RollingWindowSplitter, generate_synthetic_data
from src.models import FactorCovarianceModel, NeuralCovarianceModel
from src.optimization.qp_layer import SparseTrackingQP
from src.optimization.solver import solve_tracking_qp
from src.training import Trainer
from src.evaluation.metrics import (
    tracking_error,
    max_tracking_deviation,
    turnover,
    information_ratio,
    sparsity,
)
from src.utils import set_seed, get_logger

log = get_logger(__name__)


def _resolve_device(device: str) -> str:
    """Validate and resolve the compute device."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        log.warning("CUDA not available, falling back to CPU")
        return "cpu"
    if device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        log.warning("MPS not available, falling back to CPU")
        return "cpu"
    return device


def _get_rebalance_dates(
    dates: pd.DatetimeIndex, freq: str = "monthly"
) -> list[pd.Timestamp]:
    """Select rebalancing dates from a trading calendar."""
    if freq == "daily":
        return list(dates)
    elif freq == "weekly":
        return list(dates[dates.weekday == 4])  # Fridays
    elif freq == "monthly":
        s = pd.Series(range(len(dates)), index=dates)
        return list(s.resample("ME").last().dropna().index)
    else:
        raise ValueError(f"Unknown rebalance frequency: {freq}")


def _build_model(
    cfg: DictConfig, n_features: int, n_stocks: int
) -> torch.nn.Module:
    """Instantiate the covariance prediction model."""
    if cfg.model.name == "factor":
        return FactorCovarianceModel(
            n_features=n_features,
            n_stocks=n_stocks,
            n_factors=cfg.model.factor.n_factors,
        )
    elif cfg.model.name == "neural":
        return NeuralCovarianceModel(
            n_features=n_features,
            n_stocks=n_stocks,
            hidden_dims=list(cfg.model.neural.hidden_dims),
            dropout=cfg.model.neural.dropout,
            rank=cfg.model.neural.rank,
        )
    else:
        raise ValueError(f"Unknown model: {cfg.model.name}")


def _compute_index_weights(prices: pd.DataFrame) -> np.ndarray:
    """Market-cap-weighted index (proxy: use last available price as weight)."""
    last_prices = prices.iloc[-1].values
    w = last_prices / last_prices.sum()
    return w.astype(np.float32)


def run_experiment(cfg: DictConfig) -> dict:
    """Run a single experiment across all folds and sparsity levels."""
    set_seed(cfg.training.seed)
    device = _resolve_device(cfg.training.device)

    # ── Data ──
    if cfg.data.universe == "synthetic":
        log.info("Generating synthetic data...")
        prices, index_prices, tickers = generate_synthetic_data(
            n_stocks=cfg.data.synthetic.n_stocks,
            n_days=cfg.data.synthetic.n_days,
            n_factors=cfg.data.synthetic.n_factors,
            seed=cfg.training.seed,
        )
    else:
        log.info("Loading S&P 500 data...")
        loader = SP500DataLoader(
            start_date=cfg.data.start_date,
            end_date=cfg.data.end_date,
            top_n=cfg.data.top_n_by_mcap,
            cache_dir=cfg.data.cache_dir,
        )
        prices, index_prices, tickers = loader.load()
    n_stocks = len(tickers)

    # Validate loaded data
    assert len(prices) >= 252 * 3, f"Insufficient data: only {len(prices)} trading days"
    assert prices.shape[1] == n_stocks, f"Price columns ({prices.shape[1]}) != tickers ({n_stocks})"
    nan_count = prices.isna().sum().sum()
    if nan_count > 0:
        log.warning(f"Found {nan_count} NaN values in prices, forward-filling")
        prices = prices.ffill().bfill()

    log.info(f"Universe: {n_stocks} stocks, {len(prices)} trading days")

    # ── Features ──
    fb = FeatureBuilder(
        vol_window=cfg.features.vol_window,
        momentum_window=cfg.features.momentum_window,
        use_sector=cfg.features.use_sector,
        use_mcap_quintile=cfg.features.use_mcap_quintile,
    )
    returns = fb.compute_returns(prices)
    index_returns = fb.compute_index_returns(index_prices)

    # Determine feature dimension from a sample
    sample_idx = max(cfg.features.momentum_window, cfg.features.vol_window) + 1
    sample_date = returns.index[sample_idx]
    n_features = fb.build_features(returns, sample_date).shape[1]
    log.info(f"Feature dimension: {n_features}")

    # ── Rolling windows ──
    splitter = RollingWindowSplitter(
        train_years=cfg.rolling.train_years,
        val_months=cfg.rolling.val_months,
        test_years=cfg.rolling.test_years,
        step_years=cfg.rolling.step_years,
    )
    folds = splitter.split(returns.index)
    log.info(f"Rolling window folds: {len(folds)}")
    for f in folds:
        log.info(f"  {f}")

    min_offset = max(cfg.features.vol_window, cfg.features.momentum_window) + 1
    all_results = []

    for fold_info in folds:
        log.info(f"\n{'='*60}\n{fold_info}\n{'='*60}")

        # Split returns
        train_mask = (returns.index >= fold_info.train_start) & (returns.index <= fold_info.train_end)
        val_mask = (returns.index >= fold_info.val_start) & (returns.index <= fold_info.val_end)
        test_mask = (returns.index >= fold_info.test_start) & (returns.index <= fold_info.test_end)

        ret_train = returns.loc[train_mask]
        ret_val = returns.loc[val_mask]
        ret_test = returns.loc[test_mask]

        # Per-fold index weights (use training period prices, aligned to returns index)
        train_prices = prices.loc[prices.index.isin(ret_train.index)]
        w_index = _compute_index_weights(train_prices)

        # Rebalancing dates (filter all sets for sufficient lookback + must exist in returns)
        def _filter_rebal(dates_list: list) -> list:
            out = []
            for d in dates_list:
                if d not in returns.index:
                    continue
                if returns.index.get_loc(d) >= min_offset:
                    out.append(d)
            return out

        rebal_train = _filter_rebal(_get_rebalance_dates(ret_train.index, cfg.optimization.rebalance_freq))
        rebal_val = _filter_rebal(_get_rebalance_dates(ret_val.index, cfg.optimization.rebalance_freq))
        rebal_test = _filter_rebal(_get_rebalance_dates(ret_test.index, cfg.optimization.rebalance_freq))

        if not rebal_train or not rebal_test:
            log.warning(f"Fold {fold_info.fold}: insufficient rebalancing dates, skipping")
            continue

        for K in cfg.optimization.sparsity_levels:
            log.info(f"\n--- Sparsity K={K} ---")

            model = _build_model(cfg, n_features, n_stocks)

            trainer = Trainer(
                model=model,
                feature_builder=fb,
                loss_type=cfg.training.loss,
                lr=cfg.training.lr,
                weight_decay=cfg.training.weight_decay,
                epochs=cfg.training.epochs,
                patience=cfg.training.patience,
                device=device,
                sparsity_K=K,
                turnover_penalty=cfg.training.get("turnover_penalty", 0.0),
                selection=cfg.optimization.get("selection", "market_cap"),
            )

            history = trainer.train(
                returns_train=ret_train.values,
                returns_val=ret_val.values,
                rebalance_dates_train=rebal_train,
                rebalance_dates_val=rebal_val,
                returns_df_train=returns,
                returns_df_val=returns,
                w_index=w_index,
            )

            # ── Test ──
            log.info("Evaluating on test set...")
            model.eval()

            portfolio_rets = []
            index_rets = []
            weights_history = []

            for i, date in enumerate(rebal_test):
                loc = returns.index.get_loc(date)
                next_loc = (returns.index.get_loc(rebal_test[i + 1])
                            if i + 1 < len(rebal_test) else len(returns))

                if next_loc <= loc:
                    continue

                features_np = fb.build_features(returns, date)
                with torch.no_grad():
                    features_t = torch.tensor(features_np, dtype=torch.float32, device=device)
                    cov_pred = model(features_t).cpu().numpy()

                try:
                    w_opt = solve_tracking_qp(cov_pred, w_index, K=K, selection=cfg.optimization.get("selection", "market_cap"))
                except Exception as e:
                    log.warning(f"Test QP failed at {date}: {e}, using equal weight")
                    w_opt = np.ones(n_stocks, dtype=np.float64) / n_stocks

                weights_history.append(w_opt)

                holding_rets = returns.iloc[loc:next_loc].values
                port_r = holding_rets @ w_opt
                idx_r = holding_rets @ w_index
                portfolio_rets.extend(port_r.tolist())
                index_rets.extend(idx_r.tolist())

            if not portfolio_rets:
                log.warning(f"Fold {fold_info.fold}, K={K}: no test returns, skipping")
                continue

            portfolio_rets = np.array(portfolio_rets)
            index_rets = np.array(index_rets)

            te = tracking_error(portfolio_rets, index_rets, cfg.evaluation.annualization_factor)
            mtd = max_tracking_deviation(portfolio_rets, index_rets)
            ir = information_ratio(portfolio_rets, index_rets, cfg.evaluation.annualization_factor)
            avg_sparsity = float(np.mean([sparsity(w) for w in weights_history]))
            avg_turnover = (float(np.mean([
                turnover(weights_history[j], weights_history[j + 1])
                for j in range(len(weights_history) - 1)
            ])) if len(weights_history) > 1 else 0.0)

            result = {
                "fold": fold_info.fold,
                "K": K,
                "model": cfg.model.name,
                "loss": cfg.training.loss,
                "tracking_error": te,
                "max_tracking_deviation": mtd,
                "information_ratio": ir,
                "avg_sparsity": avg_sparsity,
                "avg_turnover": avg_turnover,
                "test_start": str(fold_info.test_start.date()),
                "test_end": str(fold_info.test_end.date()),
                "portfolio_returns": portfolio_rets.tolist(),
                "index_returns": index_rets.tolist(),
            }
            all_results.append(result)
            log.info(
                f"K={K} | TE={te:.4f} | MTD={mtd:.4f} | IR={ir:.4f} | "
                f"Sparsity={avg_sparsity:.0f} | Turnover={avg_turnover:.4f}"
            )

    return {"results": all_results, "config": OmegaConf.to_container(cfg)}


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    output = run_experiment(cfg)

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    lam = cfg.training.get("turnover_penalty", 0.0)
    lam_str = f"_lambda{lam}" if lam != 0.1 else ""
    out_path = out_dir / f"{cfg.model.name}_{cfg.training.loss}{lam_str}_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
