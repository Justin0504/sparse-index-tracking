#!/usr/bin/env python3
"""EPA air quality monitoring sparse selection experiment — cross-domain validation.

Uses real PM2.5 data from EPA AQS.
Higher spatial heterogeneity than temperature → DFL advantage may be stronger.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.epa_loader import EPAAirQualityLoader
from src.optimization.qp_layer import SparseTrackingQP
from src.optimization.solver import solve_tracking_qp, naive_index_weights
from src.training.losses import mse_loss, task_loss
from src.utils import set_seed, get_logger

log = get_logger(__name__)


class SpatialStructuredModel(nn.Module):
    """Structured shrinkage with per-region target variances."""

    def __init__(self, N: int, region_ids: np.ndarray, n_regions: int, init_alpha: float = 0.5):
        super().__init__()
        self.N = N
        self.n_regions = n_regions
        self._logit_alpha = nn.Parameter(
            torch.tensor(np.log(init_alpha / (1 - init_alpha)), dtype=torch.float32)
        )
        self._log_region_var = nn.Parameter(torch.zeros(n_regions, dtype=torch.float32))
        self.register_buffer("_region_ids", torch.tensor(region_ids, dtype=torch.long))
        self.register_buffer("_sample_cov", torch.eye(N, dtype=torch.float32))

    @property
    def alpha(self):
        return torch.sigmoid(self._logit_alpha)

    def set_sample_covariance(self, cov: torch.Tensor):
        self._sample_cov = cov

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        S = self._sample_cov
        mu = torch.trace(S) / self.N
        alpha = self.alpha
        region_vars = mu * nn.functional.softplus(self._log_region_var)
        diag_values = region_vars[self._region_ids]
        T = torch.diag(diag_values)
        return (1 - alpha) * S + alpha * T


def compute_sample_cov(data: np.ndarray) -> np.ndarray:
    T = data.shape[0]
    centered = data - data.mean(axis=0)
    return (centered.T @ centered / (T - 1)).astype(np.float32)


def train_model(data_train, data_val, N, K, region_ids, n_regions,
                loss_type, epochs=60, lr=5e-3, patience=12):
    w_idx = np.ones(N, dtype=np.float32) / N
    w_idx_t = torch.tensor(w_idx, dtype=torch.float32)

    model = SpatialStructuredModel(N, region_ids, n_regions)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    qp = SparseTrackingQP(N)

    T_train = data_train.shape[0]
    window = min(63, T_train // 4)
    forward = min(21, T_train // 6)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        epoch_losses = []

        for start in range(0, T_train - window - forward, window // 2):
            hist = data_train[start:start + window]
            fut = data_train[start + window:start + window + forward]
            if len(fut) < 5:
                continue

            sample_cov_t = torch.tensor(compute_sample_cov(hist))
            realized_cov_t = torch.tensor(compute_sample_cov(fut))
            fut_t = torch.tensor(fut, dtype=torch.float32)
            features_dummy = torch.zeros(N, 1)

            optimizer.zero_grad()
            model.set_sample_covariance(sample_cov_t)
            cov_pred = model(features_dummy)

            if loss_type == "mse":
                loss = mse_loss(cov_pred, realized_cov_t)
            else:
                try:
                    w_opt = qp.solve(cov_pred, w_idx_t, K, selection="equal")
                    loss = task_loss(w_opt, w_idx_t, fut_t, annualization_factor=1)
                except Exception:
                    loss = mse_loss(cov_pred, realized_cov_t)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        if not epoch_losses:
            continue

        model.eval()
        val_losses = []
        T_val = data_val.shape[0]
        with torch.no_grad():
            for start in range(0, T_val - window - forward, window):
                hist = data_val[start:start + window]
                fut = data_val[start + window:start + window + forward]
                if len(fut) < 5:
                    continue
                sample_cov_t = torch.tensor(compute_sample_cov(hist))
                features_dummy = torch.zeros(N, 1)
                model.set_sample_covariance(sample_cov_t)
                cov_pred_np = model(features_dummy).numpy()
                try:
                    w_opt_np = solve_tracking_qp(cov_pred_np, w_idx, K=K, selection="equal")
                except Exception:
                    w_opt_np = naive_index_weights(w_idx, K)
                port_ret = fut @ w_opt_np
                idx_ret = fut @ w_idx
                val_losses.append(np.std(port_ret - idx_ret))

        val_loss = np.mean(val_losses) if val_losses else float("inf")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate_model(model, data_test, N, K):
    w_idx = np.ones(N, dtype=np.float32) / N
    T_test = data_test.shape[0]
    window = min(63, T_test // 3)

    all_port_rets, all_idx_rets = [], []
    weights_history = []

    model.eval()
    with torch.no_grad():
        for start in range(0, T_test - window, window):
            hist = data_test[max(0, start):start + window]
            fut = data_test[start + window:min(start + 2 * window, T_test)]
            if len(fut) < 5 or len(hist) < 10:
                continue
            sample_cov_t = torch.tensor(compute_sample_cov(hist))
            features_dummy = torch.zeros(N, 1)
            model.set_sample_covariance(sample_cov_t)
            cov_pred_np = model(features_dummy).numpy()
            try:
                w_opt = solve_tracking_qp(cov_pred_np, w_idx, K=K, selection="equal")
            except Exception:
                w_opt = naive_index_weights(w_idx, K)
            weights_history.append(w_opt)
            all_port_rets.extend((fut @ w_opt).tolist())
            all_idx_rets.extend((fut @ w_idx).tolist())

    if not all_port_rets:
        return None

    from src.evaluation.metrics import tracking_error, turnover
    te = tracking_error(np.array(all_port_rets), np.array(all_idx_rets), annualization=365)
    avg_to = float(np.mean([turnover(weights_history[j], weights_history[j + 1])
                            for j in range(len(weights_history) - 1)])) if len(weights_history) > 1 else 0.0
    return {"tracking_error": te, "avg_turnover": avg_to}


def run_poet_baseline(data_test, N, K):
    w_idx = np.ones(N, dtype=np.float32) / N
    T_test = data_test.shape[0]
    window = min(63, T_test // 3)

    all_port_rets, all_idx_rets = [], []
    weights_history = []

    for start in range(0, T_test - window, window):
        hist = data_test[max(0, start):start + window]
        fut = data_test[start + window:min(start + 2 * window, T_test)]
        if len(fut) < 5 or len(hist) < 10:
            continue
        cov = compute_sample_cov(hist)
        try:
            w_opt = solve_tracking_qp(cov, w_idx, K=K, selection="equal")
        except Exception:
            w_opt = naive_index_weights(w_idx, K)
        weights_history.append(w_opt)
        all_port_rets.extend((fut @ w_opt).tolist())
        all_idx_rets.extend((fut @ w_idx).tolist())

    if not all_port_rets:
        return None

    from src.evaluation.metrics import tracking_error, turnover
    te = tracking_error(np.array(all_port_rets), np.array(all_idx_rets), annualization=365)
    avg_to = float(np.mean([turnover(weights_history[j], weights_history[j + 1])
                            for j in range(len(weights_history) - 1)])) if len(weights_history) > 1 else 0.0
    return {"tracking_error": te, "avg_turnover": avg_to}


def main():
    set_seed(42)
    all_results = []

    for N in [200, 500, 800]:
        log.info(f"\n{'='*60}\nLoading N={N} EPA PM2.5 stations...")
        loader = EPAAirQualityLoader(n_stations=N)
        anomalies, field_avg, station_ids, metadata = loader.load()
        log.info(f"  {len(station_ids)} stations, {len(anomalies)} days")

        data = anomalies.values.astype(np.float32)
        T_total = len(data)
        T_train = int(T_total * 0.6)
        T_val = int(T_total * 0.2)

        data_train = data[:T_train]
        data_val = data[T_train:T_train + T_val]
        data_test = data[T_train + T_val:]

        n_regions = 9
        grid_size = 3
        if metadata is not None and len(metadata) > 0:
            region_ids = loader.assign_regions(metadata, grid_size=grid_size)
            if len(region_ids) != N:
                region_ids = np.random.RandomState(42).randint(0, n_regions, N)
        else:
            region_ids = np.random.RandomState(42).randint(0, n_regions, N)

        K_values = [10, 20, 50]
        K_values = [k for k in K_values if k <= N // 2]

        for K in K_values:
            kn = K / N
            alignment = np.sqrt(2 * kn)
            log.info(f"\n  N={N}, K={K}, K/N={kn:.3f}, alignment={alignment:.3f}")

            log.info(f"    POET baseline...")
            poet_res = run_poet_baseline(data_test, N, K)
            if poet_res:
                result = {
                    "N": N, "K": K, "model": "POET", "loss": "N/A",
                    "alignment": round(alignment, 4),
                    **poet_res,
                }
                all_results.append(result)
                log.info(f"    POET TE={poet_res['tracking_error']:.4f}")

            for loss_type in ["mse", "task"]:
                log.info(f"    Structured {loss_type}...")
                model = train_model(data_train, data_val, N, K, region_ids, n_regions, loss_type)
                res = evaluate_model(model, data_test, N, K)
                if res:
                    result = {
                        "N": N, "K": K, "model": "spatial_structured", "loss": loss_type,
                        "alignment": round(alignment, 4),
                        **res,
                    }
                    all_results.append(result)
                    log.info(f"    {loss_type} TE={res['tracking_error']:.4f}")

    out_dir = Path("results/epa")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "epa_results.json"
    with open(out_path, "w") as f:
        json.dump({"results": all_results}, f, indent=2)
    log.info(f"\nSaved {len(all_results)} results to {out_path}")

    # Summary
    log.info(f"\n{'='*60}\nSUMMARY")
    log.info(f"{'N':>5} {'K':>5} {'K/N':>6} {'MSE':>8} {'DFL':>8} {'POET':>8} {'Δ%':>6}")
    log.info("-" * 50)

    configs = sorted(set((r["N"], r["K"]) for r in all_results))
    for N, K in configs:
        mse = [r["tracking_error"] for r in all_results if r["N"] == N and r["K"] == K and r["loss"] == "mse"]
        task = [r["tracking_error"] for r in all_results if r["N"] == N and r["K"] == K and r["loss"] == "task"]
        poet = [r["tracking_error"] for r in all_results if r["N"] == N and r["K"] == K and r["model"] == "POET"]
        mse_m = np.mean(mse) if mse else float("nan")
        task_m = np.mean(task) if task else float("nan")
        poet_m = np.mean(poet) if poet else float("nan")
        delta = (mse_m - task_m) / mse_m * 100 if mse and task and mse_m > 0 else float("nan")
        log.info(f"{N:>5} {K:>5} {K/N:>6.3f} {mse_m:>8.4f} {task_m:>8.4f} {poet_m:>8.4f} {delta:>+6.1f}")


if __name__ == "__main__":
    main()
