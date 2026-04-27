"""Training loop for covariance models with MSE or task (DFL) loss."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from src.data.features import FeatureBuilder
from src.models.shrinkage_model import (
    ShrinkageCovarianceModel,
    ConditionalShrinkageModel,
    StructuredShrinkageModel,
)
from src.optimization.qp_layer import SparseTrackingQP, L1TrackingQP
from src.training.losses import mse_loss, task_loss
from src.utils.logging import get_logger

log = get_logger(__name__)


class Trainer:
    """Train a covariance model using either MSE or task (DFL) loss.

    Supports gradient accumulation across rebalancing dates within each epoch,
    optional turnover regularization, multi-step lookahead, and curriculum
    learning (MSE → task loss transition).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        feature_builder: FeatureBuilder,
        loss_type: str = "task",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 50,
        patience: int = 10,
        device: str = "cpu",
        sparsity_K: int = 20,
        grad_accum_steps: int = 4,
        turnover_penalty: float = 0.0,
        selection: str = "market_cap",
        qp_formulation: str = "hard_k",
        l1_lambda: float = 0.01,
        lookahead_steps: int = 1,
        curriculum_epochs: int = 0,
    ):
        self.model = model.to(device)
        self.feature_builder = feature_builder
        self.loss_type = loss_type
        self.epochs = epochs
        self.patience = patience
        self.device = device
        self.sparsity_K = sparsity_K
        self.grad_accum_steps = grad_accum_steps
        self.turnover_penalty = turnover_penalty
        self.selection = selection
        self.qp_formulation = qp_formulation
        self.l1_lambda = l1_lambda
        self.tc_cost = 0.0  # transaction cost in QP objective
        self.lookahead_steps = lookahead_steps  # multi-step lookahead (1 = single-step)
        self.curriculum_epochs = curriculum_epochs  # epochs of pure MSE before transitioning

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=lr * 0.01
        )
        self._qp_layer: SparseTrackingQP | None = None
        self._l1_qp_layer: L1TrackingQP | None = None

    def _get_qp_layer(self, n_stocks: int) -> SparseTrackingQP:
        if self._qp_layer is None or self._qp_layer.n_stocks != n_stocks:
            self._qp_layer = SparseTrackingQP(n_stocks)
        return self._qp_layer

    def _get_l1_qp_layer(self, n_stocks: int) -> L1TrackingQP:
        if self._l1_qp_layer is None or self._l1_qp_layer.n_stocks != n_stocks:
            self._l1_qp_layer = L1TrackingQP(n_stocks)
        return self._l1_qp_layer

    def _effective_loss_type(self, epoch: int) -> tuple[str, float]:
        """Determine loss type and task weight for curriculum learning.

        Returns (loss_type, task_weight) where:
        - During pure MSE phase (epoch < curriculum_epochs): ("mse", 0.0)
        - During transition (curriculum_epochs <= epoch < 2*curriculum_epochs):
          ("curriculum", linearly increasing task_weight)
        - After transition or no curriculum: (self.loss_type, 1.0)
        """
        if self.curriculum_epochs <= 0 or self.loss_type == "mse":
            return self.loss_type, 1.0
        if epoch < self.curriculum_epochs:
            return "mse", 0.0
        transition_epoch = epoch - self.curriculum_epochs
        if transition_epoch < self.curriculum_epochs:
            task_weight = (transition_epoch + 1) / self.curriculum_epochs
            return "curriculum", task_weight
        return "task", 1.0

    def train(
        self,
        returns_train: np.ndarray,
        returns_val: np.ndarray,
        rebalance_dates_train: list,
        rebalance_dates_val: list,
        returns_df_train,
        returns_df_val,
        w_index: np.ndarray | None = None,
        lookback: int = 63,
        forward_window: int = 21,
    ) -> dict:
        """Run the training loop with gradient accumulation."""
        n_stocks = returns_train.shape[1]
        if w_index is None:
            w_index = np.ones(n_stocks, dtype=np.float32) / n_stocks

        w_idx_t = torch.tensor(w_index, dtype=torch.float32, device=self.device)

        history = {"train_losses": [], "val_losses": []}
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            eff_loss_type, task_weight = self._effective_loss_type(epoch)

            # ── Train with gradient accumulation ──
            self.model.train()
            self.optimizer.zero_grad()
            epoch_losses = []
            accum_count = 0
            qp_failures = 0
            w_prev = None  # Track previous weights for turnover penalty

            # Multi-step lookahead: group consecutive rebalancing dates
            if self.lookahead_steps > 1 and eff_loss_type in ("task", "curriculum"):
                groups = [
                    rebalance_dates_train[i:i + self.lookahead_steps]
                    for i in range(0, len(rebalance_dates_train), self.lookahead_steps)
                ]
            else:
                groups = [[d] for d in rebalance_dates_train]

            for group in groups:
                if len(group) > 1:
                    loss_val, w_opt = self._multi_step(
                        returns_df_train, group, w_idx_t,
                        lookback, forward_window, train=True,
                        accum_scale=min(self.grad_accum_steps, len(groups)),
                        w_prev=w_prev,
                        eff_loss_type=eff_loss_type,
                        task_weight=task_weight,
                    )
                else:
                    loss_val, w_opt = self._step(
                        returns_df_train, group[0], w_idx_t,
                        lookback, forward_window, train=True,
                        accum_scale=min(self.grad_accum_steps, len(groups)),
                        w_prev=w_prev,
                        eff_loss_type=eff_loss_type,
                        task_weight=task_weight,
                    )
                if loss_val is None:
                    qp_failures += 1
                    continue

                if w_opt is not None:
                    w_prev = w_opt.detach()

                epoch_losses.append(loss_val)
                accum_count += 1

                if accum_count % self.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # Flush remaining gradients
            if accum_count % self.grad_accum_steps != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

            train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            history["train_losses"].append(train_loss)

            if qp_failures > 0:
                log.warning(f"Epoch {epoch + 1}: {qp_failures} QP solve failures in training")

            # ── Validate ──
            self.model.eval()
            val_losses = []
            w_prev_val = None
            with torch.no_grad():
                for date in rebalance_dates_val:
                    loss_val, w_opt = self._step(
                        returns_df_val, date, w_idx_t,
                        lookback, forward_window, train=False,
                        w_prev=w_prev_val,
                    )
                    if loss_val is not None:
                        val_losses.append(loss_val)
                    if w_opt is not None:
                        w_prev_val = w_opt.detach()

            val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
            history["val_losses"].append(val_loss)

            self.scheduler.step()

            phase_str = ""
            if self.curriculum_epochs > 0:
                phase_str = f" [{eff_loss_type}(tw={task_weight:.2f})]"
            if self.lookahead_steps > 1 and eff_loss_type in ("task", "curriculum"):
                phase_str += f" [lookahead={self.lookahead_steps}]"
            log.info(
                f"Epoch {epoch + 1}/{self.epochs}{phase_str} — "
                f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
                f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
            )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    log.info(f"Early stopping at epoch {epoch + 1}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        return history

    def _step(
        self,
        returns_df,
        date,
        w_idx_t: torch.Tensor,
        lookback: int,
        forward_window: int,
        train: bool,
        accum_scale: int = 1,
        w_prev: torch.Tensor | None = None,
        eff_loss_type: str | None = None,
        task_weight: float = 1.0,
    ) -> tuple[float | None, torch.Tensor | None]:
        """Single rebalancing-date step. Returns (loss_value, w_opt).

        eff_loss_type overrides self.loss_type when set (used by curriculum).
        For "curriculum" mode, blends (1-task_weight)*MSE + task_weight*task.
        """
        loss_type = eff_loss_type if eff_loss_type is not None else self.loss_type
        loc = returns_df.index.get_loc(date)
        n_stocks = returns_df.shape[1]

        features_np = self.feature_builder.build_features(returns_df, date)
        features_t = torch.tensor(features_np, dtype=torch.float32, device=self.device)

        # For shrinkage-family models, set sample covariance before forward pass
        if isinstance(self.model, (ShrinkageCovarianceModel, ConditionalShrinkageModel, StructuredShrinkageModel)):
            sample_cov_np = self.feature_builder.compute_realized_covariance(
                returns_df, date, lookback
            )
            sample_cov_t = torch.tensor(sample_cov_np, dtype=torch.float32, device=self.device)
            self.model.set_sample_covariance(sample_cov_t)

        # For conditional shrinkage, set market features
        if isinstance(self.model, ConditionalShrinkageModel):
            mkt_feat_np = self.feature_builder.build_market_features(returns_df, date, lookback)
            mkt_feat_t = torch.tensor(mkt_feat_np, dtype=torch.float32, device=self.device)
            self.model.set_market_features(mkt_feat_t)

        cov_pred = self.model(features_t)

        if loss_type == "mse":
            cov_target_np = self.feature_builder.compute_realized_covariance(
                returns_df, date, lookback
            )
            cov_target = torch.tensor(cov_target_np, dtype=torch.float32, device=self.device)
            loss = mse_loss(cov_pred, cov_target)

            if train:
                scaled_loss = loss / accum_scale
                scaled_loss.backward()
            return float(loss.detach().cpu()), None

        # Task loss (DFL) or curriculum blend
        if loc + forward_window > len(returns_df):
            return None, None

        try:
            if self.qp_formulation == "l1":
                l1_qp = self._get_l1_qp_layer(n_stocks)
                w_opt = l1_qp.solve(cov_pred, w_idx_t, lam=self.l1_lambda, hard=False)
            else:
                qp = self._get_qp_layer(n_stocks)
                cov_qp = 252.0 * cov_pred if self.tc_cost > 0 else cov_pred
                w_opt = qp.solve(cov_qp, w_idx_t, K=self.sparsity_K, hard=False, selection=self.selection, w_prev=w_prev, tc_cost=self.tc_cost)
        except Exception as e:
            log.debug(f"QP solve failed at {date}: {e}")
            return None, None

        future_ret_np = returns_df.iloc[loc: loc + forward_window].values
        future_ret_t = torch.tensor(
            future_ret_np, dtype=torch.float32, device=self.device
        )
        t_loss = task_loss(
            w_opt, w_idx_t, future_ret_t,
            w_prev=w_prev,
            turnover_penalty=self.turnover_penalty,
        )

        # Curriculum blending: (1-task_weight)*MSE + task_weight*task
        if loss_type == "curriculum":
            cov_target_np = self.feature_builder.compute_realized_covariance(
                returns_df, date, lookback
            )
            cov_target = torch.tensor(cov_target_np, dtype=torch.float32, device=self.device)
            m_loss = mse_loss(cov_pred, cov_target)
            loss = (1 - task_weight) * m_loss + task_weight * t_loss
        else:
            loss = t_loss

        if train:
            scaled_loss = loss / accum_scale
            scaled_loss.backward()

        return float(loss.detach().cpu()), w_opt

    def _multi_step(
        self,
        returns_df,
        dates: list,
        w_idx_t: torch.Tensor,
        lookback: int,
        forward_window: int,
        train: bool,
        accum_scale: int = 1,
        w_prev: torch.Tensor | None = None,
        eff_loss_type: str | None = None,
        task_weight: float = 1.0,
    ) -> tuple[float | None, torch.Tensor | None]:
        """Multi-step lookahead: chain QP solves across consecutive dates.

        Accumulates losses across the trajectory and backprops through the
        entire chain, allowing the model to learn policies that account for
        future rebalancing costs.
        """
        loss_type = eff_loss_type if eff_loss_type is not None else self.loss_type
        total_loss = torch.tensor(0.0, device=self.device)
        n_valid = 0
        w_current = w_prev
        last_w_opt = None

        for date in dates:
            loc = returns_df.index.get_loc(date)
            n_stocks = returns_df.shape[1]

            if loc + forward_window > len(returns_df):
                break

            features_np = self.feature_builder.build_features(returns_df, date)
            features_t = torch.tensor(features_np, dtype=torch.float32, device=self.device)

            if isinstance(self.model, (ShrinkageCovarianceModel, ConditionalShrinkageModel, StructuredShrinkageModel)):
                sample_cov_np = self.feature_builder.compute_realized_covariance(
                    returns_df, date, lookback
                )
                sample_cov_t = torch.tensor(sample_cov_np, dtype=torch.float32, device=self.device)
                self.model.set_sample_covariance(sample_cov_t)

            if isinstance(self.model, ConditionalShrinkageModel):
                mkt_feat_np = self.feature_builder.build_market_features(returns_df, date, lookback)
                mkt_feat_t = torch.tensor(mkt_feat_np, dtype=torch.float32, device=self.device)
                self.model.set_market_features(mkt_feat_t)

            cov_pred = self.model(features_t)

            try:
                if self.qp_formulation == "l1":
                    l1_qp = self._get_l1_qp_layer(n_stocks)
                    w_opt = l1_qp.solve(cov_pred, w_idx_t, lam=self.l1_lambda, hard=False)
                else:
                    qp = self._get_qp_layer(n_stocks)
                    cov_qp = 252.0 * cov_pred if self.tc_cost > 0 else cov_pred
                    w_opt = qp.solve(cov_qp, w_idx_t, K=self.sparsity_K, hard=False,
                                     selection=self.selection, w_prev=w_current,
                                     tc_cost=self.tc_cost)
            except Exception as e:
                log.debug(f"Multi-step QP failed at {date}: {e}")
                break

            future_ret_np = returns_df.iloc[loc: loc + forward_window].values
            future_ret_t = torch.tensor(
                future_ret_np, dtype=torch.float32, device=self.device
            )
            step_task_loss = task_loss(
                w_opt, w_idx_t, future_ret_t,
                w_prev=w_current,
                turnover_penalty=self.turnover_penalty,
            )

            if loss_type == "curriculum":
                cov_target_np = self.feature_builder.compute_realized_covariance(
                    returns_df, date, lookback
                )
                cov_target = torch.tensor(cov_target_np, dtype=torch.float32, device=self.device)
                m_loss = mse_loss(cov_pred, cov_target)
                step_loss = (1 - task_weight) * m_loss + task_weight * step_task_loss
            else:
                step_loss = step_task_loss

            total_loss = total_loss + step_loss
            n_valid += 1
            w_current = w_opt  # chain: pass w_opt as w_prev to next step
            last_w_opt = w_opt

        if n_valid == 0:
            return None, None

        avg_loss = total_loss / n_valid

        if train:
            scaled_loss = avg_loss / accum_scale
            scaled_loss.backward()

        return float(avg_loss.detach().cpu()), last_w_opt

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        log.info(f"Model saved to {path}")

    def load(self, path: str | Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        log.info(f"Model loaded from {path}")
