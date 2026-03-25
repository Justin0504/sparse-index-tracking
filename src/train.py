"""Training routines for two-stage MSE baseline and end-to-end DFL."""

from __future__ import annotations

import copy
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .losses import mse_prediction_loss, tracking_error_loss
from .models import DFLModel, TwoStageModel


def train_two_stage(
    model: TwoStageModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 50,
    lr: float = 1e-3,
    device: torch.device | None = None,
    verbose: bool = True,
) -> TwoStageModel:
    """Train the two-stage MSE baseline.

    Stage 1: optimise the predictor with ``MSE(ĉ, σ_rb)``.
    The QP layer is not used during training.

    Args:
        model: The ``TwoStageModel`` to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        n_epochs: Number of training epochs.
        lr: Adam learning rate.
        device: Compute device (``None`` → CPU).
        verbose: Print per-epoch progress.

    Returns:
        Best (lowest val-loss) copy of the model.
    """
    if device is None:
        device = torch.device("cpu")
    model = model.to(device)
    optimiser = torch.optim.Adam(model.predictor.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_predictor_state = copy.deepcopy(model.predictor.state_dict())

    for epoch in range(n_epochs):
        # ---- training ----
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            features, _, _, _, sigma_rb = _to_device(batch, device)
            c_hat = model.predict(features)
            loss = mse_prediction_loss(c_hat, sigma_rb)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            train_loss += loss.item()

        # ---- validation ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                features, _, _, _, sigma_rb = _to_device(batch, device)
                c_hat = model.predict(features)
                val_loss += mse_prediction_loss(c_hat, sigma_rb).item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
            print(
                f"[TwoStage] epoch {epoch+1:>3}/{n_epochs} "
                f"train_mse={avg_train:.6f}  val_mse={avg_val:.6f}"
            )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_predictor_state = copy.deepcopy(model.predictor.state_dict())

    model.predictor.load_state_dict(best_predictor_state)
    return model


def train_dfl(
    model: DFLModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 30,
    lr: float = 1e-3,
    device: torch.device | None = None,
    verbose: bool = True,
    warmstart_state: Optional[Dict] = None,
) -> DFLModel:
    """Train the end-to-end Decision-Focused Learning model.

    The loss is the next-period portfolio tracking error.
    Gradients flow from the loss through the differentiable QP layer back
    into the predictor network.

    Args:
        model: The ``DFLModel`` to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        n_epochs: Number of training epochs.
        lr: Adam learning rate.
        device: Compute device (``None`` → CPU).
        verbose: Print per-epoch progress.
        warmstart_state: Optional ``state_dict`` used to initialise the
            predictor (e.g. from a pre-trained two-stage model).

    Returns:
        Best (lowest val tracking error) copy of the model.
    """
    if device is None:
        device = torch.device("cpu")
    model = model.to(device)

    if warmstart_state is not None:
        # Only load predictor weights for warm-start
        predictor_state = {
            k.replace("predictor.", ""): v
            for k, v in warmstart_state.items()
            if k.startswith("predictor.")
        }
        model.predictor.load_state_dict(predictor_state)

    optimiser = torch.optim.Adam(model.predictor.parameters(), lr=lr)

    best_val_te = float("inf")
    best_predictor_state = copy.deepcopy(model.predictor.state_dict())

    for epoch in range(n_epochs):
        # ---- training ----
        model.train()
        train_te = 0.0
        for batch in train_loader:
            features, target_return, target_index, R_scaled, _ = _to_device(
                batch, device
            )
            weights = model(features, R_scaled)
            loss = tracking_error_loss(weights, target_return, target_index)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            train_te += loss.item()

        # ---- validation ----
        model.eval()
        val_te = 0.0
        for batch in val_loader:
            features, target_return, target_index, R_scaled, _ = _to_device(
                batch, device
            )
            with torch.no_grad():
                # Use predictor only; avoid backprop overhead in validation
                c_hat = model.predictor(features)
            weights = model.qp_layer(c_hat, R_scaled)
            val_te += tracking_error_loss(weights, target_return, target_index).item()

        avg_train = train_te / len(train_loader)
        avg_val = val_te / len(val_loader)
        if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
            print(
                f"[DFL]      epoch {epoch+1:>3}/{n_epochs} "
                f"train_te={avg_train:.6f}  val_te={avg_val:.6f}"
            )

        if avg_val < best_val_te:
            best_val_te = avg_val
            best_predictor_state = copy.deepcopy(model.predictor.state_dict())

    model.predictor.load_state_dict(best_predictor_state)
    return model


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_device(batch: tuple, device: torch.device) -> tuple:
    """Move all tensors in a batch tuple to *device*."""
    return tuple(t.to(device) for t in batch)
