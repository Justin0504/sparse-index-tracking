"""Feature engineering for covariance prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureBuilder:
    """Builds per-stock features at each rebalancing date.

    Features:
        1. Trailing realized volatility (vol_window days)
        2. Price momentum (momentum_window days)
        3. GICS sector one-hot (optional)
        4. Market-cap quintile (optional)
    """

    def __init__(
        self,
        vol_window: int = 21,
        momentum_window: int = 63,
        use_sector: bool = True,
        use_mcap_quintile: bool = True,
    ):
        self.vol_window = vol_window
        self.momentum_window = momentum_window
        self.use_sector = use_sector
        self.use_mcap_quintile = use_mcap_quintile

    def compute_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute simple daily returns."""
        return prices.pct_change().iloc[1:]

    def compute_index_returns(self, index_prices: pd.Series) -> pd.Series:
        """Compute simple daily index returns."""
        return index_prices.pct_change().iloc[1:]

    def build_features(
        self,
        returns: pd.DataFrame,
        date: pd.Timestamp,
    ) -> np.ndarray:
        """Build feature vector for all stocks at a given rebalancing date.

        Parameters
        ----------
        returns : DataFrame (dates × N stocks)
        date : rebalancing date

        Returns
        -------
        features : ndarray of shape (N, n_features)
        """
        loc = returns.index.get_loc(date)
        n_stocks = returns.shape[1]
        feature_list = []

        # 1. Trailing realized volatility
        if loc >= self.vol_window:
            window = returns.iloc[loc - self.vol_window : loc]
            vol = window.std().values  # (N,)
        else:
            vol = np.zeros(n_stocks)
        feature_list.append(vol.reshape(-1, 1))

        # 2. Price momentum (cumulative return over momentum_window)
        if loc >= self.momentum_window:
            window = returns.iloc[loc - self.momentum_window : loc]
            mom = (1 + window).prod().values - 1  # (N,)
        else:
            mom = np.zeros(n_stocks)
        feature_list.append(mom.reshape(-1, 1))

        # 3. Sector one-hot (placeholder: use stock index as proxy)
        if self.use_sector:
            n_sectors = 11  # GICS sectors
            sector_ids = np.arange(n_stocks) % n_sectors
            one_hot = np.eye(n_sectors)[sector_ids]  # (N, 11)
            feature_list.append(one_hot)

        # 4. Market-cap quintile (proxy: average dollar volume rank)
        if self.use_mcap_quintile:
            if loc >= self.vol_window:
                avg_vol = returns.iloc[loc - self.vol_window : loc].abs().mean().values
                quintiles = pd.qcut(avg_vol, 5, labels=False, duplicates="drop")
                q_one_hot = np.eye(5)[quintiles.astype(int)]  # (N, 5)
            else:
                q_one_hot = np.zeros((n_stocks, 5))
            feature_list.append(q_one_hot)

        return np.hstack(feature_list).astype(np.float32)

    def compute_realized_covariance(
        self,
        returns: pd.DataFrame,
        date: pd.Timestamp,
        lookback: int = 63,
    ) -> np.ndarray:
        """Compute realized covariance matrix over a lookback window.

        Returns
        -------
        cov : ndarray of shape (N, N)
        """
        loc = returns.index.get_loc(date)
        start = max(0, loc - lookback)
        window = returns.iloc[start:loc].values  # (T, N)
        if window.shape[0] < 2:
            return np.eye(returns.shape[1], dtype=np.float32) * 1e-4
        cov = np.cov(window, rowvar=False)
        return cov.astype(np.float32)

    def compute_ledoit_wolf_covariance(
        self,
        returns: pd.DataFrame,
        date: pd.Timestamp,
        lookback: int = 63,
    ) -> tuple[np.ndarray, float]:
        """Compute Ledoit-Wolf shrinkage covariance estimate.

        Σ_shrunk = (1 - α) * S + α * μ * I

        where S is the sample covariance, μ = trace(S)/N, and α is the
        optimal shrinkage intensity from Ledoit & Wolf (2004).

        Returns
        -------
        cov_shrunk : ndarray (N, N)
        alpha : float — optimal shrinkage intensity
        """
        loc = returns.index.get_loc(date)
        start = max(0, loc - lookback)
        X = returns.iloc[start:loc].values  # (T, N)
        T, N = X.shape
        if T < 2:
            return np.eye(N, dtype=np.float32) * 1e-4, 1.0

        # Sample covariance
        X_centered = X - X.mean(axis=0)
        S = (X_centered.T @ X_centered) / (T - 1)

        # Shrinkage target: scaled identity
        mu = np.trace(S) / N

        # Optimal shrinkage intensity (Ledoit-Wolf 2004, Eq. 2)
        delta = S - mu * np.eye(N)
        # sum of squared off-diagonal + diagonal deviations
        delta_sq_sum = np.sum(delta ** 2) / N

        # Estimate of the squared Frobenius norm of the estimation error
        X2 = X_centered ** 2
        phi = np.sum((X2.T @ X2) / (T - 1) - S ** 2) / N

        # Clamp alpha to [0, 1]
        alpha = max(0.0, min(1.0, phi / (delta_sq_sum + 1e-10)))

        cov_shrunk = (1 - alpha) * S + alpha * mu * np.eye(N)
        return cov_shrunk.astype(np.float32), float(alpha)

    def compute_stock_index_covariance(
        self,
        returns: pd.DataFrame,
        index_returns: pd.Series,
        date: pd.Timestamp,
        lookback: int = 63,
    ) -> np.ndarray:
        """Compute covariance between each stock and the index.

        Returns
        -------
        sigma_idx : ndarray of shape (N,)
        """
        loc = returns.index.get_loc(date)
        start = max(0, loc - lookback)
        stock_window = returns.iloc[start:loc].values  # (T, N)
        idx_window = index_returns.iloc[start:loc].values  # (T,)
        if stock_window.shape[0] < 2:
            return np.zeros(returns.shape[1], dtype=np.float32)
        # Vectorized: cov(stock_i, idx) = E[(stock_i - μ_i)(idx - μ_idx)] for all i
        stock_centered = stock_window - stock_window.mean(axis=0)
        idx_centered = idx_window - idx_window.mean()
        cov_with_idx = (stock_centered.T @ idx_centered) / (stock_window.shape[0] - 1)
        return cov_with_idx.astype(np.float32)
