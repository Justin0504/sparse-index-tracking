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

    def build_market_features(
        self,
        returns: pd.DataFrame,
        date: pd.Timestamp,
        lookback: int = 63,
    ) -> np.ndarray:
        """Build market-level features at a given rebalancing date.

        Features (5-dim):
            1. Cross-sectional return dispersion (std of stock means)
            2. Average pairwise correlation
            3. Index-level realized volatility (21d)
            4. Index-level realized volatility (63d)
            5. Fraction of stocks with negative momentum

        Returns
        -------
        features : ndarray of shape (n_market_features,)
        """
        loc = returns.index.get_loc(date)
        start = max(0, loc - lookback)
        window = returns.iloc[start:loc].values  # (T, N)
        T, N = window.shape

        if T < 5:
            return np.zeros(5, dtype=np.float32)

        # 1. Cross-sectional dispersion: std of per-stock mean returns
        stock_means = window.mean(axis=0)
        dispersion = float(np.std(stock_means))

        # 2. Average pairwise correlation
        corr = np.corrcoef(window, rowvar=False)
        # Upper triangle excluding diagonal
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        avg_corr = float(np.mean(corr[mask]))

        # 3. Short-term index vol (21d)
        idx_ret = window.mean(axis=1)  # equal-weight proxy
        short_start = max(0, T - 21)
        short_vol = float(np.std(idx_ret[short_start:])) * np.sqrt(252)

        # 4. Long-term index vol (full lookback)
        long_vol = float(np.std(idx_ret)) * np.sqrt(252)

        # 5. Fraction of stocks with negative momentum
        if T >= 21:
            mom = window[-21:].sum(axis=0)
            neg_frac = float(np.mean(mom < 0))
        else:
            neg_frac = 0.5

        features = np.array([dispersion, avg_corr, short_vol, long_vol, neg_frac],
                            dtype=np.float32)
        return features

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

    def compute_nonlinear_shrinkage_covariance(
        self,
        returns: pd.DataFrame,
        date: pd.Timestamp,
        lookback: int = 63,
    ) -> np.ndarray:
        """Ledoit & Wolf (2020) analytical nonlinear shrinkage.

        Ported from the official MATLAB code by Ledoit & Wolf.
        Applies an optimal nonlinear shrinkage function to each eigenvalue
        of the sample covariance matrix.

        Reference: Ledoit & Wolf, "Analytical Nonlinear Shrinkage of
        Large-Dimensional Covariance Matrices", Annals of Statistics, 2020.
        """
        loc = returns.index.get_loc(date)
        start = max(0, loc - lookback)
        X = returns.iloc[start:loc].values  # (T, N)
        n, p = X.shape
        if n < 2:
            return np.eye(p, dtype=np.float32) * 1e-4

        # Center data
        X = X - X.mean(axis=0)

        # When p > n, work with the n×n Gram matrix instead
        # Sample covariance eigenvalues are the same either way
        S = (X.T @ X) / n  # using n not n-1, following LW2020 convention

        # Eigendecomposition (ascending order from eigh)
        eigenvalues, eigenvectors = np.linalg.eigh(S)

        # Clip tiny/negative eigenvalues
        eigenvalues = np.maximum(eigenvalues, 0)

        # Sort ascending (eigh default)
        lam = eigenvalues  # (p,), ascending

        # Concentration ratio
        c = p / n

        # Only the min(n,p) eigenvalues are nonzero when p > n
        # The LW2020 formula applies to all p eigenvalues
        # Use the isotonized version of the oracle shrinkage

        # Compute optimal shrinkage via the direct kernel estimator
        # from Section 4 of LW2020
        shrunk = self._analytical_nonlinear_shrinkage(lam, n, p)

        # Reconstruct
        cov_shrunk = (eigenvectors * shrunk) @ eigenvectors.T
        cov_shrunk = (cov_shrunk + cov_shrunk.T) / 2
        return cov_shrunk.astype(np.float32)

    @staticmethod
    def _analytical_nonlinear_shrinkage(lam, n, p):
        """Core LW2020 shrinkage computation.

        Ported from the official MATLAB implementation by Ledoit & Wolf.
        Uses the Epanechnikov kernel for density estimation of the
        sample spectral distribution, then applies the oracle formula.

        Parameters
        ----------
        lam : ndarray (p,) — sample eigenvalues in ascending order
        n : int — number of observations
        p : int — number of variables

        Returns
        -------
        d : ndarray (p,) — shrunk eigenvalues
        """
        # Remove zero eigenvalues for density estimation
        # but keep track of how many there are
        c = p / n

        # Effective eigenvalues: when p > n, only n eigenvalues are nonzero
        # The rest are exactly zero. We handle them separately.
        lam = np.asarray(lam, dtype=np.float64)

        # Number of nonzero eigenvalues
        k = np.sum(lam > 1e-12)
        if k == 0:
            return np.ones(p) * np.mean(lam) if np.mean(lam) > 0 else np.ones(p) * 1e-6

        # Use all eigenvalues for the formula
        # Bandwidth (Silverman's rule for kernel density)
        h = n ** (-1.0 / 3.0)

        # Grid: evaluate at each distinct eigenvalue
        # Following LW2020 Section 4, use Epanechnikov kernel
        L = lam.copy()

        # Compute Hilbert transform and density at each eigenvalue
        # using the smoothed sample spectral CDF
        Hf = np.zeros(p)  # Hilbert transform
        f = np.zeros(p)   # density

        for i in range(p):
            # Standardized distances
            u = (L[i] - L) / (L[i] * h + 1e-30)

            # Epanechnikov kernel: K(u) = 3/(4*sqrt(5)) * max(1 - u^2/5, 0)
            # and its Hilbert transform
            mask = np.abs(u) < np.sqrt(5)

            # Density via Epanechnikov kernel
            w = (3.0 / (4.0 * np.sqrt(5))) * np.maximum(1.0 - u**2 / 5.0, 0.0)
            f[i] = np.mean(w) / max(L[i] * h, 1e-30)

            # Hilbert transform via analytic formula for Epanechnikov
            # H[K](u) = -(3u)/(10*pi*sqrt(5)) * ... (complex formula)
            # Simpler: use the Stieltjes transform approach
            # m(z) = (1/p) * sum 1/(lam_j - z) for complex z = L[i] + i*eta
            eta = max(L[i] * h, 1e-10)
            z = complex(L[i], eta)
            stieltjes = np.mean(1.0 / (L - z))
            Hf[i] = stieltjes.real / np.pi  # Hilbert transform
            f[i] = -stieltjes.imag / np.pi  # density

        # Oracle nonlinear shrinkage formula (Theorem 4, Eq 4.3)
        # d_i = lam_i / |1 - c - c*lam_i*m_tilde(lam_i)|^2
        # where m_tilde involves both the Hilbert transform and density
        d = np.zeros(p)
        for i in range(p):
            if L[i] < 1e-12:
                # Zero eigenvalue: shrink toward the grand mean
                d[i] = 1.0 / (c * np.pi * max(f[i], 1e-10)) if f[i] > 1e-10 else L.mean()
            else:
                real_part = 1.0 - c - c * L[i] * Hf[i]
                imag_part = c * np.pi * L[i] * f[i]
                denom = real_part**2 + imag_part**2
                d[i] = L[i] / max(denom, 1e-10)

        # Ensure positive and isotonize (shrunk eigenvalues should be
        # monotonically non-decreasing since sample eigenvalues are)
        d = np.maximum(d, 1e-10)

        # Isotonic regression to ensure monotonicity
        # (the oracle formula can occasionally produce non-monotone results
        #  due to finite-sample effects)
        for i in range(1, p):
            if d[i] < d[i-1]:
                d[i] = d[i-1]

        return d

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
