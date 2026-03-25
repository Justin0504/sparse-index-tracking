"""Rolling-window train / val / test splitter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pandas as pd


@dataclass
class WindowSplit:
    """One rolling-window fold."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    fold: int

    def __repr__(self) -> str:
        return (
            f"Fold {self.fold}: "
            f"train [{self.train_start.date()}→{self.train_end.date()}] "
            f"val [{self.val_start.date()}→{self.val_end.date()}] "
            f"test [{self.test_start.date()}→{self.test_end.date()}]"
        )


class RollingWindowSplitter:
    """Generate expanding/rolling train-val-test splits from a date index.

    Parameters
    ----------
    train_years : int
        Length of training window in years.
    val_months : int
        Length of validation window in months.
    test_years : int
        Length of test window in years.
    step_years : int
        How many years to roll forward between folds.
    """

    def __init__(
        self,
        train_years: int = 3,
        val_months: int = 6,
        test_years: int = 1,
        step_years: int = 1,
    ):
        self.train_years = train_years
        self.val_months = val_months
        self.test_years = test_years
        self.step_years = step_years

    def split(self, dates: pd.DatetimeIndex) -> list[WindowSplit]:
        """Generate all non-overlapping rolling-window folds.

        Parameters
        ----------
        dates : sorted DatetimeIndex of all available trading days.

        Returns
        -------
        List of WindowSplit objects.
        """
        folds: list[WindowSplit] = []
        start = dates.min()
        end = dates.max()
        fold_idx = 0

        anchor = start
        while True:
            train_start = anchor
            train_end = train_start + pd.DateOffset(years=self.train_years)
            val_start = train_end
            val_end = val_start + pd.DateOffset(months=self.val_months)
            test_start = val_end
            test_end = test_start + pd.DateOffset(years=self.test_years)

            # Stop if test window exceeds available data
            if test_end > end:
                break

            # Snap to nearest trading days in the index
            folds.append(
                WindowSplit(
                    train_start=self._snap(dates, train_start, direction="forward"),
                    train_end=self._snap(dates, train_end, direction="backward"),
                    val_start=self._snap(dates, val_start, direction="forward"),
                    val_end=self._snap(dates, val_end, direction="backward"),
                    test_start=self._snap(dates, test_start, direction="forward"),
                    test_end=self._snap(dates, test_end, direction="backward"),
                    fold=fold_idx,
                )
            )
            fold_idx += 1
            anchor = anchor + pd.DateOffset(years=self.step_years)

        return folds

    def split_index(
        self, dates: pd.DatetimeIndex
    ) -> Iterator[tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]]:
        """Yield (train_dates, val_dates, test_dates) index slices per fold."""
        for w in self.split(dates):
            train_mask = (dates >= w.train_start) & (dates <= w.train_end)
            val_mask = (dates >= w.val_start) & (dates <= w.val_end)
            test_mask = (dates >= w.test_start) & (dates <= w.test_end)
            yield dates[train_mask], dates[val_mask], dates[test_mask]

    @staticmethod
    def _snap(
        dates: pd.DatetimeIndex,
        target: pd.Timestamp,
        direction: str = "forward",
    ) -> pd.Timestamp:
        """Snap a target date to the nearest trading day."""
        if target in dates:
            return target
        if direction == "forward":
            candidates = dates[dates >= target]
            return candidates[0] if len(candidates) > 0 else dates[-1]
        else:
            candidates = dates[dates <= target]
            return candidates[-1] if len(candidates) > 0 else dates[0]
