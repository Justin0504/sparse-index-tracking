"""Tests for rolling window splitter."""

import pandas as pd
import pytest

from src.data.splitter import RollingWindowSplitter


@pytest.fixture
def trading_dates():
    return pd.bdate_range("2006-01-01", "2025-12-31")


def test_split_produces_folds(trading_dates):
    splitter = RollingWindowSplitter(train_years=3, val_months=6, test_years=1, step_years=1)
    folds = splitter.split(trading_dates)
    assert len(folds) > 0


def test_no_overlap(trading_dates):
    splitter = RollingWindowSplitter(train_years=3, val_months=6, test_years=1, step_years=1)
    folds = splitter.split(trading_dates)
    for f in folds:
        assert f.train_end <= f.val_start
        assert f.val_end <= f.test_start


def test_fold_ordering(trading_dates):
    splitter = RollingWindowSplitter(train_years=3, val_months=6, test_years=1, step_years=1)
    folds = splitter.split(trading_dates)
    for f in folds:
        assert f.train_start <= f.train_end
        assert f.val_start <= f.val_end
        assert f.test_start <= f.test_end


def test_split_index_yields_correct_count(trading_dates):
    splitter = RollingWindowSplitter(train_years=3, val_months=6, test_years=1, step_years=1)
    folds = list(splitter.split_index(trading_dates))
    expected = len(splitter.split(trading_dates))
    assert len(folds) == expected


def test_split_index_non_empty(trading_dates):
    splitter = RollingWindowSplitter(train_years=3, val_months=6, test_years=1, step_years=1)
    for train, val, test in splitter.split_index(trading_dates):
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0
