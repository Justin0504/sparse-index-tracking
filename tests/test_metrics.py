"""Tests for evaluation metrics."""

import numpy as np
import pytest

from src.evaluation.metrics import (
    tracking_error,
    max_tracking_deviation,
    turnover,
    information_ratio,
    sparsity,
    effective_sparsity,
)


def test_tracking_error_zero():
    r = np.array([0.01, -0.02, 0.03, 0.01])
    assert tracking_error(r, r) == pytest.approx(0.0, abs=1e-10)


def test_tracking_error_positive():
    p = np.array([0.01, -0.02, 0.03])
    i = np.array([0.005, -0.01, 0.02])
    te = tracking_error(p, i)
    assert te > 0


def test_max_tracking_deviation():
    p = np.array([0.01, 0.02, -0.05])
    i = np.array([0.0, 0.0, 0.0])
    mtd = max_tracking_deviation(p, i)
    # cum diff = [0.01, 0.03, -0.02] → max abs = 0.03
    assert mtd == pytest.approx(0.03, abs=1e-10)


def test_turnover():
    w1 = np.array([0.5, 0.3, 0.2])
    w2 = np.array([0.4, 0.4, 0.2])
    t = turnover(w1, w2)
    # |−0.1| + |0.1| + 0 = 0.2, /2 = 0.1
    assert t == pytest.approx(0.1, abs=1e-10)


def test_information_ratio_zero_excess():
    r = np.array([0.01, 0.02, 0.03])
    ir = information_ratio(r, r)
    assert ir == pytest.approx(0.0, abs=1e-10)


def test_sparsity():
    w = np.array([0.5, 0.3, 0.0, 0.2, 0.0])
    assert sparsity(w) == 3


def test_effective_sparsity_equal_weight():
    w = np.ones(10) / 10
    es = effective_sparsity(w)
    assert es == pytest.approx(10.0, abs=0.1)


def test_effective_sparsity_concentrated():
    w = np.zeros(10)
    w[0] = 1.0
    es = effective_sparsity(w)
    assert es == pytest.approx(1.0, abs=0.1)
