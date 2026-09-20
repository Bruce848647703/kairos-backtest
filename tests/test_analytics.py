import numpy as np
import pandas as pd
import pytest

import kairos_backtest as kb


def test_total_return_known():
    r = pd.Series([0.10, 0.10])
    assert kb.total_return(r) == pytest.approx(0.21, rel=1e-9)


def test_cagr_and_vol():
    # 每期 +1%，252 期 -> 年化约 (1.01)^252 - 1
    r = pd.Series([0.01] * 252)
    assert kb.cagr(r, 252) == pytest.approx(1.01 ** 252 - 1, rel=1e-6)
    assert kb.annualized_volatility(r, 252) == pytest.approx(0.0, abs=1e-12)


def test_sharpe_zero_vol_returns_zero():
    r = pd.Series([0.001] * 100)
    # 波动为 0 -> 约定返回 0（避免除零）
    assert kb.sharpe_ratio(r) == 0.0


def test_max_drawdown_known():
    eq = pd.Series([1.0, 1.2, 0.6, 0.9])
    r = eq.pct_change().fillna(0.0)
    # 峰值 1.2 -> 谷值 0.6，回撤 50%
    assert kb.max_drawdown(r) == pytest.approx(0.5, rel=1e-9)


def test_win_rate_profit_factor():
    r = pd.Series([0.02, -0.01, 0.03, -0.01])
    assert kb.win_rate(r) == pytest.approx(0.5)
    assert kb.profit_factor(r) == pytest.approx(0.05 / 0.02, rel=1e-9)


def test_summary_keys():
    r = pd.Series(np.linspace(-0.01, 0.02, 50))
    s = kb.summary(r)
    for k in ("total_return", "cagr", "sharpe", "max_drawdown", "win_rate"):
        assert k in s


def test_sortino_no_downside():
    r = pd.Series([0.01, 0.02, 0.03])
    assert kb.sortino_ratio(r) == float("inf")
