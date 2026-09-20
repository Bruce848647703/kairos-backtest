import numpy as np
import pandas as pd
import pytest

from kairos_backtest import VectorBacktester, CostModel, make_gbm_prices


def test_zero_cost_full_investment_tracks_asset():
    idx = pd.bdate_range("2021-01-01", periods=10)
    prices = pd.DataFrame({"A": np.linspace(100, 110, 10)}, index=idx)
    w = pd.DataFrame(1.0, index=idx, columns=["A"])  # 始终满仓 A
    res = VectorBacktester(cost_rate=0.0).run(prices, w)
    # 第 0 期建仓无前收，净收益从第 1 期起等于资产收益
    asset_ret = prices["A"].pct_change().fillna(0.0)
    pd.testing.assert_series_equal(
        res.returns.iloc[1:].rename(None), asset_ret.iloc[1:].rename(None), check_names=False
    )


def test_no_lookahead_uses_previous_weight():
    idx = pd.bdate_range("2021-01-01", periods=4)
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0, 133.1]}, index=idx)
    w = pd.DataFrame({"A": [0.0, 1.0, 1.0, 1.0]}, index=idx)
    res = VectorBacktester(cost_rate=0.0).run(prices, w)
    # 第 1 期目标权重=1，但当期持有=上一期权重=0 -> 第 1 期净收益应为 0
    assert res.returns.iloc[1] == pytest.approx(0.0, abs=1e-12)
    # 第 2 期才真正享受 A 的收益
    assert res.returns.iloc[2] == pytest.approx(0.1, rel=1e-9)


def test_costs_reduce_returns_and_scale_with_turnover():
    prices = make_gbm_prices(n_periods=60, symbols=["A", "B"], seed=3)
    w_free = pd.DataFrame(0.5, index=prices.index, columns=["A", "B"])
    zero = VectorBacktester(cost_rate=0.0).run(prices, w_free)
    costly = VectorBacktester(cost_model=CostModel(0.001, 0.0, 10.0)).run(prices, w_free)
    # 恒定权重换手极低，但成本仍应 <= 零成本
    assert costly.returns.sum() <= zero.returns.sum() + 1e-12
    assert costly.cost_rate == pytest.approx(0.001 + 10 / 10000.0)


def test_turnover_first_period_is_initial_build():
    idx = pd.bdate_range("2021-01-01", periods=3)
    prices = pd.DataFrame({"A": [100.0, 100.0, 100.0]}, index=idx)
    w = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=idx)
    res = VectorBacktester(cost_rate=0.0).run(prices, w)
    assert res.turnover.iloc[0] == pytest.approx(1.0)  # 建仓换手=1
    assert res.turnover.iloc[1] == pytest.approx(0.0, abs=1e-12)  # 之后不再调仓
