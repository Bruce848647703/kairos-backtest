import numpy as np
import pandas as pd
import pytest

from kairos_backtest import BacktestEngine, Strategy, CostModel, ZERO_COST, Side


class BuyAndHold(Strategy):
    def __init__(self, symbol):
        self.symbol = symbol
        self._bought = False

    def on_bar(self, i, engine):
        if not self._bought:
            engine.order_target_percent(self.symbol, 1.0)
            self._bought = True


class BuyOnceShares(Strategy):
    def on_start(self, engine):
        pass

    def on_bar(self, i, engine):
        if i == 0:
            engine.order_shares("A", 10)


def test_order_fills_next_bar_not_same_bar():
    idx = pd.bdate_range("2021-01-01", periods=5)
    prices = pd.DataFrame({"A": [100.0, 100.0, 100.0, 100.0, 100.0]}, index=idx)
    eng = BacktestEngine(prices, BuyOnceShares(), cash=10000, cost_model=ZERO_COST)
    eng.run()
    fills = eng.trades()
    # 第 0 bar 下单，应在第 1 bar 成交
    assert len(fills) == 1
    assert fills.iloc[0]["bar"] == 1
    assert fills.iloc[0]["qty"] == pytest.approx(10.0)


def test_cash_constraint_blocks_overdraft():
    idx = pd.bdate_range("2021-01-01", periods=3)
    prices = pd.DataFrame({"A": [100.0, 100.0, 100.0]}, index=idx)

    class Greedy(Strategy):
        def on_bar(self, i, engine):
            if i == 0:
                engine.order_shares("A", 1000)  # 需 10 万，但只有 1 万现金

    eng = BacktestEngine(prices, Greedy(), cash=10000, cost_model=ZERO_COST, allow_margin=False)
    eng.run()
    assert eng.portfolio.cash >= -1e-9  # 未透支
    assert eng.portfolio.position("A").qty <= 100.0 + 1e-9  # 只买得起约 100 股


def test_buy_and_hold_equity_tracks_price_with_cost():
    n = 50
    idx = pd.bdate_range("2021-01-01", periods=n)
    prices = pd.DataFrame({"A": np.linspace(100, 150, n)}, index=idx)
    cost = CostModel(commission_rate=0.0003, commission_min=0.0, slippage_bps=5.0)
    eng = BacktestEngine(prices, BuyAndHold("A"), cash=100000, cost_model=cost)
    eng.run()
    # 净值应随价格上行而增长
    assert eng.result.iloc[-1] > eng.result.iloc[0]
    m = eng.metrics()
    assert m["total_return"] > 0
    assert m["max_drawdown"] >= 0.0


def test_slippage_worsens_buy_price():
    idx = pd.bdate_range("2021-01-01", periods=3)
    prices = pd.DataFrame({"A": [100.0, 100.0, 100.0]}, index=idx)
    eng = BacktestEngine(prices, BuyOnceShares(), cash=100000,
                         cost_model=CostModel(0.0, 0.0, slippage_bps=50.0))
    eng.run()
    fill_price = eng.trades().iloc[0]["price"]
    # 50bp 滑点买入：100 * 1.005 = 100.5
    assert fill_price == pytest.approx(100.5, rel=1e-9)


def test_returns_length_matches_bars():
    prices = pd.DataFrame({"A": np.arange(1, 21, dtype=float)},
                          index=pd.bdate_range("2021-01-01", periods=20))
    eng = BacktestEngine(prices, BuyAndHold("A"), cash=1000, cost_model=ZERO_COST)
    eng.run()
    assert len(eng.returns()) == 20
