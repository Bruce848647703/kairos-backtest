"""Kairos Backtest 演示：向量化动量 + 事件驱动双均线。

运行： python examples/demo.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kairos_backtest as kb
from kairos_backtest import CostModel, BacktestEngine, Strategy, VectorBacktester


class SmaCross(Strategy):
    """双均线策略：短均线上穿长均线满仓，下穿清仓（单标的演示）。"""

    def __init__(self, symbol: str, fast: int = 10, slow: int = 30):
        self.symbol = symbol
        self.fast = fast
        self.slow = slow

    def on_bar(self, i, engine):
        if i < self.slow:
            return
        hist = engine.history(self.symbol, self.slow)
        if len(hist) < self.slow:
            return
        fast_ma = hist.iloc[-self.fast:].mean()
        slow_ma = hist.mean()
        target = 1.0 if fast_ma > slow_ma else 0.0
        engine.order_target_percent(self.symbol, target)


def main():
    prices = kb.make_gbm_prices(n_periods=500, symbols=["AAA", "BBB", "CCC"], seed=7)
    cost = CostModel(commission_rate=0.0003, commission_min=5.0, slippage_bps=5.0)

    print("=" * 60)
    print("① 向量化动量回测（每期持有过去 20 日最强的 1 只）")
    weights = kb.momentum_weights(prices, lookback=20, top_k=1)
    vres = VectorBacktester(cost_model=cost).run(prices, weights)
    print(kb.summary_frame(vres.returns).to_string(index=False))
    print(f"累计换手率: {vres.turnover.sum():.1f}")

    print("=" * 60)
    print("② 事件驱动双均线回测（标的 AAA，下一 bar 成交）")
    engine = BacktestEngine(prices[["AAA"]], SmaCross("AAA"), cash=1_000_000, cost_model=cost)
    engine.run()
    print(kb.summary_frame(engine.returns()).to_string(index=False))
    trades = engine.trades()
    print(f"成交笔数: {len(trades)}；期末净值: {engine.result.iloc[-1]:,.0f}")
    if not trades.empty:
        print(trades.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
