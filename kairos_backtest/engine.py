"""事件驱动回测引擎。

模拟「按 bar 推进」的真实交易流程：策略在每个 bar 看到截至当前的历史数据，
下单后订单在 **下一个 bar** 成交（避免未来函数），成交价含滑点与佣金。

组件：Order / Position / Portfolio / SimBroker / Strategy / BacktestEngine。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import analytics
from .costs import CostModel, ZERO_COST


class Side(Enum):
    BUY = 1
    SELL = -1


@dataclass
class Order:
    symbol: str
    side: Side
    qty: float                 # 正数数量，方向由 side 决定
    created_bar: int = -1
    order_type: str = "market"


class Position:
    __slots__ = ("symbol", "qty", "avg_price", "realized_pnl")

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.qty = 0.0
        self.avg_price = 0.0
        self.realized_pnl = 0.0

    def market_value(self, price: float) -> float:
        return self.qty * price

    def apply_fill(self, side_sign: int, fill_qty: float, exec_price: float) -> None:
        signed = side_sign * fill_qty
        new_qty = self.qty + signed
        if side_sign > 0:  # 买入抬高/摊薄成本
            if new_qty != 0:
                self.avg_price = (self.qty * self.avg_price + fill_qty * exec_price) / new_qty
        else:              # 卖出实现盈亏
            self.realized_pnl += (self.avg_price - exec_price) * fill_qty
            if abs(new_qty) < 1e-12:
                self.avg_price = 0.0
        self.qty = new_qty


class Portfolio:
    def __init__(self, cash: float):
        self.cash = float(cash)
        self.positions: Dict[str, Position] = {}

    def position(self, symbol: str) -> Position:
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)
        return self.positions[symbol]

    def equity(self, prices: pd.Series) -> float:
        mv = 0.0
        for sym, pos in self.positions.items():
            if pos.qty == 0:
                continue
            p = prices.get(sym, np.nan)
            if not np.isnan(p):
                mv += pos.market_value(float(p))
        return self.cash + mv


class SimBroker:
    """极简撮合：市价单在给定 bar 价格上成交，含滑点与佣金。"""

    def __init__(self, cost_model: Optional[CostModel] = None, allow_margin: bool = False):
        self.cost = cost_model or ZERO_COST
        self.allow_margin = allow_margin
        self.pending: List[Order] = []
        self.fills: List[dict] = []

    def submit(self, order: Order) -> None:
        self.pending.append(order)

    def on_bar(self, bar_index: int, prices: pd.Series, portfolio: Portfolio) -> None:
        remaining: List[Order] = []
        for od in self.pending:
            price = prices.get(od.symbol, np.nan)
            if price is None or np.isnan(price):
                remaining.append(od)      # 无价，顺延
                continue
            self._fill(od, bar_index, float(price), portfolio)
        self.pending = remaining

    def _fill(self, od: Order, bar_index: int, price: float, portfolio: Portfolio) -> None:
        side_sign = od.side.value
        exec_price = self.cost.fill_price(price, side_sign)
        qty = abs(float(od.qty))
        if qty <= 0:
            return
        trade_value = exec_price * qty
        commission = self.cost.commission(trade_value)
        # 现金约束（无杠杆时，买入不得透支）
        if side_sign > 0 and not self.allow_margin:
            affordable = (portfolio.cash - commission) / exec_price if exec_price > 0 else 0.0
            if affordable <= 0:
                return
            qty = min(qty, affordable)
            trade_value = exec_price * qty
            commission = self.cost.commission(trade_value)
        if side_sign > 0:
            portfolio.cash -= trade_value + commission
        else:
            portfolio.cash += trade_value - commission
        pos = portfolio.position(od.symbol)
        pos.apply_fill(side_sign, qty, exec_price)
        self.fills.append({
            "bar": bar_index, "symbol": od.symbol, "side": od.side.name,
            "qty": qty, "price": exec_price, "commission": commission,
        })


class Strategy:
    """策略基类，子类实现 on_bar。"""

    def on_start(self, engine: "BacktestEngine") -> None:  # pragma: no cover
        pass

    def on_bar(self, i: int, engine: "BacktestEngine") -> None:
        raise NotImplementedError


class BacktestEngine:
    def __init__(self, prices: pd.DataFrame, strategy: Strategy,
                 cash: float = 1_000_000.0,
                 cost_model: Optional[CostModel] = None,
                 allow_margin: bool = False):
        self.prices = prices.astype("float64").sort_index()
        self.strategy = strategy
        self.portfolio = Portfolio(cash)
        self.broker = SimBroker(cost_model, allow_margin)
        self._equity: List[float] = []
        self.result: Optional[pd.Series] = None

    # ---- 供策略调用的接口 ----
    @property
    def bar(self) -> pd.Series:
        """当前 bar 的各资产价格。"""
        return self.prices.iloc[self._i]

    def history(self, symbol: str, n: int) -> pd.Series:
        """截至当前 bar（含）的最近 n 个价格，用于计算指标，无未来数据。"""
        end = self._i + 1
        start = max(0, end - n)
        return self.prices[symbol].iloc[start:end]

    def order_shares(self, symbol: str, qty: float) -> None:
        if qty == 0:
            return
        side = Side.BUY if qty > 0 else Side.SELL
        self.broker.submit(Order(symbol, side, abs(qty), created_bar=self._i))

    def order_target_percent(self, symbol: str, pct: float) -> None:
        """把某资产调整到组合净值的目标权重。"""
        price = self.bar.get(symbol, np.nan)
        if price is None or np.isnan(price) or price <= 0:
            return
        equity = self.portfolio.equity(self.bar)
        target_value = pct * equity
        cur_value = self.portfolio.position(symbol).qty * float(price)
        delta_value = target_value - cur_value
        qty = delta_value / float(price)
        self.order_shares(symbol, qty)

    # ---- 主循环 ----
    def run(self) -> pd.Series:
        self.strategy.on_start(self)
        idx = self.prices.index
        for i in range(len(self.prices)):
            self._i = i
            bar_prices = self.prices.iloc[i]
            # 1) 先成交上一 bar 提交的订单
            self.broker.on_bar(i, bar_prices, self.portfolio)
            # 2) 记录当 bar 净值
            self._equity.append(self.portfolio.equity(bar_prices))
            # 3) 策略据当前信息决策（订单下一 bar 成交）
            self.strategy.on_bar(i, self)
        self.result = pd.Series(self._equity, index=idx, name="equity")
        return self.result

    def returns(self) -> pd.Series:
        if self.result is None:
            raise RuntimeError("请先调用 run()")
        return self.result.pct_change().fillna(0.0)

    def metrics(self, risk_free: float = 0.0, periods_per_year: int = 252) -> Dict[str, float]:
        return analytics.summary(self.returns(), risk_free, periods_per_year)

    def trades(self) -> pd.DataFrame:
        return pd.DataFrame(self.broker.fills)
