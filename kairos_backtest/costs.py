"""交易成本模型。

包含佣金（比例 + 最低收费）与滑点（按基点 bps 影响成交价）。
设计为可组合、可继承，方便接入更复杂的冲击成本模型。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostModel:
    """比例佣金 + 最低佣金 + 滑点(基点)。

    参数
    ----
    commission_rate: 成交额佣金比例，如 0.0003 表示万三。
    commission_min:  单笔最低佣金（元）。
    slippage_bps:    滑点基点，1bp = 0.01%。买入抬价、卖出压价。
    """

    commission_rate: float = 0.0003
    commission_min: float = 0.0
    slippage_bps: float = 5.0

    def commission(self, trade_value: float) -> float:
        """按成交额计算佣金（取比例与最低收费的较大者）。"""
        value = abs(float(trade_value))
        return max(value * self.commission_rate, self.commission_min if value > 0 else 0.0)

    def fill_price(self, price: float, side_sign: int) -> float:
        """根据方向计算含滑点的成交价。

        side_sign: +1 买入（价格上移），-1 卖出（价格下移）。
        """
        slip = float(price) * self.slippage_bps / 10000.0
        return float(price) + side_sign * slip

    def total_cost(self, price: float, qty: float, side_sign: int) -> float:
        """一笔交易的总显性成本（佣金，不含已体现在成交价里的滑点）。"""
        exec_price = self.fill_price(price, side_sign)
        return self.commission(exec_price * abs(qty))


ZERO_COST = CostModel(commission_rate=0.0, commission_min=0.0, slippage_bps=0.0)
