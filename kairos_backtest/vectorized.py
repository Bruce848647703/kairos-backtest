"""向量化回测器。

给定资产价格与目标权重（均为 date x asset 的 DataFrame），
用向量化方式快速估算组合收益，适合因子/权重组的批量筛选。

关键防未来函数设计：第 t 期决定的目标权重，在 t→t+1 期间持有，
因此组合收益用「上一期权重 × 本期资产收益」。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import analytics
from .costs import CostModel, ZERO_COST


@dataclass
class VectorBacktestResult:
    returns: pd.Series                 # 每期净收益
    gross_returns: pd.Series           # 每期毛收益（未扣成本）
    equity: pd.Series                  # 净值曲线
    turnover: pd.Series                # 每期换手率
    weights: pd.DataFrame              # 实际持有权重（滞后后）
    cost_rate: float                   # 单边成本率（换手成本系数）

    def metrics(self, risk_free: float = 0.0, periods_per_year: int = 252) -> Dict[str, float]:
        return analytics.summary(self.returns, risk_free, periods_per_year)


class VectorBacktester:
    """基于目标权重的向量化回测。

    参数
    ----
    prices:     资产价格面板 (index=日期, columns=资产)。
    cost_rate:  换手成本率（对换手率收取的比例，如 0.001 表示单边千一）。
                若提供 CostModel，则用 commission_rate + slippage 近似换算。
    """

    def __init__(self, cost_rate: Optional[float] = None,
                 cost_model: Optional[CostModel] = None):
        if cost_rate is None:
            cm = cost_model or ZERO_COST
            cost_rate = cm.commission_rate + cm.slippage_bps / 10000.0
        self.cost_rate = float(cost_rate)

    def run(self, prices: pd.DataFrame, target_weights: pd.DataFrame) -> VectorBacktestResult:
        prices = prices.astype("float64").sort_index()
        w = target_weights.astype("float64").reindex(index=prices.index, columns=prices.columns).fillna(0.0)

        asset_ret = prices.pct_change().fillna(0.0)
        # 实际持有权重 = 上一期目标权重（防未来函数）
        held = w.shift(1).fillna(0.0)

        gross = (held * asset_ret).sum(axis=1)

        # 换手率：目标权重相对「随收益漂移后的持有权重」的变化
        drifted = self._drift(held, asset_ret)
        turnover = (w - drifted).abs().sum(axis=1)
        turnover.iloc[0] = w.iloc[0].abs().sum()  # 建仓换手

        costs = turnover * self.cost_rate
        net = gross - costs

        equity = (1.0 + net).cumprod()
        return VectorBacktestResult(
            returns=net, gross_returns=gross, equity=equity,
            turnover=turnover, weights=held, cost_rate=self.cost_rate,
        )

    @staticmethod
    def _drift(held: pd.DataFrame, asset_ret: pd.DataFrame) -> pd.DataFrame:
        """持有权重随收益漂移后的下一期权重（用于计算换手）。"""
        grown = held * (1.0 + asset_ret)
        total = grown.sum(axis=1).replace(0.0, np.nan)
        return grown.div(total, axis=0).fillna(0.0)
