"""合成行情工具，用于示例与测试（无需联网、结果可复现）。"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


def make_gbm_prices(n_periods: int = 500,
                    symbols: Optional[List[str]] = None,
                    start: str = "2020-01-01",
                    mu: float = 0.08,
                    sigma: float = 0.25,
                    s0: float = 100.0,
                    seed: int = 42,
                    freq: str = "B") -> pd.DataFrame:
    """生成几何布朗运动价格面板 (index=交易日, columns=资产)。

    mu/sigma 为年化漂移与波动，按 252 交易日折算到每期。
    """
    symbols = symbols or ["AAA", "BBB", "CCC"]
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_periods, freq=freq)
    dt = 1.0 / 252.0
    out = {}
    for k, sym in enumerate(symbols):
        z = rng.standard_normal(n_periods)
        # 每个资产用不同种子偏移，保证相关但有差异
        drift = (mu - 0.5 * sigma ** 2) * dt
        shock = sigma * np.sqrt(dt) * z
        log_ret = drift + shock
        prices = s0 * np.exp(np.cumsum(log_ret))
        out[sym] = prices
    return pd.DataFrame(out, index=idx)


def momentum_weights(prices: pd.DataFrame, lookback: int = 20,
                     top_k: int = 1) -> pd.DataFrame:
    """一个极简动量权重示例：每 lookback 期选过去涨幅最高的 top_k 资产等权持有。

    仅用于演示向量化回测的输入形态；不含未来数据（用截至 t 的收益决定 t 的权重）。
    """
    mom = prices.pct_change(lookback)
    ranks = mom.rank(axis=1, ascending=False, method="first")
    w = (ranks <= top_k).astype(float)
    row_sum = w.sum(axis=1).replace(0, np.nan)
    return w.div(row_sum, axis=0).fillna(0.0)
