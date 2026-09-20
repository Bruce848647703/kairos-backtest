"""绩效分析模块。

所有函数接收「每期收益率」序列（pandas.Series 或可转换对象），
默认年化频率 periods_per_year=252（A 股/美股日线）。
纯 numpy/pandas 实现，无第三方依赖。
"""
from __future__ import annotations

from typing import Dict, Union

import numpy as np
import pandas as pd

SeriesLike = Union[pd.Series, np.ndarray]


def _as_returns(returns: SeriesLike) -> pd.Series:
    if isinstance(returns, pd.Series):
        s = returns.astype("float64").dropna()
    else:
        s = pd.Series(np.asarray(returns, dtype="float64")).dropna()
    return s


def equity_curve(returns: SeriesLike) -> pd.Series:
    """由每期收益率构造净值曲线（起点=1.0）。"""
    r = _as_returns(returns)
    return (1.0 + r).cumprod()


def total_return(returns: SeriesLike) -> float:
    """累计收益率（如 0.35 表示 +35%）。"""
    r = _as_returns(returns)
    if r.empty:
        return 0.0
    return float((1.0 + r).prod() - 1.0)


def cagr(returns: SeriesLike, periods_per_year: int = 252) -> float:
    """年化复合增长率 (CAGR)。"""
    r = _as_returns(returns)
    n = len(r)
    if n == 0:
        return 0.0
    growth = float((1.0 + r).prod())
    if growth <= 0:
        return -1.0
    years = n / float(periods_per_year)
    if years <= 0:
        return 0.0
    return float(growth ** (1.0 / years) - 1.0)


def annualized_volatility(returns: SeriesLike, periods_per_year: int = 252) -> float:
    """年化波动率（标准差）。"""
    r = _as_returns(returns)
    if len(r) < 2:
        return 0.0
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(returns: SeriesLike, risk_free: float = 0.0,
                 periods_per_year: int = 252) -> float:
    """夏普比率。risk_free 为年化无风险利率。"""
    r = _as_returns(returns)
    if len(r) < 2:
        return 0.0
    rf_per = risk_free / float(periods_per_year)
    excess = r - rf_per
    sd = excess.std(ddof=1)
    if sd < 1e-12 or np.isnan(sd):
        return 0.0
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def sortino_ratio(returns: SeriesLike, risk_free: float = 0.0,
                  periods_per_year: int = 252) -> float:
    """索提诺比率，仅用下行波动。"""
    r = _as_returns(returns)
    if len(r) < 2:
        return 0.0
    rf_per = risk_free / float(periods_per_year)
    excess = r - rf_per
    downside = excess[excess < 0]
    if downside.empty:
        return float("inf") if excess.mean() > 0 else 0.0
    ddof = 1 if len(downside) > 1 else 0
    dvol = float(np.sqrt((downside ** 2).sum() / max(len(r) - 1, 1)))
    if dvol < 1e-12:
        return 0.0
    return float(excess.mean() / dvol * np.sqrt(periods_per_year))


def drawdown_series(returns: SeriesLike) -> pd.Series:
    """回撤序列（<=0）。"""
    eq = equity_curve(returns)
    if eq.empty:
        return eq
    running_max = eq.cummax()
    return eq / running_max - 1.0


def max_drawdown(returns: SeriesLike) -> float:
    """最大回撤（正数表示回撤幅度，如 0.2 表示 -20%）。"""
    dd = drawdown_series(returns)
    if dd.empty:
        return 0.0
    return float(-dd.min())


def calmar_ratio(returns: SeriesLike, periods_per_year: int = 252) -> float:
    """卡尔玛比率 = 年化收益 / 最大回撤。"""
    mdd = max_drawdown(returns)
    if mdd == 0:
        return 0.0
    return float(cagr(returns, periods_per_year) / mdd)


def win_rate(returns: SeriesLike) -> float:
    """胜率 = 正收益期数 / 总期数。"""
    r = _as_returns(returns)
    active = r[r != 0]
    if active.empty:
        return 0.0
    return float((active > 0).sum() / len(active))


def profit_factor(returns: SeriesLike) -> float:
    """盈利因子 = 总盈利 / 总亏损（绝对值）。"""
    r = _as_returns(returns)
    gains = float(r[r > 0].sum())
    losses = float(-r[r < 0].sum())
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def summary(returns: SeriesLike, risk_free: float = 0.0,
            periods_per_year: int = 252) -> Dict[str, float]:
    """一次性返回常用绩效指标字典。"""
    r = _as_returns(returns)
    return {
        "periods": float(len(r)),
        "total_return": total_return(r),
        "cagr": cagr(r, periods_per_year),
        "volatility": annualized_volatility(r, periods_per_year),
        "sharpe": sharpe_ratio(r, risk_free, periods_per_year),
        "sortino": sortino_ratio(r, risk_free, periods_per_year),
        "max_drawdown": max_drawdown(r),
        "calmar": calmar_ratio(r, periods_per_year),
        "win_rate": win_rate(r),
        "profit_factor": profit_factor(r),
    }


def summary_frame(returns: SeriesLike, risk_free: float = 0.0,
                  periods_per_year: int = 252) -> pd.DataFrame:
    """把 summary 结果整理成两列 DataFrame，便于打印。"""
    d = summary(returns, risk_free, periods_per_year)
    return pd.DataFrame({"metric": list(d.keys()), "value": list(d.values())})
