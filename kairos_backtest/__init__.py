"""Kairos Backtest —— 自研轻量回测框架。

提供两套互补的回测方式：
- VectorBacktester: 基于目标权重的向量化回测，适合因子/权重组批量筛选。
- BacktestEngine:   事件驱动回测，模拟逐 bar 下单、下一 bar 成交的真实流程。

以及绩效分析 (analytics)、成本模型 (CostModel)、合成行情 (sim)。

设计原则：防未来函数、纯 numpy/pandas 依赖、可测试、可扩展。
"""
from .analytics import (
    cagr,
    calmar_ratio,
    annualized_volatility,
    drawdown_series,
    equity_curve,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    summary,
    summary_frame,
    total_return,
    win_rate,
)
from .costs import CostModel, ZERO_COST
from .engine import BacktestEngine, Order, Portfolio, Position, Side, SimBroker, Strategy
from .sim import make_gbm_prices, momentum_weights
from .vectorized import VectorBacktester, VectorBacktestResult

__version__ = "0.1.0"

__all__ = [
    "VectorBacktester", "VectorBacktestResult",
    "BacktestEngine", "Strategy", "SimBroker", "Portfolio", "Position", "Order", "Side",
    "CostModel", "ZERO_COST",
    "make_gbm_prices", "momentum_weights",
    "summary", "summary_frame", "total_return", "cagr", "annualized_volatility",
    "sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio",
    "win_rate", "profit_factor", "equity_curve", "drawdown_series",
    "__version__",
]
