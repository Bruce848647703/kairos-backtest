# Kairos Backtest

> Kairos 量化系列的回测模块 —— 一个**自研、轻量、零重型依赖**的 Python 回测框架。

`kairos_backtest` 提供两套互补的回测方式，以及完整的绩效分析与成本模型，
专为 A 股 / 加密 / 通用资产的策略验证设计。核心代码全部原创，仅依赖 `numpy` 与 `pandas`。

## 特性
- **向量化回测 `VectorBacktester`**：输入目标权重面板，快速评估因子/权重组，内置换手率与成本扣减。
- **事件驱动回测 `BacktestEngine`**：逐 bar 推进，策略下单后**下一 bar 成交**，模拟真实撮合、滑点与佣金。
- **绩效分析 `analytics`**：CAGR、年化波动、Sharpe、Sortino、最大回撤、Calmar、胜率、盈利因子等。
- **成本模型 `CostModel`**：比例佣金 + 最低佣金 + 基点滑点，可组合可扩展。
- **防未来函数**：向量化用「上一期权重 × 本期收益」；事件驱动订单延迟一 bar 成交。
- **可测试**：自带合成行情 `sim`，示例与测试全部离线、可复现。

## 安装
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # 或 pip install numpy pandas
pip install -e ".[dev]"     # 需要跑测试时
```

## 快速开始
### ① 向量化回测
```python
import kairos_backtest as kb
from kairos_backtest import VectorBacktester, CostModel

prices = kb.make_gbm_prices(500, ["AAA", "BBB", "CCC"], seed=7)  # 合成行情
weights = kb.momentum_weights(prices, lookback=20, top_k=1)       # 动量权重
res = VectorBacktester(cost_model=CostModel(0.0003, 5.0, 5.0)).run(prices, weights)
print(kb.summary_frame(res.returns))
```

### ② 事件驱动回测
```python
from kairos_backtest import BacktestEngine, Strategy

class SmaCross(Strategy):
    def on_bar(self, i, engine):
        if i < 30:
            return
        hist = engine.history("AAA", 30)
        target = 1.0 if hist.iloc[-10:].mean() > hist.mean() else 0.0
        engine.order_target_percent("AAA", target)

eng = BacktestEngine(prices[["AAA"]], SmaCross(), cash=1_000_000)
eng.run()
print(eng.metrics())
print(eng.trades().head())
```

完整可运行示例见 [`examples/demo.py`](examples/demo.py)。

## API 概览
| 模块 | 关键对象 | 说明 |
|---|---|---|
| `vectorized` | `VectorBacktester` | 权重面板 → 净值/换手/成本 |
| `engine` | `BacktestEngine` `Strategy` `SimBroker` `Portfolio` `Order` | 事件驱动撮合 |
| `analytics` | `summary` `sharpe_ratio` `max_drawdown` `cagr` … | 绩效指标 |
| `costs` | `CostModel` | 佣金 + 滑点 |
| `sim` | `make_gbm_prices` `momentum_weights` | 合成行情与示例权重 |

## 设计要点
- **时序对齐**：所有信号在第 `t` 期用截至 `t` 的数据生成，收益在 `t→t+1` 实现，杜绝前视偏差。
- **成交模型**：市价单在下一 bar 价格上叠加滑点成交；无杠杆模式下买入受现金约束，不会透支。
- **可扩展**：`SimBroker`/`CostModel`/`Strategy` 均为可继承的清晰接口，便于接入更真实的撮合与冲击成本。

## 测试
```bash
make test          # 或 python -m pytest -q
```

## 项目结构
```
kairos_backtest/    核心包（analytics / costs / vectorized / engine / sim）
examples/           可运行示例
tests/              pytest 测试
```

## 许可
MIT © 2026 Bruce848647703，见 [LICENSE](LICENSE)。

## 参考与致谢
本项目为**独立原创实现**，未复制任何第三方代码。设计思路受业界通用回测范式
（向量化组合回测、事件驱动撮合、防未来函数的时序对齐、TCA 成本建模）启发，
在此向开源量化社区致谢。算法与接口均为本仓库自研。
