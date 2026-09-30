# Kairos Backtest

[![CI](https://github.com/Bruce848647703/kairos-backtest/actions/workflows/ci.yml/badge.svg)](https://github.com/Bruce848647703/kairos-backtest/actions/workflows/ci.yml)

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

## 真实数据回测
在**真实 A 股日线行情**上跑通两套引擎，并把结果落盘为可复现的研究产物。

```bash
python examples/real_backtest.py \
    --data-dir /home/zhuoming.wang/quant-hub/kairos/kairos-data/data/ashare
```

- **数据加载** `kairos_backtest.realdata.load_close_panel(data_dir)`：读取目录下 `*.csv`
  （`date,open,high,low,close,volume`），按**同系列口径**清洗——非正价→NaN、`ffill` 衔接
  停牌/缺失、按**全体上市日**裁剪到公共有效区间，返回无缺口的 `date x symbol` 收盘价面板。
- **向量化**：横截面动量 Top-K（自研权重面板，信号只用截至当期数据）经 `VectorBacktester`
  扣成本回测，对照**等权买入持有**基准。
- **事件驱动**：单标的 `Strategy` 子类做双均线择时，经 `BacktestEngine` 在真实价上逐 bar 运行、
  **下一 bar 成交**，对照**买入持有**基准，输出成交明细。
- **产物**（写入 `research/real_backtest/`）：`REPORT.md`（中文报告：累计收益/CAGR/夏普/索提诺/
  最大回撤/卡玛/换手 + 基准对照 + 诚实结论）、`equity.csv`（净值曲线）、`metrics.json`（结构化指标）、
  `trades.csv`（成交明细）。

> **数据声明**：示例行情为**公开来源、前复权**的 A 股日线，**仅供研究与学习**，不保证准确/完整/及时；
> 全部回测均为历史模拟，**不构成任何投资建议**。原始数据不在本仓库内，请自备或用 `--data-dir` 指向本地目录。

## API 概览
| 模块 | 关键对象 | 说明 |
|---|---|---|
| `vectorized` | `VectorBacktester` | 权重面板 → 净值/换手/成本 |
| `engine` | `BacktestEngine` `Strategy` `SimBroker` `Portfolio` `Order` | 事件驱动撮合 |
| `analytics` | `summary` `sharpe_ratio` `max_drawdown` `cagr` … | 绩效指标 |
| `costs` | `CostModel` | 佣金 + 滑点 |
| `sim` | `make_gbm_prices` `momentum_weights` | 合成行情与示例权重 |
| `realdata` | `load_close_panel` `list_symbols` | 真实行情 CSV → 收盘价面板 |

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
kairos_backtest/    核心包（analytics / costs / vectorized / engine / sim / realdata）
examples/           可运行示例（demo.py 合成行情、real_backtest.py 真实数据）
tests/              pytest 测试（全部离线）
research/           研究产物（real_backtest/ 内含真实数据回测报告与结果）
```

## 许可
MIT © 2026 Bruce848647703，见 [LICENSE](LICENSE)。

## 参考与致谢
本项目为**独立原创实现**，未复制任何第三方代码。设计思路受业界通用回测范式
（向量化组合回测、事件驱动撮合、防未来函数的时序对齐、TCA 成本建模）启发，
在此向开源量化社区致谢。算法与接口均为本仓库自研。
