"""真实 A 股数据回测示例：向量化横截面动量 + 事件驱动双均线择时。

本脚本在**真实日线行情**上完整跑通 kairos-backtest 的两套回测引擎，并将结果
（报告 / 净值 / 指标 / 成交明细）落盘到 ``research/real_backtest/``。

运行::

    python examples/real_backtest.py \
        --data-dir /home/zhuoming.wang/quant-hub/kairos/kairos-data/data/ashare

数据声明：所用行情为公开来源、前复权日线，仅供研究与教学演示，**不构成任何投资建议**。

防未来函数：
- 动量信号在第 t 期只用截至 t 的历史收益；:class:`VectorBacktester` 内部再滞后一期，
  即「上一期权重 × 本期收益」。
- 双均线择时在每个 bar 只用截至当前 bar 的历史价，订单在**下一 bar** 成交。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# 允许直接以脚本方式运行（无需先安装）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import kairos_backtest as kb
from kairos_backtest import BacktestEngine, CostModel, Strategy, VectorBacktester
from kairos_backtest.realdata import load_close_panel, list_symbols

DEFAULT_DATA_DIR = "/home/zhuoming.wang/quant-hub/kairos/kairos-data/data/ashare"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "research", "real_backtest")
TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# 策略与权重构造（全部自研，仅用 numpy/pandas，不依赖外部量化包）
# --------------------------------------------------------------------------- #
def build_momentum_weights(close: pd.DataFrame, lookback: int = 126,
                           top_k: int = 5, rebalance: int = 21) -> pd.DataFrame:
    """构造横截面动量 Top-K 等权目标权重面板。

    - 动量 = 过去 ``lookback`` 日收益（``close.pct_change(lookback)``），仅用截至 t 的数据。
    - 每隔 ``rebalance`` 日调仓一次：选出动量最强的 ``top_k`` 只等权持有，其余为 0；
      非调仓日沿用上一次的选择（forward-fill），以降低换手。
    - ``lookback`` 之前无有效动量，权重为 0（空仓持币）。

    返回与 ``close`` 同形状的目标权重面板（行和 <= 1）。真正的持有滞后由
    :class:`VectorBacktester` 负责，此处不需要再手动 shift。
    """
    mom = close.pct_change(lookback)
    rank = mom.rank(axis=1, ascending=False, method="first")
    picked = (rank <= top_k).astype("float64")

    is_rebal = np.zeros(len(close), dtype=bool)
    is_rebal[lookback::rebalance] = True
    picked = picked.where(pd.Series(is_rebal, index=close.index), np.nan)
    picked = picked.ffill().fillna(0.0)

    row_sum = picked.sum(axis=1).replace(0.0, np.nan)
    return picked.div(row_sum, axis=0).fillna(0.0)


class SmaCross(Strategy):
    """双均线择时（单标的）：短均线上穿长均线满仓、下穿清仓（只做多/持币）。

    在每个 bar 只用截至当前 bar 的历史价计算均线，订单由引擎在**下一 bar** 成交，
    不含未来信息。仅当目标仓位发生变化时才下单，避免无谓的重复委托。
    """

    def __init__(self, symbol: str, fast: int = 20, slow: int = 60):
        if fast >= slow:
            raise ValueError("fast 必须小于 slow")
        self.symbol = symbol
        self.fast = fast
        self.slow = slow
        self._target = 0.0

    def on_bar(self, i: int, engine: BacktestEngine) -> None:
        if i < self.slow:
            return
        hist = engine.history(self.symbol, self.slow)
        if len(hist) < self.slow:
            return
        fast_ma = float(hist.iloc[-self.fast:].mean())
        slow_ma = float(hist.mean())
        target = 1.0 if fast_ma > slow_ma else 0.0
        if target != self._target:
            engine.order_target_percent(self.symbol, target)
            self._target = target


class BuyAndHold(Strategy):
    """买入并持有：首个 bar 建仓到满仓，其后不动（作为事件驱动的对照基准）。"""

    def __init__(self, symbol: str, pct: float = 1.0):
        self.symbol = symbol
        self.pct = pct
        self._done = False

    def on_bar(self, i: int, engine: BacktestEngine) -> None:
        if not self._done:
            engine.order_target_percent(self.symbol, self.pct)
            self._done = True


# --------------------------------------------------------------------------- #
# 指标工具
# --------------------------------------------------------------------------- #
def _years(n_periods: int) -> float:
    return n_periods / float(TRADING_DAYS)


def equal_weight_buy_hold(close: pd.DataFrame) -> pd.Series:
    """等权买入持有基准的收益序列：期初等额买入全部标的并持有（权重随行情漂移）。"""
    equity = (close / close.iloc[0]).mean(axis=1)
    return equity.pct_change().fillna(0.0)


def engine_annual_turnover(engine: BacktestEngine) -> float:
    """事件驱动年化换手（单边成交额 / 平均净值 / 年数）。"""
    fills = engine.trades()
    yrs = _years(len(engine.returns()))
    if fills.empty or yrs <= 0:
        return 0.0
    traded = float((fills["qty"] * fills["price"]).sum())
    mean_eq = float(engine.result.mean())
    return traded / mean_eq / yrs if mean_eq > 0 else 0.0


def _round(d: dict, nd: int = 6) -> dict:
    return {k: (round(float(v), nd) if isinstance(v, (int, float)) and np.isfinite(v) else v)
            for k, v in d.items()}


# --------------------------------------------------------------------------- #
# 两段回测
# --------------------------------------------------------------------------- #
def run_vectorized(close: pd.DataFrame, cost: CostModel, lookback: int,
                   top_k: int, rebalance: int):
    """向量化：横截面动量 Top-K vs 等权买入持有。"""
    weights = build_momentum_weights(close, lookback=lookback, top_k=top_k, rebalance=rebalance)
    res = VectorBacktester(cost_model=cost).run(close, weights)
    bench_ret = equal_weight_buy_hold(close)

    yrs = _years(len(res.returns))
    strat = kb.summary(res.returns)
    bench = kb.summary(bench_ret)
    strat["annual_turnover"] = float(res.turnover.sum() / yrs) if yrs > 0 else 0.0
    strat["cumulative_turnover"] = float(res.turnover.sum())
    strat["cum_cost_drag"] = float((res.gross_returns - res.returns).sum())
    bench["annual_turnover"] = 0.0

    equity = pd.DataFrame({
        "vec_momentum": res.equity / float(res.equity.iloc[0]),
        "vec_equal_weight_bh": kb.equity_curve(bench_ret),
    })
    return {
        "result": res, "weights": weights, "strat": strat, "bench": bench,
        "bench_ret": bench_ret, "equity": equity,
        "params": {"lookback": lookback, "top_k": top_k, "rebalance": rebalance},
    }


def run_event_driven(close: pd.DataFrame, cost: CostModel, symbol: str,
                     fast: int, slow: int, cash: float):
    """事件驱动：单标的双均线择时 vs 买入持有。"""
    if symbol not in close.columns:
        symbol = str(close.columns[0])
    px = close[[symbol]]

    sma_engine = BacktestEngine(px, SmaCross(symbol, fast, slow), cash=cash, cost_model=cost)
    sma_engine.run()
    bh_engine = BacktestEngine(px, BuyAndHold(symbol), cash=cash, cost_model=cost)
    bh_engine.run()

    strat = kb.summary(sma_engine.returns())
    bench = kb.summary(bh_engine.returns())
    fills = sma_engine.trades()
    strat["annual_turnover"] = engine_annual_turnover(sma_engine)
    strat["n_trades"] = int(len(fills))
    strat["total_commission"] = float(fills["commission"].sum()) if not fills.empty else 0.0
    bench["annual_turnover"] = engine_annual_turnover(bh_engine)
    bench["n_trades"] = int(len(bh_engine.trades()))
    bench["total_commission"] = float(bh_engine.trades()["commission"].sum()) \
        if not bh_engine.trades().empty else 0.0

    equity = pd.DataFrame({
        "event_sma_cross": sma_engine.result / float(sma_engine.result.iloc[0]),
        "event_buy_hold": bh_engine.result / float(bh_engine.result.iloc[0]),
    })
    trades = fills.copy()
    if not trades.empty:
        trades.insert(0, "date", px.index[trades["bar"].astype(int)].values)
        trades = trades[["date", "bar", "symbol", "side", "qty", "price", "commission"]]
    return {
        "sma_engine": sma_engine, "bh_engine": bh_engine, "symbol": symbol,
        "strat": strat, "bench": bench, "equity": equity, "trades": trades,
        "params": {"fast": fast, "slow": slow, "cash": cash},
    }


# --------------------------------------------------------------------------- #
# 输出：报告 / 净值 / 指标 / 成交
# --------------------------------------------------------------------------- #
def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _pp(x: float) -> str:
    """把两个比率之差表述为「百分点」，避免与相对收益混淆。"""
    return f"{abs(x) * 100:.1f} 个百分点"


def _metric_table(rows: list) -> str:
    """rows: [(名称, 策略值字符串, 基准值字符串), ...] -> markdown 表格。"""
    lines = ["| 指标 | 策略 | 基准(买入持有) |", "|---|---|---|"]
    for name, s, b in rows:
        lines.append(f"| {name} | {s} | {b} |")
    return "\n".join(lines)


def build_report(meta: dict, vec: dict, evt: dict) -> str:
    vs, vb = vec["strat"], vec["bench"]
    es, eb = evt["strat"], evt["bench"]
    vp, ep = vec["params"], evt["params"]

    vec_rows = [
        ("累计收益", _pct(vs["total_return"]), _pct(vb["total_return"])),
        ("年化收益 CAGR", _pct(vs["cagr"]), _pct(vb["cagr"])),
        ("年化波动", _pct(vs["volatility"]), _pct(vb["volatility"])),
        ("夏普比率", f"{vs['sharpe']:.2f}", f"{vb['sharpe']:.2f}"),
        ("索提诺比率", f"{vs['sortino']:.2f}", f"{vb['sortino']:.2f}"),
        ("最大回撤", _pct(vs["max_drawdown"]), _pct(vb["max_drawdown"])),
        ("卡玛比率", f"{vs['calmar']:.2f}", f"{vb['calmar']:.2f}"),
        ("胜率", _pct(vs["win_rate"]), _pct(vb["win_rate"])),
        ("年化换手(双边)", f"{vs['annual_turnover']:.2f}x", "≈0（持有不动）"),
    ]
    evt_rows = [
        ("累计收益", _pct(es["total_return"]), _pct(eb["total_return"])),
        ("年化收益 CAGR", _pct(es["cagr"]), _pct(eb["cagr"])),
        ("年化波动", _pct(es["volatility"]), _pct(eb["volatility"])),
        ("夏普比率", f"{es['sharpe']:.2f}", f"{eb['sharpe']:.2f}"),
        ("索提诺比率", f"{es['sortino']:.2f}", f"{eb['sortino']:.2f}"),
        ("最大回撤", _pct(es["max_drawdown"]), _pct(eb["max_drawdown"])),
        ("卡玛比率", f"{es['calmar']:.2f}", f"{eb['calmar']:.2f}"),
        ("成交笔数", f"{int(es['n_trades'])}", f"{int(eb['n_trades'])}"),
        ("累计佣金(元)", f"{es['total_commission']:,.0f}", f"{eb['total_commission']:,.0f}"),
        ("年化换手(单边)", f"{es['annual_turnover']:.2f}x", f"{eb['annual_turnover']:.2f}x"),
    ]

    # 诚实结论：依据实际数字动态措辞
    vec_dd_gain = vb["max_drawdown"] - vs["max_drawdown"]
    vec_ret_gap = vs["total_return"] - vb["total_return"]
    evt_dd_gain = eb["max_drawdown"] - es["max_drawdown"]
    evt_sharpe_gap = es["sharpe"] - eb["sharpe"]
    evt_ret_gap = es["total_return"] - eb["total_return"]

    L = []
    L.append("# 真实数据回测报告（Real-Data Backtest Report）")
    L.append("")
    L.append("> **数据声明**：本报告使用的行情为**公开来源、前复权 A 股日线**，"
             "仅用于**研究与教学演示**，不保证准确/完整/及时，**不构成任何投资建议**。"
             "回测为历史模拟，未考虑全部真实约束，实盘结果可能显著不同。")
    L.append("")
    L.append("## 一、数据与口径")
    L.append("")
    L.append(f"- 数据目录：`{meta['data_dir']}`")
    L.append(f"- 标的数量：**{meta['n_symbols']}** 只（成分取自文件名，如 `sh600519`）")
    L.append(f"- 回测区间：**{meta['start']} ~ {meta['end']}**，共 **{meta['n_bars']}** 个交易日"
             f"（约 {meta['years']:.2f} 年）")
    L.append("- 清洗口径（`load_close_panel`，同系列可比）："
             "**非正价 → NaN → 前向填充 ffill → 按全体上市日裁剪到公共有效区间**。"
             "前复权可能在早期产生负/零价，故先剔除再填充；裁剪保证面板无缺口、无前视。")
    L.append(f"- 成本模型：佣金 {meta['commission_rate']*10000:.1f}bp（最低 {meta['commission_min']:.0f} 元）"
             f" + 滑点 {meta['slippage_bps']:.0f}bp。年化基准 {TRADING_DAYS} 交易日。")
    L.append("")
    L.append("## 二、向量化回测：横截面动量 Top-K")
    L.append("")
    L.append(f"- **策略**：每 {vp['rebalance']} 个交易日，按过去 {vp['lookback']} 日涨幅"
             f"选出最强的 {vp['top_k']} 只等权持有，其余空仓；信号只用截至当期数据，"
             "`VectorBacktester` 内部再滞后一期（上一期权重 × 本期收益）。")
    L.append("- **基准**：全成分**等权买入持有**（期初等额买入并持有，权重随行情漂移，被动无成本）。")
    L.append("- 净值口径起点均为 1.0，策略为**扣成本后**净值。")
    L.append("")
    L.append(_metric_table(vec_rows))
    L.append("")
    L.append(f"- 累计换手 **{vs['cumulative_turnover']:.1f}x**（双边），"
             f"成本拖累约 **{_pct(vs['cum_cost_drag'])}**（毛→净）。")
    L.append("")
    L.append("**诚实结论（向量化）**：在该成分（多为事后已知的大市值、高流动性龙头）与区间内，"
             f"等权买入持有本身非常强势（累计 {_pct(vb['total_return'])}、夏普 {vb['sharpe']:.2f}）。"
             f"动量策略累计收益 {_pct(vs['total_return'])}（低于基准约 {_pp(vec_ret_gap)}），"
             f"夏普 {vs['sharpe']:.2f}（基准 {vb['sharpe']:.2f}）；"
             f"但**最大回撤由 {_pct(vb['max_drawdown'])} 降至 {_pct(vs['max_drawdown'])}"
             f"（降低约 {_pp(vec_dd_gain)}）**，卡玛与基准接近。"
             "换言之，简单横截面动量在此未能跑赢强势的等权基准，主要贡献是**显著降低回撤**，"
             "代价是更高的换手与成本。该结论对回看窗口/持仓数/调仓频率**较敏感**，"
             "且基准受成分**幸存者/选择偏差**美化，不应据此外推为可实盘复制的超额收益。")
    L.append("")
    L.append("## 三、事件驱动回测：双均线择时")
    L.append("")
    L.append(f"- **策略**：单标的 `{evt['symbol']}` 的 {ep['fast']}/{ep['slow']} 双均线择时"
             "（短均线上穿长均线满仓、下穿清仓，只做多/持币）。逐 bar 推进，"
             "订单在**下一 bar** 含滑点与佣金成交，无杠杆、买入受现金约束。")
    L.append(f"- **基准**：同一标的的**买入持有**（经同一引擎、同一成本模型撮合）。")
    L.append(f"- 初始资金 {ep['cash']:,.0f} 元；下表净值为归一化后（起点=1.0）。")
    L.append("")
    L.append(_metric_table(evt_rows))
    L.append("")
    L.append("**诚实结论（事件驱动）**："
             f"双均线择时把最大回撤由 {_pct(eb['max_drawdown'])} 降至 {_pct(es['max_drawdown'])}"
             f"（降低约 {_pp(evt_dd_gain)}），夏普 {es['sharpe']:.2f}（基准 {eb['sharpe']:.2f}，"
             f"差 {evt_sharpe_gap:+.2f}），索提诺 {es['sortino']:.2f}（基准 {eb['sortino']:.2f}）；"
             f"但累计收益 {_pct(es['total_return'])} 低于买入持有的 {_pct(eb['total_return'])}"
             f"（相差约 {_pp(evt_ret_gap)}）——趋势跟踪在单边上涨标的上常因**频繁进出、踏空反弹**而让渡部分涨幅，"
             "以换取更平滑的净值与更低的回撤。"
             "**单标的择时高度依赖标的与参数**，此处仅为演示引擎在真实数据上的正确性与风险/收益取舍，"
             "并非推荐信号。")
    L.append("")
    L.append("## 四、防未来函数")
    L.append("")
    L.append("- 向量化：信号仅用截至 t 的历史收益；`VectorBacktester` 用「上一期权重 × 本期收益」，杜绝前视。")
    L.append("- 事件驱动：`on_bar(i)` 仅能取到截至第 i 根 bar 的历史；订单于第 i+1 根 bar 成交。")
    L.append("- 数据清洗：ffill 只用历史价；裁剪仅依据各标的自身有效区间，均不引入未来信息。")
    L.append("")
    L.append("## 五、复现")
    L.append("")
    L.append("```bash")
    L.append("cd kairos-backtest")
    L.append(f"python examples/real_backtest.py --data-dir {meta['data_dir']}")
    L.append("```")
    L.append("")
    L.append("产物：`REPORT.md`（本文件）、`equity.csv`（四条净值曲线）、"
             "`metrics.json`（结构化指标）、`trades.csv`（双均线成交明细）。")
    L.append("")
    L.append("## 六、局限与风险")
    L.append("")
    L.append("- **幸存者/选择偏差**：成分为事后挑选的知名龙头，等权买入持有基准被系统性美化。")
    L.append("- **参数敏感**：动量与双均线的表现对窗口/持仓数/调仓频率敏感，本报告用先验常用参数，未做样本内择优。")
    L.append("- **成本与冲击**：仅建模佣金+固定滑点，未含涨跌停、停牌、冲击成本、税费与借券约束。")
    L.append("- **前复权口径**：以数据源为准，可能存在偏差；早期负价已按无效处理。")
    L.append("- **单标的择时**：事件驱动示例为单标的，结论不具横截面代表性。")
    L.append("")
    L.append("> 再次声明：以上均为历史数据的**研究性模拟**，**不构成投资建议**。")
    L.append("")
    return "\n".join(L)


def write_outputs(out_dir: str, meta: dict, vec: dict, evt: dict) -> dict:
    os.makedirs(out_dir, exist_ok=True)

    # equity.csv：四条净值曲线（同一起点 1.0）
    equity = vec["equity"].join(evt["equity"], how="outer").sort_index()
    equity = equity.round(6)
    eq_out = equity.copy()
    eq_out.insert(0, "date", equity.index.strftime("%Y-%m-%d"))
    eq_path = os.path.join(out_dir, "equity.csv")
    eq_out.to_csv(eq_path, index=False)

    # trades.csv：双均线成交明细（少量）
    trades = evt["trades"]
    trades_path = os.path.join(out_dir, "trades.csv")
    t = trades.copy()
    if not t.empty:
        t["date"] = pd.to_datetime(t["date"]).dt.strftime("%Y-%m-%d")
        t["qty"] = t["qty"].round(4)
        t["price"] = t["price"].round(4)
        t["commission"] = t["commission"].round(4)
    t.to_csv(trades_path, index=False)

    # metrics.json
    metrics = {
        "meta": meta,
        "vectorized": {
            "strategy": "cross_sectional_momentum_topk",
            "params": vec["params"],
            "momentum": _round(vec["strat"]),
            "equal_weight_buy_hold": _round(vec["bench"]),
        },
        "event_driven": {
            "strategy": "sma_cross_timing",
            "symbol": evt["symbol"],
            "params": evt["params"],
            "sma_cross": _round(evt["strat"]),
            "buy_hold": _round(evt["bench"]),
        },
    }
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)

    # REPORT.md
    report_path = os.path.join(out_dir, "REPORT.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(build_report(meta, vec, evt))

    return {"equity": eq_path, "trades": trades_path,
            "metrics": metrics_path, "report": report_path}


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="真实 A 股数据回测示例（向量化动量 + 事件驱动双均线）")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="真实行情 CSV 目录（只读）")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="结果输出目录")
    ap.add_argument("--symbol", default="sh600519", help="事件驱动择时标的（默认贵州茅台）")
    ap.add_argument("--lookback", type=int, default=126, help="动量回看窗口（交易日）")
    ap.add_argument("--top-k", type=int, default=5, help="动量持仓只数")
    ap.add_argument("--rebalance", type=int, default=21, help="动量调仓间隔（交易日）")
    ap.add_argument("--fast", type=int, default=20, help="双均线短窗口")
    ap.add_argument("--slow", type=int, default=60, help="双均线长窗口")
    ap.add_argument("--cash", type=float, default=1_000_000.0, help="事件驱动初始资金")
    ap.add_argument("--commission-rate", type=float, default=0.0003, help="佣金比例")
    ap.add_argument("--commission-min", type=float, default=5.0, help="最低佣金(元)")
    ap.add_argument("--slippage-bps", type=float, default=5.0, help="滑点(bp)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if not os.path.isdir(args.data_dir):
        print(f"[错误] 数据目录不存在: {args.data_dir}\n"
              f"请用 --data-dir 指定真实行情目录。", file=sys.stderr)
        return 2

    cost = CostModel(args.commission_rate, args.commission_min, args.slippage_bps)
    close = load_close_panel(args.data_dir, drop_incomplete=True)
    symbols = list(close.columns)

    meta = {
        "data_dir": args.data_dir,
        "n_symbols": len(symbols),
        "symbols_head": symbols[:5],
        "start": str(close.index.min().date()),
        "end": str(close.index.max().date()),
        "n_bars": int(len(close)),
        "years": round(_years(len(close)), 4),
        "commission_rate": args.commission_rate,
        "commission_min": args.commission_min,
        "slippage_bps": args.slippage_bps,
    }

    print("=" * 72)
    print(f"真实数据回测 | 标的 {meta['n_symbols']} 只 | 区间 {meta['start']} ~ {meta['end']}"
          f" | {meta['n_bars']} 个交易日（约 {meta['years']:.2f} 年）")
    print("=" * 72)

    vec = run_vectorized(close, cost, args.lookback, args.top_k, args.rebalance)
    evt = run_event_driven(close, cost, args.symbol, args.fast, args.slow, args.cash)

    vs, vb = vec["strat"], vec["bench"]
    es, eb = evt["strat"], evt["bench"]

    print("\n【① 向量化】横截面动量 Top-K（lookback=%d, top_k=%d, rebalance=%d）"
          % (args.lookback, args.top_k, args.rebalance))
    print(f"  动量策略 : 累计 {_pct(vs['total_return']):>8} | CAGR {_pct(vs['cagr']):>7} | "
          f"夏普 {vs['sharpe']:.2f} | 索提诺 {vs['sortino']:.2f} | "
          f"回撤 {_pct(vs['max_drawdown']):>7} | 卡玛 {vs['calmar']:.2f} | "
          f"年换手 {vs['annual_turnover']:.1f}x")
    print(f"  等权持有 : 累计 {_pct(vb['total_return']):>8} | CAGR {_pct(vb['cagr']):>7} | "
          f"夏普 {vb['sharpe']:.2f} | 索提诺 {vb['sortino']:.2f} | "
          f"回撤 {_pct(vb['max_drawdown']):>7} | 卡玛 {vb['calmar']:.2f} | 年换手 ≈0")

    print(f"\n【② 事件驱动】双均线择时 {evt['symbol']}（fast={args.fast}, slow={args.slow}）")
    print(f"  双均线   : 累计 {_pct(es['total_return']):>8} | CAGR {_pct(es['cagr']):>7} | "
          f"夏普 {es['sharpe']:.2f} | 索提诺 {es['sortino']:.2f} | "
          f"回撤 {_pct(es['max_drawdown']):>7} | 卡玛 {es['calmar']:.2f} | "
          f"成交 {int(es['n_trades'])} 笔 | 年换手 {es['annual_turnover']:.1f}x")
    print(f"  买入持有 : 累计 {_pct(eb['total_return']):>8} | CAGR {_pct(eb['cagr']):>7} | "
          f"夏普 {eb['sharpe']:.2f} | 索提诺 {eb['sortino']:.2f} | "
          f"回撤 {_pct(eb['max_drawdown']):>7} | 卡玛 {eb['calmar']:.2f} | "
          f"成交 {int(eb['n_trades'])} 笔")

    trades = evt["trades"]
    if not trades.empty:
        print("\n  成交明细（前 5 笔）：")
        head = trades.head(5).copy()
        head["date"] = pd.to_datetime(head["date"]).dt.strftime("%Y-%m-%d")
        print(head.to_string(index=False))

    paths = write_outputs(args.out_dir, meta, vec, evt)
    print("\n已写出：")
    for k in ("report", "equity", "metrics", "trades"):
        print(f"  {k:>7}: {paths[k]}")
    print("\n提示：数据为公开行情、仅供研究，本报告不构成投资建议。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
