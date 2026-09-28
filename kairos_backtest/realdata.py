"""真实行情数据加载器（离线、纯 numpy/pandas）。

从目录下的 ``*.csv`` 日线行情文件加载收盘价，拼装成 ``date x symbol`` 的价格面板，
供 :class:`~kairos_backtest.vectorized.VectorBacktester` 与
:class:`~kairos_backtest.engine.BacktestEngine` 直接消费。

每个 CSV 约定至少包含 ``date`` 与 ``close`` 两列（形如
``date,open,high,low,close,volume``），标的代码取自文件名（去扩展名）。

统一口径（消除前复权/停牌带来的脏数据，保证同系列可比）：

1. **非正价 → NaN**：前复权可能在早期产生 ``<=0`` 的价格，视为无效。
2. **前向填充 ffill**：衔接停牌或缺失的交易日，用最近一个有效价补齐。
3. **按全体上市日裁剪**（``drop_incomplete=True``）：只保留「所有标的都已有
   有效价」的公共区间——起点取各标的首个有效日的最大值，终点取各标的末个
   有效日的最小值，得到一个无缺口、同口径的平衡面板。

以上处理均不引入任何未来信息：ffill 只用历史价，裁剪只依据各标的自身的有效区间。
"""
from __future__ import annotations

import glob
import os
from typing import List, Optional

import numpy as np
import pandas as pd

__all__ = ["load_close_panel", "list_symbols"]


def _read_one(path: str, price_col: str, date_col: str) -> pd.Series:
    """读取单个 CSV，返回以日期为索引、升序去重后的价格序列。"""
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if date_col.lower() not in cols or price_col.lower() not in cols:
        raise ValueError(
            f"{os.path.basename(path)} 缺少必需列 '{date_col}' 或 '{price_col}'，"
            f"实际列为 {list(df.columns)}"
        )
    dcol, pcol = cols[date_col.lower()], cols[price_col.lower()]
    s = df[[dcol, pcol]].copy()
    s[dcol] = pd.to_datetime(s[dcol])
    s = s.set_index(dcol)[pcol].astype("float64")
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s.name = os.path.splitext(os.path.basename(path))[0]
    return s


def list_symbols(data_dir: str, pattern: str = "*.csv") -> List[str]:
    """列出目录下可用的标的代码（文件名去扩展名），按字典序排序。"""
    files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    return [os.path.splitext(os.path.basename(f))[0] for f in files]


def load_close_panel(data_dir: str,
                     drop_incomplete: bool = True,
                     price_col: str = "close",
                     date_col: str = "date",
                     pattern: str = "*.csv",
                     symbols: Optional[List[str]] = None) -> pd.DataFrame:
    """加载 ``data_dir`` 下的日线行情，返回收盘价面板。

    参数
    ----
    data_dir:        存放 ``*.csv`` 行情文件的目录。
    drop_incomplete: 为 True（默认）时，按「全体标的均有有效价」的公共区间裁剪，
                     得到同系列口径的平衡面板；为 False 时保留完整历史跨度，
                     未上市/无有效价的早期时点留为 NaN。
    price_col:       价格列名，默认 ``close``。
    date_col:        日期列名，默认 ``date``。
    pattern:         文件匹配模式，默认 ``*.csv``。
    symbols:         可选，仅加载指定标的（按文件名去扩展名匹配）。

    返回
    ----
    ``pandas.DataFrame``，index 为升序 :class:`~pandas.DatetimeIndex`，
    columns 为标的代码，值为清洗后的收盘价。

    处理口径见模块 docstring：非正价→NaN、ffill、按全体上市日裁剪。
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"目录 {data_dir} 下未找到匹配 {pattern} 的行情文件")

    want = set(symbols) if symbols else None
    series = {}
    for f in files:
        sym = os.path.splitext(os.path.basename(f))[0]
        if want is not None and sym not in want:
            continue
        s = _read_one(f, price_col=price_col, date_col=date_col)
        series[s.name] = s

    if not series:
        raise ValueError(f"目录 {data_dir} 中没有匹配到任何标的（symbols={symbols}）")

    panel = pd.DataFrame(series).sort_index()

    # 1) 非正价（含前复权产生的负值/零）视为无效 → NaN
    panel = panel.where(panel > 0)
    # 2) 前向填充，衔接停牌/缺失交易日（仅用历史价，无未来信息）
    panel = panel.ffill()

    # 丢弃清洗后仍全为 NaN 的标的（自始至终无有效价）
    panel = panel.dropna(axis=1, how="all")
    if panel.shape[1] == 0:
        raise ValueError("所有标的清洗后均无有效价格，无法构建面板")

    # 3) 按全体上市日裁剪到公共有效区间
    if drop_incomplete:
        first_valid = [d for d in panel.apply(lambda c: c.first_valid_index()) if d is not None]
        last_valid = [d for d in panel.apply(lambda c: c.last_valid_index()) if d is not None]
        if not first_valid or not last_valid:
            raise ValueError("无法确定公共有效区间")
        start, end = max(first_valid), min(last_valid)
        panel = panel.loc[start:end]

    return panel
