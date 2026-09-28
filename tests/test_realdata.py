"""离线测试 kairos_backtest.realdata.load_close_panel。

全部用 tmp_path 现造小 CSV，不联网、不依赖真实数据目录。
"""
import os

import numpy as np
import pandas as pd
import pytest

from kairos_backtest.realdata import list_symbols, load_close_panel

COLS = ["date", "open", "high", "low", "close", "volume"]


def _write_csv(dirpath, name, dates, closes):
    """按真实行情格式（date,open,high,low,close,volume）写一个小 CSV。"""
    df = pd.DataFrame({
        "date": dates,
        "open": closes,
        "high": closes,
        "low": closes,
        "close": closes,
        "volume": [1000.0] * len(closes),
    })[COLS]
    path = os.path.join(str(dirpath), f"{name}.csv")
    df.to_csv(path, index=False)
    return path


def test_symbol_names_from_filenames(tmp_path):
    _write_csv(tmp_path, "aaa", ["2020-01-01", "2020-01-02"], [10.0, 11.0])
    _write_csv(tmp_path, "bbb", ["2020-01-01", "2020-01-02"], [20.0, 21.0])
    assert list_symbols(str(tmp_path)) == ["aaa", "bbb"]
    panel = load_close_panel(str(tmp_path))
    assert list(panel.columns) == ["aaa", "bbb"]
    assert isinstance(panel.index, pd.DatetimeIndex)


def test_basic_panel_values_aligned(tmp_path):
    dates = ["2020-01-01", "2020-01-02", "2020-01-03"]
    _write_csv(tmp_path, "a", dates, [1.0, 2.0, 3.0])
    _write_csv(tmp_path, "b", dates, [10.0, 20.0, 30.0])
    panel = load_close_panel(str(tmp_path))
    assert panel.shape == (3, 2)
    assert panel["a"].tolist() == [1.0, 2.0, 3.0]
    assert panel["b"].tolist() == [10.0, 20.0, 30.0]
    assert not panel.isna().any().any()


def test_nonpositive_becomes_nan_then_ffilled(tmp_path):
    dates = ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]
    # 前复权可能产生负价/零价，应视为无效并用最近有效价前向填充
    _write_csv(tmp_path, "a", dates, [100.0, -5.0, 0.0, 110.0])
    panel = load_close_panel(str(tmp_path))
    assert panel["a"].tolist() == [100.0, 100.0, 100.0, 110.0]


def test_suspension_gap_is_ffilled(tmp_path):
    # a 在 01-03 停牌（缺该日），面板对齐后应被 ffill 补齐
    _write_csv(tmp_path, "a", ["2020-01-01", "2020-01-02", "2020-01-04"], [1.0, 2.0, 4.0])
    _write_csv(tmp_path, "b", ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
               [1.0, 1.0, 1.0, 1.0])
    panel = load_close_panel(str(tmp_path))
    assert list(panel.index.strftime("%Y-%m-%d")) == \
        ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]
    assert panel["a"].tolist() == [1.0, 2.0, 2.0, 4.0]  # 01-03 被 ffill 成 2.0


def test_drop_incomplete_trims_to_common_listing_window(tmp_path):
    # a 全程有效；b 从 01-03 才上市 -> 公共区间应从 01-03 起
    _write_csv(tmp_path, "a", ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
               [1.0, 2.0, 3.0, 4.0])
    _write_csv(tmp_path, "b", ["2020-01-03", "2020-01-04"], [30.0, 40.0])

    trimmed = load_close_panel(str(tmp_path), drop_incomplete=True)
    assert list(trimmed.index.strftime("%Y-%m-%d")) == ["2020-01-03", "2020-01-04"]
    assert not trimmed.isna().any().any()

    full = load_close_panel(str(tmp_path), drop_incomplete=False)
    assert len(full) == 4
    # b 上市前为 NaN（ffill 无法回填前导缺失）
    assert np.isnan(full["b"].iloc[0]) and np.isnan(full["b"].iloc[1])
    assert full["b"].iloc[2] == 30.0


def test_leading_nonpositive_trimmed_by_drop_incomplete(tmp_path):
    # a 前两天的前复权价为负 -> 首个有效价在 01-03；应据此裁剪公共区间起点
    _write_csv(tmp_path, "a", ["2020-01-01", "2020-01-02", "2020-01-03"], [-1.0, -2.0, 50.0])
    _write_csv(tmp_path, "b", ["2020-01-01", "2020-01-02", "2020-01-03"], [1.0, 2.0, 3.0])
    panel = load_close_panel(str(tmp_path), drop_incomplete=True)
    assert list(panel.index.strftime("%Y-%m-%d")) == ["2020-01-03"]
    assert panel["a"].iloc[0] == 50.0 and panel["b"].iloc[0] == 3.0


def test_symbols_filter(tmp_path):
    dates = ["2020-01-01", "2020-01-02"]
    _write_csv(tmp_path, "a", dates, [1.0, 2.0])
    _write_csv(tmp_path, "b", dates, [3.0, 4.0])
    _write_csv(tmp_path, "c", dates, [5.0, 6.0])
    panel = load_close_panel(str(tmp_path), symbols=["a", "c"])
    assert list(panel.columns) == ["a", "c"]


def test_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_close_panel(os.path.join(str(tmp_path), "nope"))


def test_empty_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_close_panel(str(tmp_path))


def test_missing_column_raises(tmp_path):
    # 缺少 close 列应报错
    df = pd.DataFrame({"date": ["2020-01-01"], "open": [1.0]})
    df.to_csv(os.path.join(str(tmp_path), "bad.csv"), index=False)
    with pytest.raises(ValueError):
        load_close_panel(str(tmp_path))


def test_panel_feeds_vector_backtester(tmp_path):
    """加载出的面板应能直接喂给 VectorBacktester（口径衔接验证）。"""
    from kairos_backtest import VectorBacktester, ZERO_COST

    dates = list(pd.bdate_range("2020-01-01", periods=6).strftime("%Y-%m-%d"))
    _write_csv(tmp_path, "a", dates, [100.0, 101, 102, 103, 104, 105])
    _write_csv(tmp_path, "b", dates, [100.0, 99, 98, 97, 96, 95])
    close = load_close_panel(str(tmp_path))
    w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    w["a"] = 1.0  # 始终满仓 a
    res = VectorBacktester(cost_rate=0.0).run(close, w)
    # 满仓单调上涨的 a，净值应递增
    assert res.equity.iloc[-1] > res.equity.iloc[0]
    assert res.returns.shape[0] == len(close)
