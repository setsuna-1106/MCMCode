#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scipy.interpolate 一维插值最小可执行模板。

支持三种常用方法：
    linear: 分段线性插值，稳定且不容易过冲。
    cubic:  三次样条插值，曲线平滑，但可能出现过冲。
    pchip:  保形三次插值，适合要求保持单调性的工程数据。

默认不对观测区间外的数据外推，越界结果为 NaN；只有确认外推合理时，
才设置 extrapolate=True。

验收锚点：对 y=x^2 的观测点做 cubic 插值，在 x=1.5 处结果约为 2.25。
"""

import sys

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator, interp1d


def _validate_xy(x, y):
    """校验观测点，并返回严格递增的浮点数组。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size < 2:
        raise ValueError("x 和 y 必须是一维，且至少包含 2 个观测点")
    if x.size != y.size:
        raise ValueError("x 和 y 的长度必须相同")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("x 和 y 只能包含有限数值")
    if np.any(np.diff(x) <= 0):
        raise ValueError("x 必须严格递增，不能有重复横坐标")
    return x, y


def make_interpolator(x, y, *, kind="cubic", extrapolate=False):
    """根据观测点构造一维插值函数。

    Args:
        x: 严格递增的观测横坐标。
        y: 与 ``x`` 等长的观测值。
        kind: ``linear``、``cubic`` 或 ``pchip``。
        extrapolate: 是否允许在观测区间外外推。

    Returns:
        可调用的 SciPy 插值器。

    Raises:
        ValueError: 观测点或插值方法不合法时抛出。
    """
    x, y = _validate_xy(x, y)
    # 区间外默认返回 NaN；只有确认外推合理时才打开该开关。
    if kind == "linear":
        fill_value = "extrapolate" if extrapolate else np.nan
        return interp1d(
            x,
            y,
            kind="linear",
            bounds_error=False,
            fill_value=fill_value,
            assume_sorted=True,
        )
    if kind == "cubic":
        return CubicSpline(x, y, extrapolate=extrapolate)
    if kind == "pchip":
        return PchipInterpolator(x, y, extrapolate=extrapolate)
    raise ValueError("kind 必须是 'linear'、'cubic' 或 'pchip'")


def interpolate_1d(x, y, x_new, *, kind="cubic", extrapolate=False):
    """在查询点计算一维插值。

    Args:
        x: 严格递增的观测横坐标。
        y: 与 ``x`` 等长的观测值。
        x_new: 待查询的横坐标序列。
        kind: 插值方法，传给 :func:`make_interpolator`。
        extrapolate: 是否允许观测区间外外推。

    Returns:
        与 ``x_new`` 形状相同的 NumPy 浮点数组。

    Raises:
        ValueError: 输入包含非有限值或观测点不合法时抛出。
    """
    x_new = np.asarray(x_new, dtype=float)
    if not np.all(np.isfinite(x_new)):
        raise ValueError("x_new 只能包含有限数值")
    interpolator = make_interpolator(
        x,
        y,
        kind=kind,
        extrapolate=extrapolate,
    )
    return np.asarray(interpolator(x_new), dtype=float)


def main():
    # ====== 比赛时主要替换下面这部分 ======
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([0.0, 1.0, 4.0, 9.0, 16.0])  # y = x^2 的观测值
    x_new = np.linspace(0.0, 4.0, 17)
    kind = "cubic"  # 可选：linear / cubic / pchip

    y_new = interpolate_1d(x, y, x_new, kind=kind)

    print("插值方法:", kind)
    print("查询点:", np.round(x_new, 4))
    print("插值结果:", np.round(y_new, 8))

    # 本例的简单验收：替换题目后可删除或改写。
    assert np.isclose(interpolate_1d(x, y, [1.5])[0], 2.25, atol=1e-10)
    assert np.isnan(interpolate_1d(x, y, [-1.0])[0])

    if "--csv" in sys.argv:
        np.savetxt(
            "scipy_interpolate_out.csv",
            np.column_stack((x_new, y_new)),
            delimiter=",",
            fmt="%.10f",
            header="x,interpolated_y",
            comments="",
        )
        print("已保存 -> scipy_interpolate_out.csv")


if __name__ == "__main__":
    main()
