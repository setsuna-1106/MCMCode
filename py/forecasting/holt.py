"""Holt 二次指数平滑和 Holt-Winters 三次指数平滑模板。

二次指数平滑：水平项 + 趋势项。
三次指数平滑：水平项 + 趋势项 + 季节项。

比赛时主要替换 y、steps 和 seasonal_periods。
"""

import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt


def _validate_series(y, minimum_length):
    # 两种模型都按时间顺序接收一维、有限历史序列。
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or y.size < minimum_length or not np.isfinite(y).all():
        raise ValueError(
            f"y 必须是一维、至少包含 {minimum_length} 个有限数值"
        )
    return y


def _pack_result(result, steps, method):
    # 统一 Holt 与 Holt-Winters 的返回结构，调用者无需区分底层结果类型。
    forecast = np.asarray(result.forecast(steps=int(steps)), dtype=float)
    return {
        "model": result,
        "fitted": np.asarray(result.fittedvalues, dtype=float),
        "forecast": forecast,
        "method": method,
        "steps": int(steps),
    }


def fit_holt(
    y,
    steps=3,
    *,
    damped_trend=False,
    initialization_method="estimated",
):
    """拟合含水平项和趋势项的 Holt 模型。

    Args:
        y: 按时间排列的一维序列。
        steps: 未来预测期数。
        damped_trend: 是否让趋势随预测步数衰减。

    Returns:
        包含模型、历史拟合值、未来预测和方法名的字典。
    """
    y = _validate_series(y, minimum_length=3)
    if not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError("steps 必须是正整数")

    # Holt 同时估计水平项和趋势项；damped_trend 可让趋势逐步衰减。
    result = Holt(
        y,
        damped_trend=damped_trend,
        initialization_method=initialization_method,
    ).fit(optimized=True)
    return _pack_result(result, steps, method="Holt")


def fit_holt_winters(
    y,
    seasonal_periods,
    steps=3,
    *,
    trend="add",
    seasonal="add",
    damped_trend=False,
    initialization_method="estimated",
):
    """拟合含水平、趋势和季节项的 Holt-Winters 模型。

    Args:
        y: 至少包含两个完整季节周期的一维序列。
        seasonal_periods: 季节周期长度。
        steps: 未来预测期数。
        trend, seasonal: 趋势项和季节项的加法或乘法形式。

    Returns:
        包含模型、历史拟合值、未来预测和方法名的字典。
    """
    if not isinstance(seasonal_periods, (int, np.integer)) or seasonal_periods < 2:
        raise ValueError("seasonal_periods 必须是不小于 2 的整数")
    if trend not in (None, "add", "mul"):
        raise ValueError("trend 必须是 None、'add' 或 'mul'")
    if seasonal not in ("add", "mul"):
        raise ValueError("seasonal 必须是 'add' 或 'mul'")
    if trend is None and damped_trend:
        raise ValueError("damped_trend=True 时必须设置 trend")
    if not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError("steps 必须是正整数")

    # 至少需要两个完整季节周期，才能识别季节项的重复模式。
    y = _validate_series(y, minimum_length=2 * int(seasonal_periods))
    if (trend == "mul" or seasonal == "mul") and np.any(y <= 0):
        raise ValueError("乘法趋势或乘法季节项要求 y 全部大于 0")

    # Holt-Winters 在水平、趋势之外再估计季节项；乘法项要求数据为正。
    result = ExponentialSmoothing(
        y,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=int(seasonal_periods),
        damped_trend=damped_trend,
        initialization_method=initialization_method,
    ).fit(optimized=True)
    return _pack_result(result, steps, method="Holt-Winters")


def main():
    # ====== 比赛时主要替换下面这部分 ======
    y = np.array(
        [100, 106, 111, 104, 108, 114, 120, 112, 117, 123, 129, 121],
        dtype=float,
    )
    steps = 4

    double_output = fit_holt(y, steps=steps, damped_trend=True)
    triple_output = fit_holt_winters(
        y,
        seasonal_periods=4,
        steps=steps,
        trend="add",
        seasonal="add",
    )

    print("Holt 拟合值:", np.round(double_output["fitted"], 4))
    print("Holt 未来预测:", np.round(double_output["forecast"], 4))
    print("Holt-Winters 拟合值:", np.round(triple_output["fitted"], 4))
    print("Holt-Winters 未来预测:", np.round(triple_output["forecast"], 4))

    assert double_output["forecast"].shape == (steps,)
    assert triple_output["forecast"].shape == (steps,)
    assert np.isfinite(double_output["forecast"]).all()
    assert np.isfinite(triple_output["forecast"]).all()


if __name__ == "__main__":
    main()
