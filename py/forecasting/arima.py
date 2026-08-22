"""statsmodels ARIMA forecasting template."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA


def _mape(y_true, y_pred):
    # MAPE 对真实值为 0 的位置没有定义，因此只在非零样本上计算。
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def fit_arima(
    y,
    order=(1, 1, 1),
    *,
    test_size=0.2,
    trend=None,
):
    """按时间顺序拟合 ARIMA 并评估测试段。

    Args:
        y: 按时间从早到晚排列的一维序列。
        order: ``(p, d, q)`` 模型阶数。
        test_size: 测试点数量或测试比例。
        trend: 趋势项设置。

    Returns:
        包含模型、测试预测和 MAE/MSE/RMSE/MAPE 的字典。
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or len(y) < 8 or not np.isfinite(y).all():
        raise ValueError("y must be a 1D finite series with at least 8 values")
    if len(order) != 3 or any(
        not isinstance(value, (int, np.integer)) or value < 0 for value in order
    ):
        raise ValueError("order must be a tuple of three non-negative integers")

    if isinstance(test_size, (int, np.integer)):
        n_test = int(test_size)
    else:
        n_test = int(np.ceil(len(y) * float(test_size)))
    if n_test < 1 or n_test >= len(y):
        raise ValueError("test_size must leave at least one training value")

    # 时间序列按先后顺序切分，不能随机打乱，否则会用未来信息预测过去。
    split = len(y) - n_test
    y_train, y_test = y[:split], y[split:]
    # 先在历史训练段拟合，再对紧接着的测试段做多步预测。
    result = ARIMA(y_train, order=order, trend=trend).fit()
    forecast = np.asarray(result.forecast(steps=n_test))
    # 同时保留 MAE、MSE、RMSE 和 MAPE，便于按题目评价标准选用指标。
    mse = mean_squared_error(y_test, forecast)

    return {
        "model": result,
        "y_train": y_train,
        "y_test": y_test,
        "forecast": forecast,
        "mae": mean_absolute_error(y_test, forecast),
        "mse": mse,
        "rmse": np.sqrt(mse),
        "mape": _mape(y_test, forecast),
        "order": order,
    }


def predict_arima(result, steps=1):
    """用已拟合的 ARIMA 结果预测未来 ``steps`` 期。"""
    if not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError("steps must be a positive integer")
    return np.asarray(result.forecast(steps=int(steps)))


if __name__ == "__main__":
    # Replace y with a 1D series ordered from earliest to latest.
    rng = np.random.default_rng(42)
    t = np.linspace(0, 8 * np.pi, 100)
    y = 100 + np.linspace(0, 40, 100) + 10 * np.sin(t)
    y += rng.normal(0, 1, 100)

    output = fit_arima(y, order=(1, 1, 1), test_size=0.2)
    print(f"MAE:  {output['mae']:.4f}")
    print(f"RMSE: {output['rmse']:.4f}")
    print(f"MAPE: {output['mape']:.2f}%")
    print("test forecast:", output["forecast"])

    # Select order on the historical split, then refit on all data for future use.
    full_model = ARIMA(y, order=output["order"]).fit()
    print("future forecast:", predict_arima(full_model, steps=3))
