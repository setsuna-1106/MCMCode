"""Lightweight scipy curve_fit template."""

import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def _fit(model_func, x_train, x_test, y_train, y_test, p0, bounds, maxfev):
    params, _ = curve_fit(
        model_func,
        x_train,
        y_train,
        p0=p0,
        bounds=bounds,
        maxfev=maxfev,
    )
    y_pred = np.asarray(model_func(x_test, *params))
    return params, y_test, y_pred, np.sqrt(mean_squared_error(y_test, y_pred))


def fit_curve(
    model_func,
    x,
    y,
    *,
    p0=None,
    bounds=(-np.inf, np.inf),
    test_size=0.2,
    random_state=42,
    maxfev=10000,
):
    """拟合单输入任意函数 ``y = model_func(x, *params)``。

    Args:
        model_func: 第一个参数为 x、其余参数为待估系数的函数。
        x, y: 一维观测数据。
        p0: 参数初值；``bounds`` 为参数边界。

    Returns:
        包含拟合参数、测试集预测值和 RMSE 的字典。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    # 只用训练子集估计参数，再在测试子集上计算 RMSE。
    params, y_test, y_pred, rmse = _fit(
        model_func, x_train, x_test, y_train, y_test, p0, bounds, maxfev
    )
    return {"params": params, "y_test": y_test, "y_pred": y_pred, "rmse": rmse}


def fit_curve_multi(
    model_func,
    X,
    y,
    *,
    p0=None,
    bounds=(-np.inf, np.inf),
    test_size=0.2,
    random_state=42,
    maxfev=10000,
):
    """拟合多输入任意函数 ``y = model_func(X.T, *params)``。

    Args:
        model_func: 接收 ``(特征数, 样本数)`` 输入的模型函数。
        X: ``(样本数, 特征数)`` 特征矩阵。
        y: 一维目标值。

    Returns:
        包含拟合参数、测试集预测值和 RMSE 的字典。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    params, y_test, y_pred, rmse = _fit(
        model_func,
        X_train.T,
        X_test.T,
        y_train,
        y_test,
        p0,
        bounds,
        maxfev,
    )
    return {"params": params, "y_test": y_test, "y_pred": y_pred, "rmse": rmse}


def predict_curve(model_func, x_new, params):
    """用单输入拟合函数预测新样本。"""
    return np.asarray(model_func(np.asarray(x_new, dtype=float), *params))


def predict_curve_multi(model_func, X_new, params):
    """用多输入拟合函数预测 ``(样本数, 特征数)`` 新样本。"""
    X_new = np.asarray(X_new, dtype=float)
    if X_new.ndim == 1:
        X_new = X_new.reshape(1, -1)
    return np.asarray(model_func(X_new.T, *params))


if __name__ == "__main__":
    # Replace model_func, x, and y with the problem data.
    def model_func(x, a, b, c):
        return a * np.exp(-b * x) + c

    rng = np.random.default_rng(42)
    x = np.linspace(0, 5, 100)
    y = model_func(x, 8, 0.7, 1.5) + rng.normal(0, 0.2, len(x))
    result = fit_curve(model_func, x, y, p0=(7, 0.5, 1))
    print("params:", result["params"])
    print("RMSE:", result["rmse"])
