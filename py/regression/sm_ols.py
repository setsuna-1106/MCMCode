"""statsmodels OLS 回归模板。"""

from __future__ import annotations

import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

try:
    from py.regression.sm_diagnostics import diagnose_ols
except ModuleNotFoundError:
    from sm_diagnostics import diagnose_ols


def fit_ols(
    X,
    y,
    *,
    test_size=0.2,
    random_state=42,
    robust=False,
):
    """按时间无关的随机切分拟合 OLS 并返回测试集指标。

    Args:
        X, y: 特征矩阵和连续型目标值。
        robust: 是否使用 HC3 稳健协方差。

    Returns:
        包含模型、训练/测试数据、预测值和 MAE/MSE/RMSE/R2 的字典。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("X must be a 2D feature matrix and match y in length")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    # statsmodels 不会自动补截距列，训练和预测必须使用相同的列结构。
    X_train_sm = sm.add_constant(X_train, has_constant="add")
    X_test_sm = sm.add_constant(X_test, has_constant="add")

    model = sm.OLS(y_train, X_train_sm)
    result = model.fit(cov_type="HC3") if robust else model.fit()
    y_pred = np.asarray(result.predict(X_test_sm))
    mse = mean_squared_error(y_test, y_pred)

    return {
        "model": result,
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mse,
        "rmse": np.sqrt(mse),
        "r2": r2_score(y_test, y_pred),
    }


def predict_ols(result, X_new):
    """使用已拟合的 OLS 结果预测新样本。"""
    X_new = np.asarray(X_new, dtype=float)
    if X_new.ndim == 1:
        X_new = X_new.reshape(1, -1)
    return np.asarray(result.predict(sm.add_constant(X_new, has_constant="add")))


if __name__ == "__main__":
    # Replace this block with X = df[features].to_numpy(), y = df[target].to_numpy().
    from sklearn.datasets import load_diabetes

    X, y = load_diabetes(return_X_y=True)
    output = fit_ols(X, y, robust=True)
    print(f"MAE:  {output['mae']:.4f}")
    print(f"RMSE: {output['rmse']:.4f}")
    print(f"R2:   {output['r2']:.4f}")
    print("coefficients:", output["model"].params)
    print("p-values:", output["model"].pvalues)
    print("diagnostics:", diagnose_ols(output["model"], output["X_train"]))
