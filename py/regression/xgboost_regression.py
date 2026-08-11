#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""XGBoost 回归最小可执行模板。

使用 xgboost.XGBRegressor 完成：
    数据划分 -> 模型训练 -> 回归评估 -> 特征重要性 -> 新样本预测

XGBoost 不要求标准化，但 X 必须是数值特征矩阵；比赛时主要替换 main()
中的 X、y 和模型参数。
"""

import sys

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


def train_xgboost_regressor(
    X,
    y,
    *,
    test_size=0.2,
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
):
    """训练 XGBRegressor，并返回模型和测试集结果。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("X 必须是二维特征矩阵，且样本数必须与 y 一致")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("X 和 y 不能包含 NaN 或无穷值")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective="reg:squarederror",
        eval_metric="rmse",
        n_jobs=1,
        random_state=random_state,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    result = {
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mse,
        "rmse": np.sqrt(mse),
        "r2": r2_score(y_test, y_pred),
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "feature_importances": model.feature_importances_,
    }
    return model, result


def main():
    # ====== 比赛时主要替换下面这部分 ======
    X, y = load_diabetes(return_X_y=True)
    model, result = train_xgboost_regressor(X, y)

    print(f"MAE:  {result['mae']:.4f}")
    print(f"MSE:  {result['mse']:.4f}")
    print(f"RMSE: {result['rmse']:.4f}")
    print(f"R²:   {result['r2']:.4f}")
    print("特征重要性:", np.round(result["feature_importances"], 6))
    print("示例预测:", np.round(model.predict(X[:3]), 4))

    if "--csv" in sys.argv:
        np.savetxt(
            "xgboost_regression_out.csv",
            np.column_stack((result["y_test"], result["y_pred"])),
            delimiter=",",
            fmt="%.10f",
            header="真实值,预测值",
            comments="",
        )
        print("已保存 -> xgboost_regression_out.csv")


if __name__ == "__main__":
    main()
