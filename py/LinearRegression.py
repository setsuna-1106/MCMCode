"""线性回归模板。"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit_linear_regression(X, y, test_size=0.2, random_state=42, standardize=True):
    """训练线性回归并返回模型、预测值和常用评价指标。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("X 必须是二维特征矩阵，且样本数必须与 y 一致")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if standardize:
        model = make_pipeline(StandardScaler(), LinearRegression())
        regressor = model.named_steps["linearregression"]
    else:
        model = LinearRegression()
        regressor = model

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    return {
        "model": model,
        "y_test": y_test,
        "y_pred": y_pred,
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mse,
        "rmse": np.sqrt(mse),
        "r2": r2_score(y_test, y_pred),
        "coef": regressor.coef_,
        "intercept": regressor.intercept_,
    }


if __name__ == "__main__":
    # 替换为自己的数据：X 为特征，y 为连续型目标值。
    from sklearn.datasets import load_diabetes

    X, y = load_diabetes(return_X_y=True)
    result = fit_linear_regression(X, y, standardize=True)

    print(f"MAE:  {result['mae']:.4f}")
    print(f"MSE:  {result['mse']:.4f}")
    print(f"RMSE: {result['rmse']:.4f}")
    print(f"R²:   {result['r2']:.4f}")
    print("系数:", result["coef"])
    print("截距:", result["intercept"])
    # 新数据预测：result["model"].predict(X_new)
