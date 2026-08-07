"""多项式回归模板。"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def fit_polynomial_regression(
    X, y, degree=2, test_size=0.2, random_state=42, standardize=True
):
    """训练多项式回归并返回模型、预测值和常用评价指标。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("X 必须是二维特征矩阵，且样本数必须与 y 一致")
    if not isinstance(degree, int) or degree < 1:
        raise ValueError("degree 必须是大于等于 1 的整数")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler() if standardize else "passthrough"),
        ("regressor", LinearRegression()),
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    regressor = model.named_steps["regressor"]
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
        "feature_names": model.named_steps["poly"].get_feature_names_out(),
    }


if __name__ == "__main__":
    # 替换为自己的数据：X 为特征，y 为连续型目标值。
    rng = np.random.default_rng(42)
    X = np.linspace(-3, 3, 80).reshape(-1, 1)
    y = 2 * X[:, 0] ** 2 - 3 * X[:, 0] + 4 + rng.normal(0, 1, len(X))

    result = fit_polynomial_regression(X, y, degree=2, standardize=True)

    print(f"MAE:  {result['mae']:.4f}")
    print(f"MSE:  {result['mse']:.4f}")
    print(f"RMSE: {result['rmse']:.4f}")
    print(f"R²:   {result['r2']:.4f}")
    print("多项式特征:", result["feature_names"])
    print("系数:", result["coef"])
    print("截距:", result["intercept"])
    # 新数据预测：result["model"].predict(X_new)
