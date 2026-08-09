"""scipy curve_fit template for arbitrary nonlinear functions."""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def _fit_and_score(
    model_func,
    x_train,
    x_test,
    y_train,
    y_test,
    *,
    p0,
    bounds,
    maxfev,
):
    params, covariance = curve_fit(
        model_func,
        x_train,
        y_train,
        p0=p0,
        bounds=bounds,
        maxfev=maxfev,
    )
    y_pred = np.asarray(model_func(x_test, *params))
    if y_pred.shape != y_test.shape:
        raise ValueError("model_func must return one prediction for each x")

    mse = mean_squared_error(y_test, y_pred)
    return {
        "model_func": model_func,
        "params": params,
        "param_std": np.sqrt(np.diag(covariance)),
        "covariance": covariance,
        "y_pred": y_pred,
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mse,
        "rmse": np.sqrt(mse),
        "r2": r2_score(y_test, y_pred),
    }


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
    """Fit y = model_func(x, *params) and return parameters and metrics."""
    if not callable(model_func):
        raise ValueError("model_func must be callable")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.ndim != 1 or len(x) != len(y):
        raise ValueError("x must be a 1D input and match y in length")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("x and y must contain only finite values")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    output = _fit_and_score(
        model_func,
        x_train,
        x_test,
        y_train,
        y_test,
        p0=p0,
        bounds=bounds,
        maxfev=maxfev,
    )
    output.update({"x_train": x_train, "x_test": x_test, "y_test": y_test})
    return output


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
    """Fit y = model_func(X.T, *params) for multiple input variables.

    X is supplied as (samples, features); model_func receives (features, samples).
    """
    if not callable(model_func):
        raise ValueError("model_func must be callable")

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("X must be (samples, features) and match y in length")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("X and y must contain only finite values")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    output = _fit_and_score(
        model_func,
        X_train.T,
        X_test.T,
        y_train,
        y_test,
        p0=p0,
        bounds=bounds,
        maxfev=maxfev,
    )
    output.update({"X_train": X_train, "X_test": X_test, "y_test": y_test})
    return output


def predict_curve(model_func, x_new, params):
    """Predict new x values from fitted parameters."""
    x_new = np.asarray(x_new, dtype=float)
    return np.asarray(model_func(x_new, *params))


def predict_curve_multi(model_func, X_new, params):
    """Predict new rows with a fitted multivariate curve."""
    X_new = np.asarray(X_new, dtype=float)
    if X_new.ndim == 1:
        X_new = X_new.reshape(1, -1)
    if X_new.ndim != 2:
        raise ValueError("X_new must be (samples, features)")
    return np.asarray(model_func(X_new.T, *params))


if __name__ == "__main__":
    # Replace this function and the x/y block with the problem-specific model.
    def model_func(x, a, b, c):
        return a * np.exp(-b * x) + c

    rng = np.random.default_rng(42)
    x = np.linspace(0, 5, 100)
    y = model_func(x, 8.0, 0.7, 1.5) + rng.normal(0, 0.2, len(x))

    output = fit_curve(
        model_func,
        x,
        y,
        p0=(7.0, 0.5, 1.0),
        bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf]),
    )
    print("parameters:", output["params"])
    print("parameter std:", output["param_std"])
    print(f"MAE:  {output['mae']:.4f}")
    print(f"RMSE: {output['rmse']:.4f}")
    print(f"R2:   {output['r2']:.4f}")
    print("new prediction:", predict_curve(model_func, [6, 7], output["params"]))

    def plane(X, a, b, c):
        x1, x2 = X
        return a * x1 + b * x2 + c

    X = rng.uniform(0, 1, size=(100, 2))
    y = plane(X.T, 2.0, -1.0, 0.5) + rng.normal(0, 0.05, len(X))
    multi_output = fit_curve_multi(plane, X, y, p0=(1, 1, 0))
    print("multi parameters:", multi_output["params"])
    print(f"multi RMSE: {multi_output['rmse']:.4f}")
