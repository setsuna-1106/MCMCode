"""statsmodels OLS diagnostic template."""

from __future__ import annotations

import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera


def _prepare_data(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("X must be a 2D feature matrix and match y in length")
    if len(y) <= X.shape[1] + 1:
        raise ValueError("The sample size is too small for the feature count")
    return X, y


def diagnose_ols(result, X, feature_names=None):
    """Return residual, variance, autocorrelation, and VIF diagnostics."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2 or len(X) != len(result.resid):
        raise ValueError("X must be the raw feature matrix used to fit result")

    if feature_names is None:
        feature_names = [f"x{i + 1}" for i in range(X.shape[1])]
    if len(feature_names) != X.shape[1]:
        raise ValueError("feature_names must match the number of features")

    X_sm = sm.add_constant(X, has_constant="add")
    jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(result.resid)
    bp_lm, bp_lm_pvalue, bp_f, bp_f_pvalue = het_breuschpagan(
        result.resid, X_sm
    )
    vif = {
        name: variance_inflation_factor(X_sm, i + 1)
        for i, name in enumerate(feature_names)
    }

    return {
        "normality": {
            "statistic": jb_stat,
            "pvalue": jb_pvalue,
            "skew": skew,
            "kurtosis": kurtosis,
        },
        "heteroscedasticity": {
            "lm_statistic": bp_lm,
            "lm_pvalue": bp_lm_pvalue,
            "f_statistic": bp_f,
            "f_pvalue": bp_f_pvalue,
        },
        "autocorrelation": {
            "durbin_watson": durbin_watson(result.resid),
        },
        "multicollinearity": {
            "vif": vif,
        },
    }


def fit_ols_diagnostics(X, y, feature_names=None, robust=False):
    """Fit OLS on all observations and return the model plus diagnostics."""
    X, y = _prepare_data(X, y)
    X_sm = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y, X_sm)
    result = model.fit(cov_type="HC3") if robust else model.fit()
    return {
        "model": result,
        "diagnostics": diagnose_ols(result, X, feature_names),
    }


if __name__ == "__main__":
    # Replace this block with X = df[features].to_numpy(), y = df[target].to_numpy().
    from sklearn.datasets import load_diabetes

    data = load_diabetes()
    output = fit_ols_diagnostics(data.data, data.target, data.feature_names)
    print("R2:", output["model"].rsquared)
    print("diagnostics:", output["diagnostics"])
