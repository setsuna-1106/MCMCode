"""statsmodels OLS diagnostic template."""

from __future__ import annotations

import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera


def _prepare_data(X, y):
    """校验 OLS 输入并返回 NumPy 数组。"""
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
    """计算 OLS 残差、异方差、自相关和多重共线性诊断。

    Args:
        result: 已拟合的 statsmodels OLS 结果。
        X: 拟合时使用的原始特征矩阵，不含常数列。
        feature_names: 特征名；为空时自动生成。

    Returns:
        按诊断类别组织的统计量和 p 值字典。
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2 or len(X) != len(result.resid):
        raise ValueError("X must be the raw feature matrix used to fit result")

    if feature_names is None:
        feature_names = [f"x{i + 1}" for i in range(X.shape[1])]
    if len(feature_names) != X.shape[1]:
        raise ValueError("feature_names must match the number of features")

    # 诊断中的 VIF 需要显式常数列，但 feature_names 只对应原始特征。
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
    """用全部观测拟合 OLS，并返回模型和诊断结果。

    Args:
        X, y: OLS 特征矩阵和目标值。
        feature_names: 特征名。
        robust: 是否使用 HC3 稳健协方差。

    Returns:
        包含拟合结果和 ``diagnose_ols`` 输出的字典。
    """
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
