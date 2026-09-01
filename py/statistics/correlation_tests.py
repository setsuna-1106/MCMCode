"""相关分析模板。

Pearson 检验线性相关，Spearman 检验单调相关（先转成秩再算，对
异常值和单调非线性关系稳健）。

注意区分两个量：相关系数的大小衡量相关强度，p 值只回答
“相关系数是否显著异于 0”；大样本下微弱相关也可能显著，
论文中两者都要报告。
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _as_pair(x, y) -> tuple[np.ndarray, np.ndarray]:
    """校验并返回等长的一维数组。"""
    a = np.asarray(x, dtype=float).reshape(-1)
    b = np.asarray(y, dtype=float).reshape(-1)
    if a.size != b.size or a.size < 3:
        raise ValueError("x 和 y 必须是等长且至少 3 个观测的一维数组")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("x 和 y 只能包含有限数值")
    if np.std(a) == 0 or np.std(b) == 0:
        raise ValueError("x 和 y 不能是常数列，否则相关系数没有定义")
    return a, b


def pearson_correlation(x, y):
    """计算 Pearson 线性相关系数及其显著性。

    Args:
        x, y: 等长的一维数值观测。

    Returns:
        包含相关系数 r、p 值和样本量的字典。
    """
    a, b = _as_pair(x, y)
    r, pvalue = stats.pearsonr(a, b)
    return {"statistic": r, "pvalue": pvalue, "n": a.size}


def spearman_correlation(x, y):
    """计算 Spearman 秩相关系数及其显著性。

    对单调但非线性的关系（如指数增长）和异常值更稳健。

    Args:
        x, y: 等长的一维数值观测。

    Returns:
        包含相关系数 rho、p 值和样本量的字典。
    """
    a, b = _as_pair(x, y)
    rho, pvalue = stats.spearmanr(a, b)
    return {"statistic": rho, "pvalue": pvalue, "n": a.size}


def main():
    # ====== 比赛时主要替换下面这部分 ======
    rng = np.random.default_rng(42)

    # 线性相关：Pearson 与 Spearman 都接近 1。
    x = rng.uniform(0.0, 10.0, 100)
    linear_y = 2.0 * x + 1.0 + rng.normal(0.0, 1.0, x.size)
    print("Linear Pearson:", pearson_correlation(x, linear_y))
    print("Linear Spearman:", spearman_correlation(x, linear_y))

    # 单调非线性：Spearman 仍接近 1，Pearson 明显更低。
    exponential_y = np.exp(0.3 * x)
    print("Exponential Pearson:", pearson_correlation(x, exponential_y))
    print("Exponential Spearman:", spearman_correlation(x, exponential_y))

    # 本例的简单验收：替换题目后可删除或改写。
    assert pearson_correlation(x, linear_y)["statistic"] > 0.98
    assert pearson_correlation(x, linear_y)["pvalue"] < 1e-10
    r_exp = pearson_correlation(x, exponential_y)["statistic"]
    rho_exp = spearman_correlation(x, exponential_y)["statistic"]
    assert rho_exp > 0.999 and r_exp < rho_exp  # 单调关系下 Spearman 占优


if __name__ == "__main__":
    main()
