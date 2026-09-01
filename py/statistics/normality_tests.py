"""正态性检验模板。

t 检验和 ANOVA 都假设数据近似正态，正式检验前先做本模块的前置检查：

    小样本（n <= 5000）优先 Shapiro-Wilk，对偏离正态最灵敏；
    大样本用 Jarque-Bera，基于偏度与峰度；
    Lilliefors 是均值方差由样本估计时的 K-S 校正版。
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from statsmodels.stats.diagnostic import lilliefors
from statsmodels.stats.stattools import jarque_bera


def _as_1d(data, name: str) -> np.ndarray:
    """统一为一维有限数值数组。"""
    array = np.asarray(data, dtype=float).reshape(-1)
    if array.size < 3 or not np.isfinite(array).all():
        raise ValueError(f"{name} 必须是至少 3 个有限数值的一维数组")
    return array


def shapiro_wilk(data):
    """小样本正态性检验，对偏离正态最灵敏。

    Args:
        data: 一维数值观测，样本量须在 3 到 5000 之间。

    Returns:
        包含 W 统计量、p 值和样本量的字典。

    Raises:
        ValueError: 样本量超出 [3, 5000] 时抛出，大样本请用 ``jarque_bera_test``。
    """
    array = _as_1d(data, "data")
    if array.size > 5000:
        raise ValueError("Shapiro-Wilk 最多支持 5000 个样本，大样本请用 jarque_bera_test")
    statistic, pvalue = stats.shapiro(array)
    return {"statistic": statistic, "pvalue": pvalue, "n": array.size}


def jarque_bera_test(data):
    """基于偏度与峰度的正态性检验，适合大样本。

    正态分布偏度为 0、峰度为 3；样本偏离越多 JB 统计量越大。

    Args:
        data: 一维数值观测。

    Returns:
        包含 JB 统计量、p 值、偏度和峰度的字典。
    """
    array = _as_1d(data, "data")
    statistic, pvalue, skew, kurtosis = jarque_bera(array)
    return {
        "statistic": statistic,
        "pvalue": pvalue,
        "skew": skew,
        "kurtosis": kurtosis,
    }


def lilliefors_test(data, *, dist="norm"):
    """K-S 检验的 Lilliefors 校正版（分布参数由样本估计）。

    普通 K-S 需要事先给定分布参数；参数从样本估计时应使用本校正版，
    否则 p 值会偏大（过度保守）。

    Args:
        data: 一维数值观测。
        dist: 待比较的分布，``norm`` 或 ``exp``。

    Returns:
        包含 K-S 统计量和 p 值的字典。

    Raises:
        ValueError: dist 不合法时抛出。
    """
    if dist not in ("norm", "exp"):
        raise ValueError("dist 必须是 'norm' 或 'exp'")
    array = _as_1d(data, "data")
    statistic, pvalue = lilliefors(array, dist=dist)
    return {"statistic": statistic, "pvalue": pvalue, "dist": dist}


def main():
    # ====== 比赛时主要替换下面这部分 ======
    rng = np.random.default_rng(42)
    normal_data = rng.normal(100.0, 10.0, 200)     # 近似正态
    skewed_data = rng.lognormal(0.0, 0.8, 200)     # 右偏，明显非正态

    for name, data in (("normal", normal_data), ("lognormal", skewed_data)):
        print(f"[{name}] Shapiro:", shapiro_wilk(data))
        print(f"[{name}] Jarque-Bera:", jarque_bera_test(data))
        print(f"[{name}] Lilliefors:", lilliefors_test(data))

    # 本例的简单验收：替换题目后可删除或改写。
    assert shapiro_wilk(normal_data)["pvalue"] > 0.05
    assert shapiro_wilk(skewed_data)["pvalue"] < 0.05
    assert jarque_bera_test(skewed_data)["pvalue"] < 0.05
    assert jarque_bera_test(skewed_data)["skew"] > 0.5  # 右偏


if __name__ == "__main__":
    main()
