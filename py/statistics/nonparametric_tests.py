"""非参数检验模板。

数据不满足正态假设（先用 ``normality_tests`` 检查）或样本量过小时，
用基于秩的检验替代对应的参数检验：

    Mann-Whitney U  <-> Welch t 检验（两组独立样本）
    Kruskal-Wallis  <-> 单因素 ANOVA（三组及以上）
    Wilcoxon 符号秩 <-> 配对 t 检验（同一对象前后测量）

秩检验比较的是分布位置（中位数），论文措辞注意与均值区分。
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _as_1d(data, name: str) -> np.ndarray:
    """统一为一维有限数值数组。"""
    array = np.asarray(data, dtype=float).reshape(-1)
    if array.size < 3 or not np.isfinite(array).all():
        raise ValueError(f"{name} 必须是至少 3 个有限数值的一维数组")
    return array


def mann_whitney_u(group_a, group_b, *, alternative="two-sided"):
    """两独立样本的秩和检验，Welch t 的非参数替代。

    把两组观测合并排名，比较两组的秩和是否平衡；不需要正态假设。

    Args:
        group_a, group_b: 两组一维数值观测。
        alternative: ``two-sided``、``less`` 或 ``greater``。

    Returns:
        包含 U 统计量和 p 值的字典。
    """
    if alternative not in ("two-sided", "less", "greater"):
        raise ValueError("alternative 必须是 'two-sided'、'less' 或 'greater'")
    a = _as_1d(group_a, "group_a")
    b = _as_1d(group_b, "group_b")
    statistic, pvalue = stats.mannwhitneyu(a, b, alternative=alternative)
    return {"statistic": statistic, "pvalue": pvalue}


def kruskal_wallis(groups):
    """多组独立样本的秩检验，单因素 ANOVA 的非参数替代。

    Args:
        groups: 每组一个一维数组的序列，至少两组。

    Returns:
        包含 H 统计量、p 值和组数的字典。
    """
    arrays = [_as_1d(group, f"group_{i}") for i, group in enumerate(groups)]
    if len(arrays) < 2:
        raise ValueError("Kruskal-Wallis 检验至少需要两组数据")
    statistic, pvalue = stats.kruskal(*arrays)
    return {"statistic": statistic, "pvalue": pvalue, "n_groups": len(arrays)}


def wilcoxon_signed_rank(before, after, *, alternative="two-sided"):
    """配对样本的符号秩检验，配对 t 的非参数替代。

    对逐对差值的绝对值排名，再结合符号检验差值是否系统性地偏离 0。

    Args:
        before, after: 等长的两组一维观测（同一对象的配对测量）。
        alternative: ``two-sided``、``less`` 或 ``greater``。

    Returns:
        包含 W 统计量、p 值和对数的字典。
    """
    if alternative not in ("two-sided", "less", "greater"):
        raise ValueError("alternative 必须是 'two-sided'、'less' 或 'greater'")
    a = _as_1d(before, "before")
    b = _as_1d(after, "after")
    if a.size != b.size:
        raise ValueError("before 和 after 必须等长")
    if np.all(a == b):
        raise ValueError("差值全部为 0，无法执行符号秩检验")
    statistic, pvalue = stats.wilcoxon(a, b, alternative=alternative)
    return {"statistic": statistic, "pvalue": pvalue, "n_pairs": a.size}


def main():
    # ====== 比赛时主要替换下面这部分 ======
    rng = np.random.default_rng(42)

    # 两组右偏数据（对数正态），用秩检验替代 Welch t。
    group_a = rng.lognormal(0.0, 0.6, 80)
    group_b = rng.lognormal(0.4, 0.6, 80)  # 中位数整体抬高
    print("Mann-Whitney U:", mann_whitney_u(group_a, group_b))

    # 三组水平不同，用秩检验替代 ANOVA。
    groups = [
        rng.normal(100.0, 5.0, 40),
        rng.normal(103.0, 5.0, 40),
        rng.normal(110.0, 5.0, 40),
    ]
    print("Kruskal-Wallis:", kruskal_wallis(groups))

    # 同一批对象处理前后对比，用符号秩替代配对 t。
    before = rng.normal(100.0, 5.0, 30)
    after = before + rng.normal(2.0, 1.0, 30)
    print("Wilcoxon signed-rank:", wilcoxon_signed_rank(before, after))

    # 本例的简单验收：替换题目后可删除或改写。
    assert mann_whitney_u(group_a, group_b)["pvalue"] < 0.05
    assert kruskal_wallis(groups)["pvalue"] < 0.05
    assert wilcoxon_signed_rank(before, after)["pvalue"] < 0.05


if __name__ == "__main__":
    main()
