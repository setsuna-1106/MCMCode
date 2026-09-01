"""常用统计假设检验模板（statsmodels / scipy）。"""

from __future__ import annotations

import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import fisher_exact, levene, ttest_rel
from statsmodels.stats.contingency_tables import Table
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.weightstats import ttest_ind


def welch_ttest(group_a, group_b):
    """比较两个独立样本的均值，不假设方差相等。

    Args:
        group_a, group_b: 两组一维数值观测。

    Returns:
        包含 t 统计量、p 值和自由度的字典。
    """
    statistic, pvalue, df = ttest_ind(
        np.asarray(group_a, dtype=float),
        np.asarray(group_b, dtype=float),
        usevar="unequal",
    )
    return {
        "statistic": statistic,
        "pvalue": pvalue,
        "df": df,
    }


def chi_square(table):
    """检验列联表中的两个分类变量是否独立。

    Args:
        table: 非负频数列联表。

    Returns:
        包含卡方统计量、p 值和自由度的字典。
    """
    result = Table(np.asarray(table, dtype=float)).test_nominal_association()
    return {
        "statistic": result.statistic,
        "pvalue": result.pvalue,
        "df": result.df,
    }


def one_way_anova(data, target, group):
    """检验不同分组水平的目标均值是否存在差异。

    Args:
        data: 包含目标列和分组列的 DataFrame。
        target: 连续型目标列名。
        group: 分类分组列名。

    Returns:
        包含 OLS 模型、ANOVA 表、F 统计量和 p 值的字典。
    """
    # Q() 允许列名包含空格或与公式语言关键字冲突的字符。
    formula = f'Q("{target}") ~ C(Q("{group}"))'
    model = smf.ols(formula, data=data).fit()
    table = sm.stats.anova_lm(model, typ=2)
    row = table.iloc[0]
    return {
        "model": model,
        "table": table,
        "statistic": row["F"],
        "pvalue": row["PR(>F)"],
    }


def paired_ttest(before, after):
    """比较同一对象前后两次测量的均值差异。

    Args:
        before, after: 等长的两组一维观测（同一对象的配对测量）。

    Returns:
        包含 t 统计量、p 值和自由度（测量对数 - 1）的字典。
    """
    a = np.asarray(before, dtype=float).reshape(-1)
    b = np.asarray(after, dtype=float).reshape(-1)
    if a.size != b.size or a.size < 2:
        raise ValueError("before 和 after 必须是等长且至少 2 个观测的一维数组")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("before 和 after 只能包含有限数值")
    statistic, pvalue = ttest_rel(a, b)
    return {"statistic": statistic, "pvalue": pvalue, "df": a.size - 1}


def levene_test(groups, *, center="median"):
    """检验多组数据的方差是否相等（t/ANOVA 的方差齐性前置检查）。

    Args:
        groups: 每组一个一维数组的序列，至少两组。
        center: 中心化方式；``median``（Brown–Forsythe 版本）对非正态数据更稳健。

    Returns:
        包含 W 统计量、p 值和组数的字典。
    """
    arrays = [np.asarray(group, dtype=float).reshape(-1) for group in groups]
    if len(arrays) < 2:
        raise ValueError("Levene 检验至少需要两组数据")
    if any(
        array.size < 2 or not np.isfinite(array).all() for array in arrays
    ):
        raise ValueError("每组必须是至少 2 个有限数值的一维数组")
    statistic, pvalue = levene(*arrays, center=center)
    return {"statistic": statistic, "pvalue": pvalue, "n_groups": len(arrays)}


def fisher_exact_test(table):
    """2x2 列联表的 Fisher 精确检验，期望频数过小时代替卡方检验。

    Args:
        table: 只含非负数值的 2x2 频数表。

    Returns:
        包含优势比和 p 值的字典。
    """
    matrix = np.asarray(table, dtype=float)
    if (
        matrix.shape != (2, 2)
        or np.any(matrix < 0)
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("table 必须是只含非负数值的 2x2 列联表")
    odds_ratio, pvalue = fisher_exact(matrix)
    return {"odds_ratio": odds_ratio, "pvalue": pvalue}


def tukey_hsd(data, target, group, *, alpha=0.05):
    """单因素 ANOVA 显著后的两两事后比较（Tukey HSD）。

    Args:
        data: 包含目标列和分组列的 DataFrame。
        target: 连续型目标列名。
        group: 分类分组列名。
        alpha: 显著性水平。

    Returns:
        包含 TukeyHSDResults、文本汇总表、逐对拒绝结论和 p 值的字典。
    """
    endog = np.asarray(data[target], dtype=float).reshape(-1)
    groups = np.asarray(data[group]).reshape(-1)
    if endog.size != groups.size or endog.size < 4:
        raise ValueError("data 的目标列和分组列必须等长且至少 4 行")
    if not np.isfinite(endog).all():
        raise ValueError("目标列只能包含有限数值")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha 必须落在 (0, 1) 区间")
    result = pairwise_tukeyhsd(endog=endog, groups=groups, alpha=alpha)
    return {
        "model": result,
        "table": str(result.summary()),
        "reject": np.asarray(result.reject),
        "pvalues": np.asarray(result.pvalues),
    }


if __name__ == "__main__":
    # ====== 比赛时主要替换下面这部分 ======
    rng = np.random.default_rng(42)

    a = rng.normal(500, 30, 40)
    b = rng.normal(530, 30, 40)
    print("Welch t-test:", welch_ttest(a, b))

    before = rng.normal(100.0, 5.0, 30)
    after = before + rng.normal(2.0, 1.0, 30)
    print("Paired t-test:", paired_ttest(before, after))

    print("Chi-square:", chi_square([[180, 20], [150, 50]]))
    print("Fisher exact:", fisher_exact_test([[8, 2], [1, 9]]))

    equal_variance_groups = [
        rng.normal(0.0, 2.0, 50),
        rng.normal(0.5, 2.0, 50),
    ]
    print("Levene:", levene_test(equal_variance_groups))

    # Tukey HSD：三个均值不同的组，事后检验应找出存在差异的组对。
    import pandas as pd

    df = pd.DataFrame({
        "yield": np.concatenate([
            rng.normal(100.0, 5.0, 30),
            rng.normal(103.0, 5.0, 30),
            rng.normal(110.0, 5.0, 30),
        ]),
        "plan": np.repeat(["A", "B", "C"], 30),
    })
    tukey = tukey_hsd(df, target="yield", group="plan")
    print(tukey["table"])

    # 本例的简单验收：替换题目后可删除或改写。
    assert paired_ttest(before, after)["pvalue"] < 0.05
    assert fisher_exact_test([[8, 2], [1, 9]])["pvalue"] < 0.05
    assert tukey["reject"].sum() >= 1
