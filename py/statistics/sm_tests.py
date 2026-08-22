"""Common statsmodels hypothesis-test templates."""

from __future__ import annotations

import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.contingency_tables import Table
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


if __name__ == "__main__":
    # Replace these arrays/DataFrame with the data from the problem.
    rng = np.random.default_rng(42)
    a = rng.normal(500, 30, 40)
    b = rng.normal(530, 30, 40)
    print("Welch t-test:", welch_ttest(a, b))
    print("Chi-square:", chi_square([[180, 20], [150, 50]]))
