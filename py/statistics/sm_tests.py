"""Common statsmodels hypothesis-test templates."""

from __future__ import annotations

import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.contingency_tables import Table
from statsmodels.stats.weightstats import ttest_ind


def welch_ttest(group_a, group_b):
    """Compare two independent groups without assuming equal variance."""
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
    """Test independence in a contingency table."""
    result = Table(np.asarray(table, dtype=float)).test_nominal_association()
    return {
        "statistic": result.statistic,
        "pvalue": result.pvalue,
        "df": result.df,
    }


def one_way_anova(data, target, group):
    """Test whether the means differ across levels of one grouping column."""
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
