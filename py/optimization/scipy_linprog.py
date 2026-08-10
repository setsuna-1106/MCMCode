#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scipy.optimize.linprog 最小可执行模板。

标准形式：
    min c.T @ x
    s.t. A_ub @ x <= b_ub
         A_eq @ x == b_eq
         bounds[i][0] <= x[i] <= bounds[i][1]

比赛时主要替换 c、A_ub、b_ub、A_eq、b_eq 和 bounds。
本例是最大化收益问题，因此令 c = -收益系数；求解后最大收益为 -result.fun。

验收锚点：本例最优产量约为 [2.6667, 2.6667]，最大收益约为 21.3333。
"""

import sys

import numpy as np
from scipy.optimize import linprog


def solve_linprog(
    c,
    *,
    A_ub=None,
    b_ub=None,
    A_eq=None,
    b_eq=None,
    bounds=None,
    method="highs",
    options=None,
):
    """求解线性规划并返回完整的 OptimizeResult。"""
    c = np.asarray(c, dtype=float)
    if c.ndim != 1 or c.size == 0 or not np.all(np.isfinite(c)):
        raise ValueError("c 必须是一维且只包含有限数值")

    if A_ub is not None:
        A_ub = np.asarray(A_ub, dtype=float)
    if b_ub is not None:
        b_ub = np.asarray(b_ub, dtype=float)
    if A_eq is not None:
        A_eq = np.asarray(A_eq, dtype=float)
    if b_eq is not None:
        b_eq = np.asarray(b_eq, dtype=float)

    return linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method=method,
        options=options,
    )


def main():
    # ====== 比赛时主要替换下面这部分 ======
    # x[0], x[1]：两种产品的生产量
    profit = np.array([3.0, 5.0])  # 单位收益，题目若是收益最大化就取负
    c = -profit

    # 资源约束：A_ub @ x <= b_ub
    A_ub = np.array([
        [2.0, 1.0],  # 资源 1：2*x1 + x2 <= 8
        [1.0, 2.0],  # 资源 2：x1 + 2*x2 <= 8
    ])
    b_ub = np.array([8.0, 8.0])

    # 等式约束示例：A_eq @ x == b_eq；没有时保持 None。
    A_eq = None
    b_eq = None

    bounds = [
        (0.0, None),  # x1 >= 0
        (0.0, None),  # x2 >= 0
    ]

    result = solve_linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
    )

    print("是否收敛:", result.success)
    print("求解信息:", result.message)
    print("最优变量:", np.round(result.x, 8))
    print("最小化目标值:", f"{result.fun:.8f}")
    print("最大收益:", f"{-result.fun:.8f}")
    print("资源余量:", np.round(b_ub - A_ub @ result.x, 8))

    if not result.success:
        raise RuntimeError(f"线性规划失败: {result.message}")

    # 本例的简单验收：替换题目后可删除或改写。
    assert np.allclose(result.x, [8.0 / 3.0, 8.0 / 3.0], atol=1e-6)
    assert np.all(A_ub @ result.x <= b_ub + 1e-8)

    if "--csv" in sys.argv:
        np.savetxt(
            "scipy_linprog_out.csv",
            np.column_stack((np.arange(1, result.x.size + 1), result.x)),
            delimiter=",",
            fmt=["%d", "%.10f"],
            header="变量编号,最优值",
            comments="",
        )
        print("已保存 -> scipy_linprog_out.csv")


if __name__ == "__main__":
    main()
