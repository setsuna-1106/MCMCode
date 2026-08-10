#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scipy.optimize.minimize 最小可执行模板。

适用：带变量边界、线性约束或非线性约束的连续优化问题。
比赛时主要替换 objective、x0、bounds 和 constraints。

验收锚点：本例最优解约为 [1.5, 2.5]，最小目标值约为 0.5。
"""

import sys

import numpy as np
from scipy.optimize import minimize


def objective(x):
    """目标函数：min f(x)。把这里替换成题目中的目标函数。"""
    x1, x2 = x
    return (x1 - 2.0) ** 2 + (x2 - 3.0) ** 2


def solve_minimize(
    objective,
    x0,
    *,
    bounds=None,
    constraints=(),
    method="SLSQP",
    options=None,
):
    """调用 scipy.optimize.minimize 并返回完整的 OptimizeResult。"""
    x0 = np.asarray(x0, dtype=float)
    if x0.ndim != 1 or x0.size == 0 or not np.all(np.isfinite(x0)):
        raise ValueError("x0 必须是一维且只包含有限数值")

    return minimize(
        objective,
        x0,
        method=method,
        bounds=bounds,
        constraints=constraints,
        options=options,
    )


def main():
    # ====== 比赛时主要替换下面这部分 ======
    x0 = np.array([1.0, 1.0])  # 初始解，长度必须与变量个数一致
    bounds = [
        (0.0, None),  # x1 >= 0
        (0.0, None),  # x2 >= 0
    ]

    # scipy 的不等式约束统一写成 g(x) >= 0。
    constraints = [
        {"type": "ineq", "fun": lambda x: 4.0 - x[0] - x[1]},  # x1+x2 <= 4
        # {"type": "eq", "fun": lambda x: x[0] - x[1]},  # 等式约束示例
    ]

    result = solve_minimize(
        objective,
        x0,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={"ftol": 1e-9, "maxiter": 1000},
    )

    print("是否收敛:", result.success)
    print("求解信息:", result.message)
    print("最优变量:", np.round(result.x, 8))
    print("最小目标值:", f"{result.fun:.8f}")

    if not result.success:
        raise RuntimeError(f"优化失败: {result.message}")

    # 本例的简单验收：检查最优解和约束，替换题目后可删除或改写。
    assert np.allclose(result.x, [1.5, 2.5], atol=1e-5)
    assert result.x[0] + result.x[1] <= 4.0 + 1e-8

    if "--csv" in sys.argv:
        np.savetxt(
            "scipy_minimize_out.csv",
            np.column_stack((np.arange(1, result.x.size + 1), result.x)),
            delimiter=",",
            fmt=["%d", "%.10f"],
            header="变量编号,最优值",
            comments="",
        )
        print("已保存 -> scipy_minimize_out.csv")


if __name__ == "__main__":
    main()
