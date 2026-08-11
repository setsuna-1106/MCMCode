#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OR-Tools CBC 混合整数线性规划最小可执行模板。

通过 categories 指定每个变量的类型："C"、"I" 或 "B"，分别表示连续、整数和 0-1。
比赛时主要替换 objective、约束矩阵、bounds 和 categories。
"""

import sys

import numpy as np
from ortools.linear_solver import pywraplp


def _as_matrix(matrix, n_variables, name):
    if matrix is None:
        return None
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[1] != n_variables:
        raise ValueError(f"{name} 必须是列数为变量数的二维数组")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} 必须只包含有限数值")
    return matrix


def _as_vector(vector, n_rows, name):
    if vector is None:
        if n_rows:
            raise ValueError(f"{name} 不能为空")
        return None
    vector = np.asarray(vector, dtype=float).reshape(-1)
    if vector.size != n_rows or not np.isfinite(vector).all():
        raise ValueError(f"{name} 的长度必须与约束行数一致且只包含有限数值")
    return vector


def _status_name(status):
    return {
        pywraplp.Solver.OPTIMAL: "OPTIMAL",
        pywraplp.Solver.FEASIBLE: "FEASIBLE",
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    }.get(status, str(status))


def solve_milp(
    objective,
    *,
    A_ub=None,
    b_ub=None,
    A_eq=None,
    b_eq=None,
    bounds=None,
    categories=None,
    maximize=True,
    solver_name="CBC_MIXED_INTEGER_PROGRAMMING",
):
    """求解混合整数线性规划，返回 (solver, variables, status)。"""
    objective = np.asarray(objective, dtype=float)
    if objective.ndim != 1 or objective.size == 0 or not np.isfinite(objective).all():
        raise ValueError("objective 必须是一维且只包含有限数值")

    n_variables = objective.size
    A_ub = _as_matrix(A_ub, n_variables, "A_ub")
    A_eq = _as_matrix(A_eq, n_variables, "A_eq")
    b_ub = _as_vector(b_ub, 0 if A_ub is None else A_ub.shape[0], "b_ub")
    b_eq = _as_vector(b_eq, 0 if A_eq is None else A_eq.shape[0], "b_eq")

    if bounds is None:
        bounds = [(0.0, None)] * n_variables
    if len(bounds) != n_variables:
        raise ValueError("bounds 的长度必须与变量数一致")
    if categories is None:
        categories = ["I"] * n_variables
    if len(categories) != n_variables:
        raise ValueError("categories 的长度必须与变量数一致")

    solver = pywraplp.Solver.CreateSolver(solver_name)
    if solver is None:
        raise RuntimeError(f"无法创建 OR-Tools 求解器: {solver_name}")

    variables = []
    for i, ((lower, upper), category) in enumerate(zip(bounds, categories)):
        category = str(category).upper()
        if category in ("B", "BINARY"):
            variables.append(solver.BoolVar(f"x_{i + 1}"))
            continue

        lower = 0.0 if lower is None else lower
        upper = solver.infinity() if upper is None else upper
        if category in ("I", "INTEGER"):
            variables.append(solver.IntVar(lower, upper, f"x_{i + 1}"))
        elif category in ("C", "CONTINUOUS"):
            variables.append(solver.NumVar(lower, upper, f"x_{i + 1}"))
        else:
            raise ValueError(f"未知变量类型: {category}")

    if A_ub is not None:
        for i, row in enumerate(A_ub):
            expression = solver.Sum(row[j] * variables[j] for j in range(n_variables))
            solver.Add(expression <= b_ub[i])
    if A_eq is not None:
        for i, row in enumerate(A_eq):
            expression = solver.Sum(row[j] * variables[j] for j in range(n_variables))
            solver.Add(expression == b_eq[i])

    expression = solver.Sum(objective[i] * variables[i] for i in range(n_variables))
    solver.Maximize(expression) if maximize else solver.Minimize(expression)
    status = solver.Solve()
    return solver, variables, status


def main():
    # ====== 比赛时主要替换下面这部分 ======
    # x1、x2 是整数产量，open 是是否启用生产线的 0-1 变量。
    objective = np.array([3.0, 5.0, -4.0])
    A_ub = np.array([
        [2.0, 1.0, 0.0],  # 资源 1：2*x1 + x2 <= 8
        [1.0, 2.0, 0.0],  # 资源 2：x1 + 2*x2 <= 8
        [1.0, 1.0, -8.0],  # linking：没有启用生产线时不能生产
    ])
    b_ub = np.array([8.0, 8.0, 0.0])
    bounds = [(0.0, None), (0.0, None), (0.0, 1.0)]
    categories = ["I", "I", "B"]

    solver, variables, status = solve_milp(
        objective,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        categories=categories,
    )

    values = np.array([variable.solution_value() for variable in variables])
    print("求解状态:", _status_name(status))
    print("最优变量:", np.round(values, 8))
    print("目标值:", f"{solver.Objective().Value():.8f}")
    print("约束余量:", np.round(b_ub - A_ub @ values, 8))

    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError(f"混合整数线性规划失败，状态码: {status}")

    assert np.allclose(values, [2.0, 3.0, 1.0], atol=1e-6)
    assert np.all(A_ub @ values <= b_ub + 1e-6)
    assert np.allclose(values[:2], np.round(values[:2]), atol=1e-8)
    assert values[2] in (0.0, 1.0)

    if "--csv" in sys.argv:
        np.savetxt(
            "ortools_milp_out.csv",
            np.column_stack((np.arange(1, values.size + 1), values)),
            delimiter=",",
            fmt=["%d", "%.10f"],
            header="变量编号,最优值",
            comments="",
        )
        print("已保存 -> ortools_milp_out.csv")


if __name__ == "__main__":
    main()
