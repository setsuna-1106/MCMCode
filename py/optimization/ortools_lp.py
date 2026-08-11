#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OR-Tools GLOP 线性规划最小可执行模板。

模型形式：
    max/min objective.T @ x
    s.t. A_ub @ x <= b_ub
         A_eq @ x == b_eq
         bounds[i][0] <= x[i] <= bounds[i][1]

本模板的变量默认是连续变量。比赛时主要替换 objective、约束矩阵和 bounds。
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


def solve_lp(
    objective,
    *,
    A_ub=None,
    b_ub=None,
    A_eq=None,
    b_eq=None,
    bounds=None,
    maximize=True,
    solver_name="GLOP",
):
    """求解连续线性规划，返回 (solver, variables, status)。"""
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

    solver = pywraplp.Solver.CreateSolver(solver_name)
    if solver is None:
        raise RuntimeError(f"无法创建 OR-Tools 求解器: {solver_name}")

    variables = []
    for i, (lower, upper) in enumerate(bounds):
        lower = 0.0 if lower is None else lower
        upper = solver.infinity() if upper is None else upper
        variables.append(solver.NumVar(lower, upper, f"x_{i + 1}"))

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
    profit = np.array([3.0, 5.0])
    A_ub = np.array([
        [2.0, 1.0],  # 资源 1：2*x1 + x2 <= 8
        [1.0, 2.0],  # 资源 2：x1 + 2*x2 <= 8
    ])
    b_ub = np.array([8.0, 8.0])
    bounds = [(0.0, None), (0.0, None)]

    solver, variables, status = solve_lp(
        profit,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
    )

    values = np.array([variable.solution_value() for variable in variables])
    print("求解状态:", _status_name(status))
    print("最优变量:", np.round(values, 8))
    print("目标值:", f"{solver.Objective().Value():.8f}")
    print("资源余量:", np.round(b_ub - A_ub @ values, 8))

    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError(f"线性规划失败，状态码: {status}")

    assert np.allclose(values, [8.0 / 3.0, 8.0 / 3.0], atol=1e-6)
    assert np.all(A_ub @ values <= b_ub + 1e-6)

    if "--csv" in sys.argv:
        np.savetxt(
            "ortools_lp_out.csv",
            np.column_stack((np.arange(1, values.size + 1), values)),
            delimiter=",",
            fmt=["%d", "%.10f"],
            header="变量编号,最优值",
            comments="",
        )
        print("已保存 -> ortools_lp_out.csv")


if __name__ == "__main__":
    main()
