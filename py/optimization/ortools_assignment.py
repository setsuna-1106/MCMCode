#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OR-Tools CP-SAT 指派问题最小可执行模板。

x[i, j] = 1 表示对象 i 被分配给任务 j。
每个对象必须分配一次；每个任务默认至多被分配一次。
"""

import numpy as np
from ortools.sat.python import cp_model


def solve_assignment(cost, *, maximize=False, cost_scale=100, time_limit=10.0):
    """求解矩阵型指派问题，返回 (model, solver, variables, status)。"""
    cost = np.asarray(cost, dtype=float)
    if cost.ndim != 2 or cost.size == 0 or not np.isfinite(cost).all():
        raise ValueError("cost 必须是非空二维有限数值数组")
    if cost_scale <= 0 or int(cost_scale) != cost_scale:
        raise ValueError("cost_scale 必须是正整数")

    n_objects, n_tasks = cost.shape
    integer_cost = np.rint(cost * int(cost_scale)).astype(np.int64)
    model = cp_model.CpModel()
    variables = [
        [model.NewBoolVar(f"x_{i}_{j}") for j in range(n_tasks)]
        for i in range(n_objects)
    ]

    for row in variables:
        model.AddExactlyOne(row)
    for j in range(n_tasks):
        model.AddAtMostOne(variables[i][j] for i in range(n_objects))

    objective = sum(
        int(integer_cost[i, j]) * variables[i][j]
        for i in range(n_objects)
        for j in range(n_tasks)
    )
    model.Maximize(objective) if maximize else model.Minimize(objective)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 1
    status = solver.Solve(model)
    return model, solver, variables, status


def main():
    # ====== 比赛时主要替换下面这部分 ======
    cost = np.array([
        [9.0, 2.0, 7.0, 8.0],
        [6.0, 4.0, 3.0, 7.0],
        [5.0, 8.0, 1.0, 8.0],
    ])

    _, solver, variables, status = solve_assignment(cost)
    print("求解状态:", solver.StatusName(status))

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"指派问题失败: {solver.StatusName(status)}")

    assignment = [
        next(j for j in range(cost.shape[1]) if solver.Value(variables[i][j]))
        for i in range(cost.shape[0])
    ]
    objective = sum(cost[i, assignment[i]] for i in range(cost.shape[0]))
    print("任务分配:", assignment)
    print("总成本:", f"{objective:.8f}")
    assert assignment == [1, 0, 2]
    assert np.isclose(objective, 9.0)


if __name__ == "__main__":
    main()
