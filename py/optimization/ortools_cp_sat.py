#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OR-Tools CP-SAT 单机器排程最小可执行模板。

适用：排班、任务调度、先后顺序、资源不重叠和整数逻辑约束。
CP-SAT 的核心变量是整数或布尔变量；含小数数据时应先统一放大并转成整数。
"""

import numpy as np
from ortools.sat.python import cp_model


def solve_schedule(durations, precedences=(), horizon=None, time_limit=10.0):
    """求解单机器排程并最小化总工期。

    Args:
        durations: 每个任务的正整数工时。
        precedences: ``(before, after)`` 先后约束序列。
        horizon: 时间上界；为空时使用工时总和。
        time_limit: 最大求解时间（秒）。

    Returns:
        ``(model, solver, starts, ends, status)``。
    """
    durations = np.asarray(durations, dtype=int).reshape(-1)
    if durations.size == 0 or np.any(durations <= 0):
        raise ValueError("durations 必须包含正整数")

    if horizon is None:
        horizon = int(durations.sum())
    if horizon < int(durations.max()):
        raise ValueError("horizon 必须不小于最长任务时长")

    n_tasks = durations.size
    model = cp_model.CpModel()
    starts = [
        model.NewIntVar(0, horizon - int(durations[i]), f"start_{i}")
        for i in range(n_tasks)
    ]
    ends = [
        model.NewIntVar(int(durations[i]), horizon, f"end_{i}")
        for i in range(n_tasks)
    ]
    # 区间变量把开始、持续时间和结束时间绑定起来。
    intervals = [
        model.NewIntervalVar(starts[i], int(durations[i]), ends[i], f"task_{i}")
        for i in range(n_tasks)
    ]
    model.AddNoOverlap(intervals)

    for before, after in precedences:
        if not 0 <= before < n_tasks or not 0 <= after < n_tasks:
            raise ValueError("precedences 中的任务编号超出范围")
        model.Add(ends[before] <= starts[after])

    # makespan 是所有任务结束时间的最大值，即项目总工期。
    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, ends)
    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 1
    status = solver.Solve(model)
    return model, solver, starts, ends, status


def main():
    # ====== 比赛时主要替换下面这部分 ======
    durations = [3, 2, 4]
    precedences = [(0, 2)]  # 任务 0 完成后才能开始任务 2

    _, solver, starts, ends, status = solve_schedule(durations, precedences)
    print("求解状态:", solver.StatusName(status))

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"排程失败: {solver.StatusName(status)}")

    print("任务开始时间:", [solver.Value(start) for start in starts])
    print("任务结束时间:", [solver.Value(end) for end in ends])
    makespan = max(solver.Value(end) for end in ends)
    print("总工期:", makespan)
    assert makespan == 9
    assert solver.Value(ends[0]) <= solver.Value(starts[2])


if __name__ == "__main__":
    main()
