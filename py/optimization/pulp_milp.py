#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PuLP 混合整数线性规划最小可执行模板。

模型形式：
    max/min objective.T @ x
    s.t. A_ub @ x <= b_ub
         A_eq @ x == b_eq
         bounds[i][0] <= x[i] <= bounds[i][1]

通过 categories 指定每个变量的类型："Integer"、"Binary" 或 "Continuous"。
比赛时主要替换 objective、约束矩阵、bounds 和 categories。

验收锚点：本例最优解为 [2, 3, 1]，最大收益为 17。
"""

import sys

import numpy as np
import pulp


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


def solve_milp(
    objective,
    *,
    sense=pulp.LpMaximize,
    A_ub=None,
    b_ub=None,
    A_eq=None,
    b_eq=None,
    bounds=None,
    categories=None,
    names=None,
    problem_name="mixed_integer_linear_programming",
    solver=None,
):
    """构造并求解混合整数线性规划。

    Args:
        objective: 目标函数系数向量。
        A_ub, b_ub: 不等式约束 ``A_ub @ x <= b_ub``。
        A_eq, b_eq: 等式约束 ``A_eq @ x == b_eq``。
        bounds: 每个变量的上下界。
        categories: PuLP 变量类型列表。
        names: 变量名；为空时自动生成。
        sense: ``pulp.LpMaximize`` 或 ``pulp.LpMinimize``。

    Returns:
        ``(model, variables)``，模型状态见 ``model.status``。
    """
    objective = np.asarray(objective, dtype=float).reshape(-1)
    if objective.size == 0 or not np.isfinite(objective).all():
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
        categories = [pulp.LpInteger] * n_variables
    if len(categories) != n_variables:
        raise ValueError("categories 的长度必须与变量数一致")

    if names is None:
        names = [f"x_{i + 1}" for i in range(n_variables)]
    if len(names) != n_variables:
        raise ValueError("names 的长度必须与变量数一致")

    model = pulp.LpProblem(problem_name, sense)
    # categories 决定变量是否必须取整数或只能取 0/1。
    variables = [
        pulp.LpVariable(
            names[i],
            lowBound=bounds[i][0],
            upBound=bounds[i][1],
            cat=categories[i],
        )
        for i in range(n_variables)
    ]
    model += pulp.lpSum(objective[i] * variables[i] for i in range(n_variables)), "objective"

    if A_ub is not None:
        for i, row in enumerate(A_ub):
            model += pulp.lpSum(row[j] * variables[j] for j in range(n_variables)) <= b_ub[i], f"ub_{i + 1}"
    if A_eq is not None:
        for i, row in enumerate(A_eq):
            model += pulp.lpSum(row[j] * variables[j] for j in range(n_variables)) == b_eq[i], f"eq_{i + 1}"

    if solver is None:
        solver = pulp.PULP_CBC_CMD(msg=False)
    model.solve(solver)
    return model, variables


def main():
    # ====== 比赛时主要替换下面这部分 ======
    # x1、x2 是整数产量，open 是是否启用生产线的 0-1 变量。
    objective = np.array([3.0, 5.0, -4.0])  # 最后一项是启用生产线的固定成本
    A_ub = np.array([
        [2.0, 1.0, 0.0],  # 资源 1：2*x1 + x2 <= 8
        [1.0, 2.0, 0.0],  # 资源 2：x1 + 2*x2 <= 8
        [1.0, 1.0, -8.0],  # linking：没有启用生产线时不能生产
    ])
    b_ub = np.array([8.0, 8.0, 0.0])
    bounds = [(0.0, None), (0.0, None), (0.0, 1.0)]
    categories = [pulp.LpInteger, pulp.LpInteger, pulp.LpBinary]

    model, variables = solve_milp(
        objective,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        categories=categories,
        names=["product_1", "product_2", "open"],
    )

    status = pulp.LpStatus[model.status]
    values = np.array([pulp.value(variable) for variable in variables], dtype=float)
    objective_value = pulp.value(model.objective)
    print("求解状态:", status)
    print("最优变量:", np.round(values, 8))
    print("目标值:", f"{objective_value:.8f}")
    print("约束余量:", np.round(b_ub - A_ub @ values, 8))

    if status != "Optimal":
        raise RuntimeError(f"混合整数线性规划失败: {status}")

    # 本例的简单验收：替换题目后可删除或改写。
    assert np.allclose(values, [2.0, 3.0, 1.0], atol=1e-6)
    assert np.all(A_ub @ values <= b_ub + 1e-6)
    assert np.allclose(values[:2], np.round(values[:2]), atol=1e-8)
    assert values[2] in (0.0, 1.0)

    if "--csv" in sys.argv:
        np.savetxt(
            "pulp_milp_out.csv",
            np.column_stack((np.arange(1, values.size + 1), values)),
            delimiter=",",
            fmt=["%d", "%.10f"],
            header="变量编号,最优值",
            comments="",
        )
        print("已保存 -> pulp_milp_out.csv")


if __name__ == "__main__":
    main()
