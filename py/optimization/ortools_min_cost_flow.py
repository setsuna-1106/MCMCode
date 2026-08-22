#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OR-Tools 最小费用流最小可执行模板。

节点供给约定：供给节点为正，需求节点为负，所有节点供给量之和必须为 0。
每条弧使用 (tail, head, capacity, unit_cost) 表示。
"""

from ortools.graph.python import min_cost_flow


def solve_min_cost_flow(num_nodes, arcs, supplies):
    """求解带节点供需的最小费用流。

    Args:
        num_nodes: 节点数量，节点编号为 ``0`` 到 ``num_nodes - 1``。
        arcs: ``(tail, head, capacity, unit_cost)`` 弧序列。
        supplies: 节点供给量；需求节点使用负数。

    Returns:
        ``(model, status)``。
    """
    if len(supplies) != num_nodes or sum(supplies) != 0:
        raise ValueError("supplies 长度必须等于节点数且总和为 0")

    model = min_cost_flow.SimpleMinCostFlow()
    # 先注册容量和单位费用，再设置每个节点的供给或需求。
    for tail, head, capacity, unit_cost in arcs:
        model.add_arc_with_capacity_and_unit_cost(tail, head, capacity, unit_cost)
    for node, supply in enumerate(supplies):
        model.set_node_supply(node, supply)

    status = model.solve()
    return model, status


def main():
    # ====== 比赛时主要替换下面这部分 ======
    # 两个仓库向两个客户供货，总供给 9，总需求 9。
    arcs = [
        (0, 2, 5, 2),
        (0, 3, 5, 4),
        (1, 2, 4, 3),
        (1, 3, 4, 1),
    ]
    supplies = [5, 4, -4, -5]
    model, status = solve_min_cost_flow(4, arcs, supplies)

    if status != model.OPTIMAL:
        raise RuntimeError(f"最小费用流失败，状态码: {status}")

    print("求解状态:", status)
    print("最小总费用:", model.optimal_cost())
    for arc in range(model.num_arcs()):
        if model.flow(arc) > 0:
            print(
                f"弧 {model.tail(arc)} -> {model.head(arc)}:",
                f"流量={model.flow(arc)}, 单位费用={arcs[arc][3]}",
            )
    assert model.optimal_cost() == 16


if __name__ == "__main__":
    main()
