#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OR-Tools RoutingModel TSP 最小可执行模板。

比赛时主要替换 distance_matrix、depot、车辆数，并按题意增加容量或时间窗维度。
距离矩阵应使用整数；小数距离可先统一放大后取整。
"""

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


def solve_tsp(distance_matrix, depot=0, time_limit=5):
    """求解单车辆 TSP，返回 (route, objective_value)。"""
    distance_matrix = np.asarray(distance_matrix, dtype=int)
    if (
        distance_matrix.ndim != 2
        or distance_matrix.shape[0] != distance_matrix.shape[1]
        or distance_matrix.shape[0] < 2
        or np.any(distance_matrix < 0)
    ):
        raise ValueError("distance_matrix 必须是至少 2 个节点的非负方阵")
    if not 0 <= depot < distance_matrix.shape[0]:
        raise ValueError("depot 超出节点范围")

    n_nodes = distance_matrix.shape[0]
    manager = pywrapcp.RoutingIndexManager(n_nodes, 1, depot)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node, to_node])

    transit_callback = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback)

    parameters = pywrapcp.DefaultRoutingSearchParameters()
    parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    parameters.time_limit.seconds = int(time_limit)

    solution = routing.SolveWithParameters(parameters)
    if solution is None:
        raise RuntimeError("TSP 没有找到可行路线")

    route = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        route.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
    route.append(manager.IndexToNode(index))
    return route, solution.ObjectiveValue()


def main():
    # ====== 比赛时主要替换下面这部分 ======
    distance_matrix = np.array([
        [0, 4, 8, 7, 6],
        [4, 0, 5, 9, 3],
        [8, 5, 0, 6, 7],
        [7, 9, 6, 0, 4],
        [6, 3, 7, 4, 0],
    ])
    route, distance = solve_tsp(distance_matrix)
    print("路线:", route)
    print("总距离:", distance)
    assert route[0] == route[-1] == 0
    assert len(route) == distance_matrix.shape[0] + 1
    assert len(set(route[:-1])) == distance_matrix.shape[0]


if __name__ == "__main__":
    main()
