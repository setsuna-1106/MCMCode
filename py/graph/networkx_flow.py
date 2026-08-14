#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NetworkX 最大流和最小割最小可执行模板。

有向边使用 (u, v, capacity)；容量放在 edge['capacity'] 中。
"""

import networkx as nx


def solve_max_flow(edges, source, sink):
    """返回最大流值、流字典、最小割值和割的两侧节点集合。"""
    graph = nx.DiGraph()
    for u, v, capacity in edges:
        graph.add_edge(u, v, capacity=capacity)

    flow_value, flow_dict = nx.maximum_flow(
        graph,
        source,
        sink,
        capacity="capacity",
    )
    cut_value, partition = nx.minimum_cut(
        graph,
        source,
        sink,
        capacity="capacity",
    )
    return {
        "flow_value": flow_value,
        "flow_dict": flow_dict,
        "cut_value": cut_value,
        "partition": partition,
    }


def main():
    # ====== 比赛时主要替换下面这部分 ======
    edges = [
        ("s", "a", 10),
        ("s", "b", 5),
        ("a", "b", 15),
        ("a", "t", 10),
        ("b", "t", 10),
    ]
    result = solve_max_flow(edges, "s", "t")
    print("最大流:", result["flow_value"])
    print("最小割:", result["cut_value"])
    print("流量方案:", result["flow_dict"])
    assert result["flow_value"] == 15
    assert result["cut_value"] == result["flow_value"]


if __name__ == "__main__":
    main()
