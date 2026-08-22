"""NetworkX 最大流和最小割最小可执行模板。

有向边使用 (u, v, capacity)；容量放在 edge['capacity'] 中。
"""

import networkx as nx


def solve_max_flow(edges, source, sink):
    """同时计算网络最大流和对应的最小割。

    Args:
        edges: 形如 ``(u, v, capacity)`` 的有向容量边序列。
        source: 流的起点。
        sink: 流的终点。

    Returns:
        包含最大流值、边流量、最小割值及割两侧节点集合的字典。
    """
    graph = nx.DiGraph()
    for u, v, capacity in edges:
        graph.add_edge(u, v, capacity=capacity)

    # 最大流值和最小割值理论上相等，可用作结果一致性检查。
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
