"""NetworkX 最大权匹配最小可执行模板。

无向边使用 (u, v, weight)；二分图也可以使用该模板。
"""

import networkx as nx


def solve_matching(edges, maxcardinality=True):
    """求无向图的最大权匹配。

    Args:
        edges: 形如 ``(u, v, weight)`` 的无向边序列。
        maxcardinality: 权重相同时是否优先选择边数更多的匹配。

    Returns:
        ``(matching, total_weight)``，分别为匹配边集合和总权重。
    """
    graph = nx.Graph()
    for u, v, weight in edges:
        graph.add_edge(u, v, weight=weight)

    # matching 中每个节点最多出现一次，适合表达一对一指派关系。
    matching = nx.max_weight_matching(
        graph,
        maxcardinality=maxcardinality,
        weight="weight",
    )
    total_weight = sum(graph[u][v].get("weight", 1) for u, v in matching)
    return matching, float(total_weight)


def main():
    # ====== 比赛时主要替换下面这部分 ======
    edges = [
        ("worker_1", "task_1", 5),
        ("worker_1", "task_2", 1),
        ("worker_2", "task_1", 4),
        ("worker_2", "task_2", 3),
    ]
    matching, total_weight = solve_matching(edges)
    print("匹配:", matching)
    print("匹配总权重:", total_weight)
    assert len(matching) == 2
    assert total_weight == 8.0


if __name__ == "__main__":
    main()
