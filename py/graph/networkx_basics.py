#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NetworkX 建图、遍历和连通性最小可执行模板。

边数据使用 (u, v, weight)；无权边也可以只写 (u, v)。
"""

import networkx as nx


def build_graph(edges, directed=False):
    """根据边列表创建图，返回 Graph 或 DiGraph。"""
    graph = nx.DiGraph() if directed else nx.Graph()
    for edge in edges:
        edge = tuple(edge)
        if len(edge) == 2:
            u, v = edge
            weight = 1.0
        elif len(edge) == 3:
            u, v, weight = edge
        else:
            raise ValueError("每条边必须是 (u, v) 或 (u, v, weight)")
        graph.add_edge(u, v, weight=float(weight))
    return graph


def analyze_graph(graph, source=None):
    """返回节点数、边数、度数和可选的 BFS/DFS 结果。"""
    result = {
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "degree": dict(graph.degree()),
    }
    if source is not None:
        result["bfs"] = list(nx.bfs_tree(graph, source))
        result["dfs"] = list(nx.dfs_tree(graph, source))
    if graph.is_directed():
        result["components"] = list(nx.weakly_connected_components(graph))
    else:
        result["components"] = list(nx.connected_components(graph))
    return result


def main():
    # ====== 比赛时主要替换下面这部分 ======
    edges = [
        ("A", "B", 2),
        ("B", "C", 1),
        ("C", "D", 3),
        ("A", "D", 8),
    ]
    graph = build_graph(edges)
    result = analyze_graph(graph, source="A")
    print("节点:", list(graph.nodes))
    print("边:", list(graph.edges(data=True)))
    print("BFS:", result["bfs"])
    print("DFS:", result["dfs"])
    print("连通分量:", result["components"])
    assert result["node_count"] == 4
    assert len(result["components"]) == 1


if __name__ == "__main__":
    main()
