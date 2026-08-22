"""NetworkX 加权最短路径最小可执行模板。"""

import networkx as nx


def solve_shortest_path(graph, source, target, weight="weight"):
    """求两个节点之间的加权最短路径。

    Args:
        graph: 待搜索的 NetworkX 图。
        source: 路径起点。
        target: 路径终点。
        weight: 边权属性名；设为 ``None`` 时按边数计算。

    Returns:
        包含节点路径和路径长度的字典。

    Raises:
        NetworkXNoPath: 起点无法到达终点时由 NetworkX 抛出。
    """
    # shortest_path 和 shortest_path_length 使用同一权重规则，避免结果不一致。
    path = nx.shortest_path(graph, source, target, weight=weight)
    distance = nx.shortest_path_length(graph, source, target, weight=weight)
    return {"path": path, "distance": float(distance)}


def main():
    # ====== 比赛时主要替换下面这部分 ======
    graph = nx.Graph()
    graph.add_weighted_edges_from([
        ("A", "B", 2),
        ("B", "C", 1),
        ("C", "D", 3),
        ("A", "C", 10),
        ("B", "D", 8),
    ])

    result = solve_shortest_path(graph, "A", "D")
    print("最短路径:", result["path"])
    print("最短距离:", result["distance"])
    assert result["path"] == ["A", "B", "C", "D"]
    assert result["distance"] == 6.0


if __name__ == "__main__":
    main()
