"""NetworkX 建图、遍历和连通性最小可执行模板。

边数据使用 (u, v, weight)；无权边也可以只写 (u, v)。
"""

import networkx as nx


def build_graph(edges, directed=False):
    """根据边列表创建图。

    Args:
        edges: 形如 ``(u, v)`` 或 ``(u, v, weight)`` 的边序列。
        directed: 是否创建有向图。

    Returns:
        根据 ``directed`` 创建的 ``Graph`` 或 ``DiGraph``。

    Raises:
        ValueError: 某条边既不是二元组也不是三元组时抛出。
    """
    graph = nx.DiGraph() if directed else nx.Graph()
    for edge in edges:
        # 统一保存 weight 属性，使后续加权算法可以直接复用这张图。
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
    """统计图结构，并在给定起点时执行 BFS 和 DFS。

    Args:
        graph: 待分析的 NetworkX 图。
        source: 可选遍历起点；为空时跳过 BFS/DFS。

    Returns:
        包含节点数、边数、度数、遍历序列和连通分量的字典。
    """
    result = {
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "degree": dict(graph.degree()),
    }
    if source is not None:
        result["bfs"] = list(nx.bfs_tree(graph, source))
        result["dfs"] = list(nx.dfs_tree(graph, source))
    # 有向图使用弱连通分量，避免方向导致本应相连的节点被拆开。
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
