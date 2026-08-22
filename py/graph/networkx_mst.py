"""NetworkX 最小生成树最小可执行模板。"""

import networkx as nx


def solve_mst(graph, weight="weight"):
    """求无向图的最小生成树。

    Args:
        graph: 待连接的无向 NetworkX 图。
        weight: 边权属性名。

    Returns:
        ``(tree, total_weight)``，分别为最小生成树和树边权重总和。

    Raises:
        ValueError: 输入图是有向图时抛出。
    """
    if graph.is_directed():
        raise ValueError("最小生成树需要无向图")
    # 生成树连接所有可达节点且不含环，最小化总建设成本。
    tree = nx.minimum_spanning_tree(graph, weight=weight)
    total_weight = sum(
        data.get(weight, 1.0) for _, _, data in tree.edges(data=True)
    )
    return tree, float(total_weight)


def main():
    # ====== 比赛时主要替换下面这部分 ======
    graph = nx.Graph()
    graph.add_weighted_edges_from([
        ("A", "B", 1),
        ("B", "C", 2),
        ("C", "D", 1),
        ("A", "D", 5),
        ("B", "D", 4),
    ])

    tree, total_weight = solve_mst(graph)
    print("生成树边:", list(tree.edges(data=True)))
    print("最小总权重:", total_weight)
    assert tree.number_of_edges() == graph.number_of_nodes() - 1
    assert total_weight == 4.0


if __name__ == "__main__":
    main()
