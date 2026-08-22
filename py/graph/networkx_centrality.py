"""NetworkX 中心性分析最小可执行模板。"""

import networkx as nx


def analyze_centrality(graph, weight="weight"):
    """计算四种常用节点中心性指标。

    Args:
        graph: 待分析的 NetworkX 图。
        weight: 边权属性名；不存在时按 NetworkX 的无权规则处理。

    Returns:
        以指标名称为键、以节点到数值映射为值的字典。
    """
    # 不同中心性衡量的“重要”含义不同，结果应结合题意解释。
    return {
        "degree": nx.degree_centrality(graph),
        "betweenness": nx.betweenness_centrality(graph, weight=weight),
        "closeness": nx.closeness_centrality(graph, distance=weight),
        "pagerank": nx.pagerank(graph, weight=weight),
    }


def main():
    # ====== 比赛时主要替换下面这部分 ======
    graph = nx.Graph()
    graph.add_weighted_edges_from([
        ("A", "B", 1),
        ("A", "C", 1),
        ("A", "D", 1),
        ("B", "C", 1),
        ("C", "D", 1),
        ("D", "E", 1),
    ])

    centrality = analyze_centrality(graph)
    for name, values in centrality.items():
        ranking = sorted(values.items(), key=lambda item: item[1], reverse=True)
        print(f"{name} 排名:", ranking)
    assert set(centrality) == {"degree", "betweenness", "closeness", "pagerank"}
    assert all(len(values) == graph.number_of_nodes() for values in centrality.values())


if __name__ == "__main__":
    main()
