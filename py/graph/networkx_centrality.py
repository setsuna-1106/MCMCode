"""NetworkX 中心性分析最小可执行模板。"""

import networkx as nx


def analyze_centrality(graph, weight="weight"):
    """返回度中心性、介数中心性、接近中心性和 PageRank。"""
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
