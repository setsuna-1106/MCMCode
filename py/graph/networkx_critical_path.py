"""NetworkX DAG 关键路径最小可执行模板。

有向边使用 (before, after, duration)，图必须是有向无环图。
"""

import networkx as nx


def solve_critical_path(edges, duration="duration"):
    """返回关键路径节点序列和最短项目完成时间下的最长工期。"""
    graph = nx.DiGraph()
    for before, after, time in edges:
        graph.add_edge(before, after, **{duration: time})
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("关键路径分析需要有向无环图")

    path = nx.dag_longest_path(graph, weight=duration)
    length = nx.dag_longest_path_length(graph, weight=duration)
    return {"graph": graph, "path": path, "length": float(length)}


def main():
    # ====== 比赛时主要替换下面这部分 ======
    edges = [
        ("start", "A", 3),
        ("start", "B", 2),
        ("A", "C", 4),
        ("B", "C", 1),
        ("C", "end", 2),
    ]
    result = solve_critical_path(edges)
    print("关键路径:", result["path"])
    print("项目工期:", result["length"])
    assert result["path"] == ["start", "A", "C", "end"]
    assert result["length"] == 9.0


if __name__ == "__main__":
    main()
