# 图论与网络分析（graph）

把问题抽象成「节点 + 边」的网络结构再求解。对应 `py/graph/`，基于 NetworkX 提供 7 个模板。统一的边属性约定：`weight` 表示距离/成本，`capacity` 表示流量上限，`duration` 表示任务工期。

| 模板 | 问题 | 典型题目 |
| --- | --- | --- |
| networkx_basics | 建图、遍历、连通性 | 网络结构描述、可达性 |
| networkx_shortest_path | 加权最短路径 | 运输、导航、管网 |
| networkx_mst | 最小生成树 | 最低成本连通所有节点 |
| networkx_flow | 最大流 / 最小割 | 运力瓶颈、物流上限 |
| networkx_matching | 最大权匹配 | 一对一指派 |
| networkx_critical_path | DAG 关键路径 | 项目工期、工序依赖 |
| networkx_centrality | 中心性 | 找关键节点 |

## 建图与遍历（networkx_basics.py）

### 原理

**BFS（广度优先）** 借助队列逐层访问邻居，可给出无权最短跳数；**DFS（深度优先）** 沿一条分支走到底再回溯，适合搜索连通性、检测环。**连通分量**是图中「彼此可达」的极大节点组；有向图用弱连通分量（忽略方向），避免方向把本应相连的节点拆开。

### 用法

```python
from py.graph.networkx_basics import analyze_graph, build_graph

graph = build_graph([("A", "B", 2), ("B", "C", 1), ("C", "D", 3)], directed=False)
result = analyze_graph(graph, source="A")
print(result["bfs"], result["dfs"], result["components"])
```

- 边用 `(u, v)` 或 `(u, v, weight)`；`directed=True` 建有向图。
- 返回节点数、边数、度数、BFS/DFS 序列和连通分量。

## 最短路径（networkx_shortest_path.py）

### 原理

求起点到终点**边权总和最小**的路径，NetworkX 对非负权使用 Dijkstra 算法：维护「已确定最短距离」的节点集合，每轮把当前距离最小的节点并入集合并松弛其邻居，直至终点被确定。`weight=None` 时退化为按边数（跳数）最少。

### 用法

```python
import networkx as nx
from py.graph.networkx_shortest_path import solve_shortest_path

graph = nx.Graph()
graph.add_weighted_edges_from([("A", "B", 2), ("B", "C", 1), ("C", "D", 3)])

result = solve_shortest_path(graph, "A", "D")
print(result["path"], result["distance"])
```

起点不可达终点时抛 `NetworkXNoPath`。

### 注意

- 边权含负数时 Dijkstra 不保证正确，应换 Bellman-Ford（`nx.bellman_ford_path`）。

## 最小生成树（networkx_mst.py）

### 原理

在保持**所有节点连通**的前提下选边，使总边权最小。生成树恰有 $n-1$ 条边且不含环；MST 是其中总权重最小的一棵（Kruskal「按权排序贪心加边、跳过成环的边」）。典型语义：修路/铺管线连通所有城市，成本最低。

### 用法

```python
from py.graph.networkx_mst import solve_mst

tree, total_weight = solve_mst(graph)   # 要求无向图
print(list(tree.edges(data=True)), total_weight)
```

## 最大流与最小割（networkx_flow.py）

### 原理

有向容量网络中，`source` 到 `sink` 最多能输送多少流量。**最大流最小割定理**：最大流量 = 最小割容量——把节点分成含源、含汇两份所需切断的最小容量总和，即网络瓶颈。每次找一条还有剩余容量的增广路径加流，直到不存在为止（NetworkX 内置实现）。

### 用法

```python
from py.graph.networkx_flow import solve_max_flow

result = solve_max_flow(
    [("s", "a", 10), ("s", "b", 5), ("a", "t", 10), ("b", "t", 10)],
    source="s", sink="t",
)
print(result["flow_value"])    # 最大流 = 最小割
print(result["flow_dict"])     # 每条边的具体流量分配
print(result["partition"])     # 最小割两侧的节点集合
```

边用 `(u, v, capacity)` 有向三元组。

## 最大权匹配（networkx_matching.py）

### 原理

匹配是「任意两个节点最多出现一次」的边集合，即**一对一配对**。最大权匹配在所有匹配中找边权总和最大的（如工人-任务的收益矩阵），二分图是特例。`maxcardinality=True` 在权重相同时优先匹配更多对。

### 用法

```python
from py.graph.networkx_matching import solve_matching

matching, total_weight = solve_matching(
    [("worker_1", "task_1", 5), ("worker_1", "task_2", 1),
     ("worker_2", "task_1", 4), ("worker_2", "task_2", 3)]
)
print(matching, total_weight)
```

### 注意

- 需要带容量、多对多的指派应改用[优化模块](optimization.md)的 `ortools_min_cost_flow` 或 `ortools_assignment`。

## 关键路径（networkx_critical_path.py）

### 原理

项目活动用 DAG（有向无环图）表示，边 `duration` 是工期，边 $(u, v)$ 表示 $v$ 必须等 $u$ 完成。项目总工期由**最长路径**决定：缩短非最长链上的活动不影响工期，最长链即**关键路径**，其上任何延误都直接推迟完工。模板先校验无环，再求最长路径与长度。

### 用法

```python
from py.graph.networkx_critical_path import solve_critical_path

result = solve_critical_path([
    ("start", "A", 3), ("start", "B", 2),
    ("A", "C", 4), ("B", "C", 1), ("C", "end", 2),
])
print(result["path"], result["length"])   # 关键路径与项目工期
```

## 中心性分析（networkx_centrality.py）

### 原理

四种「重要性」指标，含义不同，应按题意选用：

- **度中心性**：直接邻居数占比，衡量局部连接广度；
- **介数中心性**：节点出现在其他节点对最短路径上的比例，衡量「桥梁 / 咽喉」地位；
- **接近中心性**：到其余节点最短距离之和的倒数，衡量到达全网的平均便利度；
- **PageRank**：随机游走长期停留在该节点的概率，重要节点指向的节点更重要，适合有向网络（如引用、链接）。

### 用法

```python
from py.graph.networkx_centrality import analyze_centrality

centrality = analyze_centrality(graph)   # weight 参与介数/接近/PageRank
for name, values in centrality.items():
    print(name, sorted(values.items(), key=lambda kv: kv[1], reverse=True))
```

## 注意

- NetworkX 适合图结构与中等规模图算法；超大规模或需要整数最优解的路径规划（VRP/TSP）应结合[优化模块](optimization.md)的 OR-Tools。
- 有向 / 无向、`weight` 属性名要与所用算法匹配；`betweenness` 的 `weight` 按距离解释，值越大表示越远。
