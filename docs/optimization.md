# 优化（optimization）

在约束条件下求目标函数的最优值。对应 `py/optimization/`，按求解器分为 SciPy、PuLP、OR-Tools 三套共 10 个模板。三套 LP/MILP 模板共用同一标准形式：

$$\max / \min \; c^\top x \quad \text{s.t.} \quad A_{ub} x \le b_{ub}, \quad A_{eq} x = b_{eq}, \quad l_i \le x_i \le u_i$$

| 工具 | 模板 | 适用 |
| --- | --- | --- |
| SciPy | linprog / minimize | 纯连续问题，接口最直接 |
| PuLP | lp / milp | 代数建模，整数与 0-1 变量，默认可直接写 max |
| OR-Tools | lp / milp / cp_sat / assignment / min_cost_flow / routing | 大规模或专用结构：排程、指派、运输网络、路径规划 |

## scipy 线性规划（scipy_linprog.py）

### 原理

目标与约束均为线性时，可行域是一个**凸多面体**，若最优解存在则必在某个顶点取得。`linprog` 的 `highs` 方法（单纯形 / 对偶单纯形 + 预处理）从顶点到顶点迭代改进，直到无可改进方向。注意 `linprog` 固定求**最小值**：最大化问题传 $c = -$收益系数，最优收益为 `-result.fun`。

### 用法

```python
import numpy as np
from py.optimization.scipy_linprog import solve_linprog

c = -np.array([3.0, 5.0])                       # 最大化收益
result = solve_linprog(
    c,
    A_ub=np.array([[2.0, 1.0], [1.0, 2.0]]),     # 2x1 + x2 <= 8 等
    b_ub=np.array([8.0, 8.0]),
    bounds=[(0, None), (0, None)],
)
print(result.x, -result.fun)
```

不等式统一写成 `A_ub @ x <= b_ub`，等式写成 `A_eq @ x == b_eq`；返回 SciPy 的 `OptimizeResult`（`.x` 最优解、`.fun` 目标值、`.success`）。

## scipy 连续优化（scipy_minimize.py）

### 原理

目标或约束**非线性**的连续问题。默认 `SLSQP`（序列二次规划）：在当前解处把问题近似为一个二次规划子问题求解，再迭代更新，直到收敛。注意两点：

- 得到的是**局部最优**，结果依赖初值 `x0`，多初值尝试更稳妥；
- SciPy 约定不等式约束写成 $g(x) \ge 0$，例如 $x_1 + x_2 \le 4$ 要写成 `4 - x[0] - x[1]`。

### 用法

```python
from py.optimization.scipy_minimize import solve_minimize

def objective(x):
    return (x[0] - 2) ** 2 + (x[1] - 3) ** 2

result = solve_minimize(
    objective, x0=[1, 1],
    bounds=[(0, None), (0, None)],
    constraints=[{"type": "ineq", "fun": lambda x: 4 - x[0] - x[1]}],
)
print(result.x, result.fun)
```

无约束时可换 `BFGS`；只有边界约束时 `L-BFGS-B` 更合适。

## PuLP 线性规划（pulp_lp.py）

### 原理

PuLP 是**代数建模**库：变量、目标、约束都以表达式书写，再交给 CBC 求解器。与 SciPy 的区别：目标方向由 `sense` 指定（默认 `LpMaximize`），最大化**不需要**对系数取负；约束仍按矩阵形式传入，模板逐行转为 PuLP 表达式。

### 用法

```python
import numpy as np
import pulp
from py.optimization.pulp_lp import solve_lp

model, variables = solve_lp(
    objective=[3, 5],
    A_ub=np.array([[2, 1], [1, 2]]),
    b_ub=[8, 8],
    bounds=[(0, None), (0, None)],
)
print(pulp.LpStatus[model.status])                      # Optimal
print([variable.value() for variable in variables])     # 最优解
print(pulp.value(model.objective))                      # 目标值
```

最小化传 `sense=pulp.LpMinimize`；`bounds` 中 `None` 表示无界。

## PuLP 混合整数规划（pulp_milp.py）

### 原理

变量被限制为整数或 0-1 后，问题成为组合优化，LP 松弛的最优解一般不再是整数解。求解器用**分支定界**：先解 LP 松弛，若有整数变量取了小数值就分支固定其上/下取整，递归搜索并在界劣于已知解时剪枝。整数建模常用技巧：

- 0-1 变量表达「是否启用」，配 **linking 约束**控制连续量（示例中 `x1 + x2 - 8*open <= 0`：不启用则不能生产）；
- 固定成本、互斥选择、逻辑条件都可以用 0-1 变量 + 大 M 常数表达。

### 用法

```python
import pulp
from py.optimization.pulp_milp import solve_milp

model, variables = solve_milp(
    objective=[3, 5, -4],
    A_ub=[[2, 1, 0], [1, 2, 0], [1, 1, -8]],
    b_ub=[8, 8, 0],
    bounds=[(0, None), (0, None), (0, 1)],
    categories=[pulp.LpInteger, pulp.LpInteger, pulp.LpBinary],
)
print([variable.value() for variable in variables])
```

`categories` 用 `pulp.LpInteger` / `pulp.LpBinary` / `pulp.LpContinuous` 指定每个变量的类型；默认全部整数变量。

## OR-Tools 线性与混合整数（ortools_lp.py / ortools_milp.py）

### 原理

接口与 PuLP 版本一致（同样的矩阵约束和 `bounds`），底层用 OR-Tools `pywraplp`：`ortools_lp.py` 默认 **GLOP** 求解连续 LP，`ortools_milp.py` 默认 **CBC** 求解 MILP。变量类型用字符串指定：`"C"` 连续、`"I"` 整数、`"B"` 0-1；最大化为默认，最小化传 `maximize=False`。

### 用法

```python
from py.optimization.ortools_lp import solve_lp

solver, variables, status = solve_lp(
    objective=[3, 5],
    A_ub=[[2, 1], [1, 2]],
    b_ub=[8, 8],
    bounds=[(0, None), (0, None)],
)
print([variable.solution_value() for variable in variables])
print(solver.Objective().Value())
```

MILP 传 `categories=["I", "I", "B"]`，变量取值同样用 `variable.solution_value()` 读取。

## CP-SAT 排程（ortools_cp_sat.py）

### 原理

约束规划（CP-SAT）不依赖线性结构，核心是**整数/布尔变量 + 逻辑约束**，靠布尔传播和冲突驱动的子句学习搜索。模板把每个任务表示为「开始时间 + 工期 + 结束时间」的区间变量：

- `AddNoOverlap`：单机器同一时刻只能做一个任务；
- `Add(ends[before] <= starts[after])`：先后顺序约束；
- 目标最小化 `makespan`（所有结束时间的最大值），即项目总工期。

CP-SAT 系数必须为整数；含小数时先统一乘 10/100 取整。

### 用法

```python
from py.optimization.ortools_cp_sat import solve_schedule

_, solver, starts, ends, status = solve_schedule(
    durations=[3, 2, 4],
    precedences=[(0, 2)],   # 任务 0 完成后才能开始任务 2
)
print([solver.Value(start) for start in starts])
```

`horizon` 默认取工时总和（时间上界）；`time_limit` 控制最长求解时间。

## CP-SAT 指派（ortools_assignment.py）

### 原理

指派问题的布尔矩阵建模：$x_{ij} = 1$ 表示对象 $i$ 分给任务 $j$，约束「每行恰好一个 1（`AddExactlyOne`）、每列至多一个 1（`AddAtMostOne`）」，最小化 $\sum c_{ij} x_{ij}$。非方阵（对象与任务数不等）也直接支持。小数成本通过 `cost_scale` 放大取整为 CP-SAT 可用的整数。

### 用法

```python
import numpy as np
from py.optimization.ortools_assignment import solve_assignment

cost = np.array([[9, 2, 7, 8], [6, 4, 3, 7], [5, 8, 1, 8]])
_, solver, variables, status = solve_assignment(cost)          # 成本最小化
assignment = [
    next(j for j in range(cost.shape[1]) if solver.Value(variables[i][j]))
    for i in range(cost.shape[0])
]
```

矩阵为收益时传 `maximize=True`。

## 最小费用流（ortools_min_cost_flow.py）

### 原理

在[最大流](graph.md)基础上给每条弧加**单位费用**，在满足供需的前提下最小化总运费 $\sum_{(u,v)} f_{uv} \cdot \text{cost}_{uv}$。约束包括弧容量上限、流守恒，以及节点供给平衡（供给为正、需求为负，总和必须为 0）。运输、供货、资源调配类问题的原生模型。

### 用法

```python
from py.optimization.ortools_min_cost_flow import solve_min_cost_flow

# 弧：(tail, head, capacity, unit_cost)；节点 0、1 供货，2、3 需求
model, status = solve_min_cost_flow(
    4,
    arcs=[(0, 2, 5, 2), (0, 3, 5, 4), (1, 2, 4, 3), (1, 3, 4, 1)],
    supplies=[5, 4, -4, -5],
)
print(model.optimal_cost())
for arc in range(model.num_arcs()):
    if model.flow(arc) > 0:
        print(model.tail(arc), "->", model.head(arc), model.flow(arc))
```

## 旅行商 / 路径规划（ortools_routing.py）

### 原理

TSP 要找经过所有节点并回到起点的最短环路，搜索空间随节点数阶乘增长，精确求解只适用于小规模。`RoutingModel` 用**启发式**策略：先用贪心（`PATH_CHEAPEST_ARC`）构造初始路线，再用元启发式局部搜索（`GUIDED_LOCAL_SEARCH`）在时间预算内持续改进，不保证全局最优但解质量高。

### 用法

```python
import numpy as np
from py.optimization.ortools_routing import solve_tsp

distance_matrix = np.array([
    [0, 4, 8, 7, 6],
    [4, 0, 5, 9, 3],
    [8, 5, 0, 6, 7],
    [7, 9, 6, 0, 4],
    [6, 3, 7, 4, 0],
])
route, distance = solve_tsp(distance_matrix, depot=0)
print(route, distance)   # 首尾都是 depot
```

距离矩阵须为非负**整数**（小数先放大取整）；多车辆、容量、时间窗在此基础上增加对应维度。

## 选型建议

- 连续线性/非线性小问题：SciPy 两个模板最省事。
- 有整数或 0-1 变量的通用问题：PuLP（建模直观、默认最大化）或 OR-Tools（性能更强）。
- 排班调度、复杂逻辑约束：CP-SAT；矩阵指派：`ortools_assignment`；运输网络：最小费用流；TSP/VRP：`ortools_routing`。
- 所有模板都应在论文中检查求解状态（Optimal / FEASIBLE）并做约束余量或灵敏度分析（见[评估模块](evaluation.md)）。
