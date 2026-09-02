# 优化（optimization）

在约束条件下求目标函数的最优值。对应 `py/optimization/`，按求解器分为 SciPy、PuLP、OR-Tools 三套共 10 个模板。三套 LP/MILP 模板共用同一标准形式：

$$\max / \min \; c^\top x \quad \text{s.t.} \quad A_{ub} x \le b_{ub}, \quad A_{eq} x = b_{eq}, \quad l_i \le x_i \le u_i$$

| 工具 | 模板 | 适用 |
| --- | --- | --- |
| SciPy | linprog / minimize | 纯连续问题，接口最直接 |
| PuLP | lp / milp | 代数建模，整数与 0-1 变量，默认可直接写 max |
| OR-Tools | lp / milp / cp_sat / assignment / min_cost_flow / routing | 大规模或专用结构：排程、指派、运输网络、路径规划 |
| 元启发式 | sa / ga / pso | 黑箱、不可导、强非凸或组合爆炸问题 |

## scipy 线性规划（scipy_linprog.py）

### 原理

目标与约束均为线性时，可行域是一个**凸多面体**，若最优解存在则必在某个顶点取得。`linprog` 的 `highs` 方法（单纯形 / 对偶单纯形 + 预处理）从顶点到顶点迭代改进，直到无可改进方向。注意 `linprog` 固定求**最小值**：最大化问题传 $c = -$收益系数，最优收益为 `-result.fun`。

### 优缺点与使用场景

- **优点**：HiGHS 求解器快而精确，中小规模 LP 秒级出结果；SciPy 生态内闭环，不引入额外建模库。
- **缺点**：固定求最小值，最大化要手动对 $c$ 取负；约束以矩阵形式传入，变量多时建模繁琐；不支持整数变量。
- **使用场景**：连续变量的经典线性规划（生产计划、资源分配、配方），变量与约束数量不大时。

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

### 优缺点与使用场景

- **优点**：一个接口覆盖无约束 / 边界 / 等式 / 不等式约束，算法可按问题切换（SLSQP、BFGS、L-BFGS-B）；轻量，适合参数标定类问题。
- **缺点**：只保证局部最优，结果依赖初值；要求目标可导（至少数值可导），黑箱与强非凸问题低效。
- **使用场景**：连续参数最优化——模型参数标定、带约束的损失最小化、小型非线性规划。

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

### 优缺点与使用场景

- **优点**：目标方向用 `sense` 指定，最大化不用取负；代数式表达比矩阵直观；内置 CBC 求解器开箱即用。
- **缺点**：模板仍以矩阵形式接收约束，复杂逻辑不如逐条书写表达式灵活；CBC 求解大规模 LP 慢于专业求解器。
- **使用场景**：中小规模线性规划，或作为 PuLP 建模语法的起点（比赛中最常用的 LP 写法之一）。

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

### 优缺点与使用场景

- **优点**：整数 / 0-1 变量能直接表达「是否启用、互斥、固定成本」等逻辑决策；分支定界给出精确最优解，结论有最优性证明。
- **缺点**：最坏情况指数时间，规模大时可能超时；大 M、linking 等建模技巧取值不当会显著拖慢求解。
- **使用场景**：选址、生产计划、背包 / 装载等含离散决策的组合优化问题。

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

### 优缺点与使用场景

- **优点**：与 PuLP 版本同一套接口，迁移成本低；GLOP / CBC 是工业级实现，同样规模下求解更快更稳；与 CP-SAT、路由等专用求解器同生态，便于后续升级。
- **缺点**：结果要用 `solution_value()` 等原生 API 读取，写法略啰嗦；MILP 规模上限仍受 CBC 能力约束。
- **使用场景**：LP / MILP 规模较大或求解时间敏感时的升级选项。

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

### 优缺点与使用场景

- **优点**：`AddNoOverlap` 等约束直接表达「互斥占用资源」，逻辑建模能力远强于线性规划；冲突驱动搜索在大排程问题上效率高；可设时限先拿可行解再改进。
- **缺点**：所有系数必须为整数，小数缩放带来精度损失；目标形式限于线性 / 区间组合；建模概念多，上手门槛较高。
- **使用场景**：排班排程、机器调度、资源冲突类问题——约束复杂但目标相对简单。

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

### 优缺点与使用场景

- **优点**：布尔矩阵建模直白，行列约束直接对应指派规则；非方阵（人数 ≠ 任务数）天然支持；求解快。
- **缺点**：只能表达「每人一项、每项至多一人」的纯指派；技能限制、容量、多目标等复杂要求要自行加约束。
- **使用场景**：人员-任务、机组-航线等一对一分配；比[图模块](graph.md)的匹配模板多了成本矩阵与非方阵封装。

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

### 优缺点与使用场景

- **优点**：容量、费用、供需三类信息一体化建模，是运输问题的原生模型；网络流算法高效，规模大也不怕；同时给出每条弧的流量分配方案。
- **缺点**：只支持单一种类流（多种货物不能直接共模型）；供给总和必须与需求平衡；容量与费用须为整数。
- **使用场景**：多产地到多销地的运输方案、供货网络、物资调配与总运费最小化。

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

### 优缺点与使用场景

- **优点**：专为大规模 TSP / VRP 设计，实践中远胜通用求解器与手写启发式；多车辆、容量、时间窗等约束有原生支持；时限内解质量持续改进。
- **缺点**：启发式搜索不保证全局最优；距离矩阵必须为非负整数；模型、维度、搜索参数概念多，学习成本高。
- **使用场景**：配送路线、巡检路线、多车带容量 / 时间窗约束的路径规划（VRP）。

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

## 元启发式优化（sa.py / ga.py / pso.py）

### 原理

前面的求解器都依赖问题结构：LP/MILP 要求线性、`minimize` 要求目标可导且只保证局部最优。当目标是**仿真结果、查表规则**等黑箱（不可导），或景观**强非凸、坑坑洼洼**，或解空间是**排列组合**时，用元启发式兜底：把优化问题抽象为「给一个候选解、返回一个好坏分数」，在有界空间内随机搜索，通过**有策略地接受劣解**跳出局部最优，在有限预算内给出足够好的解——但不保证最优。

三个模板共用同一套约定：`objective(x) -> 标量`（黑箱函数）、`bounds` 给出有界搜索空间、`maximize` 切换方向、`random_state` 固定种子保证可复现；返回字典统一为 `x_best`（最优解）、`fun_best`（原始目标值）、`history`（逐步/逐代最优值序列）、`n_evaluations`（目标评价次数）。

| 模板 | 搜索方式 | 跳出局部最优的机制 | 适用 |
| --- | --- | --- | --- |
| SA 模拟退火 | 单点迭代 + 邻域扰动 | Metropolis 准则：以 $\exp(-\Delta E/T)$ 概率接受劣解，温度几何降温 | 实现最简；连续离散通用 |
| GA 遗传算法 | 种群进化 | 变异维持多样性 + 交叉重组；锦标赛选择 + 精英保留 | 离散/组合问题（改写编码与交叉变异即可） |
| PSO 粒子群 | 群体协作 | 粒子同时拉向个体历史最佳 pbest 与全群最佳 gbest，多点并行 | 连续问题，收敛通常最快 |

### 优缺点与使用场景

- **SA 模拟退火**——优点：实现最简单，单点迭代内存占用小，连续 / 离散问题通用；缺点：结果对降温日程敏感，单点搜索在复杂景观中容易停滞。
- **GA 遗传算法**——优点：种群并行探索、抗局部最优能力强，改写编码即可适配各类组合问题；缺点：实现最重（编码、交叉、变异都要设计），超参数多，可能早熟收敛。
- **PSO 粒子群**——优点：连续问题收敛通常最快，参数少、实现量中等；缺点：高维或离散问题需改造，多样性丧失后易早熟。
- **共同取舍**：不保证最优、结果带随机性，必须多随机种子运行并报告最好 / 最差 / 均值；但只要能「给候选解打分」就能优化，是黑箱问题的最后兜底。

### 用法

```python
from py.optimization.pso import particle_swarm

def objective(x):
    return (x[0] - 2.0) ** 2 + (x[1] + 1.0) ** 2   # 任意黑箱函数

result = particle_swarm(objective, [(-10.0, 10.0), (-10.0, 10.0)],
                        n_particles=30, iterations=100)
print(result["x_best"], result["fun_best"])
```

`simulated_annealing` / `genetic_algorithm` 接口相同；收敛过程用 `result["history"]` 记录，可直接交给[可视化模块](visualization.md)的 `plot_line` 画收敛曲线。

### 注意

- **没有最优性证明**：论文中应说明为启发式解，用多个随机种子独立运行，报告最好/最差/均值以证明稳定；
- 结果随机，固定 `random_state` 保证复现；参数（温度、种群、惯性等）敏感，建议附收敛曲线；
- GA 处理组合问题（排列、0-1）时替换编码与交叉/变异算子，进化框架不变；
- 能用精确求解器就用精确的——有最优性证明，评审更认可。

## 选型建议

- 连续线性/非线性小问题：SciPy 两个模板最省事。
- 有整数或 0-1 变量的通用问题：PuLP（建模直观、默认最大化）或 OR-Tools（性能更强）。
- 排班调度、复杂逻辑约束：CP-SAT；矩阵指派：`ortools_assignment`；运输网络：最小费用流；TSP/VRP：`ortools_routing`。
- 目标黑箱、不可导或强非凸：元启发式——连续优先 PSO，离散/组合改 GA 编码，快速实现用 SA。
- 所有模板都应在论文中检查求解状态（Optimal / FEASIBLE）并做约束余量或灵敏度分析（见[评估模块](evaluation.md)）。
