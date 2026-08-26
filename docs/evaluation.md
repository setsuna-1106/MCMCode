# 综合评价与模型评估（evaluation）

两类工具：**评价排序**（AHP、熵权、TOPSIS、灰色关联——把多指标方案排出名次）和**结果检验**（灵敏度分析、evaluate 标准流程）。对应 `py/evaluation/`。

通用数据约定：决策矩阵 `X` 形状为 `(方案数, 指标数)`，每行一个方案、每列一个指标；`direction` 用 `1` 标收益型指标（越大越好）、`-1` 标成本型指标（越小越好）。

## AHP 层次分析法（ahp.py）

### 原理

**主观赋权**方法。把决策问题拆成「准则层—方案层」，对同层元素做两两比较，填入正互反判断矩阵（$a_{ij} > 0$，$a_{ii}=1$，$a_{ij} a_{ji}=1$，常用 Saaty 1–9 标度）。权重取**最大特征值** $\lambda_{\max}$ 对应特征向量的归一化结果。

人的两两比较可能自相矛盾（A>B、B>C 却 C>A），因此做一致性检验：

$$CI = \frac{\lambda_{\max} - n}{n - 1}, \qquad CR = \frac{CI}{RI}$$

$RI$ 是同阶随机矩阵的平均一致性指标（模板内置 1~15 阶表）。**CR < 0.1 通过检验**，否则需要重新审视比较值。方案总权重 = 各准则下的局部权重矩阵右乘准则权重：$\text{global} = \text{local} \cdot w_{\text{criteria}}$。

### 用法

```python
import numpy as np
from py.evaluation.ahp import solve_ahp, ahp_weights

criteria = np.array([[1, 2, 4], [1/2, 1, 2], [1/4, 1/2, 1]])
alternatives = [np.array([[1, 2], [1/2, 1]]), ...]   # 每个准则一个方案矩阵

result = solve_ahp(criteria, alternatives, require_consistent=True)
print(result["criteria"]["weights"])   # 准则权重
print(result["global_weights"])        # 方案总权重
print(result["best_index"] + 1)        # 最优方案编号

# 只分析单个判断矩阵：
single = ahp_weights(criteria)
```

### 注意

- 主观性强，适合指标难以全部量化、需要体现决策者偏好的题目。
- 阶数超过 2 时一致性检验才有效；不通过时优先检查两两比较及其倒数关系。

## 熵权法（entropy_weight.py）

### 原理

**客观赋权**方法，权重完全由数据差异决定。先把各指标按方向做极差标准化到 $[0,1]$，再计算列内比例 $p_{ij}$ 与信息熵：

$$e_j = -\frac{1}{\ln m} \sum_{i=1}^{m} p_{ij} \ln p_{ij} \in [0, 1]$$

熵越小 → 该指标取值差异越大 → 携带信息越多 → **差异系数** $d_j = 1 - e_j$ 越大 → 权重 $w_j = d_j / \sum_k d_k$ 越高。若某列几乎不变（无区分度），熵趋近 1，权重自动趋近 0。

### 用法

```python
import numpy as np
from py.evaluation.entropy_weight import entropy_weight

X = np.array([[10, 100], [20, 80], [30, 120]])
direction = [-1, 1]   # 第一列成本型，第二列收益型
weights = entropy_weight(X, direction)
```

### 注意

- 客观但不体现题意偏好；数据异常值会直接扭曲权重。
- 常与 TOPSIS 组成「熵权 TOPSIS」；也可与 AHP 权重取平均做主客观组合赋权。

## TOPSIS（topsis.py）

### 原理

**逼近理想解排序**：量化每个方案到「最好方案」（正理想解）和「最差方案」（负理想解）的距离。步骤：

1. 各列向量归一化 $x_{ij} / \sqrt{\sum_i x_{ij}^2}$ 消除量纲；成本型列取倒数转成「越大越好」；
2. 乘权重得加权矩阵 $V$，取每列最大/最小构成正、负理想解 $A^+, A^-$；
3. 计算欧氏距离 $D_i^+, D_i^-$，**贴近度**：

$$C_i = \frac{D_i^-}{D_i^+ + D_i^-} \in [0, 1]$$

$C_i$ 越大表示越贴近正理想解、同时离最差方案越远，方案越优。

### 用法

```python
import numpy as np
from py.evaluation.topsis import topsis

w = np.array([0.4, 0.6])
direction = np.array([1, -1])
C, order, D_plus, D_minus = topsis(X, w, direction)
print("贴近度:", C, "从优到劣的行号:", order)
```

- 入参 `w` 应已归一化；返回 `(C, order, D_plus, D_minus)`。
- 权重可来自熵权法或 AHP，形成完整的组合评价流程。

## 灰色关联度分析（gra.py）

### 原理

度量各方案序列与**参考序列**（通常取各指标的最优值构成的理想序列）的几何相似程度。先按方向极差标准化，再计算每个方案在各指标上与参考的差距 $\Delta_{ij}$，转换为关联系数：

$$\gamma_{ij} = \frac{\Delta_{\min} + \rho\, \Delta_{\max}}{\Delta_{ij} + \rho\, \Delta_{\max}}$$

差距越小关联系数越接近 1；分辨系数 $\rho$（默认 0.5）调节区分度。关联度为关联系数的加权平均，越大表示该方案与理想序列越接近。

### 用法

```python
import numpy as np
from py.evaluation.gra import grey_relation

reference = [1.0, 1.0, 0.0]                    # 理想参考序列
comparison = np.array([[0.9, 0.8, 0.4], ...])  # 每行一个方案

result = grey_relation(
    comparison=comparison,
    reference=reference,
    direction=[1, 1, -1],
    weights=[0.4, 0.4, 0.2],
    rho=0.5,
)
print(result["grades"], result["order"] + 1)
```

- 数据已无量纲化时传 `normalize=False`。
- 对小样本、指标含义模糊的数据比 TOPSIS 更稳健。

## 灵敏度分析（sensitivity.py）

### 原理

检验结论对参数扰动的稳健性——评审常问「参数变了结论还成立吗」。统一接口 `model(params) -> 标量`，提供四种分析：

| 方法 | 做法 | 输出与用途 |
| --- | --- | --- |
| `local_sensitivity` | 中心差分 $\frac{f(p_0+h) - f(p_0-h)}{2h}$ | 导数 + **弹性系数**（参数变 1% 输出变百分之几），判断基准点附近谁最敏感 |
| `one_way_sensitivity` | 固定其他参数，扫描一个参数 | 输出曲线，画折线图 |
| `two_way_sensitivity` | 同时扫描两个参数 | 输出矩阵（行对应 x、列对应 y），画热力图 |
| `monte_carlo_sensitivity` | 按 `sampler` 随机扰动多参数 | 输出分布 + 各参数与输出的 Pearson 相关系数 |

### 用法

```python
import numpy as np
from py.evaluation.sensitivity import (
    local_sensitivity, monte_carlo_sensitivity,
    one_way_sensitivity, two_way_sensitivity,
)

def model(params):
    return params["price"] * params["quantity"] - params["cost"]

base = {"price": 10.0, "quantity": 80.0, "cost": 100.0}

local = local_sensitivity(model, base, "price", step=0.01)
one_way = one_way_sensitivity(model, base, "quantity", np.linspace(60, 100, 9))
two_way = two_way_sensitivity(model, base, "price", [8, 10, 12], "cost", [80, 100, 120])

def sampler(rng):
    return {"price": rng.uniform(8, 12), "quantity": rng.uniform(70, 90),
            "cost": rng.uniform(80, 120)}

mc = monte_carlo_sensitivity(model, sampler, n_samples=2000)
```

### 注意

- 相关系数只反映单变量线性关联，最终应结合输出分布与题目实际意义解释。
- 比赛写作套路：选 2~3 个关键参数，一因素折线 + 双因素热力图，说明结论在扰动下稳健。

## 机器学习标准流程（evaluate.py）

### 原理

以 SVM 为例演示防数据泄漏的完整评估链：**划分 → Pipeline → 交叉验证 → 网格调参 → 测试**。两个关键点：

- 标准化放在 **Pipeline** 内：交叉验证的每一折都只用该折训练数据重新拟合 scaler，测试信息不进入训练；
- **测试集只在最后使用一次**：交叉验证和网格搜索都只碰训练集，最后用从未参与任何环节的测试集报告最终指标。

### 用法

```bash
.venv/bin/python py/evaluation/evaluate.py
```

脚本式模板（无函数接口），替换其中的 `X`、`y`、`param_grid` 与模型即可，`clf__C` 这类参数名对应 Pipeline 中 `clf` 步骤的超参数。交叉验证的均值±标准差反映稳定性，`GridSearchCV` 的 `best_params_` 给出最优组合。

## 组合建议

- 经典组合：**熵权（或 AHP）定权重 → TOPSIS / GRA 排序 → 灵敏度分析检验稳健性**。
- AHP 偏主观、熵权偏客观，题目要求「主客观结合」时可将两套权重加权平均后再进 TOPSIS。
