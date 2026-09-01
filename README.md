# MCMCode

这是本人的在准备数学建模竞赛中积累的常用算法与数据处理代码仓库。
以“最小化、可运行、便于改写”为目标，适合比赛时快速复制模板，再根据题目数据补充清洗、建模和可视化部分。

## 快速开始

### 1. 准备环境

项目使用 Python 3.10+，依赖记录在 `requirements.txt` 中。

可以创建新的虚拟环境：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

### 2. 运行示例

所有 Python 模板都可以从项目根目录直接运行：

```bash
.venv/bin/python py/classification/knn.py
.venv/bin/python py/classification/svm.py
.venv/bin/python py/classification/xgboost_classification.py
.venv/bin/python py/forecasting/gm11.py
.venv/bin/python py/forecasting/ese.py
.venv/bin/python py/forecasting/holt.py
.venv/bin/python py/forecasting/arima.py
.venv/bin/python py/regression/LinearRegression.py
.venv/bin/python py/regression/scipy_curve_fit.py
.venv/bin/python py/regression/xgboost_regression.py
.venv/bin/python py/optimization/scipy_minimize.py
.venv/bin/python py/optimization/scipy_linprog.py
.venv/bin/python py/optimization/pulp_lp.py
.venv/bin/python py/optimization/pulp_milp.py
.venv/bin/python py/optimization/ortools_lp.py
.venv/bin/python py/optimization/ortools_milp.py
.venv/bin/python py/optimization/ortools_cp_sat.py
.venv/bin/python py/optimization/ortools_assignment.py
.venv/bin/python py/optimization/ortools_min_cost_flow.py
.venv/bin/python py/optimization/ortools_routing.py
.venv/bin/python py/optimization/sa.py
.venv/bin/python py/optimization/ga.py
.venv/bin/python py/optimization/pso.py
.venv/bin/python py/evaluation/sensitivity.py
.venv/bin/python py/evaluation/ahp.py
.venv/bin/python py/evaluation/gra.py
.venv/bin/python py/graph/networkx_basics.py
.venv/bin/python py/graph/networkx_shortest_path.py
.venv/bin/python py/graph/networkx_mst.py
.venv/bin/python py/graph/networkx_flow.py
.venv/bin/python py/graph/networkx_matching.py
.venv/bin/python py/graph/networkx_critical_path.py
.venv/bin/python py/graph/networkx_centrality.py
.venv/bin/python py/ode/scipy_solve_ivp.py
.venv/bin/python py/interpolation/scipy_interpolate.py
.venv/bin/python py/integration/scipy_integrate.py
.venv/bin/python py/regression/sm_ols.py
.venv/bin/python py/classification/sm_logit.py
.venv/bin/python py/statistics/sm_tests.py
.venv/bin/python py/statistics/normality_tests.py
.venv/bin/python py/statistics/nonparametric_tests.py
.venv/bin/python py/statistics/correlation_tests.py
.venv/bin/python py/statistics/bootstrap_interval.py
.venv/bin/python py/visualization/plot_style.py
.venv/bin/python py/visualization/heatmap_plots.py
.venv/bin/python py/visualization/model_plots.py
.venv/bin/python py/visualization/analysis_plots.py
.venv/bin/python py/visualization/basic_plots.py
```

其中机器学习示例默认使用 sklearn 自带的 Iris 数据集；预处理模板使用自包含的合成数据演示，实际使用时替换为 `pd.read_csv` / `pd.read_excel` 读入的数据：

```bash
.venv/bin/python py/preprocessing/data_clean.py
.venv/bin/python py/preprocessing/outlier_detection.py
.venv/bin/python py/preprocessing/encode_categorical.py
```

## 项目结构

```text
MCMCode/
├── cpp/                         # C++ 代码目录
├── docs/                        # 各模块的原理与用法文档
├── py/                          # Python 模板包
│   ├── [preprocessing/](docs/preprocessing.md)           # 数据清洗
│   ├── [evaluation/](docs/evaluation.md)              # 熵权、TOPSIS 和模型评估
│   ├── [graph/](docs/graph.md)                   # 图论和网络分析
│   ├── [forecasting/](docs/forecasting.md)             # GM(1,1)、指数平滑和 ARIMA
│   ├── [classification/](docs/classification.md)          # 分类模型
│   ├── [clustering/](docs/clustering.md)              # 聚类模型
│   ├── [dimensionality_reduction/](docs/dimensionality_reduction.md) # PCA 降维模型
│   ├── [regression/](docs/regression.md)              # 回归模型和任意函数拟合
│   ├── [optimization/](docs/optimization.md)            # 优化模型（SciPy、PuLP、OR-Tools）
│   ├── [ode/](docs/ode.md)                     # 常微分方程数值求解
│   ├── [interpolation/](docs/interpolation.md)           # 插值模型
│   ├── [integration/](docs/integration.md)             # 数值积分计算
│   ├── [statistics/](docs/statistics.md)              # 统计假设检验
│   └── [visualization/](docs/visualization.md)        # 论文绘图
├── requirements.txt             # Python 依赖
└── README.md
```

目录名表示建模用途；
所有模块接口保持彼此独立，使用时直接替换示例中的 `X`、`y` 或数据表。

## Python 模块

| 文件 | 类型 | 主要用途 |
| --- | --- | --- |
| [`py/preprocessing/data_clean.py`](docs/preprocessing.md) | 数据处理 | 缺失值报告、数值列中位数填充和去重 |
| [`py/preprocessing/outlier_detection.py`](docs/preprocessing.md) | 数据处理 | IQR 与 3σ 异常值检测及盖帽处理 |
| [`py/preprocessing/encode_categorical.py`](docs/preprocessing.md) | 数据处理 | one-hot 哑变量编码与标签编码 |
| [`py/evaluation/entropy_weight.py`](docs/evaluation.md) | 评价方法 | 极差标准化后计算熵权 |
| [`py/evaluation/topsis.py`](docs/evaluation.md) | 评价方法 | 根据权重和指标方向计算 TOPSIS 贴近度并排序 |
| [`py/forecasting/gm11.py`](docs/forecasting.md) | 预测 | 使用 GM(1,1) 对非负时间序列进行短期预测 |
| [`py/forecasting/ese.py`](docs/forecasting.md) | 预测 | 使用一次指数平滑进行短期预测 |
| [`py/forecasting/holt.py`](docs/forecasting.md) | 预测 | Holt 二次指数平滑和 Holt-Winters 三次指数平滑 |
| [`py/forecasting/arima.py`](docs/forecasting.md) | 预测 | statsmodels ARIMA 评估和未来预测 |
| [`py/dimensionality_reduction/pca.py`](docs/dimensionality_reduction.md) | 降维 | PCA 主成分、贡献率、累计贡献率和载荷分析 |
| [`py/evaluation/evaluate.py`](docs/evaluation.md) | 模型评估 | 以 SVM 为例，演示交叉验证、网格调参和测试集评估 |
| [`py/evaluation/sensitivity.py`](docs/evaluation.md) | 灵敏度分析 | 局部、一因素、双因素和 Monte Carlo 参数扰动 |
| [`py/evaluation/ahp.py`](docs/evaluation.md) | 评价方法 | AHP 权重、层次总排序和一致性检验 |
| [`py/evaluation/gra.py`](docs/evaluation.md) | 评价方法 | 灰色关联度、指标方向和加权排序 |
| [`py/graph/networkx_basics.py`](docs/graph.md) | 图论 | 建图、遍历和连通性分析 |
| [`py/graph/networkx_shortest_path.py`](docs/graph.md) | 图论 | 加权最短路径 |
| [`py/graph/networkx_mst.py`](docs/graph.md) | 图论 | 最小生成树和网络建设成本 |
| [`py/graph/networkx_flow.py`](docs/graph.md) | 图论 | 最大流和最小割 |
| [`py/graph/networkx_matching.py`](docs/graph.md) | 图论 | 最大权匹配和指派关系 |
| [`py/graph/networkx_critical_path.py`](docs/graph.md) | 图论 | DAG 关键路径和项目工期 |
| [`py/graph/networkx_centrality.py`](docs/graph.md) | 图论 | 度、介数、接近中心性和 PageRank |
| [`py/regression/sm_ols.py`](docs/regression.md) | 统计建模 | OLS 回归、测试集指标和回归诊断 |
| [`py/regression/sm_diagnostics.py`](docs/regression.md) | 回归诊断 | 残差正态性、异方差、自相关和 VIF |
| [`py/classification/sm_logit.py`](docs/classification.md) | 统计建模 | Logit 二分类、优势比和分类评估 |
| [`py/statistics/sm_tests.py`](docs/statistics.md) | 统计检验 | Welch t、配对 t、卡方、Fisher、单因素 ANOVA、Levene 和 Tukey HSD |
| [`py/statistics/normality_tests.py`](docs/statistics.md) | 统计检验 | Shapiro-Wilk、Jarque-Bera 和 Lilliefors 正态性检验 |
| [`py/statistics/nonparametric_tests.py`](docs/statistics.md) | 统计检验 | Mann-Whitney U、Kruskal-Wallis 和 Wilcoxon 符号秩 |
| [`py/statistics/correlation_tests.py`](docs/statistics.md) | 统计检验 | Pearson 和 Spearman 相关分析 |
| [`py/statistics/bootstrap_interval.py`](docs/statistics.md) | 统计检验 | 任意统计量的非参数 Bootstrap 置信区间 |
| [`py/visualization/plot_style.py`](docs/visualization.md) | 绘图 | 统一论文绘图风格、中文字体和图片导出 |
| [`py/visualization/heatmap_plots.py`](docs/visualization.md) | 绘图 | 相关系数与通用矩阵热力图 |
| [`py/visualization/model_plots.py`](docs/visualization.md) | 绘图 | 回归、分类、聚类和 PCA 的模型诊断图 |
| [`py/visualization/analysis_plots.py`](docs/visualization.md) | 绘图 | 灵敏度分析和时间序列预测图 |
| [`py/visualization/basic_plots.py`](docs/visualization.md) | 绘图 | 折线、散点、柱状、箱线图等基础图 |
| [`py/classification/knn.py`](docs/classification.md) | 分类 | 标准化 + KNN 分类 |
| [`py/classification/svm.py`](docs/classification.md) | 分类 | 标准化 + SVM 分类 |
| [`py/classification/xgboost_classification.py`](docs/classification.md) | 分类 | XGBoost 分类、评估和特征重要性 |
| [`py/classification/Logistics.py`](docs/classification.md) | 分类 | 标准化 + Logistic 回归分类 |
| [`py/classification/rf_iris.py`](docs/classification.md) | 分类 | 随机森林分类与特征重要性 |
| [`py/clustering/kmeans.py`](docs/clustering.md) | 聚类 | 标准化 + KMeans 聚类 |
| [`py/clustering/dbscan.py`](docs/clustering.md) | 聚类 | 标准化 + DBSCAN 聚类与噪声识别 |
| [`py/regression/LinearRegression.py`](docs/regression.md) | 回归 | 线性回归与常用回归指标 |
| [`py/regression/PolynomialRegression.py`](docs/regression.md) | 回归 | 多项式回归与多项式特征 |
| [`py/regression/RidgeRegression.py`](docs/regression.md) | 回归 | 带 L2 正则化的岭回归 |
| [`py/regression/scipy_curve_fit.py`](docs/regression.md) | 回归 | scipy 任意函数拟合与参数评估 |
| [`py/regression/xgboost_regression.py`](docs/regression.md) | 回归 | XGBoost 回归、评估和特征重要性 |
| [`py/optimization/scipy_linprog.py`](docs/optimization.md) | 优化 | scipy 线性规划、资源约束和变量边界 |
| [`py/optimization/scipy_minimize.py`](docs/optimization.md) | 优化 | scipy 连续优化、变量边界和非线性约束 |
| [`py/optimization/pulp_lp.py`](docs/optimization.md) | 优化 | PuLP 连续线性规划、资源约束和变量边界 |
| [`py/optimization/pulp_milp.py`](docs/optimization.md) | 优化 | PuLP 混合整数线性规划、整数变量和 0-1 变量 |
| [`py/optimization/ortools_lp.py`](docs/optimization.md) | 优化 | OR-Tools GLOP 连续线性规划 |
| [`py/optimization/ortools_milp.py`](docs/optimization.md) | 优化 | OR-Tools CBC 混合整数线性规划 |
| [`py/optimization/ortools_cp_sat.py`](docs/optimization.md) | 优化 | OR-Tools CP-SAT 单机器排程和先后约束 |
| [`py/optimization/ortools_assignment.py`](docs/optimization.md) | 优化 | OR-Tools CP-SAT 指派问题 |
| [`py/optimization/ortools_min_cost_flow.py`](docs/optimization.md) | 优化 | OR-Tools 最小费用流和运输网络 |
| [`py/optimization/ortools_routing.py`](docs/optimization.md) | 优化 | OR-Tools RoutingModel 单车辆 TSP |
| [`py/optimization/sa.py`](docs/optimization.md) | 优化 | 模拟退火：黑箱目标的单点随机搜索和退火接受 |
| [`py/optimization/ga.py`](docs/optimization.md) | 优化 | 实数编码遗传算法：选择、交叉、变异和精英保留 |
| [`py/optimization/pso.py`](docs/optimization.md) | 优化 | 粒子群：惯性、个体认知和社会认知的速度更新 |
| [`py/ode/scipy_solve_ivp.py`](docs/ode.md) | 微分方程 | scipy 常微分方程数值积分和事件检测 |
| [`py/interpolation/scipy_interpolate.py`](docs/interpolation.md) | 插值 | scipy 线性、三次样条和 PCHIP 插值 |
| [`py/integration/scipy_integrate.py`](docs/integration.md) | 数值积分 | scipy 一维定积分、误差估计和反常积分 |

## 常用接口

展示常用接口的使用示例，使用时要遵守注释中的规范

### 数据清洗

```python
import pandas as pd
from py.preprocessing.data_clean import handle_missing

df = pd.read_csv("附件.csv")
df = handle_missing(df)
```

`handle_missing` 只处理数值列；文本列、分类列和日期列需要根据题意单独处理。配套的 `missing_report` 报告各列缺失比例，`drop_duplicates` 在划分数据前去除重复行。异常值检测与分类编码：

```python
from py.preprocessing.encode_categorical import one_hot_encode
from py.preprocessing.outlier_detection import (
    cap_outliers,
    detect_outliers_iqr,
)

iqr = detect_outliers_iqr(df)            # IQR 法检测，返回掩码与边界
df = cap_outliers(df, iqr)               # 盖帽到边界（或 df.mask(iqr["mask"]) 置 NaN）

encoded, info = one_hot_encode(df, drop_first=True)   # 分类列转哑变量
```

### 灵敏度分析

`sensitivity.py` 统一使用 `model(params) -> 标量` 接口，包含四种常用分析：

```python
import numpy as np
from py.evaluation.sensitivity import (
    local_sensitivity,
    monte_carlo_sensitivity,
    one_way_sensitivity,
    two_way_sensitivity,
)

def model(params):
    return params["price"] * params["quantity"] - params["cost"]

base_params = {"price": 10.0, "quantity": 80.0, "cost": 100.0}

# 局部灵敏度：中心差分导数和弹性系数
local = local_sensitivity(model, base_params, "price", step=0.01)

# 一因素：固定其他参数，扫描 quantity
one_way = one_way_sensitivity(
    model, base_params, "quantity", np.linspace(60, 100, 9)
)

# 双因素：输出矩阵的行对应 price，列对应 cost
two_way = two_way_sensitivity(
    model, base_params, "price", [8, 10, 12], "cost", [80, 100, 120]
)

# Monte Carlo：sampler 每次返回一组扰动后的参数
def sampler(rng):
    return {
        "price": rng.uniform(8, 12),
        "quantity": rng.uniform(70, 90),
        "cost": rng.uniform(80, 120),
    }

monte_carlo = monte_carlo_sensitivity(model, sampler, n_samples=2000)
print(local["elasticity"])
print(one_way["outputs"])
print(two_way["outputs"])
print(monte_carlo["correlation"])
```

局部分析适合判断基准点附近的影响；一因素和双因素分析适合画折线图、热力图；Monte Carlo 适合参数存在区间或概率分布时判断输出稳定性。相关系数只能反映单变量线性关联，最终应结合输出分布、分位数和题目实际意义解释。

### AHP 层次分析法

`ahp.py` 使用最大特征值法计算判断矩阵权重，并完成准则层、方案层和总排序。判断矩阵必须是正互反矩阵；通常 `CR < 0.1` 时通过一致性检验。

```python
import numpy as np
from py.evaluation.ahp import solve_ahp

criteria = np.array([
    [1, 2, 4],
    [1 / 2, 1, 2],
    [1 / 4, 1 / 2, 1],
])
alternatives = [
    np.array([[1, 2], [1 / 2, 1]]),
    np.array([[1, 1 / 3], [3, 1]]),
    np.array([[1, 4], [1 / 4, 1]]),
]

result = solve_ahp(criteria, alternatives, require_consistent=True)
print("准则权重:", result["criteria"]["weights"])
print("方案总权重:", result["global_weights"])
print("最优方案编号:", result["best_index"] + 1)
```

`result["criteria"]` 和 `result["alternatives"]` 中包含 `lambda_max`、`CI`、`RI`、`CR` 和 `consistent`；若只分析一个判断矩阵，直接调用 `ahp_weights`。判断矩阵不一致时，优先检查两两比较值及其倒数关系，再决定是否重新赋值。

### 灰色关联度分析

`gra.py` 用参考序列和多个比较序列计算灰色关联系数与灰色关联度，支持收益型/成本型指标、指标权重和分辨系数 `rho`：

```python
import numpy as np
from py.evaluation.gra import grey_relation

reference = [1.0, 1.0, 0.0]
comparison = np.array([
    [0.9, 0.8, 0.4],
    [0.8, 0.9, 0.3],
    [0.7, 0.95, 0.5],
])

result = grey_relation(
    reference,
    comparison,
    direction=[1, 1, -1],
    weights=[0.4, 0.4, 0.2],
    rho=0.5,
)
print("灰色关联度:", result["grades"])
print("方案排序:", result["order"] + 1)
```

`reference` 是理想参考序列，`comparison` 每一行是一个方案；`direction` 中 `1` 表示收益型指标，`-1` 表示成本型指标。默认先按指标列做极差标准化，再计算关联系数；若数据已经完成无量纲化，可传入 `normalize=False`。

### NetworkX 图论模板

NetworkX 模板统一使用节点和边表示网络结构，边属性根据问题选择 `weight`、`capacity` 或 `duration`。当前模板覆盖建图、最短路径、最小生成树、最大流、匹配、关键路径和中心性分析：

```python
import networkx as nx
from py.graph.networkx_shortest_path import solve_shortest_path

graph = nx.Graph()
graph.add_weighted_edges_from([
    ("A", "B", 2),
    ("B", "C", 1),
    ("C", "D", 3),
])

result = solve_shortest_path(graph, "A", "D")
print(result["path"])
print(result["distance"])
```

使用建议：`weight` 表示距离或成本，`capacity` 表示流量上限，`duration` 表示任务时长；最短路径使用 Dijkstra，网络建设使用最小生成树，运输瓶颈使用最大流/最小割，任务依赖使用 DAG 关键路径。NetworkX 负责图结构和图算法，复杂整数优化或车辆路径规划应结合 PuLP、SciPy 或 OR-Tools。

### 熵权法

```python
import numpy as np
from py.evaluation.entropy_weight import entropy_weight

X = np.array([
    [10, 100],
    [20, 80],
    [30, 120],
])
direction = [
    -1,  # 成本型指标：越小越好
    1,   # 收益型指标：越大越好
]
weights = entropy_weight(X, direction)
print(weights)
```

约定：`X` 的每一行是一个方案或样本，每一列是一个指标；`direction` 长度应与指标数一致，`1` 表示收益型指标，`-1` 表示成本型指标。

### TOPSIS

```python
import numpy as np
from py.evaluation.topsis import topsis

w = np.array([0.4, 0.6])
direction = np.array([1, -1])
C, order, D_plus, D_minus = topsis(X, w, direction)
print("贴近度:", C)
print("从优到劣的行号:", order)
```

`C` 越大表示方案越接近正理想解；`order` 是按 `C` 从大到小排列后的行号。使用前应确认输入矩阵没有缺失值，且权重已归一化。

### GM(1,1)

```python
from py.forecasting.gm11 import gm11

forecast, (a, b), fitted = gm11(
    [12, 15, 19, 24, 30],
    steps=3,
)
print("参数:", a, b)
print("拟合值:", fitted)
print("预测值:", forecast)
```

返回值依次为：未来 `steps` 个预测值、参数 `(a, b)`、原始样本对应的拟合值。GM(1,1) 适合样本量较小、趋势较明显的序列；正式建模时还应检查残差、后验差比和适用性。

### 一次指数平滑

```python
import numpy as np
from py.forecasting.ese import exponential_smoothing

y = np.array([102, 105, 107, 111, 115, 114, 119, 123], dtype=float)
level, forecast = exponential_smoothing(y, alpha=0.3, steps=3)
print("平滑值:", level)
print("预测值:", forecast)
```

`y` 是按时间排列的一维序列，`alpha` 越大越重视最新观测值。返回的 `level` 是历史平滑值，`forecast` 是未来 `steps` 期预测；该方法不单独建模趋势和季节性。

### Holt 二次和三次指数平滑

```python
import numpy as np
from py.forecasting.holt import fit_holt, fit_holt_winters

y = np.array(
    [100, 106, 111, 104, 108, 114, 120, 112, 117, 123, 129, 121],
    dtype=float,
)

# 二次指数平滑：水平项 + 趋势项
holt = fit_holt(y, steps=4, damped_trend=True)
print("Holt 预测:", holt["forecast"])

# 三次指数平滑：水平项 + 趋势项 + 季节项
holt_winters = fit_holt_winters(
    y,
    seasonal_periods=4,
    steps=4,
    trend="add",
    seasonal="add",
)
print("Holt-Winters 预测:", holt_winters["forecast"])
```

`fit_holt` 适合有趋势但无明显季节性的序列；`fit_holt_winters` 需要设置季节周期，通常至少提供两个完整季节。`trend` 和 `seasonal` 可选 `"add"` 或 `"mul"`；使用乘法项时，`y` 必须全部大于 0。`damped_trend=True` 会逐步减弱远期趋势，适合不希望趋势无限增长的场景。两个函数都返回 `model`、`fitted`、`forecast` 和 `method`。

### ARIMA

```python
from py.forecasting.arima import fit_arima, predict_arima

output = fit_arima(y, order=(1, 1, 1), test_size=0.2)
print(output["rmse"], output["mape"])

# 确定阶数后，用全部历史数据重新拟合，再预测未来。
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(y, order=output["order"]).fit()
future = predict_arima(model, steps=3)
```

ARIMA 按时间顺序使用前段数据训练、后段数据测试，不随机打乱序列。`order=(p, d, q)` 分别表示自回归阶数、差分阶数和移动平均阶数。

### 机器学习分类与聚类

分类模板通常接收：

- `X`：形状为 `(样本数, 特征数)` 的数值特征矩阵；
- `y`：形状为 `(样本数,)` 的分类标签。

以 SVM 为例：

```python
from py.classification.svm import train_svm

model, result = train_svm(X, y)
print(result["accuracy"])
predictions = model.predict(X_new)
```

`train_knn`、`train_svm`、`fit_logistic` 和 `train_random_forest` 都会划分训练集与测试集并返回模型和结果。KNN、SVM、Logistic 回归使用了 `StandardScaler`；标准化被放在 Pipeline 中，以避免测试集信息泄漏到训练过程。

### XGBoost 分类与回归

分类和回归模板分别使用 `XGBClassifier` 与 `XGBRegressor`。XGBoost 是树模型，通常不需要标准化；输入 `X` 仍需是数值特征矩阵。

```python
from py.classification.xgboost_classification import (
    predict_xgboost_classifier,
    train_xgboost_classifier,
)
from py.regression.xgboost_regression import train_xgboost_regressor

classifier, classification_result = train_xgboost_classifier(X, y_class)
print(classification_result["accuracy"])
print(predict_xgboost_classifier(
    classifier,
    X_new,
    classification_result["label_encoder"],
))

regressor, regression_result = train_xgboost_regressor(X, y_value)
print(regression_result["rmse"], regression_result["r2"])
print(regressor.predict(X_new))
```

`n_estimators`、`max_depth`、`learning_rate`、`subsample` 和 `colsample_bytree` 是最常用的调参入口；正式建模时应在训练集上配合交叉验证或验证集调参，避免测试集信息泄漏。

回归模板位于 `py/regression/`，提供 `fit_linear_regression`、`fit_polynomial_regression` 和 `fit_ridge_regression`。它们都返回训练后的 `model`、测试集预测值和 `mae`、`mse`、`rmse`、`r2` 等指标；`standardize=True` 时，标准化在 Pipeline 中完成。

### scipy 任意函数拟合

```python
import numpy as np
from py.regression.scipy_curve_fit import (
    fit_curve,
    fit_curve_multi,
    predict_curve,
    predict_curve_multi,
)

def model_func(x, a, b, c):
    return a * np.exp(-b * x) + c

output = fit_curve(
    model_func,
    x,
    y,
    p0=(1, 1, 0),
    bounds=([-np.inf, 0, -np.inf], [np.inf, np.inf, np.inf]),
)
print("参数:", output["params"])
print("RMSE:", output["rmse"])
future = predict_curve(model_func, x_new, output["params"])
```

多元函数使用 `fit_curve_multi`，`X` 按 `(样本数, 特征数)` 传入：

```python
def plane(X, a, b, c):
    x1, x2 = X
    return a * x1 + b * x2 + c

output = fit_curve_multi(plane, X, y, p0=(1, 1, 0))
future = predict_curve_multi(plane, X_new, output["params"])
```

一元函数必须写成 `model_func(x, *params)`；多元函数接收的 `X` 形状为 `(特征数, 样本数)`。`p0` 是参数初值，`bounds` 是参数上下界。两种入口都会先划分训练集和测试集，再用 `curve_fit` 拟合训练数据并评估测试数据。

### scipy 线性规划

`scipy_linprog.py` 使用标准形式求解连续线性规划：

```python
import numpy as np
from py.optimization.scipy_linprog import solve_linprog

c = -np.array([3.0, 5.0])  # 最大化收益时取负，linprog 本身执行最小化
A_ub = np.array([[2.0, 1.0], [1.0, 2.0]])
b_ub = np.array([8.0, 8.0])
bounds = [(0, None), (0, None)]

result = solve_linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
print("最优变量:", result.x)
print("最大收益:", -result.fun)
```

不等式约束统一写成 `A_ub @ x <= b_ub`，等式约束写成 `A_eq @ x == b_eq`。需要整数变量时，也可以使用下面的 PuLP 模板。

### PuLP 线性规划

`pulp_lp.py` 使用 PuLP 的代数建模方式求解连续线性规划，默认最大化目标：

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
print("状态:", pulp.LpStatus[model.status])
print("最优变量:", [variable.value() for variable in variables])
print("目标值:", pulp.value(model.objective))
```

`objective` 是目标函数系数；`bounds` 中的 `None` 表示无界。最小化时传入 `sense=pulp.LpMinimize`。函数返回 `(model, variables)`，约束仍按 `A_ub @ x <= b_ub` 和 `A_eq @ x == b_eq` 传入。

### PuLP 混合整数线性规划

`pulp_milp.py` 与 LP 模板的约束写法相同，通过 `categories` 指定变量类型：

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
print("最优变量:", [variable.value() for variable in variables])
print("目标值:", pulp.value(model.objective))
```

`categories` 可使用 `pulp.LpInteger`、`pulp.LpBinary` 和 `pulp.LpContinuous`；默认全部为整数变量。带有整数或 0-1 决策变量的生产、选址、分配和排班问题可从此模板改写。

### OR-Tools 常用优化模板

OR-Tools 模板按问题结构选择接口：`pywraplp` 用于 LP/MILP，`CP-SAT` 用于整数逻辑和排程，最小费用流用于网络运输，`RoutingModel` 用于 TSP/VRP。

#### OR-Tools 线性规划与混合整数规划

`ortools_lp.py` 默认使用 `GLOP` 求解连续 LP；`ortools_milp.py` 默认使用 `CBC_MIXED_INTEGER_PROGRAMMING` 求解 MILP。两个模板都使用 `A_ub @ x <= b_ub`、`A_eq @ x == b_eq` 和 `bounds`：

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

MILP 中通过 `categories=["I", "I", "B"]` 指定整数、整数和 0-1 变量；连续变量使用 `"C"`。最大化使用默认参数，最小化传入 `maximize=False`。

#### CP-SAT 排程与指派

`ortools_cp_sat.py` 使用整数开始时间、结束时间和 `AddNoOverlap` 表示单机器排程；`ortools_assignment.py` 使用布尔变量表示对象-任务分配。CP-SAT 适合排班、先后关系、逻辑条件和组合约束，不适合作为普通连续优化器。

```python
from py.optimization.ortools_cp_sat import solve_schedule

_, solver, starts, ends, status = solve_schedule(
    durations=[3, 2, 4],
    precedences=[(0, 2)],
)
print([solver.Value(start) for start in starts])
```

CP-SAT 的系数和时间通常使用整数；若原始数据含小数，先统一乘以 `10`、`100` 等倍数再取整。

#### 最小费用流与路径规划

`ortools_min_cost_flow.py` 使用 `(tail, head, capacity, unit_cost)` 表示网络弧，节点供给为正、需求为负，适合运输和资源流动问题。`ortools_routing.py` 使用 `RoutingModel` 求解 TSP；车辆容量、时间窗和多车辆问题应在此基础上增加对应维度。

### scipy 连续优化

`scipy_minimize.py` 使用 `scipy.optimize.minimize` 求解带边界或非线性约束的连续优化问题：

```python
from py.optimization.scipy_minimize import solve_minimize

def objective(x):
    x1, x2 = x
    return (x1 - 2) ** 2 + (x2 - 3) ** 2

result = solve_minimize(
    objective,
    x0=[1, 1],
    bounds=[(0, None), (0, None)],
    constraints=[
        {"type": "ineq", "fun": lambda x: 4 - x[0] - x[1]},
    ],
)
print("最优变量:", result.x)
print("最小目标值:", result.fun)
```

`SLSQP` 的不等式约束必须写成 `g(x) >= 0`；例如 `x1 + x2 <= 4` 应写成 `4 - x1 - x2`。无约束问题可以省略 `bounds` 和 `constraints`，再根据需要选择 `BFGS` 或 `L-BFGS-B`。

### scipy 常微分方程

`scipy_solve_ivp.py` 使用 `scipy.integrate.solve_ivp` 求解初值问题：

```python
import numpy as np
from py.ode.scipy_solve_ivp import solve_ode

def rhs(t, y, rate):
    return [-rate * y[0]]

result = solve_ode(
    rhs,
    t_span=(0, 10),
    y0=[1.0],
    t_eval=np.linspace(0, 10, 101),
    args=(0.3,),
)
print("时间:", result.t)
print("状态:", result.y)
```

方程函数必须返回与 `y0` 同长度的导数向量；`t_eval` 控制输出时刻，`args` 传入方程参数，`events` 可用于检测达到阈值、碰撞或终止条件。默认使用 `RK45`，刚性方程可改用 `BDF` 或 `Radau`。

### scipy 插值

`scipy_interpolate.py` 用观测点构造一维插值函数：

```python
import numpy as np
from py.interpolation.scipy_interpolate import interpolate_1d

x = np.array([0.0, 1.0, 2.0, 3.0])
y = np.array([0.0, 1.0, 4.0, 9.0])
x_new = np.linspace(0.0, 3.0, 31)
y_new = interpolate_1d(x, y, x_new, kind="cubic")
```

`linear` 稳定且不易过冲，`cubic` 更平滑，`pchip` 适合保持单调性的数据。默认不外推观测区间外的点，若确认外推合理，再设置 `extrapolate=True`。

### scipy 数值积分

`scipy_integrate.py` 使用 `scipy.integrate.quad` 计算一维定积分，并返回积分值和误差估计：

```python
import numpy as np
from py.integration.scipy_integrate import integrate_1d

def integrand(x, rate):
    return np.exp(-rate * x)

value, error = integrate_1d(
    integrand,
    0.0,
    np.inf,
    args=(0.3,),
)
print("积分值:", value)
print("误差估计:", error)
```

`args` 用于传递被积函数参数；函数存在已知间断点、尖点或剧烈变化时，将分割点传给 `points`，例如 `points=[1.0, 2.0]`。积分值应结合误差估计、函数连续性和题目物理意义检查。

### statsmodels 统计建模

连续型目标使用 OLS：

```python
from py.regression.sm_ols import fit_ols, diagnose_ols

output = fit_ols(X, y, robust=True)
print(output["rmse"], output["r2"])
print(output["model"].summary())
print(diagnose_ols(output["model"], output["X_train"]))
```

也可以直接使用独立的诊断模板：

```python
from py.regression.sm_diagnostics import fit_ols_diagnostics

output = fit_ols_diagnostics(X, y, feature_names=feature_names)
print(output["diagnostics"])
```

诊断结果包括 Jarque-Bera 正态性检验、Breusch-Pagan 异方差检验、Durbin-Watson 自相关统计量和各特征 VIF。

二分类目标使用 Logit，`y` 必须是 `0/1`：

```python
from py.classification.sm_logit import fit_logit

output = fit_logit(X, y)
print(output["accuracy"])
print(output["odds_ratio"])
print(output["model"].summary())
```

这两个模板都先划分训练集和测试集；`summary()`、系数、p 值和优势比来自训练集，MAE/RMSE/R² 或准确率来自独立测试集。

常用假设检验：

```python
from py.statistics.sm_tests import chi_square, one_way_anova, welch_ttest

print(welch_ttest(group_a, group_b))
print(chi_square([[180, 20], [150, 50]]))
print(one_way_anova(df, target="yield", group="plan"))
```

聚类模板 `train_kmeans` 和 `train_dbscan` 接收二维特征矩阵 `X`，返回模型和聚类结果。DBSCAN 的标签 `-1` 表示噪声点；KMeans 需要预先指定 `n_clusters`。

## 推荐的建模流程

```text
读取数据
  -> 缺失值与异常值处理
  -> 选择指标、构造特征
  -> 按题意进行标准化或指标正负向处理
  -> 训练模型 / 评价排序 / 预测
  -> 用交叉验证、残差或敏感性分析检验结果
  -> 输出表格、图像和论文中的模型解释
```

使用时的相关注意事项：

1. `X` 的行列含义必须在论文和代码中保持一致。
2. 评价指标的收益型、成本型方向要先确认，再传入 `direction`。
3. 标准化、缺失值填补和特征选择都应只使用训练数据拟合，避免数据泄漏。
4. 算法输出不能替代题目分析；应结合假设、误差、稳定性和敏感性分析解释结论。

## 依赖

- Python 3.10+
- NumPy 2.x
- pandas 2.x
- scikit-learn 1.5+
- statsmodels 0.14+
- matplotlib 3.8+
- seaborn 0.13+
- XGBoost 2.1+
- PuLP 2.9+
- OR-Tools 9.15+
- NetworkX 3.3+

具体版本范围见 [`requirements.txt`](requirements.txt)。

## 参考资料

### SciPy

- [SciPy 官方文档](https://docs.scipy.org/doc/scipy/)
- [SciPy 用户指南](https://docs.scipy.org/doc/scipy/tutorial/index.html)
- [`scipy.optimize.minimize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
- [`scipy.optimize.linprog`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html)
- [`scipy.optimize.milp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html)
- [`scipy.optimize.curve_fit`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html)
- [`scipy.integrate.solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)
- [`scipy.integrate.quad`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html)
- [`scipy.interpolate.interp1d`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html)
- [`scipy.interpolate.CubicSpline`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html)
- [`scipy.interpolate.PchipInterpolator`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html)

### PuLP

- [PuLP 官方文档](https://coin-or.github.io/pulp/)
- [PuLP GitHub](https://github.com/coin-or/pulp)

### OR-Tools

- [OR-Tools 官方文档](https://developers.google.com/optimization)
- [OR-Tools Python API](https://or-tools.github.io/docs/python/)

### NetworkX

- [NetworkX 官方文档](https://networkx.org/documentation/stable/)
- [NetworkX 算法文档](https://networkx.org/documentation/stable/reference/algorithms/index.html)

### Python 科学计算

- [Python 官方文档](https://docs.python.org/3/)
- [NumPy 官方文档](https://numpy.org/doc/stable/)
- [pandas 官方文档](https://pandas.pydata.org/docs/)
- [scikit-learn 官方文档](https://scikit-learn.org/stable/)
- [scikit-learn 用户指南](https://scikit-learn.org/stable/user_guide.html)
- [statsmodels 官方文档](https://www.statsmodels.org/stable/)
- [statsmodels 用户指南](https://www.statsmodels.org/stable/user-guide.html)

### XGBoost

- [XGBoost 官方文档](https://xgboost.readthedocs.io/en/stable/)
- [XGBoost Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html)
- [XGBoost 参数说明](https://xgboost.readthedocs.io/en/stable/parameter.html)
