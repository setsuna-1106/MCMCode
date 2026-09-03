# MCMCode

这是本人的在准备数学建模竞赛中积累的常用算法与数据处理代码仓库。
以“最小化、可运行、便于改写”为目标，适合比赛时快速复制模板，再根据题目数据补充清洗、建模和可视化部分。
各模块的方法原理与使用说明按模块整理在 [docs/](docs/README.md)；README 提供快速开始、模板索引和统一约定。

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
├── [docs/](docs/README.md)      # 各模块的原理与用法文档（含索引）
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

## 模块文档与统一约定

各模块的**方法原理、接口用法和选型建议**统一整理在 [`docs/`](docs/README.md)，可按建模需求从[文档索引](docs/README.md)查找；模板源码与文档的对应关系见上方「Python 模块」表格。

所有建模模板遵循统一约定，以分类为例：

```python
from py.classification.svm import train_svm

model, result = train_svm(X, y)    # X: (样本数, 特征数) 特征矩阵；y: 一维标签
print(result["accuracy"])          # 指标来自独立测试集
predictions = model.predict(X_new) # 新数据预测
```

- 输入 `X` 为 `(样本数, 特征数)` 数值矩阵，`y`（或序列）为对应标签、目标或时间序列；
- 涉及 train/test 划分的模板统一返回 `(model, result)`，`result` 含评估指标与中间结果；
- 标准化、缺失填充等预处理放在 Pipeline 内、只在训练集上拟合，避免测试集信息泄漏；
- 评价类接口统一用 `direction` 表示指标方向：`1` 收益型（越大越好）、`-1` 成本型（越小越好）。

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
