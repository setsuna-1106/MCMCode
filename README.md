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
.venv/bin/python py/forecasting/gm11.py
.venv/bin/python py/forecasting/ese.py
.venv/bin/python py/forecasting/arima.py
.venv/bin/python py/regression/LinearRegression.py
.venv/bin/python py/regression/sm_ols.py
.venv/bin/python py/classification/sm_logit.py
.venv/bin/python py/statistics/sm_tests.py
```

其中机器学习示例默认使用 sklearn 自带的 Iris 数据集；运行 `data_clean.py` 前，需要在项目根目录放置名为 `附件.csv` 的输入文件：

```bash
.venv/bin/python py/preprocessing/data_clean.py
```

## 项目结构

```text
MCMCode/
├── cpp/                         # C++ 代码目录
├── py/                          # Python 模板包
│   ├── preprocessing/           # 数据清洗
│   ├── evaluation/              # 熵权、TOPSIS 和模型评估
│   ├── forecasting/             # GM(1,1)、指数平滑和 ARIMA
│   ├── classification/          # 分类模型
│   ├── clustering/              # 聚类模型
│   ├── dimensionality_reduction/ # PCA
│   ├── regression/              # 回归模型
│   └── statistics/              # 假设检验
├── requirements.txt             # Python 依赖
└── README.md
```

目录名表示建模用途；模块接口保持独立，使用时直接替换示例中的 `X`、`y` 或数据表。

## Python 模块

| 文件 | 类型 | 主要用途 |
| --- | --- | --- |
| `py/preprocessing/data_clean.py` | 数据处理 | 读取根目录 `附件.csv`，用各数值列的中位数填充缺失值 |
| `py/evaluation/entropy_weight.py` | 评价方法 | 极差标准化后计算熵权 |
| `py/evaluation/topsis.py` | 评价方法 | 根据权重和指标方向计算 TOPSIS 贴近度并排序 |
| `py/forecasting/gm11.py` | 预测 | 使用 GM(1,1) 对非负时间序列进行短期预测 |
| `py/forecasting/ese.py` | 预测 | 使用一次指数平滑进行短期预测 |
| `py/forecasting/arima.py` | 预测 | statsmodels ARIMA 评估和未来预测 |
| `py/dimensionality_reduction/pca.py` | 降维 | PCA 主成分、贡献率、累计贡献率和载荷分析 |
| `py/evaluation/evaluate.py` | 模型评估 | 以 SVM 为例，演示交叉验证、网格调参和测试集评估 |
| `py/regression/sm_ols.py` | 统计建模 | OLS 回归、测试集指标和回归诊断 |
| `py/regression/sm_diagnostics.py` | 回归诊断 | 残差正态性、异方差、自相关和 VIF |
| `py/classification/sm_logit.py` | 统计建模 | Logit 二分类、优势比和分类评估 |
| `py/statistics/sm_tests.py` | 统计检验 | Welch t、卡方和单因素 ANOVA |
| `py/classification/knn.py` | 分类 | 标准化 + KNN 分类 |
| `py/classification/svm.py` | 分类 | 标准化 + SVM 分类 |
| `py/classification/Logistics.py` | 分类 | 标准化 + Logistic 回归分类 |
| `py/classification/rf_iris.py` | 分类 | 随机森林分类与特征重要性 |
| `py/clustering/kmeans.py` | 聚类 | 标准化 + KMeans 聚类 |
| `py/clustering/dbscan.py` | 聚类 | 标准化 + DBSCAN 聚类与噪声识别 |
| `py/regression/LinearRegression.py` | 回归 | 线性回归与常用回归指标 |
| `py/regression/PolynomialRegression.py` | 回归 | 多项式回归与多项式特征 |
| `py/regression/RidgeRegression.py` | 回归 | 带 L2 正则化的岭回归 |

## 常用接口

展示常用接口使用示例

### 数据清洗

```python
import pandas as pd
from py.preprocessing.data_clean import handle_missing

df = pd.read_csv("附件.csv")
df = handle_missing(df)
```

`handle_missing` 只处理数值列；文本列、分类列和日期列需要根据题意单独处理。

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

回归模板位于 `py/regression/`，提供 `fit_linear_regression`、`fit_polynomial_regression` 和 `fit_ridge_regression`。它们都返回训练后的 `model`、测试集预测值和 `mae`、`mse`、`rmse`、`r2` 等指标；`standardize=True` 时，标准化在 Pipeline 中完成。

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

具体版本范围见 [`requirements.txt`](requirements.txt)。

## 参考资料

https://scikit-learn.org
https://www.statsmodels.org
