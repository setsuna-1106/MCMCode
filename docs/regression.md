# 回归（regression）

预测连续型目标值。对应 `py/regression/`，提供 7 个模板。统一约定：`X` 为 `(样本数, 特征数)` 矩阵、`y` 为连续目标；先划分 train/test，返回 `model` 与含 `mae` / `mse` / `rmse` / `r2` 的结果字典。

| 模板 | 模型形式 | 适用 |
| --- | --- | --- |
| LinearRegression | $y = w^\top x + b$ | 线性关系基线 |
| PolynomialRegression | 多项式展开后线性 | 曲线趋势 |
| RidgeRegression | 线性 + L2 正则 | 多重共线性、特征多 |
| scipy_curve_fit | 任意给定函数 | 已知机理形式（增长、衰减等） |
| sm_ols | 线性 + 统计推断 | 需要系数 p 值、置信区间 |
| sm_diagnostics | 回归诊断 | 检验 OLS 假设是否成立 |
| xgboost_regression | 梯度提升树 | 非线性、表格数据、精度优先 |

## 线性回归（LinearRegression.py）

### 原理

最小化残差平方和（OLS），有解析解 $\hat{w} = (X^\top X)^{-1} X^\top y$：

$$\min_{w, b} \sum_i \left( y_i - w^\top x_i - b \right)^2$$

系数 $w_j$ 表示其他特征不变时 $x_j$ 每增加 1 单位引起的 $y$ 平均变化量。`standardize=True`（默认）把标准化放进 Pipeline，此时系数相互可比；关闭后系数直接对应原始量纲。

### 优缺点与使用场景

- **优点**：有解析解、训练快而稳定；系数有清晰的边际解释（标准化后还可比大小）；是一切回归任务的对照基线。
- **缺点**：只能拟合线性关系；对多重共线性和离群点敏感；非线性与交互效应需要手工构造特征。
- **使用场景**：关系近似线性的建模；需要解释「某因素每增加一单位，目标平均变化多少」的问题；复杂模型前的第一版结果。

### 用法

```python
from py.regression.LinearRegression import fit_linear_regression

result = fit_linear_regression(X, y, standardize=True)
print(result["r2"], result["coef"], result["intercept"])
predictions = result["model"].predict(X_new)
```

## 多项式回归（PolynomialRegression.py）

### 原理

先用 `PolynomialFeatures` 把特征**展开**成高次项与交叉项（如 $(x_1, x_2) \to x_1^2, x_1 x_2, x_2^2, \dots$），再对展开后的特征做普通线性回归。整体仍是对系数的线性模型，但拟合的是曲线：

$$y = \beta_0 + \sum_j \beta_j x_j + \sum_{j \le k} \beta_{jk}\, x_j x_k + \cdots$$

次数 `degree` 越高越灵活，特征数也组合式增长，极易过拟合（高次多项式摆动剧烈）。

### 优缺点与使用场景

- **优点**：以线性模型的技术代价捕捉曲线关系，实现与理解成本低；交叉项可表达特征交互。
- **缺点**：特征数随次数组合式增长；高次项使曲线剧烈摆动，过拟合与外推失控风险大。
- **使用场景**：单变量或低维数据的曲线趋势拟合；物理量之间光滑的非线性关系。

### 用法

```python
from py.regression.PolynomialRegression import fit_polynomial_regression

result = fit_polynomial_regression(X, y, degree=2)
print(result["feature_names"])   # 展开后的多项式特征名
```

### 注意

- `degree` 一般不超过 3~4；对比相邻次数的测试集 $R^2$ 判断是否继续加复杂度。
- 展开后强烈依赖标准化，保持 `standardize=True`。

## 岭回归（RidgeRegression.py）

### 原理

在 OLS 基础上给系数加 **L2 惩罚**，牺牲无偏性换取方差下降：

$$\min_{w} \; \| y - Xw \|^2 + \alpha \| w \|^2$$

$\alpha$ 越大，系数被整体收缩得越小，模型越平滑。当特征之间存在**多重共线性**（$X^\top X$ 接近奇异、OLS 系数剧烈波动）时，岭回归显著更稳定。惩罚作用在系数尺度上，因此标准化与模型绑定在 Pipeline 中。

### 优缺点与使用场景

- **优点**：L2 惩罚消除 $X^\top X$ 的病态，共线 / 宽表下系数稳定、方差小；只有一个超参数 $\alpha$，好调。
- **缺点**：估计有偏；不产生稀疏解，变量一个都不删；$\alpha$ 需交叉验证选择。
- **使用场景**：特征多（one-hot 后的宽表）、特征间高度相关、普通线性回归系数异常波动时。

### 用法

```python
from py.regression.RidgeRegression import fit_ridge_regression

result = fit_ridge_regression(X, y, alpha=1.0)
```

`alpha` 可用交叉验证扫参选取（见[评估模块](evaluation.md)）。

## 任意函数拟合（scipy_curve_fit.py）

### 原理

题目已知机理形式（Logistic 增长、指数衰减、多项式物理公式等）时，直接对参数做**非线性最小二乘**：`curve_fit` 从初值 $p_0$ 出发，用 Levenberg–Marquardt / Trust Region Reflective 算法迭代调整参数使 $\sum_i (y_i - f(x_i; p))^2$ 最小。给定 `bounds` 时自动切换有界算法，参数被约束在物理合理范围内。

初值 $p_0$ 和边界对收敛影响很大：初值差可能停在局部极小或直接不收敛，应结合参数量级粗略估计给出。

### 优缺点与使用场景

- **优点**：拟合的是题给机理公式，参数有物理含义，结果可解释、可在机理成立范围内外推；能施加参数边界保证物理合理。
- **缺点**：需要人工指定函数形式与初值，初值不当不收敛或陷入局部极小；形式选错则一切结论失效。
- **使用场景**：已知增长 / 衰减 / 饱和等形式机理的建模（Logistic 人口、指数衰减、产能饱和曲线）。

### 用法

```python
import numpy as np
from py.regression.scipy_curve_fit import (
    fit_curve, fit_curve_multi, predict_curve,
)

def model_func(x, a, b, c):        # 一元：model_func(x, *params)
    return a * np.exp(-b * x) + c

output = fit_curve(model_func, x, y, p0=(7, 0.5, 1))
print(output["params"], output["rmse"])
future = predict_curve(model_func, x_new, output["params"])

# 多元：model_func(X, *params)，X 形状 (特征数, 样本数)
def plane(X, a, b, c):
    x1, x2 = X
    return a * x1 + b * x2 + c

output = fit_curve_multi(plane, X_2d, y, p0=(1, 1, 0))
```

### 注意

- 模板只在训练子集上估参数、在测试子集上算 RMSE；初值不当报错时优先调整 `p0`。

## OLS 统计建模（sm_ols.py）

### 原理

与 sklearn 线性回归同一模型，但 statsmodels 输出**统计推断**：系数标准误、t 值、p 值、置信区间、整体显著性（F 检验）。`robust=True` 使用 HC3 稳健协方差，在异方差存在时仍给出可信的标准误与 p 值。需要手动 `add_constant` 补截距列，模板已处理（训练与预测使用相同列结构）。

### 优缺点与使用场景

- **优点**：完整统计推断（t 检验、置信区间、F 检验）支撑「影响是否显著」的论证；`robust=True` 在异方差下仍可信。
- **缺点**：模型本身的局限与线性回归相同；预测性能不是目标；结论依赖假设成立（配合诊断模板）。
- **使用场景**：论文中需要报告系数显著性与区间的回归分析（政策效应、因素影响类问题）。

### 用法

```python
from py.regression.sm_ols import fit_ols, predict_ols

output = fit_ols(X, y, robust=True)
print(output["r2"])
print(output["model"].summary())     # 完整回归表
predictions = predict_ols(output["model"], X_new)
```

## 回归诊断（sm_diagnostics.py）

### 原理

OLS 的推断（p 值、置信区间）建立在若干假设上，该模板一次给出四类检验：

| 假设 | 检验 | 统计量 | 判读 |
| --- | --- | --- | --- |
| 残差正态 | Jarque-Bera | JB 及 p 值 | p > 0.05 不拒绝正态 |
| 同方差 | Breusch-Pagan | LM 及 p 值 | p < 0.05 存在异方差 |
| 无自相关 | Durbin-Watson | DW ∈ [0, 4] | 接近 2 无自相关；趋 0 正相关，趋 4 负相关 |
| 无多重共线 | VIF | 每特征一个值 | 经验上 > 10 严重共线 |

异方差时用 `robust=True` 的稳健标准误；共线性强时改用岭回归；残差自相关多见于时间序列，应改用时序模型。

### 优缺点与使用场景

- **优点**：一份输出覆盖 OLS 四大假设的检查，直接指明补救方向（稳健标准误 / 岭回归 / 时序模型）。
- **缺点**：各检验自身也有前提（如 Jarque-Bera 依赖较大样本）；只能「查出问题」，修复要配合相应对策。
- **使用场景**：把 OLS 结果写进论文前的例行检查；回应「回归假设是否成立」这类评审质疑。

### 用法

```python
from py.regression.sm_diagnostics import diagnose_ols, fit_ols_diagnostics

# 方式一：对已拟合的 OLS 结果诊断（X 为拟合用的原始特征矩阵）
diag = diagnose_ols(ols_result, X, feature_names)

# 方式二：一步拟合 + 诊断
output = fit_ols_diagnostics(X, y, feature_names=names)
print(output["diagnostics"])
```

## XGBoost 回归（xgboost_regression.py）

### 原理

与[分类模块](classification.md)的 XGBoost 同源：串行加法训练一系列回归树，每棵新树拟合当前整体的负梯度，$F_m(x) = F_{m-1}(x) + \eta\, T_m(x)$，目标为平方误差。能自动捕捉非线性与特征交互，不需要标准化，附带 `feature_importances_`。代价是可解释性弱于线性模型。

### 优缺点与使用场景

- **优点**：自动捕捉非线性与特征交互，表格数据精度高；免标准化；附带特征重要性。
- **缺点**：解释性弱于线性模型；树模型无法外推出训练数据的目标范围（预测值被限制在已见区间）；超参数多需调优。
- **使用场景**：精度优先、关系复杂难写机理公式的回归任务；线性模型精度不足时的升级选项。

### 用法

```python
from py.regression.xgboost_regression import train_xgboost_regressor

model, result = train_xgboost_regressor(X, y)
print(result["rmse"], result["r2"], result["feature_importances"])
predictions = model.predict(X_new)
```

关键参数：`n_estimators`、`max_depth`、`learning_rate`、`subsample`、`colsample_bytree`。
