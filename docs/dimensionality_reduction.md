# 降维（dimensionality_reduction）

把高维指标压缩成少数几个综合变量，同时尽量保留原始信息。对应模板 `py/dimensionality_reduction/pca.py`。常见用途：指标太多导致的共线性处理、二维可视化、为后续建模构造综合得分。

## pca_analysis

### 原理

主成分分析（PCA）寻找使数据**方差最大**的投影方向。对（标准化后的）数据矩阵 $X$ 求协方差矩阵 $\Sigma = \frac{1}{n-1} X^\top X$，其特征值与特征向量 $\{\lambda_j, v_j\}$ 给出：

- 第 $j$ 主成分 = 数据在 $v_j$ 上的投影得分：$z_j = X v_j$；
- $\lambda_j$ 是该方向上的方差，**贡献率** $= \lambda_j / \sum_k \lambda_k$，即该主成分解释的信息占比；
- 各主成分互不相关（正交），按 $\lambda_j$ 从大到小排列，前几个通常就能覆盖大部分方差。

`standardize=True` 时先按列做 z-score 标准化：量纲不同的指标若不标准化，方差大的列会霸占第一主成分。**载荷**（loadings）是主成分与原始指标的相关系数，$v_j \sqrt{\lambda_j}$，用于解释每个主成分「主要代表哪些原始指标」。

### 用法

```python
import numpy as np
from py.dimensionality_reduction.pca import pca_analysis

result = pca_analysis(X, n_components=0.9)  # 保留累计贡献率 90%
# 或 pca_analysis(X, n_components=2)        # 固定保留 2 个主成分

print(result["contribution"])             # 各主成分贡献率
print(result["cumulative_contribution"])  # 累计贡献率
print(result["scores"])                   # (样本数, 主成分数) 的主成分得分
print(result["loadings"])                 # (指标数, 主成分数) 的载荷
```

- 入参 `X`：`(样本数, 指标数)` 矩阵，每行一个样本。
- `n_components`：整数表示保留的主成分个数；`0~1` 的小数表示目标累计贡献率（如 `0.9`）。
- `standardize`：是否先按列标准化，指标量纲不同时保持 `True`。
- 返回字典常用字段：`scores`（主成分得分）、`contribution` / `cumulative_contribution`（贡献率）、`loadings`（载荷）、`eigenvalues`（特征值）、`pca` / `scaler`（拟合对象）。
- 直接运行模板：

```bash
.venv/bin/python py/dimensionality_reduction/pca.py
```

### 注意

- 主成分是原始指标的线性组合，本身没有直接物理含义，需借助载荷大小进行命名和解释。
- PCA 是无监督的，不使用标签；若目标是提高预测效果，可考虑监督式的降维（如 LDA）。
- 累计贡献率的常用阈值是 80%–90%，论文中应同时报告各主成分贡献率与载荷表。
