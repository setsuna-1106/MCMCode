# 分类（classification）

用带标签的数据训练模型预测离散类别。对应 `py/classification/`，提供 6 个模板。统一约定：

- `X`：`(样本数, 特征数)` 数值特征矩阵；`y`：一维分类标签；
- 训练前先做**分层** train/test 划分（各类别比例一致），测试集只用于最终评估；
- 返回 `(model, result)`，`result` 含准确率、分类报告、混淆矩阵（个别模板还含特征重要性）。

| 模板 | 是否需要标准化 | 特点 |
| --- | --- | --- |
| KNN | 是（已内置） | 直观、无需训练，预测慢 |
| SVM | 是（已内置） | 小样本、高维数据强 |
| Logistic（sklearn） | 是（已内置） | 线性、快、带正则化 |
| 随机森林 | 否 | 抗过拟合、给特征重要性 |
| XGBoost | 否 | 精度高、表格数据常用强模型 |
| Logit（statsmodels） | 否（建议自备） | 统计推断：p 值、优势比 |

## KNN（knn.py）

### 原理

「近朱者赤」：不显式训练模型，预测时计算新样本与全部训练样本的距离，取最近的 $k$ 个邻居投票决定类别。

$$\hat{y} = \operatorname{mode}\{ y_i : x_i \in \text{k 近邻}(x) \}$$

$k$ 小容易被局部噪声带偏（过拟合），$k$ 大则决策边界过粗（欠拟合）；`weights="distance"` 让越近的邻居票权越大。因为依赖距离，**必须标准化**，否则量纲大的特征会主导距离计算。

### 优缺点与使用场景

- **优点**：原理直观、没有显式训练过程；天然形成非线性决策边界；可调项少（$k$、权重）。
- **缺点**：预测时需计算与全部训练样本的距离，样本多时又慢又占内存；高维空间距离失效；类别不平衡时偏向多数类。
- **使用场景**：小规模数据集上的快速基线；特征维度不高、需要「相似样本给相似结论」这种可解释性的场合。

### 用法

```python
from py.classification.knn import train_knn

model, result = train_knn(X, y, n_neighbors=5, weights="uniform")
print(result["accuracy"])
predictions = model.predict(X_new)
```

关键参数：`n_neighbors`（邻居数，建议奇数避免平票）、`weights`（`uniform` / `distance`）。

## SVM（svm.py）

### 原理

寻找把两类样本分开且**间隔最大**的超平面 $w^\top x + b = 0$。软间隔允许部分样本越界，目标为：

$$\min \frac{1}{2}\|w\|^2 + C \sum_i \xi_i$$

$C$ 越大对误分类惩罚越重（间隔窄、易过拟合），越小则容忍越界（间隔宽、更平滑）。线性不可分时用**核技巧**把数据隐式映射到高维再找超平面，默认 RBF 核 $K(x, x') = \exp(-\gamma \|x - x'\|^2)$：$\gamma$ 大则每个点影响范围小（易过拟合）。SVM 对特征尺度敏感，模板内置标准化。

### 优缺点与使用场景

- **优点**：小样本、高维数据上表现强；最大间隔带来良好泛化能力；核技巧灵活处理非线性边界。
- **缺点**：大规模样本训练慢；`C` / `gamma` 敏感，需要交叉验证调参；概率输出需额外校准（模板只给类别）。
- **使用场景**：样本量中等以下、特征维度高的数据（文本特征、传感器频谱）；追求决策边界质量的二分类。

### 用法

```python
from py.classification.svm import train_svm

model, result = train_svm(X, y, kernel="rbf", C=1.0, gamma="scale")
```

关键参数：`kernel`（`linear` / `rbf` / `poly`）、`C`（惩罚系数）、`gamma`（RBF 核宽度）。

## Logistic 回归（Logistics.py）

### 原理

对二分类（可扩展多分类），用 sigmoid 把线性组合压成概率：

$$P(y=1 \mid x) = \frac{1}{1 + e^{-(w^\top x + b)}}$$

训练即最大化对数似然（等价于最小化 log-loss），sklearn 默认带 L2 正则，强度由 $1/C$ 控制：$C$ 越小正则越强、系数越收缩。系数符号可直接给出特征方向性解释（概率增减），配合混淆矩阵和 `predict_proba` 概率输出使用。

### 优缺点与使用场景

- **优点**：训练快、输出概率可直接用于排序与阈值调整；系数符号给出特征方向性解释；自带正则化防过拟合。
- **缺点**：本质是线性决策边界，复杂关系需要手工构造特征；对共线性与离群点敏感。
- **使用场景**：需要可解释性的分类基线；风险评估、评分卡等需要概率而非仅类别的场合。

### 用法

```python
from py.classification.Logistics import fit_logistic

result = fit_logistic(X, y, C=1.0)
print(result["accuracy"], result["confusion_matrix"])
print(result["y_proba"])       # 测试集类别概率
```

## 随机森林（rf_iris.py）

### 原理

决策树容易过拟合，随机森林用 **bagging** 平均化方差：每棵树在 bootstrap 重采样的数据子集上训练，节点分裂时只在随机抽取的部分特征中选最优，最后多树投票。

$$\hat{y} = \operatorname{mode}\{\text{tree}_1(x), \dots, \text{tree}_B(x)\}$$

随机化让各棵树的错误互不相关，平均后方差显著下降。`feature_importances_` 按各特征带来的不纯度下降汇总，是论文中常用的特征排序依据。树模型按阈值切分特征，**不需要标准化**。

### 优缺点与使用场景

- **优点**：bagging + 随机特征使模型稳健、不易过拟合；几乎免调参、免标准化；附带特征重要性。
- **缺点**：树多时模型大、预测比单棵树慢；特征高度相关时重要性分配会失真；概率输出偏保守。
- **使用场景**：表格数据的稳健基线；需要回答「哪些特征重要」的分析；不想做预处理的快速建模。

### 用法

```python
from py.classification.rf_iris import train_random_forest

model, result = train_random_forest(
    X, y, feature_names=["a", "b", "c", "d"],
)
print(result["feature_importances"])
```

关键参数：`n_estimators`（树数，越多越稳）、`max_depth`（限制深度防过拟合）。

## XGBoost 分类（xgboost_classification.py）

### 原理

梯度提升：串行训练一系列小树，**每棵新树拟合当前整体的负梯度（残差）**，累加进模型：

$$F_m(x) = F_{m-1}(x) + \eta \cdot T_m(x)$$

学习率 $\eta$（`learning_rate`）收缩每棵树的贡献，配合较多树数通常更稳；`subsample` 和 `colsample_bytree` 分别对样本和特征做随机抽样进一步抗过拟合。二分类用 logistic 目标，多分类自动切换 `multi:softprob`。模板会自动把字符串标签编码为整数并在输出时还原。

### 优缺点与使用场景

- **优点**：正则化 + 二阶梯度 + 行列抽样，表格数据上精度通常最高；训练效率高；能直接处理缺失值。
- **缺点**：超参数多，不调参容易过拟合；黑箱模型，解释性弱于线性模型；小样本上相对随机森林优势不明显。
- **使用场景**：追求预测精度的主力模型；特征关系复杂、存在交互效应的表格数据。

### 用法

```python
from py.classification.xgboost_classification import (
    predict_xgboost_classifier,
    train_xgboost_classifier,
)

model, result = train_xgboost_classifier(X, y)
print(result["accuracy"], result["feature_importances"])
predictions = predict_xgboost_classifier(model, X_new, result["label_encoder"])
```

关键参数：`n_estimators`、`max_depth`、`learning_rate`、`subsample`、`colsample_bytree`。

## Logit 统计建模（sm_logit.py）

### 原理

与 sklearn 的 Logistic 回归同属一个模型，但用 statsmodels 做**极大似然估计且不加正则化**，因此输出完整的统计推断：系数 p 值（特征是否显著）、优势比 $e^{\beta_j}$（该特征每增加 1 单位，事件优势变为原来的多少倍）、LLR p 值（模型整体是否显著）。适合论文中需要「系数显著性与解释」而非单纯预测精度的场景。只支持二分类，`y` 必须是 0/1。

### 优缺点与使用场景

- **优点**：输出完整统计推断——系数 p 值、优势比、模型整体显著性，论文可直接引用；不加正则，估计无收缩偏差。
- **缺点**：只支持二分类（0/1 标签）；高维共线数据下不加正则会不稳定；预测精度通常不是强项。
- **使用场景**：回答「哪些因素显著影响结果、影响多大」的分析型问题，而非纯预测任务。

### 用法

```python
from py.classification.sm_logit import fit_logit, predict_logit

output = fit_logit(X, y, threshold=0.5)
print(output["accuracy"], output["odds_ratio"], output["llr_pvalue"])
print(output["model"].summary())   # 完整回归表
proba, pred = predict_logit(output["model"], X_new)
```

## 注意

- 所有模板的评估指标都来自独立测试集；正式建模时应配合交叉验证选参数（见[评估模块](evaluation.md)的 `evaluate.py`）。
- 类别不平衡时准确率会失真，应看分类报告中的召回率与 F1，或调整 `threshold`。
