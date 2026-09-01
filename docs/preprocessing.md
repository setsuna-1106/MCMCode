# 数据预处理（preprocessing）

建模的第一步：把原始数据表处理成可建模的干净输入。本模块对应 `py/preprocessing/`，覆盖数据清洗的四个环节：

| 文件 | 功能 |
| --- | --- |
| data_clean.py | 缺失值报告、数值列中位数填充、去重 |
| outlier_detection.py | IQR / 3σ 异常值检测与盖帽处理 |
| encode_categorical.py | one-hot 哑变量编码、标签编码 |

**推荐的处理顺序**：去重 → 缺失报告（决定填充/删列）→ 中位数填充 → 异常值检测与处理 → 分类变量编码 → 建模（标准化在各模型的 Pipeline 内完成，避免数据泄漏）。

## 缺失值与去重（data_clean.py）

### missing_report / drop_duplicates 原理

`missing_report` 统计每列缺失数量与比例并按严重程度降序——先用数据决定策略（缺失 50% 的列通常应剔除而非填充），而不是盲目填充。`drop_duplicates` 删除完全重复的行并保留首次出现；重复行会让同一样本在训练集与测试集中各出现一次，造成评估虚高，应在划分数据**之前**去除。

### handle_missing 原理

对每个**数值列**，用该列的中位数填充缺失值（`NaN`）。中位数是列数据排序后位于中间位置的值：

$$\tilde{x}_j = \begin{cases} x_{j,(n/2)} & n \text{ 为偶数（取中间两值的均值）} \\ x_{j,((n+1)/2)} & n \text{ 为奇数} \end{cases}$$

与均值填充相比，中位数只依赖数据的**顺序**而非数值大小：极端值再大也只占据排序两端的一个位置，不会拉动中位数。因此当列中存在异常值或分布偏斜（如收入、面积类指标）时，中位数填充更稳健。

### 用法

```python
import pandas as pd
from py.preprocessing.data_clean import (
    drop_duplicates, handle_missing, missing_report,
)

df = pd.read_csv("附件.csv")

print(missing_report(df))          # 各列缺失数量与比例（只列有缺失的列）
df = drop_duplicates(df)           # 整行去重；subset=["id"] 可按关键列去重
df = handle_missing(df)            # 数值列缺失用中位数填充
```

- 三个函数都不修改原始 DataFrame；`handle_missing` 只处理数值列，文本/分类/日期列原样保留。
- 直接运行模板（自包含合成数据示例）：`.venv/bin/python py/preprocessing/data_clean.py`

### 注意

- 中位数填充隐含「缺失是随机的」这一假设；若缺失本身有规律（如仪器超量程才缺失），应在论文中说明并考虑其他处理方式。
- 分类列的缺失不能用中位数填充，应在编码前按题意单独处理（填充众数、单独"未知"类别或删除）。

## 异常值检测（outlier_detection.py）

### 原理

两种划界方法，输出的都是**越界掩码和边界**，不直接删改数据：

- **IQR 法（箱线图法则，推荐默认）**：以四分位数划界，$[\,Q_1 - 1.5 \cdot IQR,\ \ Q_3 + 1.5 \cdot IQR\,]$，其中 $IQR = Q_3 - Q_1$。四分位数只依赖数据的**排序位置**，极端值再大也不会拉动边界，因此对偏态分布稳健。
- **3σ 法**：把每个值标准化为 $z = \dfrac{x - \bar{x}}{s}$，$|z| > 3$ 判为异常（正态假设下概率约 0.27%）。它假设数据近似正态；偏态数据中极端值会**抬高均值和标准差**，反而掩盖自身（掩蔽效应），漏检明显。

检测出异常后如何处理由题目背景决定，常用三种：

1. **盖帽（winsorize）**：`cap_outliers` 把越界值收缩到边界，保留样本量且不改变正常值，适合「极端但真实」的指标；
2. **置 NaN 后填充**：`data.mask(detection["mask"])` 再交给 `handle_missing` 中位数填充，等效于用稳健统计量替换；
3. **删行**：`data[~detection["mask"].any(axis=1)]`，异常是录入错误且数量很少时适用。

### 用法

```python
from py.preprocessing.outlier_detection import (
    cap_outliers, detect_outliers_iqr, detect_outliers_zscore,
)

iqr = detect_outliers_iqr(df)                  # 默认检测全部数值列
print(iqr["counts"])                           # 每列异常值个数
print(iqr["lower"], iqr["upper"])              # 每列边界

capped = cap_outliers(df, iqr)                 # 盖帽到边界
masked = df.mask(iqr["mask"])                  # 或置 NaN 后交给 handle_missing

zscore = detect_outliers_zscore(df, threshold=3.0)   # 近似正态的列可用
```

- 返回值 `mask` 是与原数据对齐的布尔表，可按行、按列统计。
- 直接运行模板：`.venv/bin/python py/preprocessing/outlier_detection.py`（demo 中右偏列 IQR 检出 9 个异常、3σ 仅 3 个，正是掩蔽效应的展示）

### 注意

- 「异常」不等于「错误」：高价值订单、极端天气是真实数据，删掉会扭曲结论——论文中应说明处理依据。
- 检测和处理的统计量（分位数、均值）应只在**训练集**上计算，再应用到测试集，避免数据泄漏。

## 分类变量编码（encode_categorical.py）

### 原理

多数模型只接受数值输入，分类列必须先编码，两种方式对应不同假设：

- **one-hot 编码**：每个类别展开成一列 0/1 哑变量（`color_red`、`color_blue`、…）。类别之间没有大小与顺序关系（颜色、城市），哑变量如实表达「属于/不属于」，不会把虚假的数值距离引入模型。列数随类别数增长，类别过多（如邮政编码）会导致维度爆炸。
- **标签编码**：类别按排序映射为整数（低/中/高 → 0/1/2）。编码暗示了顺序和距离，只适合**有序类别**，或对数值大小不敏感的树模型（XGBoost、随机森林按阈值切分，天然免疫虚假距离）。

线性回归中使用 one-hot 时应设 `drop_first=True`：每组哑变量之和恒为 1，与截距列完全共线性（多重共线性，见[回归诊断](regression.md)的 VIF）。

### 用法

```python
from py.preprocessing.encode_categorical import label_encode, one_hot_encode

# 默认自动选择全部分类列（非数值、非日期）
encoded, info = one_hot_encode(df, drop_first=True)   # 线性模型用 drop_first
print(info["dummy_columns"])                          # 新增的哑变量列名

labeled, mappings = label_encode(df)
print(mappings)        # {列名: {原类别: 编码}}，论文中报告此对照表
```

- `columns` 参数可指定只编码部分列。
- 直接运行模板：`.venv/bin/python py/preprocessing/encode_categorical.py`

### 注意

- one-hot 中缺失值表现为该组哑变量全 0；标签编码中缺失保留为 NaN——两者都建议先处理缺失再编码。
- 编码后的新列名（`列名_类别`）应保持稳定，预测新数据时必须用**同样的映射**，不要对训练集和预测数据分别独立编码。
