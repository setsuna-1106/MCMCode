# 数据预处理（preprocessing）

建模的第一步：把带缺失值的原始数据表处理成可建模的干净输入。本模块对应 `py/preprocessing/`，当前提供一个中位数填充模板。

## handle_missing

### 原理

对每个**数值列**，用该列的中位数填充缺失值（`NaN`）。中位数是列数据排序后位于中间位置的值：

$$\tilde{x}_j = \begin{cases} x_{j,(n/2)} & n \text{ 为偶数（取中间两值的均值）} \\ x_{j,((n+1)/2)} & n \text{ 为奇数} \end{cases}$$

与均值填充相比，中位数只依赖数据的**顺序**而非数值大小：极端值再大也只占据排序两端的一个位置，不会拉动中位数。因此当列中存在异常值或分布偏斜（如收入、面积类指标）时，中位数填充更稳健。

### 用法

```python
import pandas as pd
from py.preprocessing.data_clean import handle_missing

df = pd.read_csv("附件.csv")
df = handle_missing(df)
```

- 入参 `df`：待处理的 `pandas.DataFrame`，函数内部会复制一份，**不修改原始数据**。
- 返回：数值列缺失值已填充的新数据表；文本、分类、日期列原样保留。
- 直接运行模板（读取项目根目录的 `附件.csv`）：

```bash
.venv/bin/python py/preprocessing/data_clean.py
```

### 注意

- 只自动处理数值列；非数值列需要按题目含义单独决定填充或删除策略。
- 中位数填充隐含「缺失是随机的」这一假设；若缺失本身有规律（如仪器超量程才缺失），应在论文中说明并考虑其他处理方式。
- 比赛常见流程：先 `df.info()` 查看各列缺失比例，再决定填充、插值（见[插值模块](interpolation.md)）还是剔除。
