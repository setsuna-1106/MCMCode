# 统计假设检验（statistics）

用样本数据判断「组间差异」或「变量关联」是否显著，是论文中支撑结论的常用证据。本模块对应 `py/statistics/sm_tests.py`，基于 statsmodels 提供三个检验模板。

三个检验的原假设 $H_0$ 都是「没有差异 / 没有关联」：p 值小于显著性水平（通常 0.05）时拒绝 $H_0$，认为存在显著差异或关联。

| 检验 | 适用问题 | 原假设 $H_0$ |
| --- | --- | --- |
| `welch_ttest` | 两组独立样本的均值比较 | 两组均值相等 |
| `chi_square` | 两个分类变量是否相关 | 两变量相互独立 |
| `one_way_anova` | 三组及以上的均值比较 | 各组均值全相等 |

## welch_ttest

### 原理

Welch t 检验比较**两个独立组**的均值，不要求两组方差相等（比经典 Student t 检验更稳妥，推荐默认使用）：

$$t = \frac{\bar{x}_a - \bar{x}_b}{\sqrt{\dfrac{s_a^2}{n_a} + \dfrac{s_b^2}{n_b}}}, \qquad df \approx \frac{\left(\dfrac{s_a^2}{n_a} + \dfrac{s_b^2}{n_b}\right)^2}{\dfrac{(s_a^2/n_a)^2}{n_a-1} + \dfrac{(s_b^2/n_b)^2}{n_b-1}}$$

其中 $\bar{x}$ 为样本均值，$s^2$ 为样本方差，$n$ 为样本量，自由度采用 Welch–Satterthwaite 近似。$|t|$ 越大，均值差异相对抽样波动越显著。

### 用法

```python
from py.statistics.sm_tests import welch_ttest

result = welch_ttest(group_a, group_b)
print(result["statistic"], result["pvalue"], result["df"])
```

- 入参 `group_a, group_b`：两组一维数值观测（列表或数组均可）。
- 返回：包含 `statistic`（t 统计量）、`pvalue`、`df`（自由度）的字典。

### 注意

- 只适用于**独立**样本；同一对象前后两次测量应使用配对检验。
- 两组需近似正态；样本量小且严重偏态时可考虑非参数的 Mann–Whitney U 检验。

## chi_square

### 原理

卡方独立性检验判断列联表中两个**分类变量**是否关联。用行列边际估计每个格子的期望频数 $E_{ij} = \dfrac{n_{i\cdot} \, n_{\cdot j}}{n}$，再比较观测频数与期望的偏离：

$$\chi^2 = \sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}, \qquad df = (r-1)(c-1)$$

观测与期望偏离越大，$\chi^2$ 越大，两变量越可能不独立。

### 用法

```python
from py.statistics.sm_tests import chi_square

result = chi_square([[180, 20], [150, 50]])
print(result["statistic"], result["pvalue"], result["df"])
```

- 入参 `table`：非负频数列联表，行为一种分类、列为另一种分类。
- 返回：包含 `statistic`（卡方统计量）、`pvalue`、`df` 的字典。

### 注意

- 输入是**频数**（计数），不是比例。
- 期望频数过小（经验上超过 20% 的格子 $E_{ij}<5$）时结果不可靠，应合并类别或改用 Fisher 精确检验。

## one_way_anova

### 原理

单因素方差分析比较**三组及以上**的均值是否全相等。把总变异分解为组间与组内两部分：

$$F = \frac{\text{组间均方 } MSB}{\text{组内均方 } MSW} = \frac{SSB/(k-1)}{SSW/(n-k)}$$

$SSB$ 度量各组均值对总均值的偏离，$SSW$ 度量组内随机波动。若 $H_0$ 成立，$F$ 应接近 1；$F$ 显著偏大说明至少有一组均值不同。实现上等价于对 `目标 ~ C(分组)` 做一次 OLS 回归再做 F 检验，模板返回的 `table` 即标准 ANOVA 表。

### 用法

```python
from py.statistics.sm_tests import one_way_anova

result = one_way_anova(df, target="yield", group="plan")
print(result["statistic"], result["pvalue"])
print(result["table"])   # 完整 ANOVA 表
```

- 入参 `data` 为 DataFrame，`target` 是连续型目标列名，`group` 是分类分组列名（列名可含空格）。
- 返回：包含 `model`（OLS 模型）、`table`、`statistic`（F 值）、`pvalue` 的字典。

### 注意

- ANOVA 显著只说明「至少两组不同」，具体哪两组不同需做事后检验（如 Tukey HSD）。
- 依赖各组近似正态、方差齐性；方差不齐时可改用 Welch ANOVA 或 Kruskal–Wallis 检验。
