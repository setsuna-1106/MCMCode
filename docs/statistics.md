# 统计假设检验（statistics）

用样本数据判断「组间差异」「变量关联」或「分布形态」是否显著，是论文中支撑结论的常用证据。本模块对应 `py/statistics/`，覆盖经典参数检验、正态性前置检查、非参数替代、相关分析和 Bootstrap 区间。

除 Bootstrap 外，各检验的原假设 $H_0$ 都是「没有差异 / 没有关联 / 符合假设」：p 值小于显著性水平（通常 0.05）时拒绝 $H_0$。

| 检验 | 适用问题 | 原假设 $H_0$ | 所在文件 |
| --- | --- | --- | --- |
| `welch_ttest` | 两组独立样本的均值比较 | 两组均值相等 | sm_tests |
| `paired_ttest` | 同一对象前后两次测量 | 差值均值为 0 | sm_tests |
| `levene_test` | 多组方差齐性前置检查 | 各组方差相等 | sm_tests |
| `chi_square` | 两个分类变量是否相关 | 两变量相互独立 | sm_tests |
| `fisher_exact_test` | 2x2 表、期望频数过小时 | 两变量相互独立 | sm_tests |
| `one_way_anova` | 三组及以上的均值比较 | 各组均值全相等 | sm_tests |
| `tukey_hsd` | ANOVA 显著后的两两比较 | （逐对）均值相等 | sm_tests |
| 正态性三检验 | 数据是否近似正态 | 来自正态分布 | normality_tests |
| `mann_whitney_u` 等 | 正态不成立时的替代 | 分布位置相同 | nonparametric_tests |
| `pearson/spearman` | 两变量相关性 | 相关系数为 0 | correlation_tests |
| Bootstrap | 任意统计量的置信区间 | —（估计而非检验） | bootstrap_interval |

典型的检验流程：先 `normality_tests` 查正态 → 正态用参数检验（t / ANOVA，ANOVA 前配 `levene_test` 查方差齐性）→ 不正态换非参数版本 → ANOVA 显著后用 `tukey_hsd` 找出差异组对。

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

### 注意

- 只适用于**独立**样本；同一对象前后测量用 `paired_ttest`。
- 两组需近似正态——先做[正态性检验](#正态性检验normality_testspy)；严重偏态时改用 `mann_whitney_u`。

## paired_ttest

### 原理

配对 t 检验处理**同一对象前后两次测量**（如处理前/后指标）。先逐对求差 $d_i = \text{after}_i - \text{before}_i$，再对差值做单样本 t 检验：

$$t = \frac{\bar{d}}{s_d / \sqrt{n}}, \qquad df = n - 1$$

配对消除了对象间固有差异，只保留处理效应，通常比独立样本 t 检验功效更高。

### 用法

```python
from py.statistics.sm_tests import paired_ttest

result = paired_ttest(before, after)   # 等长的两组配对观测
print(result["statistic"], result["pvalue"], result["df"])
```

### 注意

- 差值需近似正态；不满足时用 `wilcoxon_signed_rank`。

## levene_test

### 原理

检验多组数据的**方差是否相等**（方差齐性），是经典 ANOVA 和 pooled t 检验的前提。把每个观测对本组中心（均值或中位数）的偏离绝对值做 ANOVA——偏离程度一致即方差齐。默认 `center="median"`（Brown–Forsythe 版本），对非正态数据更稳健。

### 用法

```python
from py.statistics.sm_tests import levene_test

result = levene_test([group_a, group_b, group_c])   # 至少两组
print(result["pvalue"])   # p < 0.05 说明方差不齐
```

### 注意

- Welch t 检验**不需要**方差齐性；经典 ANOVA 方差不齐时改用 Welch ANOVA 或 `kruskal_wallis`。

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
- 期望频数过小（经验上超过 20% 的格子 $E_{ij}<5$）时结果不可靠，2x2 表改用 `fisher_exact_test`，更大的表可合并类别。

## fisher_exact_test

### 原理

2x2 列联表的**精确检验**：在行列边际固定的条件下，直接用超几何分布计算「当前或更极端」表型的概率，不依赖大样本渐近——小样本或期望频数过小时比卡方检验可靠。同时输出优势比（odds ratio）。

### 用法

```python
from py.statistics.sm_tests import fisher_exact_test

result = fisher_exact_test([[8, 2], [1, 9]])
print(result["odds_ratio"], result["pvalue"])
```

只支持 2x2 表；更大的表应合并类别后用卡方检验。

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
```

### 注意

- 前提是各组近似正态、方差齐性——先做 `levene_test`。
- ANOVA 显著只说明「至少两组不同」，具体哪两组用 `tukey_hsd`；正态或齐性不满足时改用 `kruskal_wallis`。

## tukey_hsd

### 原理

ANOVA 显著后的**事后多重比较**：对全部组对做两两均值比较，同时把族错误率（FWER）控制在 `alpha`——若逐对直接做 t 检验，比较次数一多假阳性会失控。Tukey HSD 基于 studentized range 分布给出每对的均值差、置信区间和调整后 p 值。

### 用法

```python
from py.statistics.sm_tests import tukey_hsd

result = tukey_hsd(df, target="yield", group="plan")
print(result["table"])       # 逐对比较表（差值、p 值、置信区间、是否显著）
print(result["reject"])      # 布尔数组：每对是否显著
```

与 `one_way_anova` 使用相同的 `data / target / group` 接口。

## 正态性检验（normality_tests.py）

### 原理

三个检验的原假设都是「数据来自正态分布」，适用场景不同：

- `shapiro_wilk`：小样本（$3 \le n \le 5000$）首选，对偏离正态最灵敏，检验 W 统计量（观测值与正态期望值的相关性）；
- `jarque_bera_test`：基于**偏度**（对称性）与**峰度**（尾部厚度）的联合检验，正态分布下两者分别为 0 和 3，大样本下渐近有效，还返回这两个描述统计量；
- `lilliefors_test`：K–S 检验的校正版——普通 K–S 要求分布参数已知，参数由样本估计时它给出正确的 p 值。

### 用法

```python
from py.statistics.normality_tests import (
    jarque_bera_test, lilliefors_test, shapiro_wilk,
)

print(shapiro_wilk(data)["pvalue"])        # n <= 5000 首选
print(jarque_bera_test(data)["pvalue"])    # 大样本
print(lilliefors_test(data)["pvalue"])     # K-S 校正版
```

### 注意

- p > 0.05 只是「没有足够证据拒绝正态」，不等于「一定正态」。
- 大样本下检验极灵敏：轻微偏态也会被拒绝，应结合直方图 / QQ 图（可视化模块）判断是否「实用地接近正态」。

## 非参数检验（nonparametric_tests.py）

### 原理

基于**秩**（排序位置）而非原始数值，不要正态假设，对异常值稳健。与参数检验一一对应：

| 非参数检验 | 替代的参数检验 | 检验什么 |
| --- | --- | --- |
| `mann_whitney_u` | `welch_ttest` | 两组分布位置（中位数）是否不同 |
| `kruskal_wallis` | `one_way_anova` | 多组分布位置是否不同 |
| `wilcoxon_signed_rank` | `paired_ttest` | 配对差值是否系统性偏离 0 |

秩检验把数值映射为名次，损失了距离信息，正态假设成立时功效略低于参数检验；作为代价，偏态、离群值都影响不了名次。

### 用法

```python
from py.statistics.nonparametric_tests import (
    kruskal_wallis, mann_whitney_u, wilcoxon_signed_rank,
)

print(mann_whitney_u(group_a, group_b)["pvalue"])
print(kruskal_wallis([group_a, group_b, group_c])["pvalue"])
print(wilcoxon_signed_rank(before, after)["pvalue"])
```

### 注意

- 结论措辞是「分布位置（中位数）不同」，不是「均值不同」。

## 相关分析（correlation_tests.py）

### 原理

`pearson_correlation` 度量**线性**相关：

$$r = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_i (x_i - \bar{x})^2} \sqrt{\sum_i (y_i - \bar{y})^2}} \in [-1, 1]$$

`spearman_correlation` 先把数值转成秩再算同一公式，度量**单调**相关：对指数增长这类「单调但非线性」的关系仍给出接近 ±1 的值，且不受个别异常值拉动。

**相关强度与显著性是两回事**：$r$ 的大小衡量强度，p 值只回答「是否显著异于 0」。大样本下 $r=0.1$ 也可以极显著，论文中必须同时报告两者。

### 用法

```python
from py.statistics.correlation_tests import pearson_correlation, spearman_correlation

print(pearson_correlation(x, y))    # {"statistic": r, "pvalue": ..., "n": ...}
print(spearman_correlation(x, y))   # {"statistic": rho, "pvalue": ..., "n": ...}
```

### 注意

- 相关不等于因果；关系弯曲时 Pearson 会低估关联，先画散点图（可视化模块 `plot_scatter`）确认形态。
- 多个指标一起看相关结构时用可视化模块的相关热力图（`plot_correlation_heatmap`）。

## Bootstrap 置信区间（bootstrap_interval.py）

### 原理

不假设任何分布形状，直接用**经验分布**估计统计量的不确定性：有放回地重采样原始数据 $B$ 次（每次样本量与原数据相同），得到 $B$ 个统计量取值，取 $[\alpha/2,\ 1-\alpha/2]$ 分位数作为置信区间。原理是「样本的经验分布是总体分布的合理近似」，样本量越大该近似越好。

适用于中位数、分位数、自定综合指标等难以解析求标准误的统计量。

### 用法

```python
import numpy as np
from py.statistics.bootstrap_interval import bootstrap_confidence_interval

ci = bootstrap_confidence_interval(data, statistic=np.median,
                                   n_resamples=5000, confidence=0.95)
print(ci["estimate"], (ci["low"], ci["high"]))
```

`statistic` 可以是任意 `f(sample) -> 标量` 函数。返回 `estimate`（原始样本上的统计量）与区间端点 `low` / `high`。

### 注意

- 结果随种子波动，固定 `random_state` 保证复现；`n_resamples` 越大区间端点越稳定（默认 5000 足够）。
- 极小样本（n < 20）或统计量非常极端时区间覆盖率可能不足，应结合题目背景说明。
