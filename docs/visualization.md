# 可视化（visualization）

为论文产出统一风格的图。对应 `py/visualization/`，基于 matplotlib（热力图借助 seaborn），提供样式基础设施和四类绘图模板。统一约定：

- 画图前先 `set_paper_style()`——风格在创建画布时生效，顺序不能反；
- 每个函数返回 `(fig, ax)`，传 `ax` 可把多张图拼进同一 Figure；
- 用 `save_figure(fig, name)` 导出（PNG 300 dpi，可选 PDF 矢量图）；
- 标签默认英文（美赛论文），国赛中文标签传 `set_paper_style(chinese=True)`。

## 样式与导出（plot_style.py）

### 原理

matplotlib 的样式由全局 `rcParams` 字典控制。模板把论文需要的项（字号、线宽、去掉上方和右侧边框、浅色虚线网格、300 dpi 导出）集中成一份配置，所有图共享同一观感。中文字体按**回退列表**处理：把 macOS / Windows 常见中文字体按优先级排列，函数只保留系统实际安装的项（逐个探测，避免缺字体警告），英文论文则完全绕开中文字体问题。

### 优缺点与使用场景

- **优点**：一份配置统一全文观感，省去逐图调样式；中文字体回退机制跨平台免告警；PNG + PDF 双导出兼顾排版与放大。
- **缺点**：`rcParams` 是全局状态，设置后影响之后的所有图；样式面向学术论文，交互式探索不一定合适。
- **使用场景**：论文出图前的统一初始化——任何要导出进论文的图，都应先 `set_paper_style()` 再画。

### 用法

```python
from py.visualization.plot_style import new_axes, save_figure, set_paper_style

set_paper_style()                 # 英文论文；国赛传 chinese=True
fig, ax = new_axes()              # 或 plt.subplots() 后自己画
ax.plot(x, y)
path = save_figure(fig, "my_plot", pdf=True)   # 300 dpi PNG + 矢量 PDF
```

### 注意

- `axes.unicode_minus = False` 已处理部分字体不含 Unicode 负号的问题。
- 论文插图建议同时导出 PDF（放大不失真），排版用 `\includegraphics` 直接引用。

## 热力图（heatmap_plots.py）

### 原理

相关系数矩阵用**以 0 为中心的发散色**（`RdBu_r`，红正蓝负、颜色深浅对应强度），并遮住上三角避免重复信息；通用矩阵（双因素灵敏度、产量表、计数矩阵）用顺序色 `viridis`。数值直接标注在格子里，评审不需要对照色条读数。计算相关系数的部分独立成 `correlation_matrix` 函数，Pearson（线性）与 Spearman（秩，对异常值稳健）可选。

### 优缺点与使用场景

- **优点**：一张图呈现全部两两关系，格内标数值免对照色条；发散色让正负相关一眼可分。
- **缺点**：变量多于 15 个左右时格子过小难读；只表达两两关系，看不出高阶结构。
- **使用场景**：建模前的指标相关结构检查（顺带发现共线性）、双因素灵敏度矩阵、产量 / 计数表。

### 用法

```python
import pandas as pd
from py.visualization.heatmap_plots import (
    correlation_matrix, plot_correlation_heatmap, plot_matrix_heatmap,
)
from py.visualization.plot_style import set_paper_style

set_paper_style()
df = pd.read_csv("附件.csv")

fig, ax = plot_correlation_heatmap(df, method="pearson")  # 指标相关结构
fig, ax = plot_matrix_heatmap(outputs, xlabels=..., ylabels=...)  # 通用矩阵
corr = correlation_matrix(df)   # 只算矩阵不画图，用于报告数值
```

注意：常数列的相关系数无定义，模板会直接报错，应先剔除。

## 模型诊断图（model_plots.py）

每个函数对接现有建模模板的输出字段，出图不需要重新组织数据：

| 图 | 函数 | 数据来源 |
| --- | --- | --- |
| 真实 vs 预测（y=x 参考线，标 R²/RMSE） | `plot_true_vs_pred` | 回归 `result["y_test"]` / `["y_pred"]` |
| 残差-拟合值散点 | `plot_residuals` | 拟合值与残差 |
| 残差直方图 + 正态参考线 | `plot_residual_histogram` | `resid` |
| 混淆矩阵热力图 | `plot_confusion_matrix` | 分类 `result["confusion_matrix"]` |
| ROC 曲线 + AUC | `plot_roc_curve` | 二分类 `result["y_proba"]` 正类列 |
| 聚类散点（噪声点 -1 灰色 x） | `plot_cluster_scatter` | `result["labels"]` + 两维坐标 |
| PCA 双标图（得分 + 载荷箭头） | `plot_pca_biplot` | `pca_analysis` 的 `scores` / `loadings` |

### 优缺点与使用场景

- **优点**：直接消费各建模模板的 `result` 字段，出图零数据整理；每张图都是评审习惯看的「标准配图」（y=x 参考线、ROC 对角基线、噪声点灰色标记等）。
- **缺点**：字段名与既有模板耦合，自定义流程需自己拼数据；每张图的定制空间有限。
- **使用场景**：论文「模型评估」小节——回归配真实 vs 预测 + 残差图，分类配混淆矩阵 + ROC，聚类 / 降维配散点与 biplot。

### 用法

```python
from py.visualization.model_plots import plot_true_vs_pred, plot_pca_biplot
from py.visualization.plot_style import set_paper_style

set_paper_style()
fig, ax = plot_true_vs_pred(result["y_test"], result["y_pred"])
fig, ax = plot_pca_biplot(pca_result["scores"], pca_result["loadings"],
                          feature_names=["a", "b", "c", "d"])
```

残差图看「漏斗形」（异方差）与「弯曲」（非线性结构）；biplot 箭头越长、指向越一致的指标，对前两个主成分影响越大。

## 分析图（analysis_plots.py）

对接灵敏度分析和预测模板的输出：

- `plot_one_way`：一因素扫描折线（`one_way_sensitivity` 的 `values`/`outputs`）；
- `plot_two_way_heatmap`：双因素矩阵热力图（`two_way_sensitivity` 的输出）；
- `plot_monte_carlo`：输出分布直方图 + 分位数竖线（`monte_carlo_sensitivity` 的 `outputs`）；
- `plot_forecast`：历史（实线）+ 拟合（虚线）+ 未来预测（方点虚线）+ 置信带（`fill_between`），竖直虚线标记历史与预测分界；`lower`/`upper` 与 `forecast` 等长时自动绘制置信区间。

```python
from py.visualization.analysis_plots import plot_forecast
from py.visualization.plot_style import set_paper_style

set_paper_style()
fig, ax = plot_forecast(y, fitted=holt["fitted"], forecast=holt["forecast"])
```

灵敏度三张图（折线 + 热力图 + 分布图）是论文「稳健性检验」小节的标准配图。

### 优缺点与使用场景

- **优点**：与灵敏度 / 预测模板的输出结构一一对应，置信带、历史-预测分界线等细节已处理好；图型覆盖稳健性检验与预测展示的全部标准配图。
- **缺点**：`plot_forecast` 的置信带要求 `lower` / `upper` 与 `forecast` 等长，模型不输出区间时画不了。
- **使用场景**：灵敏度分析小节的三张标准图；时间序列预测结果的历史-预测对照展示。

## 基础图（basic_plots.py）

不带建模语义的通用图形，做数据探索时使用：

| 函数 | 用途 |
| --- | --- |
| `plot_line` | 多序列折线，可选误差棒 |
| `plot_scatter` | 散点，可选第三变量着色 + 颜色条 |
| `plot_bar` | 柱状对比，可选误差棒 |
| `plot_box` | 多组分布与离群点对比 |
| `plot_histogram` | 单变量分布形状 |

### 优缺点与使用场景

- **`plot_line`**——优点：趋势与拐点最清晰；缺点：序列多于 5~6 条时拥挤。
- **`plot_scatter`**——优点：直观展示关系形态与聚集；缺点：点数大时相互重叠，需透明度或抽样。
- **`plot_bar`**——优点：类别间对比直观；缺点：只显示汇总值（均值），掩盖分布形状。
- **`plot_box`**——优点：同时给出分位数与离群点；缺点：看不出双峰等多峰结构（此时用直方图）。
- **`plot_histogram`**——优点：分布形状一目了然；缺点：对分箱宽度敏感，多组对比不如箱线图。
- **使用场景**：数据探索（EDA）阶段的快速作图；确认数据形态后，再交给各专用模板出论文级图。

## 图选型指南

| 想表达的内容 | 推荐图 |
| --- | --- |
| 随时间 / 参数变化的趋势 | 折线 `plot_line` / `plot_one_way` |
| 两变量关系（含第三个变量） | 散点 `plot_scatter` |
| 指标间相关结构 | 相关热力图 `plot_correlation_heatmap` |
| 多组数值对比 | 柱状 `plot_bar`（均值）/ 箱线 `plot_box`（分布） |
| 回归质量 | `plot_true_vs_pred` + `plot_residuals` |
| 分类质量 | `plot_confusion_matrix` + `plot_roc_curve` |
| 聚类 / 降维结果 | `plot_cluster_scatter` / `plot_pca_biplot` |
| 参数组合的影响 | `plot_two_way_heatmap` |
| 结果稳健性 | `plot_monte_carlo` + 灵敏度图 |
| 预测结果 | `plot_forecast` |

## 注意

- 所有 demo 运行后会把示例图保存为当前目录的 `*_out.png`（模板验证用，可删除）；
- 模板函数不弹窗显示（`plt.show()` 未调用），批量出图更稳；交互查看时自己加 `plt.show()`；
- 多图导出后记得 `plt.close(fig)` 释放内存；seaborn 只用了 `heatmap`，没有调用 `set_theme`，不会覆盖论文风格。
