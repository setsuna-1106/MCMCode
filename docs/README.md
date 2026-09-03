# 模块文档索引

每个模块一份文档：方法原理、接口用法、选型对比和使用注意。模板源码在 [`py/`](../py/) 对应目录，仓库总览与快速开始见 [README](../README.md)。

## 按建模需求查找

| 需求 | 模块 | 文档 |
| --- | --- | --- |
| 缺失值、重复行、异常值、分类编码 | preprocessing | [preprocessing.md](preprocessing.md) |
| 显著性检验、正态性、非参数检验、相关分析、Bootstrap | statistics | [statistics.md](statistics.md) |
| 二分类 / 多分类 | classification | [classification.md](classification.md) |
| 无标签分簇 | clustering | [clustering.md](clustering.md) |
| 连续目标预测（含统计推断与回归诊断） | regression | [regression.md](regression.md) |
| 指标压缩、综合得分 | dimensionality_reduction | [dimensionality_reduction.md](dimensionality_reduction.md) |
| 时间序列外推 | forecasting | [forecasting.md](forecasting.md) |
| 线性/整数/黑箱优化、排程、指派、运输与路径 | optimization | [optimization.md](optimization.md) |
| 网络结构：最短路、流、匹配、关键路径、中心性 | graph | [graph.md](graph.md) |
| 常微分方程 | ode | [ode.md](ode.md) |
| 离散点之间的取值 | interpolation | [interpolation.md](interpolation.md) |
| 无法解析求积的定积分 | integration | [integration.md](integration.md) |
| 权重、排序、灵敏度分析、模型评估流程 | evaluation | [evaluation.md](evaluation.md) |
| 论文出图 | visualization | [visualization.md](visualization.md) |

## 推荐阅读顺序

按建模流程阅读：

1. [preprocessing](preprocessing.md) —— 数据先干净（去重、缺失、异常值、编码）；
2. [statistics](statistics.md) —— 看清数据再建模（分布、差异、相关性）；
3. 按题意选建模模块：classification / clustering / regression / dimensionality_reduction / forecasting / optimization / graph / ode / interpolation / integration；
4. [evaluation](evaluation.md) —— 赋权排序、灵敏度检验结论的稳健性；
5. [visualization](visualization.md) —— 统一风格的论文配图。

配合根 [README](../README.md) 的「推荐的建模流程」一节使用。
