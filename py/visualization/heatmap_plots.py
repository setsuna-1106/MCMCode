"""矩阵热力图模板：相关系数矩阵和通用二维矩阵。

基于 ``seaborn.heatmap`` 绘制带数值注释的热力图；只使用热力图功能，
不调用 ``sns.set_theme``，避免覆盖 ``plot_style`` 设置的论文风格。

相关系数使用以 0 为中心的发散色（RdBu_r）：正相关偏红、负相关偏蓝，
与通用矩阵的顺序色（viridis）区分开。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from py.visualization.plot_style import new_axes
except ModuleNotFoundError:
    from plot_style import new_axes


def _as_numeric_frame(data, minimum_columns: int, name: str) -> pd.DataFrame:
    """把数组或 DataFrame 统一为纯数值 DataFrame，列名自动生成。"""
    if isinstance(data, pd.DataFrame):
        frame = data.select_dtypes(include="number")
        columns = list(frame.columns)
    else:
        matrix = np.asarray(data, dtype=float)
        if matrix.ndim != 2 or not np.isfinite(matrix).all():
            raise ValueError(f"{name} 必须是只含有限数值的二维数组")
        frame = pd.DataFrame(matrix)
        columns = [f"x{j + 1}" for j in range(matrix.shape[1])]
    if frame.shape[1] < minimum_columns:
        raise ValueError(f"{name} 至少需要 {minimum_columns} 个数值列")
    frame.columns = columns
    return frame


def correlation_matrix(data, method: str = "pearson") -> pd.DataFrame:
    """计算相关系数矩阵（绘图与验收共用的入口）。

    Args:
        data: DataFrame 或 ``(样本数, 指标数)`` 数值矩阵。
        method: ``pearson``（线性相关）或 ``spearman``（秩相关）。

    Returns:
        对称的相关系数矩阵 DataFrame。

    Raises:
        ValueError: 方法名不合法或数值列不足时抛出。
    """
    if method not in ("pearson", "spearman"):
        raise ValueError("method 必须是 'pearson' 或 'spearman'")
    frame = _as_numeric_frame(data, minimum_columns=2, name="data")
    corr = frame.corr(method=method)
    if corr.isna().any().any():
        # 常数列与任何变量的相关系数都没有定义，绘图前应剔除。
        raise ValueError("存在无法计算相关系数的列（如常数列），请先剔除")
    return corr


def plot_correlation_heatmap(data, *, method: str = "pearson",
                             annot: bool = True, ax=None):
    """绘制相关系数下三角热力图。

    Args:
        data: DataFrame 或 ``(样本数, 指标数)`` 数值矩阵。
        method: ``pearson`` 或 ``spearman``。
        annot: 是否在格子中标注数值。
        ax: 可选的已有坐标系，用于拼多子图。

    Returns:
        ``(fig, ax)``。
    """
    corr = correlation_matrix(data, method=method)
    if ax is None:
        fig, ax = new_axes()
    else:
        fig = ax.figure
    # 遮住上三角避免重复信息；对角线恒为 1，保留作为行列标签的锚点。
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        annot=annot,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        square=True,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title(f"{method.capitalize()} correlation")
    return fig, ax


def plot_matrix_heatmap(matrix, *, xlabels=None, ylabels=None,
                        cmap: str = "viridis", fmt: str = ".3g",
                        title: str = "", ax=None):
    """绘制通用二维矩阵热力图。

    适合双因素灵敏度输出、产量/成本表、混淆计数矩阵等已经算好的数据。

    Args:
        matrix: 二维数值矩阵，行对应 ``ylabels``、列对应 ``xlabels``。
        xlabels: 列标签；为空时自动编号。
        ylabels: 行标签；为空时自动编号。
        cmap: 顺序型色图名，数值大小按颜色深浅映射。
        fmt: 注释数字格式（如 ``".2f"``、``".3g"``）。
        title: 图标题。
        ax: 可选的已有坐标系。

    Returns:
        ``(fig, ax)``。

    Raises:
        ValueError: 矩阵不是二维有限数值或标签数量不匹配时抛出。
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.size == 0 or not np.isfinite(matrix).all():
        raise ValueError("matrix 必须是非空、只含有限数值的二维数组")

    if xlabels is None:
        xlabels = [f"c{j + 1}" for j in range(matrix.shape[1])]
    if ylabels is None:
        ylabels = [f"r{i + 1}" for i in range(matrix.shape[0])]
    if len(xlabels) != matrix.shape[1] or len(ylabels) != matrix.shape[0]:
        raise ValueError("xlabels 和 ylabels 的数量必须与矩阵形状一致")

    if ax is None:
        fig, ax = new_axes()
    else:
        fig = ax.figure
    frame = pd.DataFrame(matrix, index=list(ylabels), columns=list(xlabels))
    sns.heatmap(
        frame,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title(title)
    return fig, ax


def main() -> None:
    # ====== 比赛时主要替换下面这部分 ======
    try:
        from py.visualization.plot_style import save_figure, set_paper_style
    except ModuleNotFoundError:
        from plot_style import save_figure, set_paper_style

    # 先定样式再建画布：spines、网格等风格在创建 Axes 时生效。
    set_paper_style()

    # 构造已知相关性的数据：x2 与 x1 完全正相关，x3 与 x1 完全负相关。
    rng = np.random.default_rng(42)
    x1 = rng.normal(0.0, 1.0, 200)
    x2 = 2.0 * x1 + 1.0
    x3 = -x1 + 0.5
    x4 = rng.normal(0.0, 1.0, 200)
    data = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4})

    fig1, _ = plot_correlation_heatmap(data, method="pearson")
    fig2, _ = plot_correlation_heatmap(data, method="spearman")

    # 通用矩阵示例：双因素灵敏度输出的形状。
    outputs = np.array([
        [600.0, 500.0, 400.0],
        [800.0, 700.0, 600.0],
        [1000.0, 900.0, 800.0],
    ])
    fig3, _ = plot_matrix_heatmap(
        outputs,
        xlabels=["cost=80", "cost=100", "cost=120"],
        ylabels=["price=8", "price=10", "price=12"],
        title="Two-way sensitivity",
    )

    for fig, name in ((fig1, "heatmap_corr_pearson_out"),
                      (fig2, "heatmap_corr_spearman_out"),
                      (fig3, "heatmap_matrix_out")):
        print("已保存 ->", save_figure(fig, name))
        plt.close(fig)  # 释放画布内存，多图批量导出时避免累积占用。

    # 本例的简单验收：替换题目后可删除或改写。
    corr = correlation_matrix(data)
    assert np.isclose(corr.loc["x1", "x2"], 1.0, atol=1e-8)
    assert np.isclose(corr.loc["x1", "x3"], -1.0, atol=1e-8)


if __name__ == "__main__":
    main()
