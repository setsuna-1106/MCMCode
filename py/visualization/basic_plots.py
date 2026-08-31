"""基础图模板：折线、散点、柱状、箱线和直方图。

这些函数只负责通用图形，不带建模语义；论文中最常见的
“数据长什么样”一类的探索图从这里取。
"""

from __future__ import annotations

import numpy as np

import matplotlib.pyplot as plt

try:
    from py.visualization.plot_style import new_axes
except ModuleNotFoundError:
    from plot_style import new_axes


def _as_series(values, name: str) -> list[np.ndarray]:
    """把一维或二维输入统一为“每条曲线一个一维数组”的列表。"""
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} 必须是非空且只含有限数值的一维或二维数组")
    return [array[i] for i in range(array.shape[0])]


def plot_line(y, *, x=None, labels=None, errors=None, xlabel: str = "",
              ylabel: str = "", title: str = "", ax=None):
    """绘制一条或多条折线，可选误差棒。

    Args:
        y: 单条曲线的一维数组，或 ``(曲线数, 样本数)`` 矩阵。
        x: 横坐标；为空时按 0, 1, 2, ... 编号。
        labels: 每条曲线的图例名。
        errors: 与 ``y`` 同结构的误差棒大小。
        xlabel: 横轴标签。
        ylabel: 纵轴标签。
        title: 图标题。
        ax: 可选的已有坐标系。

    Returns:
        ``(fig, ax)``。
    """
    series = _as_series(y, "y")
    n_series = len(series)
    n_points = series[0].size
    if any(curve.size != n_points for curve in series):
        raise ValueError("y 的每条曲线长度必须一致")

    if x is None:
        x = np.arange(n_points)
    else:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != n_points:
            raise ValueError("x 的长度必须与曲线长度一致")
    if labels is None:
        labels = [f"series {i + 1}" for i in range(n_series)]
    if len(labels) != n_series:
        raise ValueError("labels 的数量必须与曲线数一致")

    error_series = None
    if errors is not None:
        error_series = _as_series(errors, "errors")
        if len(error_series) != n_series or any(
            error.size != n_points for error in error_series
        ):
            raise ValueError("errors 必须与 y 的结构一致")

    if ax is None:
        fig, ax = new_axes()
    else:
        fig = ax.figure

    for i, curve in enumerate(series):
        if error_series is None:
            ax.plot(x, curve, marker="o", markersize=3, label=labels[i])
        else:
            ax.errorbar(x, curve, yerr=error_series[i], marker="o",
                        markersize=3, capsize=3, label=labels[i])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return fig, ax


def plot_scatter(x, y, *, color=None, colorbar_label: str = "",
                 xlabel: str = "", ylabel: str = "", title: str = "",
                 ax=None):
    """绘制散点图，可选第三个变量着色并显示颜色条。

    Args:
        x: 横坐标一维数组。
        y: 纵坐标一维数组。
        color: 可选的着色变量，与 ``x`` 等长。
        colorbar_label: 颜色条标签。
        xlabel: 横轴标签。
        ylabel: 纵轴标签。
        title: 图标题。
        ax: 可选的已有坐标系。

    Returns:
        ``(fig, ax)``。
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size == 0 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("x 和 y 必须是非空且只含有限数值的一维数组")
    if x.size != y.size:
        raise ValueError("x 和 y 的长度必须一致")

    if ax is None:
        fig, ax = new_axes()
    else:
        fig = ax.figure

    if color is None:
        ax.scatter(x, y, s=18, alpha=0.7)
    else:
        color = np.asarray(color, dtype=float).reshape(-1)
        if color.size != x.size:
            raise ValueError("color 必须与 x 等长")
        points = ax.scatter(x, y, s=18, c=color, cmap="viridis",
                            alpha=0.7)
        fig.colorbar(points, ax=ax, shrink=0.8, label=colorbar_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return fig, ax


def plot_bar(labels, values, *, errors=None, ylabel: str = "",
             title: str = "", ax=None):
    """绘制柱状图，可选误差棒。

    Args:
        labels: 每根柱子的类别名。
        values: 柱高数组。
        errors: 可选的误差棒大小，与 ``values`` 等长。
        ylabel: 纵轴标签。
        title: 图标题。
        ax: 可选的已有坐标系。

    Returns:
        ``(fig, ax)``。
    """
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("values 必须是非空且只含有限数值的一维数组")
    labels = [str(label) for label in labels]
    if len(labels) != values.size:
        raise ValueError("labels 的数量必须与 values 一致")
    if errors is not None:
        errors = np.asarray(errors, dtype=float).reshape(-1)
        if errors.size != values.size:
            raise ValueError("errors 必须与 values 等长")

    if ax is None:
        fig, ax = new_axes()
    else:
        fig = ax.figure

    positions = np.arange(values.size)
    ax.bar(positions, values, yerr=errors, capsize=3, alpha=0.8)
    ax.set_xticks(positions, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return fig, ax


def plot_box(groups, *, labels=None, ylabel: str = "", title: str = "",
             ax=None):
    """绘制一组或多组数据的箱线图，比较分布与离群点。

    Args:
        groups: 每组一个一维数组的序列。
        labels: 每组的类别名。
        ylabel: 纵轴标签。
        title: 图标题。
        ax: 可选的已有坐标系。

    Returns:
        ``(fig, ax)``。
    """
    groups = [np.asarray(group, dtype=float).reshape(-1) for group in groups]
    if not groups or any(group.size == 0 for group in groups):
        raise ValueError("每组数据必须是非空数组")
    for group in groups:
        if not np.isfinite(group).all():
            raise ValueError("groups 只能包含有限数值")
    if labels is None:
        labels = [f"group {i + 1}" for i in range(len(groups))]
    if len(labels) != len(groups):
        raise ValueError("labels 的数量必须与分组数一致")

    if ax is None:
        fig, ax = new_axes()
    else:
        fig = ax.figure

    ax.boxplot(groups)
    # 手动设置刻度标签，兼容不同版本的 boxplot 参数命名。
    ax.set_xticks(range(1, len(groups) + 1), [str(label) for label in labels])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return fig, ax


def plot_histogram(data, *, bins: int = 20, density: bool = False,
                   xlabel: str = "", title: str = "", ax=None):
    """绘制直方图，观察单变量分布形状。

    Args:
        data: 一维数值数组。
        bins: 分组数。
        density: 是否归一化为密度（面积和为 1）。
        xlabel: 横轴标签。
        title: 图标题。
        ax: 可选的已有坐标系。

    Returns:
        ``(fig, ax)``。
    """
    data = np.asarray(data, dtype=float).reshape(-1)
    if data.size == 0 or not np.isfinite(data).all():
        raise ValueError("data 必须是非空且只含有限数值的一维数组")

    if ax is None:
        fig, ax = new_axes()
    else:
        fig = ax.figure

    ax.hist(data, bins=bins, density=density, alpha=0.7,
            edgecolor="white")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density" if density else "Frequency")
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
    rng = np.random.default_rng(42)

    # 折线：两条带误差棒的序列。
    x = np.linspace(0.0, 10.0, 11)
    y1 = 2.0 * x + 1.0
    y2 = 1.5 * x
    fig1, _ = plot_line(
        np.vstack([y1, y2]),
        x=x,
        labels=["model A", "model B"],
        errors=np.vstack([rng.uniform(0.2, 0.8, x.size)] * 2),
        xlabel="x", ylabel="y", title="Line with error bars",
    )

    # 散点：第三个变量着色。
    xs = rng.uniform(0.0, 10.0, 120)
    ys = xs + rng.normal(0.0, 1.0, xs.size)
    fig2, _ = plot_scatter(
        xs, ys, color=ys, colorbar_label="value",
        xlabel="x", ylabel="y", title="Scatter with color",
    )

    # 柱状：三方案对比 + 误差棒。
    fig3, _ = plot_bar(
        ["plan A", "plan B", "plan C"],
        [85.0, 92.0, 78.0],
        errors=[3.0, 4.0, 2.5],
        ylabel="score", title="Bar with error bars",
    )

    # 箱线：三组分布对比。
    fig4, _ = plot_box(
        [rng.normal(50, 5, 60), rng.normal(53, 8, 60),
         rng.normal(48, 3, 60)],
        labels=["A", "B", "C"],
        ylabel="value", title="Box plot",
    )

    # 直方图。
    fig5, _ = plot_histogram(
        rng.normal(0.0, 1.0, 500), bins=25, density=True,
        xlabel="value", title="Histogram",
    )

    for fig, name in ((fig1, "basic_line_out"),
                      (fig2, "basic_scatter_out"),
                      (fig3, "basic_bar_out"),
                      (fig4, "basic_box_out"),
                      (fig5, "basic_histogram_out")):
        print("已保存 ->", save_figure(fig, name))
        plt.close(fig)

    # 本例的简单验收：替换题目后可删除或改写。
    assert y1.size == x.size and y2.size == x.size
    assert xs.size == ys.size


if __name__ == "__main__":
    main()
