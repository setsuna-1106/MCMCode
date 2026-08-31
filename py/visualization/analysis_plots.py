"""分析图模板：灵敏度分析与时间序列预测。

直接对接 ``py/evaluation/sensitivity.py`` 的四种输出
（一因素序列、双因素矩阵、Monte Carlo 输出数组）和
``py/forecasting/`` 各模板的 ``fitted``/``forecast``。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

try:
    from py.visualization.plot_style import new_axes
except ModuleNotFoundError:
    from plot_style import new_axes


def _as_1d(values, name: str) -> np.ndarray:
    """统一为一维有限数值数组。"""
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} 必须是非空且只含有限数值的一维数组")
    return array


def plot_one_way(values, outputs, *, parameter_name: str = "parameter",
                 output_name: str = "output", ax=None):
    """绘制一因素灵敏度扫描折线图。

    Args:
        values: 扫描的参数值（``one_way_sensitivity`` 的 ``values``）。
        outputs: 对应的输出序列（``outputs``）。
        parameter_name: 横轴参数名，用于图例和轴标签。
        output_name: 纵轴输出名。
        ax: 可选的已有坐标系。

    Returns:
        ``(fig, ax)``。
    """
    values = _as_1d(values, "values")
    outputs = _as_1d(outputs, "outputs")
    if values.size != outputs.size:
        raise ValueError("values 和 outputs 的长度必须一致")

    if ax is None:
        fig, ax = new_axes()
    else:
        fig = ax.figure

    ax.plot(values, outputs, marker="o")
    ax.set_xlabel(parameter_name)
    ax.set_ylabel(output_name)
    ax.set_title(f"One-way sensitivity: {parameter_name}")
    return fig, ax


def plot_two_way_heatmap(values_x, values_y, outputs, *,
                         name_x: str = "x", name_y: str = "y",
                         fmt: str = ".3g", ax=None):
    """绘制双因素灵敏度热力图。

    Args:
        values_x: 行方向参数扫描值（``two_way_sensitivity`` 的 ``values_x``）。
        values_y: 列方向参数扫描值（``values_y``）。
        outputs: ``(len(values_x), len(values_y))`` 输出矩阵。
        name_x: 行方向参数名。
        name_y: 列方向参数名。
        fmt: 注释数字格式。
        ax: 可选的已有坐标系。

    Returns:
        ``(fig, ax)``。
    """
    values_x = _as_1d(values_x, "values_x")
    values_y = _as_1d(values_y, "values_y")
    matrix = np.asarray(outputs, dtype=float)
    if matrix.shape != (values_x.size, values_y.size):
        raise ValueError("outputs 形状必须是 (len(values_x), len(values_y))")

    if ax is None:
        fig, ax = new_axes()
    else:
        fig = ax.figure

    frame = pd.DataFrame(matrix, index=[f"{v:g}" for v in values_x],
                         columns=[f"{v:g}" for v in values_y])
    sns.heatmap(frame, annot=True, fmt=fmt, cmap="viridis",
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_xlabel(name_y)
    ax.set_ylabel(name_x)
    ax.set_title(f"Two-way: {name_x} x {name_y}")
    return fig, ax


def plot_monte_carlo(outputs, *, quantiles=(0.05, 0.5, 0.95),
                     bins: int = 30, output_name: str = "output", ax=None):
    """绘制 Monte Carlo 输出分布直方图并标注分位数。

    Args:
        outputs: 随机扰动的输出数组（``monte_carlo_sensitivity`` 的 ``outputs``）。
        quantiles: 要标注的分位数序列。
        bins: 直方图分组数。
        output_name: 输出量名称，用于轴标签。
        ax: 可选的已有坐标系。

    Returns:
        ``(fig, ax)``。
    """
    outputs = _as_1d(outputs, "outputs")
    quantiles = np.asarray(quantiles, dtype=float).reshape(-1)
    if np.any((quantiles <= 0.0) | (quantiles >= 1.0)):
        raise ValueError("quantiles 必须落在 (0, 1) 区间")

    if ax is None:
        fig, ax = new_axes()
    else:
        fig = ax.figure

    ax.hist(outputs, bins=bins, density=True, alpha=0.6,
            edgecolor="white", label="Outputs")
    # 分位线回答评审常问的“输出最差/最好能到多少”。
    marks = np.quantile(outputs, quantiles)
    for q, value in zip(quantiles, marks):
        ax.axvline(value, linestyle="--", linewidth=1.2,
                   label=f"q{q:.0%} = {value:.3g}")
    ax.set_xlabel(output_name)
    ax.set_ylabel("Density")
    ax.set_title("Monte Carlo output distribution")
    ax.legend()
    return fig, ax


def plot_forecast(history, *, fitted=None, forecast=None, lower=None,
                  upper=None, ax=None):
    """绘制历史序列、拟合曲线、未来预测和置信区间。

    历史段实线、拟合段虚线、预测段虚线方点，竖直虚线标记
    历史与预测的分界；``lower``/``upper`` 与预测段等长时绘制置信带。

    Args:
        history: 按时间排列的历史观测序列。
        fitted: 与 ``history`` 等长的历史拟合值（可选）。
        forecast: 未来 ``steps`` 期预测值（可选）。
        lower: 置信区间下界，与 ``forecast`` 等长（可选）。
        upper: 置信区间上界，与 ``forecast`` 等长（可选）。
        ax: 可选的已有坐标系。

    Returns:
        ``(fig, ax)``。
    """
    history = _as_1d(history, "history")
    n_history = history.size
    if n_history < 2:
        raise ValueError("history 至少需要 2 个观测值")

    if fitted is not None:
        fitted = _as_1d(fitted, "fitted")
        if fitted.size != n_history:
            raise ValueError("fitted 必须与 history 等长")
    if forecast is not None:
        forecast = _as_1d(forecast, "forecast")
        if forecast.size < 1:
            raise ValueError("forecast 至少包含 1 期")
    if (lower is None) != (upper is None):
        raise ValueError("lower 和 upper 必须同时提供")
    if lower is not None:
        lower = _as_1d(lower, "lower")
        upper = _as_1d(upper, "upper")
        if forecast is None or lower.size != forecast.size \
                or upper.size != forecast.size:
            raise ValueError("lower/upper 必须与 forecast 等长")

    if ax is None:
        fig, ax = new_axes()
    else:
        fig = ax.figure

    ax.plot(np.arange(n_history), history, marker="o", markersize=3,
            label="History")
    if fitted is not None:
        ax.plot(np.arange(n_history), fitted, linestyle="--",
                label="Fitted")
    if forecast is not None:
        n_forecast = forecast.size
        # 预测段从最后一个历史点起步，保证曲线视觉上连续。
        xs = np.arange(n_history - 1, n_history + n_forecast)
        ax.plot(xs, np.r_[history[-1], forecast], linestyle="--",
                marker="s", markersize=3, label="Forecast")
        if lower is not None:
            ax.fill_between(
                xs,
                np.r_[history[-1], lower],
                np.r_[history[-1], upper],
                alpha=0.2,
                label="Confidence band",
            )
        ax.axvline(n_history - 1, linestyle=":", color="gray",
                   linewidth=1.0)
    ax.set_xlabel("Time index")
    ax.set_ylabel("Value")
    ax.set_title("Forecast")
    ax.legend()
    return fig, ax


def _bootstrap_repo_root() -> None:
    """直接运行本文件时把仓库根目录加入 sys.path 以导入其他模板。"""
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def main() -> None:
    # ====== 比赛时主要替换下面这部分 ======
    _bootstrap_repo_root()
    from py.evaluation.sensitivity import (
        monte_carlo_sensitivity,
        one_way_sensitivity,
        two_way_sensitivity,
    )
    from py.forecasting.holt import fit_holt
    from py.visualization.plot_style import save_figure, set_paper_style

    # 先定样式再建画布：spines、网格等风格在创建 Axes 时生效。
    set_paper_style()

    def model(params):
        return params["price"] * params["quantity"] - params["cost"]

    base_params = {"price": 10.0, "quantity": 80.0, "cost": 100.0}

    one_way = one_way_sensitivity(
        model, base_params, "quantity", np.linspace(60.0, 100.0, 9)
    )
    fig1, _ = plot_one_way(
        one_way["values"], one_way["outputs"],
        parameter_name="quantity", output_name="profit",
    )

    two_way = two_way_sensitivity(
        model, base_params, "price", [8.0, 10.0, 12.0],
        "cost", [80.0, 100.0, 120.0],
    )
    fig2, _ = plot_two_way_heatmap(
        two_way["values_x"], two_way["values_y"], two_way["outputs"],
        name_x="price", name_y="cost",
    )

    def sampler(rng):
        return {
            "price": rng.uniform(8.0, 12.0),
            "quantity": rng.uniform(70.0, 90.0),
            "cost": rng.uniform(80.0, 120.0),
        }

    mc = monte_carlo_sensitivity(model, sampler, n_samples=2000,
                                 random_state=42)
    fig3, _ = plot_monte_carlo(mc["outputs"], output_name="profit")

    # 预测示例：确定性趋势 + 波动序列，Holt 拟合并预测 4 期。
    t = np.arange(16)
    series = 100.0 + 2.5 * t + 6.0 * np.sin(2.0 * np.pi * t / 8.0)
    holt = fit_holt(series, steps=4)
    fig4, _ = plot_forecast(
        series,
        fitted=holt["fitted"],
        forecast=holt["forecast"],
        lower=holt["forecast"] * 0.94,
        upper=holt["forecast"] * 1.06,
    )

    for fig, name in ((fig1, "analysis_one_way_out"),
                      (fig2, "analysis_two_way_out"),
                      (fig3, "analysis_monte_carlo_out"),
                      (fig4, "analysis_forecast_out")):
        print("已保存 ->", save_figure(fig, name))
        plt.close(fig)

    # 本例的简单验收：替换题目后可删除或改写。
    assert one_way["outputs"].shape == (9,)
    assert two_way["outputs"].shape == (3, 3)
    assert mc["outputs"].shape == (2000,)
    assert holt["forecast"].shape == (4,)


if __name__ == "__main__":
    main()
