"""模型诊断图模板：回归、分类、聚类和 PCA。

每个函数直接对接现有建模模板的返回值（``result`` 字典中的
``y_test``/``y_pred``/``y_proba``/``confusion_matrix``/``labels``、
PCA 的 ``scores``/``loadings``），论文出图时不用重新组织数据。
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import auc, mean_squared_error, r2_score, roc_curve

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


def plot_true_vs_pred(y_true, y_pred, *, ax=None):
    """绘制真实值-预测值散点图与 y=x 参考线。

    Args:
        y_true: 测试集真实值（回归模板 ``result["y_test"]``）。
        y_pred: 测试集预测值（回归模板 ``result["y_pred"]``）。
        ax: 可选的已有坐标系。

    Returns:
        ``(fig, ax)``，标题标注 R2 和 RMSE。
    """
    y_true = _as_1d(y_true, "y_true")
    y_pred = _as_1d(y_pred, "y_pred")
    if y_true.size != y_pred.size:
        raise ValueError("y_true 和 y_pred 的长度必须一致")

    if ax is None:
        fig, ax = new_axes()
    else:
        fig = ax.figure

    lower = min(y_true.min(), y_pred.min())
    upper = max(y_true.max(), y_pred.max())
    ax.plot([lower, upper], [lower, upper], linestyle="--", color="gray",
            linewidth=1.0, label="y = x")
    ax.scatter(y_true, y_pred, s=18, alpha=0.7, label="Test samples")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal")
    ax.set_xlabel("True value")
    ax.set_ylabel("Predicted value")

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    ax.set_title(f"$R^2$ = {r2:.3f}, RMSE = {rmse:.3f}")
    ax.legend()
    return fig, ax


def plot_residuals(fitted, residuals, *, ax=None):
    """绘制残差-拟合值散点图，检查异方差与非线性结构。

    Args:
        fitted: 拟合值（或预测值），横坐标。
        residuals: 残差 = 真实值 - 拟合值，纵坐标。
        ax: 可选的已有坐标系。

    Returns:
        ``(fig, ax)``。
    """
    fitted = _as_1d(fitted, "fitted")
    residuals = _as_1d(residuals, "residuals")
    if fitted.size != residuals.size:
        raise ValueError("fitted 和 residuals 的长度必须一致")

    if ax is None:
        fig, ax = new_axes()
    else:
        fig = ax.figure

    ax.axhline(0.0, linestyle="--", color="gray", linewidth=1.0)
    ax.scatter(fitted, residuals, s=18, alpha=0.7)
    ax.set_xlabel("Fitted value")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals vs fitted")
    return fig, ax


def plot_residual_histogram(residuals, *, bins: int = 20, ax=None):
    """绘制残差直方图并叠加同均值方差的正态参考曲线。

    Args:
        residuals: 残差序列。
        bins: 直方图分组数。
        ax: 可选的已有坐标系。

    Returns:
        ``(fig, ax)``。
    """
    from scipy.stats import norm

    residuals = _as_1d(residuals, "residuals")
    if ax is None:
        fig, ax = new_axes()
    else:
        fig = ax.figure

    ax.hist(residuals, bins=bins, density=True, alpha=0.6,
            edgecolor="white", label="Residuals")
    # 正态参考线用于目测偏态与厚尾，不替代 Jarque-Bera 等正式检验。
    grid = np.linspace(residuals.min(), residuals.max(), 200)
    ax.plot(grid, norm.pdf(grid, residuals.mean(), residuals.std(ddof=1)),
            linestyle="--", label="Normal reference")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Density")
    ax.set_title("Residual distribution")
    ax.legend()
    return fig, ax


def plot_confusion_matrix(cm, *, labels=None, normalize: bool = False,
                          ax=None):
    """绘制混淆矩阵热力图。

    Args:
        cm: ``(类别数, 类别数)`` 计数矩阵，行是真实类别、列是预测类别。
        labels: 类别名；为空时自动编号。
        normalize: 是否按真实类别行归一化为比例。
        ax: 可选的已有坐标系。

    Returns:
        ``(fig, ax)``。

    Raises:
        ValueError: 矩阵不是非负方阵或归一化时存在全零行时抛出。
    """
    cm = np.asarray(cm, dtype=float)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1] or cm.size == 0:
        raise ValueError("cm 必须是非空方阵")
    if np.any(cm < 0) or not np.isfinite(cm).all():
        raise ValueError("cm 只能包含非负有限数值")
    if normalize and np.any(cm.sum(axis=1) == 0):
        raise ValueError("归一化要求 cm 的每一行至少有一个样本")

    shown = cm / cm.sum(axis=1, keepdims=True) if normalize else cm
    if labels is None:
        labels = [str(i) for i in range(cm.shape[0])]
    if len(labels) != cm.shape[0]:
        raise ValueError("labels 的数量必须与类别数一致")

    if ax is None:
        fig, ax = new_axes()
    else:
        fig = ax.figure

    image = ax.imshow(shown, cmap="Blues")
    fig.colorbar(image, ax=ax, shrink=0.8)
    threshold = shown.max() / 2.0
    fmt = ".2f" if normalize else "g"
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # 深色格子用白字，保证注释在任何底色上可读。
            color = "white" if shown[i, j] > threshold else "black"
            ax.text(j, i, format(shown[i, j], fmt), ha="center",
                    va="center", color=color, fontsize=10)
    ax.set_xticks(range(cm.shape[1]), list(labels))
    ax.set_yticks(range(cm.shape[0]), list(labels))
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix (row-normalized)" if normalize
                 else "Confusion matrix")
    ax.grid(False)
    return fig, ax


def plot_roc_curve(y_true, y_proba, *, ax=None):
    """绘制二分类 ROC 曲线并标注 AUC。

    Args:
        y_true: 只含 0/1 的真实标签。
        y_proba: 正类概率（``result["y_proba"]`` 的正类列）。
        ax: 可选的已有坐标系。

    Returns:
        ``(fig, ax)``，标题标注 AUC。

    Raises:
        ValueError: 标签不是 0/1 或概率越界时抛出。
    """
    y_true = _as_1d(y_true, "y_true")
    y_proba = _as_1d(y_proba, "y_proba")
    if not np.isin(y_true, [0.0, 1.0]).all():
        raise ValueError("y_true 必须只包含 0 和 1")
    if y_true.size != y_proba.size:
        raise ValueError("y_true 和 y_proba 的长度必须一致")
    if np.any((y_proba < 0.0) | (y_proba > 1.0)):
        raise ValueError("y_proba 必须落在 [0, 1] 区间")

    if ax is None:
        fig, ax = new_axes()
    else:
        fig = ax.figure

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_value = float(auc(fpr, tpr))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {auc_value:.3f})")
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray",
            linewidth=1.0, label="Random guess")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend()
    return fig, ax


def plot_cluster_scatter(X2d, labels, *, centers=None, ax=None):
    """绘制二维聚类散点图，标签 -1 按噪声点显示。

    Args:
        X2d: ``(样本数, 2)`` 坐标矩阵（原始两特征或 PCA 前两主成分）。
        labels: 每个样本的簇标签（DBSCAN 的噪声点为 -1）。
        centers: 可选的 ``(簇数, 2)`` 簇中心坐标。
        ax: 可选的已有坐标系。

    Returns:
        ``(fig, ax)``。
    """
    points = np.asarray(X2d, dtype=float)
    labels = np.asarray(labels).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] != labels.size:
        raise ValueError("X2d 必须是 (样本数, 2)，且与 labels 长度一致")

    if ax is None:
        fig, ax = new_axes()
    else:
        fig = ax.figure

    # 按簇分别绘制以获得图例；噪声点 -1 单独用灰色 x 表示。
    for cluster in np.unique(labels):
        mask = labels == cluster
        if cluster == -1:
            ax.scatter(points[mask, 0], points[mask, 1], marker="x",
                       color="gray", alpha=0.7, label="Noise")
        else:
            ax.scatter(points[mask, 0], points[mask, 1], s=22, alpha=0.7,
                       label=f"Cluster {cluster}")
    if centers is not None:
        centers = np.asarray(centers, dtype=float)
        ax.scatter(centers[:, 0], centers[:, 1], marker="X", s=90,
                   color="black", label="Centers")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title("Cluster result")
    ax.legend()
    return fig, ax


def plot_pca_biplot(scores, loadings, *, feature_names=None, ax=None):
    """绘制 PCA 得分-载荷双标图（biplot）。

    散点是样本在前两个主成分上的得分，箭头是原始指标在该平面的
    载荷方向：箭头越长，指标对这两个主成分的影响越大。

    Args:
        scores: ``(样本数, 2)`` 前两主成分得分（``result["scores"]``）。
        loadings: ``(指标数, 2)`` 载荷（``result["loadings"]``）。
        feature_names: 指标名；为空时自动编号。
        ax: 可选的已有坐标系。

    Returns:
        ``(fig, ax)``。
    """
    scores = np.asarray(scores, dtype=float)
    loadings = np.asarray(loadings, dtype=float)
    if scores.ndim != 2 or scores.shape[1] != 2:
        raise ValueError("scores 必须是 (样本数, 2)")
    if loadings.ndim != 2 or loadings.shape[1] != 2:
        raise ValueError("loadings 必须是 (指标数, 2)")
    if feature_names is None:
        feature_names = [f"x{j + 1}" for j in range(loadings.shape[0])]
    if len(feature_names) != loadings.shape[0]:
        raise ValueError("feature_names 的数量必须与指标数一致")

    if ax is None:
        fig, ax = new_axes()
    else:
        fig = ax.figure

    ax.scatter(scores[:, 0], scores[:, 1], s=18, alpha=0.6,
               color="tab:blue", label="Scores")
    # 载荷按主成分分别缩放，让最长的箭头到达约 85% 的绘图范围。
    limits = np.abs(scores).max(axis=0)
    scales = limits * 0.85 / np.abs(loadings).max(axis=0)
    scaled = loadings * scales
    for name, (tip_x, tip_y) in zip(feature_names, scaled):
        ax.annotate(
            "",
            xy=(tip_x, tip_y),
            xytext=(0.0, 0.0),
            arrowprops={"arrowstyle": "->", "color": "tab:red",
                        "linewidth": 1.2},
        )
        ax.text(tip_x * 1.08, tip_y * 1.08, name, color="tab:red",
                fontsize=9)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA biplot")
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
    from sklearn.datasets import load_iris

    from py.dimensionality_reduction.pca import pca_analysis
    from py.visualization.plot_style import save_figure, set_paper_style

    # 先定样式再建画布：spines、网格等风格在创建 Axes 时生效。
    set_paper_style()
    rng = np.random.default_rng(42)

    # 回归诊断：拟合带噪声的直线，制造 y_test / y_pred / 残差。
    x = rng.uniform(0.0, 10.0, 80)
    y_true = 3.0 * x + 2.0 + rng.normal(0.0, 1.0, x.size)
    y_pred = 3.0 * x + 2.5  # 故意带一点系统偏差
    residuals = y_true - y_pred

    fig1, _ = plot_true_vs_pred(y_true, y_pred)
    fig2, _ = plot_residuals(y_pred, residuals)
    fig3, _ = plot_residual_histogram(residuals)

    # 分类评估：混淆矩阵与 ROC。
    fig4, _ = plot_confusion_matrix(
        np.array([[8, 2], [1, 9]]), labels=["0", "1"]
    )
    fig5, _ = plot_roc_curve(
        [0, 0, 0, 1, 1, 1], [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]
    )

    # 聚类：三个手工簇 + 已知中心。
    blob = lambda center, n: center + rng.normal(0.0, 0.6, (n, 2))
    cluster_centers = np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 3.5]])
    X2d = np.vstack([blob(c, 40) for c in cluster_centers])
    labels = np.repeat([0, 1, 2], 40)
    fig6, _ = plot_cluster_scatter(X2d, labels, centers=cluster_centers)

    # PCA biplot：直接消费 pca_analysis 的输出。
    iris = load_iris()
    pca_result = pca_analysis(iris.data, n_components=2)
    fig7, _ = plot_pca_biplot(
        pca_result["scores"],
        pca_result["loadings"],
        feature_names=["sepal_l", "sepal_w", "petal_l", "petal_w"],
    )

    for fig, name in (
        (fig1, "model_true_vs_pred_out"),
        (fig2, "model_residuals_out"),
        (fig3, "model_residual_hist_out"),
        (fig4, "model_confusion_out"),
        (fig5, "model_roc_out"),
        (fig6, "model_cluster_out"),
        (fig7, "model_biplot_out"),
    ):
        print("已保存 ->", save_figure(fig, name))
        plt.close(fig)

    # 本例的简单验收：替换题目后可删除或改写。
    assert residuals.size == y_true.size
    assert pca_result["scores"].shape == (iris.data.shape[0], 2)
    assert pca_result["loadings"].shape == (iris.data.shape[1], 2)


if __name__ == "__main__":
    main()
