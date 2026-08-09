"""轻量 DBSCAN 聚类模板。

DBSCAN 是无监督学习算法，只需要准备：
    X: shape 为 (样本数, 特征数) 的数值特征矩阵

DBSCAN 不需要预先指定簇数，标签 -1 表示噪声点。
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def train_dbscan(
    X: np.ndarray,
    *,
    eps: float = 0.8, # 每个点的计算半径
    min_samples: int = 5,
) -> tuple[object, dict[str, object]]:
    """标准化数据、训练 DBSCAN，并返回模型和聚类结果。"""
    
    X = np.asarray(X)
    
    if X.ndim != 2:
        raise ValueError("X 必须是二维特征矩阵，形状为 (样本数, 特征数)")
    if eps <= 0:
        raise ValueError("eps 必须大于 0")
    if min_samples < 1:
        raise ValueError("min_samples 必须大于等于 1")

    model = make_pipeline(
        StandardScaler(),
        DBSCAN(eps=eps, min_samples=min_samples),
    )
    labels = model.fit_predict(X)

    unique_labels, counts = np.unique(labels, return_counts=True)
    distribution = dict(zip(unique_labels.tolist(), counts.tolist()))
    n_noise = int(distribution.get(-1, 0))
    n_clusters = len(distribution) - (1 if -1 in distribution else 0)

    print("DBSCAN 簇大小:", distribution)
    print(f"簇数量: {n_clusters}, 噪声点数量: {n_noise}")

    result = {
        "labels": labels,
        "distribution": distribution,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
    }
    return model, result


def main() -> None:
    """默认使用 Iris 特征演示
    实际使用时替换这一段数据。"""
    iris = load_iris()
    X = iris.data

    _, result = train_dbscan(X, eps=0.8, min_samples=5)
    print("前 3 个样本的簇标签:", result["labels"][:3])


if __name__ == "__main__":
    main()
