""" KMeans 聚类模板。

KMeans 是无监督学习算法，只需要准备：
    X: shape 为 (样本数, 特征数) 的数值特征矩阵

例如从 Excel 读取：
    import pandas as pd
    X = pd.read_excel("附件.xlsx").to_numpy()
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def train_kmeans(
    X: np.ndarray,
    *,
    n_clusters: int = 3,  # 聚类数
    random_state: int = 42,
    n_init: int = 10,  # KMeans 算法总运行次数，选择最优的结构
) -> tuple[object, dict[str, object]]:
    """标准化数据、训练 KMeans，并返回模型和聚类结果。"""
    # 聚类没有监督标签，模型会直接利用全部样本寻找簇结构。
    X = np.asarray(X)
    
    if X.ndim != 2:
        raise ValueError("X 必须是二维特征矩阵，形状为 (样本数, 特征数)")
    if n_clusters < 2:
        raise ValueError("n_clusters 必须大于等于 2")
    if n_clusters > len(X):
        raise ValueError("n_clusters 不能大于样本数")

    # KMeans 按距离分簇，先标准化可以避免量纲较大的指标主导距离。
    model = make_pipeline(
        StandardScaler(),
        KMeans(
            n_clusters=n_clusters,
            n_init=n_init,
            random_state=random_state,
        ),
    )
    # fit_predict 同时完成聚类拟合和每个样本的簇标签分配。
    labels = model.fit_predict(X)
    cluster_sizes = np.bincount(labels, minlength=n_clusters)

    print("KMeans 簇大小:", cluster_sizes)
    print("惯性:", model[-1].inertia_)
    # 惯性是样本到所属簇中心的距离平方和，可用于比较不同 k 的紧凑程度。

    result = {
        "labels": labels,
        "cluster_sizes": cluster_sizes,
        "inertia": model[-1].inertia_,
    }
    return model, result


def main() -> None:
    """默认使用 Iris 特征演示
    实际使用时替换这一段数据。"""
    iris = load_iris()
    X = iris.data

    model, _ = train_kmeans(X, n_clusters=3)
    print("示例预测簇:", model.predict(X[:3]))


if __name__ == "__main__":
    main()
