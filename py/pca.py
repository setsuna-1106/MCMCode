""" PCA 模板。
数据约定：X 的每一行是一个样本，每一列是一个指标。
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def pca_analysis(X, n_components=0.9, standardize=True):
    """执行 PCA，并返回常用结果

    n_components 可以写成：
    - 2：保留 2 个主成分；
    - 0.9：保留累计贡献率达到 90% 所需的主成分。
    """
    X = np.asarray(X, dtype=float)
    
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 1:
        raise ValueError("X 必须是至少包含 2 行的二维数值矩阵")

    scaler = StandardScaler() if standardize else None
    X_used = scaler.fit_transform(X) if scaler else X

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_used)
    contribution = pca.explained_variance_ratio_

    # 行对应原始指标，列对应主成分；标准化后可直接作载荷解释。
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    return {
        "scores": scores,  # 样本在主成分上的投影坐标
        "contribution": contribution,  # 各主成分贡献率
        "cumulative_contribution": np.cumsum(contribution),
        "components": pca.components_,  # 主轴系数：主成分 x 原始指标
        "loadings": loadings,
        "eigenvalues": pca.explained_variance_,
        "pca": pca,
        "scaler": scaler,
    }


if __name__ == "__main__":
    # 替换为题目数据：行 = 样本，列 = 指标。
    # 例如：X = np.loadtxt("data.csv", delimiter=",", skiprows=1)
    X = np.array([
        [5.1, 3.5, 1.4, 0.2],
        [4.9, 3.0, 1.4, 0.2],
        [6.3, 3.3, 4.7, 1.6],
        [5.8, 2.7, 4.1, 1.0],
        [6.7, 3.1, 5.6, 2.4],
    ])

    result = pca_analysis(X, n_components=2)

    print("各主成分贡献率：", result["contribution"])
    print("累计贡献率：", result["cumulative_contribution"])
    print("主成分得分：\n", result["scores"])
    print("主成分载荷：\n", result["loadings"])
