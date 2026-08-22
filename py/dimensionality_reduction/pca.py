""" PCA 模板。
数据约定：X 的每一行是一个样本，每一列是一个指标。
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def pca_analysis(X, n_components=0.9, standardize=True):
    """执行 PCA 并返回降维与解释结果。

    Args:
        X: ``(样本数, 指标数)`` 数据矩阵。
        n_components: 主成分数量，或目标累计贡献率（如 ``0.9``）。
        standardize: 是否先按列标准化。

    Returns:
        包含得分、贡献率、主轴系数、载荷、特征值和拟合对象的字典。
    """
    # 每行是样本、每列是指标；PCA 只能处理有限的数值矩阵。
    X = np.asarray(X, dtype=float)

    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 1:
        raise ValueError("X 必须是至少包含 2 行的二维数值矩阵")

    # 标准化后各指标以标准差为单位参与方差分解，适合量纲不同的指标。
    scaler = StandardScaler() if standardize else None
    X_used = scaler.fit_transform(X) if scaler else X

    # fit_transform 得到样本在保留主成分方向上的投影坐标，即主成分得分。
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_used)
    contribution = pca.explained_variance_ratio_

    # explained_variance_ratio_ 表示每个主成分解释的方差比例。
    # components_ 的行是主成分、列是原指标；转置后乘以标准差可得载荷。
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    return {
        "scores": scores,  # 样本在主成分上的投影坐标
        "contribution": contribution,  # 各主成分贡献率
        "cumulative_contribution": np.cumsum(contribution),  # 累计贡献率
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
