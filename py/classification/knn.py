""" KNN （k邻类）分类模板。

使用时，只需要把 main() 中的 X、y 替换为：
    X: shape 为 (样本数, 特征数) 的数值特征矩阵
    y: shape 为 (样本数,) 的分类标签
"""

from __future__ import annotations

import numpy as np

from sklearn.datasets import load_iris  # 示例数据

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def train_knn(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float = 0.2,
    n_neighbors: int = 5,  # 邻近元素的数量，奇数避免平票，多分类时仍有可能出现平票
    weights: str = "uniform",  # KNN 中邻居的权重
    random_state: int = 42,  # 随机种子
) -> tuple[object, dict[str, object]]:
    """标准化数据并训练 KNN 分类器。

    Args:
        X: ``(样本数, 特征数)`` 数值特征矩阵。
        y: 一维分类标签。
        n_neighbors: 投票时使用的邻居数。
        weights: 邻居权重方式。

    Returns:
        ``(model, result)``，model 为含标准化步骤的 Pipeline。
    """
    X = np.asarray(X)
    y = np.asarray(y)

    # 避免不规范输入
    if X.ndim != 2:
        raise ValueError("X 必须是二维特征矩阵，形状为 (样本数, 特征数)")
    if y.ndim != 1 or len(X) != len(y):
        raise ValueError("y 必须是一维标签，且样本数必须与 X 一致")
    if n_neighbors < 1:
        raise ValueError("n_neighbors 必须大于等于 1")

    # 测试集只用于最后评估，避免把待预测样本的信息带入训练过程。
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # KNN 基于距离，通常需要先统一各特征的量纲。
    model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights),
    )  # 使用 pipeline，保证标准化与模型训练始终绑定。
    model.fit(X_train, y_train)

    # 模型内部先在缩放后的训练集上寻找邻居，再对测试集分类。
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"KNN 准确率: {accuracy:.2%}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, zero_division=0))

    result = {
        "accuracy": accuracy,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }
    # 返回完整 pipeline，预测新样本时会自动执行相同的标准化。
    return model, result
 

def main() -> None:
    """默认使用 Iris 数据集演示
    使用时替换这一段数据。"""
    iris = load_iris()
    X, y = iris.data, iris.target

    model, _ = train_knn(X, y)
    print("示例预测:", model.predict(X[:3]))


if __name__ == "__main__":
    main()
