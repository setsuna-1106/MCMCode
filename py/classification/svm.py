""" SVM 分类模板。

使用时，只需要把 main() 中的 X、y 替换为：
    X: shape 为 (样本数, 特征数) 的数值特征矩阵
    y: shape 为 (样本数,) 的分类标签
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def train_svm(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float = 0.2,
    kernel: str = "rbf",  # 训练核函数
    C: float = 1.0,  # 误分类惩罚系数
    gamma: str | float = "scale",
    random_state: int = 42,
) -> tuple[object, dict[str, object]]:
    """标准化数据并训练 SVM 分类器。

    Args:
        X: ``(样本数, 特征数)`` 数值特征矩阵。
        y: 一维分类标签。
        kernel: SVM 核函数。
        C: 误分类惩罚系数。

    Returns:
        ``(model, result)``，model 为含标准化步骤的 Pipeline。
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError("X 必须是二维特征矩阵，形状为 (样本数, 特征数)")
    if y.ndim != 1 or len(X) != len(y):
        raise ValueError("y 必须是一维标签，且样本数必须与 X 一致")

    # 测试集只在训练结束后使用，评估结果才具有独立性。
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # SVM 对特征尺度敏感，因此把标准化和分类器放进同一个 pipeline。
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel=kernel, C=C, gamma=gamma),
    )
    model.fit(X_train, y_train)

    # pipeline 会先缩放测试特征，再调用 SVM 进行预测。
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"SVM 准确率: {accuracy:.2%}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, zero_division=0))

    result = {
        "accuracy": accuracy,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }
    return model, result


def main() -> None:
    """使用时替换这一段数据。"""
    iris = load_iris()
    X, y = iris.data, iris.target

    model, _ = train_svm(X, y)
    print("示例预测:", model.predict(X[:3]))


if __name__ == "__main__":
    main()
