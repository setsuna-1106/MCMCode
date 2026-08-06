"""随机森林分类模板。

使用时：
    X: shape 为 (样本数, 特征数) 的数值特征矩阵
    y: shape 为 (样本数,) 的分类标签

可以从 CSV 读取：
    import pandas as pd
    df = pd.read_csv("data.csv")
    X = df.drop(columns="target").to_numpy()
    y = df["target"].to_numpy()
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def train_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str] | None = None,
    *,
    test_size: float = 0.2, # 测试集所占大小
    n_estimators: int = 100, # 随机森林中树的数量
    max_depth: int | None = None, # 决策树所能达到的最大深度
    random_state: int = 42,
) -> tuple[RandomForestClassifier, dict[str, object]]:
    """训练随机森林分类器并返回模型和评估结果。"""
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError("X 必须是二维特征矩阵，形状为 (样本数, 特征数)")
    if y.ndim != 1 or len(X) != len(y):
        raise ValueError("y 必须是一维标签，且样本数必须与 X 一致")
    if feature_names is not None and len(feature_names) != X.shape[1]:
        raise ValueError("feature_names 的数量必须与特征数一致")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1, # 训练时cpu并行数量，-1表示使用全部核心
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    importances = model.feature_importances_

    print(f"随机森林准确率: {accuracy:.2%}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, zero_division=0))

    if feature_names is not None:
        order = np.argsort(importances)[::-1]
        print("特征重要性（从高到低）:")
        for index in order:
            print(f"  {feature_names[index]}: {importances[index]:.4f}")

    result = {
        "accuracy": accuracy,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "feature_importances": importances,
    }
    return model, result


def main() -> None:
    """使用 Iris 数据集演示；
    实际使用时替换这一段数据。"""
    iris = load_iris()
    X, y = iris.data, iris.target

    model, result = train_random_forest(
        X,
        y,
        feature_names=iris.feature_names,
    )

    # 训练完成后可直接预测新样本：
    new_samples = X[:3]
    print("预测结果:", model.predict(new_samples))


if __name__ == "__main__":
    main()
