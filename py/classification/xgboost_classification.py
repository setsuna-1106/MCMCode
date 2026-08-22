#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""XGBoost 分类最小可执行模板。

使用 xgboost.XGBClassifier 完成：
    数据划分 -> 模型训练 -> 分类评估 -> 特征重要性 -> 新样本预测

XGBoost 不要求标准化，但 X 必须是数值特征矩阵；比赛时主要替换 main()
中的 X、y 和模型参数。
"""

import sys

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


def train_xgboost_classifier(
    X,
    y,
    *,
    test_size=0.2,
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
):
    """训练 XGBClassifier 并返回测试集分类结果。

    Args:
        X, y: 数值特征矩阵和分类标签。
        n_estimators, max_depth, learning_rate: 树模型主要超参数。

    Returns:
        ``(model, result)``，result 含原始类别预测、分类指标和标签编码器。
    """
    # XGBoost 直接接收数值特征；先统一类型并检查有限性。
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("X 必须是二维特征矩阵，且样本数必须与 y 一致")
    if not np.all(np.isfinite(X)):
        raise ValueError("X 不能包含 NaN 或无穷值")
    if y.ndim != 1:
        raise ValueError("y 必须是一维分类标签")

    # 将字符串或非连续标签编码为从 0 开始的整数，训练后再还原。
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    n_classes = len(label_encoder.classes_)
    if n_classes < 2:
        raise ValueError("y 至少需要包含 2 个类别")

    # 对编码后的标签做分层切分，保持各类别在两部分中的比例。
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )

    # 二分类和多分类需要不同的目标函数；n_jobs=1 便于比赛环境稳定运行。
    model_params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "objective": "binary:logistic" if n_classes == 2 else "multi:softprob",
        "eval_metric": "logloss" if n_classes == 2 else "mlogloss",
        "n_jobs": 1,
        "random_state": random_state,
        "verbosity": 0,
    }
    if n_classes > 2:
        model_params["num_class"] = n_classes

    model = XGBClassifier(**model_params)
    model.fit(X_train, y_train)

    # 评估前将预测标签还原为调用者传入的原始类别类型。
    y_pred_encoded = model.predict(X_test).astype(int)
    y_test_original = label_encoder.inverse_transform(y_test)
    y_pred_original = label_encoder.inverse_transform(y_pred_encoded)
    accuracy = accuracy_score(y_test_original, y_pred_original)

    result = {
        "accuracy": accuracy,
        "report": classification_report(
            y_test_original,
            y_pred_original,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_test_original,
            y_pred_original,
        ),
        "X_test": X_test,
        "y_test": y_test_original,
        "y_pred": y_pred_original,
        "classes": label_encoder.classes_,
        "label_encoder": label_encoder,
        "feature_importances": model.feature_importances_,
    }
    return model, result


def predict_xgboost_classifier(model, X_new, label_encoder):
    """预测新样本，并将编码后的类别还原为原始标签。"""
    X_new = np.asarray(X_new, dtype=float)
    if X_new.ndim == 1:
        X_new = X_new.reshape(1, -1)
    if X_new.ndim != 2 or not np.all(np.isfinite(X_new)):
        raise ValueError("X_new 必须是二维且只包含有限数值")
    encoded = model.predict(X_new).astype(int)
    return label_encoder.inverse_transform(encoded)


def main():
    # ====== 比赛时主要替换下面这部分 ======
    data = load_iris()
    X, y = data.data, data.target

    model, result = train_xgboost_classifier(X, y)

    print(f"XGBoost 准确率: {result['accuracy']:.2%}")
    print("分类报告:")
    print(result["report"])
    print("混淆矩阵:\n", result["confusion_matrix"])
    print("特征重要性:", np.round(result["feature_importances"], 6))
    print(
        "示例预测:",
        predict_xgboost_classifier(model, X[:3], result["label_encoder"]),
    )

    if "--csv" in sys.argv:
        np.savetxt(
            "xgboost_classification_out.csv",
            np.column_stack((result["y_test"], result["y_pred"])),
            delimiter=",",
            fmt="%s",
            header="真实标签,预测标签",
            comments="",
        )
        print("已保存 -> xgboost_classification_out.csv")


if __name__ == "__main__":
    main()
