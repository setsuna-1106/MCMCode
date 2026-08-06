"""Logistic 回归分类模板。"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit_logistic(X, y, test_size=0.2, random_state=42, C=1.0, max_iter=1000):
    """训练模型并返回模型与测试集结果。X 为二维特征，y 为一维标签。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).reshape(-1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # 标准化只在 pipeline 中拟合，避免测试集信息泄漏到训练过程。
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=max_iter, random_state=random_state),
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    classifier = model.named_steps["logisticregression"]
    return {
        "model": model,
        "y_pred": y_pred,
        "y_proba": model.predict_proba(X_test),
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(
            y_test, y_pred, labels=classifier.classes_
        ),
        "classes": classifier.classes_,
        "coef": classifier.coef_,
    }


if __name__ == "__main__":
    # 示例数据 X = df[features], y = df[target]。
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)
    result = fit_logistic(X, y)
    print(f"准确率: {result['accuracy']:.2%}")
    print(result["report"])
    print("混淆矩阵:\n", result["confusion_matrix"])
    # 新数据预测：result["model"].predict(X_new)
