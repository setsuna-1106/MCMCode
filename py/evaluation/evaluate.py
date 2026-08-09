"""机器学习分类标准流程：划分数据 -> 预处理 -> 交叉验证 -> 调参 -> 测试。"""

from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 1. 准备数据：替换 X（特征）和 y（标签）
X, y = load_iris(return_X_y=True)

# 2. 划分数据：测试集只在最后使用一次
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. 建立Pipeline ：每次训练内部完成标准化，避免数据泄漏
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC())
])

# 4. 在训练集上交叉验证，评估模型稳定性
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy") # cv-交叉验证的折数
print(f"交叉验证准确率: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}") # std 方差衡量模型的稳定性

# 5. 在训练集上网格调参
param_grid = {
    "clf__C": [0.1, 1, 10],
    "clf__gamma": [0.01, 0.1, 1],
}
search = GridSearchCV(
    model, param_grid, cv=5, scoring="accuracy", n_jobs=1
)
search.fit(X_train, y_train)
print("最优参数:", search.best_params_)

# 6. 用最优模型测试：测试集不参与训练和调参
best_model = search.best_estimator_
y_pred = best_model.predict(X_test)
print(f"测试集准确率: {accuracy_score(y_test, y_pred):.2%}")
print(classification_report(y_test, y_pred, zero_division=0))
print("混淆矩阵:\n", confusion_matrix(y_test, y_pred))
