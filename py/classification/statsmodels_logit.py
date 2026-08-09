"""statsmodels Logit classification template."""

from __future__ import annotations

import numpy as np
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def fit_logit(
    X,
    y,
    *,
    test_size=0.2,
    random_state=42,
    threshold=0.5,
):
    """Fit binary Logit and return inference and test-set metrics."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).reshape(-1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("X must be a 2D feature matrix and match y in length")
    if not np.all(np.isin(np.unique(y), [0, 1])):
        raise ValueError("y must contain binary labels 0 and 1")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    X_train_sm = sm.add_constant(X_train, has_constant="add")
    X_test_sm = sm.add_constant(X_test, has_constant="add")

    result = sm.Logit(y_train, X_train_sm).fit(disp=False)
    y_proba = np.asarray(result.predict(X_test_sm))
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "model": result,
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,
        "y_proba": y_proba,
        "y_pred": y_pred,
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]),
        "odds_ratio": np.exp(result.params),
        "llr_pvalue": result.llr_pvalue,
    }


def predict_logit(result, X_new, threshold=0.5):
    """Return probabilities and labels for new rows."""
    X_new = np.asarray(X_new, dtype=float)
    if X_new.ndim == 1:
        X_new = X_new.reshape(1, -1)
    proba = np.asarray(result.predict(sm.add_constant(X_new, has_constant="add")))
    return proba, (proba >= threshold).astype(int)


if __name__ == "__main__":
    # Replace this block with X = df[features].to_numpy(), y = df[target].to_numpy().
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=400,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )
    output = fit_logit(X, y)
    print(f"accuracy: {output['accuracy']:.2%}")
    print(output["report"])
    print("confusion matrix:\n", output["confusion_matrix"])
    print("odds ratios:", output["odds_ratio"])
    print("LLR p-value:", output["llr_pvalue"])
