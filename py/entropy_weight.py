import numpy as np


def entropy_weight(X, direction):
    """放回熵权法中的权重"""
    X = np.asarray(X, dtype=float)
    direction = np.asarray(direction, dtype=float)
    # 1: 收益性, -1: 成本性
    
    Xn = np.zeros_like(X)
    for j in range(X.shape[1]):
        column = X[:, j]
        span = column.max() - column.min()
        if span <1e-7 :
            Xn[:, j] = 1 #差别可以忽略，权重为0
        elif direction[j] == 1:
            Xn[:, j] = (column - column.min()) / span
        else:
            Xn[:, j] = (column.max() - column) / span

    P = Xn / Xn.sum(axis=0)
    P[P == 0] = 1  # p * log(p) = 0 when p = 0
    entropy = -(P * np.log(P)).sum(axis=0) / np.log(X.shape[0])
    difference = 1 - entropy
    weights = difference / difference.sum()
    return weights


if __name__ == "__main__":
    #将X,direction替换成实际矩阵
    X = np.array([[10, 100], [20, 80], [30, 120]])
    direction = [
        -1,
        1,
    ]
    weights= entropy_weight(X, direction)
    print("weights:", weights)
