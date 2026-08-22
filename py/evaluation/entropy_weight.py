import numpy as np


def entropy_weight(X, direction):
    """返回熵权法中的指标权重。"""
    X = np.asarray(X, dtype=float)
    direction = np.asarray(direction, dtype=float)
    # direction=1 表示收益型，direction=-1 表示成本型。
    
    # 先把各指标转为同向的 [0, 1] 数值，后续熵值才可以跨列比较。
    Xn = np.zeros_like(X)
    for j in range(X.shape[1]):
        column = X[:, j]
        span = column.max() - column.min()
        if span < 1e-7:
            # 常数列没有区分度，归一化后熵为 1，最终差异系数为 0。
            Xn[:, j] = 1
        elif direction[j] == 1:
            Xn[:, j] = (column - column.min()) / span
        else:
            Xn[:, j] = (column.max() - column) / span

    # P 是列内比例；p=0 时按极限约定 p*log(p)=0。
    P = Xn / Xn.sum(axis=0)
    P[P == 0] = 1
    entropy = -(P * np.log(P)).sum(axis=0) / np.log(X.shape[0])
    # 熵越小表示差异越大，因此差异系数越大、权重越高。
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
