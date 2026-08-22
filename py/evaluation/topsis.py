import numpy as np

def topsis(X, w, direction):
    """计算 TOPSIS 贴近度和方案排序。

    Args:
        X: ``(方案数, 指标数)`` 决策矩阵。
        w: 指标权重向量。
        direction: 指标方向，成本型指标取负值。

    Returns:
        ``(C, order, D_plus, D_minus)``，分别为贴近度、排序和两类距离。
    """
    # 先按列做向量归一化，消除不同指标量纲的影响。
    Xn = X / np.sqrt((X**2).sum(axis=0))

    # 成本型指标取倒数转为“越大越好”，再乘指标权重得到加权矩阵。
    Xb = np.where(direction[:, None] < 0, 1.0 / np.maximum(Xn, 1e-12), Xn)
    V = Xb * w

    # 正理想解取每列最大值，负理想解取每列最小值。
    A_plus = V.max(axis=0)
    A_minus = V.min(axis=0)

    # 计算每个方案到正、负理想解的欧氏距离。
    D_plus = np.sqrt(((V - A_plus) ** 2).sum(axis=1))
    D_minus = np.sqrt(((V - A_minus) ** 2).sum(axis=1))

    # 越靠近正理想解、越远离负理想解，贴近度 C 越大。
    C = D_minus / (D_plus + D_minus)
    order = np.argsort(-C)

    return C, order, D_plus, D_minus
