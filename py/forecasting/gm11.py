import numpy as np


def gm11(data, steps=1):
    """Return (forecast, (a, b), fitted) for a GM(1,1) model."""
    # GM(1,1) 适合短序列趋势外推，输入要求为非负时间序列。
    x0 = np.asarray(data, dtype=float).reshape(-1)
    if len(x0) < 4:
        raise ValueError("GM(1,1) 至少需要 4 个数据")
    if not np.isfinite(x0).all() or (x0 < 0).any():
        raise ValueError("数据必须是非负有限数")
    if steps < 1:
        raise ValueError("steps 必须大于 0")

    # 一次累加生成 x1，降低原始序列随机波动后建立灰色微分方程。
    x1 = np.cumsum(x0)
    # 邻均值 z1 作为背景值，B @ [a, b] 拟合 x0[k] = -a*z1[k] + b。
    z1 = 0.5 * (x1[:-1] + x1[1:])
    B = np.column_stack((-z1, np.ones(len(z1))))
    a, b = np.linalg.lstsq(B, x0[1:], rcond=None)[0]

    # 先生成包含历史和未来时点的累加序列解析解，再做逆累加。
    k = np.arange(len(x0) + steps)
    if abs(a) < 1e-12:
        x1_hat = x0[0] + b * k
    else:
        x1_hat = (x0[0] - b / a) * np.exp(-a * k) + b / a
    fitted = np.r_[x0[0], np.diff(x1_hat)]
    # 前 len(x0) 项是历史拟合值，最后 steps 项是未来预测值。
    return fitted[-steps:], (a, b), fitted[: len(x0)]


if __name__ == "__main__":
    data = [12, 15, 19, 24, 30]
    forecast, (a, b), fitted = gm11(data, steps=3)
    print("a, b:", a, b)
    print("fitted:", fitted)
    print("forecast:", forecast)
