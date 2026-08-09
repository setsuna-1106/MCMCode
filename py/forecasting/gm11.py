import numpy as np


def gm11(data, steps=1):
    """Return (forecast, (a, b), fitted) for a GM(1,1) model."""
    x0 = np.asarray(data, dtype=float).reshape(-1)
    if len(x0) < 4:
        raise ValueError("GM(1,1) 至少需要 4 个数据")
    if not np.isfinite(x0).all() or (x0 < 0).any():
        raise ValueError("数据必须是非负有限数")
    if steps < 1:
        raise ValueError("steps 必须大于 0")

    x1 = np.cumsum(x0)
    z1 = 0.5 * (x1[:-1] + x1[1:])
    B = np.column_stack((-z1, np.ones(len(z1))))
    a, b = np.linalg.lstsq(B, x0[1:], rcond=None)[0]

    k = np.arange(len(x0) + steps)
    if abs(a) < 1e-12:
        x1_hat = x0[0] + b * k
    else:
        x1_hat = (x0[0] - b / a) * np.exp(-a * k) + b / a
    fitted = np.r_[x0[0], np.diff(x1_hat)]
    return fitted[-steps:], (a, b), fitted[: len(x0)]


if __name__ == "__main__":
    data = [12, 15, 19, 24, 30]
    forecast, (a, b), fitted = gm11(data, steps=3)
    print("a, b:", a, b)
    print("fitted:", fitted)
    print("forecast:", forecast)
