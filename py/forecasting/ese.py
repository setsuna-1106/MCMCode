"""一次指数平滑模板。"""

import numpy as np


def exponential_smoothing(y, alpha=0.3, steps=3):
    """计算一次指数平滑及未来预测。

    Args:
        y: 按时间排列的一维序列。
        alpha: 最新观测值的平滑权重。
        steps: 未来预测期数。

    Returns:
        ``(level, forecast)``，分别为历史水平项和未来预测。
    """
    # 一次指数平滑只建模水平项，不包含趋势或季节项。
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or y.size < 2 or not np.all(np.isfinite(y)):
        raise ValueError("y 必须是一维且至少包含 2 个有限数值")
    if not 0 < alpha <= 1:
        raise ValueError("alpha 必须满足 0 < alpha <= 1")
    if not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError("steps 必须是正整数")

    # l_t = alpha*y_t + (1-alpha)*l_{t-1}
    # 用第一期观测初始化水平项，之后递推更新每一期的平滑值。
    level = np.empty(y.size)
    level[0] = y[0]
    for t in range(1, y.size):
        level[t] = alpha * y[t] + (1 - alpha) * level[t - 1]

    # 无趋势项时，未来各期都使用最后一期水平项。
    forecast = np.full(steps, level[-1])
    return level, forecast


def main():
    # 替换为题目数据：y 为按时间排列的一维序列。
    y = np.array([102, 105, 107, 111, 115, 114, 119, 123], dtype=float)
    alpha = 0.3  # 越大越重视最新观测值
    steps = 3

    level, forecast = exponential_smoothing(y, alpha, steps)

    print("平滑值:", level)
    print("未来预测:", forecast)


if __name__ == "__main__":
    main()
