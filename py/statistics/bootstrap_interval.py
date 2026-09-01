"""Bootstrap 置信区间模板。

对任意统计量（均值、中位数、自定指标）做非参数置信区间：
有放回地重采样原始数据 ``n_resamples`` 次，每次计算统计量，
再取分位数区间。不假设分布形状，适合难以解析求方差的指标。
"""

from __future__ import annotations

import numpy as np


def _as_1d(data, name: str) -> np.ndarray:
    """统一为一维有限数值数组。"""
    array = np.asarray(data, dtype=float).reshape(-1)
    if array.size < 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} 必须是至少 2 个有限数值的一维数组")
    return array


def _evaluate_statistic(statistic, sample) -> float:
    """计算统计量并校验返回值是有限标量。"""
    value = np.asarray(statistic(sample), dtype=float)
    if value.ndim != 0 or not np.isfinite(value):
        raise ValueError("statistic(sample) 必须返回一个有限标量")
    return float(value)


def bootstrap_confidence_interval(
    data,
    statistic=np.mean,
    *,
    n_resamples: int = 5000,
    confidence: float = 0.95,
    random_state: int = 42,
):
    """用非参数 Bootstrap 计算统计量的置信区间（百分位法）。

    Args:
        data: 一维数值观测。
        statistic: 统计量函数，如 ``np.mean``、``np.median`` 或自定指标。
        n_resamples: 重采样次数，越多区间越稳定。
        confidence: 置信水平，如 ``0.95``。
        random_state: 随机种子，保证结果可复现。

    Returns:
        包含 ``estimate``（原始样本统计量）、``low``、``high``
        （区间端点）、``confidence`` 和 ``n_resamples`` 的字典。

    Raises:
        ValueError: 参数不合法或统计量返回非有限标量时抛出。
    """
    array = _as_1d(data, "data")
    if n_resamples < 100:
        raise ValueError("n_resamples 至少为 100，否则区间不稳定")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence 必须落在 (0, 1) 区间")

    rng = np.random.default_rng(random_state)
    # 每次重采样与原样本等长、有放回，等价于从经验分布抽样。
    estimates = np.fromiter(
        (
            _evaluate_statistic(
                statistic, rng.choice(array, size=array.size, replace=True)
            )
            for _ in range(n_resamples)
        ),
        dtype=float,
        count=n_resamples,
    )

    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, [alpha, 1.0 - alpha])
    return {
        "estimate": _evaluate_statistic(statistic, array),
        "low": float(low),
        "high": float(high),
        "confidence": confidence,
        "n_resamples": n_resamples,
    }


def main():
    # ====== 比赛时主要替换下面这部分 ======
    rng = np.random.default_rng(42)

    # 指数分布样本：均值、中位数都应给出覆盖真值的区间。
    data = rng.exponential(1.0, 300)  # 均值真值为 1.0，中位数真值为 ln2
    mean_ci = bootstrap_confidence_interval(data, statistic=np.mean)
    median_ci = bootstrap_confidence_interval(data, statistic=np.median)
    print("Mean CI:", mean_ci)
    print("Median CI:", median_ci)

    # 本例的简单验收：替换题目后可删除或改写。
    assert mean_ci["low"] < 1.0 < mean_ci["high"]
    assert median_ci["low"] < np.log(2.0) < median_ci["high"]
    assert mean_ci["low"] < mean_ci["estimate"] < mean_ci["high"]


if __name__ == "__main__":
    main()
