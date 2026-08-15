"""灰色关联度分析（GRA）最小可执行模板。

输入约定：
    reference: 参考序列，形状为 (n_indicators,)
    comparison: 比较序列，形状为 (n_objects, n_indicators)
    direction: 1 表示收益型指标，-1 表示成本型指标

标准计算：
    gamma_ij = (delta_min + rho * delta_max)
               / (delta_ij + rho * delta_max)
    grade_i = sum(weights_j * gamma_ij)
"""

import sys

import numpy as np


def _validate_input(reference, comparison, direction, weights, rho):
    reference = np.asarray(reference, dtype=float).reshape(-1)
    comparison = np.asarray(comparison, dtype=float)
    if comparison.ndim == 1:
        comparison = comparison.reshape(1, -1)
    if (
        reference.size == 0
        or comparison.ndim != 2
        or comparison.shape[0] == 0
        or comparison.shape[1] != reference.size
        or not np.isfinite(reference).all()
        or not np.isfinite(comparison).all()
    ):
        raise ValueError("reference 和 comparison 的维度或数值无效")
    if not 0 < rho <= 1:
        raise ValueError("rho 必须满足 0 < rho <= 1")

    if direction is None:
        direction = np.ones(reference.size, dtype=int)
    direction = np.asarray(direction, dtype=int).reshape(-1)
    if direction.size != reference.size or not np.isin(direction, [-1, 1]).all():
        raise ValueError("direction 必须由与指标数相同的 1 和 -1 组成")

    if weights is None:
        weights = np.ones(reference.size, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if weights.size != reference.size or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("weights 必须是非负有限数值，且长度与指标数一致")
    if weights.sum() <= 0:
        raise ValueError("weights 的总和必须大于 0")
    weights = weights / weights.sum()
    return reference, comparison, direction, weights


def _minmax_normalize(data, direction):
    """按指标列做极差标准化，支持收益型和成本型指标。"""
    normalized = np.empty_like(data, dtype=float)
    for j, kind in enumerate(direction):
        column = data[:, j]
        minimum = column.min()
        maximum = column.max()
        span = maximum - minimum
        if span == 0:
            normalized[:, j] = 1.0
        elif kind == 1:
            normalized[:, j] = (column - minimum) / span
        else:
            normalized[:, j] = (maximum - column) / span
    return normalized


def grey_relation(
    reference,
    comparison,
    *,
    direction=None,
    weights=None,
    rho=0.5,
    normalize=True,
):
    """计算灰色关联系数、关联度和排序结果。"""
    reference, comparison, direction, weights = _validate_input(
        reference, comparison, direction, weights, rho
    )
    all_data = np.vstack((reference, comparison))
    if normalize:
        all_data = _minmax_normalize(all_data, direction)
    normalized_reference = all_data[0]
    normalized_comparison = all_data[1:]

    difference = np.abs(normalized_comparison - normalized_reference)
    delta_min = difference.min()
    delta_max = difference.max()
    if delta_max == 0:
        coefficients = np.ones_like(difference)
    else:
        coefficients = (delta_min + rho * delta_max) / (
            difference + rho * delta_max
        )
    grades = coefficients @ weights

    return {
        "normalized_reference": normalized_reference,
        "normalized_comparison": normalized_comparison,
        "coefficients": coefficients,
        "grades": grades,
        "order": np.argsort(-grades),
        "weights": weights,
        "delta_min": delta_min,
        "delta_max": delta_max,
    }


def main():
    # ====== 比赛时主要替换下面这部分 ======
    # 三个方案，四个指标；第三列是成本型指标。
    reference = np.array([1.0, 1.0, 0.0, 1.0])
    comparison = np.array([
        [0.9, 0.80, 0.4, 0.9],
        [0.8, 0.90, 0.3, 0.8],
        [0.7, 0.95, 0.5, 0.7],
    ])
    direction = [1, 1, -1, 1]
    weights = [0.30, 0.30, 0.20, 0.20]

    result = grey_relation(
        reference,
        comparison,
        direction=direction,
        weights=weights,
        rho=0.5,
        normalize=True,
    )
    print("灰色关联度:", np.round(result["grades"], 8))
    print("方案排序（从优到劣）:", result["order"] + 1)
    print("关联系数:\n", np.round(result["coefficients"], 8))

    assert result["grades"].shape == (comparison.shape[0],)
    assert np.all((result["grades"] >= 0) & (result["grades"] <= 1))
    assert sorted(result["order"].tolist()) == list(range(comparison.shape[0]))

    if "--csv" in sys.argv:
        np.savetxt(
            "gra_out.csv",
            np.column_stack((np.arange(1, result["grades"].size + 1), result["grades"])),
            delimiter=",",
            fmt=["%d", "%.10f"],
            header="方案编号,灰色关联度",
            comments="",
        )
        print("已保存 -> gra_out.csv")


if __name__ == "__main__":
    main()
