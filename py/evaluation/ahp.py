#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AHP 层次分析法和一致性检验最小可执行模板。

输入是正互反判断矩阵：
    a[i, j] > 0
    a[i, i] = 1
    a[i, j] * a[j, i] = 1

结果使用最大特征值法计算权重：
    CI = (lambda_max - n) / (n - 1)
    CR = CI / RI

通常 CR < 0.1 时认为判断矩阵通过一致性检验。
"""

import sys

import numpy as np


RANDOM_INDEX = {
    1: 0.00,
    2: 0.00,
    3: 0.58,
    4: 0.90,
    5: 1.12,
    6: 1.24,
    7: 1.32,
    8: 1.41,
    9: 1.45,
    10: 1.49,
    11: 1.51,
    12: 1.48,
    13: 1.56,
    14: 1.57,
    15: 1.59,
}


def _validate_matrix(matrix, name="matrix", tolerance=1e-8):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} 必须是非空方阵")
    if matrix.shape[0] == 0 or not np.isfinite(matrix).all() or np.any(matrix <= 0):
        raise ValueError(f"{name} 必须只包含正的有限数值")
    if not np.allclose(np.diag(matrix), 1.0, atol=tolerance, rtol=0):
        raise ValueError(f"{name} 的对角线必须全为 1")
    if not np.allclose(matrix * matrix.T, 1.0, atol=tolerance, rtol=0):
        raise ValueError(f"{name} 必须满足正互反关系 a[i,j] * a[j,i] = 1")
    if matrix.shape[0] not in RANDOM_INDEX:
        raise ValueError("当前 RI 表支持 1 到 15 阶判断矩阵")
    return matrix


def ahp_weights(matrix, consistency_threshold=0.1, require_consistent=False):
    """计算单个判断矩阵的权重、CI、CR 和一致性结论。"""
    matrix = _validate_matrix(matrix)
    n = matrix.shape[0]
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    index = int(np.argmax(eigenvalues.real))
    if abs(eigenvalues[index].imag) > 1e-7:
        raise ValueError("最大特征值存在不可忽略的虚部")

    lambda_max = float(eigenvalues[index].real)
    weights = eigenvectors[:, index].real
    if weights.sum() < 0:
        weights = -weights
    if np.any(weights <= 0):
        raise ValueError("无法得到全为正的特征向量")
    weights = weights / weights.sum()

    ci = 0.0 if n <= 2 else max(0.0, (lambda_max - n) / (n - 1))
    ri = RANDOM_INDEX[n]
    cr = 0.0 if ri == 0 else ci / ri
    consistent = cr <= consistency_threshold
    if require_consistent and not consistent:
        raise ValueError(f"判断矩阵未通过一致性检验: CR={cr:.6f}")

    return {
        "weights": weights,
        "lambda_max": lambda_max,
        "CI": ci,
        "RI": ri,
        "CR": cr,
        "consistent": consistent,
    }


def solve_ahp(
    criteria_matrix,
    alternative_matrices,
    consistency_threshold=0.1,
    require_consistent=False,
):
    """完成准则层和方案层 AHP 总排序。"""
    criteria_result = ahp_weights(
        criteria_matrix,
        consistency_threshold=consistency_threshold,
        require_consistent=require_consistent,
    )
    matrices = list(alternative_matrices)
    n_criteria = len(criteria_result["weights"])
    if len(matrices) != n_criteria:
        raise ValueError("alternative_matrices 的数量必须等于准则数量")

    alternative_results = [
        ahp_weights(
            matrix,
            consistency_threshold=consistency_threshold,
            require_consistent=require_consistent,
        )
        for matrix in matrices
    ]
    n_alternatives = len(alternative_results[0]["weights"])
    if any(result["weights"].size != n_alternatives for result in alternative_results):
        raise ValueError("所有方案判断矩阵的阶数必须相同")

    local_weights = np.column_stack(
        [result["weights"] for result in alternative_results]
    )
    global_weights = local_weights @ criteria_result["weights"]
    all_consistent = criteria_result["consistent"] and all(
        result["consistent"] for result in alternative_results
    )

    return {
        "criteria": criteria_result,
        "alternatives": alternative_results,
        "local_weights": local_weights,
        "global_weights": global_weights,
        "best_index": int(np.argmax(global_weights)),
        "all_consistent": all_consistent,
    }


def main():
    # ====== 比赛时主要替换下面这部分 ======
    # 三个准则：成本、质量、服务；三个方案：A、B、C。
    criteria_matrix = np.array([
        [1.0, 2.0, 4.0],
        [1.0 / 2.0, 1.0, 2.0],
        [1.0 / 4.0, 1.0 / 2.0, 1.0],
    ])
    alternative_matrices = [
        np.array([
            [1.0, 2.0, 6.0],
            [1.0 / 2.0, 1.0, 3.0],
            [1.0 / 6.0, 1.0 / 3.0, 1.0],
        ]),
        np.array([
            [1.0, 0.4, 2.0 / 3.0],
            [2.5, 1.0, 5.0 / 3.0],
            [1.5, 0.6, 1.0],
        ]),
        np.array([
            [1.0, 1.0, 0.5],
            [1.0, 1.0, 0.5],
            [2.0, 2.0, 1.0],
        ]),
    ]

    result = solve_ahp(
        criteria_matrix,
        alternative_matrices,
        consistency_threshold=0.1,
        require_consistent=True,
    )
    print("准则层权重:", np.round(result["criteria"]["weights"], 8))
    print("准则层 CR:", f"{result['criteria']['CR']:.8f}")
    for i, alternative in enumerate(result["alternatives"], start=1):
        print(f"方案层 {i} 的局部权重:", np.round(alternative["weights"], 8))
        print(f"方案层 {i} 的 CR:", f"{alternative['CR']:.8f}")
    print("方案总权重:", np.round(result["global_weights"], 8))
    print("最优方案编号:", result["best_index"] + 1)
    print("是否全部通过一致性检验:", result["all_consistent"])

    assert result["all_consistent"]
    assert np.isclose(result["global_weights"].sum(), 1.0)
    assert result["best_index"] == 0

    if "--csv" in sys.argv:
        np.savetxt(
            "ahp_global_weights.csv",
            np.column_stack((np.arange(1, result["global_weights"].size + 1), result["global_weights"])),
            delimiter=",",
            fmt=["%d", "%.10f"],
            header="方案编号,总权重",
            comments="",
        )
        print("已保存 -> ahp_global_weights.csv")


if __name__ == "__main__":
    main()
