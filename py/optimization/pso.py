"""粒子群优化（PSO）最小可执行模板。

适用于连续黑箱目标函数的群体搜索：
    min/max objective(x)    x 为一维决策向量，上下界由 bounds 给定

每个粒子同时受三项力更新速度：惯性（保持原方向）、个体认知
（拉向自己的历史最佳）和社会认知（拉向全群历史最佳）；
速度限幅防止粒子一步飞出搜索域反复震荡。
"""

from __future__ import annotations

import numpy as np


def _as_bounds(bounds) -> tuple[np.ndarray, np.ndarray]:
    """解析并校验每个变量的上下界，返回 (lower, upper) 向量。"""
    bounds = np.asarray(bounds, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2 or bounds.shape[0] == 0:
        raise ValueError("bounds 必须是 (变量数, 2) 的上下界数组")
    if not np.isfinite(bounds).all():
        # 元启发式在有界空间内随机搜索，无界的方向必须由使用者自行截断。
        raise ValueError("bounds 只能包含有限的上下界")
    if np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("每个变量的下界必须小于上界")
    return bounds[:, 0].copy(), bounds[:, 1].copy()


def _evaluate_population(objective, positions, sign) -> np.ndarray:
    """逐个评价粒子并校验返回值，内部统一按最小化处理。"""
    values = np.array(
        [
            np.asarray(objective(np.asarray(x, dtype=float)), dtype=float)
            for x in positions
        ],
        dtype=float,
    )
    if values.ndim != 1 or values.size != positions.shape[0]:
        raise ValueError("objective(x) 必须为每个粒子返回一个标量")
    if not np.isfinite(values).all():
        raise ValueError("objective(x) 必须返回有限标量")
    return values * sign


def particle_swarm(
    objective,
    bounds,
    *,
    n_particles: int = 30,
    iterations: int = 100,
    inertia: float = 0.7,
    cognitive: float = 1.5,
    social: float = 1.5,
    maximize: bool = False,
    random_state: int = 42,
) -> dict[str, object]:
    """用粒子群算法搜索目标函数的最优解。

    Args:
        objective: 目标函数 ``objective(x) -> 标量``。
        bounds: 每个变量的 ``(下界, 上界)``。
        n_particles: 粒子数量，每代评价这么多个候选解。
        iterations: 迭代轮数。
        inertia: 惯性权重，越大越延续原速度方向、全局探索越强。
        cognitive: 个体认知系数，拉向粒子自身历史最佳的力度。
        social: 社会认知系数，拉向全群历史最佳的力度。
        maximize: 是否最大化目标（默认最小化）。
        random_state: 随机种子，保证结果可复现。

    Returns:
        含 ``x_best``、``fun_best``（原始目标值）、``history``
        （每轮历史最优，单调不增）和 ``n_evaluations`` 的字典。

    Raises:
        ValueError: 参数不合法或目标函数返回非有限标量时抛出。
    """
    lower, upper = _as_bounds(bounds)
    if n_particles < 3:
        raise ValueError("n_particles 至少为 3")
    if iterations < 1:
        raise ValueError("iterations 必须是正整数")
    if inertia < 0.0 or cognitive < 0.0 or social < 0.0:
        raise ValueError("inertia、cognitive 和 social 必须非负")

    rng = np.random.default_rng(random_state)
    sign = -1.0 if maximize else 1.0
    spans = upper - lower
    # 速度限幅取各维范围的 20%：过大会来回穿越搜索域，过小收敛缓慢。
    v_max = 0.2 * spans

    # 初始位置均匀撒点，初始速度在限幅内随机取值。
    positions = lower + rng.random((n_particles, lower.size)) * spans
    velocities = (rng.random((n_particles, lower.size)) * 2.0 - 1.0) * v_max
    values = _evaluate_population(objective, positions, sign)
    n_evaluations = n_particles

    # pbest 是每个粒子的历史最佳，gbest 是全群历史最佳。
    pbest = positions.copy()
    pbest_values = values.copy()
    best_index = int(np.argmin(pbest_values))
    gbest = pbest[best_index].copy()
    gbest_value = pbest_values[best_index]
    history = np.empty(iterations)

    for t in range(iterations):
        r1 = rng.random(positions.shape)
        r2 = rng.random(positions.shape)
        velocities = (
            inertia * velocities
            + cognitive * r1 * (pbest - positions)
            + social * r2 * (gbest - positions)
        )
        np.clip(velocities, -v_max, v_max, out=velocities)
        positions = positions + velocities
        np.clip(positions, lower, upper, out=positions)
        values = _evaluate_population(objective, positions, sign)
        n_evaluations += n_particles

        improved = values < pbest_values
        pbest[improved] = positions[improved]
        pbest_values[improved] = values[improved]
        best_index = int(np.argmin(pbest_values))
        if pbest_values[best_index] < gbest_value:
            gbest = pbest[best_index].copy()
            gbest_value = pbest_values[best_index]
        history[t] = gbest_value * sign  # 还原为原始目标值

    return {
        "x_best": gbest,
        "fun_best": gbest_value * sign,
        "history": history,
        "n_evaluations": n_evaluations,
    }


def main():
    # ====== 比赛时主要替换下面这部分 ======
    def rastrigin(x):
        # Rastrigin 函数有大量局部极小值，全局最优在原点、最优值为 0，
        # 用于演示群体搜索跳出局部最优的能力。
        return 10.0 * x.size + float(
            np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x))
        )

    bounds = [(-5.12, 5.12)] * 2
    result = particle_swarm(rastrigin, bounds,
                            n_particles=30, iterations=100)

    print("最优变量:", np.round(result["x_best"], 6))
    print("最优目标值:", f"{result['fun_best']:.8f}")
    print("目标评价次数:", result["n_evaluations"])
    print("收敛过程(每 20 轮):", np.round(result["history"][::20], 6))

    # 本例的简单验收：替换题目后可删除或改写。
    assert np.allclose(result["x_best"], [0.0, 0.0], atol=0.2)
    assert result["fun_best"] < 1e-3
    assert result["history"].shape == (100,)
    # gbest 只会改进不会退化，收敛曲线必须单调不增。
    assert np.all(np.diff(result["history"]) <= 1e-12)


if __name__ == "__main__":
    main()
