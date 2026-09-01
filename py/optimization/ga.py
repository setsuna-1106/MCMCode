"""遗传算法（GA）最小可执行模板。

实数编码的种群进化搜索，适用于黑箱、不可导或离散组合问题：
    min/max objective(x)    x 为一维决策向量，上下界由 bounds 给定

进化循环：适应度评价 -> 锦标赛选择 -> 线性组合交叉 -> 高斯变异
-> 精英保留。组合问题的使用者可把编码与交叉、变异操作替换为
排列算子，框架其余部分不变。
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


def _evaluate(objective, x, sign: float) -> float:
    """评价一个候选解并校验返回值，内部统一按最小化处理。"""
    value = np.asarray(objective(np.asarray(x, dtype=float)), dtype=float)
    if value.ndim != 0 or not np.isfinite(value):
        raise ValueError("objective(x) 必须返回一个有限标量")
    return float(value) * sign


def _tournament(population, fitness, size: int, rng) -> np.ndarray:
    """锦标赛选择：随机抽 size 个个体，返回其中适应度最优者。"""
    contenders = rng.choice(population.shape[0], size=size, replace=False)
    winner = contenders[np.argmin(fitness[contenders])]
    return population[winner]


def genetic_algorithm(
    objective,
    bounds,
    *,
    population_size: int = 50,
    generations: int = 100,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.1,
    mutation_scale: float = 0.1,
    tournament_size: int = 3,
    elite_size: int = 2,
    maximize: bool = False,
    random_state: int = 42,
) -> dict[str, object]:
    """用实数编码遗传算法搜索目标函数的最优解。

    Args:
        objective: 目标函数 ``objective(x) -> 标量``。
        bounds: 每个变量的 ``(下界, 上界)``。
        population_size: 种群规模，每代评价这么多个个体。
        generations: 进化代数。
        crossover_rate: 两个亲本发生交叉的概率。
        mutation_rate: 每个基因发生变异的概率。
        mutation_scale: 变异强度，按各变量取值范围的倍数计算。
        tournament_size: 锦标赛选择的抽样人数。
        elite_size: 直接进入下一代的精英数量。
        maximize: 是否最大化目标（默认最小化）。
        random_state: 随机种子，保证结果可复现。

    Returns:
        含 ``x_best``、``fun_best``（原始目标值）、``history``
        （每代历史最优）和 ``n_evaluations`` 的字典。

    Raises:
        ValueError: 参数不合法或目标函数返回非有限标量时抛出。
    """
    lower, upper = _as_bounds(bounds)
    d = lower.size
    if population_size < 4:
        raise ValueError("population_size 至少为 4")
    if generations < 1:
        raise ValueError("generations 必须是正整数")
    if not 0.0 <= crossover_rate <= 1.0:
        raise ValueError("crossover_rate 必须落在 [0, 1] 区间")
    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError("mutation_rate 必须落在 [0, 1] 区间")
    if not 0.0 < mutation_scale <= 1.0:
        raise ValueError("mutation_scale 必须落在 (0, 1] 区间")
    if not 1 <= tournament_size <= population_size:
        raise ValueError("tournament_size 必须落在 1 到种群规模之间")
    if not 0 <= elite_size < population_size:
        raise ValueError("elite_size 必须落在 0 到种群规模之间")

    rng = np.random.default_rng(random_state)
    sign = -1.0 if maximize else 1.0
    spans = upper - lower

    # 初始种群在界内均匀撒点，保证开局覆盖整个搜索空间。
    population = lower + rng.random((population_size, d)) * spans
    fitness = np.array(
        [_evaluate(objective, individual, sign) for individual in population]
    )
    n_evaluations = population_size
    history = np.empty(generations)

    for generation in range(generations):
        # 精英保留：当代最优个体不参与交叉变异，直接复制到下一代。
        elite_order = np.argsort(fitness)[:elite_size]
        offspring = [population[i].copy() for i in elite_order]

        while len(offspring) < population_size:
            parent_a = _tournament(population, fitness, tournament_size, rng)
            parent_b = _tournament(population, fitness, tournament_size, rng)
            if rng.random() < crossover_rate:
                # 线性组合交叉：系数取 (-0.25, 1.25) 略超出父母区间，
                # 避免后代范围逐代收缩、探索性丧失。
                alpha = rng.uniform(-0.25, 1.25)
                child = parent_a + alpha * (parent_b - parent_a)
            else:
                child = parent_a.copy()
            # 高斯变异：少量基因随机扰动，维持跳出局部最优的多样性。
            mutation_mask = rng.random(d) < mutation_rate
            if mutation_mask.any():
                child[mutation_mask] += rng.normal(
                    0.0, mutation_scale, mutation_mask.sum()
                ) * spans[mutation_mask]
            offspring.append(np.clip(child, lower, upper))

        population = np.array(offspring)
        fitness = np.array(
            [_evaluate(objective, individual, sign) for individual in population]
        )
        n_evaluations += population_size
        history[generation] = fitness.min() * sign  # 还原为原始目标值

    best_index = int(np.argmin(fitness))
    return {
        "x_best": population[best_index],
        "fun_best": fitness[best_index] * sign,
        "history": history,
        "n_evaluations": n_evaluations,
    }


def main():
    # ====== 比赛时主要替换下面这部分 ======
    target = np.array([2.0, -3.0, 5.0, 1.0])  # 真实最优点仅用于验收

    def objective(x):
        return float(np.sum((x - target) ** 2))

    bounds = [(-10.0, 10.0)] * 4
    result = genetic_algorithm(objective, bounds,
                               population_size=50, generations=100)

    print("最优变量:", np.round(result["x_best"], 6))
    print("最优目标值:", f"{result['fun_best']:.8f}")
    print("目标评价次数:", result["n_evaluations"])
    print("收敛过程(每 25 代):", np.round(result["history"][::25], 6))

    # 本例的简单验收：替换题目后可删除或改写。
    assert np.allclose(result["x_best"], target, atol=0.2)
    assert result["history"].shape == (100,)
    assert result["history"][-1] <= result["history"][0]


if __name__ == "__main__":
    main()
