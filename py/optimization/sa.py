"""模拟退火（SA）最小可执行模板。

适用于黑箱、不可导、强非凸目标函数的单点随机搜索：
    min/max objective(x)    x 为一维决策向量，上下界由 bounds 给定

核心机制是 Metropolis 准则：更优解必接受，更差解按 exp(-delta/T)
的概率接受；温度 T 从高到低几何降温，早期大胆探索、后期收敛细化。
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


def simulated_annealing(
    objective,
    bounds,
    *,
    x0=None,
    iterations: int = 2000,
    initial_temp: float = 100.0,
    cooling_alpha: float = 0.995,
    step_fraction: float = 0.15,
    maximize: bool = False,
    random_state: int = 42,
) -> dict[str, object]:
    """用模拟退火搜索目标函数的最优解。

    Args:
        objective: 目标函数 ``objective(x) -> 标量``，x 为一维决策向量。
        bounds: 每个变量的 ``(下界, 上界)``，搜索空间必须有界。
        x0: 可选初始解；为空时在界内均匀随机采样。
        iterations: 总迭代步数，每步评价一次候选解。
        initial_temp: 初始温度，越高早期越敢接受劣解。
        cooling_alpha: 几何降温系数，T_t = initial_temp * alpha**t。
        step_fraction: 扰动幅度，按各变量取值范围的倍数计算。
        maximize: 是否最大化目标（默认最小化）。
        random_state: 随机种子，保证结果可复现。

    Returns:
        含 ``x_best``、``fun_best``（原始目标值）、``history``
        （每步历史最优）和 ``n_evaluations`` 的字典。

    Raises:
        ValueError: 参数不合法或目标函数返回非有限标量时抛出。
    """
    lower, upper = _as_bounds(bounds)
    d = lower.size
    if iterations < 1:
        raise ValueError("iterations 必须是正整数")
    if initial_temp <= 0:
        raise ValueError("initial_temp 必须大于 0")
    if not 0.0 < cooling_alpha < 1.0:
        raise ValueError("cooling_alpha 必须落在 (0, 1) 区间")
    if not 0.0 < step_fraction <= 1.0:
        raise ValueError("step_fraction 必须落在 (0, 1] 区间")
    if x0 is not None:
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        if x0.size != d or not np.isfinite(x0).all():
            raise ValueError("x0 必须与变量数一致且只含有限数值")
        if np.any(x0 < lower) or np.any(x0 > upper):
            raise ValueError("x0 必须落在 bounds 内")
    rng = np.random.default_rng(random_state)
    sign = -1.0 if maximize else 1.0
    spans = upper - lower

    # 当前解与历史最优分开维护：接受劣解的是搜索路径，报告的是历史最优。
    x = x0.copy() if x0 is not None else lower + rng.random(d) * spans
    f = _evaluate(objective, x, sign)
    x_best, f_best = x.copy(), f
    history = np.empty(iterations)
    n_evaluations = 1

    for t in range(iterations):
        temp = initial_temp * cooling_alpha**t
        # 候选解 = 当前解 + 高斯扰动（幅度按变量范围缩放），并裁剪回界内。
        candidate = x + rng.normal(0.0, step_fraction, d) * spans
        np.clip(candidate, lower, upper, out=candidate)
        f_new = _evaluate(objective, candidate, sign)
        n_evaluations += 1

        delta = f_new - f
        # Metropolis 准则：更优必接受；更差按温度概率接受，用于跳出局部最优。
        if delta < 0.0 or (
            temp > 0.0 and rng.random() < np.exp(-delta / temp)
        ):
            x, f = candidate, f_new
            if f < f_best:
                x_best, f_best = x.copy(), f
        history[t] = f_best * sign  # 还原为原始目标值

    return {
        "x_best": x_best,
        "fun_best": f_best * sign,
        "history": history,
        "n_evaluations": n_evaluations,
    }


def main():
    # ====== 比赛时主要替换下面这部分 ======
    def objective(x):
        # 替换为题目模型：可以是仿真、查表或任意不可导的黑箱函数。
        return (x[0] - 3.0) ** 2 + (x[1] + 1.0) ** 2

    bounds = [(-10.0, 10.0), (-10.0, 10.0)]
    result = simulated_annealing(objective, bounds, iterations=2000)

    print("最优变量:", np.round(result["x_best"], 6))
    print("最优目标值:", f"{result['fun_best']:.8f}")
    print("目标评价次数:", result["n_evaluations"])
    print("收敛过程(每 500 步):", np.round(result["history"][::500], 6))

    # 本例的简单验收：替换题目后可删除或改写。
    assert np.allclose(result["x_best"], [3.0, -1.0], atol=0.1)
    assert result["fun_best"] < 1e-2
    assert result["history"].shape == (2000,)
    assert result["history"][-1] <= result["history"][0]


if __name__ == "__main__":
    main()
