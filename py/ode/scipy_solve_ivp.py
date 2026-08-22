#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scipy.integrate.solve_ivp 最小可执行模板。

标准形式：
    dy/dt = rhs(t, y, *args)
    y(t0) = y0

比赛时主要替换 rhs、t_span、y0、t_eval 和 args。
如果方程刚性明显，可以把 method 从 RK45 改为 BDF 或 Radau。

验收锚点：本例 Logistic 方程在 y=90 时触发事件，终止值约为 90。
"""

import sys

import numpy as np
from scipy.integrate import solve_ivp


def solve_ode(
    rhs,
    t_span,
    y0,
    *,
    t_eval=None,
    args=(),
    method="RK45",
    rtol=1e-6,
    atol=1e-9,
    max_step=np.inf,
    events=None,
):
    """数值求解一阶或高阶常微分方程组。

    Args:
        rhs: 右端函数，签名为 ``rhs(t, y, *args)``。
        t_span: ``(起始时间, 终止时间)``。
        y0: 初始状态向量。
        t_eval: 可选的输出采样时间点。
        args: 传给 ``rhs`` 和事件函数的额外位置参数。
        method: SciPy 求解器名称，如 ``RK45``、``BDF`` 或 ``Radau``。
        rtol: 相对误差容限。
        atol: 绝对误差容限。
        max_step: 最大积分步长。
        events: 可选事件函数或事件函数序列。

    Returns:
        SciPy 的 ``OdeResult``，包含时间、状态、成功标志和事件结果。

    Raises:
        ValueError: 时间区间或初始状态不合法时抛出。
    """
    t_span = np.asarray(t_span, dtype=float)
    y0 = np.asarray(y0, dtype=float)
    if t_span.shape != (2,) or not np.all(np.isfinite(t_span)):
        raise ValueError("t_span 必须包含两个有限数值")
    if t_span[0] == t_span[1]:
        raise ValueError("t_span 的起止时间不能相同")
    if y0.ndim != 1 or y0.size == 0 or not np.all(np.isfinite(y0)):
        raise ValueError("y0 必须是一维且只包含有限数值")

    # solve_ivp 负责自适应选步；t_eval 只控制返回采样点，不改变积分过程。
    return solve_ivp(
        rhs,
        (t_span[0], t_span[1]),
        y0,
        method=method,
        t_eval=t_eval,
        args=args,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        events=events,
    )


def logistic_rhs(t, y, growth_rate, carrying_capacity):
    """计算 Logistic 方程右端项 ``dy/dt = r*y*(1-y/K)``。"""
    return [growth_rate * y[0] * (1.0 - y[0] / carrying_capacity)]


def reach_target(t, y, growth_rate, carrying_capacity):
    """返回达到环境容纳量 90% 时的事件函数值；过零即触发事件。"""
    return y[0] - 0.9 * carrying_capacity


reach_target.terminal = True
reach_target.direction = 1


def main():
    # ====== 比赛时主要替换下面这部分 ======
    growth_rate = 0.4
    carrying_capacity = 100.0
    t_span = (0.0, 20.0)
    t_eval = np.linspace(t_span[0], t_span[1], 201)
    y0 = [10.0]

    result = solve_ode(
        logistic_rhs,
        t_span,
        y0,
        t_eval=t_eval,
        args=(growth_rate, carrying_capacity),
        method="RK45",
        events=reach_target,
    )

    print("是否成功:", result.success)
    print("求解信息:", result.message)
    print("积分步数:", result.t.size)
    print("最后采样时间:", f"{result.t[-1]:.8f}")
    print("最后采样状态:", np.round(result.y[:, -1], 8))

    event_state = None
    if result.t_events and result.t_events[0].size:
        event_time = result.t_events[0][0]
        event_state = result.y_events[0][0]
        print("事件时间:", f"{event_time:.8f}")
        print("事件状态:", np.round(event_state, 8))

    if not result.success:
        raise RuntimeError(f"ODE 求解失败: {result.message}")

    # 本例的简单验收：替换题目后可删除或改写。
    assert event_state is not None
    assert np.isclose(event_state[0], 0.9 * carrying_capacity, atol=1e-5)
    assert result.t_events[0].size == 1

    if "--csv" in sys.argv:
        columns = np.column_stack((result.t, result.y.T))
        header = "t," + ",".join(
            f"y{i + 1}" for i in range(result.y.shape[0])
        )
        np.savetxt(
            "scipy_solve_ivp_out.csv",
            columns,
            delimiter=",",
            fmt="%.10f",
            header=header,
            comments="",
        )
        print("已保存 -> scipy_solve_ivp_out.csv")


if __name__ == "__main__":
    main()
