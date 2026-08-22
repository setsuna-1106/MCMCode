#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scipy.integrate.quad 一维数值积分最小可执行模板。

计算定积分：
    I = integral_a^b f(x, *args) dx

适用：普通定积分、带参数积分和反常积分（a 或 b 可取 +/-np.inf）。
如果被积函数在已知位置有间断、尖点或剧烈变化，可通过 points 传入分割点。

验收锚点：integral_0^pi sin(x) dx = 2。
"""

import sys

import numpy as np
from scipy.integrate import quad


def integrate_1d(
    func,
    a,
    b,
    *,
    args=(),
    points=None,
    epsabs=1e-9,
    epsrel=1e-9,
    limit=100,
):
    """计算一维定积分。

    Args:
        func: 被积函数，签名为 ``func(x, *args)``。
        a: 积分下限，可使用 ``-np.inf``。
        b: 积分上限，可使用 ``np.inf``。
        args: 传给被积函数的额外位置参数。
        points: 已知的间断点或尖点，用于提示数值积分器分段。
        epsabs: 绝对误差容限。
        epsrel: 相对误差容限。
        limit: 自适应积分允许的最大子区间数。

    Returns:
        ``(value, error)``，分别为积分值和 SciPy 的误差估计。

    Raises:
        ValueError: 参数容限或积分上下限不合法时抛出。
    """
    if np.isnan(a) or np.isnan(b):
        raise ValueError("积分上下限不能是 NaN")
    if epsabs <= 0 or epsrel <= 0:
        raise ValueError("epsabs 和 epsrel 必须为正数")
    if limit < 1:
        raise ValueError("limit 必须是正整数")

    # points 只对有限区间有效；反常积分应通过无穷上下限直接传入。
    value, error = quad(
        func,
        a,
        b,
        args=args,
        points=points,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=limit,
    )
    return value, error


def main():
    # ====== 比赛时主要替换下面这部分 ======
    def integrand(x):
        return np.sin(x)

    a = 0.0
    b = np.pi
    value, error = integrate_1d(integrand, a, b)

    print("积分区间:", f"[{a:.6f}, {b:.6f}]")
    print("积分值:", f"{value:.12f}")
    print("误差估计:", f"{error:.3e}")

    # 本例的简单验收：替换题目后可删除或改写。
    assert np.isclose(value, 2.0, atol=1e-10)

    if "--csv" in sys.argv:
        np.savetxt(
            "scipy_integrate_out.csv",
            np.array([[a, b, value, error]]),
            delimiter=",",
            fmt="%.12f",
            header="a,b,integral,estimated_error",
            comments="",
        )
        print("已保存 -> scipy_integrate_out.csv")


if __name__ == "__main__":
    main()
