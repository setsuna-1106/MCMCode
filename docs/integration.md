# 数值积分（integration）

计算无法解析求积的定积分 $I = \int_a^b f(x)\, dx$，并给出误差估计。对应模板 `py/integration/scipy_integrate.py`，基于 `scipy.integrate.quad`。

## 原理

`quad` 使用**自适应 Gauss–Kronrod 求积**：在整个区间上同时用高阶（Kronrod，21 点）与低阶（Gauss，15 点）公式计算积分，两者的差作为**误差估计**；若误差超过容限（`epsabs`、`epsrel`），就把区间对半细分并在子区间上重复该过程，直至收敛或达到最大子区间数 `limit`。

这种「先估计误差、再决定细分」的策略使它对平滑函数非常高效，通常几十次函数求值即可达到默认的 $10^{-9}$ 精度。

反常积分（上下限取 $\pm\infty$，或区间端点函数奇异）通过变量变换处理：例如上限为 $+\infty$ 时内部做 $x = a + \frac{t}{1-t}$ 的代换，把无限区间映射到 $[0, 1)$。

## 用法

```python
import numpy as np
from py.integration.scipy_integrate import integrate_1d

def integrand(x, rate):
    return np.exp(-rate * x)

value, error = integrate_1d(integrand, 0.0, np.inf, args=(0.3,))
print("积分值:", value)
print("误差估计:", error)
```

- `func`：被积函数，签名 `func(x, *args)`，`args` 传入额外参数。
- `a`、`b`：积分上下限，可取 `np.inf` / `-np.inf`。
- `points`：函数已知间断点、尖点或剧烈变化位置（如 `[1.0, 2.0]`），提示积分器分段处理；仅对有限区间有效。
- `epsabs` / `epsrel`：绝对 / 相对误差容限（默认 $10^{-9}$）；`limit`：最大子区间数（默认 100）。
- 返回 `(value, error)`：积分值与 SciPy 的误差估计。
- 直接运行模板（含 `--csv` 导出）：

```bash
.venv/bin/python py/integration/scipy_integrate.py --csv
```

## 注意

- 结果应结合 `error` 与题目物理量级检查；误差估计偏大说明函数可能震荡或奇异，优先用 `points` 分段，其次放宽 `limit`。
- 强震荡被积函数（如 $\sin(100x)$）可能需要增大 `limit` 或手工分段积分再求和。
- 二维及以上积分应使用 `scipy.integrate.dblquad` 等多元接口，本模板只处理一维。
