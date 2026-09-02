# 常微分方程数值求解（ode）

求解常微分方程初值问题。对应模板 `py/ode/scipy_solve_ivp.py`，基于 `scipy.integrate.solve_ivp`。

## 原理

模板求解一阶常微分方程组的**初值问题**：

$$\frac{d\mathbf{y}}{dt} = \mathbf{f}(t, \mathbf{y}), \qquad \mathbf{y}(t_0) = \mathbf{y}_0$$

高阶方程先降维：如二阶方程 $y'' = g(t, y, y')$ 令 $y_1 = y,\ y_2 = y'$，化为 $\mathbf{y}' = (y_2,\ g(t, y_1, y_2))$ 的方程组。

默认方法 `RK45` 是 **Dormand–Prince 自适应龙格库塔**：每步同时算出 5 阶和 4 阶两个解，用其差估计局部截断误差，据此自动放大或缩小步长，使误差控制在 `rtol`/`atol` 之下。显式方法对**刚性**方程（快慢过程耦合，如大系数衰减 + 缓慢增长）步长会被迫极小，此时应换隐式方法 `BDF` 或 `Radau`。

**事件检测**：传入的事件函数 $g(t, \mathbf{y})$ 由正变负（过零）时触发记录；设置 `terminal = True` 可在该处终止积分，常用于「达到阈值 / 碰撞 / 耗尽」时刻的精确捕获。

## 优缺点与使用场景

- **优点**：自适应步长把误差控制在容限内，无需手工调步长；事件检测可精确捕获「何时达到某状态」；接口统一，遇到刚性问题时换 `method="BDF"` 即可。
- **缺点**：`RK45` 等显式方法解刚性方程效率急剧下降；`t_eval` 只控制采样、不影响精度，误以为加密输出点能提高精度是常见错误；只处理初值问题（边值问题用 `solve_bvp`）。
- **使用场景**：动力学系统、传染病（SIR）、种群增长等初值问题仿真；需要「污染物何时超标」「库存何时耗尽」这类触发时刻的分析。

## 用法

```python
import numpy as np
from py.ode.scipy_solve_ivp import solve_ode

def rhs(t, y, rate):
    return [-rate * y[0]]

result = solve_ode(
    rhs,
    t_span=(0, 10),
    y0=[1.0],
    t_eval=np.linspace(0, 10, 101),
    args=(0.3,),
)
print(result.t)        # 输出时间点
print(result.y)        # 形状 (状态数, 时间点数) 的状态矩阵
print(result.success)  # 是否成功积分到终止时间
```

- `rhs`：右端函数，返回与 `y0` 等长的导数列表；参数经 `args` 传入。
- `t_span`：`(起始时间, 终止时间)`；`y0`：初始状态向量。
- `t_eval`：只控制**输出哪些时刻**的采样，不影响内部积分步长与精度。
- `events`：事件函数（需与 `rhs` 同签名），设置 `event.terminal = True` 与 `event.direction = 1/-1` 控制终止与过零方向；触发时刻和状态在 `result.t_events` / `result.y_events`。
- 返回 `OdeResult`：`.t`、`.y`、`.success`、`.message`、`.t_events` 等字段。
- 直接运行模板（Logistic 增长 + 事件检测示例，含 `--csv` 导出）：

```bash
.venv/bin/python py/ode/scipy_solve_ivp.py --csv
```

## 注意

- 误差控制靠 `rtol`/`atol`（默认 $10^{-6}$/$10^{-9}$），不是靠加密 `t_eval`。
- 积分失败（`result.success` 为 `False`）通常意味着步长耗尽或函数发散，先检查方程量纲与刚性。
- 边值问题（给两端条件而非初值）应使用 `scipy.integrate.solve_bvp`，不在本模板范围内。
