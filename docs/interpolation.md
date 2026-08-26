# 插值（interpolation）

在离散观测点之间估计中间值：构造一条**穿过所有观测点**的曲线 $f(x)$，再查询区间内的任意位置。对应模板 `py/interpolation/scipy_interpolate.py`，提供三种常用方法。

插值与拟合（见[回归模块](regression.md)）的区别：插值要求曲线精确穿过每个观测点，适合「数据本身可靠、只缺中间值」的场景；拟合允许曲线偏离观测点，适合数据含噪声、需要趋势规律的场景。

## 三种方法

| 方法 | 每段曲线 | 特点 | 适用 |
| --- | --- | --- | --- |
| `linear` | 直线 | 稳定、绝不过冲，节点处有折角 | 快速估算、数据本身分段变化 |
| `cubic` | 三次多项式（样条） | 整体二阶连续、曲线平滑，可能过冲 | 平滑绘制曲线、光滑物理量 |
| `pchip` | 三次多项式（保形） | 保持观测数据的单调性与局部形状 | 单调工程数据、不允许过冲时 |

### 原理要点

- **linear**：相邻观测点 $(x_i, y_i)$ 与 $(x_{i+1}, y_{i+1})$ 之间直接连线，节点处斜率突变。
- **cubic**：在每个子区间上取一个三次多项式，并要求相邻多项式在节点处的函数值、一阶导数、二阶导数都相等，因此曲线光滑；但局部大陡变可能在附近产生「过冲」（超出观测值范围）。
- **pchip**：同样分段三次，但节点处导数被刻意限制为与相邻数据单调性一致，保证曲线不会在数据递增区间内局部下跌，代价是光滑度略低。

## 用法

```python
import numpy as np
from py.interpolation.scipy_interpolate import interpolate_1d, make_interpolator

x = np.array([0.0, 1.0, 2.0, 3.0])
y = np.array([0.0, 1.0, 4.0, 9.0])

# 一次性查询
y_new = interpolate_1d(x, y, np.linspace(0.0, 3.0, 31), kind="cubic")

# 或先构造插值器，反复调用
f = make_interpolator(x, y, kind="pchip")
value = f(1.5)
```

- `x` 必须严格递增且无重复，`y` 与其等长；输入不合法会抛 `ValueError`。
- `kind` 取 `linear` / `cubic` / `pchip`，默认 `cubic`。
- `extrapolate` 默认 `False`：查询点超出观测区间 $[\min x, \max x]$ 时返回 `NaN`，防止误用外推。
- 直接运行模板（含 `--csv` 导出结果）：

```bash
.venv/bin/python py/interpolation/scipy_interpolate.py --csv
```

## 注意

- 外推（查询区间外的点）风险远大于内插，开启 `extrapolate=True` 前应在论文中说明其合理性。
- 观测点稀疏时三次样条的过冲会被放大，可先降级到 `pchip` 或 `linear` 观察。
- 多维或散点插值需要 `griddata` 等其他工具，本模板只处理一维。
