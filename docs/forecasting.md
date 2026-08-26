# 预测（forecasting）

对时间序列做短期外推。对应 `py/forecasting/`，提供四个模板：GM(1,1)、一次指数平滑、Holt / Holt-Winters、ARIMA。共同约定：`y` 是**按时间从早到晚排列**的一维序列，预测时不能随机打乱或用未来信息训练。

| 方法 | 数据量要求 | 建模成分 | 适用场景 |
| --- | --- | --- | --- |
| GM(1,1) | 至少 4 期 | 累加后的指数趋势 | 小样本、近似单调的增长/衰减 |
| 一次指数平滑 | 至少 2 期 | 水平项 | 无趋势无季节的平稳波动 |
| Holt / Holt-Winters | 3 期 / 两个完整季节 | 水平 + 趋势（+ 季节） | 有趋势（或季节）的序列 |
| ARIMA | 建议 30+ 期 | 自回归 + 差分 + 移动平均 | 较长序列、无显式季节结构 |

## GM(1,1)

### 原理

灰色模型，核心技巧是**一次累加生成（AGO）**：对非负序列 $x^{(0)}$ 做累加 $x^{(1)}_k = \sum_{i \le k} x^{(0)}_i$，把随机波动削弱成近似单调的增长曲线，再对累加序列建立白化微分方程：

$$\frac{d x^{(1)}}{dt} + a\, x^{(1)} = b$$

用邻均值 $z^{(1)}_k = \frac{1}{2}(x^{(1)}_k + x^{(1)}_{k-1})$ 作背景值，最小二乘估计参数 $(a, b)$，得到时间响应式并**累减还原**回原始尺度：

$$\hat{x}^{(1)}_{k+1} = \left(x^{(0)}_1 - \frac{b}{a}\right) e^{-ak} + \frac{b}{a}, \qquad \hat{x}^{(0)}_{k+1} = \hat{x}^{(1)}_{k+1} - \hat{x}^{(1)}_{k}$$

### 用法

```python
from py.forecasting.gm11 import gm11

forecast, (a, b), fitted = gm11([12, 15, 19, 24, 30], steps=3)
```

- 入参：`data`（至少 4 个非负观测）、`steps`（预测期数）。
- 返回：未来预测值、参数 `(a, b)`、历史拟合值。
- 直接运行：`.venv/bin/python py/forecasting/gm11.py`

### 注意

- 要求数据非负（含 0 也不行需平移）、级比近似落在可容覆盖区间内；波动大的序列不适用。
- 只适合短期外推；论文中应报告残差检验（后验差比 $C$、小误差概率 $P$）。

## 一次指数平滑（ese）

### 原理

只维护一个**水平项**（对当前水平的估计），按指数权重更新：

$$l_t = \alpha\, y_t + (1 - \alpha)\, l_{t-1}$$

递推展开后 $l_t$ 是全部历史的加权平均，且**越近的观测权重越大**（权重按 $(1-\alpha)^k$ 衰减）。没有趋势和季节成分，未来各期预测都是常数 $\hat{y}_{T+h} = l_T$。

$\alpha$ 越大越「信任」最新数据、响应快但波动大；越小越平滑但反应滞后。

### 用法

```python
from py.forecasting.ese import exponential_smoothing

level, forecast = exponential_smoothing(y, alpha=0.3, steps=3)
```

- 返回 `(level, forecast)`：历史平滑值与未来 `steps` 期预测（常数）。
- 直接运行：`.venv/bin/python py/forecasting/ese.py`

### 注意

- 数据有趋势时预测系统性滞后，应改用 Holt；有季节改用 Holt-Winters。

## Holt 与 Holt-Winters（holt）

### 原理

在水平项之外逐步增加成分（statsmodels 自动优化平滑参数）：

- **Holt（二次平滑）**：增加趋势项 $b_t$，预测 $\hat{y}_{T+h} = l_T + h\, b_T$；
  $$l_t = \alpha y_t + (1-\alpha)(l_{t-1} + b_{t-1}), \qquad b_t = \beta (l_t - l_{t-1}) + (1-\beta) b_{t-1}$$
- **Holt-Winters（三次平滑）**：再增加季节项 $s_t$，按周期 `seasonal_periods` 重复。季节影响可分为：
  - 加法 `add`：季节波动幅度不随水平变化，$y_t = l_t + b_t t + s_t$；
  - 乘法 `mul`：波动幅度与水平成比例（要求数据全为正），$y_t = (l_t + b_t t) \cdot s_t$。
- `damped_trend=True` 给趋势乘上衰减因子 $\phi^h$，避免趋势无限外推。

### 用法

```python
from py.forecasting.holt import fit_holt, fit_holt_winters

holt = fit_holt(y, steps=4, damped_trend=True)
hw = fit_holt_winters(y, seasonal_periods=4, steps=4, trend="add", seasonal="add")
print(holt["forecast"], hw["forecast"])
```

- `fit_holt`：序列至少 3 期；`fit_holt_winters`：至少两个完整季节周期。
- 两者都返回字典：`model`、`fitted`（历史拟合）、`forecast`（未来预测）、`method`。
- 直接运行：`.venv/bin/python py/forecasting/holt.py`

### 注意

- 季节周期要按数据频率设定（季度数据 `4`、月度 `12`）；周期设错结果会明显异常。
- 乘法形式要求 `y` 全部大于 0。

## ARIMA

### 原理

ARIMA$(p, d, q)$ 用三部分刻画序列：

- **AR($p$) 自回归**：当前值 = 过去 $p$ 期值的线性组合 $+ \varepsilon_t$；
- **$d$ 阶差分**：差分 $d$ 次把非平稳序列（有趋势）化为平稳序列；
- **MA($q$) 移动平均**：误差项本身存在 $q$ 期的自相关。

$$\phi(B)\, (1-B)^d y_t = \theta(B)\, \varepsilon_t$$

其中 $B$ 是滞后算子，$\phi$、$\theta$ 分别是 AR、MA 多项式。平稳性是前提，因此靠差分处理趋势；阶数可结合 ACF/PACF 图或 AIC/BIC 选择。模板按时间顺序划分训练/测试段做**多步预测评估**，指标为 MAE、MSE、RMSE、MAPE。

### 用法

```python
from py.forecasting.arima import fit_arima, predict_arima
from statsmodels.tsa.arima.model import ARIMA

output = fit_arima(y, order=(1, 1, 1), test_size=0.2)   # 评估
print(output["rmse"], output["mape"])

model = ARIMA(y, order=output["order"]).fit()           # 阶数确定后全量重拟合
future = predict_arima(model, steps=3)
```

- `order=(p, d, q)`；`test_size` 可传比例（如 `0.2`）或整数个数。
- 返回字典含 `model`、`forecast`（测试段预测）、`mae`/`mse`/`rmse`/`mape`、`order`。
- 直接运行：`.venv/bin/python py/forecasting/arima.py`

### 注意

- 序列至少 8 期（模板校验），但要有可信的阶数估计通常需要更多历史。
- 强季节模式应使用 SARIMA（`seasonal_order` 参数），本模板只建非季节 ARIMA。
- MAPE 在真实值接近 0 时不稳定，模板自动只在非零样本上计算。
