#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""常用灵敏度分析最小可执行模板。

统一接口：model(params) -> 一个有限标量。
比赛时主要替换 model、base_params、扫描范围和 sampler。

包含：
    1. 局部灵敏度：中心差分和弹性系数
    2. 一因素分析：固定其他参数，扫描一个参数
    3. 双因素分析：扫描两个参数并生成输出矩阵
    4. Monte Carlo：随机扰动多个参数并计算输出相关性
"""

import sys

import numpy as np


def _evaluate(model, params):
    # 复制参数后再交给模型，避免模型意外修改调用者的原始字典。
    value = np.asarray(model(dict(params)), dtype=float)
    if value.ndim != 0 or not np.isfinite(value):
        raise ValueError("model(params) 必须返回一个有限标量")
    return float(value)


def local_sensitivity(model, base_params, parameter, step):
    """计算一个参数的中心差分导数和弹性系数。"""
    if parameter not in base_params:
        raise KeyError(f"参数不存在: {parameter}")
    if step <= 0:
        raise ValueError("step 必须大于 0")

    base_value = float(base_params[parameter])
    # 中心差分同时计算上、下扰动，通常比单边差分更稳定。
    params_plus = dict(base_params)
    params_minus = dict(base_params)
    params_plus[parameter] = base_value + step
    params_minus[parameter] = base_value - step

    output_plus = _evaluate(model, params_plus)
    output_minus = _evaluate(model, params_minus)
    output_base = _evaluate(model, base_params)
    derivative = (output_plus - output_minus) / (2.0 * step)
    elasticity = np.nan if output_base == 0 else derivative * base_value / output_base

    return {
        "parameter": parameter,
        "base_value": base_value,
        "base_output": output_base,
        "output_plus": output_plus,
        "output_minus": output_minus,
        "derivative": derivative,
        "elasticity": elasticity,
    }


def one_way_sensitivity(model, base_params, parameter, values):
    """固定其他参数，扫描一个参数并返回参数值和模型输出。"""
    if parameter not in base_params:
        raise KeyError(f"参数不存在: {parameter}")
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("values 必须是非空有限数值序列")

    # 每次只替换一个参数，其余参数保持基准值不变。
    outputs = []
    for value in values:
        params = dict(base_params)
        params[parameter] = value
        outputs.append(_evaluate(model, params))

    return {
        "parameter": parameter,
        "values": values,
        "outputs": np.asarray(outputs),
    }


def two_way_sensitivity(
    model,
    base_params,
    parameter_x,
    values_x,
    parameter_y,
    values_y,
):
    """扫描两个参数并返回形状为 (len(values_x), len(values_y)) 的输出矩阵。"""
    if parameter_x not in base_params or parameter_y not in base_params:
        raise KeyError("parameter_x 或 parameter_y 不存在")
    if parameter_x == parameter_y:
        raise ValueError("两个扫描参数必须不同")

    values_x = np.asarray(values_x, dtype=float).reshape(-1)
    values_y = np.asarray(values_y, dtype=float).reshape(-1)
    if (
        values_x.size == 0
        or values_y.size == 0
        or not np.isfinite(values_x).all()
        or not np.isfinite(values_y).all()
    ):
        raise ValueError("values_x 和 values_y 必须是非空有限数值序列")

    # 输出矩阵的行对应 values_x，列对应 values_y，便于直接绘制热力图。
    outputs = np.empty((values_x.size, values_y.size), dtype=float)
    for i, value_x in enumerate(values_x):
        for j, value_y in enumerate(values_y):
            params = dict(base_params)
            params[parameter_x] = value_x
            params[parameter_y] = value_y
            outputs[i, j] = _evaluate(model, params)

    return {
        "parameter_x": parameter_x,
        "parameter_y": parameter_y,
        "values_x": values_x,
        "values_y": values_y,
        "outputs": outputs,
    }


def monte_carlo_sensitivity(model, sampler, n_samples=1000, random_state=42):
    """随机扰动参数，返回样本、输出和各参数的 Pearson 相关系数。"""
    if n_samples < 1:
        raise ValueError("n_samples 必须大于 0")

    # 固定随机种子后可复现实验；sampler 决定各参数的扰动分布。
    rng = np.random.default_rng(random_state)
    samples = []
    outputs = []
    for _ in range(n_samples):
        params = dict(sampler(rng))
        if not params:
            raise ValueError("sampler(rng) 必须返回非空参数字典")
        samples.append(params)
        outputs.append(_evaluate(model, params))

    keys = list(samples[0])
    if any(set(sample) != set(keys) for sample in samples):
        raise ValueError("sampler 返回的参数键必须保持一致")

    outputs = np.asarray(outputs, dtype=float)
    # 相关系数只描述单个参数与输出的线性关系，不等同于因果影响。
    correlation = {}
    for key in keys:
        values = np.asarray([sample[key] for sample in samples], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("sampler 返回的参数必须只包含有限数值")
        correlation[key] = np.nan if np.std(values) == 0 else np.corrcoef(values, outputs)[0, 1]

    return {
        "samples": samples,
        "outputs": outputs,
        "correlation": correlation,
    }


def example_model(params):
    """示例收益模型：产量受需求和产能中的较小值限制。"""
    quantity = min(params["demand"], params["capacity"])
    return params["price"] * quantity - params["cost"] * quantity**2


def main():
    # ====== 比赛时主要替换下面这部分 ======
    base_params = {
        "demand": 100.0,
        "capacity": 80.0,
        "price": 10.0,
        "cost": 0.1,
    }

    local = local_sensitivity(example_model, base_params, "price", step=0.1)
    one_way = one_way_sensitivity(
        example_model,
        base_params,
        "capacity",
        values=np.linspace(60.0, 100.0, 5),
    )
    two_way = two_way_sensitivity(
        example_model,
        base_params,
        "price",
        values_x=[8.0, 10.0, 12.0],
        parameter_y="cost",
        values_y=[0.08, 0.10, 0.12],
    )

    def sampler(rng):
        return {
            "demand": 100.0,
            "capacity": rng.uniform(70.0, 90.0),
            "price": rng.uniform(9.0, 11.0),
            "cost": rng.uniform(0.08, 0.12),
        }

    monte_carlo = monte_carlo_sensitivity(
        example_model,
        sampler,
        n_samples=1000,
        random_state=42,
    )

    print("局部导数:", f"{local['derivative']:.8f}")
    print("局部弹性系数:", f"{local['elasticity']:.8f}")
    print("一因素输出:", np.round(one_way["outputs"], 8))
    print("双因素输出矩阵:\n", np.round(two_way["outputs"], 8))
    print("Monte Carlo 输出均值:", f"{monte_carlo['outputs'].mean():.8f}")
    print("参数相关性:", monte_carlo["correlation"])

    assert np.isclose(local["derivative"], 80.0)
    assert one_way["outputs"].shape == (5,)
    assert two_way["outputs"].shape == (3, 3)
    assert monte_carlo["outputs"].shape == (1000,)

    if "--csv" in sys.argv:
        np.savetxt(
            "sensitivity_one_way_out.csv",
            np.column_stack((one_way["values"], one_way["outputs"])),
            delimiter=",",
            fmt="%.10f",
            header="parameter,output",
            comments="",
        )
        np.savetxt(
            "sensitivity_two_way_out.csv",
            two_way["outputs"],
            delimiter=",",
            fmt="%.10f",
        )
        np.savetxt(
            "sensitivity_monte_carlo_out.csv",
            np.column_stack((np.arange(1, monte_carlo["outputs"].size + 1), monte_carlo["outputs"])),
            delimiter=",",
            fmt=["%d", "%.10f"],
            header="sample,output",
            comments="",
        )
        print("已保存 -> sensitivity_*_out.csv")


if __name__ == "__main__":
    main()
