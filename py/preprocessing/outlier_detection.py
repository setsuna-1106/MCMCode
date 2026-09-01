"""异常值检测模板。

IQR 法按四分位间距划界（箱线图法则），对分布形状稳健，推荐默认；
3sigma 法假设数据近似正态，对偏离正态的偏态数据会产生大量误报。

检测只输出越界掩码和边界，不直接删改数据——
异常值是"盖帽、置 NaN 后填充、删行"还是保留，应由题目背景决定。
"""

from __future__ import annotations

import pandas as pd


def _numeric_frame(data: pd.DataFrame, columns) -> pd.DataFrame:
    """选取数值列子集；columns 为空时取全部数值列。"""
    if columns is None:
        frame = data.select_dtypes(include="number")
    else:
        columns = list(columns)
        missing = [name for name in columns if name not in data.columns]
        if missing:
            raise ValueError(f"data 中不存在这些列: {missing}")
        frame = data[columns]
        non_numeric = frame.select_dtypes(exclude="number").columns
        if len(non_numeric):
            raise ValueError(f"这些列不是数值列: {list(non_numeric)}")
    if frame.shape[1] == 0:
        raise ValueError("没有可检测的数值列")
    return frame


def detect_outliers_iqr(data: pd.DataFrame, *, columns=None, factor: float = 1.5):
    """用 IQR（箱线图）法则检测每列异常值。

    边界为 ``Q1 - factor * IQR`` 与 ``Q3 + factor * IQR``，
    其中 Q1/Q3 是 25%/75% 分位数；四分位数只依赖排序位置，
    不受极端值本身拉动，因此对偏态分布稳健。

    Args:
        data: 待检测的 DataFrame。
        columns: 要检测的数值列；为空时检测全部数值列。
        factor: IQR 的放大倍数，1.5 是箱线图标准取值。

    Returns:
        含 ``mask``（与原数据对齐的布尔表）、``lower``/``upper``
        （每列边界）、``counts`` 和 ``ratio``（每列异常值个数与占比）的字典。

    Raises:
        ValueError: 没有数值列、列名不存在或指定列非数值时抛出。
    """
    if factor <= 0:
        raise ValueError("factor 必须大于 0")
    frame = _numeric_frame(data, columns)

    q1 = frame.quantile(0.25)
    q3 = frame.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    mask = (frame < lower) | (frame > upper)

    return {
        "mask": mask,
        "lower": lower,
        "upper": upper,
        "counts": mask.sum(),
        "ratio": mask.sum() / len(frame),
    }


def detect_outliers_zscore(data: pd.DataFrame, *, columns=None, threshold: float = 3.0):
    """用 3sigma 法则检测每列异常值。

    把每个值标准化为 z 分数，|z| 超过阈值判为异常。
    正态假设下 |z| > 3 的概率约 0.27%；分布偏态或异常值本身
    会抬高均值和标准差，导致漏检，因此偏态数据建议用 IQR 法。

    Args:
        data: 待检测的 DataFrame。
        columns: 要检测的数值列；为空时检测全部数值列。
        threshold: z 分数阈值，默认 3.0。

    Returns:
        含 ``mask``（布尔表）、``zscores``（z 分数表）、
        ``counts`` 和 ``ratio`` 的字典。

    Raises:
        ValueError: 参数不合法、存在常数列或没有数值列时抛出。
    """
    if threshold <= 0:
        raise ValueError("threshold 必须大于 0")
    frame = _numeric_frame(data, columns)

    std = frame.std(ddof=1)
    constant_columns = list(std[std == 0].index)
    if constant_columns:
        # 常数列的标准差为 0，z 分数没有定义。
        raise ValueError(f"常数列无法计算 z 分数，请先剔除: {constant_columns}")

    zscores = (frame - frame.mean()) / std
    mask = zscores.abs() > threshold

    return {
        "mask": mask,
        "zscores": zscores,
        "counts": mask.sum(),
        "ratio": mask.sum() / len(frame),
    }


def cap_outliers(data: pd.DataFrame, detection: dict) -> pd.DataFrame:
    """把越界值收缩（盖帽）到检测边界内，返回新数据表。

    保留样本量且不改变正常值，适合"极端但真实"的指标；
    另一种常用做法是 ``data.mask(detection["mask"])`` 把异常值
    置 NaN，再交给 ``data_clean.handle_missing`` 填充。

    Args:
        data: 原始 DataFrame。
        detection: ``detect_outliers_iqr`` 的返回值。

    Returns:
        异常值已被裁剪到边界的新数据表，原始 data 不变。
    """
    lower = detection["lower"]
    upper = detection["upper"]
    result = data.copy()
    for column in lower.index:
        if column in result.columns:
            result[column] = result[column].clip(
                lower=lower[column], upper=upper[column]
            )
    return result


def main():
    # ====== 比赛时主要替换下面这部分（实际数据 pd.read_csv/read_excel 读入） ======
    import numpy as np

    rng = np.random.default_rng(42)
    normal = rng.normal(50.0, 5.0, 200)
    normal[0] = 999.0   # 注入高位异常
    normal[1] = -400.0  # 注入低位异常
    df = pd.DataFrame({
        "normal": normal,
        "skewed": rng.lognormal(3.0, 0.5, 200),  # 右偏，上尾天然存在异常
    })

    iqr = detect_outliers_iqr(df)
    print("IQR 边界:\n", pd.concat([iqr["lower"], iqr["upper"]], axis=1,
                                   keys=["lower", "upper"]))
    print("IQR 异常值个数:", iqr["counts"].to_dict())

    zscore = detect_outliers_zscore(df)
    print("3sigma 异常值个数:", zscore["counts"].to_dict())

    capped = cap_outliers(df, iqr)
    print("盖帽后 normal 列范围:", capped["normal"].min(), capped["normal"].max())

    # 本例的简单验收：替换题目后可删除或改写。
    assert iqr["mask"]["normal"].iloc[[0, 1]].all()   # 注入点必被检出
    assert zscore["mask"]["normal"].iloc[[0, 1]].all()
    assert iqr["counts"]["skewed"] > 0                # 右偏上尾有异常
    assert capped["normal"].max() <= iqr["upper"]["normal"]
    assert capped["normal"].min() >= iqr["lower"]["normal"]


if __name__ == "__main__":
    main()
