"""分类变量编码模板。

one_hot_encode 把无序类别展开为 0/1 哑变量列，不引入虚假的
大小关系（红 > 蓝没有意义），线性模型的默认选择；
label_encode 把类别映射为从 0 开始的整数，适合有序类别
（低/中/高）或树模型（XGBoost、随机森林）。

默认自动处理所有非数值列，与 data_clean 只处理数值列的行为互补。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _categorical_columns(data: pd.DataFrame, columns) -> list[str]:
    """选出待编码的分类列；columns 为空时取全部非数值、非日期列。"""
    if columns is None:
        selected = list(
            data.select_dtypes(exclude=["number", "datetime"]).columns
        )
    else:
        columns = list(columns)
        missing = [name for name in columns if name not in data.columns]
        if missing:
            raise ValueError(f"data 中不存在这些列: {missing}")
        numeric = list(
            data[columns].select_dtypes(include="number").columns
        )
        if numeric:
            # 数值列做 one-hot 或标签编码通常是误用，直接拦截。
            raise ValueError(f"这些列是数值列，无需编码: {numeric}")
        selected = columns
    if not selected:
        raise ValueError("没有可编码的分类列")
    return selected


def one_hot_encode(data: pd.DataFrame, *, columns=None, drop_first: bool = False):
    """把分类列展开为 0/1 哑变量列。

    Args:
        data: 含分类列的 DataFrame。
        columns: 要编码的列；为空时自动选择全部分类列。
        drop_first: 每个类别组是否丢弃第一个哑变量列。线性回归中
            全部保留会产生完全共线性（哑变量之和恒为 1），应设 True。

    Returns:
        ``(encoded, info)``：编码后的 DataFrame，以及记录
        ``columns``（被编码列）、``drop_first`` 和 ``dummy_columns``
        （新增哑变量列名）的字典。

    Raises:
        ValueError: 没有分类列、列名不存在或误传数值列时抛出。
    """
    cats = _categorical_columns(data, columns)
    encoded = pd.get_dummies(data, columns=cats, drop_first=drop_first,
                             dtype=float)
    info = {
        "columns": cats,
        "drop_first": drop_first,
        "dummy_columns": [
            name for name in encoded.columns if name not in data.columns
        ],
    }
    return encoded, info


def label_encode(data: pd.DataFrame, *, columns=None):
    """把分类列映射为从 0 开始的整数编码。

    编码按类别排序后依次分配（低/中/高 -> 0/1/2），保证可复现，
    论文中应报告映射对照表；缺失值保留为 NaN，需再交给
    ``data_clean.handle_missing`` 之外的策略处理（分类缺失不能中位数填充）。

    Args:
        data: 含分类列的 DataFrame。
        columns: 要编码的列；为空时自动选择全部分类列。

    Returns:
        ``(encoded, mappings)``：编码后的 DataFrame，以及
        ``{列名: {原类别: 编码}}`` 的映射字典。

    Raises:
        ValueError: 没有分类列或列名不存在时抛出。
    """
    cats = _categorical_columns(data, columns)
    encoded = data.copy()
    mappings = {}
    for name in cats:
        column = encoded[name]
        # 按字符串顺序排序，混合类型时也有确定性的编码顺序。
        categories = sorted(column.dropna().unique(), key=str)
        mapping = {category: code for code, category in enumerate(categories)}
        # map 对缺失值返回 NaN，因此整列转 float 保留缺失信息。
        encoded[name] = column.map(mapping).astype("float64")
        mappings[name] = mapping
    return encoded, mappings


def main():
    # ====== 比赛时主要替换下面这部分（实际数据 pd.read_csv/read_excel 读入） ======
    df = pd.DataFrame({
        "yield": [500.0, 520.0, 530.0, 510.0, 540.0],
        "color": ["red", "blue", "red", "green", np.nan],
        "plan": ["B", "A", "B", "A", "A"],
    })

    encoded, info = one_hot_encode(df)
    print("one-hot 新增列:", info["dummy_columns"])
    print(encoded.head())

    encoded_first, info_first = one_hot_encode(df, drop_first=True)
    print("drop_first 新增列:", info_first["dummy_columns"])

    labeled, mappings = label_encode(df)
    print("标签编码映射:", mappings)
    print(labeled.head())

    # 本例的简单验收：替换题目后可删除或改写。
    assert set(info["columns"]) == {"color", "plan"}
    assert encoded.shape[1] == 1 + 3 + 2          # yield + 3 色哑变量 + 2 方案
    assert encoded_first.shape[1] == 1 + 2 + 1    # 每组各丢一个哑变量
    assert encoded.loc[0, "color_red"] == 1.0
    assert mappings["plan"] == {"A": 0, "B": 1}
    assert labeled.loc[0, "plan"] == 1.0
    assert np.isnan(labeled.loc[4, "color"])      # 缺失保留为 NaN


if __name__ == "__main__":
    main()
