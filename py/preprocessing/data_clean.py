"""竞赛数据清洗模板：缺失值报告、中位数填充和去重。

数据约定：只自动处理数值列的缺失；文本、分类和日期列不自动填充，
分类变量的处理见 encode_categorical.py，异常值处理见 outlier_detection.py。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """统计各列缺失值的数量与比例，按比例降序排列。

    Args:
        df: 待检查的数据表。

    Returns:
        含 ``missing_count``（缺失数）和 ``missing_ratio``（缺失比例）两列的
        DataFrame，只包含存在缺失的列；无缺失时返回空表。
    """
    counts = df.isna().sum()
    report = pd.DataFrame({
        "missing_count": counts,
        "missing_ratio": counts / len(df),
    })
    # 只保留有缺失的列，并按比例降序，优先暴露问题最重的列。
    report = report[report["missing_count"] > 0]
    return report.sort_values("missing_ratio", ascending=False)


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """填充数值列缺失值，并保留原始 DataFrame 不变。

    Args:
        df: 待处理的数据表。

    Returns:
        数值列缺失值已用各列中位数填充的数据表。
    """
    result = df.copy()
    # 文本、分类和日期列不自动填充，避免把题意相关信息强行数值化。
    numeric = result.select_dtypes(include="number").columns
    result[numeric] = result[numeric].fillna(result[numeric].median())
    return result


def drop_duplicates(df: pd.DataFrame, *, subset=None) -> pd.DataFrame:
    """删除完全重复的行，保留每个重复组第一次出现。

    Args:
        df: 待清洗的数据表。
        subset: 只根据这些列判断重复；为空时要求整行完全一致。

    Returns:
        去重后的新数据表，原始 df 不变，保留原索引。
    """
    return df.drop_duplicates(subset=subset, keep="first")


if __name__ == "__main__":
    # ====== 比赛时主要替换下面这部分（实际数据 pd.read_csv/read_excel 读入） ======
    df = pd.DataFrame({
        "yield": [500.0, 520.0, np.nan, 520.0],
        "cost": [12.0, np.nan, 15.0, np.nan],
        "plan": ["A", "B", "A", "B"],
    })

    print("缺失值报告:\n", missing_report(df))
    df = handle_missing(df)
    print("中位数填充后:\n", df.head())
    df = drop_duplicates(df)
    print("去重后行数:", len(df))

    # 本例的简单验收：替换题目后可删除或改写。
    demo = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    assert missing_report(demo)["missing_ratio"].iloc[0] == 1.0 / 3.0
    assert not handle_missing(demo).isna().any().any()
    # 第 2、4 行填充后完全相同，整行去重应剩 3 行；按 plan 列去重应剩 2 行。
    assert len(drop_duplicates(df)) == 3
    assert len(drop_duplicates(df, subset=["plan"])) == 2
