"""竞赛数据清洗模板：用数值列中位数填充缺失值。"""

from pathlib import Path

import pandas as pd


DATA_FILE = Path(__file__).resolve().parents[1] / "附件.csv"


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


if __name__ == "__main__":
    df = pd.read_csv(DATA_FILE)
    df = handle_missing(df)
    print(df.head())
