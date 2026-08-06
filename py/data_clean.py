from pathlib import Path

import pandas as pd


DATA_FILE = Path(__file__).resolve().parents[1] / "附件.csv"


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """用每列的中位数填充数值型缺失值"""
    result = df.copy()
    numeric = result.select_dtypes(include="number").columns
    result[numeric] = result[numeric].fillna(result[numeric].median())
    return result


if __name__ == "__main__":
    df = pd.read_csv(DATA_FILE)
    df = handle_missing(df)
    print(df.head())
