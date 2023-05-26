import pandas as pd
from typing import Union, Dict


def rolling_avgs(
    df: pd.DataFrame,
    window: int,
    column_mapping: Dict[str, str],
    groupby: Union[str, list, None] = None,
    on: Union[str, None] = None,
) -> pd.DataFrame:
    cols = list(column_mapping.keys())
    rolling_avg = (
        (df.groupby(groupby) if groupby else df)
        .rolling(window, on=on)[cols]
        .mean()
        .rename(columns=column_mapping)
        .reset_index()
        .dropna()
    )
    return rolling_avg
