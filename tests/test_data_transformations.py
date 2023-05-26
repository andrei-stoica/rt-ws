import sys

sys.path += ["dags"]

from data_transformations import rolling_avgs
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


import pytest


@pytest.fixture()
def df() -> pd.DataFrame:
    days = 31
    labels = ["A", "B"]
    dfs = []
    for label in labels:
        ones = np.ones(days)
        arange = np.arange(days)

        start = datetime(2023, 1, 1)
        stop = start + timedelta(days=days)
        dates = np.arange(start, stop, timedelta(days=1))

        df = pd.DataFrame(
            {
                "ones": ones,
                "arange": arange,
                "label": label,
                "date": dates,
            }
        )
        dfs.append((df))

    return pd.concat(dfs)


def test_rolling_avgs(df):
    result = rolling_avgs(
        df, 30, {"ones": "ones_avg", "arange": "arange_avg"}, "label", "date"
    )
    print(result)
    assert result.shape == (4, 4)
    assert (result["label"] == ["A", "A", "B", "B"]).all()
    assert (
        result["date"]
        == [
            datetime(2023, 1, 30),
            datetime(2023, 1, 31),
            datetime(2023, 1, 30),
            datetime(2023, 1, 31),
        ]
    ).all()
    assert (result["arange_avg"] == [14.5, 15.5, 14.5, 15.5]).all()
    assert (result["ones_avg"] == [1, 1, 1, 1]).all()
