from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ibis
from ibis.backends.tests.errors import ClickHouseDatabaseError

cc = pytest.importorskip("clickhouse_connect")


@pytest.fixture(scope="module")
def win_df():
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7],
            "g": ["a", "a", "a", "a", "b", "b", "b"],
            "x": [1, 1, 2, 3, 5, 5, 8],
            "y": [10.0, 20.0, np.nan, 40.0, 50.0, 60.0, 70.0],
        }
    )


@pytest.fixture(scope="module")
def win_t(win_df):
    return ibis.memtable(win_df)


def test_window_functions_sql_snapshot(alltypes, assert_sql):
    t = alltypes
    expr = t.select(
        t.id,
        t.double_col.sum()
        .over(ibis.window(order_by=t.id, following=0))
        .name("cumulative_sum"),
        ibis.rank().over(ibis.window(order_by=t.id)).name("rank"),
        ibis.cume_dist().over(ibis.window(order_by=t.id)).name("cume_dist"),
        ibis.percent_rank().over(ibis.window(order_by=t.id)).name("percent_rank"),
        t.double_col.lag().over(ibis.window(order_by=t.id)).name("lag"),
        t.double_col.lead().over(ibis.window(order_by=t.id)).name("lead"),
    )
    assert_sql(expr)


def test_lag_lead_ignore_frame_bounds(con, win_t, win_df):
    # lagInFrame/leadInFrame are constrained by the frame, but the standard
    # lag/lead functions used by the compiler are not; a bounded frame must
    # not make lead return NULL.
    frame = ibis.window(order_by=win_t.id, preceding=2, following=0)
    expr = win_t.mutate(
        lag_y=win_t.y.lag().over(frame),
        lead_y=win_t.y.lead().over(frame),
        lead_default=win_t.y.lead(1, -1.0).over(frame),
    ).order_by("id")
    result = con.execute(expr)

    # The default only fills rows whose target row is outside the partition;
    # rows whose target value is NULL stay NULL.
    lead_default = win_df.y.shift(-1)
    lead_default.iloc[-1] = -1.0
    expected = win_df.assign(
        lag_y=win_df.y.shift(1),
        lead_y=win_df.y.shift(-1),
        lead_default=lead_default,
    )
    pd.testing.assert_frame_equal(
        result[["id", "lag_y", "lead_y", "lead_default"]].reset_index(drop=True),
        expected[["id", "lag_y", "lead_y", "lead_default"]].reset_index(drop=True),
    )


def test_cume_dist_percent_rank_match_pandas(con, win_t, win_df):
    frame = ibis.window(order_by=win_t.x)
    expr = win_t.mutate(
        cume_dist=ibis.cume_dist().over(frame),
        percent_rank=ibis.percent_rank().over(frame),
    ).order_by("id")
    result = con.execute(expr)

    n_rows = win_df.x.shape[0]
    expected_cume_dist = win_df.x.rank(method="max") / n_rows
    expected_percent_rank = win_df.x.rank(method="min").sub(1).div(n_rows - 1)

    pd.testing.assert_series_equal(
        result.cume_dist.reset_index(drop=True),
        expected_cume_dist.reset_index(drop=True),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        result.percent_rank.reset_index(drop=True),
        expected_percent_rank.reset_index(drop=True),
        check_names=False,
    )


def test_partitioned_cume_dist_percent_rank_match_pandas(con, win_t, win_df):
    frame = ibis.window(group_by=win_t.g, order_by=win_t.x)
    expr = win_t.mutate(
        cume_dist=ibis.cume_dist().over(frame),
        percent_rank=ibis.percent_rank().over(frame),
    ).order_by("id")
    result = con.execute(expr)

    sizes = win_df.groupby("g").x.transform("size")
    min_ranks = win_df.groupby("g").x.transform("rank", method="min")
    max_ranks = win_df.groupby("g").x.transform("rank", method="max")

    pd.testing.assert_series_equal(
        result.cume_dist.reset_index(drop=True),
        (max_ranks / sizes).reset_index(drop=True),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        result.percent_rank.reset_index(drop=True),
        min_ranks.sub(1).div(sizes - 1).reset_index(drop=True),
        check_names=False,
    )


def test_rows_frame_bounds_match_pandas(con, win_t, win_df):
    expr = win_t.mutate(
        cumulative=win_t.y.sum().over(ibis.window(order_by=win_t.id, following=0)),
        centered=win_t.y.sum().over(
            ibis.window(order_by=win_t.id, preceding=1, following=1)
        ),
        trailing=win_t.y.sum().over(
            ibis.window(order_by=win_t.id, preceding=0, following=2)
        ),
    ).order_by("id")
    result = con.execute(expr)

    # SQL aggregates ignore NULLs, whereas pandas rolling/cumulative ops
    # propagate NaN, so normalize the input before computing expectations.
    filled = win_df.y.fillna(0.0)
    pd.testing.assert_series_equal(
        result.cumulative.reset_index(drop=True),
        filled.cumsum().reset_index(drop=True),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        result.centered.reset_index(drop=True),
        filled.rolling(3, min_periods=1, center=True).sum().reset_index(drop=True),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        result.trailing.reset_index(drop=True),
        filled[::-1].rolling(3, min_periods=1).sum()[::-1].reset_index(drop=True),
        check_names=False,
    )


def test_grouped_rows_frame_match_pandas(con, win_t, win_df):
    frame = ibis.window(group_by=win_t.g, order_by=win_t.id, following=0)
    expr = win_t.mutate(cumulative=win_t.y.sum().over(frame)).order_by("id")
    result = con.execute(expr)
    expected = win_df.assign(y=win_df.y.fillna(0.0)).groupby("g").y.cumsum()

    pd.testing.assert_series_equal(
        result.cumulative.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_range_numeric_frame_match_pandas(con, win_t, win_df):
    frame = ibis.window(range=(-1, 1), order_by=win_t.x)
    expr = win_t.mutate(window_sum=win_t.x.sum().over(frame)).order_by("id")
    result = con.execute(expr)
    expected = win_df.x.apply(
        lambda value: win_df.x[(win_df.x >= value - 1) & (win_df.x <= value + 1)].sum()
    )

    pd.testing.assert_series_equal(
        result.window_sum.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_range_interval_frame_unsupported(con, win_t):
    frame = ibis.window(range=(-ibis.interval(seconds=10), 0), order_by=win_t.id)
    with pytest.raises(ClickHouseDatabaseError):
        con.execute(win_t.mutate(window_sum=win_t.x.sum().over(frame)))
