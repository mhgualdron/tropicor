"""Unit tests for common-period alignment."""

import numpy as np
import pandas as pd
import pytest

from tropicor.core.alignment import align_common_period


def _series(start: str, periods: int, values=None) -> pd.Series:
    idx = pd.date_range(start, periods=periods, freq="1MS")
    data = np.arange(periods, dtype=float) if values is None else values
    return pd.Series(data, index=idx)


def test_exact_intersection_of_mismatched_periods():
    obs = _series("1990-01-01", 12 * 26)  # 1990-2015
    mod = _series("1990-01-01", 12 * 30)  # 1990-2019
    a_obs, a_mod = align_common_period(obs, mod)
    assert len(a_obs) == len(a_mod) == 12 * 26
    assert a_obs.index.equals(a_mod.index)
    assert a_obs.index.max() == pd.Timestamp("2015-12-01")


def test_model_masked_where_observation_is_nan_without_filling():
    obs = _series("2000-01-01", 6, [1.0, np.nan, 3.0, 4.0, np.nan, 6.0])
    mod = _series("2000-01-01", 6, [10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    a_obs, a_mod = align_common_period(obs, mod, drop_na=False)
    assert len(a_obs) == 6
    assert a_mod.isna().tolist() == [False, True, False, False, True, False]
    assert a_obs.isna().tolist() == a_mod.isna().tolist()
    assert a_mod.iloc[0] == 10.0 and a_mod.iloc[5] == 60.0


def test_observation_masked_where_model_is_nan():
    obs = _series("2000-01-01", 3, [1.0, 2.0, 3.0])
    mod = _series("2000-01-01", 3, [10.0, np.nan, 30.0])
    a_obs, a_mod = align_common_period(obs, mod, drop_na=False)
    assert np.isnan(a_obs.iloc[1]) and np.isnan(a_mod.iloc[1])


def test_drop_na_returns_only_valid_pairs():
    obs = _series("2000-01-01", 4, [1.0, np.nan, 3.0, 4.0])
    mod = _series("2000-01-01", 4, [10.0, 20.0, np.nan, 40.0])
    a_obs, a_mod = align_common_period(obs, mod, drop_na=True)
    assert a_obs.tolist() == [1.0, 4.0]
    assert a_mod.tolist() == [10.0, 40.0]
    assert not a_obs.isna().any()


def test_drop_na_false_keeps_full_intersection_index():
    obs = _series("2000-01-01", 4, [1.0, np.nan, 3.0, 4.0])
    mod = _series("2000-01-01", 4, [10.0, 20.0, 30.0, 40.0])
    a_obs, _ = align_common_period(obs, mod, drop_na=False)
    assert len(a_obs) == 4


def test_non_contiguous_dates_are_not_filled():
    idx_obs = pd.DatetimeIndex(["2000-01-01", "2000-03-01", "2000-05-01"])
    obs = pd.Series([1.0, 2.0, 3.0], index=idx_obs)
    mod = _series("2000-01-01", 6)
    a_obs, a_mod = align_common_period(obs, mod)
    assert list(a_obs.index) == list(idx_obs)
    assert a_mod.tolist() == [0.0, 2.0, 4.0]


def test_unsorted_input_is_sorted():
    obs = _series("2000-01-01", 4).iloc[::-1]
    mod = _series("2000-01-01", 4)
    a_obs, a_mod = align_common_period(obs, mod)
    assert a_obs.index.is_monotonic_increasing
    assert a_obs.index.equals(a_mod.index)


def test_inputs_are_not_modified():
    obs = _series("2000-01-01", 3, [1.0, np.nan, 3.0])
    mod = _series("2000-01-01", 3, [10.0, 20.0, 30.0])
    align_common_period(obs, mod, drop_na=False)
    assert mod.tolist() == [10.0, 20.0, 30.0]
    assert np.isnan(obs.iloc[1])


def test_no_overlap_raises():
    obs = _series("1990-01-01", 12)
    mod = _series("2010-01-01", 12)
    with pytest.raises(ValueError, match="no common timestamps"):
        align_common_period(obs, mod)


def test_empty_series_raises():
    empty = pd.Series(dtype="float64", index=pd.DatetimeIndex([]))
    with pytest.raises(ValueError, match="no common timestamps"):
        align_common_period(empty, _series("2000-01-01", 3))


def test_non_datetime_index_raises():
    obs = pd.Series([1.0, 2.0, 3.0])
    mod = _series("2000-01-01", 3)
    with pytest.raises(ValueError, match="DatetimeIndex"):
        align_common_period(obs, mod)


def test_duplicate_timestamps_raise():
    idx = pd.DatetimeIndex(["2000-01-01", "2000-01-01", "2000-02-01"])
    obs = pd.Series([1.0, 2.0, 3.0], index=idx)
    mod = _series("2000-01-01", 3)
    with pytest.raises(ValueError, match="duplicated"):
        align_common_period(obs, mod)


def test_overlap_without_simultaneous_valid_values_returns_empty():
    obs = _series("2000-01-01", 2, [1.0, np.nan])
    mod = _series("2000-01-01", 2, [np.nan, 2.0])
    a_obs, a_mod = align_common_period(obs, mod, drop_na=True)
    assert a_obs.empty and a_mod.empty
