"""Unit tests for thermal range diagnostics."""

import numpy as np
import pandas as pd
import pytest

from tropicor.core.dtr import (
    compute_double_difference,
    compute_dtr,
    compute_monthly_extreme_range,
)


def _s(values, start: str = "2000-01-01") -> pd.Series:
    idx = pd.date_range(start, periods=len(values), freq="1MS")
    return pd.Series(values, index=idx, dtype=float)


# ---------------------------------------------------------------- compute_dtr


def test_compute_dtr_known_values() -> None:
    tmax = _s([28.0, 27.0, 29.5])
    tmin = _s([18.0, 17.5, 19.0])
    dtr = compute_dtr(tmax, tmin)
    assert dtr.tolist() == pytest.approx([10.0, 9.5, 10.5])
    assert dtr.name == "dtr"
    assert dtr.index.equals(tmax.index)


def test_compute_dtr_equal_extremes_is_zero_without_warning(recwarn) -> None:
    dtr = compute_dtr(_s([20.0, 21.0]), _s([20.0, 21.0]))
    assert dtr.tolist() == [0.0, 0.0]
    assert len(recwarn) == 0


def test_compute_dtr_tmin_above_tmax_becomes_nan_with_warning() -> None:
    tmax = _s([28.0, 18.0, 27.0, 17.0, 26.0])
    tmin = _s([18.0, 25.0, 17.0, 24.0, 16.0])
    with pytest.warns(UserWarning, match=r"Detected 2 timesteps \(40\.0%"):
        dtr = compute_dtr(tmax, tmin)
    assert dtr.isna().tolist() == [False, True, False, True, False]
    assert dtr.dropna().tolist() == pytest.approx([10.0, 10.0, 10.0])
    assert len(dtr) == 5


def test_compute_dtr_warning_counts_only_valid_pairs() -> None:
    tmax = _s([28.0, np.nan, 18.0, 27.0])
    tmin = _s([18.0, 10.0, 25.0, np.nan])
    # 2 valid pairs (Jan, Mar); 1 inverted -> 50% of valid pairs, not of all steps.
    with pytest.warns(UserWarning, match=r"Detected 1 timesteps \(50\.0%"):
        dtr = compute_dtr(tmax, tmin)
    assert dtr.isna().tolist() == [False, True, True, True]


def test_compute_dtr_nan_in_either_input_propagates() -> None:
    tmax = _s([28.0, np.nan, 27.0])
    tmin = _s([18.0, 17.0, np.nan])
    dtr = compute_dtr(tmax, tmin)
    assert dtr.isna().tolist() == [False, True, True]
    assert dtr.iloc[0] == pytest.approx(10.0)


def test_compute_dtr_uses_common_timestamps_only() -> None:
    tmax = _s([28.0, 27.0, 26.0, 25.0], start="2000-01-01")
    tmin = _s([18.0, 17.0, 16.0], start="2000-02-01")
    dtr = compute_dtr(tmax, tmin)
    assert list(dtr.index) == list(pd.date_range("2000-02-01", periods=3, freq="1MS"))
    # Feb: 27-18, Mar: 26-17, Apr: 25-16 -> 9 each
    assert dtr.tolist() == pytest.approx([9.0, 9.0, 9.0])


def test_compute_dtr_no_common_timestamps_raises() -> None:
    with pytest.raises(ValueError, match="no common timestamps"):
        compute_dtr(_s([28.0], "1990-01-01"), _s([18.0], "2010-01-01"))


def test_compute_dtr_empty_raises() -> None:
    empty = pd.Series(dtype="float64", index=pd.DatetimeIndex([]))
    with pytest.raises(ValueError, match="must not be empty"):
        compute_dtr(empty, _s([18.0]))


def test_compute_dtr_all_inverted_raises() -> None:
    tmax = _s([10.0, 11.0])
    tmin = _s([20.0, 21.0])
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="No valid"):
        compute_dtr(tmax, tmin)


def test_compute_dtr_all_nan_raises() -> None:
    nan = _s([np.nan, np.nan])
    with pytest.raises(ValueError, match="No valid"):
        compute_dtr(nan, _s([18.0, 17.0]))


def test_compute_dtr_requires_datetime_index() -> None:
    with pytest.raises(ValueError, match="DatetimeIndex"):
        compute_dtr(pd.Series([28.0, 27.0]), pd.Series([18.0, 17.0]))


def test_compute_dtr_does_not_modify_inputs() -> None:
    tmax = _s([28.0, 18.0])
    tmin = _s([18.0, 25.0])
    with pytest.warns(UserWarning):
        compute_dtr(tmax, tmin)
    assert tmax.tolist() == [28.0, 18.0]
    assert tmin.tolist() == [18.0, 25.0]


# ------------------------------------------------ compute_monthly_extreme_range


def test_extreme_range_known_values_and_name() -> None:
    tmax_abs = _s([31.0, 30.0])
    tmin_abs = _s([12.0, 13.5])
    rng = compute_monthly_extreme_range(tmax_abs, tmin_abs)
    assert rng.tolist() == pytest.approx([19.0, 16.5])
    assert rng.name == "extreme_range"


def test_extreme_range_is_never_smaller_than_mean_dtr() -> None:
    """Absolute extremes bound the means, so the extreme range is >= DTR."""
    tmax_mean = _s([26.0, 27.0, 25.5, 28.0])
    tmin_mean = _s([15.0, 16.0, 14.5, 15.5])
    tmax_abs = tmax_mean + _s([3.0, 4.5, 2.0, 5.0])
    tmin_abs = tmin_mean - _s([2.5, 1.0, 3.0, 2.0])
    dtr = compute_dtr(tmax_mean, tmin_mean)
    extreme = compute_monthly_extreme_range(tmax_abs, tmin_abs)
    assert (extreme >= dtr).all()
    assert (extreme > dtr).all()


def test_extreme_range_tmin_above_tmax_warning_names_inputs() -> None:
    with pytest.warns(UserWarning, match=r"tmin_abs > tmax_abs"):
        out = compute_monthly_extreme_range(_s([30.0, 10.0]), _s([12.0, 20.0]))
    assert out.isna().tolist() == [False, True]


def test_extreme_range_errors_match_dtr_contract() -> None:
    empty = pd.Series(dtype="float64", index=pd.DatetimeIndex([]))
    with pytest.raises(ValueError, match="must not be empty"):
        compute_monthly_extreme_range(empty, _s([10.0]))
    with pytest.raises(ValueError, match="no common timestamps"):
        compute_monthly_extreme_range(
            _s([30.0], "1990-01-01"), _s([10.0], "2010-01-01")
        )


# ------------------------------------------------- compute_double_difference


def test_double_difference_positive_when_model_underestimates_range() -> None:
    obs = _s([14.0, 14.0, 14.0])
    mod = _s([8.0, 8.0, 8.0])
    dd = compute_double_difference(obs, mod)
    assert dd.tolist() == pytest.approx([6.0, 6.0, 6.0])
    assert dd.name == "double_difference"


def test_double_difference_negative_when_model_overestimates_range() -> None:
    dd = compute_double_difference(_s([6.0]), _s([9.0]))
    assert dd.iloc[0] == pytest.approx(-3.0)


def test_double_difference_cancels_constant_temperature_offset() -> None:
    """A constant bias between model and station does not enter the ranges."""
    tmax_obs, tmin_obs = _s([26.0, 27.0]), _s([14.0, 15.0])
    offset = 4.0
    dtr_obs = compute_dtr(tmax_obs, tmin_obs)
    dtr_mod = compute_dtr(tmax_obs + offset, tmin_obs + offset)
    assert compute_double_difference(dtr_obs, dtr_mod).tolist() == pytest.approx(
        [0.0, 0.0]
    )


def test_double_difference_keeps_gaps_without_filling() -> None:
    obs = _s([14.0, np.nan, 13.0])
    mod = _s([8.0, 9.0, np.nan])
    dd = compute_double_difference(obs, mod)
    assert dd.isna().tolist() == [False, True, True]
    assert dd.iloc[0] == pytest.approx(6.0)


def test_double_difference_restricts_to_common_period() -> None:
    obs = _s([10.0, 11.0, 12.0, 13.0], start="1990-01-01")
    mod = _s([5.0, 5.0], start="1990-03-01")
    dd = compute_double_difference(obs, mod)
    assert list(dd.index) == list(pd.date_range("1990-03-01", periods=2, freq="1MS"))
    assert dd.tolist() == pytest.approx([7.0, 8.0])


def test_double_difference_works_with_extreme_range_outputs() -> None:
    obs = compute_monthly_extreme_range(_s([30.0, 31.0]), _s([10.0, 11.0]))
    mod = compute_monthly_extreme_range(_s([27.0, 28.0]), _s([13.0, 14.0]))
    dd = compute_double_difference(obs, mod)
    assert dd.tolist() == pytest.approx([6.0, 6.0])


def test_double_difference_no_overlap_raises() -> None:
    with pytest.raises(ValueError, match="no common timestamps"):
        compute_double_difference(_s([1.0], "1990-01-01"), _s([1.0], "2010-01-01"))


def test_double_difference_requires_datetime_index() -> None:
    with pytest.raises(ValueError, match="DatetimeIndex"):
        compute_double_difference(pd.Series([1.0]), _s([1.0]))
