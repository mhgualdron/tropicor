"""Thermal range diagnostics: diurnal temperature range and model double difference.

Two physically different quantities are kept strictly apart:

* **Mean diurnal temperature range (DTR)**: the monthly mean of the daily
  ``Tmax - Tmin``. It requires monthly means of the daily maximum and daily
  minimum temperature.
* **Monthly extreme range**: the difference between the single highest and the
  single lowest temperature recorded in the month. It is computed from absolute
  monthly extremes, which is what monthly IDEAM DHIME temperature sheets report.
  For the same month it is always greater than or equal to the mean DTR, and it
  must never be called DTR.

No function here can tell from the numbers which kind of input it was given.
Choosing the right function for the available series is the caller's
responsibility.
"""

import warnings

import pandas as pd

from tropicor.core.alignment import align_common_period


def _range_between(
    upper: pd.Series,
    lower: pd.Series,
    *,
    upper_name: str,
    lower_name: str,
    output_name: str,
) -> pd.Series:
    """Compute ``upper - lower`` on common timestamps with safe coercion.

    Timesteps where ``lower > upper`` are physically inconsistent (typically
    typing errors in historical records). They are set to NaN and reported with
    a single warning, so one bad month does not discard the whole series.

    Raises:
        ValueError: If an input is empty, the series share no timestamps, or no
            valid timestep remains after coercion.
    """
    if len(upper) == 0 or len(lower) == 0:
        raise ValueError(f"'{upper_name}' and '{lower_name}' must not be empty.")

    up, lo = align_common_period(upper, lower, drop_na=False)

    both_valid = up.notna() & lo.notna()
    inverted = both_valid & (lo > up)
    n_inverted = int(inverted.sum())

    result = (up - lo).mask(inverted)

    if n_inverted > 0:
        pct = 100.0 * n_inverted / int(both_valid.sum())
        warnings.warn(
            f"Detected {n_inverted} timesteps ({pct:.1f}% of valid pairs) where "
            f"{lower_name} > {upper_name}; set to NaN.",
            UserWarning,
            stacklevel=3,
        )

    if int(result.notna().sum()) == 0:
        raise ValueError(
            f"No valid timesteps remain between '{upper_name}' and '{lower_name}' "
            "(all values are missing or inconsistent)."
        )

    result.name = output_name
    return result


def compute_dtr(tmax_mean: pd.Series, tmin_mean: pd.Series) -> pd.Series:
    """Compute the mean monthly diurnal temperature range (DTR).

    DTR is the monthly mean of the daily range:

        DTR = mean(Tmax_daily) - mean(Tmin_daily)

    Note:
        Inputs must be monthly means of the daily maximum and daily minimum
        temperature. This function cannot verify that: if absolute monthly
        extremes are passed instead, the result is a monthly extreme range, not
        a DTR. For that case use :func:`compute_monthly_extreme_range`.

    Args:
        tmax_mean: Monthly mean of daily maximum temperature (DatetimeIndex).
        tmin_mean: Monthly mean of daily minimum temperature (DatetimeIndex).

    Returns:
        Series named ``"dtr"`` on the timestamps both inputs share. Timesteps
        where either input is NaN are NaN. Timesteps where ``tmin_mean`` is
        greater than ``tmax_mean`` are set to NaN, with one ``UserWarning``
        reporting how many.

    Raises:
        ValueError: If an input is empty, lacks a DatetimeIndex, the inputs
            share no timestamps, or no valid timestep remains.
    """
    return _range_between(
        tmax_mean,
        tmin_mean,
        upper_name="tmax_mean",
        lower_name="tmin_mean",
        output_name="dtr",
    )


def compute_monthly_extreme_range(
    tmax_abs: pd.Series, tmin_abs: pd.Series
) -> pd.Series:
    """Compute the monthly extreme temperature range.

    This is the difference between the highest and the lowest temperature
    recorded in each month:

        extreme range = Tmax_absolute - Tmin_absolute

    Note:
        This is **not** the diurnal temperature range. For the same month it is
        greater than or equal to the mean DTR, because a single warmest reading
        and a single coldest reading bound every daily range. Use it only with
        absolute monthly extremes, such as the monthly maximum and minimum
        temperature series of IDEAM DHIME.

    Args:
        tmax_abs: Absolute monthly maximum temperature (DatetimeIndex).
        tmin_abs: Absolute monthly minimum temperature (DatetimeIndex).

    Returns:
        Series named ``"extreme_range"`` on the timestamps both inputs share.
        Timesteps where either input is NaN are NaN. Timesteps where
        ``tmin_abs`` is greater than ``tmax_abs`` are set to NaN, with one
        ``UserWarning`` reporting how many.

    Raises:
        ValueError: If an input is empty, lacks a DatetimeIndex, the inputs
            share no timestamps, or no valid timestep remains.
    """
    return _range_between(
        tmax_abs,
        tmin_abs,
        upper_name="tmax_abs",
        lower_name="tmin_abs",
        output_name="extreme_range",
    )


def compute_double_difference(obs_range: pd.Series, mod_range: pd.Series) -> pd.Series:
    """Compute the double difference between observed and modeled thermal range.

    The double difference compares the range of the observation with the range
    of the model, instead of comparing temperatures directly:

        double difference = range_obs - range_mod

    A positive value means the model has a smaller thermal range than the
    station, that is, it underestimates the amplitude. A negative value means
    it overestimates it. Because each range is a difference of two
    temperatures, a constant offset between model and station cancels out.

    Note:
        Both inputs must be the same kind of range, for example the DTR of the
        station and the DTR of the model, or the extreme range of each. Mixing a
        DTR with an extreme range is meaningless, and this function cannot
        detect it.

    Args:
        obs_range: Observed thermal range (DatetimeIndex).
        mod_range: Modeled thermal range of the same kind (DatetimeIndex).

    Returns:
        Series named ``"double_difference"`` on the common timestamps. Months
        missing in either input are NaN; no value is filled or interpolated.

    Raises:
        ValueError: If an input lacks a DatetimeIndex, has duplicated
            timestamps, or the inputs share no timestamps.
    """
    obs, mod = align_common_period(obs_range, mod_range, drop_na=False)
    result = obs - mod
    result.name = "double_difference"
    return result
