"""Temporal alignment of observation and model series (Layer 0 foundation).

Comparing climatologies, thermal ranges or error statistics is only meaningful
when both series describe exactly the same months. This module provides the
single place where that synchronization is defined, so higher layers
(climatology, thermal range, bias correction) share one implementation.
"""

from typing import Tuple

import pandas as pd


def _validate_series(name: str, series: pd.Series) -> None:
    """Check that a series has a unique DatetimeIndex."""
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError(f"'{name}' must have a pandas DatetimeIndex.")
    if series.index.has_duplicates:
        raise ValueError(
            f"'{name}' has duplicated timestamps; deduplicate before aligning."
        )


def align_common_period(
    obs: pd.Series,
    mod: pd.Series,
    drop_na: bool = True,
) -> Tuple[pd.Series, pd.Series]:
    """Synchronize observation and model series on their common valid period.

    Applies pairwise synchronous masking without any temporal imputation:

    1. Both series are restricted to the intersection of their timestamps.
    2. The model is masked to NaN wherever the observation is NaN, and the
       observation is masked wherever the model is NaN.
    3. Gaps are never filled or interpolated.

    Args:
        obs: Ground-truth observation series with a DatetimeIndex.
        mod: Model or reanalysis series with a DatetimeIndex.
        drop_na: If True (default), timestamps where either series is NaN are
            removed, so only synchronous valid pairs are returned. If False,
            the full timestamp intersection is kept and both series share
            identical NaN positions.

    Returns:
        Tuple ``(aligned_obs, aligned_mod)`` with identical, sorted indexes.
        With ``drop_na=True`` the result may be empty if the series overlap in
        time but never have a simultaneous valid value.

    Raises:
        ValueError: If either series lacks a DatetimeIndex, has duplicated
            timestamps, or if the two indexes share no timestamps at all.
    """
    _validate_series("obs", obs)
    _validate_series("mod", mod)

    common = obs.index.intersection(mod.index).sort_values()
    if len(common) == 0:
        raise ValueError("Observation and model series share no common timestamps.")

    obs_c = obs.loc[common].astype("float64")
    mod_c = mod.loc[common].astype("float64")

    invalid = obs_c.isna() | mod_c.isna()
    obs_c = obs_c.mask(invalid)
    mod_c = mod_c.mask(invalid)

    if drop_na:
        obs_c = obs_c[~invalid]
        mod_c = mod_c[~invalid]

    return obs_c, mod_c
