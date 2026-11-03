"""Benchmark evaluator helpers under Protocol Appendix J and K.

Provides pure helpers for the benchmark evaluator:
- Valid-year rule of J.7.1
- Calendar-year masking to valid years
- Ledger-free 5-year block partitioning
- J.7.2 sample status classification
"""

from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from tropicor.core.protocol import MIN_BOOTSTRAP_BLOCKS

BENCHMARK_STATION_IDS: Tuple[str, str] = ("21205710", "23215030")


def valid_years_j71(
    tmax_obs: pd.Series,
    tmin_obs: pd.Series,
    tmax_mod: pd.Series,
    tmin_mod: pd.Series,
) -> List[int]:
    """Return sorted calendar years containing at least one valid day.

    Under Protocol J.7.1, a calendar year is valid if and only if there is at least
    one calendar day on which all four temperature series (tmax_obs, tmin_obs,
    tmax_mod, tmin_mod) are simultaneously non-NaN.

    Parameters
    ----------
    tmax_obs : pd.Series
        Observed daily maximum temperature series.
    tmin_obs : pd.Series
        Observed daily minimum temperature series.
    tmax_mod : pd.Series
        Model/predicted daily maximum temperature series.
    tmin_mod : pd.Series
        Model/predicted daily minimum temperature series.

    Returns
    -------
    List[int]
        Sorted list of valid calendar years.

    Raises
    ------
    ValueError
        If series indexes are not pd.DatetimeIndex or are not identical.
    """
    for s, name in (
        (tmax_obs, "tmax_obs"),
        (tmin_obs, "tmin_obs"),
        (tmax_mod, "tmax_mod"),
        (tmin_mod, "tmin_mod"),
    ):
        if not isinstance(s.index, pd.DatetimeIndex):
            raise ValueError(f"{name} index must be a pd.DatetimeIndex.")

    base_index = tmax_obs.index
    if not (
        tmin_obs.index.equals(base_index)
        and tmax_mod.index.equals(base_index)
        and tmin_mod.index.equals(base_index)
    ):
        raise ValueError("All four series must have identical indexes.")
    assert isinstance(base_index, pd.DatetimeIndex)

    valid_mask = (
        tmax_obs.notna() & tmin_obs.notna() & tmax_mod.notna() & tmin_mod.notna()
    )
    if not valid_mask.any():
        return []

    valid_years = sorted(int(y) for y in set(base_index[valid_mask].year))
    return valid_years


def mask_to_valid_years(
    tmax_obs: pd.Series,
    tmin_obs: pd.Series,
    tmax_mod: pd.Series,
    tmin_mod: pd.Series,
    valid_years: Sequence[int],
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return copies of the 4 series with non-valid calendar years masked to NaN.

    Inputs are never modified.

    Parameters
    ----------
    tmax_obs : pd.Series
        Observed daily maximum temperature series.
    tmin_obs : pd.Series
        Observed daily minimum temperature series.
    tmax_mod : pd.Series
        Model/predicted daily maximum temperature series.
    tmin_mod : pd.Series
        Model/predicted daily minimum temperature series.
    valid_years : Sequence[int]
        Sequence of valid calendar years to preserve.

    Returns
    -------
    Tuple[pd.Series, pd.Series, pd.Series, pd.Series]
        Copies of the 4 series with values in non-valid years set to NaN.

    Raises
    ------
    ValueError
        If series indexes are not pd.DatetimeIndex or are not identical.
    """
    for s, name in (
        (tmax_obs, "tmax_obs"),
        (tmin_obs, "tmin_obs"),
        (tmax_mod, "tmax_mod"),
        (tmin_mod, "tmin_mod"),
    ):
        if not isinstance(s.index, pd.DatetimeIndex):
            raise ValueError(f"{name} index must be a pd.DatetimeIndex.")

    base_index = tmax_obs.index
    if not (
        tmin_obs.index.equals(base_index)
        and tmax_mod.index.equals(base_index)
        and tmin_mod.index.equals(base_index)
    ):
        raise ValueError("All four series must have identical indexes.")
    assert isinstance(base_index, pd.DatetimeIndex)

    valid_set = set(valid_years)
    invalid_mask = ~base_index.year.isin(valid_set)

    s1 = tmax_obs.copy()
    s2 = tmin_obs.copy()
    s3 = tmax_mod.copy()
    s4 = tmin_mod.copy()

    if invalid_mask.any():
        s1.loc[invalid_mask] = np.nan
        s2.loc[invalid_mask] = np.nan
        s3.loc[invalid_mask] = np.nan
        s4.loc[invalid_mask] = np.nan

    return s1, s2, s3, s4


def five_year_blocks(years: Sequence[int]) -> List[Tuple[int, int]]:
    """Partition calendar years into contiguous 5-year blocks.

    Follows the same block algorithm as contiguous_5year_block_cv in
    protocol.py (blocks of 5 from the first year; when remaining years
    number fewer than 10 and are not exactly 5, the last block takes all
    of them), but with no validation ledger access.

    Parameters
    ----------
    years : Sequence[int]
        Sequence of calendar years.

    Returns
    -------
    List[Tuple[int, int]]
        List of (first_year, last_year) tuples defining the blocks.

    Raises
    ------
    ValueError
        If fewer than 5 unique years are provided.
    """
    sorted_years = sorted(set(years))
    n_years = len(sorted_years)
    if n_years < 5:
        raise ValueError(
            f"5-year block partition requires at least 5 years, found {n_years}."
        )

    blocks: List[Tuple[int, int]] = []
    i = 0
    while i < n_years:
        rem = n_years - i
        if rem < 10 and rem != 5:
            block = sorted_years[i:]
            blocks.append((min(block), max(block)))
            break
        else:
            block = sorted_years[i : i + 5]
            blocks.append((min(block), max(block)))
            i += 5

    return blocks


def track_sample_status(n_valid_years: int) -> str:
    """Classify sample status under Protocol J.7.2.

    Parameters
    ----------
    n_valid_years : int
        Number of valid calendar years under J.7.1.

    Returns
    -------
    str
        "ok" if n_valid_years >= MIN_BOOTSTRAP_BLOCKS, else
        "not_established_insufficient_sample".
    """
    if n_valid_years >= MIN_BOOTSTRAP_BLOCKS:
        return "ok"
    return "not_established_insufficient_sample"
