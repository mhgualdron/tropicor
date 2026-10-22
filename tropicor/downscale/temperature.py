"""Daily temperature downscaling and bias correction engine (Protocol Levels 1-3).

Implements the pre-specified temperature downscaling hierarchy from Section 6.2
of docs/science/evaluation-protocol.md:
- Level 1: Independent monthly additive bias correction for Tmax and Tmin.
- Level 2: Mean state monthly additive bias correction + Diurnal Temperature Range
  (DTR) monthly multiplicative scaling, with daily extremes reconstruction.
- Level 3: Non-parametric empirical quantile mapping (EQM) with deterministic
  physical reordering via the universal crossing rule (Section 3.5).
- Nested annual block cross-validation hyperparameter tuning strictly within the
  calibration partition (Section 4.1), minimizing temperature_tuning_loss.
- Calibration-only data loading with immediate validation partition discarding.
"""

from __future__ import annotations

import glob
import re
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple, cast

import numpy as np
import pandas as pd

from tropicor.core.andean_registry import AndeanStation, get_andean_station
from tropicor.core.protocol import (
    CrossingDiagnostic,
    apply_universal_crossing_rule,
    get_benchmark_partitions,
    temperature_tuning_loss,
)
from tropicor.downscale.precipitation import date_to_leap_doy

# Physically plausible range for surface air temperature in terrestrial climates
MIN_PLAUSIBLE_TEMP_C: float = -60.0
MAX_PLAUSIBLE_TEMP_C: float = 60.0

MIN_SAMPLES_PER_GROUP: int = 10
"""Minimum number of valid paired observation/model samples required per group
(month or DOY window) to fit empirical quantile nodes."""

# Pre-specified Level 3 hyperparameter configurations (Section 6.3)
# 2 window definitions x 2 quantile bin sets x 2 extrapolation schemes = 8 configs
LEVEL3_TEMPERATURE_CONFIGS: List[Dict[str, Any]] = [
    {
        "config_id": 0,
        "grouping": "monthly",
        "n_quantiles": 50,
        "extrapolation": "constant",
    },
    {
        "config_id": 1,
        "grouping": "monthly",
        "n_quantiles": 50,
        "extrapolation": "linear",
    },
    {
        "config_id": 2,
        "grouping": "monthly",
        "n_quantiles": 100,
        "extrapolation": "constant",
    },
    {
        "config_id": 3,
        "grouping": "monthly",
        "n_quantiles": 100,
        "extrapolation": "linear",
    },
    {
        "config_id": 4,
        "grouping": "moving_window",
        "n_quantiles": 50,
        "extrapolation": "constant",
    },
    {
        "config_id": 5,
        "grouping": "moving_window",
        "n_quantiles": 50,
        "extrapolation": "linear",
    },
    {
        "config_id": 6,
        "grouping": "moving_window",
        "n_quantiles": 100,
        "extrapolation": "constant",
    },
    {
        "config_id": 7,
        "grouping": "moving_window",
        "n_quantiles": 100,
        "extrapolation": "linear",
    },
]


def _validate_temperature_range(series: pd.Series, name: str) -> None:
    """Validate that non-null temperature values lie within [-60, 60] °C."""
    valid = series.dropna()
    if valid.empty:
        return
    min_val = float(valid.min())
    max_val = float(valid.max())
    if min_val < MIN_PLAUSIBLE_TEMP_C or max_val > MAX_PLAUSIBLE_TEMP_C:
        raise ValueError(
            f"Temperature series '{name}' has values outside the physically plausible "
            f"range [{MIN_PLAUSIBLE_TEMP_C}, {MAX_PLAUSIBLE_TEMP_C}] °C: "
            f"min={min_val:.2f} °C, max={max_val:.2f} °C."
        )


def _validate_datetime_indices_equal(
    *series_list: pd.Series,
    names: Sequence[str],
) -> None:
    """Validate that all input series have identical pd.DatetimeIndex."""
    for s, name in zip(series_list, names):
        if not isinstance(s.index, pd.DatetimeIndex):
            raise ValueError(f"Series '{name}' must have a pd.DatetimeIndex.")
    first = series_list[0]
    for s, name in zip(series_list[1:], names[1:]):
        if not first.index.equals(s.index):
            raise ValueError(f"DatetimeIndex of '{name}' does not match '{names[0]}'.")


class BaseTemperatureDownscaler(ABC):
    """Abstract base class for temperature downscalers."""

    def __init__(self) -> None:
        self.last_crossing_diagnostic_: Optional[CrossingDiagnostic] = None
        self.n_crossings_before_rule_: int = 0
        self.n_crossings_after_rule_: int = 0

    @abstractmethod
    def fit(
        self,
        obs_tmax: pd.Series,
        obs_tmin: pd.Series,
        mod_tmax: pd.Series,
        mod_tmin: pd.Series,
    ) -> BaseTemperatureDownscaler:
        """Fit downscaling parameters on calibration paired series."""

    @abstractmethod
    def transform(
        self,
        mod_tmax: pd.Series,
        mod_tmin: pd.Series,
    ) -> Tuple[pd.Series, pd.Series]:
        """Apply fitted correction to modeled temperature series."""


class Level1MonthlyAdditiveTemperature(BaseTemperatureDownscaler):
    """Level 1 Mean State Correction: Independent Monthly Additive Bias Correction.

    Protocol Section 6.2:
    - Fits independent monthly additive biases for Tmax and Tmin:
        Bias_m(Tmax) = mean(Tmax_mod,m) - mean(Tmax_obs,m)
        Bias_m(Tmin) = mean(Tmin_mod,m) - mean(Tmin_obs,m)
    - Applied as:
        Tmax_corr = Tmax_mod - Bias_m(Tmax)
        Tmin_corr = Tmin_mod - Bias_m(Tmin)
    - Followed by the universal crossing rule (Section 3.5).
    """

    def __init__(self) -> None:
        super().__init__()
        self.tmax_biases_: Dict[int, float] = {}
        self.tmin_biases_: Dict[int, float] = {}

    def fit(
        self,
        obs_tmax: pd.Series,
        obs_tmin: pd.Series,
        mod_tmax: pd.Series,
        mod_tmin: pd.Series,
    ) -> Level1MonthlyAdditiveTemperature:
        """Estimate independent monthly additive biases."""
        _validate_datetime_indices_equal(
            obs_tmax,
            obs_tmin,
            mod_tmax,
            mod_tmin,
            names=["obs_tmax", "obs_tmin", "mod_tmax", "mod_tmin"],
        )
        _validate_temperature_range(obs_tmax, "obs_tmax")
        _validate_temperature_range(obs_tmin, "obs_tmin")
        _validate_temperature_range(mod_tmax, "mod_tmax")
        _validate_temperature_range(mod_tmin, "mod_tmin")

        idx = cast(pd.DatetimeIndex, obs_tmax.index)
        self.tmax_biases_ = {}
        self.tmin_biases_ = {}

        for m in range(1, 13):
            # Tmax mask and bias (variable-by-variable paired mask)
            m_mask_tmax = (idx.month == m) & obs_tmax.notna() & mod_tmax.notna()
            if m_mask_tmax.sum() == 0:
                raise ValueError(
                    f"Month {m} has zero valid observation/model pairs for Tmax."
                )
            bias_tmax = float(
                mod_tmax[m_mask_tmax].mean() - obs_tmax[m_mask_tmax].mean()
            )
            self.tmax_biases_[m] = bias_tmax

            # Tmin mask and bias (variable-by-variable paired mask)
            m_mask_tmin = (idx.month == m) & obs_tmin.notna() & mod_tmin.notna()
            if m_mask_tmin.sum() == 0:
                raise ValueError(
                    f"Month {m} has zero valid observation/model pairs for Tmin."
                )
            bias_tmin = float(
                mod_tmin[m_mask_tmin].mean() - obs_tmin[m_mask_tmin].mean()
            )
            self.tmin_biases_[m] = bias_tmin

        return self

    def transform(
        self,
        mod_tmax: pd.Series,
        mod_tmin: pd.Series,
    ) -> Tuple[pd.Series, pd.Series]:
        """Apply monthly additive corrections and universal crossing rule."""
        if not self.tmax_biases_ or not self.tmin_biases_:
            raise ValueError("Downscaler must be fitted before calling transform.")

        _validate_datetime_indices_equal(
            mod_tmax, mod_tmin, names=["mod_tmax", "mod_tmin"]
        )
        _validate_temperature_range(mod_tmax, "mod_tmax")
        _validate_temperature_range(mod_tmin, "mod_tmin")

        idx = cast(pd.DatetimeIndex, mod_tmax.index)
        months = idx.month

        tmax_corr = mod_tmax.copy()
        tmin_corr = mod_tmin.copy()

        for m in range(1, 13):
            m_mask = months == m
            if m_mask.any():
                tmax_corr.loc[m_mask] = tmax_corr.loc[m_mask] - self.tmax_biases_[m]
                tmin_corr.loc[m_mask] = tmin_corr.loc[m_mask] - self.tmin_biases_[m]

        # Count crossings before universal crossing rule
        valid_both = tmax_corr.notna() & tmin_corr.notna()
        raw_crossing_mask = valid_both & (tmin_corr > tmax_corr)
        self.n_crossings_before_rule_ = int(raw_crossing_mask.sum())

        # Apply universal crossing rule where both series are valid
        final_tmax, final_tmin, diag = apply_universal_crossing_rule(
            tmax_corr, tmin_corr
        )
        self.last_crossing_diagnostic_ = diag
        valid_final = final_tmax.notna() & final_tmin.notna()
        self.n_crossings_after_rule_ = int(
            (valid_final & (final_tmin > final_tmax)).sum()
        )

        final_tmax.name = "tmax_corrected"
        final_tmin.name = "tmin_corrected"
        return final_tmax, final_tmin


class Level2MeanStateDTRScalingTemperature(BaseTemperatureDownscaler):
    """Level 2 Diurnal Structure Correction: Mean State + DTR Scaling.

    Protocol Section 6.2:
    - Decomposes daily extremes into daily mean temperature and DTR:
        Tmean(t) = (Tmax(t) + Tmin(t)) / 2
        DTR(t)   = Tmax(t) - Tmin(t)
    - Four-value paired mask: only days where all four (obs/mod x tmax/tmin)
      are non-null enter the monthly estimation.
    - Estimates:
        Tmean_bias_m = mean(Tmean_mod,m) - mean(Tmean_obs,m)
        s_dtr_m      = mean(DTR_obs,m) / mean(DTR_mod,m)
    - If mean modeled DTR in any month is <= 1e-4 °C, raises ValueError.
    - Reconstruction:
        Tmean_corr(t) = Tmean_mod(t) - Tmean_bias_m
        DTR_corr(t)   = DTR_mod(t) * s_dtr_m
        Tmax_corr(t)  = Tmean_corr(t) + DTR_corr(t) / 2
        Tmin_corr(t)  = Tmean_corr(t) - DTR_corr(t) / 2
    - Followed by the universal crossing rule (Section 3.5).
    """

    def __init__(self) -> None:
        super().__init__()
        self.tmean_biases_: Dict[int, float] = {}
        self.dtr_scaling_factors_: Dict[int, float] = {}

    def fit(
        self,
        obs_tmax: pd.Series,
        obs_tmin: pd.Series,
        mod_tmax: pd.Series,
        mod_tmin: pd.Series,
    ) -> Level2MeanStateDTRScalingTemperature:
        """Estimate monthly mean Tmean bias and DTR scaling factors."""
        _validate_datetime_indices_equal(
            obs_tmax,
            obs_tmin,
            mod_tmax,
            mod_tmin,
            names=["obs_tmax", "obs_tmin", "mod_tmax", "mod_tmin"],
        )
        _validate_temperature_range(obs_tmax, "obs_tmax")
        _validate_temperature_range(obs_tmin, "obs_tmin")
        _validate_temperature_range(mod_tmax, "mod_tmax")
        _validate_temperature_range(mod_tmin, "mod_tmin")

        idx = cast(pd.DatetimeIndex, obs_tmax.index)
        self.tmean_biases_ = {}
        self.dtr_scaling_factors_ = {}

        # 4-value paired mask: all four values must be valid simultaneously
        paired_mask = (
            obs_tmax.notna() & obs_tmin.notna() & mod_tmax.notna() & mod_tmin.notna()
        )

        obs_tmean = (obs_tmax + obs_tmin) / 2.0
        obs_dtr = obs_tmax - obs_tmin

        mod_tmean = (mod_tmax + mod_tmin) / 2.0
        mod_dtr = mod_tmax - mod_tmin

        for m in range(1, 13):
            m_mask = (idx.month == m) & paired_mask
            if m_mask.sum() == 0:
                raise ValueError(
                    f"Month {m} has zero valid observation/model pairs with all "
                    "4 values valid."
                )

            mean_mod_dtr = float(mod_dtr[m_mask].mean())
            if mean_mod_dtr <= 1e-4:
                raise ValueError(
                    f"Level 2 DTR scaling undefined: modeled mean DTR in month {m} "
                    f"is {mean_mod_dtr:.6f} °C <= 1e-4 °C."
                )

            mean_obs_dtr = float(obs_dtr[m_mask].mean())
            s_dtr = mean_obs_dtr / mean_mod_dtr

            mean_mod_tmean = float(mod_tmean[m_mask].mean())
            mean_obs_tmean = float(obs_tmean[m_mask].mean())
            bias_tmean = mean_mod_tmean - mean_obs_tmean

            self.tmean_biases_[m] = bias_tmean
            self.dtr_scaling_factors_[m] = s_dtr

        return self

    def transform(
        self,
        mod_tmax: pd.Series,
        mod_tmin: pd.Series,
    ) -> Tuple[pd.Series, pd.Series]:
        """Apply Level 2 decomposition, scaling, and reconstruction."""
        if not self.tmean_biases_ or not self.dtr_scaling_factors_:
            raise ValueError("Downscaler must be fitted before calling transform.")

        _validate_datetime_indices_equal(
            mod_tmax, mod_tmin, names=["mod_tmax", "mod_tmin"]
        )
        _validate_temperature_range(mod_tmax, "mod_tmax")
        _validate_temperature_range(mod_tmin, "mod_tmin")

        idx = cast(pd.DatetimeIndex, mod_tmax.index)
        months = idx.month

        mod_tmean = (mod_tmax + mod_tmin) / 2.0
        mod_dtr = mod_tmax - mod_tmin

        tmean_corr = pd.Series(np.nan, index=idx, dtype=float)
        dtr_corr = pd.Series(np.nan, index=idx, dtype=float)

        for m in range(1, 13):
            m_mask = months == m
            if m_mask.any():
                tmean_corr.loc[m_mask] = mod_tmean.loc[m_mask] - self.tmean_biases_[m]
                dtr_corr.loc[m_mask] = (
                    mod_dtr.loc[m_mask] * self.dtr_scaling_factors_[m]
                )

        tmax_corr = tmean_corr + dtr_corr / 2.0
        tmin_corr = tmean_corr - dtr_corr / 2.0

        # NaN propagation: if either modeled tmax or tmin is NaN, both outputs are NaN
        nan_either = mod_tmax.isna() | mod_tmin.isna()
        tmax_corr.loc[nan_either] = np.nan
        tmin_corr.loc[nan_either] = np.nan

        # Count crossings before universal crossing rule
        valid_both = tmax_corr.notna() & tmin_corr.notna()
        raw_crossing_mask = valid_both & (tmin_corr > tmax_corr)
        self.n_crossings_before_rule_ = int(raw_crossing_mask.sum())

        # Apply universal crossing rule where both series are valid
        final_tmax, final_tmin, diag = apply_universal_crossing_rule(
            tmax_corr, tmin_corr
        )
        self.last_crossing_diagnostic_ = diag
        valid_final = final_tmax.notna() & final_tmin.notna()
        self.n_crossings_after_rule_ = int(
            (valid_final & (final_tmin > final_tmax)).sum()
        )

        final_tmax.name = "tmax_corrected"
        final_tmin.name = "tmin_corrected"
        return final_tmax, final_tmin


def _doy_window_mask(
    leap_doys: np.ndarray,
    target_doy: int,
    half_window: int = 15,
) -> np.ndarray:
    """Return boolean mask for leap_doys within circular distance of target_doy.

    Protocol Section 6.3 & D.14:
    Circular distance along reference leap calendar (366 days):
      Delta(d1, d2) = min(|d1 - d2|, 366 - |d1 - d2|)
      window_mask = Delta <= half_window
    """
    diff = np.abs(leap_doys - target_doy)
    circ_dist = np.minimum(diff, 366 - diff)
    return circ_dist <= half_window


class Level3QuantileMappingTemperature(BaseTemperatureDownscaler):
    """Level 3 Empirical Quantile Mapping (EQM) for daily temperature extremes.

    Protocol Section 6.3:
    1. Grouping: Calendar monthly window vs. 31-day circular moving window centered
       on day of year (leap calendar 1..366).
    2. Quantile Bins:
       - 50 bins: 49 empirical quantile nodes at 100*k/50 for k=1..49
         (2%, 4%, ..., 98%).
       - 100 bins: 99 empirical quantile nodes at 100*k/100 for k=1..99
         (1%, 2%, ..., 99%).
    3. Extrapolation outside calibration range:
       - 'constant': Constant residual anomaly extension (slope 1.0):
           T_corr = Q_obs,K + (T - Q_mod,K) for T > Q_mod,K
           T_corr = Q_obs,1 + (T - Q_mod,1) for T < Q_mod,1
       - 'linear': Linear extension of the transfer function
         (slope s from extreme nodes):
           s_upper = (Q_obs,K - Q_obs,K-1) / (Q_mod,K - Q_mod,K-1)
           s_lower = (Q_obs,2 - Q_obs,1) / (Q_mod,2 - Q_mod,1)
           T_corr = Q_obs,K + s_upper * (T - Q_mod,K) for T > Q_mod,K
           T_corr = Q_obs,1 + s_lower * (T - Q_mod,1) for T < Q_mod,1
           If extreme model interval (denominator) <= 1e-9, falls back to
           constant residual (s = 1.0).
           No clipping of s is applied.
    Followed by the universal deterministic crossing rule (Section 3.5).
    """

    def __init__(
        self,
        grouping: Literal["monthly", "moving_window"] = "monthly",
        n_quantiles: int = 50,
        extrapolation: Literal["constant", "linear"] = "constant",
    ) -> None:
        super().__init__()
        if grouping not in ("monthly", "moving_window"):
            raise ValueError(
                f"Invalid grouping '{grouping}'. Must be 'monthly' or 'moving_window'."
            )
        if n_quantiles not in (50, 100):
            raise ValueError(
                f"Invalid n_quantiles {n_quantiles}. Must be 50 or 100 (Section 6.3)."
            )
        if extrapolation not in ("constant", "linear"):
            raise ValueError(
                f"Invalid extrapolation '{extrapolation}'. Must be 'constant' or "
                "'linear'."
            )

        self.grouping = grouping
        self.n_quantiles = n_quantiles
        self.extrapolation = extrapolation

        # Fitted quantile nodes and tail slopes per group key (month or leap doy)
        self.tmax_quantiles_mod_: Dict[int, np.ndarray] = {}
        self.tmax_quantiles_obs_: Dict[int, np.ndarray] = {}
        self.tmax_slopes_lower_: Dict[int, float] = {}
        self.tmax_slopes_upper_: Dict[int, float] = {}

        self.tmin_quantiles_mod_: Dict[int, np.ndarray] = {}
        self.tmin_quantiles_obs_: Dict[int, np.ndarray] = {}
        self.tmin_slopes_lower_: Dict[int, float] = {}
        self.tmin_slopes_upper_: Dict[int, float] = {}

        self.min_n_per_group: Optional[int] = None
        self.n_extrapolated_days_: int = 0
        self.max_abs_extrapolation_correction_: float = 0.0

    @staticmethod
    def _compute_slopes(
        q_mod: np.ndarray,
        q_obs: np.ndarray,
        extrapolation: str,
    ) -> Tuple[float, float]:
        """Compute lower and upper tail extrapolation slopes."""
        if extrapolation == "constant":
            return 1.0, 1.0

        # Linear transfer function extension
        denom_lower = float(q_mod[1] - q_mod[0])
        num_lower = float(q_obs[1] - q_obs[0])
        s_lower = num_lower / denom_lower if denom_lower > 1e-9 else 1.0

        denom_upper = float(q_mod[-1] - q_mod[-2])
        num_upper = float(q_obs[-1] - q_obs[-2])
        s_upper = num_upper / denom_upper if denom_upper > 1e-9 else 1.0

        return s_lower, s_upper

    def fit(
        self,
        obs_tmax: pd.Series,
        obs_tmin: pd.Series,
        mod_tmax: pd.Series,
        mod_tmin: pd.Series,
    ) -> Level3QuantileMappingTemperature:
        """Estimate empirical quantile mapping nodes and tail slopes per group."""
        _validate_datetime_indices_equal(
            obs_tmax,
            obs_tmin,
            mod_tmax,
            mod_tmin,
            names=["obs_tmax", "obs_tmin", "mod_tmax", "mod_tmin"],
        )
        _validate_temperature_range(obs_tmax, "obs_tmax")
        _validate_temperature_range(obs_tmin, "obs_tmin")
        _validate_temperature_range(mod_tmax, "mod_tmax")
        _validate_temperature_range(mod_tmin, "mod_tmin")

        # 49 nodes for 50 bins; 99 nodes for 100 bins (Protocol Section 6.3 & D.6)
        if self.n_quantiles == 50:
            quantiles = np.arange(1, 50) * (100.0 / 50.0)  # 2, 4, ..., 98 (49 nodes)
        else:
            quantiles = np.arange(1, 100) * (100.0 / 100.0)  # 1, 2, ..., 99 (99 nodes)

        idx = cast(pd.DatetimeIndex, obs_tmax.index)

        self.tmax_quantiles_mod_ = {}
        self.tmax_quantiles_obs_ = {}
        self.tmax_slopes_lower_ = {}
        self.tmax_slopes_upper_ = {}

        self.tmin_quantiles_mod_ = {}
        self.tmin_quantiles_obs_ = {}
        self.tmin_slopes_lower_ = {}
        self.tmin_slopes_upper_ = {}

        sample_counts: List[int] = []

        if self.grouping == "monthly":
            for m in range(1, 13):
                # Tmax
                mask_tmax = (idx.month == m) & obs_tmax.notna() & mod_tmax.notna()
                sample_counts.append(int(mask_tmax.sum()))
                if mask_tmax.sum() < MIN_SAMPLES_PER_GROUP:
                    raise ValueError(
                        f"Month {m} has insufficient valid samples for Tmax EQM: "
                        f"{mask_tmax.sum()} (< {MIN_SAMPLES_PER_GROUP})."
                    )
                q_mod_tmax = np.percentile(
                    mod_tmax[mask_tmax].to_numpy(), quantiles, method="linear"
                )
                q_obs_tmax = np.percentile(
                    obs_tmax[mask_tmax].to_numpy(), quantiles, method="linear"
                )
                s_low_max, s_up_max = self._compute_slopes(
                    q_mod_tmax, q_obs_tmax, self.extrapolation
                )

                self.tmax_quantiles_mod_[m] = q_mod_tmax
                self.tmax_quantiles_obs_[m] = q_obs_tmax
                self.tmax_slopes_lower_[m] = s_low_max
                self.tmax_slopes_upper_[m] = s_up_max

                # Tmin
                mask_tmin = (idx.month == m) & obs_tmin.notna() & mod_tmin.notna()
                sample_counts.append(int(mask_tmin.sum()))
                if mask_tmin.sum() < MIN_SAMPLES_PER_GROUP:
                    raise ValueError(
                        f"Month {m} has insufficient valid samples for Tmin EQM: "
                        f"{mask_tmin.sum()} (< {MIN_SAMPLES_PER_GROUP})."
                    )
                q_mod_tmin = np.percentile(
                    mod_tmin[mask_tmin].to_numpy(), quantiles, method="linear"
                )
                q_obs_tmin = np.percentile(
                    obs_tmin[mask_tmin].to_numpy(), quantiles, method="linear"
                )
                s_low_min, s_up_min = self._compute_slopes(
                    q_mod_tmin, q_obs_tmin, self.extrapolation
                )

                self.tmin_quantiles_mod_[m] = q_mod_tmin
                self.tmin_quantiles_obs_[m] = q_obs_tmin
                self.tmin_slopes_lower_[m] = s_low_min
                self.tmin_slopes_upper_[m] = s_up_min

        else:
            # Vectorized 31-day circular moving window centered on day of year
            obs_leap_doys = np.array([date_to_leap_doy(ts) for ts in idx])

            for doy in range(1, 367):
                window_mask = _doy_window_mask(obs_leap_doys, doy, half_window=15)

                # Tmax
                mask_tmax = window_mask & obs_tmax.notna() & mod_tmax.notna()
                sample_counts.append(int(mask_tmax.sum()))
                if mask_tmax.sum() < MIN_SAMPLES_PER_GROUP:
                    raise ValueError(
                        f"DOY {doy} has insufficient valid samples for Tmax EQM: "
                        f"{mask_tmax.sum()} (< {MIN_SAMPLES_PER_GROUP})."
                    )
                q_mod_tmax = np.percentile(
                    mod_tmax[mask_tmax].to_numpy(), quantiles, method="linear"
                )
                q_obs_tmax = np.percentile(
                    obs_tmax[mask_tmax].to_numpy(), quantiles, method="linear"
                )
                s_low_max, s_up_max = self._compute_slopes(
                    q_mod_tmax, q_obs_tmax, self.extrapolation
                )

                self.tmax_quantiles_mod_[doy] = q_mod_tmax
                self.tmax_quantiles_obs_[doy] = q_obs_tmax
                self.tmax_slopes_lower_[doy] = s_low_max
                self.tmax_slopes_upper_[doy] = s_up_max

                # Tmin
                mask_tmin = window_mask & obs_tmin.notna() & mod_tmin.notna()
                sample_counts.append(int(mask_tmin.sum()))
                if mask_tmin.sum() < MIN_SAMPLES_PER_GROUP:
                    raise ValueError(
                        f"DOY {doy} has insufficient valid samples for Tmin EQM: "
                        f"{mask_tmin.sum()} (< {MIN_SAMPLES_PER_GROUP})."
                    )
                q_mod_tmin = np.percentile(
                    mod_tmin[mask_tmin].to_numpy(), quantiles, method="linear"
                )
                q_obs_tmin = np.percentile(
                    obs_tmin[mask_tmin].to_numpy(), quantiles, method="linear"
                )
                s_low_min, s_up_min = self._compute_slopes(
                    q_mod_tmin, q_obs_tmin, self.extrapolation
                )

                self.tmin_quantiles_mod_[doy] = q_mod_tmin
                self.tmin_quantiles_obs_[doy] = q_obs_tmin
                self.tmin_slopes_lower_[doy] = s_low_min
                self.tmin_slopes_upper_[doy] = s_up_min

        self.min_n_per_group = min(sample_counts) if sample_counts else 0
        return self

    @staticmethod
    def _apply_eqm_to_array(
        vals: np.ndarray,
        q_mod: np.ndarray,
        q_obs: np.ndarray,
        s_lower: float,
        s_upper: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply piecewise linear quantile mapping with tail extrapolation.

        Returns:
            Tuple of (corrected_values, extrapolation_bool_mask,
            abs_correction_magnitudes).
        """
        res = np.empty_like(vals, dtype=float)
        nan_mask = np.isnan(vals)
        res[nan_mask] = np.nan

        extrap_mask = np.zeros(vals.shape, dtype=bool)
        abs_corr = np.zeros(vals.shape, dtype=float)

        valid_mask = ~nan_mask
        valid_vals = vals[valid_mask]
        if len(valid_vals) == 0:
            return res, extrap_mask, abs_corr

        # 1. Piecewise linear interpolation on interior nodes
        interp_vals = np.interp(valid_vals, q_mod, q_obs)

        # 2. Lower tail extension
        low_tail = valid_vals < q_mod[0]
        if low_tail.any():
            delta_low = valid_vals[low_tail] - q_mod[0]
            interp_vals[low_tail] = q_obs[0] + s_lower * delta_low

        # 3. Upper tail extension
        high_tail = valid_vals > q_mod[-1]
        if high_tail.any():
            delta_high = valid_vals[high_tail] - q_mod[-1]
            interp_vals[high_tail] = q_obs[-1] + s_upper * delta_high

        res[valid_mask] = interp_vals

        tail_in_valid = low_tail | high_tail
        if tail_in_valid.any():
            tail_indices = np.where(valid_mask)[0][tail_in_valid]
            extrap_mask[tail_indices] = True
            abs_corr[tail_indices] = np.abs(res[tail_indices] - vals[tail_indices])

        return res, extrap_mask, abs_corr

    def transform(
        self,
        mod_tmax: pd.Series,
        mod_tmin: pd.Series,
    ) -> Tuple[pd.Series, pd.Series]:
        """Apply EQM transformation and universal crossing rule."""
        if not self.tmax_quantiles_mod_ or not self.tmin_quantiles_mod_:
            raise ValueError("Downscaler must be fitted before calling transform.")

        _validate_datetime_indices_equal(
            mod_tmax, mod_tmin, names=["mod_tmax", "mod_tmin"]
        )
        _validate_temperature_range(mod_tmax, "mod_tmax")
        _validate_temperature_range(mod_tmin, "mod_tmin")

        idx = cast(pd.DatetimeIndex, mod_tmax.index)
        tmax_corr = pd.Series(np.nan, index=idx, dtype=float)
        tmin_corr = pd.Series(np.nan, index=idx, dtype=float)

        extrap_mask_tmax = np.zeros(len(idx), dtype=bool)
        extrap_mask_tmin = np.zeros(len(idx), dtype=bool)
        abs_corr_tmax = np.zeros(len(idx), dtype=float)
        abs_corr_tmin = np.zeros(len(idx), dtype=float)

        if self.grouping == "monthly":
            months = idx.month
            for m in range(1, 13):
                m_mask = months == m
                if m_mask.any():
                    tmax_arr, e_mask_max, a_corr_max = self._apply_eqm_to_array(
                        mod_tmax.loc[m_mask].to_numpy(),
                        self.tmax_quantiles_mod_[m],
                        self.tmax_quantiles_obs_[m],
                        self.tmax_slopes_lower_[m],
                        self.tmax_slopes_upper_[m],
                    )
                    tmax_corr.loc[m_mask] = tmax_arr
                    extrap_mask_tmax[m_mask] = e_mask_max
                    abs_corr_tmax[m_mask] = a_corr_max

                    tmin_arr, e_mask_min, a_corr_min = self._apply_eqm_to_array(
                        mod_tmin.loc[m_mask].to_numpy(),
                        self.tmin_quantiles_mod_[m],
                        self.tmin_quantiles_obs_[m],
                        self.tmin_slopes_lower_[m],
                        self.tmin_slopes_upper_[m],
                    )
                    tmin_corr.loc[m_mask] = tmin_arr
                    extrap_mask_tmin[m_mask] = e_mask_min
                    abs_corr_tmin[m_mask] = a_corr_min
        else:
            leap_doys = np.array([date_to_leap_doy(ts) for ts in idx])
            for doy in range(1, 367):
                doy_mask = leap_doys == doy
                if doy_mask.any():
                    tmax_arr, e_mask_max, a_corr_max = self._apply_eqm_to_array(
                        mod_tmax.loc[doy_mask].to_numpy(),
                        self.tmax_quantiles_mod_[doy],
                        self.tmax_quantiles_obs_[doy],
                        self.tmax_slopes_lower_[doy],
                        self.tmax_slopes_upper_[doy],
                    )
                    tmax_corr.loc[doy_mask] = tmax_arr
                    extrap_mask_tmax[doy_mask] = e_mask_max
                    abs_corr_tmax[doy_mask] = a_corr_max

                    tmin_arr, e_mask_min, a_corr_min = self._apply_eqm_to_array(
                        mod_tmin.loc[doy_mask].to_numpy(),
                        self.tmin_quantiles_mod_[doy],
                        self.tmin_quantiles_obs_[doy],
                        self.tmin_slopes_lower_[doy],
                        self.tmin_slopes_upper_[doy],
                    )
                    tmin_corr.loc[doy_mask] = tmin_arr
                    extrap_mask_tmin[doy_mask] = e_mask_min
                    abs_corr_tmin[doy_mask] = a_corr_min

        # Diagnostic metrics on extrapolation
        any_extrap_mask = extrap_mask_tmax | extrap_mask_tmin
        self.n_extrapolated_days_ = int(any_extrap_mask.sum())
        all_extrap_corrs: List[float] = []
        if extrap_mask_tmax.any():
            all_extrap_corrs.append(float(abs_corr_tmax[extrap_mask_tmax].max()))
        if extrap_mask_tmin.any():
            all_extrap_corrs.append(float(abs_corr_tmin[extrap_mask_tmin].max()))
        self.max_abs_extrapolation_correction_ = (
            max(all_extrap_corrs) if all_extrap_corrs else 0.0
        )

        # Count crossings before universal crossing rule
        valid_both = tmax_corr.notna() & tmin_corr.notna()
        raw_crossing_mask = valid_both & (tmin_corr > tmax_corr)
        self.n_crossings_before_rule_ = int(raw_crossing_mask.sum())

        # Apply universal crossing rule where both series are valid
        final_tmax, final_tmin, diag = apply_universal_crossing_rule(
            tmax_corr, tmin_corr
        )
        self.last_crossing_diagnostic_ = diag
        valid_final = final_tmax.notna() & final_tmin.notna()
        self.n_crossings_after_rule_ = int(
            (valid_final & (final_tmin > final_tmax)).sum()
        )

        final_tmax.name = "tmax_corrected"
        final_tmin.name = "tmin_corrected"
        return final_tmax, final_tmin

    @property
    def slope_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Return (min_slope, max_slope) per variable and tail across all groups."""

        def _rng(d: Dict[int, float]) -> Tuple[float, float]:
            if not d:
                return (1.0, 1.0)
            vals = list(d.values())
            return (float(min(vals)), float(max(vals)))

        return {
            "tmax_upper": _rng(self.tmax_slopes_upper_),
            "tmax_lower": _rng(self.tmax_slopes_lower_),
            "tmin_upper": _rng(self.tmin_slopes_upper_),
            "tmin_lower": _rng(self.tmin_slopes_lower_),
        }

    @property
    def slope_range(self) -> Tuple[float, float]:
        """Return overall (min_slope, max_slope) across all fitted tails and groups."""
        all_s: List[float] = []
        for d in (
            self.tmax_slopes_upper_,
            self.tmax_slopes_lower_,
            self.tmin_slopes_upper_,
            self.tmin_slopes_lower_,
        ):
            all_s.extend(d.values())
        if not all_s:
            return (1.0, 1.0)
        return (float(min(all_s)), float(max(all_s)))


@dataclass
class Level3TemperatureTuningResult:
    """Container for Level 3 Temperature Hyperparameter Grid CV tuning."""

    config_id: int
    grouping: str
    n_quantiles: int
    extrapolation: str
    overall_tuning_loss: float
    annual_fold_losses: List[float]
    crossings_before_rule: int
    crossings_after_rule: int
    crossing_fraction: float
    n_extrapolated_days: int
    max_abs_extrapolation_correction: float
    min_n_per_group: int


def select_best_level3_temperature_result(
    results: Sequence[Level3TemperatureTuningResult],
    tolerance: float = 1e-6,
) -> Level3TemperatureTuningResult:
    """Select winning Level 3 temperature configuration with parsimony tie-breaking.

    Tie-breaking hierarchy when losses are within tolerance (1e-6 °C):
    1. grouping: 'monthly' (simpler) < 'moving_window' (more complex)
    2. n_quantiles: 50 (simpler) < 100 (more complex)
    3. extrapolation: 'constant' (simpler) < 'linear' (more complex)
    4. final tie-break: config_id (ascending order).
    """
    if not results:
        raise ValueError("Cannot select from empty tuning results.")

    finite_results = [r for r in results if np.isfinite(r.overall_tuning_loss)]
    if not finite_results:
        raise ValueError("No candidate configuration produced a finite tuning loss.")

    min_loss = min(r.overall_tuning_loss for r in finite_results)
    candidates = [
        r for r in finite_results if r.overall_tuning_loss <= min_loss + tolerance
    ]

    def _parsimony_key(
        r: Level3TemperatureTuningResult,
    ) -> Tuple[int, int, int, int]:
        group_rank = 0 if r.grouping == "monthly" else 1
        bins_rank = 0 if r.n_quantiles == 50 else 1
        extrap_rank = 0 if r.extrapolation == "constant" else 1
        return (group_rank, bins_rank, extrap_rank, r.config_id)

    return min(candidates, key=_parsimony_key)


def tune_level3_temperature_hyperparameters(
    obs_tmax: pd.Series,
    obs_tmin: pd.Series,
    mod_tmax: pd.Series,
    mod_tmin: pd.Series,
    configs: Optional[Sequence[Dict[str, Any]]] = None,
    parsimony_tolerance: float = 1e-6,
) -> Tuple[
    Level3TemperatureTuningResult,
    Level3QuantileMappingTemperature,
    List[Level3TemperatureTuningResult],
]:
    """Tune Level 3 temperature hyperparameter grid via nested annual block CV.

    Protocol Section 4.1:
    - Partitioning: Leave-One-Year-Out whole calendar years.
    - Loss: temperature_tuning_loss evaluated on concatenated out-of-sample series.
    - Tie-breaking: Parsimony (1e-6 tol) via select_best_level3_temperature_result.
    - Final Model: Best configuration refitted on full calibration partition.

    Returns:
        Tuple of (best_result, fitted_best_model, all_tuning_results).
    """
    if configs is None:
        configs = LEVEL3_TEMPERATURE_CONFIGS

    _validate_datetime_indices_equal(
        obs_tmax,
        obs_tmin,
        mod_tmax,
        mod_tmin,
        names=["obs_tmax", "obs_tmin", "mod_tmax", "mod_tmin"],
    )
    _validate_temperature_range(obs_tmax, "obs_tmax")
    _validate_temperature_range(obs_tmin, "obs_tmin")
    _validate_temperature_range(mod_tmax, "mod_tmax")
    _validate_temperature_range(mod_tmin, "mod_tmin")

    idx = cast(pd.DatetimeIndex, obs_tmax.index)
    all_years = sorted(idx.year.unique())

    # Only include calendar years with at least one valid paired day for both extremes
    valid_years = []
    for y in all_years:
        y_mask = (
            (idx.year == y)
            & obs_tmax.notna()
            & obs_tmin.notna()
            & mod_tmax.notna()
            & mod_tmin.notna()
        )
        if y_mask.sum() > 0:
            valid_years.append(y)

    if len(valid_years) < 2:
        raise ValueError(
            f"Annual block CV requires at least 2 valid years, got {len(valid_years)}."
        )

    all_results: List[Level3TemperatureTuningResult] = []

    for cfg in configs:
        cfg_id = int(cfg["config_id"])
        grouping = cast(Literal["monthly", "moving_window"], cfg["grouping"])
        n_quantiles = int(cfg["n_quantiles"])
        extrapolation = cast(
            Literal["constant", "linear"],
            cfg["extrapolation"],
        )

        oos_tmax_preds: List[pd.Series] = []
        oos_tmin_preds: List[pd.Series] = []
        fold_losses: List[float] = []

        fold_crossings_before: List[int] = []
        fold_extrap_days: List[int] = []
        fold_max_extrap_corrs: List[float] = []
        min_n_samples_folds: List[int] = []

        for val_year in valid_years:
            val_mask = idx.year == val_year
            train_mask = ~val_mask

            train_obs_tmax = obs_tmax.loc[train_mask]
            train_obs_tmin = obs_tmin.loc[train_mask]
            train_mod_tmax = mod_tmax.loc[train_mask]
            train_mod_tmin = mod_tmin.loc[train_mask]

            val_mod_tmax = mod_tmax.loc[val_mask]
            val_mod_tmin = mod_tmin.loc[val_mask]

            val_obs_tmax = obs_tmax.loc[val_mask]
            val_obs_tmin = obs_tmin.loc[val_mask]

            fold_model = Level3QuantileMappingTemperature(
                grouping=grouping,
                n_quantiles=n_quantiles,
                extrapolation=extrapolation,
            )
            fold_model.fit(
                train_obs_tmax,
                train_obs_tmin,
                train_mod_tmax,
                train_mod_tmin,
            )
            pred_tmax, pred_tmin = fold_model.transform(val_mod_tmax, val_mod_tmin)

            oos_tmax_preds.append(pred_tmax)
            oos_tmin_preds.append(pred_tmin)

            fold_crossings_before.append(fold_model.n_crossings_before_rule_)
            fold_extrap_days.append(fold_model.n_extrapolated_days_)
            fold_max_extrap_corrs.append(fold_model.max_abs_extrapolation_correction_)
            if fold_model.min_n_per_group is not None:
                min_n_samples_folds.append(fold_model.min_n_per_group)

            # Fold-level diagnostic loss
            f_loss = temperature_tuning_loss(
                val_obs_tmax, pred_tmax, val_obs_tmin, pred_tmin
            )
            fold_losses.append(f_loss)

        # Concatenate out-of-sample predictions
        concat_pred_tmax = pd.concat(oos_tmax_preds).sort_index()
        concat_pred_tmin = pd.concat(oos_tmin_preds).sort_index()

        # Overall loss evaluated on concatenated out-of-sample series (Protocol 4.1)
        overall_loss = temperature_tuning_loss(
            obs_tmax.loc[concat_pred_tmax.index],
            concat_pred_tmax,
            obs_tmin.loc[concat_pred_tmin.index],
            concat_pred_tmin,
        )

        total_cr_before = sum(fold_crossings_before)
        valid_oos = concat_pred_tmax.notna() & concat_pred_tmin.notna()
        total_cr_after = int((valid_oos & (concat_pred_tmin > concat_pred_tmax)).sum())

        # Total valid model days across out-of-sample evaluation
        total_eval_days = int(valid_oos.sum())
        cr_fraction = total_cr_before / total_eval_days if total_eval_days > 0 else 0.0

        res = Level3TemperatureTuningResult(
            config_id=cfg_id,
            grouping=grouping,
            n_quantiles=n_quantiles,
            extrapolation=extrapolation,
            overall_tuning_loss=overall_loss,
            annual_fold_losses=fold_losses,
            crossings_before_rule=total_cr_before,
            crossings_after_rule=total_cr_after,
            crossing_fraction=cr_fraction,
            n_extrapolated_days=sum(fold_extrap_days),
            max_abs_extrapolation_correction=(
                max(fold_max_extrap_corrs) if fold_max_extrap_corrs else 0.0
            ),
            min_n_per_group=(min(min_n_samples_folds) if min_n_samples_folds else 0),
        )
        all_results.append(res)

    best_res = select_best_level3_temperature_result(
        all_results, tolerance=parsimony_tolerance
    )

    # Refit the winning configuration on the full calibration partition
    final_model = Level3QuantileMappingTemperature(
        grouping=cast(Literal["monthly", "moving_window"], best_res.grouping),
        n_quantiles=best_res.n_quantiles,
        extrapolation=cast(Literal["constant", "linear"], best_res.extrapolation),
    )
    final_model.fit(obs_tmax, obs_tmin, mod_tmax, mod_tmin)

    return best_res, final_model, all_results


def load_calibration_temperature_pair(
    station_id: str,
    obs_tmax_path: Path,
    obs_tmin_path: Path,
    era5_target: Path,
    window_start_utc: int,
    lag_days: int,
    min_hours: int,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Load calibration partition paired series for daily temperature downscaling.

    Protocol Section 2.2, 4.1, 8, E.6.10:
    - Resolves calibration partition range from protocol.get_benchmark_partitions.
    - Requires columns 'Fecha' and 'Valor' in DHIME CSVs without positional fallback.
    - Trims observations immediately after reading, before any statistic is computed.
    - Disallows unregistered stations with explicit ValueError (no spatial fallback).
    - Filters ERA5 directory files strictly to calibration years with regex.
    - Prohibits missing values (NaN) in reanalysis calibration daily series.
    - Enforces physical temperature validity [-60, 60] °C.
    - Requires explicit window_start_utc, lag_days, and min_hours (no silent defaults).

    Args:
        station_id: Benchmark station ID (e.g. '21205710', '23215030').
        obs_tmax_path: Path to DHIME daily Tmax CSV.
        obs_tmin_path: Path to DHIME daily Tmin CSV.
        era5_target: Path to ERA5 NetCDF file or directory of NetCDF files.
        window_start_utc: Explicit start hour in UTC for aggregation window (e.g. 5).
        lag_days: Lag offset in days (e.g. 0).
        min_hours: Minimum valid hours required per day (e.g. 24).

    Returns:
        Tuple of (obs_tmax_cal, obs_tmin_cal, mod_tmax_cal, mod_tmin_cal).
    """
    clean_id = re.sub(r"\.0$", "", str(station_id).strip())
    valid_years_filter: Optional[Set[int]] = None
    andean_st: Optional[AndeanStation] = None
    try:
        cal_range = get_benchmark_partitions("temperature", clean_id)[0]
    except ValueError:
        try:
            andean_st = get_andean_station(clean_id)
            cal_range = andean_st.cal_block
            valid_years_filter = set(andean_st.valid_years)
        except ValueError:
            raise ValueError(
                f"Station '{clean_id}' has no pre-specified temperature calibration "
                "partition."
            ) from None

    cal_start, cal_end = cal_range
    full_cal_index = pd.date_range(f"{cal_start}-01-01", f"{cal_end}-12-31", freq="D")

    # 1. Load and slice observations strictly to calibration
    def _read_and_slice_obs(path: Path) -> pd.Series:
        if not path.exists():
            raise FileNotFoundError(f"Observation file not found: {path}")

        df_raw = pd.read_csv(path)
        if "CodigoEstacion" not in df_raw.columns:
            raise ValueError(
                "Required column 'CodigoEstacion' not found in observation file "
                f"'{path}'."
            )
        if "Fecha" not in df_raw.columns or "Valor" not in df_raw.columns:
            raise ValueError(
                f"Required columns 'Fecha' and 'Valor' must be present in '{path}'. "
                f"Found columns: {list(df_raw.columns)}."
            )

        st_codes = (
            df_raw["CodigoEstacion"]
            .astype(str)
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
        )
        if clean_id not in st_codes.values:
            raise ValueError(
                f"Station ID '{clean_id}' not found in observation file '{path}'."
            )
        df_st = df_raw[st_codes == clean_id].copy()

        # Validate parseable dates (strict ISO8601)
        parsed_dt = pd.to_datetime(df_st["Fecha"], format="ISO8601", errors="coerce")
        unparseable_mask = (
            parsed_dt.isna()
            & df_st["Fecha"].notna()
            & (df_st["Fecha"].astype(str).str.strip() != "")
        )
        unparseable_count = int(unparseable_mask.sum())
        if unparseable_count > 0:
            raise ValueError(
                f"Found {unparseable_count} unparseable date(s) in '{path}'."
            )

        # Validate non-numeric non-empty values
        raw_val_str = df_st["Valor"].astype(str).str.strip().str.replace(",", ".")
        num_vals = pd.to_numeric(raw_val_str, errors="coerce")
        non_numeric_mask = (
            num_vals.isna()
            & df_st["Valor"].notna()
            & ~raw_val_str.str.upper().isin(["", "NAN", "NONE", "NULL"])
        )
        non_numeric_count = int(non_numeric_mask.sum())
        if non_numeric_count > 0:
            raise ValueError(
                f"Found {non_numeric_count} non-numeric non-empty value(s) in '{path}'."
            )

        # Check duplicate dates with distinct values
        df_st["dt_norm"] = parsed_dt.dt.normalize()
        dup_mask = df_st["dt_norm"].duplicated(keep=False)
        if dup_mask.any():
            dup_rows = df_st[dup_mask]
            distinct_dup_count = 0
            for _, group in dup_rows.groupby("dt_norm"):
                valid_distinct = group["Valor"].dropna().unique()
                if len(valid_distinct) > 1:
                    distinct_dup_count += 1
            if distinct_dup_count > 0:
                raise ValueError(
                    f"Found {distinct_dup_count} duplicate date(s) with "
                    f"distinct values in '{path}'."
                )

        # Drop NA dates and deduplicate identical values
        df_st = df_st.dropna(subset=["dt_norm"]).sort_values("dt_norm")
        df_st = df_st.drop_duplicates(subset=["dt_norm"], keep="first")

        clean_num = pd.to_numeric(
            df_st["Valor"].astype(str).str.strip().str.replace(",", "."),
            errors="coerce",
        )
        s_full = pd.Series(clean_num.values, index=pd.DatetimeIndex(df_st["dt_norm"]))
        s_idx = cast(pd.DatetimeIndex, s_full.index)
        # Immediately trim to calibration years before any statistic is computed
        cal_mask = (s_idx.year >= cal_start) & (s_idx.year <= cal_end)
        if valid_years_filter is not None:
            cal_mask = cal_mask & s_idx.year.isin(valid_years_filter)
        s_cal = s_full[cal_mask]
        return s_cal.reindex(full_cal_index)

    obs_tmax_cal = _read_and_slice_obs(obs_tmax_path)
    obs_tmin_cal = _read_and_slice_obs(obs_tmin_path)

    _validate_temperature_range(obs_tmax_cal, "obs_tmax")
    _validate_temperature_range(obs_tmin_cal, "obs_tmin")

    # 2. Load ERA5 hourly data and aggregate with explicit window
    from tropicor.io.era5 import ERA5Adapter, aggregate_hourly_to_daily_temperature
    from tropicor.io.stations import StationCatalog

    if era5_target.is_dir():
        nc_files = sorted(glob.glob(str(era5_target / "*2m_temperature*.nc")))
        if not nc_files:
            raise ValueError(
                f"No '*2m_temperature*.nc' files found in directory {era5_target}."
            )
        # Filter files to calibration years plus the boundary years needed for window
        cal_years = list(range(cal_start, cal_end + 2))
        cal_nc_files = [
            f
            for f in nc_files
            if any(re.search(rf"(?<!\d){y}(?!\d)", Path(f).name) for y in cal_years)
        ]
        if not cal_nc_files:
            raise ValueError(
                f"No ERA5 NetCDF files matching calibration years "
                f"{cal_start}-{cal_end} found in directory {era5_target}."
            )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*Dask is not installed.*", category=UserWarning
            )
            adapter = ERA5Adapter(cal_nc_files)
    else:
        adapter = ERA5Adapter(era5_target)

    catalog = StationCatalog.from_benchmark()
    st_meta = catalog.get(clean_id)
    if st_meta is not None:
        lat, lon = st_meta.latitude, st_meta.longitude
    elif andean_st is not None:
        lat, lon = andean_st.latitud_cne, andean_st.longitud_cne
    else:
        try:
            andean_st = get_andean_station(clean_id)
            lat, lon = andean_st.latitud_cne, andean_st.longitud_cne
        except ValueError:
            raise ValueError(
                f"Station ID '{clean_id}' not found in benchmark station catalog."
            ) from None

    var_name = adapter._resolve_variable_name("t2m")
    ds_pt = adapter.extract_point(lat, lon, method="nearest")

    # Verify that the extracted ERA5 grid cell matches the registered cell
    if andean_st is not None:
        grid_lat = float(ds_pt["latitude"].values)
        grid_lon = float(ds_pt["longitude"].values)
        exp_lat = andean_st.celda_era5["lat"]
        exp_lon = andean_st.celda_era5["lon"]
        if not (
            np.isclose(grid_lat, exp_lat, atol=1e-3)
            and np.isclose(grid_lon, exp_lon, atol=1e-3)
        ):
            raise ValueError(
                f"ERA5 grid cell ({grid_lat:.2f}, {grid_lon:.2f}) does not match "
                f"registered cell ({exp_lat:.2f}, {exp_lon:.2f}) "
                f"for Andean station '{clean_id}'."
            )

    da = ds_pt[var_name]

    if "step" in da.dims:
        da = da.squeeze("step")
    if "surface" in da.dims:
        da = da.squeeze("surface")
    if "number" in da.dims:
        da = da.isel(number=0)

    # Validate ERA5 units attribute: demands 'K' before conversion
    units = str(da.attrs.get("units", "")).strip()
    if units != "K":
        raise ValueError(
            f"ERA5 variable '{var_name}' must have units 'K' (Kelvin), found '{units}'."
        )

    s_hourly_raw = da.to_series()
    s_hourly_raw.index = pd.DatetimeIndex(s_hourly_raw.index)

    # Explicit unit conversion from Kelvin to Celsius
    s_hourly_c = s_hourly_raw - 273.15

    df_daily_mod = aggregate_hourly_to_daily_temperature(
        s_hourly_c,
        window_start_utc=window_start_utc,
        min_hours=min_hours,
        lag_days=lag_days,
        convert_to_celsius=False,  # Already explicitly converted above
    )

    # Immediately slice model series to calibration years
    mod_idx = cast(pd.DatetimeIndex, df_daily_mod.index)
    cal_mod_mask = (mod_idx.year >= cal_start) & (mod_idx.year <= cal_end)
    df_daily_mod_cal = df_daily_mod.loc[cal_mod_mask].reindex(full_cal_index)

    mod_tmax_cal = df_daily_mod_cal["tmax"]
    mod_tmin_cal = df_daily_mod_cal["tmin"]

    # Validate that model calibration daily series has no NaNs (ERA5 has no gaps)
    nan_tmax_count = int(mod_tmax_cal.isna().sum())
    nan_tmin_count = int(mod_tmin_cal.isna().sum())
    if nan_tmax_count > 0 or nan_tmin_count > 0:
        nan_idx = (
            mod_tmax_cal[mod_tmax_cal.isna()].index[:5]
            if nan_tmax_count > 0
            else mod_tmin_cal[mod_tmin_cal.isna()].index[:5]
        )
        first_nan_dates = [str(ts)[:10] for ts in nan_idx]
        raise ValueError(
            f"ERA5 daily model series contains missing values (NaN) within calibration "
            f"partition {cal_start}-{cal_end}: {nan_tmax_count} NaNs in Tmax, "
            f"{nan_tmin_count} NaNs in Tmin. First NaN dates: {first_nan_dates}."
        )

    _validate_temperature_range(mod_tmax_cal, "mod_tmax")
    _validate_temperature_range(mod_tmin_cal, "mod_tmin")

    obs_tmax_cal.name = "obs_tmax"
    obs_tmin_cal.name = "obs_tmin"
    mod_tmax_cal.name = "mod_tmax"
    mod_tmin_cal.name = "mod_tmin"

    return obs_tmax_cal, obs_tmin_cal, mod_tmax_cal, mod_tmin_cal
