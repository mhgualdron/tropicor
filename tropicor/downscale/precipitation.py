"""Daily precipitation downscaling hierarchy (Levels 1–3) and calibration utilities.

This module implements the daily precipitation downscaling ladder defined in
Protocol Section 6.1, Section 6.3, and Section 7:
- Level 1: Linear Scaling (monthly multiplicative factors)
- Level 2: Threshold Matching & Scaling (dry-day frequency adjustment +
  mass-conserving linear scaling)
- Level 3: Empirical Quantile Mapping (wet-day EQM with monthly or circular
  31-day windows)
- Hyperparameter tuning via nested annual block cross-validation
- Calibration data loading utility with immediate trimming of validation years
"""

from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd

from tropicor.core.protocol import (
    get_benchmark_partitions,
    precipitation_tuning_loss,
)

# Reference leap calendar (366 days) cumulative day counts before month 1..12
LEAP_CUMULATIVE_DAYS: Tuple[int, ...] = (
    0,  # Jan (days 1..31)
    31,  # Feb (days 32..60; Feb 28 = 59, Feb 29 = 60)
    60,  # Mar (days 61..91; Mar 1 = 61)
    91,  # Apr (days 92..121)
    121,  # May (days 122..152)
    152,  # Jun (days 153..182)
    182,  # Jul (days 183..213)
    213,  # Aug (days 214..244)
    244,  # Sep (days 245..274)
    274,  # Oct (days 275..305)
    305,  # Nov (days 306..335)
    335,  # Dec (days 336..366; Dec 17 = 352, Dec 31 = 366)
)


def date_to_leap_doy(date: pd.Timestamp) -> int:
    """Map a calendar date to day-of-year in a 366-day leap reference calendar.

    For non-leap years, Feb 28 is day 59 and Mar 1 is day 61 (day 60 is omitted).
    In leap years, Feb 29 is day 60.

    Args:
        date: Calendar date timestamp.

    Returns:
        Integer day of year in [1, 366].
    """
    month = date.month
    day = date.day
    return LEAP_CUMULATIVE_DAYS[month - 1] + day


def leap_doy_circular_distance(d1: int, d2: int) -> int:
    """Compute circular distance between two days in a 366-day reference calendar.

    Args:
        d1: Day of year (1..366).
        d2: Day of year (1..366).

    Returns:
        Shortest distance along the circular 366-day calendar.
    """
    diff = abs(d1 - d2)
    return min(diff, 366 - diff)


class BasePrecipitationDownscaler:
    """Abstract base class for precipitation downscalers."""

    def _validate_inputs(self, obs: pd.Series, mod: pd.Series) -> None:
        """Validate input series structure, physical non-negativity, and units.

        Raises:
            ValueError: If indices are not DatetimeIndex, values are negative,
                or overall series mean is < 0.01 mm (indicating incorrect units).
        """
        if not isinstance(obs.index, pd.DatetimeIndex):
            raise ValueError("Observation series must have a pd.DatetimeIndex.")
        if not isinstance(mod.index, pd.DatetimeIndex):
            raise ValueError("Model series must have a pd.DatetimeIndex.")

        obs_valid = obs.dropna()
        mod_valid = mod.dropna()

        if len(obs_valid) > 0 and (obs_valid < -1e-6).any():
            raise ValueError("Observed precipitation contains negative values.")
        if len(mod_valid) > 0 and (mod_valid < -1e-6).any():
            raise ValueError("Model precipitation contains negative values.")

        if len(obs_valid) > 0 and obs_valid.mean() < 0.01:
            raise ValueError(
                f"Observed precipitation series mean is {obs_valid.mean():.6f} mm "
                "(< 0.01 mm). Inputs must be strictly in millimeters (mm)."
            )
        if len(mod_valid) > 0 and mod_valid.mean() < 0.01:
            raise ValueError(
                f"Model precipitation series mean is {mod_valid.mean():.6f} mm "
                "(< 0.01 mm). Inputs must be strictly in millimeters (mm)."
            )

    def fit(
        self, obs_cal: pd.Series, mod_cal: pd.Series
    ) -> BasePrecipitationDownscaler:
        """Estimate correction parameters using paired calibration series."""
        raise NotImplementedError

    def transform(self, mod: pd.Series) -> pd.Series:
        """Apply calibrated correction parameters to a model series."""
        raise NotImplementedError

    def fit_transform(self, obs_cal: pd.Series, mod_cal: pd.Series) -> pd.Series:
        """Fit on calibration series and transform model calibration series."""
        return self.fit(obs_cal, mod_cal).transform(mod_cal)


class Level1LinearScalingPrecipitation(BasePrecipitationDownscaler):
    """Level 1: Monthly multiplicative linear scaling for daily precipitation.

    Reference: Protocol Section 6.1.
    Computes monthly multiplicative factors:
        s_m = sum(P_obs, m) / sum(P_mod, m)
    over paired valid calibration days. No drizzle truncation is applied.
    """

    def __init__(self) -> None:
        self.monthly_factors: Dict[int, float] = {}

    def fit(
        self, obs_cal: pd.Series, mod_cal: pd.Series
    ) -> Level1LinearScalingPrecipitation:
        """Fit monthly scaling factors on paired calibration data.

        Args:
            obs_cal: Observed daily precipitation series in mm.
            mod_cal: Model daily precipitation series in mm.

        Returns:
            Fitted estimator instance.
        """
        self._validate_inputs(obs_cal, mod_cal)

        # Synchronous paired masking
        df = pd.DataFrame({"obs": obs_cal, "mod": mod_cal}).dropna()
        dt_idx = cast(pd.DatetimeIndex, df.index)
        df["month"] = dt_idx.month
        self.monthly_factors = {}

        for month in range(1, 13):
            sub = df[df["month"] == month]
            if sub.empty:
                raise ValueError(f"no paired valid calibration days for month {month}")

            sum_obs = float(sub["obs"].sum())
            sum_mod = float(sub["mod"].sum())

            if sum_mod == 0.0:
                if sum_obs > 0.0:
                    raise ValueError(
                        f"Level 1 fitting failed for month {month}: zero model "
                        f"accumulation with positive observed accumulation "
                        f"({sum_obs:.2f} mm)."
                    )
                self.monthly_factors[month] = 0.0
            else:
                self.monthly_factors[month] = float(sum_obs / sum_mod)

        return self

    def transform(self, mod: pd.Series) -> pd.Series:
        """Apply monthly scaling factors to daily model precipitation.

        Args:
            mod: Daily model precipitation series in mm.

        Returns:
            Corrected precipitation series with NaN values strictly preserved.
        """
        if not self.monthly_factors:
            raise RuntimeError("Estimator has not been fitted. Call fit() first.")
        if not isinstance(mod.index, pd.DatetimeIndex):
            raise ValueError("Input series must have a pd.DatetimeIndex.")

        months = mod.index.month
        factors = months.map(self.monthly_factors).to_numpy(dtype=float)
        values = mod.to_numpy(dtype=float)

        # Compute scaled values, preserving NaNs
        res_values = values * factors
        # Enforce physical non-negativity clamp on valid values
        res_values = np.where(np.isnan(res_values), np.nan, np.maximum(0.0, res_values))

        return pd.Series(res_values, index=mod.index, name=mod.name)


class Level2ThresholdMatchingPrecipitation(BasePrecipitationDownscaler):
    """Level 2: Drizzle threshold matching and mass-conserving scaling.

    Reference: Protocol Section 7.1–7.4.
    For each calendar month m:
    1. f_obs,m^wet is the observed wet-day frequency (P_obs >= 1.0 mm).
    2. P_th,m is computed as the (1 - f_obs,m^wet) quantile of calibration model days
       via Type 7 linear interpolation.
    3. Model precipitation < P_th,m is censored to 0.0 mm.
    4. s_m = sum(P_obs, m) / sum(P_censored, m) scales surviving precipitation.
    """

    def __init__(self) -> None:
        self.monthly_observed_f_wet: Dict[int, float] = {}
        self.monthly_thresholds: Dict[int, float] = {}
        self.monthly_scaling_factors: Dict[int, float] = {}

    def fit(
        self, obs_cal: pd.Series, mod_cal: pd.Series
    ) -> Level2ThresholdMatchingPrecipitation:
        """Fit monthly thresholds and scaling factors on paired calibration data.

        Args:
            obs_cal: Observed daily precipitation series in mm.
            mod_cal: Model daily precipitation series in mm.

        Returns:
            Fitted estimator instance.
        """
        self._validate_inputs(obs_cal, mod_cal)

        # Synchronous paired masking
        df = pd.DataFrame({"obs": obs_cal, "mod": mod_cal}).dropna()
        dt_idx = cast(pd.DatetimeIndex, df.index)
        df["month"] = dt_idx.month
        self.monthly_observed_f_wet = {}
        self.monthly_thresholds = {}
        self.monthly_scaling_factors = {}

        for month in range(1, 13):
            sub = df[df["month"] == month]
            if sub.empty:
                raise ValueError(f"no paired valid calibration days for month {month}")

            y_obs = sub["obs"].to_numpy(dtype=float)
            y_mod = sub["mod"].to_numpy(dtype=float)
            n_days = len(y_obs)

            # 1. Observed wet-day frequency (P_obs >= 1.0 mm)
            n_wet_obs = int(np.sum(y_obs >= 1.0))
            f_wet = float(n_wet_obs / n_days)
            self.monthly_observed_f_wet[month] = f_wet

            # 2. Threshold P_th,m as (1 - f_wet) quantile of model series (Type 7)
            q_prob = max(0.0, min(1.0, 1.0 - f_wet))
            p_th = float(np.percentile(y_mod, q_prob * 100.0, method="linear"))
            self.monthly_thresholds[month] = p_th

            # 3. Censoring and scaling factor
            y_censored = np.where(y_mod >= p_th, y_mod, 0.0)
            sum_censored = float(np.sum(y_censored))
            sum_obs = float(np.sum(y_obs))

            if sum_censored == 0.0:
                if sum_obs > 0.0:
                    raise ValueError(
                        f"Level 2 fitting failed for month {month}: zero model "
                        f"censored precipitation with positive observed "
                        f"precipitation ({sum_obs:.2f} mm)."
                    )
                self.monthly_scaling_factors[month] = 0.0
            else:
                self.monthly_scaling_factors[month] = float(sum_obs / sum_censored)

        return self

    def transform(self, mod: pd.Series) -> pd.Series:
        """Apply drizzle threshold truncation and monthly scaling to model series.

        Args:
            mod: Daily model precipitation series in mm.

        Returns:
            Corrected precipitation series with NaN values strictly preserved.
        """
        if not self.monthly_thresholds:
            raise RuntimeError("Estimator has not been fitted. Call fit() first.")
        if not isinstance(mod.index, pd.DatetimeIndex):
            raise ValueError("Input series must have a pd.DatetimeIndex.")

        months = mod.index.month
        thresholds = months.map(self.monthly_thresholds).to_numpy(dtype=float)
        factors = months.map(self.monthly_scaling_factors).to_numpy(dtype=float)
        values = mod.to_numpy(dtype=float)

        is_nan = np.isnan(values)
        res_values = np.zeros_like(values)

        is_wet = values >= thresholds
        res_values[is_wet] = values[is_wet] * factors[is_wet]
        # Truncated model values (< threshold) are strictly 0.0
        res_values[~is_wet] = 0.0

        # Enforce physical floor and preserve NaNs
        res_values = np.where(is_nan, np.nan, np.maximum(0.0, res_values))

        return pd.Series(res_values, index=mod.index, name=mod.name)


class Level3QuantileMappingPrecipitation(BasePrecipitationDownscaler):
    """Level 3: Empirical Quantile Mapping (EQM) with drizzle truncation.

    Reference: Protocol Section 6.1, Section 6.3, and Section 7.4.
    Fitted on wet days only (P_obs >= 1.0 mm vs censored model P_mod >= P_th,m).
    Supports calendar monthly grouping or 31-day moving circular window.
    Quantile transfer nodes are evaluated with 50 or 100 bins.

    Substantive Design Decision (Linear Extrapolation):
        In contrast to the literal reading of Section 6.3 (extrapolating the ratio),
        for P > Q_mod,top, this implementation performs the linear extension of
        the transfer function itself:
            P_corr = Q_obs,top + s * (P - Q_mod,top)
        with slope s = (Q_obs,top - Q_obs,top-1) / (Q_mod,top - Q_mod,top-1).
        If the denominator <= 1e-9, the estimator falls back to the constant
        ratio r_top = Q_obs,top / Q_mod,top.
        This substantive design decision was adopted because the literal reading
        (extrapolating the ratio r_top + slope_r * (P - Q_mod,top)) produced
        negative ratios when upper quantile empirical ratios were declining
        (11 days with raw storm events of 63–154 mm erroneously truncated
        to 0 mm in calibration linear configurations).
        With the transfer function linear extension and the non-decreasing
        monotonicity of empirical quantiles, s >= 0 mathematically guarantees
        physically non-negative corrections; the zero floor is retained as a
        defensive safeguard and its counter now evaluates strictly to 0 during
        extrapolation.
    """

    def __init__(
        self,
        window: Literal["monthly", "moving_window_31"] = "monthly",
        n_quantiles: Literal[50, 100] = 50,
        extrapolation: Literal["constant", "linear"] = "constant",
    ) -> None:
        if window not in ("monthly", "moving_window_31"):
            raise ValueError(
                f"Invalid window '{window}'. Expected 'monthly' or 'moving_window_31'."
            )
        if n_quantiles not in (50, 100):
            raise ValueError(f"Invalid n_quantiles {n_quantiles}. Expected 50 or 100.")
        if extrapolation not in ("constant", "linear"):
            raise ValueError(
                f"Invalid extrapolation '{extrapolation}'. Expected 'constant' "
                "or 'linear'."
            )

        self.window = window
        self.n_quantiles = n_quantiles
        self.extrapolation = extrapolation

        self.level2_thresholds: Dict[int, float] = {}
        self.quantile_transfer_nodes: Dict[
            Any, Tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = {}
        self.min_n_wet_per_group: int = 0
        self.n_clipped_below_zero: int = 0

    def fit(
        self, obs_cal: pd.Series, mod_cal: pd.Series
    ) -> Level3QuantileMappingPrecipitation:
        """Fit Level 2 thresholds and Level 3 quantile transfer nodes.

        Args:
            obs_cal: Observed daily precipitation series in mm.
            mod_cal: Model daily precipitation series in mm.

        Returns:
            Fitted estimator instance.
        """
        self._validate_inputs(obs_cal, mod_cal)

        # Synchronous paired masking
        df = pd.DataFrame({"obs": obs_cal, "mod": mod_cal}).dropna()
        dt_idx = cast(pd.DatetimeIndex, df.index)
        df["month"] = dt_idx.month
        df["leap_doy"] = [date_to_leap_doy(t) for t in dt_idx]

        # 1. Fit Level 2 monthly drizzle thresholds P_th,m
        self.level2_thresholds = {}
        for month in range(1, 13):
            sub = df[df["month"] == month]
            if sub.empty:
                self.level2_thresholds[month] = 0.0
                continue
            y_obs = sub["obs"].to_numpy(dtype=float)
            y_mod = sub["mod"].to_numpy(dtype=float)
            f_wet = float(np.sum(y_obs >= 1.0) / len(y_obs))
            q_prob = max(0.0, min(1.0, 1.0 - f_wet))
            p_th = float(np.percentile(y_mod, q_prob * 100.0, method="linear"))
            self.level2_thresholds[month] = p_th

        # Assign threshold to each day in calibration frame
        th_series = df["month"].map(self.level2_thresholds).to_numpy(dtype=float)
        df["p_th"] = th_series
        # Censored model values
        df["mod_censored"] = np.where(df["mod"] >= df["p_th"], df["mod"], 0.0)

        # 2. Quantile probabilities
        if self.n_quantiles == 50:
            probs = np.linspace(0.02, 0.98, 49)
        else:  # 100 bins
            probs = np.linspace(0.01, 0.99, 99)

        self.quantile_transfer_nodes = {}
        min_wet_count = 10**9
        self.n_clipped_below_zero = 0

        if self.window == "monthly":
            for month in range(1, 13):
                sub = df[df["month"] == month]
                obs_wet = sub[sub["obs"] >= 1.0]["obs"].to_numpy(dtype=float)
                mod_wet = sub[sub["mod_censored"] > 0.0]["mod_censored"].to_numpy(
                    dtype=float
                )

                if len(obs_wet) < 2 or len(mod_wet) < 2:
                    raise ValueError(
                        f"Level 3 fitting failed for month {month}: fewer than "
                        f"2 wet days found (obs: {len(obs_wet)}, mod: {len(mod_wet)}). "
                        "Empirical quantile mapping requires at least 2 wet "
                        "days per group."
                    )

                min_wet_count = min(min_wet_count, len(obs_wet), len(mod_wet))

                q_mod = np.percentile(mod_wet, probs * 100.0, method="linear")
                q_obs = np.percentile(obs_wet, probs * 100.0, method="linear")
                # Compute ratios Q_obs / Q_mod
                r = np.where(q_mod > 0, q_obs / q_mod, 1.0)
                self.quantile_transfer_nodes[month] = (q_mod, q_obs, r)

        else:  # moving_window_31
            all_doys = df["leap_doy"].to_numpy(dtype=int)
            y_obs_all = df["obs"].to_numpy(dtype=float)
            y_mod_cen_all = df["mod_censored"].to_numpy(dtype=float)

            # Circular window evaluation for each target day in 1..366
            for target_doy in range(1, 367):
                diff = np.abs(all_doys - target_doy)
                circ_dist = np.minimum(diff, 366 - diff)
                mask = circ_dist <= 15

                obs_win = y_obs_all[mask]
                mod_win = y_mod_cen_all[mask]

                obs_wet = obs_win[obs_win >= 1.0]
                mod_wet = mod_win[mod_win > 0.0]

                if len(obs_wet) < 2 or len(mod_wet) < 2:
                    raise ValueError(
                        f"Level 3 fitting failed for day of year {target_doy}: "
                        f"fewer than 2 wet days found (obs: {len(obs_wet)}, "
                        f"mod: {len(mod_wet)}). Empirical quantile mapping requires "
                        "at least 2 wet days per group."
                    )

                min_wet_count = min(min_wet_count, len(obs_wet), len(mod_wet))

                q_mod = np.percentile(mod_wet, probs * 100.0, method="linear")
                q_obs = np.percentile(obs_wet, probs * 100.0, method="linear")
                r = np.where(q_mod > 0, q_obs / q_mod, 1.0)
                self.quantile_transfer_nodes[target_doy] = (q_mod, q_obs, r)

        self.min_n_wet_per_group = min_wet_count if min_wet_count != 10**9 else 0
        return self

    def transform(self, mod: pd.Series) -> pd.Series:
        """Apply Level 3 EQM transfer function to model daily precipitation.

        Args:
            mod: Daily model precipitation series in mm.

        Returns:
            Corrected precipitation series with NaN values strictly preserved.
        """
        if not self.quantile_transfer_nodes:
            raise RuntimeError("Estimator has not been fitted. Call fit() first.")
        if not isinstance(mod.index, pd.DatetimeIndex):
            raise ValueError("Input series must have a pd.DatetimeIndex.")

        values = mod.to_numpy(dtype=float)
        months = mod.index.month
        thresholds = months.map(self.level2_thresholds).to_numpy(dtype=float)

        is_nan = np.isnan(values)
        res_values = np.zeros_like(values)

        if self.window == "monthly":
            group_keys = months.to_numpy()
        else:
            group_keys = np.array([date_to_leap_doy(t) for t in mod.index])

        # Apply transformation per group
        unique_groups = np.unique(group_keys[~is_nan])

        for g in unique_groups:
            q_mod, q_obs, r = self.quantile_transfer_nodes[g]
            q_mod_min = q_mod[0]
            q_obs_min = q_obs[0]
            q_mod_max = q_mod[-1]
            q_obs_max = q_obs[-1]
            q_mod_prev = q_mod[-2]
            q_obs_prev = q_obs[-2]
            r_top = r[-1]

            mask_g = (group_keys == g) & (~is_nan)
            if not np.any(mask_g):
                continue

            vals_g = values[mask_g]
            th_g = thresholds[mask_g]
            res_g = np.zeros_like(vals_g)

            # Wet days: vals >= threshold
            wet_mask = vals_g >= th_g
            if np.any(wet_mask):
                p_wet = vals_g[wet_mask]
                corr_wet = np.zeros_like(p_wet)

                # Segment 1: below first quantile node (interpolate to (0, 0))
                sub_low = p_wet <= q_mod_min
                if np.any(sub_low):
                    slope_low = (q_obs_min / q_mod_min) if q_mod_min > 0 else 1.0
                    corr_wet[sub_low] = slope_low * p_wet[sub_low]

                # Segment 2: above top quantile node
                sub_high = p_wet >= q_mod_max
                if np.any(sub_high):
                    if self.extrapolation == "constant":
                        corr_wet[sub_high] = r_top * p_wet[sub_high]
                    else:
                        # Linear extension: P_corr = Q_obs,top + s*(P - Q_mod,top)
                        denom = q_mod_max - q_mod_prev
                        if denom <= 1e-9:
                            corr_wet[sub_high] = r_top * p_wet[sub_high]
                        else:
                            slope_s = (q_obs_max - q_obs_prev) / denom
                            corr_wet[sub_high] = q_obs_max + slope_s * (
                                p_wet[sub_high] - q_mod_max
                            )

                # Segment 3: interior nodes (piecewise linear interpolation)
                sub_mid = (~sub_low) & (~sub_high)
                if np.any(sub_mid):
                    corr_wet[sub_mid] = np.interp(p_wet[sub_mid], q_mod, q_obs)

                # Count clipping below zero
                clipped_count = int(np.sum(corr_wet < 0.0))
                if clipped_count > 0:
                    self.n_clipped_below_zero += clipped_count
                    corr_wet = np.maximum(0.0, corr_wet)

                res_g[wet_mask] = corr_wet

            res_values[mask_g] = res_g

        # Preserve NaNs
        res_values = np.where(is_nan, np.nan, np.maximum(0.0, res_values))

        return pd.Series(res_values, index=mod.index, name=mod.name)


@dataclass(frozen=True)
class Level3TuningResult:
    """Diagnostic outcome of Level 3 hyperparameter grid evaluation."""

    config_id: int
    window: Literal["monthly", "moving_window_31"]
    n_quantiles: Literal[50, 100]
    extrapolation: Literal["constant", "linear"]
    overall_tuning_loss: float
    fold_tuning_losses: List[float]
    mean_fold_loss: float
    std_fold_loss: float
    min_n_wet_per_group: int
    n_clipped_below_zero: int

    @property
    def loss(self) -> float:
        """Alias for overall_tuning_loss."""
        return self.overall_tuning_loss


def select_best_level3_result(
    results: List[Level3TuningResult],
    tol: float = 1e-6,
) -> Level3TuningResult:
    """Select the best Level 3 configuration adhering to D.17 parsimony tie-breaking.

    Configurations whose overall tuning loss is within `tol` (absolute) of the
    minimum finite loss are tied; non-finite losses are ineligible. Among tied
    candidates, selects the most parsimonious configuration according to:
        window: 'monthly' > 'moving_window_31'
        n_quantiles: 50 > 100
        extrapolation: 'constant' > 'linear'
        config_id: lowest integer

    Args:
        results: List of evaluated Level3TuningResult objects.
        tol: Absolute tolerance for tying with minimum loss (default 1e-6).

    Returns:
        Best Level3TuningResult.

    Raises:
        ValueError: If results is empty or no results have finite tuning loss.
    """
    finite_results = [r for r in results if np.isfinite(r.overall_tuning_loss)]
    if not finite_results:
        raise ValueError("No finite tuning losses among candidates.")

    min_loss = min(r.overall_tuning_loss for r in finite_results)
    candidates = [r for r in finite_results if r.overall_tuning_loss <= min_loss + tol]

    def parsimony_key(r: Level3TuningResult) -> Tuple[int, int, int, int]:
        win_rank = 0 if r.window == "monthly" else 1
        bins_rank = 0 if r.n_quantiles == 50 else 1
        extrap_rank = 0 if r.extrapolation == "constant" else 1
        return (win_rank, bins_rank, extrap_rank, r.config_id)

    return min(candidates, key=parsimony_key)


def tune_level3_precipitation_hyperparameters(
    obs_cal: pd.Series,
    mod_cal: pd.Series,
) -> Tuple[
    Level3QuantileMappingPrecipitation,
    List[Level3TuningResult],
    Level3TuningResult,
]:
    """Optimize Level 3 EQM hyperparameters using nested annual block CV.

    Evaluates the 8 configurations of the discrete grid (Section 6.3):
    - window: monthly vs. moving_window_31
    - n_quantiles: 50 vs. 100
    - extrapolation: constant vs. linear

    Cross-validation folds use strictly years with at least one valid paired day.
    Tuning loss is evaluated on concatenated out-of-sample predictions.
    Ties within numerical tolerance (1e-6) are broken by parsimony:
        monthly > moving_window_31 -> 50 bins > 100 bins -> constant > linear.

    Args:
        obs_cal: Calibration daily observations in mm.
        mod_cal: Calibration daily model precipitation in mm.

    Returns:
        Tuple of (fitted_winner, all_configuration_results, winning_result).
    """
    # 1. Identify valid years with at least one paired observation
    valid_paired = (~obs_cal.isna()) & (~mod_cal.isna())
    obs_idx = cast(pd.DatetimeIndex, obs_cal.index)
    valid_years = sorted(obs_idx[valid_paired].year.unique())
    if len(valid_years) < 2:
        raise ValueError(
            "Nested annual block cross-validation requires at least 2 valid "
            f"years with paired data, found {len(valid_years)}."
        )

    configs: List[
        Tuple[
            Literal["monthly", "moving_window_31"],
            Literal[50, 100],
            Literal["constant", "linear"],
        ]
    ] = [
        ("monthly", 50, "constant"),
        ("monthly", 50, "linear"),
        ("monthly", 100, "constant"),
        ("monthly", 100, "linear"),
        ("moving_window_31", 50, "constant"),
        ("moving_window_31", 50, "linear"),
        ("moving_window_31", 100, "constant"),
        ("moving_window_31", 100, "linear"),
    ]

    results: List[Level3TuningResult] = []
    mod_idx = cast(pd.DatetimeIndex, mod_cal.index)

    for cfg_id, (win, nq, ext) in enumerate(configs):
        y_oos = pd.Series(np.nan, index=mod_cal.index, dtype=float)
        fold_losses: List[float] = []
        total_clipped = 0
        min_wet_observed = 10**9

        for val_year in valid_years:
            val_mask = mod_idx.year == val_year
            train_mask = (mod_idx.year != val_year) & (mod_idx.year.isin(valid_years))

            est = Level3QuantileMappingPrecipitation(
                window=win,
                n_quantiles=nq,
                extrapolation=ext,
            )
            est.fit(obs_cal[train_mask], mod_cal[train_mask])

            y_val_pred = est.transform(mod_cal[val_mask])
            y_oos.loc[val_mask] = y_val_pred

            # Diagnostic fold loss
            loss_fold = precipitation_tuning_loss(obs_cal[val_mask], y_val_pred)
            fold_losses.append(loss_fold)
            total_clipped += est.n_clipped_below_zero
            min_wet_observed = min(min_wet_observed, est.min_n_wet_per_group)

        # Evaluate overall tuning loss on concatenated out-of-sample series
        overall_loss = precipitation_tuning_loss(obs_cal, y_oos)
        mean_fold = float(np.mean(fold_losses))
        std_fold = float(np.std(fold_losses, ddof=1)) if len(fold_losses) > 1 else 0.0

        res = Level3TuningResult(
            config_id=cfg_id,
            window=win,
            n_quantiles=nq,
            extrapolation=ext,
            overall_tuning_loss=overall_loss,
            fold_tuning_losses=fold_losses,
            mean_fold_loss=mean_fold,
            std_fold_loss=std_fold,
            min_n_wet_per_group=min_wet_observed,
            n_clipped_below_zero=total_clipped,
        )
        results.append(res)

    best_res = select_best_level3_result(results, tol=1e-6)

    # Refit winning estimator on full calibration data
    winner_est = Level3QuantileMappingPrecipitation(
        window=best_res.window,
        n_quantiles=best_res.n_quantiles,
        extrapolation=best_res.extrapolation,
    )
    winner_est.fit(obs_cal, mod_cal)

    return winner_est, results, best_res


def load_calibration_precipitation_pair(
    station_id: str,
    dhime_path: Optional[Union[str, Path]] = None,
    era5_path_or_dir: Optional[Union[str, Path]] = None,
    cal_range: Optional[Tuple[int, int]] = None,
) -> Tuple[pd.Series, pd.Series]:
    """Load observed and model daily precipitation pairs strictly for calibration.

    Immediately discards records outside the calibration period without calculating
    anything with validation years.

    Args:
        station_id: Station code (e.g. '21205710', '23215030').
        dhime_path: Optional path to DHIME CSV file. If None, resolves default.
        era5_path_or_dir: Optional path or directory to ERA5 NetCDF/CSV files.
        cal_range: Optional explicit calibration year range (start_year, end_year).
            If None, retrieves pre-specified calibration partition from protocol.

    Returns:
        Tuple of (obs_cal, mod_cal) pandas Series with DatetimeIndex normalized
        to the exact calibration date range.
    """
    clean_id = str(station_id).strip().replace(".0", "")

    if cal_range is None:
        try:
            cal_range = get_benchmark_partitions("precipitation", clean_id)[0]
        except ValueError:
            raise ValueError(
                f"no pre-specified calibration partition for station {station_id}; "
                "pass cal_range explicitly"
            ) from None

    cal_start, cal_end = cal_range
    full_cal_index = pd.date_range(f"{cal_start}-01-01", f"{cal_end}-12-31", freq="D")

    # 1. Load Observation series
    if dhime_path is None:
        if clean_id == "21205710":
            dhime_path = Path(
                "local/newFiles/DiaPluviometrico_JardinBotanico_21205710.csv"
            )
        elif clean_id == "23215030":
            dhime_path = Path("local/newFiles/DiaPluviometrico_AguasClaras.csv")
        else:
            dhime_path = Path(f"local/newFiles/DiaPluviometrico_{clean_id}.csv")

    dhime_path = Path(dhime_path)
    if not dhime_path.exists():
        raise FileNotFoundError(f"DHIME observation file not found: {dhime_path}")

    # Read CSV
    df_raw = pd.read_csv(dhime_path)
    date_col = "Fecha" if "Fecha" in df_raw.columns else df_raw.columns[0]
    val_col = "Valor" if "Valor" in df_raw.columns else df_raw.columns[1]

    df_raw["dt"] = pd.to_datetime(df_raw[date_col], errors="coerce").dt.normalize()
    df_raw = df_raw.dropna(subset=["dt"]).sort_values("dt")
    df_raw[val_col] = pd.to_numeric(df_raw[val_col], errors="coerce")

    # Filter station if multiple stations present
    if "CodigoEstacion" in df_raw.columns:
        df_raw["st_clean"] = (
            df_raw["CodigoEstacion"]
            .astype(str)
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
        )
        if clean_id in df_raw["st_clean"].values:
            df_raw = df_raw[df_raw["st_clean"] == clean_id]

    df_raw = df_raw.drop_duplicates(subset=["dt"], keep="first")
    s_obs_full = pd.Series(
        df_raw[val_col].values,
        index=pd.DatetimeIndex(df_raw["dt"]),
        name="obs",
    )

    # Immediately discard validation years without computing anything with them
    s_obs_idx = cast(pd.DatetimeIndex, s_obs_full.index)
    obs_cal = s_obs_full[(s_obs_idx.year >= cal_start) & (s_obs_idx.year <= cal_end)]
    obs_cal = obs_cal.reindex(full_cal_index)
    obs_cal.name = "obs"

    # 2. Load Model (ERA5) series
    if era5_path_or_dir is None:
        era5_path_or_dir = Path(f"local/era5_hourly/{clean_id}")

    era5_target = Path(era5_path_or_dir)
    if not era5_target.exists():
        raise FileNotFoundError(f"ERA5 input path not found: {era5_target}")

    if era5_target.is_file() and era5_target.suffix.lower() == ".csv":
        # CSV format model file
        df_mod_raw = pd.read_csv(era5_target)
        m_date_col = "Fecha" if "Fecha" in df_mod_raw.columns else df_mod_raw.columns[0]
        m_val_col = (
            "Valor"
            if "Valor" in df_mod_raw.columns
            else ("mod" if "mod" in df_mod_raw.columns else df_mod_raw.columns[1])
        )
        df_mod_raw["dt"] = pd.to_datetime(
            df_mod_raw[m_date_col], errors="coerce"
        ).dt.normalize()
        df_mod_raw = df_mod_raw.dropna(subset=["dt"]).sort_values("dt")
        df_mod_raw[m_val_col] = pd.to_numeric(df_mod_raw[m_val_col], errors="coerce")
        s_mod_full = pd.Series(
            df_mod_raw[m_val_col].values,
            index=pd.DatetimeIndex(df_mod_raw["dt"]),
            name="mod",
        )
    else:
        # Directory or NetCDF file pattern
        from tropicor.io.era5 import ERA5Adapter
        from tropicor.io.stations import StationCatalog

        if era5_target.is_dir():
            nc_files = sorted(
                glob.glob(str(era5_target / "era5_total_precipitation_*.nc"))
            )
            if not nc_files:
                nc_files = sorted(glob.glob(str(era5_target / "*.nc")))
            if not nc_files:
                raise FileNotFoundError(
                    f"No NetCDF precipitation files found in {era5_target}"
                )
            adapter = ERA5Adapter(nc_files)
        else:
            adapter = ERA5Adapter(era5_target)

        catalog = StationCatalog.from_benchmark()
        st_meta = catalog.get(clean_id)
        if st_meta is not None:
            lat, lon = st_meta.latitude, st_meta.longitude
        else:
            lat = float(adapter.dataset.latitude.mean().values)
            lon = float(adapter.dataset.longitude.mean().values)

        s_mod_full = adapter.get_daily_precipitation(
            latitude=lat,
            longitude=lon,
            method="nearest",
            window_start_utc=12,
            min_hours=24,
            lag_days=0,
        )

    # Immediately discard validation years without computing anything with them
    s_mod_idx = cast(pd.DatetimeIndex, s_mod_full.index)
    mod_cal = s_mod_full[(s_mod_idx.year >= cal_start) & (s_mod_idx.year <= cal_end)]
    mod_cal = mod_cal.reindex(full_cal_index)
    mod_cal.name = "mod"

    return obs_cal, mod_cal
