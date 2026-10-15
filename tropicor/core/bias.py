"""Classical statistical bias correction for tropical climate reanalyses.

Implements Linear Scaling and Empirical Quantile Mapping (EQM) tailored for
monthly and multi-annual climate series in complex tropical terrain.

Unlike daily downscaling, monthly aggregated records require:
1. Sample-size bounds on quantile estimation (preventing overfitting on ~30 points).
2. Moving-window sample expansion (pooling 3-month seasonal windows during fitting,
   while applying corrections strictly to the target month).
3. Physical constraints by variable (multiplicative non-negative scaling for
   precipitation, additive scaling for temperature).
4. Safe handling of zero-model precipitation without mixing additive fallbacks.
"""

import warnings
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from tropicor.core.alignment import align_common_period

VariableType = Literal["temperature", "precipitation"]
GroupingType = Literal["monthly", "moving_window", "pooled"]
ScalingMethod = Literal["additive", "multiplicative"]
ExtrapolationType = Literal["constant", "linear", "delta"]


def _get_window_months(target_month: int, window_size: int = 3) -> List[int]:
    """Return circular list of calendar months centered on target_month.

    For example, for target_month=1 (January) and window_size=3, returns [12, 1, 2].

    Args:
        target_month: Calendar month (1 to 12).
        window_size: Number of months in window (must be an odd integer >= 1).

    Returns:
        List of integer calendar months (1 to 12).
    """
    if window_size % 2 == 0 or window_size < 1:
        raise ValueError(
            f"window_size must be an odd positive integer, got {window_size}."
        )
    half = window_size // 2
    months: List[int] = []
    for offset in range(-half, half + 1):
        m = (target_month - 1 + offset) % 12 + 1
        months.append(m)
    return months


class LinearScalingCorrection:
    """Parametric linear scaling correction with monthly or moving-window grouping.

    Adjusts modeled series to match the mean of observed ground truth:
    - **Additive scaling** (standard for temperature):
      :math:`x_{\\text{corr}}(t) = x_{\\text{mod}}(t) +
      (\\bar{x}_{\\text{obs}, m} - \\bar{x}_{\\text{mod}, m})`
    - **Multiplicative scaling** (standard for precipitation):
      :math:`x_{\\text{corr}}(t) = x_{\\text{mod}}(t) \\times
      \\frac{\\bar{x}_{\\text{obs}, m}}{\\bar{x}_{\\text{mod}, m}}`

    Preserves in-sample seasonal volume exactly under ``grouping='monthly'``
    or ``grouping='pooled'``. Under ``grouping='moving_window'``, scaling factors
    are fitted on the multi-month window but applied strictly to the target month.
    """

    def __init__(
        self,
        variable: VariableType = "temperature",
        method: Optional[ScalingMethod] = None,
        grouping: Optional[GroupingType] = None,
        window_size: int = 3,
        min_threshold: Optional[float] = None,
    ) -> None:
        """Initialize LinearScalingCorrection.

        Args:
            variable: Climate variable kind ("temperature" or "precipitation").
                Defaults to "temperature".
            method: Explicit scaling method ("additive" or "multiplicative").
                If None, defaults to "multiplicative" for precipitation and
                "additive" for temperature.
            grouping: Grouping strategy ("monthly", "moving_window", or "pooled").
                If None, defaults to "monthly" (matches calendar-month means
                and preserves annual-cycle amplitude). Note: defaults rest on
                limited exploratory evidence (4 stations, thesis files pending
                primary source audit) and may be revised upon rebuilding primary data.
            window_size: Circular window size in months for moving_window
                (default 3). Must be odd.
            min_threshold: Lower bound clip value. If None, defaults to 0.0 for
                precipitation and None for temperature.
        """
        self.variable = variable
        if method is None:
            self.method: ScalingMethod = (
                "multiplicative" if variable == "precipitation" else "additive"
            )
        else:
            self.method = method

        if grouping is None:
            self.grouping: GroupingType = "monthly"
        else:
            self.grouping = grouping
        self.window_size = window_size

        if min_threshold is None and self.variable == "precipitation":
            self.min_threshold: Optional[float] = 0.0
        else:
            self.min_threshold = min_threshold

        self.factors_: Dict[int, float] = {}
        self.is_fitted_: bool = False

    def fit(self, obs: pd.Series, mod: pd.Series) -> "LinearScalingCorrection":
        """Fit scaling factors or offsets on calibration series.

        Args:
            obs: Ground-truth observed series with DatetimeIndex.
            mod: Modeled/reanalysis series with DatetimeIndex.

        Returns:
            Fitted instance of LinearScalingCorrection.

        Raises:
            ValueError: If series lacks DatetimeIndex or has no synchronous pairs.
        """
        obs_sync, mod_sync = align_common_period(obs, mod, drop_na=True)
        if len(obs_sync) == 0:
            raise ValueError(
                "No common synchronous observations found to fit "
                "LinearScalingCorrection."
            )
        assert isinstance(obs_sync.index, pd.DatetimeIndex)

        self.factors_.clear()

        if self.grouping == "pooled":
            factor = self._compute_factor(
                obs_sync.to_numpy(), mod_sync.to_numpy(), group_id=0
            )
            self.factors_[0] = factor

        elif self.grouping == "monthly":
            for m in range(1, 13):
                mask = obs_sync.index.month == m
                obs_m = obs_sync[mask].to_numpy()
                mod_m = mod_sync[mask].to_numpy()
                self.factors_[m] = self._compute_factor(obs_m, mod_m, group_id=m)

        elif self.grouping == "moving_window":
            for m in range(1, 13):
                window_months = _get_window_months(m, self.window_size)
                mask = obs_sync.index.month.isin(window_months)
                obs_win = obs_sync[mask].to_numpy()
                mod_win = mod_sync[mask].to_numpy()
                # Fitted on window, stored for target month m
                self.factors_[m] = self._compute_factor(obs_win, mod_win, group_id=m)
        else:
            raise ValueError(f"Unknown grouping mode: '{self.grouping}'.")

        self.is_fitted_ = True
        return self

    def _compute_factor(
        self, obs_arr: np.ndarray, mod_arr: np.ndarray, group_id: int
    ) -> float:
        """Compute additive delta or multiplicative ratio for a sample subset."""
        if len(obs_arr) == 0 or len(mod_arr) == 0:
            warnings.warn(
                f"No valid observations for group {group_id}. "
                "Setting scaling factor to default neutral value.",
                UserWarning,
            )
            return 0.0 if self.method == "additive" else 1.0

        mean_obs = float(np.mean(obs_arr))
        mean_mod = float(np.mean(mod_arr))

        if self.method == "additive":
            return mean_obs - mean_mod
        else:  # multiplicative
            if abs(mean_mod) < 1e-12:
                warnings.warn(
                    f"Modeled mean is ~0.0 for group {group_id}. Multiplicative "
                    "scaling factor cannot be computed; leaving values unscaled "
                    "(factor=1.0).",
                    UserWarning,
                )
                return 1.0
            return mean_obs / mean_mod

    def transform(self, mod: pd.Series) -> pd.Series:
        """Apply fitted linear scaling to a modeled series.

        Args:
            mod: Modeled series with a pandas DatetimeIndex.

        Returns:
            Corrected series with identical DatetimeIndex and name.

        Raises:
            RuntimeError: If called before fit().
            ValueError: If series lacks DatetimeIndex.
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "LinearScalingCorrection instance is not fitted yet. Call fit() first."
            )
        if not isinstance(mod.index, pd.DatetimeIndex):
            raise ValueError("mod series must have a pandas DatetimeIndex.")

        result = mod.copy().astype(float)

        if self.grouping == "pooled":
            factor = self.factors_[0]
            if self.method == "additive":
                result = result + factor
            else:
                result = result * factor
        else:
            # Both monthly and moving_window apply factors strictly by month
            for m in range(1, 13):
                mask = mod.index.month == m
                if not mask.any():
                    continue
                factor = self.factors_.get(m, 0.0 if self.method == "additive" else 1.0)
                if self.method == "additive":
                    result.loc[mask] = result.loc[mask] + factor
                else:
                    result.loc[mask] = result.loc[mask] * factor

        if self.min_threshold is not None:
            result = result.clip(lower=self.min_threshold)

        return result

    def fit_transform(self, obs: pd.Series, mod: pd.Series) -> pd.Series:
        """Fit scaling factors on calibration series and return corrected mod."""
        return self.fit(obs, mod).transform(mod)


class EmpiricalQuantileMapping:
    """Non-parametric Empirical Quantile Mapping (EQM) for distribution adjustment.

    Maps modeled cumulative distribution functions (CDFs) to observed distributions
    via empirical quantile transfer functions:
    :math:`x_{\\text{corr}} = F_{\\text{obs}}^{-1}(F_{\\text{mod}}(x_{\\text{mod}}))`

    Key Features:
    - Automatically caps the number of quantiles to the available sample size
      (:math:`n \\le N_{\\text{group}}`) to avoid interpolating noise on small samples.
    - Supports moving-window sample pooling (:math:`N \\approx 90`) during fitting,
      while applying corrections strictly to the target month.
    - Physical constraints by variable: strict non-negativity and conservative
      extrapolation ("constant" or "delta") for precipitation, avoiding unphysical
      explosive linear extremes.
    """

    def __init__(
        self,
        variable: VariableType = "temperature",
        grouping: Optional[GroupingType] = None,
        window_size: int = 3,
        n_quantiles: Optional[int] = None,
        extrapolation: Optional[ExtrapolationType] = None,
        dry_threshold: Optional[float] = None,
    ) -> None:
        """Initialize EmpiricalQuantileMapping.

        Args:
            variable: "temperature" (extrapolation="linear") or "precipitation"
                (extrapolation="constant", clipped at >= 0.0).
            grouping: "monthly", "moving_window" (3-month window), or "pooled".
                If None, defaults to variable-aware exploratory defaults:
                - "monthly" for precipitation (preserves annual cycle amplitude);
                - "moving_window" for temperature (avoids small-sample tail
                  overfitting that distorts monthly means and annual cycle amplitude).
                Note: Defaults rest on limited exploratory evidence (4 stations,
                thesis files pending primary source audit) and may be revised
                upon rebuilding primary data.
            window_size: Circular window size in months for moving_window
                (default 3). Must be odd.
            n_quantiles: Number of quantile bins to evaluate. Capped automatically
                at the sample size of each grouping.
            extrapolation: Handling out-of-sample tail extremes: "constant",
                "linear", or "delta". Defaults to "constant" for precipitation
                and "linear" for temperature.
            dry_threshold: Daily-scale threshold in mm. For monthly series,
                must be None or 0.0; emits a UserWarning if a positive
                threshold is passed.
        """
        self.variable = variable
        if grouping is None:
            self.grouping: GroupingType = (
                "monthly" if variable == "precipitation" else "moving_window"
            )
        else:
            self.grouping = grouping
        self.window_size = window_size
        self.n_quantiles = n_quantiles

        if extrapolation is None:
            self.extrapolation: ExtrapolationType = (
                "constant" if variable == "precipitation" else "linear"
            )
        else:
            self.extrapolation = extrapolation

        if dry_threshold is not None and dry_threshold > 0:
            warnings.warn(
                f"dry_threshold={dry_threshold} mm is a daily-scale concept "
                "and not applicable to monthly accumulated precipitation "
                "totals. Thresholding is deactivated.",
                UserWarning,
            )
            self.dry_threshold: Optional[float] = None
        else:
            self.dry_threshold = dry_threshold

        # Dictionary of fitted transfer data per group:
        # group_id -> {"prob": ndarray, "q_mod": ndarray, "q_obs": ndarray}
        self.transfer_tables_: Dict[int, Dict[str, np.ndarray]] = {}
        self.is_fitted_: bool = False

    def fit(self, obs: pd.Series, mod: pd.Series) -> "EmpiricalQuantileMapping":
        """Fit empirical quantile transfer functions on calibration series.

        Args:
            obs: Ground-truth observed series with DatetimeIndex.
            mod: Modeled/reanalysis series with DatetimeIndex.

        Returns:
            Fitted instance of EmpiricalQuantileMapping.

        Raises:
            ValueError: If series lacks DatetimeIndex or has no synchronous pairs.
        """
        obs_sync, mod_sync = align_common_period(obs, mod, drop_na=True)
        if len(obs_sync) == 0:
            raise ValueError(
                "No common synchronous observations found to fit "
                "EmpiricalQuantileMapping."
            )
        assert isinstance(obs_sync.index, pd.DatetimeIndex)

        self.transfer_tables_.clear()

        if self.grouping == "pooled":
            table = self._fit_group(
                obs_sync.to_numpy(), mod_sync.to_numpy(), group_id=0
            )
            self.transfer_tables_[0] = table

        elif self.grouping == "monthly":
            for m in range(1, 13):
                mask = obs_sync.index.month == m
                obs_m = obs_sync[mask].to_numpy()
                mod_m = mod_sync[mask].to_numpy()
                self.transfer_tables_[m] = self._fit_group(obs_m, mod_m, group_id=m)

        elif self.grouping == "moving_window":
            for m in range(1, 13):
                window_months = _get_window_months(m, self.window_size)
                mask = obs_sync.index.month.isin(window_months)
                obs_win = obs_sync[mask].to_numpy()
                mod_win = mod_sync[mask].to_numpy()
                # Fitted on window, stored for target month m
                self.transfer_tables_[m] = self._fit_group(obs_win, mod_win, group_id=m)
        else:
            raise ValueError(f"Unknown grouping mode: '{self.grouping}'.")

        self.is_fitted_ = True
        return self

    def _fit_group(
        self, obs_arr: np.ndarray, mod_arr: np.ndarray, group_id: int
    ) -> Dict[str, np.ndarray]:
        """Compute empirical quantile mapping table for a single group."""
        n_obs = len(obs_arr)
        n_mod = len(mod_arr)

        if n_obs < 2 or n_mod < 2:
            warnings.warn(
                f"Insufficient samples (n_obs={n_obs}, n_mod={n_mod}) for "
                f"group {group_id}. Fitting identity transfer function.",
                UserWarning,
            )
            return {
                "prob": np.array([0.0, 1.0]),
                "q_mod": np.array([0.0, 1.0]),
                "q_obs": np.array([0.0, 1.0]),
            }

        eff_n = min(n_obs, n_mod)
        # Cap n_quantiles at the effective sample size
        k = eff_n if self.n_quantiles is None else min(self.n_quantiles, eff_n)
        k = max(2, k)

        probabilities = np.linspace(0.0, 1.0, k)
        q_obs = np.quantile(obs_arr, probabilities)
        q_mod = np.quantile(mod_arr, probabilities)

        return {
            "prob": probabilities,
            "q_mod": q_mod,
            "q_obs": q_obs,
        }

    def _map_values(
        self, values: np.ndarray, table: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Map model values through empirical quantile table."""
        q_mod = table["q_mod"]
        q_obs = table["q_obs"]

        # Deduplicate q_mod for robust monotonicity during interpolation
        unique_mod, unique_indices = np.unique(q_mod, return_index=True)
        unique_obs = q_obs[unique_indices]

        if len(unique_mod) < 2:
            # Degenerate (constant) model distribution in this group.
            if self.variable == "precipitation":
                mean_mod = float(np.mean(q_mod))
                if abs(mean_mod) < 1e-12:
                    warnings.warn(
                        "Constant zero model distribution in group; "
                        "leaving values unchanged.",
                        UserWarning,
                    )
                    return values.copy()
                return values * (float(np.mean(q_obs)) / mean_mod)
            offset = float(np.mean(q_obs) - np.mean(q_mod))
            return values + offset

        # Standard piecewise linear interpolation within calibration domain
        mapped = np.interp(values, unique_mod, unique_obs)

        # Handle out-of-range tail extremes according to extrapolation policy
        lower_mask = values < unique_mod[0]
        upper_mask = values > unique_mod[-1]

        if self.extrapolation == "constant":
            # np.interp clamps to unique_obs[0] and unique_obs[-1] by default
            pass

        elif self.extrapolation == "delta":
            if self.variable == "precipitation":
                # Multiplicative delta: preserve the model's relative
                # exceedance of the calibration boundary (ratio, not offset).
                # Lower tail stays clamped (ratio below a ~0 boundary is
                # ill-defined and output is clipped at 0 anyway).
                if upper_mask.any() and unique_mod[-1] > 1e-12:
                    ratio = values[upper_mask] / unique_mod[-1]
                    mapped[upper_mask] = unique_obs[-1] * ratio
            else:
                # Additive delta: preserve the model's absolute deviation
                if lower_mask.any():
                    delta_lower = values[lower_mask] - unique_mod[0]
                    mapped[lower_mask] = unique_obs[0] + delta_lower
                if upper_mask.any():
                    delta_upper = values[upper_mask] - unique_mod[-1]
                    mapped[upper_mask] = unique_obs[-1] + delta_upper

        elif self.extrapolation == "linear":
            # Linear slope extrapolation based on boundary segments
            if lower_mask.any():
                slope_lower = (unique_obs[1] - unique_obs[0]) / (
                    unique_mod[1] - unique_mod[0] + 1e-12
                )
                mapped[lower_mask] = unique_obs[0] + slope_lower * (
                    values[lower_mask] - unique_mod[0]
                )
            if upper_mask.any():
                slope_upper = (unique_obs[-1] - unique_obs[-2]) / (
                    unique_mod[-1] - unique_mod[-2] + 1e-12
                )
                mapped[upper_mask] = unique_obs[-1] + slope_upper * (
                    values[upper_mask] - unique_mod[-1]
                )

        if self.variable == "precipitation":
            mapped = np.maximum(0.0, mapped)

        return mapped

    def transform(self, mod: pd.Series) -> pd.Series:
        """Apply fitted Empirical Quantile Mapping to a modeled series.

        Args:
            mod: Modeled series with a pandas DatetimeIndex.

        Returns:
            Corrected series with identical DatetimeIndex and name.

        Raises:
            RuntimeError: If called before fit().
            ValueError: If series lacks DatetimeIndex.
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "EmpiricalQuantileMapping instance is not fitted yet. Call fit() first."
            )
        if not isinstance(mod.index, pd.DatetimeIndex):
            raise ValueError("mod series must have a pandas DatetimeIndex.")

        result = mod.copy().astype(float)

        if self.grouping == "pooled":
            table = self.transfer_tables_[0]
            valid_mask = result.notna()
            if valid_mask.any():
                vals = result.loc[valid_mask].to_numpy()
                result.loc[valid_mask] = self._map_values(vals, table)
        else:
            # Both monthly and moving_window apply transfer functions by month
            for m in range(1, 13):
                mask = (mod.index.month == m) & result.notna()
                if not mask.any():
                    continue
                table_m = self.transfer_tables_.get(m)
                if table_m is None:
                    continue
                vals = result.loc[mask].to_numpy()
                result.loc[mask] = self._map_values(vals, table_m)

        if self.variable == "precipitation":
            result = result.clip(lower=0.0)

        return result

    def fit_transform(self, obs: pd.Series, mod: pd.Series) -> pd.Series:
        """Fit quantile transfer functions on calibration series and transform mod."""
        return self.fit(obs, mod).transform(mod)


def linear_scale_series(
    obs: pd.Series,
    mod: pd.Series,
    variable: VariableType = "temperature",
    method: Optional[ScalingMethod] = None,
    grouping: Optional[GroupingType] = None,
    window_size: int = 3,
) -> pd.Series:
    """Convenience functional wrapper for LinearScalingCorrection."""
    scaler = LinearScalingCorrection(
        variable=variable,
        method=method,
        grouping=grouping,
        window_size=window_size,
    )
    return scaler.fit_transform(obs, mod)


def quantile_map_series(
    obs: pd.Series,
    mod: pd.Series,
    variable: VariableType = "temperature",
    grouping: Optional[GroupingType] = None,
    window_size: int = 3,
    n_quantiles: Optional[int] = None,
    extrapolation: Optional[ExtrapolationType] = None,
) -> pd.Series:
    """Convenience functional wrapper for EmpiricalQuantileMapping."""
    eqm = EmpiricalQuantileMapping(
        variable=variable,
        grouping=grouping,
        window_size=window_size,
        n_quantiles=n_quantiles,
        extrapolation=extrapolation,
    )
    return eqm.fit_transform(obs, mod)
