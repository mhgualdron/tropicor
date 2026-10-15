"""Physics-based orographic and topographic vertical lapse-rate corrections.

This module implements Layer 0 leaf utilities for vertical temperature downscaling
based on environmental lapse-rate adjustments. It bridges the vertical elevation
discrepancies between coarse atmospheric reanalysis grids (e.g., ERA5 ~30 km) and
high-relief ground-truth observation stations in complex topography such as the
Colombian Andes.
"""

from typing import Any, Dict, Optional, Union, overload

import numpy as np
import pandas as pd
from scipy import stats

STANDARD_LAPSE_RATE: float = 0.0065
"""Standard environmental lapse rate in the free troposphere (0.0065 °C/m)."""

DRY_ADIABATIC_LAPSE_RATE: float = 0.0098
"""Dry adiabatic lapse rate (g / c_p ≈ 0.0098 °C/m or 9.8 °C/km)."""

MOIST_ADIABATIC_LAPSE_RATE: float = 0.0050
"""Typical moist/saturated pseudoadiabatic lapse rate (≈ 0.0050 °C/m)."""

STANDARD_GRAVITY: float = 9.80665
"""Standard acceleration due to gravity (WMO standard g_0 = 9.80665 m/s^2)."""


@overload
def geopotential_to_height(
    geopotential: float,
    gravity: float = ...,
) -> float: ...


@overload
def geopotential_to_height(
    geopotential: np.ndarray,
    gravity: float = ...,
) -> np.ndarray: ...


@overload
def geopotential_to_height(
    geopotential: pd.Series,
    gravity: float = ...,
) -> pd.Series: ...


def geopotential_to_height(
    geopotential: Union[float, np.ndarray, pd.Series],
    gravity: float = STANDARD_GRAVITY,
) -> Union[float, np.ndarray, pd.Series]:
    """Convert surface geopotential (m^2/s^2) to geopotential height in meters (masl).

    In atmospheric reanalyses such as ECMWF ERA5, surface topography is stored as
    surface geopotential Phi (variable 'z') in units of m^2/s^2 (J/kg). The geometric
    or geopotential height h in meters above sea level (masl) is derived by dividing
    by the standard gravitational acceleration:
        h = Phi / g_0
    where g_0 = 9.80665 m/s^2.

    Args:
        geopotential: Geopotential value, array, or Series in m^2/s^2.
        gravity: Gravitational acceleration in m/s^2 (default: 9.80665 m/s^2).

    Returns:
        Geopotential height in meters (masl) matching the input type.

    Raises:
        ValueError: If gravity is non-positive.
    """
    if gravity <= 0:
        raise ValueError(f"Gravity must be positive. Got {gravity}.")

    if isinstance(geopotential, (pd.Series, np.ndarray)):
        return geopotential / gravity
    return float(geopotential) / gravity


def compute_elevation_offset(
    station_elevation: float,
    model_elevation: float,
) -> float:
    """Compute the topographic elevation discrepancy between station and model grid.

    Calculates the vertical elevation difference:
        Δz = z_station - z_model

    A negative discrepancy (Δz < 0) indicates that the station lies below the smoothed
    model topography (e.g., deep Andean valleys or plateaus like Bucaramanga), meaning
    the station is physically lower and expected to be warmer than raw model output.
    A positive discrepancy (Δz > 0) indicates that the station sits above the model
    topography (e.g., mountain summits or paramos).

    Args:
        station_elevation: Ground station elevation in meters above sea level (masl).
        model_elevation: Model grid surface elevation in meters above sea level (masl).

    Returns:
        Elevation offset in meters (z_station - z_model).

    Raises:
        ValueError: If either elevation is not a finite numeric value.
    """
    if not (np.isfinite(station_elevation) and np.isfinite(model_elevation)):
        msg = (
            f"Elevations must be finite numbers. "
            f"Got station={station_elevation}, model={model_elevation}."
        )
        raise ValueError(msg)

    return float(station_elevation - model_elevation)


@overload
def lapse_rate_temperature_correction(
    temperature: pd.Series,
    station_elevation: float,
    model_elevation: float,
    lapse_rate: Union[float, pd.Series, np.ndarray] = STANDARD_LAPSE_RATE,
    allow_inversion: bool = False,
) -> pd.Series: ...


@overload
def lapse_rate_temperature_correction(
    temperature: pd.DataFrame,
    station_elevation: float,
    model_elevation: float,
    lapse_rate: Union[float, pd.Series, np.ndarray] = STANDARD_LAPSE_RATE,
    allow_inversion: bool = False,
) -> pd.DataFrame: ...


@overload
def lapse_rate_temperature_correction(
    temperature: np.ndarray,
    station_elevation: float,
    model_elevation: float,
    lapse_rate: Union[float, pd.Series, np.ndarray] = STANDARD_LAPSE_RATE,
    allow_inversion: bool = False,
) -> np.ndarray: ...


@overload
def lapse_rate_temperature_correction(
    temperature: float,
    station_elevation: float,
    model_elevation: float,
    lapse_rate: Union[float, pd.Series, np.ndarray] = STANDARD_LAPSE_RATE,
    allow_inversion: bool = False,
) -> float: ...


def lapse_rate_temperature_correction(
    temperature: Union[pd.Series, pd.DataFrame, np.ndarray, float],
    station_elevation: float,
    model_elevation: float,
    lapse_rate: Union[float, pd.Series, np.ndarray] = STANDARD_LAPSE_RATE,
    allow_inversion: bool = False,
) -> Union[pd.Series, pd.DataFrame, np.ndarray, float]:
    """Adjust reanalysis temperature for elevation discrepancy via lapse rate.

    Applies the first-order physical vertical adjustment:
        Δz = z_station - z_model
        ΔT = -Γ · Δz
        T_corrected = T_raw + ΔT = T_raw - Γ · (z_station - z_model)

    Where:
        - z_station: Real ground elevation (masl).
        - z_model: Smoothed reanalysis elevation (masl).
        - Γ: Environmental lapse rate in °C/m (default: 0.0065 °C/m).

    Mathematical & Statistical Properties:
        - Affine Transformation: For a constant lapse rate, adding ΔT is a rigid scalar
          shift across the time series. Consequently, the Pearson correlation
          coefficient r is mathematically invariant:
          r(T_raw + ΔT, T_obs) = r(T_raw, T_obs).
        - Error Reduction: The adjustment drastically reduces systematic Mean Bias
          (driving it toward 0 °C), Root Mean Square Error (RMSE), Mean Absolute
          Error (MAE), and Euclidean distance between climatological curves.

    Physical Scope & Limitations:
        - This constant lapse-rate adjustment represents a first-order
          free-tropospheric model. It assumes a uniform linear decrease of temperature
          with altitude.
        - In deeply incised Andean valleys and plateaus, nocturnal radiative cooling
          and katabatic drainage frequently generate cold-air pooling and thermal
          inversions where the local boundary-layer lapse rate flattens or reverses
          (Γ ≤ 0).
        - Consequently, while uniform lapse-rate adjustments resolve systematic bias
          for daily mean and maximum temperatures, they may over-correct minimum
          temperatures, which motivates non-linear, terrain-aware ML downscaling.

    Args:
        temperature: Uncorrected temperature data in degrees Celsius (°C). Supports
            pandas Series, pandas DataFrame, numpy ndarray, or a single float.
        station_elevation: True ground elevation in meters above sea level (masl).
        model_elevation: Model grid or reanalysis elevation in masl.
        lapse_rate: Vertical environmental lapse rate in °C/m. Defaults to 0.0065 °C/m
            (6.5 °C/km). Can be provided as a scalar float or a time-varying
            Series/array.
        allow_inversion: If False (default), raises ValueError if lapse_rate < 0,
            ensuring unintended inverted lapse rates are caught unless configured.

    Returns:
        Corrected temperature in degrees Celsius, matching the input data structure,
        index, column headers, and data type.

    Raises:
        ValueError: If elevations are non-finite, if lapse_rate is negative and
            allow_inversion is False, or if input dimensions/shapes are incompatible.
        TypeError: If temperature input is not a supported type.
    """
    delta_z = compute_elevation_offset(station_elevation, model_elevation)

    if isinstance(lapse_rate, (int, float)):
        if not np.isfinite(lapse_rate):
            msg = f"Lapse rate must be finite. Got {lapse_rate}."
            raise ValueError(msg)
        if lapse_rate < 0 and not allow_inversion:
            msg = (
                f"Negative lapse rate ({lapse_rate} °C/m) detected. Inverted lapse "
                "rates indicate temperature increasing with height. "
                "Set allow_inversion=True to permit."
            )
            raise ValueError(msg)
    elif isinstance(lapse_rate, (pd.Series, np.ndarray)):
        if not allow_inversion and np.any(np.asarray(lapse_rate) < 0):
            msg = (
                "Negative lapse rate detected in series/array. "
                "Set allow_inversion=True to permit."
            )
            raise ValueError(msg)

    # Physical adjustment: delta_t = -1.0 * lapse_rate * delta_z
    # For UIS (station=898, model=2118, delta_z=-1220, gamma=0.0065):
    # delta_t = -0.0065 * (-1220) = +7.93 °C
    delta_t = -1.0 * lapse_rate * delta_z

    if isinstance(temperature, pd.Series):
        corrected_series = temperature + delta_t
        corrected_series.name = temperature.name
        return corrected_series

    if isinstance(temperature, pd.DataFrame):
        if isinstance(delta_t, pd.Series):
            # Align by index across DataFrame columns
            return temperature.add(delta_t, axis=0)
        return temperature + delta_t

    if isinstance(temperature, np.ndarray):
        return np.asarray(temperature + delta_t)

    if isinstance(temperature, (int, float)):
        if not np.isfinite(temperature):
            return float(temperature)
        if isinstance(delta_t, (pd.Series, np.ndarray)):
            msg = "Cannot apply time-varying vector lapse rate to a scalar temperature."
            raise ValueError(msg)
        return float(temperature + delta_t)

    msg = (
        f"Unsupported temperature type: {type(temperature)}. "
        "Expected Series, DataFrame, ndarray, or float."
    )
    raise TypeError(msg)


altitude_temperature_adjustment = lapse_rate_temperature_correction
"""Alias for lapse_rate_temperature_correction emphasizing altitude adjustment."""


def compute_orographic_bias_report(
    y_obs: Union[pd.Series, np.ndarray],
    y_mod_raw: Union[pd.Series, np.ndarray],
    station_elevation: float,
    model_elevation: float,
    lapse_rate: float = STANDARD_LAPSE_RATE,
    station_name: Optional[str] = None,
    allow_inversion: bool = False,
) -> Dict[str, Any]:
    """Compute and format temperature bias before and after altitude adjustment.

    Calculates:
        - Elevation discrepancy Δz and temperature adjustment ΔT = -Γ * Δz
        - Bias, MAE, RMSE, and Pearson r for raw model temperature
        - Bias, MAE, RMSE, and Pearson r for altitude-adjusted temperature
        - Percentage reduction in absolute bias and RMSE
        - Summary comparison table (DataFrame and formatted report text)

    Args:
        y_obs: Observed temperature series in °C.
        y_mod_raw: Raw model temperature series in °C.
        station_elevation: Station ground elevation in masl.
        model_elevation: Model grid ground elevation in masl.
        lapse_rate: Vertical environmental lapse rate in °C/m (default 0.0065).
        station_name: Optional descriptive station label.
        allow_inversion: Whether to allow negative lapse rates.

    Returns:
        Dict containing scalar metrics, comparison DataFrame, and formatted text report.
    """
    delta_z = compute_elevation_offset(station_elevation, model_elevation)
    delta_t = -1.0 * lapse_rate * delta_z

    if isinstance(y_obs, pd.Series) and isinstance(y_mod_raw, pd.Series):
        df_sync = pd.concat(
            [y_obs.rename("obs"), y_mod_raw.rename("raw")], axis=1
        ).dropna()
        obs_arr = df_sync["obs"].to_numpy(dtype=float)
        raw_arr = df_sync["raw"].to_numpy(dtype=float)
    else:
        obs_arr = np.asarray(y_obs, dtype=float)
        raw_arr = np.asarray(y_mod_raw, dtype=float)
        mask = np.isfinite(obs_arr) & np.isfinite(raw_arr)
        obs_arr = obs_arr[mask]
        raw_arr = raw_arr[mask]

    n_samples = len(obs_arr)
    if n_samples < 2:
        raise ValueError(f"Insufficient synchronous valid samples: N={n_samples}")

    corr_arr = lapse_rate_temperature_correction(
        raw_arr,
        station_elevation=station_elevation,
        model_elevation=model_elevation,
        lapse_rate=lapse_rate,
        allow_inversion=allow_inversion,
    )

    raw_bias = float(np.mean(raw_arr - obs_arr))
    raw_mae = float(np.mean(np.abs(raw_arr - obs_arr)))
    raw_rmse = float(np.sqrt(np.mean((raw_arr - obs_arr) ** 2)))
    raw_r = (
        float(stats.pearsonr(raw_arr, obs_arr)[0])
        if np.std(raw_arr) > 1e-12 and np.std(obs_arr) > 1e-12
        else 0.0
    )

    adj_bias = float(np.mean(corr_arr - obs_arr))
    adj_mae = float(np.mean(np.abs(corr_arr - obs_arr)))
    adj_rmse = float(np.sqrt(np.mean((corr_arr - obs_arr) ** 2)))
    adj_r = (
        float(stats.pearsonr(corr_arr, obs_arr)[0])
        if np.std(corr_arr) > 1e-12 and np.std(obs_arr) > 1e-12
        else 0.0
    )

    bias_reduction_pct = (
        float((abs(raw_bias) - abs(adj_bias)) / abs(raw_bias) * 100.0)
        if abs(raw_bias) > 1e-6
        else 0.0
    )
    rmse_reduction_pct = (
        float((raw_rmse - adj_rmse) / raw_rmse * 100.0)
        if raw_rmse > 1e-6
        else 0.0
    )

    mae_diff = raw_mae - adj_mae
    mae_reduction_pct = (
        float(mae_diff / raw_mae * 100.0)
        if raw_mae > 1e-6
        else 0.0
    )

    df_comparison = pd.DataFrame(
        [
            {
                "Stage": "Raw Reanalysis",
                "Bias (°C)": round(raw_bias, 3),
                "MAE (°C)": round(raw_mae, 3),
                "RMSE (°C)": round(raw_rmse, 3),
                "Pearson r": round(raw_r, 4),
            },
            {
                "Stage": "Altitude Adjusted",
                "Bias (°C)": round(adj_bias, 3),
                "MAE (°C)": round(adj_mae, 3),
                "RMSE (°C)": round(adj_rmse, 3),
                "Pearson r": round(adj_r, 4),
            },
        ]
    )

    st_label = f" ({station_name})" if station_name else ""
    rate_str = f"{lapse_rate * 1000.0:.2f} °C/km ({lapse_rate:.6f} °C/m)"
    report_text = (
        "======================================================================\n"
        f"TOPOGRAPHIC ALTITUDE ADJUSTMENT REPORT{st_label}\n"
        "======================================================================\n"
        f"Station elevation:     {station_elevation:.1f} masl\n"
        f"Model grid elevation:  {model_elevation:.1f} masl\n"
        f"Discrepancy (Delta_z): {delta_z:+.1f} m\n"
        f"Lapse rate (Gamma):    {rate_str}\n"
        f"Temperature shift (dT):{delta_t:+.2f} °C\n"
        f"Synchronous samples:   N = {n_samples:,}\n"
        "----------------------------------------------------------------------\n"
        "Metric               Before (Raw)       After (Adjusted)   Improvement\n"
        "----------------------------------------------------------------------\n"
        f"Mean Bias (°C)       {raw_bias:+10.3f}         {adj_bias:+10.3f}"
        f"         {bias_reduction_pct:+.1f}%\n"
        f"MAE (°C)             {raw_mae:10.3f}         {adj_mae:10.3f}"
        f"         {mae_reduction_pct:+.1f}%\n"
        f"RMSE (°C)            {raw_rmse:10.3f}         {adj_rmse:10.3f}"
        f"         {rmse_reduction_pct:+.1f}%\n"
        f"Pearson r            {raw_r:10.4f}         {adj_r:10.4f}"
        "         (Invariant)\n"
        "======================================================================\n"
    )

    return {
        "station_name": station_name,
        "station_elevation": station_elevation,
        "model_elevation": model_elevation,
        "delta_z": delta_z,
        "lapse_rate": lapse_rate,
        "delta_t": delta_t,
        "n_samples": n_samples,
        "raw_bias": raw_bias,
        "raw_mae": raw_mae,
        "raw_rmse": raw_rmse,
        "raw_r": raw_r,
        "adjusted_bias": adj_bias,
        "adjusted_mae": adj_mae,
        "adjusted_rmse": adj_rmse,
        "adjusted_r": adj_r,
        "bias_reduction_pct": bias_reduction_pct,
        "rmse_reduction_pct": rmse_reduction_pct,
        "table": df_comparison,
        "report_text": report_text,
    }

