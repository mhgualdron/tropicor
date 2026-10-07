"""Physics-based orographic and topographic vertical lapse-rate corrections.

This module implements Layer 0 leaf utilities for vertical temperature downscaling
based on environmental lapse-rate adjustments. It bridges the vertical elevation
discrepancies between coarse atmospheric reanalysis grids (e.g., ERA5 ~30 km) and
high-relief ground-truth observation stations in complex topography such as the
Colombian Andes.
"""

from typing import Union, overload

import numpy as np
import pandas as pd

STANDARD_LAPSE_RATE: float = 0.0065
"""Standard environmental lapse rate in the free troposphere (0.0065 °C/m)."""

DRY_ADIABATIC_LAPSE_RATE: float = 0.0098
"""Dry adiabatic lapse rate (g / c_p ≈ 0.0098 °C/m or 9.8 °C/km)."""

MOIST_ADIABATIC_LAPSE_RATE: float = 0.0050
"""Typical moist/saturated pseudoadiabatic lapse rate (≈ 0.0050 °C/m)."""


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
