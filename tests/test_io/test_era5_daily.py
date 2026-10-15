"""Unit tests for ERA5 hourly to daily precipitation aggregation and comparison."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from scripts.compare_era5_ideam_daily import (
    evaluate_synchronous_metrics,
    haversine_km,
)
from tropicor.io.era5 import (
    ERA5Adapter,
    aggregate_hourly_to_daily_precipitation,
)


def test_aggregate_hourly_to_daily_window():
    """Verify that 12:00 UTC D to 12:00 UTC D+1 is accumulated to day D."""
    # 72 hours covering 2025-07-01 00:00 to 2025-07-03 23:00
    times = pd.date_range("2025-07-01 00:00", "2025-07-03 23:00", freq="1h")
    # Set 1.0 mm for every hour
    s = pd.Series(1.0, index=times)

    daily = aggregate_hourly_to_daily_precipitation(
        s, window_start_utc=12, min_hours=24
    )

    # 2025-06-30 covers [06-30 12:00 to 07-01 12:00) -> only 13 hours present -> NaN
    assert pd.isna(daily.loc["2025-06-30"])
    # 2025-07-01 covers [07-01 12:00 to 07-02 12:00) -> full 24 hours -> 24.0 mm
    assert daily.loc["2025-07-01"] == pytest.approx(24.0)
    # 2025-07-02 covers [07-02 12:00 to 07-03 12:00) -> full 24 hours -> 24.0 mm
    assert daily.loc["2025-07-02"] == pytest.approx(24.0)
    # 2025-07-03 covers [07-03 12:00 to 07-04 12:00) -> only 11 hours present -> NaN
    assert pd.isna(daily.loc["2025-07-03"])


def test_aggregate_hourly_to_daily_min_hours_masking():
    """Verify that incomplete days are masked to NaN when min_hours is enforced."""
    times = pd.date_range("2025-07-01 12:00", "2025-07-02 12:00", freq="1h")
    s = pd.Series(2.0, index=times)
    # Inject 2 NaNs so valid hours = 22 < 24
    s.iloc[5] = np.nan
    s.iloc[6] = np.nan

    # With min_hours=24, should be NaN
    daily_strict = aggregate_hourly_to_daily_precipitation(
        s, window_start_utc=12, min_hours=24
    )
    assert pd.isna(daily_strict.loc["2025-07-01"])

    # With min_hours=20, should be 22 * 2.0 = 44.0 mm
    daily_relaxed = aggregate_hourly_to_daily_precipitation(
        s, window_start_utc=12, min_hours=20
    )
    assert daily_relaxed.loc["2025-07-01"] == pytest.approx(44.0)


def test_aggregate_hourly_to_daily_lags():
    """Verify that lag offsets shift values correctly."""
    times = pd.date_range("2025-07-01 00:00", "2025-07-04 23:00", freq="1h")
    # Rainfall only on 2025-07-02 between 14:00 and 15:00 UTC (10 mm)
    s = pd.Series(0.0, index=times)
    s.loc["2025-07-02 14:00"] = 10.0

    d_lag0 = aggregate_hourly_to_daily_precipitation(s, lag_days=0, min_hours=24)
    d_lag_plus1 = aggregate_hourly_to_daily_precipitation(s, lag_days=1, min_hours=24)
    d_lag_minus1 = aggregate_hourly_to_daily_precipitation(s, lag_days=-1, min_hours=24)

    # In lag 0, event [07-02 12:00 to 07-03 12:00) falls on 2025-07-02
    assert d_lag0.loc["2025-07-02"] == pytest.approx(10.0)
    # In lag +1, shifted forward by 1 day -> appears on 2025-07-03
    assert d_lag_plus1.loc["2025-07-03"] == pytest.approx(10.0)
    # In lag -1, shifted backward by 1 day -> appears on 2025-07-01
    assert d_lag_minus1.loc["2025-07-01"] == pytest.approx(10.0)


def test_era5_adapter_get_daily_precipitation(tmp_path):
    """Verify ERA5Adapter correctly converts units and extracts daily series."""
    times = pd.date_range("2025-07-01 12:00", "2025-07-03 12:00", freq="1h")
    lats = [4.5, 4.75]
    lons = [-74.25, -74.0]

    # Shape: (time, lat, lon)
    # Value in meters: 0.001 m/hr = 1 mm/hr
    data = np.full((len(times), len(lats), len(lons)), 0.001, dtype=np.float32)

    ds = xr.Dataset(
        data_vars={
            "tp": (("valid_time", "latitude", "longitude"), data, {"units": "m"}),
        },
        coords={
            "valid_time": times,
            "latitude": lats,
            "longitude": lons,
        },
    )

    nc_path = tmp_path / "test_era5_hourly.nc"
    ds.to_netcdf(nc_path)

    with ERA5Adapter(nc_path) as adapter:
        s_daily = adapter.get_daily_precipitation(4.75, -74.0, min_hours=24)
        # Should be converted to mm: 24 hours * 1 mm/hr = 24.0 mm
        assert s_daily.loc["2025-07-01"] == pytest.approx(24.0)
        assert s_daily.loc["2025-07-02"] == pytest.approx(24.0)


def test_evaluate_synchronous_metrics():
    """Verify calculation of hydroclimatic metrics."""
    idx = pd.date_range("2025-07-01", periods=10, freq="D")
    obs = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], index=idx)
    mod = obs + 1.0  # Constant bias +1.0

    m = evaluate_synchronous_metrics(obs, mod)
    assert m["n"] == 10
    assert m["bias"] == pytest.approx(1.0)
    assert m["rmse"] == pytest.approx(1.0)
    assert m["mae"] == pytest.approx(1.0)
    assert m["r"] == pytest.approx(1.0)
    assert m["rho"] == pytest.approx(1.0)


def test_haversine_distance():
    """Verify haversine formula distance."""
    # Distance from Bogotá (4.6679, -74.1010) to same point is 0
    assert haversine_km(4.6679, -74.1010, 4.6679, -74.1010) == pytest.approx(0.0)
    # Roughly 1 degree of latitude is ~111 km
    d = haversine_km(4.0, -74.0, 5.0, -74.0)
    assert 110.0 < d < 112.0


def test_synthetic_ideam_convention_spikes():
    """Verify hourly rain 13-18 UTC D and 01-06 UTC D+1 both map to Day D (lag 0)."""
    times = pd.date_range("2025-07-01 00:00", "2025-07-04 23:00", freq="1h")

    # Serie 1: Lluvia solo entre 13:00 y 18:00 UTC del día D (2025-07-02)
    s1 = pd.Series(0.0, index=times)
    s1.loc["2025-07-02 13:00":"2025-07-02 18:00"] = 2.0  # 6 hours * 2.0 = 12.0 mm

    daily1 = aggregate_hourly_to_daily_precipitation(
        s1, window_start_utc=12, min_hours=24, lag_days=0
    )
    assert daily1.loc["2025-07-02"] == pytest.approx(12.0)
    assert daily1.loc["2025-07-01"] == pytest.approx(0.0)
    assert daily1.loc["2025-07-03"] == pytest.approx(0.0)

    # Serie 2: Lluvia solo entre 01:00 y 06:00 UTC del día D+1 (2025-07-03)
    s2 = pd.Series(0.0, index=times)
    s2.loc["2025-07-03 01:00":"2025-07-03 06:00"] = 3.0  # 6 hours * 3.0 = 18.0 mm

    daily2 = aggregate_hourly_to_daily_precipitation(
        s2, window_start_utc=12, min_hours=24, lag_days=0
    )
    # Ambas quedan asignadas al día D (2025-07-02)
    assert daily2.loc["2025-07-02"] == pytest.approx(18.0)
    assert daily2.loc["2025-07-01"] == pytest.approx(0.0)
    assert daily2.loc["2025-07-03"] == pytest.approx(0.0)

