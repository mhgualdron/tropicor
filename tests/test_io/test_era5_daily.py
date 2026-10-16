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
    aggregate_hourly_to_daily_temperature,
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


def test_aggregate_hourly_to_daily_temperature_kelvin_conversion():
    """Verify that Kelvin is properly converted to Celsius with Tmax, Tmin, Tmean."""
    # 72 hours from 2025-07-01 00:00 to 2025-07-03 23:00
    times = pd.date_range("2025-07-01 00:00", "2025-07-03 23:00", freq="1h")
    # Base temp in Kelvin: 273.15 + 20.0 = 293.15 K (20 °C)
    s_k = pd.Series(293.15, index=times)

    # For 2025-07-02 12:00 to 2025-07-03 12:00 (Day 2025-07-02 in 12-12 window):
    # Set maximum spike at 15:00 UTC: 303.15 K (30.0 °C)
    # Set minimum dip at 06:00 UTC (07-03): 288.15 K (15.0 °C)
    s_k.loc["2025-07-02 15:00"] = 303.15
    s_k.loc["2025-07-03 06:00"] = 288.15

    daily_df = aggregate_hourly_to_daily_temperature(
        s_k, window_start_utc=12, min_hours=24, convert_to_celsius=True
    )

    assert isinstance(daily_df, pd.DataFrame)
    assert list(daily_df.columns) == ["tmax", "tmin", "tmean"]

    # Day 2025-07-02 has full 24h
    d2 = daily_df.loc["2025-07-02"]
    assert d2["tmax"] == pytest.approx(30.0)
    assert d2["tmin"] == pytest.approx(15.0)
    # 22 hours at 20 °C + 1 at 30 °C + 1 at 15 °C = 485 / 24 ≈ 20.208 °C
    assert d2["tmean"] == pytest.approx((22 * 20.0 + 30.0 + 15.0) / 24.0)

    # Incomplete boundary days should be NaN
    assert pd.isna(daily_df.loc["2025-06-30", "tmax"])
    assert pd.isna(daily_df.loc["2025-07-03", "tmax"])


def test_aggregate_hourly_to_daily_temperature_masking():
    """Verify that days with fewer than min_hours valid observations mask to NaN."""
    times = pd.date_range("2025-07-01 12:00", "2025-07-02 12:00", freq="1h")
    s = pd.Series(295.15, index=times)  # 22 °C
    # Inject 3 NaNs -> 21 valid hours < 24
    s.iloc[3:6] = np.nan

    # With min_hours=24 -> all NaN
    df_strict = aggregate_hourly_to_daily_temperature(
        s, window_start_utc=12, min_hours=24
    )
    assert pd.isna(df_strict.loc["2025-07-01", "tmax"])
    assert pd.isna(df_strict.loc["2025-07-01", "tmin"])
    assert pd.isna(df_strict.loc["2025-07-01", "tmean"])

    # With min_hours=20 -> valid values computed
    df_relaxed = aggregate_hourly_to_daily_temperature(
        s, window_start_utc=12, min_hours=20
    )
    assert df_relaxed.loc["2025-07-01", "tmax"] == pytest.approx(22.0)
    assert df_relaxed.loc["2025-07-01", "tmin"] == pytest.approx(22.0)
    assert df_relaxed.loc["2025-07-01", "tmean"] == pytest.approx(22.0)


def test_aggregate_hourly_to_daily_temperature_windows_and_lags():
    """Verify configurable windows (00-24 UTC vs 12-12 UTC) and lag shifts."""
    times = pd.date_range("2025-07-01 00:00", "2025-07-04 23:00", freq="1h")
    s = pd.Series(290.15, index=times)  # 17 °C
    # Spike at 2025-07-02 02:00 UTC (00-24 window of day 2, but 12-12 window of day 1)
    s.loc["2025-07-02 02:00"] = 305.15  # 32 °C

    # Window 00-24 UTC (calendar day UTC)
    df_win0 = aggregate_hourly_to_daily_temperature(s, window_start_utc=0, min_hours=24)
    assert df_win0.loc["2025-07-02", "tmax"] == pytest.approx(32.0)
    assert df_win0.loc["2025-07-01", "tmax"] == pytest.approx(17.0)

    # Window 12-12 UTC (07-01 12:00 to 07-02 12:00 maps to day 07-01)
    df_win12 = aggregate_hourly_to_daily_temperature(
        s, window_start_utc=12, min_hours=24
    )
    assert df_win12.loc["2025-07-01", "tmax"] == pytest.approx(32.0)
    assert df_win12.loc["2025-07-02", "tmax"] == pytest.approx(17.0)

    # Window 05-05 UTC (local civil day 00:00 to 24:00 local;
    # 07-02 02:00 UTC is 07-01 21:00 local)
    df_win5 = aggregate_hourly_to_daily_temperature(s, window_start_utc=5, min_hours=24)
    assert df_win5.loc["2025-07-01", "tmax"] == pytest.approx(32.0)
    assert df_win5.loc["2025-07-02", "tmax"] == pytest.approx(17.0)

    # Lag test
    df_lag1 = aggregate_hourly_to_daily_temperature(
        s, window_start_utc=0, min_hours=24, lag_days=1
    )
    # Shifted forward: spike on 2025-07-02 now appears on 2025-07-03
    assert df_lag1.loc["2025-07-03", "tmax"] == pytest.approx(32.0)


def test_era5_adapter_get_daily_temperature(tmp_path):
    """Verify ERA5Adapter extracts daily temperature (Tmax, Tmin, Tmean) from NetCDF."""
    times = pd.date_range("2025-07-01 05:00", "2025-07-03 05:00", freq="1h")
    lats = [4.5, 4.75]
    lons = [-74.25, -74.0]

    # Shape: (time, lat, lon) with Kelvin values: 295.15 K = 22.0 °C
    data = np.full((len(times), len(lats), len(lons)), 295.15, dtype=np.float32)

    ds = xr.Dataset(
        data_vars={
            "t2m": (("valid_time", "latitude", "longitude"), data, {"units": "K"}),
            "lsm": (("latitude", "longitude"), [[0.8, 0.9], [0.85, 0.95]]),
        },
        coords={
            "valid_time": times,
            "latitude": lats,
            "longitude": lons,
        },
    )

    nc_path = tmp_path / "test_era5_temp.nc"
    ds.to_netcdf(nc_path)

    with ERA5Adapter(nc_path) as adapter:
        df_temp = adapter.get_daily_temperature(4.75, -74.0, min_hours=24)
        assert "tmax" in df_temp.columns
        assert "tmin" in df_temp.columns
        assert "tmean" in df_temp.columns
        assert df_temp.loc["2025-07-01", "tmax"] == pytest.approx(22.0)
        assert df_temp.loc["2025-07-01", "tmin"] == pytest.approx(22.0)
        assert df_temp.loc["2025-07-01", "tmean"] == pytest.approx(22.0)

        # Land sea mask
        lsm_val = adapter.get_land_sea_mask(4.75, -74.0)
        assert lsm_val == pytest.approx(0.95)


def test_synthetic_temperature_civil_day_convention():
    """Verify maximum at 19 UTC and minimum at 11 UTC on Day D
    are assigned to Day D (lag 0).
    """
    # 05:00 UTC D to 05:00 UTC D+1 defines local civil day D in Colombia (UTC-5)
    times = pd.date_range("2025-07-01 00:00", "2025-07-04 23:00", freq="1h")
    s = pd.Series(20.0, index=times)

    # Maximum at 19:00 UTC on day D (2025-07-02 19:00 UTC = 14:00 local time) -> 35.0 °C
    # Minimum at 11:00 UTC on day D (2025-07-02 11:00 UTC = 06:00 local time) -> 10.0 °C
    s.loc["2025-07-02 19:00"] = 35.0
    s.loc["2025-07-02 11:00"] = 10.0

    # Default convention: window_start_utc=5 (05-05 UTC, local civil day), lag_days=0
    df_daily = aggregate_hourly_to_daily_temperature(
        s, window_start_utc=5, min_hours=24, lag_days=0, convert_to_celsius=False
    )

    # Both extreme events must be assigned to Day D (2025-07-02)
    assert df_daily.loc["2025-07-02", "tmax"] == pytest.approx(35.0)
    assert df_daily.loc["2025-07-02", "tmin"] == pytest.approx(10.0)

    # Adjacent days should not capture these spikes
    assert df_daily.loc["2025-07-01", "tmax"] == pytest.approx(20.0)
    assert df_daily.loc["2025-07-01", "tmin"] == pytest.approx(20.0)
    assert df_daily.loc["2025-07-03", "tmax"] == pytest.approx(20.0)
    assert df_daily.loc["2025-07-03", "tmin"] == pytest.approx(20.0)


def test_variable_aggregation_conventions_registry():
    """Verify configured temporal conventions for precipitation and temperature."""
    from tropicor.io.era5 import get_variable_convention

    precip_conv = get_variable_convention("precipitation")
    assert precip_conv.window_start_utc == 12
    assert precip_conv.lag_days == 0
    assert "12:00–12:00 UTC" in precip_conv.description
    assert "date_convention" in dir(precip_conv)

    temp_conv = get_variable_convention("temperature")
    assert temp_conv.window_start_utc == 5
    assert temp_conv.lag_days == 0
    assert "05:00–05:00 UTC" in temp_conv.description
    expected_sub = "05–05 UTC maintained as local civil day default"
    assert expected_sub in temp_conv.date_convention

    # Aliases
    assert get_variable_convention("tp").window_start_utc == 12
    assert get_variable_convention("t2m").window_start_utc == 5
