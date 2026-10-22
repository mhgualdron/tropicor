"""Unit tests for Andean temperature loader with synthetic data (Protocol v1.4, App F).

All tests use purely synthetic data (no real station observations or online ERA5 files).
Verifies:
1. Validation range out-of-physical-range values do not fail calibration loading.
2. Corrupted ERA5 files in validation years are never opened.
3. Unregistered stations raise ValueError.
4. Requesting validation raises ValueError.
5. Returned observations never contain dates > cal_end.
6. Cropping respects invalid years inside the block (excluded by >= 330 days rule).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import cast

import netCDF4
import numpy as np
import pandas as pd
import pytest

from tropicor.core.andean_registry import request_andean_validation_partition
from tropicor.downscale.temperature import load_calibration_temperature_pair

# Andean test station: Berlín (37015020)
# Cal block: [1991, 2009], Val block: [2010, 2020]
# Valid years in cal: 1991-1997, 2005, 2007-2009 (1998-2004, 2006 are invalid/missing)
TEST_STATION_ID = "37015020"
CAL_START = 1991
CAL_END = 2009


def _create_synthetic_era5_nc(
    path: Path,
    start_date: str,
    end_date: str,
    t2m_k: float = 293.15,
    lat: float = 7.25,
    lon: float = -72.75,
) -> None:
    """Helper to create a valid synthetic ERA5 NetCDF file without warnings."""
    hourly_dates = pd.date_range(start_date, end_date, freq="h")
    n_hours = len(hourly_dates)
    with netCDF4.Dataset(str(path), "w", format="NETCDF4") as nc:
        nc.createDimension("time", n_hours)
        nc.createDimension("latitude", 1)
        nc.createDimension("longitude", 1)
        time_var = nc.createVariable("time", "f8", ("time",))
        time_var.units = f"hours since {start_date} 00:00:00"
        lat_var = nc.createVariable("latitude", "f4", ("latitude",))
        lon_var = nc.createVariable("longitude", "f4", ("longitude",))
        t2m_var = nc.createVariable("t2m", "f4", ("time", "latitude", "longitude"))
        t2m_var.units = "K"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            time_var[:] = np.arange(n_hours)
            lat_var[:] = [lat]
            lon_var[:] = [lon]
            hours_of_day = hourly_dates.hour.values
            diurnal = 5.0 * np.sin(2 * np.pi * (hours_of_day - 9) / 24.0)
            vals = (t2m_k + diurnal).astype(np.float32).reshape(n_hours, 1, 1)
            t2m_var[:] = vals


def test_andean_loader_validation_year_out_of_range_does_not_fail(
    tmp_path: Path,
) -> None:
    """Out-of-range value (99.0 C) in a validation year (2015) causes no error."""
    # Observations cover 1991 to 2020
    obs_dates = pd.date_range("1991-01-01", "2020-12-31", freq="D")
    df_obs = pd.DataFrame(
        {
            "CodigoEstacion": [TEST_STATION_ID] * len(obs_dates),
            "Fecha": obs_dates.strftime("%Y-%m-%d"),
            "Valor": [15.0] * len(obs_dates),
        }
    )
    # Inject 99.0 C in validation year 2015
    df_obs.loc[df_obs["Fecha"] == "2015-06-15", "Valor"] = 99.0

    tmax_path = tmp_path / "tmax.csv"
    tmin_path = tmp_path / "tmin.csv"
    df_obs.to_csv(tmax_path, index=False)
    df_obs.to_csv(tmin_path, index=False)

    era5_file = tmp_path / "era5_cal.nc"
    _create_synthetic_era5_nc(
        era5_file, f"{CAL_START}-01-01", f"{CAL_END + 1}-01-02", 288.15
    )

    o_tmax, o_tmin, m_tmax, m_tmin = load_calibration_temperature_pair(
        station_id=TEST_STATION_ID,
        obs_tmax_path=tmax_path,
        obs_tmin_path=tmin_path,
        era5_target=era5_file,
        window_start_utc=5,
        lag_days=0,
        min_hours=24,
    )
    # Validation data is strictly excluded; loader completes successfully
    assert o_tmax.index[-1] == pd.Timestamp(f"{CAL_END}-12-31")


def test_andean_loader_corrupted_validation_era5_file_not_opened(
    tmp_path: Path,
) -> None:
    """A corrupted ERA5 file from validation year is never opened by cal loader."""
    era5_dir = tmp_path / "era5_dir"
    era5_dir.mkdir()

    # Valid calibration files
    _create_synthetic_era5_nc(
        era5_dir / f"era5_2m_temperature_{CAL_START}_{CAL_END}.nc",
        f"{CAL_START}-01-01",
        f"{CAL_END + 1}-01-02",
        288.15,
    )

    # Corrupted validation year file
    corrupt_file = era5_dir / "era5_2m_temperature_2015_corrupt.nc"
    corrupt_file.write_bytes(b"CORRUPTED_BYTES_NOT_NETCDF")

    obs_dates = pd.date_range(f"{CAL_START}-01-01", f"{CAL_END}-12-31", freq="D")
    df_obs = pd.DataFrame(
        {
            "CodigoEstacion": [TEST_STATION_ID] * len(obs_dates),
            "Fecha": obs_dates.strftime("%Y-%m-%d"),
            "Valor": [15.0] * len(obs_dates),
        }
    )
    tmax_path = tmp_path / "tmax.csv"
    tmin_path = tmp_path / "tmin.csv"
    df_obs.to_csv(tmax_path, index=False)
    df_obs.to_csv(tmin_path, index=False)

    # Must succeed without error because 2015 is excluded by calibration year regex
    o_tmax, o_tmin, m_tmax, m_tmin = load_calibration_temperature_pair(
        station_id=TEST_STATION_ID,
        obs_tmax_path=tmax_path,
        obs_tmin_path=tmin_path,
        era5_target=era5_dir,
        window_start_utc=5,
        lag_days=0,
        min_hours=24,
    )
    assert len(o_tmax) == len(obs_dates)


def test_andean_loader_unregistered_station_raises_value_error(tmp_path: Path) -> None:
    """Requesting an unregistered station raises ValueError."""
    dummy_csv = tmp_path / "obs.csv"
    dummy_csv.write_text(
        "CodigoEstacion,Fecha,Valor\n99999999,1995-01-01,15.0\n", encoding="utf-8"
    )
    dummy_nc = tmp_path / "dummy.nc"

    with pytest.raises(
        ValueError, match="has no pre-specified temperature calibration partition"
    ):
        load_calibration_temperature_pair(
            station_id="99999999",
            obs_tmax_path=dummy_csv,
            obs_tmin_path=dummy_csv,
            era5_target=dummy_nc,
            window_start_utc=5,
            lag_days=0,
            min_hours=24,
        )


def test_andean_loader_request_validation_raises_value_error() -> None:
    """Requesting validation partition for an Andean station raises ValueError."""
    with pytest.raises(ValueError, match="strictly closed.*Protocol v1.4 Appendix F.7"):
        request_andean_validation_partition(TEST_STATION_ID)


def test_andean_loader_returned_observations_never_contain_dates_gt_cal_end(
    tmp_path: Path,
) -> None:
    """Observations returned by loader strictly terminate on or before cal_end."""
    obs_dates = pd.date_range("1991-01-01", "2025-12-31", freq="D")
    df_obs = pd.DataFrame(
        {
            "CodigoEstacion": [TEST_STATION_ID] * len(obs_dates),
            "Fecha": obs_dates.strftime("%Y-%m-%d"),
            "Valor": [15.0] * len(obs_dates),
        }
    )
    tmax_path = tmp_path / "tmax.csv"
    tmin_path = tmp_path / "tmin.csv"
    df_obs.to_csv(tmax_path, index=False)
    df_obs.to_csv(tmin_path, index=False)

    era5_file = tmp_path / "era5_cal.nc"
    _create_synthetic_era5_nc(
        era5_file, f"{CAL_START}-01-01", f"{CAL_END + 1}-01-02", 288.15
    )

    o_tmax, o_tmin, _, _ = load_calibration_temperature_pair(
        station_id=TEST_STATION_ID,
        obs_tmax_path=tmax_path,
        obs_tmin_path=tmin_path,
        era5_target=era5_file,
        window_start_utc=5,
        lag_days=0,
        min_hours=24,
    )
    idx_max = cast(pd.DatetimeIndex, o_tmax.index)
    idx_min = cast(pd.DatetimeIndex, o_tmin.index)
    assert (idx_max.year > CAL_END).sum() == 0
    assert (idx_min.year > CAL_END).sum() == 0
    assert o_tmax.index.max() == pd.Timestamp(f"{CAL_END}-12-31")


def test_andean_loader_cropping_respects_invalid_years_inside_block(
    tmp_path: Path,
) -> None:
    """Cropping masks out invalid years within the calibration block (< 330 days rule).

    For Berlin (37015020), 1998-2004 and 2006 were invalid years.
    Even if raw CSV contains entries in 1998, they must be masked to NaN.
    """
    obs_dates = pd.date_range(f"{CAL_START}-01-01", f"{CAL_END}-12-31", freq="D")
    df_obs = pd.DataFrame(
        {
            "CodigoEstacion": [TEST_STATION_ID] * len(obs_dates),
            "Fecha": obs_dates.strftime("%Y-%m-%d"),
            "Valor": [15.0] * len(obs_dates),
        }
    )
    tmax_path = tmp_path / "tmax.csv"
    tmin_path = tmp_path / "tmin.csv"
    df_obs.to_csv(tmax_path, index=False)
    df_obs.to_csv(tmin_path, index=False)

    era5_file = tmp_path / "era5_cal.nc"
    _create_synthetic_era5_nc(
        era5_file, f"{CAL_START}-01-01", f"{CAL_END + 1}-01-02", 288.15
    )

    o_tmax, o_tmin, _, _ = load_calibration_temperature_pair(
        station_id=TEST_STATION_ID,
        obs_tmax_path=tmax_path,
        obs_tmin_path=tmin_path,
        era5_target=era5_file,
        window_start_utc=5,
        lag_days=0,
        min_hours=24,
    )

    dt_idx = cast(pd.DatetimeIndex, o_tmax.index)

    # Valid year 1995: must have non-null observations
    mask_1995 = dt_idx.year == 1995
    assert o_tmax[mask_1995].notna().all()
    assert o_tmin[mask_1995].notna().all()

    # Invalid year 1998: must be masked to NaN
    mask_1998 = dt_idx.year == 1998
    assert o_tmax[mask_1998].isna().all()
    assert o_tmin[mask_1998].isna().all()

    # Invalid year 2000: must be masked to NaN
    mask_2000 = dt_idx.year == 2000
    assert o_tmax[mask_2000].isna().all()
    assert o_tmin[mask_2000].isna().all()


def test_andean_loader_mismatched_era5_grid_cell_raises_value_error(
    tmp_path: Path,
) -> None:
    """ERA5 file with grid coordinates not matching registry cell raises ValueError."""
    era5_file = tmp_path / "era5_mismatched_cell.nc"
    # Create file at lat=10.0, lon=-70.0 (does not match Berlin's cell 7.25, -72.75)
    _create_synthetic_era5_nc(
        era5_file,
        f"{CAL_START}-01-01",
        f"{CAL_END + 1}-01-02",
        288.15,
        lat=10.0,
        lon=-70.0,
    )

    dummy_csv = tmp_path / "obs.csv"
    dummy_csv.write_text(
        f"CodigoEstacion,Fecha,Valor\n{TEST_STATION_ID},{CAL_START}-01-01,15.0\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"ERA5 grid cell \(10\.00, -70\.00\) does not match registered cell",
    ):
        load_calibration_temperature_pair(
            station_id=TEST_STATION_ID,
            obs_tmax_path=dummy_csv,
            obs_tmin_path=dummy_csv,
            era5_target=era5_file,
            window_start_utc=5,
            lag_days=0,
            min_hours=24,
        )
