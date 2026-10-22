from __future__ import annotations

import warnings
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import pytest

from scripts.run_andean_temperature_calibration import calibrate_station, run_dry_run
from tropicor.core.andean_registry import AndeanStation

# Andean test station: Berlín (37015020)
# Cal block: [1991, 2009], Val block: [2010, 2020]
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


def test_run_dry_run_with_both_variables_selects_only_temperature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify run_dry_run selects only era5_2m_temperature_{year}.nc when both exist."""
    st_code = "990011"
    st_dir = tmp_path / "station_data"
    st_dir.mkdir(parents=True)

    tmax_file = st_dir / "TempMaxima_Synthetic.csv"
    tmin_file = st_dir / "TempMinima_Synthetic.csv"
    tmax_file.write_text("dummy", encoding="utf-8")
    tmin_file.write_text("dummy", encoding="utf-8")

    mock_station = AndeanStation(
        codigo=st_code,
        nombre="SYNTHETIC",
        latitud_cne=6.25,
        longitud_cne=-75.5,
        altitud_cne=1500.0,
        celda_lat=6.25,
        celda_lon=-75.5,
        cal_block=(2000, 2001),
        val_block=(2002, 2003),
        valid_years=(2000, 2001, 2002, 2003),
    )
    monkeypatch.setattr(
        "scripts.run_andean_temperature_calibration.get_andean_station",
        lambda _: mock_station,
    )

    def _write_nc(path: Path, var_name: str, units: str) -> None:
        with netCDF4.Dataset(str(path), "w") as nc:
            nc.createDimension("time", 1)
            lat_v = nc.createVariable("latitude", "f4")
            lon_v = nc.createVariable("longitude", "f4")
            v = nc.createVariable(var_name, "f4", ("time",))
            lat_v[:] = 6.25
            lon_v[:] = -75.5
            v.units = units
            v[:] = [285.0]

    for y in [2000, 2001, 2002]:
        _write_nc(st_dir / f"era5_2m_temperature_{y}.nc", "t2m", "K")
        _write_nc(st_dir / f"era5_total_precipitation_{y}.nc", "tp", "m")

    res = run_dry_run(
        station_id=st_code,
        obs_tmax_path=tmax_file,
        obs_tmin_path=tmin_file,
        era5_dir=st_dir,
    )
    assert res["era5_years_verified"] == [2000, 2001, 2002]
    assert res["dry_run"] is True

    (st_dir / "era5_2m_temperature_2002.nc").unlink()
    with pytest.raises(
        ValueError,
        match="Expected exactly 1 NetCDF file matching 'era5_2m_temperature_2002\\.nc'",
    ):
        run_dry_run(
            station_id=st_code,
            obs_tmax_path=tmax_file,
            obs_tmin_path=tmin_file,
            era5_dir=st_dir,
        )


def test_calibrate_andean_station_synthetic(tmp_path: Path) -> None:
    """Verify calibration runner on an Andean station using purely synthetic data."""
    era5_dir = tmp_path / "era5_dir"
    era5_dir.mkdir()

    # Create synthetic ERA5 files for Berlin calibration years 1991 to 2010
    # Note: Berlin valid years in cal are 1991-1997, 2005, 2007-2009
    for y in range(CAL_START, CAL_END + 2):
        _create_synthetic_era5_nc(
            era5_dir / f"era5_2m_temperature_{y}.nc",
            f"{y}-01-01",
            f"{y}-12-31 23:00:00" if y <= CAL_END else f"{y}-01-01 23:00:00",
            288.15,
        )

    obs_dates = pd.date_range(f"{CAL_START}-01-01", f"{CAL_END}-12-31", freq="D")
    df_tmax = pd.DataFrame(
        {
            "CodigoEstacion": [TEST_STATION_ID] * len(obs_dates),
            "Fecha": obs_dates.strftime("%Y-%m-%d"),
            "Valor": [20.0] * len(obs_dates),
        }
    )
    df_tmin = pd.DataFrame(
        {
            "CodigoEstacion": [TEST_STATION_ID] * len(obs_dates),
            "Fecha": obs_dates.strftime("%Y-%m-%d"),
            "Valor": [10.0] * len(obs_dates),
        }
    )

    tmax_path = tmp_path / "tmax.csv"
    tmin_path = tmp_path / "tmin.csv"
    df_tmax.to_csv(tmax_path, index=False)
    df_tmin.to_csv(tmin_path, index=False)

    reports_dir = tmp_path / "reports"
    report = calibrate_station(
        station_id=TEST_STATION_ID,
        obs_tmax_path=tmax_path,
        obs_tmin_path=tmin_path,
        era5_dir=era5_dir,
        output_dir=reports_dir,
    )

    out_file = reports_dir / f"andes_calibration_{TEST_STATION_ID}.json"
    assert out_file.exists()

    assert report["report_type"] == "andes_temperature_calibration"
    assert report["protocol_version"] == "1.4"
    assert len(report["git_commit"]) > 0
    assert report["station_id"] == TEST_STATION_ID
    assert report["calibration_block"] == [CAL_START, CAL_END]

    # Verify raw bias metrics present
    assert "tmax" in report["raw_bias"]
    assert "tmin" in report["raw_bias"]

    # Verify Levels 1, 2, 3 metrics
    assert "calibration_loss" in report["level1"]
    assert "calibration_loss" in report["level2"]
    assert "cv_loss" in report["level3"]
    assert "winner_config_id" in report["level3"]
    assert "n_extrapolated_days" in report["level3"]
    assert "max_abs_extrapolation_correction" in report["level3"]
    assert len(report["level3"]["cv_configurations"]) == 8


def test_resolve_default_paths_strictly_requires_single_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """resolve_default_paths requires 1 file per pattern; 0 or >1 -> ValueError."""
    from scripts.run_andean_temperature_calibration import resolve_default_paths

    st_code = "990099"
    st_dir = tmp_path / "local" / "era5_ts" / st_code
    st_dir.mkdir(parents=True)

    monkeypatch.chdir(tmp_path)

    # 0 files -> ValueError
    with pytest.raises(
        ValueError, match="Expected exactly 1 file matching 'TempMaxima_\\*\\.csv'"
    ):
        resolve_default_paths(st_code)

    # 1 Tmax file, 0 Tmin files -> ValueError
    (st_dir / "TempMaxima_Berlin.csv").write_text("dummy", encoding="utf-8")
    with pytest.raises(
        ValueError, match="Expected exactly 1 file matching 'TempMinima_\\*\\.csv'"
    ):
        resolve_default_paths(st_code)

    # 1 Tmax, 1 Tmin -> Success
    (st_dir / "TempMinima_Berlin.csv").write_text("dummy", encoding="utf-8")
    tx, tn, d = resolve_default_paths(st_code)
    assert tx.name == "TempMaxima_Berlin.csv"
    assert tn.name == "TempMinima_Berlin.csv"

    # > 1 Tmax file -> ValueError
    (st_dir / "TempMaxima_Berlin_extra.csv").write_text("dummy", encoding="utf-8")
    with pytest.raises(
        ValueError, match="Expected exactly 1 file matching 'TempMaxima_\\*\\.csv'"
    ):
        resolve_default_paths(st_code)
