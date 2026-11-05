"""Tests for benchmark ERA5 daily loader, station constants, and file hashing."""

import hashlib
import inspect
import re
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tropicor.core.benchmark_loaders import (
    BENCHMARK_STATION_CONSTANTS,
    ERA5DailyRecord,
    StationConstants,
    list_era5_files,
    read_era5_daily,
    sha256_file,
)
from tropicor.core.phase3_evaluator import compute_sha256


def _make_synthetic_era5_file(
    directory: Path,
    year: int,
    cell_lat: float = 4.75,
    cell_lon: float = -74.00,
    hourly_generator: Optional[Callable[[pd.DatetimeIndex], np.ndarray]] = None,
    units: str = "K",
    var_name: str = "t2m",
) -> Path:
    """Create a synthetic ERA5 NetCDF file for testing."""
    times = pd.date_range(f"{year}-01-01 00:00", f"{year}-12-31 23:00", freq="1h")
    lats = [cell_lat - 0.25, cell_lat, cell_lat + 0.25]
    lons = [cell_lon - 0.25, cell_lon, cell_lon + 0.25]

    n_times = len(times)
    n_lats = len(lats)
    n_lons = len(lons)

    if hourly_generator is not None:
        vals_1d = hourly_generator(times)
    else:
        hour_of_day = times.hour.values
        celsius = 15.0 + 8.0 * np.sin(2 * np.pi * (hour_of_day - 9) / 24.0)
        vals_1d = celsius + 273.15

    data = np.zeros((n_times, n_lats, n_lons), dtype=np.float32)
    for i in range(n_lats):
        for j in range(n_lons):
            data[:, i, j] = vals_1d

    ds = xr.Dataset(
        data_vars={
            var_name: (("time", "latitude", "longitude"), data, {"units": units}),
        },
        coords={
            "time": times,
            "latitude": lats,
            "longitude": lons,
        },
    )
    nc_path = directory / f"era5_2m_temperature_{year}.nc"
    ds.to_netcdf(nc_path)
    return nc_path


def test_j2_constants_equal_protocol_table():
    """Verify Section J.2 constants match docs/science/evaluation-protocol.md."""
    protocol_path = Path("docs/science/evaluation-protocol.md")
    assert protocol_path.is_file()
    content = protocol_path.read_text(encoding="utf-8").replace("\u2212", "-")

    assert set(BENCHMARK_STATION_CONSTANTS.keys()) == {"21205710", "23215030"}

    for st_id, consts in BENCHMARK_STATION_CONSTANTS.items():
        assert isinstance(consts, StationConstants)
        pattern = (
            rf"\|\s*`{st_id}`\s*\|\s*([\d\.-]+)\s*\|\s*([\d\.-]+)\s*\|\s*"
            rf"([\d\.-]+)\s*\|\s*([\d\.-]+)°\s*N,\s*([\d\.-]+)°\s*W\s*\|\s*"
            rf"([\d\.-]+)\s*\|\s*([\d\.-]+)\s*\|"
        )
        m = re.search(pattern, content)
        assert m is not None, f"Could not parse J.2 table row for station {st_id}."
        proto_lat = float(m.group(1))
        proto_lon = float(m.group(2))
        proto_elev = float(m.group(3))
        proto_cell_lat = float(m.group(4))
        proto_cell_lon = -float(m.group(5))  # West is negative
        proto_z_era5 = float(m.group(6))
        proto_delta_z = float(m.group(7))

        assert np.isclose(consts.latitude, proto_lat, atol=1e-9)
        assert np.isclose(consts.longitude, proto_lon, atol=1e-9)
        assert np.isclose(consts.elevation_m, proto_elev, atol=1e-9)
        assert np.isclose(consts.cell_lat, proto_cell_lat, atol=1e-9)
        assert np.isclose(consts.cell_lon, proto_cell_lon, atol=1e-9)
        assert np.isclose(consts.z_era5_m, proto_z_era5, atol=1e-9)
        assert np.isclose(consts.delta_z_m, proto_delta_z, atol=1e-9)

        # Convention: delta_z_m = elevation_m - z_era5_m
        assert np.isclose(
            consts.delta_z_m, consts.elevation_m - consts.z_era5_m, atol=1e-9
        )

    assert np.isclose(
        BENCHMARK_STATION_CONSTANTS["21205710"].delta_z_m, -201.4, atol=1e-9
    )
    assert np.isclose(
        BENCHMARK_STATION_CONSTANTS["23215030"].delta_z_m, -628.6, atol=1e-9
    )


def test_source_guard():
    """Verify benchmark_loaders avoids forbidden imports, classes, and print calls."""
    import tropicor.core.benchmark_loaders as mod

    source = inspect.getsource(mod)
    forbidden_terms = [
        "StationCatalog",
        "from_benchmark",
        "get_andean_station",
        "validate_validation_access",
        "record_validation_opening",
        "validation_openings.jsonl",
        "print(",
    ]
    for term in forbidden_terms:
        assert term not in source, f"Forbidden term '{term}' found in module source."


def test_aggregation_correctness_and_boundary_year(tmp_path: Path):
    """Verify daily aggregation matches independent calculation with boundary year."""
    y = 2001
    y_next = 2002

    def hourly_pattern(times: pd.DatetimeIndex) -> np.ndarray:
        # Deterministic function of hour and day of year
        h = times.hour.values
        doy = times.dayofyear.values
        celsius = 12.0 + 7.0 * np.sin(2 * np.pi * (h - 8) / 24.0) + (doy % 5)
        return celsius + 273.15

    f1 = _make_synthetic_era5_file(tmp_path, y, hourly_generator=hourly_pattern)
    f2 = _make_synthetic_era5_file(tmp_path, y_next, hourly_generator=hourly_pattern)

    times_all = pd.date_range(f"{y}-01-01 00:00", f"{y_next}-12-31 23:00", freq="1h")
    s_hourly_c = pd.Series(hourly_pattern(times_all) - 273.15, index=times_all)

    # Independent daily calculation for window 05:00 to 05:00 UTC
    expected_days = pd.date_range(f"{y}-01-01", f"{y}-12-31", freq="D")
    expected_tmax_list: list[float] = []
    expected_tmin_list: list[float] = []
    for day in expected_days:
        w_start = day + pd.Timedelta(hours=5)
        w_end = day + pd.Timedelta(days=1, hours=4)
        sub = s_hourly_c.loc[w_start:w_end]
        assert len(sub) == 24
        expected_tmax_list.append(float(sub.max()))
        expected_tmin_list.append(float(sub.min()))

    rec = read_era5_daily(
        tmp_path,
        "21205710",
        y,
        y,
        y,
        window_start_utc=5,
        lag_days=0,
        min_hours=24,
    )

    assert isinstance(rec, ERA5DailyRecord)
    assert rec.station_id == "21205710"
    assert rec.files_used == (f1.name, f2.name)

    np.testing.assert_allclose(rec.tmax.values, expected_tmax_list, atol=1e-9)
    np.testing.assert_allclose(rec.tmin.values, expected_tmin_list, atol=1e-9)

    # Check December 31 boundary day specifically
    dec31 = f"{y}-12-31"
    assert rec.tmax.loc[dec31] == pytest.approx(expected_tmax_list[-1], abs=1e-9)
    assert rec.tmin.loc[dec31] == pytest.approx(expected_tmin_list[-1], abs=1e-9)

    # Dataclass repr excludes series
    rep = repr(rec)
    assert "tmax" not in rep
    assert "tmin" not in rep
    assert "21205710" in rep


def test_cell_verification_nearest_mismatch_raises(tmp_path: Path):
    """Verify that a grid lacking the registered J.2 cell raises ValueError."""
    # Create grid far from (4.75, -74.00)
    _make_synthetic_era5_file(tmp_path, 2001, cell_lat=0.0, cell_lon=0.0)
    _make_synthetic_era5_file(tmp_path, 2002, cell_lat=0.0, cell_lon=0.0)

    with pytest.raises(ValueError, match="does not match registered cell"):
        read_era5_daily(
            tmp_path,
            "21205710",
            2001,
            2001,
            2001,
            window_start_utc=5,
            lag_days=0,
            min_hours=24,
        )


def test_units_and_missing_year_raises(tmp_path: Path):
    """Verify units 'degC' and missing year files raise ValueError."""
    # Units != 'K'
    dir_units = tmp_path / "units"
    dir_units.mkdir()
    _make_synthetic_era5_file(dir_units, 2001, units="degC")
    _make_synthetic_era5_file(dir_units, 2002, units="degC")

    with pytest.raises(ValueError, match="must have units 'K'"):
        read_era5_daily(
            dir_units,
            "21205710",
            2001,
            2001,
            2001,
            window_start_utc=5,
            lag_days=0,
            min_hours=24,
        )

    # Missing Y+1 file
    dir_missing = tmp_path / "missing"
    dir_missing.mkdir()
    _make_synthetic_era5_file(dir_missing, 2001)  # 2002 omitted

    with pytest.raises(ValueError, match="2002"):
        read_era5_daily(
            dir_missing,
            "21205710",
            2001,
            2001,
            2001,
            window_start_utc=5,
            lag_days=0,
            min_hours=24,
        )


def test_nan_prohibition_and_plausibility_raise_counts_only(tmp_path: Path):
    """Verify NaN prohibition and plausibility raise counts without data values."""
    dir_nan = tmp_path / "nan_test"
    dir_nan.mkdir()

    def nan_pattern(times: pd.DatetimeIndex) -> np.ndarray:
        celsius = np.full(len(times), 15.0, dtype=np.float32)
        # Set 5 hours of 2001-06-15 to NaN -> 19 hours < 24 min_hours
        mask = (
            (times.month == 6)
            & (times.day == 15)
            & (times.hour >= 8)
            & (times.hour <= 12)
        )
        celsius[mask] = np.nan
        return celsius + 273.15

    _make_synthetic_era5_file(dir_nan, 2001, hourly_generator=nan_pattern)
    _make_synthetic_era5_file(dir_nan, 2002)

    with pytest.raises(ValueError) as exc_nan:
        read_era5_daily(
            dir_nan,
            "21205710",
            2001,
            2001,
            2001,
            window_start_utc=5,
            lag_days=0,
            min_hours=24,
        )
    msg_nan = str(exc_nan.value)
    assert "21205710" in msg_nan
    assert "missing value" in msg_nan
    assert "found 2 missing value(s)" in msg_nan  # 1 day missing tmax & tmin
    # Assert no floating point temperature values leaked
    assert not re.findall(r"\b\d+\.\d+\b", msg_nan)

    # Plausibility test: value above 60°C (e.g. 70°C = 343.15 K)
    dir_plaus = tmp_path / "plaus_test"
    dir_plaus.mkdir()

    def extreme_pattern(times: pd.DatetimeIndex) -> np.ndarray:
        celsius = np.full(len(times), 15.0, dtype=np.float32)
        # Single afternoon hour above 60°C -> tmax is 70°C, tmin remains 15°C
        mask = (times.month == 7) & (times.day == 20) & (times.hour == 14)
        celsius[mask] = 70.0
        return celsius + 273.15

    _make_synthetic_era5_file(dir_plaus, 2001, hourly_generator=extreme_pattern)
    _make_synthetic_era5_file(dir_plaus, 2002)

    with pytest.raises(ValueError) as exc_plaus:
        read_era5_daily(
            dir_plaus,
            "21205710",
            2001,
            2001,
            2001,
            window_start_utc=5,
            lag_days=0,
            min_hours=24,
        )
    msg_plaus = str(exc_plaus.value)
    assert "21205710" in msg_plaus
    assert "E.6.1 plausibility limits" in msg_plaus
    assert "found 1 value(s)" in msg_plaus  # tmax is 70, tmin is 15
    # Assert actual temperature values are not leaked in error message
    assert "70" not in msg_plaus
    assert "343" not in msg_plaus


def test_opening_guard_and_no_file_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Verify validation authorization guard blocks access before touching files."""
    non_existent = tmp_path / "does_not_exist"

    # Monkeypatch file openers to assert no file access occurs before guard
    def _forbidden_access(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            "File access was attempted when authorization should block!"
        )

    monkeypatch.setattr("tropicor.io.era5.ERA5Adapter.__init__", _forbidden_access)
    monkeypatch.setattr("xarray.open_dataset", _forbidden_access)

    # Validation period requested (2005 > calibration 2004) without authorization
    with pytest.raises(PermissionError):
        read_era5_daily(
            non_existent,
            "21205710",
            2001,
            2005,
            2004,
            opening_records=None,
            opening_purpose="benchmark_formal",
            window_start_utc=5,
            lag_days=0,
            min_hours=24,
        )

    # Wrong station record
    wrong_st_record = [{"station_id": "99999999", "purpose": "benchmark_formal"}]
    with pytest.raises(PermissionError):
        read_era5_daily(
            non_existent,
            "21205710",
            2001,
            2005,
            2004,
            opening_records=wrong_st_record,
            opening_purpose="benchmark_formal",
            window_start_utc=5,
            lag_days=0,
            min_hours=24,
        )

    # Wrong purpose record
    wrong_purpose_record = [{"station_id": "21205710", "purpose": "other_purpose"}]
    with pytest.raises(PermissionError):
        read_era5_daily(
            non_existent,
            "21205710",
            2001,
            2005,
            2004,
            opening_records=wrong_purpose_record,
            opening_purpose="benchmark_formal",
            window_start_utc=5,
            lag_days=0,
            min_hours=24,
        )

    # Unpatch to test valid access
    monkeypatch.undo()

    # Create synthetic files for years 2004, 2005, 2006
    valid_dir = tmp_path / "valid_files"
    valid_dir.mkdir()
    _make_synthetic_era5_file(valid_dir, 2004)
    _make_synthetic_era5_file(valid_dir, 2005)
    _make_synthetic_era5_file(valid_dir, 2006)

    # Correct record passes
    correct_record = [{"station_id": "21205710", "purpose": "benchmark_formal"}]
    rec_val = read_era5_daily(
        valid_dir,
        "21205710",
        2004,
        2005,
        2004,
        opening_records=correct_record,
        opening_purpose="benchmark_formal",
        window_start_utc=5,
        lag_days=0,
        min_hours=24,
    )
    assert len(rec_val.tmax) == 731  # 2004 (leap, 366) + 2005 (365)
    assert "era5_2m_temperature_2006.nc" in rec_val.files_used

    # Calibration only (last_year == calibration_last_year) without records
    rec_cal = read_era5_daily(
        valid_dir,
        "21205710",
        2004,
        2004,
        2004,
        opening_records=None,
        opening_purpose=None,
        window_start_utc=5,
        lag_days=0,
        min_hours=24,
    )
    assert len(rec_cal.tmax) == 366
    # Confirms K.4.4: year last_year+1 (2005) is still read for boundary UTC window
    assert "era5_2m_temperature_2005.nc" in rec_cal.files_used


def test_required_keyword_only_arguments(tmp_path: Path):
    """Verify window, lag, and min_hours are required keyword-only args."""
    # Missing all three
    with pytest.raises(TypeError):
        read_era5_daily(tmp_path, "21205710", 2001, 2001, 2001)  # type: ignore[call-arg]

    # Missing two
    with pytest.raises(TypeError):
        read_era5_daily(  # type: ignore[call-arg]
            tmp_path, "21205710", 2001, 2001, 2001, window_start_utc=5
        )

    # Positional invocation instead of keyword-only
    with pytest.raises(TypeError):
        read_era5_daily(  # type: ignore[misc]
            tmp_path, "21205710", 2001, 2001, 2001, None, None, 5, 0, 24
        )


def test_sha256_file_helper(tmp_path: Path):
    """Verify sha256_file reads bytes in chunks and does not use NetCDF or xarray."""
    test_file = tmp_path / "sample_data.txt"
    content = b"Benchmark file hashing verification line 1\nLine 2\n"
    test_file.write_bytes(content)

    expected_hash = hashlib.sha256(content).hexdigest()
    assert sha256_file(test_file) == expected_hash
    assert sha256_file(test_file) == compute_sha256(test_file)

    # Verify source does not reference NetCDF or xarray
    src = inspect.getsource(sha256_file)
    assert "xarray" not in src
    assert "open_dataset" not in src
    assert "netCDF" not in src

    # Missing file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        sha256_file(tmp_path / "non_existent.txt")


def test_list_era5_files(tmp_path: Path):
    """Verify list_era5_files sorts, matches whole years, and reports missing years."""
    # Create files
    f2001 = tmp_path / "era5_2m_temperature_2001.nc"
    f2002 = tmp_path / "era5_2m_temperature_2002.nc"
    f2003 = tmp_path / "era5_2m_temperature_2003.nc"
    f20011 = tmp_path / "era5_2m_temperature_20011.nc"
    f_other = tmp_path / "era5_total_precipitation_2001.nc"

    for f in (f2001, f2002, f2003, f20011, f_other):
        f.touch()

    # Sorted matched files
    res = list_era5_files(tmp_path, [2002, 2001])
    assert res == [f2001, f2002]

    # Whole year regex: 20011 should not match 2001
    dir_single = tmp_path / "single"
    dir_single.mkdir()
    (dir_single / "era5_2m_temperature_20011.nc").touch()
    with pytest.raises(ValueError, match="2001"):
        list_era5_files(dir_single, [2001])

    # Missing year
    with pytest.raises(ValueError, match="2005"):
        list_era5_files(tmp_path, [2001, 2005])

    # Missing directory
    with pytest.raises(FileNotFoundError):
        list_era5_files(tmp_path / "non_existent_dir", [2001])


def test_duplicate_timestamps_raise(tmp_path: Path):
    """Verify overlapping ERA5 files with duplicated timestamps raise ValueError."""
    y = 2001
    y_next = 2002

    # Normal files for year Y and Y+1
    _make_synthetic_era5_file(tmp_path, y)
    _make_synthetic_era5_file(tmp_path, y_next)

    # Second file for year Y matching glob and whole-number regex with different values
    times = pd.date_range(f"{y}-01-01 00:00", f"{y}-12-31 23:00", freq="1h")
    lats = [4.5, 4.75, 5.0]
    lons = [-74.25, -74.0, -73.75]
    data = np.full((len(times), len(lats), len(lons)), 298.15, dtype=np.float32)
    ds = xr.Dataset(
        data_vars={"t2m": (("time", "latitude", "longitude"), data, {"units": "K"})},
        coords={"time": times, "latitude": lats, "longitude": lons},
    )
    copy_path = tmp_path / f"era5_2m_temperature_{y}_copy.nc"
    ds.to_netcdf(copy_path)

    # No match string: with Dask, xarray raises ValueError on combine_by_coords overlap;
    # without Dask, our explicit safeguard raises ValueError on duplicate timestamps.
    with pytest.raises(ValueError):
        read_era5_daily(
            tmp_path,
            "21205710",
            y,
            y,
            y,
            window_start_utc=5,
            lag_days=0,
            min_hours=24,
        )
