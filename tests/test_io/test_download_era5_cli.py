"""Unit tests for ERA5 year-by-year download CLI (Fase 2)."""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from scripts.download_era5_years import (
    build_cds_request,
    build_static_cds_request,
    build_timeseries_cds_request,
    check_or_update_manifest,
    compute_nearest_grid_center,
    load_cds_credentials,
    parse_args,
    run_dry_run,
    run_timeseries,
    run_verify,
    split_and_save_annual_files,
    validate_cli_args,
    verify_timeseries_dataset,
)
from tropicor.io.era5 import ERA5Adapter


def test_build_cds_request_structure() -> None:
    """Verify CDS request parameters for a given year."""
    req = build_cds_request(
        1995, variable="total_precipitation", area=[5.0, -75.0, 4.0, -74.0]
    )
    assert req["year"] == ["1995"]
    assert req["variable"] == ["total_precipitation"]
    assert req["product_type"] == ["reanalysis"]
    assert len(req["month"]) == 12
    assert len(req["day"]) == 31
    assert len(req["time"]) == 24
    assert req["area"] == [5.0, -75.0, 4.0, -74.0]


def test_load_cds_credentials_missing(tmp_path: Path) -> None:
    """Missing credentials must raise RuntimeError with clean message."""
    empty_env = tmp_path / "empty.env"
    empty_env.write_text("# No credentials here\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Faltan credenciales de Copernicus CDS"):
        load_cds_credentials(empty_env)


def test_load_cds_credentials_colon_format(tmp_path: Path) -> None:
    """Supports CDS .cdsapirc colon format (url: ... / key: ...)."""
    env_file = tmp_path / "cds.env"
    env_file.write_text(
        "url: https://cds.example.com/api\nkey: 12345-secret-key-uuid\n",
        encoding="utf-8",
    )
    url, key = load_cds_credentials(env_file)
    assert url == "https://cds.example.com/api"
    assert key == "12345-secret-key-uuid"


def test_dry_run_execution(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Dry run prints summary plan without performing downloads."""
    out_dir = tmp_path / "era5_test"
    run_dry_run(
        start_year=2000,
        end_year=2002,
        area=[5.0, -75.0, 4.0, -74.0],
        out_dir=out_dir,
        workers=2,
        variable="total_precipitation",
        dataset="reanalysis-era5-single-levels",
        url="https://cds.example.com/api",
    )
    captured = capsys.readouterr().out
    assert "PLAN DE EJECUCIÓN (DRY RUN)" in captured
    assert "2000 a 2002 (3 años)" in captured
    assert "MODO DRY-RUN COMPLETADO" in captured


def test_verify_empty_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify mode lists all years as missing when directory is empty."""
    out_dir = tmp_path / "empty_dir"
    run_verify(out_dir=out_dir, start_year=2020, end_year=2021)
    captured = capsys.readouterr().out
    assert "VERIFICACIÓN DE ARCHIVOS ERA5" in captured
    assert "Años faltantes: 2" in captured


def test_build_static_cds_request_structure() -> None:
    """Verify static invariant fields CDS request."""
    req = build_static_cds_request(area=[5.25, -74.5, 4.25, -73.5])
    assert req["product_type"] == ["reanalysis"]
    assert req["variable"] == ["geopotential", "land_sea_mask"]
    assert req["year"] == ["2020"]
    assert req["month"] == ["01"]
    assert req["day"] == ["01"]
    assert req["time"] == ["00:00"]
    assert req["area"] == [5.25, -74.5, 4.25, -73.5]


def test_manifest_multi_variable_tracking(tmp_path: Path) -> None:
    """Verify that downloading multiple variables retains both in manifest.json."""
    import json

    out_dir = tmp_path / "st_manifest"
    area = [5.25, -74.5, 4.25, -73.5]

    # First download: precipitation
    check_or_update_manifest(
        out_dir=out_dir,
        requested_area=area,
        station="21205710",
        variable="total_precipitation",
    )
    with open(out_dir / "manifest.json", "r", encoding="utf-8") as f:
        m1 = json.load(f)
    assert m1["variable"] == "total_precipitation"
    assert m1["variables"] == ["total_precipitation"]

    # Second download: 2m_temperature
    check_or_update_manifest(
        out_dir=out_dir,
        requested_area=area,
        station="21205710",
        variable="2m_temperature",
        static_downloaded=True,
    )
    with open(out_dir / "manifest.json", "r", encoding="utf-8") as f:
        m2 = json.load(f)
    assert m2["variable"] == "2m_temperature"
    assert "total_precipitation" in m2["variables"]
    assert "2m_temperature" in m2["variables"]
    assert m2["static_file"] == "era5_static.nc"
    assert m2["static_variables"] == ["geopotential", "land_sea_mask"]


def test_compute_nearest_grid_center() -> None:
    """Verify Section 2.1.1 snapping rule to nearest 0.25 deg ERA5 center."""
    # Jardín Botánico: 4.669333, -74.102667 -> 4.75, -74.00
    lat_jb, lon_jb = compute_nearest_grid_center(4.669333, -74.102667)
    assert lat_jb == 4.75
    assert lon_jb == -74.00

    # Aguas Claras: 8.228889, -73.602778 -> 8.25, -73.50
    lat_ac, lon_ac = compute_nearest_grid_center(8.228889, -73.602778)
    assert lat_ac == 8.25
    assert lon_ac == -73.50


def test_build_timeseries_cds_request_structure() -> None:
    """Verify CDS point timeseries request dictionary format."""
    req = build_timeseries_cds_request(
        lat_center=4.75,
        lon_center=-74.0,
        start_year=1990,
        end_year=2011,
        variables=["2m_temperature", "total_precipitation"],
    )
    assert req["variable"] == ["2m_temperature", "total_precipitation"]
    assert req["location"] == {"latitude": 4.75, "longitude": -74.0}
    assert req["date"] == "1990-01-01/2012-01-01"
    assert req["data_format"] == "netcdf"


def _make_dummy_timeseries_dataset(
    lat: float = 4.75,
    lon: float = -74.0,
    start_year: int = 1990,
    end_year: int = 1990,
    t2m_units: str = "K",
    tp_units: str = "m",
    include_nan: bool = False,
    extra_dim: bool = False,
) -> xr.Dataset:
    """Helper to generate synthetic point timeseries dataset for tests."""
    times = pd.date_range(
        start=f"{start_year}-01-01 00:00:00",
        end=f"{end_year + 1}-01-01 23:00:00",
        freq="h",
    )
    n = len(times)
    t2m_vals = np.ones(n, dtype=np.float32) * 290.0
    tp_vals = np.ones(n, dtype=np.float32) * 0.001
    if include_nan:
        t2m_vals[5] = np.nan

    coords: Dict[str, Any] = {
        "valid_time": times,
        "latitude": np.float64(lat),
        "longitude": np.float64(lon),
    }
    data_vars: Dict[str, Any] = {
        "t2m": (
            ("valid_time",),
            t2m_vals,
            {"units": t2m_units, "long_name": "2 metre temperature"},
        ),
        "tp": (
            ("valid_time",),
            tp_vals,
            {"units": tp_units, "long_name": "Total precipitation"},
        ),
    }
    if extra_dim:
        coords["expver"] = ["0001", "0005"]
        data_vars["t2m"] = (
            ("valid_time", "expver"),
            np.tile(t2m_vals[:, None], (1, 2)),
            {"units": t2m_units},
        )

    return xr.Dataset(data_vars=data_vars, coords=coords)


def test_verify_timeseries_lat_lon_mismatch_raises() -> None:
    """Verify ValueError when returned coordinates differ from requested center."""
    ds = _make_dummy_timeseries_dataset(lat=4.76, lon=-74.0)
    with pytest.raises(ValueError, match="Latitude mismatch"):
        verify_timeseries_dataset(
            ds,
            lat_center=4.75,
            lon_center=-74.0,
            start_year=1990,
            end_year=1990,
            requested_variables=["2m_temperature"],
        )

    ds2 = _make_dummy_timeseries_dataset(lat=4.75, lon=-74.1)
    with pytest.raises(ValueError, match="Longitude mismatch"):
        verify_timeseries_dataset(
            ds2,
            lat_center=4.75,
            lon_center=-74.0,
            start_year=1990,
            end_year=1990,
            requested_variables=["2m_temperature"],
        )


def test_verify_timeseries_units_mismatch_raises() -> None:
    """Verify ValueError if variable units differ from expected ('K' or 'm')."""
    ds = _make_dummy_timeseries_dataset(t2m_units="degC")

    with pytest.raises(ValueError, match="Unit mismatch"):
        verify_timeseries_dataset(
            ds,
            lat_center=4.75,
            lon_center=-74.0,
            start_year=1990,
            end_year=1990,
            requested_variables=["2m_temperature"],
        )

    ds2 = _make_dummy_timeseries_dataset(tp_units="mm")
    with pytest.raises(ValueError, match="Unit mismatch"):
        verify_timeseries_dataset(
            ds2,
            lat_center=4.75,
            lon_center=-74.0,
            start_year=1990,
            end_year=1990,
            requested_variables=["total_precipitation"],
        )


def test_verify_timeseries_nan_raises() -> None:
    """Verify ValueError is raised if NaNs exist in the returned timeseries."""
    ds = _make_dummy_timeseries_dataset(include_nan=True)
    with pytest.raises(ValueError, match="Found.*NaN values"):
        verify_timeseries_dataset(
            ds,
            lat_center=4.75,
            lon_center=-74.0,
            start_year=1990,
            end_year=1990,
            requested_variables=["2m_temperature"],
        )


def test_verify_timeseries_unexpected_dims_raises() -> None:
    """Verify ValueError is raised if unexpected dimension (e.g. expver) is present."""
    ds = _make_dummy_timeseries_dataset(extra_dim=True)
    with pytest.raises(ValueError, match="Unexpected dimensions"):
        verify_timeseries_dataset(
            ds,
            lat_center=4.75,
            lon_center=-74.0,
            start_year=1990,
            end_year=1990,
            requested_variables=["2m_temperature"],
        )


def test_verify_timeseries_time_gaps_or_duplicates_raises() -> None:
    """Verify ValueError is raised if timeseries has gaps or duplicates."""
    # Duplicate
    times = pd.date_range("1990-01-01 00:00:00", "1991-01-01 23:00:00", freq="h")
    dup_times = times.insert(2, times[1])
    t2m_vals = np.ones(len(dup_times), dtype=np.float32) * 290.0
    ds_dup = xr.Dataset(
        data_vars={"t2m": (("valid_time",), t2m_vals, {"units": "K"})},
        coords={"valid_time": dup_times, "latitude": 4.75, "longitude": -74.0},
    )
    with pytest.raises(ValueError, match="duplicate timestamps"):
        verify_timeseries_dataset(
            ds_dup,
            lat_center=4.75,
            lon_center=-74.0,
            start_year=1990,
            end_year=1990,
            requested_variables=["2m_temperature"],
        )

    # Missing hour
    gap_times = times.delete(10)
    t2m_vals2 = np.ones(len(gap_times), dtype=np.float32) * 290.0
    ds_gap = xr.Dataset(
        data_vars={"t2m": (("valid_time",), t2m_vals2, {"units": "K"})},
        coords={"valid_time": gap_times, "latitude": 4.75, "longitude": -74.0},
    )
    with pytest.raises(ValueError, match="Time index mismatch"):
        verify_timeseries_dataset(
            ds_gap,
            lat_center=4.75,
            lon_center=-74.0,
            start_year=1990,
            end_year=1990,
            requested_variables=["2m_temperature"],
        )


def test_timeseries_split_and_era5adapter_compatibility(tmp_path: Path) -> None:
    """Verify splitting timeseries into annual files and ERA5Adapter equivalence."""

    ds = _make_dummy_timeseries_dataset(
        lat=4.75,
        lon=-74.0,
        start_year=1990,
        end_year=1991,
    )
    # 1990: 8760 h, 1991: 8760 h, 1992: 24 h (boundary)
    created = split_and_save_annual_files(
        ds=ds,
        lat_center=4.75,
        lon_center=-74.0,
        start_year=1990,
        end_year=1991,
        requested_variables=["2m_temperature", "total_precipitation"],
        out_dir=tmp_path,
    )
    assert len(created) == 6  # 3 years * 2 variables

    # Check files exist
    assert (tmp_path / "era5_2m_temperature_1990.nc").exists()
    assert (tmp_path / "era5_2m_temperature_1991.nc").exists()
    assert (tmp_path / "era5_2m_temperature_1992.nc").exists()
    assert (tmp_path / "era5_total_precipitation_1990.nc").exists()
    assert (tmp_path / "era5_total_precipitation_1991.nc").exists()
    assert (tmp_path / "era5_total_precipitation_1992.nc").exists()

    # Boundary file must have exactly 24 hours
    with xr.open_dataset(tmp_path / "era5_2m_temperature_1992.nc") as ds_b:
        assert ds_b.sizes["time"] == 24
        assert ds_b["latitude"].shape == (1,)
        assert ds_b["longitude"].shape == (1,)
        assert ds_b["t2m"].shape == (24, 1, 1)
        assert ds_b["t2m"].attrs["units"] == "K"

    # Verify ERA5Adapter loads these files identically
    t2m_files = sorted(tmp_path.glob("era5_2m_temperature_*.nc"))
    adapter = ERA5Adapter(t2m_files)
    pt = adapter.extract_point(4.75, -74.0, method="nearest")
    s_hourly = pt["t2m"].to_series()
    assert len(s_hourly) == 8760 + 8760 + 24
    assert float(pt.latitude.values) == 4.75
    assert float(pt.longitude.values) == -74.0
    assert np.allclose(np.asarray(s_hourly), 290.0)

    # Test daily temperature aggregation works cleanly
    df_daily = adapter.get_daily_temperature(
        latitude=4.75, longitude=-74.0, method="nearest", min_hours=24
    )
    assert not df_daily.empty
    assert "tmax" in df_daily.columns
    assert "tmin" in df_daily.columns

    # Test daily precipitation aggregation works cleanly
    tp_files = sorted(tmp_path.glob("era5_total_precipitation_*.nc"))
    adapter_tp = ERA5Adapter(tp_files)
    s_daily_tp = adapter_tp.get_daily_precipitation(
        latitude=4.75, longitude=-74.0, method="nearest", min_hours=24
    )
    assert not s_daily_tp.empty
    # tp synthetic was 0.001 m/h * 24 h = 0.024 m = 24.0 mm/day
    valid_tp = s_daily_tp.dropna()
    assert np.allclose(np.asarray(valid_tp), 24.0)


def test_run_timeseries_with_mock_client(tmp_path: Path) -> None:
    """End-to-end test of run_timeseries using a mocked CDS client."""
    mock_ds = _make_dummy_timeseries_dataset(
        lat=4.75,
        lon=-74.0,
        start_year=1990,
        end_year=1990,
    )
    dummy_nc_path = tmp_path / "mock_download.nc"
    mock_ds.to_netcdf(dummy_nc_path)

    class MockResult:
        def download(self, target: str) -> None:
            import shutil

            shutil.copy(dummy_nc_path, target)

    class MockClient:
        def retrieve(self, dataset: str, req: Dict[str, Any]) -> MockResult:
            return MockResult()

    out_dir = tmp_path / "ts_out"
    created = run_timeseries(
        station="test_st",
        lat=4.67,
        lon=-74.08,
        variables=["2m_temperature"],
        start_year=1990,
        end_year=1990,
        out_dir=out_dir,
        client=MockClient(),
    )
    assert len(created) == 2  # 1990 and 1991 (boundary)
    assert (out_dir / "era5_2m_temperature_1990.nc").exists()
    assert (out_dir / "era5_2m_temperature_1991.nc").exists()
    assert (out_dir / "manifest.json").exists()


def test_parse_args_source_required(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify --source is mandatory and rejects missing source without default."""

    import sys

    monkeypatch.setattr(sys, "argv", ["download_era5_years.py", "--start-year", "1990"])
    with pytest.raises(SystemExit):
        parse_args()


def test_parse_args_timeseries_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify parse_args parses timeseries arguments correctly."""
    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_era5_years.py",
            "--source",
            "timeseries",
            "--station",
            "21205710",
            "--lat",
            "4.67",
            "--lon",
            "-74.08",
            "--variable",
            "2m_temperature",
            "total_precipitation",
            "--start-year",
            "1990",
            "--end-year",
            "2011",
        ],
    )
    args = parse_args()
    assert args.source == "timeseries"
    assert args.station == "21205710"
    assert args.lat == 4.67
    assert args.lon == -74.08
    assert args.variable == ["2m_temperature", "total_precipitation"]
    assert args.start_year == 1990
    assert args.end_year == 2011


def test_manifest_timeseries_single_variable_does_not_list_total_precipitation(
    tmp_path: Path,
) -> None:
    """Requirement 6a: In timeseries mode, manifest must not list total_precipitation

    when only 2m_temperature was requested.
    """
    import json

    out_dir = tmp_path / "ts_manifest"
    check_or_update_manifest(
        out_dir=out_dir,
        source="timeseries",
        station="37015020",
        variables=["2m_temperature"],
        location={"latitude": 7.25, "longitude": -72.75},
    )

    with open(out_dir / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["source"] == "timeseries"
    assert manifest["station"] == "37015020"
    assert manifest["variables"] == ["2m_temperature"]
    assert manifest["variable"] == "2m_temperature"
    assert "total_precipitation" not in manifest["variables"]


def test_validate_cli_single_levels_multiple_variables_raises_value_error() -> None:
    """Requirement 6b: single-levels with >1 variable raises explicit ValueError."""
    args = parse_args(
        [
            "--source",
            "single-levels",
            "--station",
            "21205710",
            "--variable",
            "2m_temperature",
            "total_precipitation",
        ]
    )
    with pytest.raises(
        ValueError, match="El modo single-levels solo admite exactamente una variable"
    ):
        validate_cli_args(args)


def test_validate_cli_single_levels_with_lat_lon_raises_value_error() -> None:
    """Requirement 6b: single-levels with --lat/--lon raises explicit ValueError."""
    args = parse_args(
        [
            "--source",
            "single-levels",
            "--station",
            "21205710",
            "--lat",
            "4.6",
            "--lon",
            "-74.1",
        ]
    )
    with pytest.raises(
        ValueError,
        match="Los argumentos --lat y --lon son exclusivos del modo timeseries",
    ):
        validate_cli_args(args)


def test_validate_cli_timeseries_with_area_raises_value_error() -> None:
    """Requirement 6b: timeseries with --area raises explicit ValueError."""
    args = parse_args(
        [
            "--source",
            "timeseries",
            "--station",
            "37015020",
            "--lat",
            "7.18",
            "--lon",
            "-72.86",
            "--variable",
            "2m_temperature",
            "--area",
            "7.3",
            "-72.9",
            "7.2",
            "-72.8",
        ]
    )
    with pytest.raises(
        ValueError, match="El argumento --area es exclusivo del modo single-levels"
    ):
        validate_cli_args(args)


def test_validate_cli_timeseries_invalid_variable_raises_value_error() -> None:
    """Requirement 6c: timeseries with invalid variable raises ValueError."""
    args = parse_args(
        [
            "--source",
            "timeseries",
            "--station",
            "37015020",
            "--lat",
            "7.18",
            "--lon",
            "-72.86",
            "--variable",
            "10m_u_component_of_wind",
        ]
    )
    with pytest.raises(
        ValueError,
        match="Variable '10m_u_component_of_wind' no es válida para el modo timeseries",
    ):
        validate_cli_args(args)


# ---------------------------------------------------------------------------
# Item 11: CLI tests for flags, verify multi-var, dry-run no-write, manifest timing
# ---------------------------------------------------------------------------


def test_validate_cli_timeseries_rejects_download_static() -> None:
    """timeseries mode rejects --download-static with ValueError."""
    args = parse_args(
        [
            "--source",
            "timeseries",
            "--station",
            "37015020",
            "--lat",
            "7.18",
            "--lon",
            "-72.86",
            "--variable",
            "2m_temperature",
            "--download-static",
        ]
    )
    with pytest.raises(ValueError, match="exclusivos del modo single-levels"):
        validate_cli_args(args)


def test_validate_cli_timeseries_rejects_static_only() -> None:
    """timeseries mode rejects --static-only with ValueError."""
    args = parse_args(
        [
            "--source",
            "timeseries",
            "--station",
            "37015020",
            "--lat",
            "7.18",
            "--lon",
            "-72.86",
            "--variable",
            "2m_temperature",
            "--static-only",
        ]
    )
    with pytest.raises(ValueError, match="exclusivos del modo single-levels"):
        validate_cli_args(args)


def test_validate_cli_single_levels_rejects_base_out_dir(tmp_path: Path) -> None:
    """single-levels mode rejects --base-out-dir with ValueError."""
    args = parse_args(
        [
            "--source",
            "single-levels",
            "--station",
            "21205710",
            "--base-out-dir",
            str(tmp_path),
        ]
    )
    with pytest.raises(
        ValueError, match="El argumento --base-out-dir es exclusivo del modo timeseries"
    ):
        validate_cli_args(args)


def test_timeseries_verify_verifies_all_requested_variables(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--verify in timeseries mode calls run_verify for every requested variable."""
    import sys

    from scripts import download_era5_years

    verified_vars: list[str] = []

    def mock_run_verify(
        out_dir: Path, start_year: int, end_year: int, variable: str
    ) -> None:
        verified_vars.append(variable)

    monkeypatch.setattr(download_era5_years, "run_verify", mock_run_verify)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_era5_years.py",
            "--source",
            "timeseries",
            "--station",
            "37015020",
            "--lat",
            "7.18",
            "--lon",
            "-72.86",
            "--variable",
            "2m_temperature",
            "total_precipitation",
            "--out-dir",
            str(tmp_path),
            "--verify",
        ],
    )
    download_era5_years.main()
    assert verified_vars == ["2m_temperature", "total_precipitation"]


def test_timeseries_dry_run_does_not_write_manifest_or_create_dir(
    tmp_path: Path,
) -> None:
    """dry-run in timeseries mode must not create directory or write manifest."""
    from scripts.download_era5_years import run_timeseries

    out_dir = tmp_path / "never_created_station"
    assert not out_dir.exists()

    res = run_timeseries(
        station="37015020",
        lat=7.1869,
        lon=-72.8686,
        variables=["2m_temperature"],
        start_year=2000,
        end_year=2002,
        out_dir=out_dir,
        dry_run=True,
    )
    assert res == []
    assert not out_dir.exists()
    assert not (out_dir / "manifest.json").exists()


def test_manifest_written_only_after_successful_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """manifest.json is written only after a successful download and file splitting."""
    from scripts.download_era5_years import run_timeseries

    out_dir = tmp_path / "mock_station"
    manifest_file = out_dir / "manifest.json"

    dummy_nc_path = tmp_path / "mock_download.nc"
    mock_ds = _make_dummy_timeseries_dataset(
        lat=7.25,
        lon=-72.75,
        start_year=2000,
        end_year=2000,
    )
    mock_ds.to_netcdf(dummy_nc_path)

    class MockResult:
        def download(self, target: str) -> None:
            import shutil

            # Before download finishes, manifest must NOT exist
            assert not manifest_file.exists()
            shutil.copy(dummy_nc_path, target)

    class MockClient:
        def retrieve(self, dataset: str, req: Dict[str, Any]) -> MockResult:
            return MockResult()

    assert not manifest_file.exists()
    res = run_timeseries(
        station="37015020",
        lat=7.1869,
        lon=-72.8686,
        variables=["2m_temperature"],
        start_year=2000,
        end_year=2000,
        out_dir=out_dir,
        dry_run=False,
        client=MockClient(),
    )
    assert len(res) == 2
    assert manifest_file.exists()
