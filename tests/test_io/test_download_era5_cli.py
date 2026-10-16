"""Unit tests for ERA5 year-by-year download CLI (Fase 2)."""

from pathlib import Path

import pytest

from scripts.download_era5_years import (
    build_cds_request,
    build_static_cds_request,
    check_or_update_manifest,
    load_cds_credentials,
    run_dry_run,
    run_verify,
)


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
