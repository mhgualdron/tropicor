"""Unit tests for the Andean station registry and loader (Protocol v1.4, Appendix F).

All tests use either the official versioned registry or synthetic test registries.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tropicor.core.andean_registry import (
    DEFAULT_ANDEAN_REGISTRY_PATH,
    AndeanStation,
    get_andean_partition,
    get_andean_station,
    load_andean_registry,
    request_andean_validation_partition,
)


def test_default_registry_exists_and_contains_5_complete_stations() -> None:
    """Verify default registry file loads 5 complete stations with valid metadata."""
    assert DEFAULT_ANDEAN_REGISTRY_PATH.exists()
    stations = load_andean_registry()

    expected_codes = {"24025040", "26135040", "27015070", "37015020", "52045010"}
    assert set(stations.keys()) == expected_codes

    for code, st in stations.items():
        assert isinstance(st, AndeanStation)
        assert st.codigo == code
        assert len(st.nombre) > 0
        assert -5.0 <= st.latitud_cne <= 14.0
        assert -80.0 <= st.longitud_cne <= -66.0
        assert st.altitud_cne > 0
        assert st.cal_block[0] <= st.cal_block[1]
        assert st.val_block[0] <= st.val_block[1]
        # cal_end must be strictly less than val_start
        assert st.cal_block[1] < st.val_block[0]
        # valid_years must be non-empty sequence of ints
        assert len(st.valid_years) >= 20


def test_load_registry_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    """Non-existent registry file raises FileNotFoundError."""
    missing = tmp_path / "non_existent.json"
    with pytest.raises(FileNotFoundError, match="not found"):
        load_andean_registry(missing)


def test_load_registry_missing_required_key_raises_value_error(tmp_path: Path) -> None:
    """Registry entry missing a required key raises ValueError."""
    bad_registry = {
        "version": "1.4",
        "stations": {
            "99001": {
                "codigo": "99001",
                "nombre": "BAD STATION",
                # missing latitud_cne
                "longitud_cne": -73.0,
                "altitud_cne": 2000.0,
                "celda_era5": {"lat": 6.5, "lon": -73.0},
                "cal_block": [1990, 2000],
                "val_block": [2001, 2010],
                "valid_years": [1990, 2000],
            }
        },
    }
    p = tmp_path / "bad_reg.json"
    p.write_text(json.dumps(bad_registry), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required keys.*latitud_cne"):
        load_andean_registry(p)


def test_load_registry_cal_end_ge_val_start_raises_value_error(tmp_path: Path) -> None:
    """Registry with cal_end >= val_start raises ValueError."""
    bad_registry = {
        "version": "1.4",
        "stations": {
            "99002": {
                "codigo": "99002",
                "nombre": "OVERLAPPING DATES",
                "latitud_cne": 6.0,
                "longitud_cne": -73.0,
                "altitud_cne": 2000.0,
                "celda_era5": {"lat": 6.0, "lon": -73.0},
                "cal_block": [1990, 2005],
                "val_block": [2005, 2015],  # cal_end == val_start!
                "valid_years": [1990, 2005, 2015],
            }
        },
    }
    p = tmp_path / "bad_overlap.json"
    p.write_text(json.dumps(bad_registry), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="calibration end .* must be strictly less than validation start",
    ):
        load_andean_registry(p)


def test_load_registry_overlapping_blocks_raises_value_error(tmp_path: Path) -> None:
    """Registry with inverted overlapping blocks raises ValueError."""
    bad_registry = {
        "version": "1.4",
        "stations": {
            "99003": {
                "codigo": "99003",
                "nombre": "INVERTED BLOCKS",
                "latitud_cne": 6.0,
                "longitud_cne": -73.0,
                "altitud_cne": 2000.0,
                "celda_era5": {"lat": 6.0, "lon": -73.0},
                "cal_block": [2005, 2015],
                "val_block": [1990, 2004],
                "valid_years": [1990, 2005],
            }
        },
    }
    p = tmp_path / "bad_inverted.json"
    p.write_text(json.dumps(bad_registry), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="calibration end .* must be strictly less than validation start",
    ):
        load_andean_registry(p)


def test_get_andean_station_not_found_raises_value_error() -> None:
    """Unregistered station raises ValueError with informative message."""
    with pytest.raises(
        ValueError, match="Station '99999999' not found in Andean registry"
    ):
        get_andean_station("99999999")


def test_get_andean_partition() -> None:
    """get_andean_partition retrieves pre-specified cal and val ranges."""
    cal, val = get_andean_partition("temperature", "37015020")
    assert cal == (1991, 2009)
    assert val == (2010, 2020)

    # Unknown variable raises ValueError
    with pytest.raises(ValueError, match="not currently supported"):
        get_andean_partition("precipitation", "37015020")


def test_request_andean_validation_partition_raises_value_error() -> None:
    """Requesting validation data for an Andean station strictly raises ValueError."""
    with pytest.raises(
        ValueError,
        match="Validation partition for Andean station '37015020' is strictly closed",
    ):
        request_andean_validation_partition("37015020")


def test_andean_registry_package_data_loading() -> None:
    """Verify andean_station_registry.json can be loaded from package data."""
    import importlib.resources

    # Verify resource file exists in package data
    ref = importlib.resources.files("tropicor.data").joinpath(
        "andean_station_registry.json"
    )
    assert ref.is_file()

    # Load using default loader path
    stations = load_andean_registry()
    assert len(stations) == 5
    assert "37015020" in stations

    # Verify source SHA hashes are recorded
    with ref.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert "source_coverage_audit_sha256" in data
    assert len(data["source_coverage_audit_sha256"]) == 64
    assert "source_download_plan_sha256" in data
    assert len(data["source_download_plan_sha256"]) == 64


# ---------------------------------------------------------------------------
# Consistency guards (temporary registries only; the real JSON is untouched)
# ---------------------------------------------------------------------------


def _valid_station(**overrides: object) -> dict:
    """Synthetic station: valid years 1990-2009 -> cal [1990,1999], val [2000,2009]."""
    st: dict = {
        "codigo": "99100",
        "nombre": "SYNTHETIC",
        "latitud_cne": 6.1,
        "longitud_cne": -73.1,
        "altitud_cne": 2000.0,
        "celda_era5": {"lat": 6.0, "lon": -73.0},
        "cal_block": [1990, 1999],
        "val_block": [2000, 2009],
        "valid_years": list(range(1990, 2010)),
    }
    st.update(overrides)
    return st


def _write_registry(tmp_path: Path, station: dict, version: object = "1.4") -> Path:
    p = tmp_path / "reg.json"
    p.write_text(
        json.dumps({"version": version, "stations": {"99100": station}}),
        encoding="utf-8",
    )
    return p


def test_synthetic_valid_registry_loads(tmp_path: Path) -> None:
    """Baseline: the synthetic station passes every guard."""
    stations = load_andean_registry(_write_registry(tmp_path, _valid_station()))
    assert stations["99100"].cal_block == (1990, 1999)


def test_blocks_not_matching_optimal_partition_raise(tmp_path: Path) -> None:
    """(a) Endpoints valid, but cut differs from compute_optimal_partition."""
    st = _valid_station(cal_block=[1990, 2001], val_block=[2002, 2009])
    with pytest.raises(ValueError, match="do not match compute_optimal_partition"):
        load_andean_registry(_write_registry(tmp_path, st))


def test_block_endpoints_not_valid_years_raise(tmp_path: Path) -> None:
    """(b) cal_block start not in valid_years; and cal_end not last before cut."""
    years = [y for y in range(1990, 2010) if y != 1999]
    st1 = _valid_station(cal_block=[1989, 1998], valid_years=years)
    with pytest.raises(ValueError, match="must both be in valid_years"):
        load_andean_registry(_write_registry(tmp_path, st1))

    st2 = _valid_station(cal_block=[1990, 1999], val_block=[2001, 2009])
    with pytest.raises(ValueError, match="must be the last valid year before"):
        load_andean_registry(_write_registry(tmp_path, st2))


def test_celda_not_grid_multiple_raises(tmp_path: Path) -> None:
    """(c) celda_era5 not a multiple of 0.25 degrees."""
    st = _valid_station(celda_era5={"lat": 6.1, "lon": -73.0})
    with pytest.raises(ValueError, match="not a multiple of 0.25"):
        load_andean_registry(_write_registry(tmp_path, st))


def test_registry_version_mismatch_raises(tmp_path: Path) -> None:
    """(d) version != '1.4' (wrong value and missing key)."""
    with pytest.raises(ValueError, match="version must be '1.4'"):
        load_andean_registry(_write_registry(tmp_path, _valid_station(), "1.3"))
    p = tmp_path / "nover.json"
    p.write_text(json.dumps({"stations": {"99100": _valid_station()}}), "utf-8")
    with pytest.raises(ValueError, match="version must be '1.4'"):
        load_andean_registry(p)


def test_celda_not_nearest_grid_node_raises(tmp_path: Path) -> None:
    """(e) celda_era5 displaced 0.25 deg (still multiple of 0.25) raises ValueError."""
    # latitud_cne=6.1 -> nearest is 6.0. Displaced +0.25 -> 6.25 (multiple, not nearest)
    st = _valid_station(celda_era5={"lat": 6.25, "lon": -73.0})
    with pytest.raises(ValueError, match="not the nearest 0.25-degree grid node"):
        load_andean_registry(_write_registry(tmp_path, st))

    # Verify that the real production registry continues to load without error
    real_stations = load_andean_registry()
    assert len(real_stations) == 5
