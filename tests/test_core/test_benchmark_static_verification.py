"""Tests for Section J.2 ERA5 static-file verification."""

import importlib.util
import inspect
import io
import sys
from pathlib import Path
from typing import Any

import netCDF4  # Must fail if netCDF4 is missing, never skip
import numpy as np
import pytest
import xarray as xr

from tropicor.core.benchmark_loaders import (
    BENCHMARK_STATION_CONSTANTS,
    GRAVITY_M_S2,
    StaticVerification,
    verify_era5_static,
)

assert netCDF4 is not None  # Ensure engine import


def _make_synthetic_static(
    directory: Path,
    filename: str,
    cell_lat: float,
    cell_lon: float,
    z_m2_s2: float,
) -> Path:
    """Create a synthetic ERA5 static NetCDF file with surface geopotential."""
    ds = xr.Dataset(
        data_vars={
            "z": (
                ("latitude", "longitude"),
                np.array([[z_m2_s2]], dtype=np.float64),
                {"units": "m**2 s**-2"},
            )
        },
        coords={"latitude": [cell_lat], "longitude": [cell_lon]},
    )
    p = directory / filename
    ds.to_netcdf(p)
    return p


def _make_script_meta(station_id: str, static_file: Path) -> dict[str, Any]:
    """Build meta dict matching scripts/run_benchmark_lag_audit.py expectations."""
    c = BENCHMARK_STATION_CONSTANTS[station_id]
    return {
        "static_file": static_file,
        "latitude": c.latitude,
        "longitude": c.longitude,
        "elevation_m": c.elevation_m,
        "expected_cell_lat": c.cell_lat,
        "expected_cell_lon": c.cell_lon,
        "expected_z_era5_m": c.z_era5_m,
        "expected_delta_z_m": c.delta_z_m,
        "name": f"Station_{station_id}",
    }


def _load_frozen_audit_script(monkeypatch: pytest.MonkeyPatch):
    """Load verify_station_static_parameters from frozen script.

    Import-time side effects of the script (sys.path insertion, stdout/stderr
    UTF-8 reconfiguration) are isolated with monkeypatch.
    """
    script_path = Path("scripts/run_benchmark_lag_audit.py")
    assert script_path.is_file(), "Frozen audit script not found."
    with monkeypatch.context() as m:
        m.setattr(sys, "path", list(sys.path))
        m.setattr(sys, "stdout", io.StringIO())
        m.setattr(sys, "stderr", io.StringIO())
        spec = importlib.util.spec_from_file_location(
            "frozen_benchmark_lag_audit_l3a", script_path
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        func = mod.verify_station_static_parameters
    return func


def test_static_verification_match_both_stations(tmp_path: Path):
    """Test 2a: verify_era5_static matches exact J.2 table values for JB and AC."""
    for st_id, expected_delta_z in [("21205710", -201.4), ("23215030", -628.6)]:
        c = BENCHMARK_STATION_CONSTANTS[st_id]
        geopotential = c.z_era5_m * GRAVITY_M_S2
        static_file = _make_synthetic_static(
            tmp_path, f"static_{st_id}.nc", c.cell_lat, c.cell_lon, geopotential
        )

        res = verify_era5_static(static_file, st_id)
        assert isinstance(res, StaticVerification)
        assert res.station_id == st_id
        assert np.isclose(res.grid_lat, c.cell_lat, atol=1e-9)
        assert np.isclose(res.grid_lon, c.cell_lon, atol=1e-9)
        assert np.isclose(res.z_era5_m, c.z_era5_m, atol=1e-9)
        assert np.isclose(res.table_z_era5_m, c.z_era5_m, atol=1e-9)
        assert np.isclose(res.abs_difference_m, 0.0, atol=1e-9)
        assert np.isclose(res.delta_z_m, expected_delta_z, atol=1e-9)

        # delta_z_m = elevation_m - z_era5_m using recomputed z within 0.1 m
        recomputed_delta_z = c.elevation_m - res.z_era5_m
        assert np.isclose(res.delta_z_m, recomputed_delta_z, atol=0.1)


def test_static_verification_tolerances(tmp_path: Path):
    """Test 2b: z differing by 0.05m passes, 0.2m fails; boundary passes."""
    c = BENCHMARK_STATION_CONSTANTS["21205710"]

    # Difference +0.05 m passes
    z_pass = (c.z_era5_m + 0.05) * GRAVITY_M_S2
    f_pass = _make_synthetic_static(
        tmp_path, "static_pass_005.nc", c.cell_lat, c.cell_lon, z_pass
    )
    res_pass = verify_era5_static(f_pass, "21205710")
    assert np.isclose(res_pass.abs_difference_m, 0.05, atol=1e-5)

    # Difference +0.20 m fails with ValueError mentioning station and difference
    z_fail = (c.z_era5_m + 0.20) * GRAVITY_M_S2
    f_fail = _make_synthetic_static(
        tmp_path, "static_fail_020.nc", c.cell_lat, c.cell_lon, z_fail
    )
    with pytest.raises(ValueError) as exc_info:
        verify_era5_static(f_fail, "21205710")
    msg = str(exc_info.value)
    assert "21205710" in msg
    assert "0.20" in msg or "0.2" in msg
    assert "differs from Table J.2" in msg

    # Difference of 0.09999 m: Table J.2 condition is |diff| > 0.1 m,
    # so differences up to 0.10 m (e.g. 0.09999 m) must pass.
    z_boundary = (c.z_era5_m + 0.09999) * GRAVITY_M_S2
    f_boundary = _make_synthetic_static(
        tmp_path, "static_boundary.nc", c.cell_lat, c.cell_lon, z_boundary
    )
    res_boundary = verify_era5_static(f_boundary, "21205710")
    assert np.isclose(res_boundary.abs_difference_m, 0.09999, atol=1e-5)


def test_static_verification_cell_mismatch_raises(tmp_path: Path):
    """Test 2c: grid lacking the J.2 cell raises ValueError without return value."""
    # Place cell far away at (0.00, 0.00)
    c = BENCHMARK_STATION_CONSTANTS["21205710"]
    z_val = c.z_era5_m * GRAVITY_M_S2
    f_mismatch = _make_synthetic_static(
        tmp_path, "static_mismatch.nc", 0.00, 0.00, z_val
    )

    with pytest.raises(ValueError, match="does not match Table J.2 expected cell"):
        verify_era5_static(f_mismatch, "21205710")


def test_static_verification_matches_frozen_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Test 2d: comparison against frozen verify_station_static_parameters."""
    path_before = list(sys.path)
    stdout_before = sys.stdout
    stderr_before = sys.stderr

    frozen_verify = _load_frozen_audit_script(monkeypatch)

    assert sys.path == path_before
    assert sys.stdout is stdout_before
    assert sys.stderr is stderr_before

    # Scenario 1: Exact match on JB (21205710)
    c_jb = BENCHMARK_STATION_CONSTANTS["21205710"]
    f_jb = _make_synthetic_static(
        tmp_path,
        "scen_jb_match.nc",
        c_jb.cell_lat,
        c_jb.cell_lon,
        c_jb.z_era5_m * GRAVITY_M_S2,
    )
    meta_jb = _make_script_meta("21205710", f_jb)
    res_pkg_jb = verify_era5_static(f_jb, "21205710")
    res_scr_jb = frozen_verify("21205710", meta_jb)
    assert np.isclose(res_pkg_jb.grid_lat, res_scr_jb["cell_latitude"], atol=1e-9)
    assert np.isclose(res_pkg_jb.grid_lon, res_scr_jb["cell_longitude"], atol=1e-9)
    assert np.isclose(res_pkg_jb.z_era5_m, res_scr_jb["z_era5_m"], atol=1e-9)
    assert np.isclose(res_pkg_jb.delta_z_m, res_scr_jb["delta_z_m"], atol=1e-9)

    # Scenario 2: Exact match on AC (23215030)
    c_ac = BENCHMARK_STATION_CONSTANTS["23215030"]
    f_ac = _make_synthetic_static(
        tmp_path,
        "scen_ac_match.nc",
        c_ac.cell_lat,
        c_ac.cell_lon,
        c_ac.z_era5_m * GRAVITY_M_S2,
    )
    meta_ac = _make_script_meta("23215030", f_ac)
    res_pkg_ac = verify_era5_static(f_ac, "23215030")
    res_scr_ac = frozen_verify("23215030", meta_ac)
    assert np.isclose(res_pkg_ac.grid_lat, res_scr_ac["cell_latitude"], atol=1e-9)
    assert np.isclose(res_pkg_ac.grid_lon, res_scr_ac["cell_longitude"], atol=1e-9)
    assert np.isclose(res_pkg_ac.z_era5_m, res_scr_ac["z_era5_m"], atol=1e-9)
    assert np.isclose(res_pkg_ac.delta_z_m, res_scr_ac["delta_z_m"], atol=1e-9)

    # Scenario 3: +0.05m tolerance pass
    f_tol = _make_synthetic_static(
        tmp_path,
        "scen_tol_pass.nc",
        c_jb.cell_lat,
        c_jb.cell_lon,
        (c_jb.z_era5_m + 0.05) * GRAVITY_M_S2,
    )
    meta_tol = _make_script_meta("21205710", f_tol)
    res_pkg_tol = verify_era5_static(f_tol, "21205710")
    res_scr_tol = frozen_verify("21205710", meta_tol)
    assert np.isclose(res_pkg_tol.z_era5_m, res_scr_tol["z_era5_m"], atol=1e-9)

    # Scenario 4: +0.20m tolerance fail
    f_fail = _make_synthetic_static(
        tmp_path,
        "scen_tol_fail.nc",
        c_jb.cell_lat,
        c_jb.cell_lon,
        (c_jb.z_era5_m + 0.20) * GRAVITY_M_S2,
    )
    meta_fail = _make_script_meta("21205710", f_fail)
    with pytest.raises(ValueError):
        verify_era5_static(f_fail, "21205710")
    with pytest.raises(ValueError):
        frozen_verify("21205710", meta_fail)

    # Scenario 5: Cell mismatch fail
    f_mis = _make_synthetic_static(
        tmp_path, "scen_mis_fail.nc", 1.0, 1.0, c_jb.z_era5_m * GRAVITY_M_S2
    )
    meta_mis = _make_script_meta("21205710", f_mis)
    with pytest.raises(ValueError):
        verify_era5_static(f_mis, "21205710")
    with pytest.raises(ValueError):
        frozen_verify("21205710", meta_mis)


def test_static_verification_sign_guard_and_no_paths_in_exceptions(tmp_path: Path):
    """Test 2e & 2f: source guard, negative delta_z, and no paths in exceptions."""
    import tropicor.core.benchmark_loaders as bmod

    src = inspect.getsource(bmod)
    assert "delta_h_m" not in src
    assert "statics_g7" not in src

    # Negative delta_z: station elevation < z_era5 (e.g. JB 2552.0 < 2753.4)
    c = BENCHMARK_STATION_CONSTANTS["21205710"]
    assert c.elevation_m < c.z_era5_m
    f_neg = _make_synthetic_static(
        tmp_path, "static_neg.nc", c.cell_lat, c.cell_lon, c.z_era5_m * GRAVITY_M_S2
    )
    res = verify_era5_static(f_neg, "21205710")
    assert res.delta_z_m < 0.0
    assert np.isclose(res.delta_z_m, -201.4, atol=1e-9)

    # Missing file exception must contain only filename, no dir path
    non_existent = tmp_path / "sub_dir" / "nested_missing_static.nc"
    with pytest.raises(FileNotFoundError) as exc_fnf:
        verify_era5_static(non_existent, "21205710")
    msg_fnf = str(exc_fnf.value)
    assert "nested_missing_static.nc" in msg_fnf
    assert "/" not in msg_fnf and "\\" not in msg_fnf


def test_static_verification_unregistered_station_raises(tmp_path: Path):
    """Verify unregistered station raises ValueError."""
    c = BENCHMARK_STATION_CONSTANTS["21205710"]
    f = _make_synthetic_static(
        tmp_path, "static_unreg.nc", c.cell_lat, c.cell_lon, c.z_era5_m * GRAVITY_M_S2
    )
    with pytest.raises(ValueError, match="not in BENCHMARK_STATION_CONSTANTS"):
        verify_era5_static(f, "99999999")
