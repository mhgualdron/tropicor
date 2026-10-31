"""Unit tests for primary benchmark lag audit (Protocol Appendix J.5 / Section 8).

Verifies:
1. Validation mode call order: static/ERA5 checks precede ledger call,
   and ledger call precedes any observation file read (independent of local data).
2. Preflight failure on second station blocks ledger access for first station.
3. Missing ERA5 year aborts with FileNotFoundError and never calls ledger.
4. Source equivalence failure aborts with ValueError and never calls ledger.
5. Validation mode requires explicit --confirm-written-authorization CLI flag.
6. Validation mode enforces git-clean guard on scripts and tropicor/.
7. Calibration mode verifies exact equivalence against official loader.
8. Duplicate dates and non-numeric entries reporting (value+NaN duplicate handling).
9. Calibration mode never accesses the ledger or calls validate_validation_access.
10. Static parameter verification strictly enforces Table J.2 coordinates and z_era5.
11. Metric boundary enforcement: output contains zero bias or dispersion metrics.
12. Completeness, preflight, and source-equivalence unit functions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from scripts.run_benchmark_lag_audit import (
    BENCHMARK_STATIONS,
    DEFAULT_LEDGER_PATH,
    REPO_ROOT,
    audit_benchmark_station,
    check_era5_completeness,
    inspect_observation_file_quality,
    main,
    preflight_validation_checks,
    read_and_slice_benchmark_observations,
    verify_era5_source_equivalence,
    verify_station_static_parameters,
)


def _make_synthetic_benchmark_audit_data(
    start_year: int, end_year: int
) -> Tuple[pd.Series, pd.Series, Dict[int, pd.Series], Dict[int, pd.Series]]:
    """Generate synthetic observation and model series without reading real files."""
    dt_index = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
    n = len(dt_index)
    rng = np.random.default_rng(42)

    obs_tx = pd.Series(20.0 + rng.normal(0, 2.0, n), index=dt_index, name="obs_tmax")
    obs_tn = pd.Series(10.0 + rng.normal(0, 1.5, n), index=dt_index, name="obs_tmin")

    mod_tx_by_lag: Dict[int, pd.Series] = {}
    mod_tn_by_lag: Dict[int, pd.Series] = {}
    for k in [-2, -1, 0, 1, 2]:
        mod_tx_by_lag[k] = obs_tx.shift(k) + rng.normal(0, 0.5, n)
        mod_tn_by_lag[k] = obs_tn.shift(k) + rng.normal(0, 0.5, n)

    return obs_tx, obs_tn, mod_tx_by_lag, mod_tn_by_lag


def _make_mock_benchmark_stations(tmp_path: Path) -> Dict[str, Dict[str, Any]]:
    """Generate mock BENCHMARK_STATIONS dictionary with isolated files in tmp_path."""
    mock_dict: Dict[str, Dict[str, Any]] = {}
    for st_id, meta in BENCHMARK_STATIONS.items():
        st_meta = dict(meta)
        tmax_csv = tmp_path / f"TempMaxima_{st_id}.csv"
        tmin_csv = tmp_path / f"TempMinima_{st_id}.csv"
        tmax_csv.write_text("CodigoEstacion,Fecha,Valor\n", encoding="utf-8")
        tmin_csv.write_text("CodigoEstacion,Fecha,Valor\n", encoding="utf-8")
        st_meta["obs_tmax_csv"] = tmax_csv
        st_meta["obs_tmin_csv"] = tmin_csv
        mock_dict[st_id] = st_meta
    return mock_dict


def test_validation_mode_calls_ledger_before_file_read(tmp_path: Path) -> None:
    """Verify static/ERA5 checks precede ledger, and ledger precedes read."""
    call_sequence: List[str] = []
    obs_tx, obs_tn, mod_tx_dict, mod_tn_dict = _make_synthetic_benchmark_audit_data(
        2005, 2006
    )
    mock_stations = _make_mock_benchmark_stations(tmp_path)

    def mock_verify_static(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        call_sequence.append("static_verification")
        return {"static_sha256": "mock_static_sha", "verified": True}

    def mock_check_era5(
        era5_dir: Path, station_id: str, required_years: Sequence[int]
    ) -> Tuple[List[Path], Dict[str, str]]:
        call_sequence.append("era5_completeness")
        return [Path("mock.nc")], {"mock.nc": "hash_val"}

    def mock_source_equiv(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        call_sequence.append("source_equivalence")
        return {
            "calibration_years": [1990, 2004],
            "equivalent": True,
            "era5_hourly_hashes": {},
            "era5_ts_hashes": {},
            "max_abs_diff_k": 1e-4,
            "n_hours": 1000,
            "station_id": "21205710",
        }

    def mock_compute_sha(p: Path) -> str:
        call_sequence.append("obs_hashes")
        return "mock_obs_hash"

    def mock_validate_access(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        call_sequence.append("ledger_access")
        return {
            "timestamp": "2026-10-30T00:00:00Z",
            "commit_hash": "mock_commit",
            "purpose": kwargs.get("purpose", "lag_audit"),
            "station_id": kwargs.get("station_id", "21205710"),
            "variable": kwargs.get("variable", "temperature"),
            "seed": kwargs.get("seed"),
            "config": kwargs.get("config", {}),
        }

    read_calls = [obs_tx, obs_tn]

    def mock_read_obs(*args: Any, **kwargs: Any) -> pd.Series:
        call_sequence.append("read_obs")
        return read_calls.pop(0)

    def mock_extract_era5(
        *args: Any, **kwargs: Any
    ) -> Tuple[Dict[int, pd.Series], Dict[int, pd.Series]]:
        call_sequence.append("extract_era5")
        return mod_tx_dict, mod_tn_dict

    with (
        patch.dict(
            "scripts.run_benchmark_lag_audit.BENCHMARK_STATIONS",
            mock_stations,
        ),
        patch(
            "scripts.run_benchmark_lag_audit.verify_station_static_parameters",
            side_effect=mock_verify_static,
        ),
        patch(
            "scripts.run_benchmark_lag_audit.check_era5_completeness",
            side_effect=mock_check_era5,
        ),
        patch(
            "scripts.run_benchmark_lag_audit.verify_era5_source_equivalence",
            side_effect=mock_source_equiv,
        ),
        patch(
            "scripts.run_benchmark_lag_audit.compute_file_sha256",
            side_effect=mock_compute_sha,
        ),
        patch(
            "scripts.run_benchmark_lag_audit.validate_validation_access",
            side_effect=mock_validate_access,
        ) as mock_ledger,
        patch(
            "scripts.run_benchmark_lag_audit.read_and_slice_benchmark_observations",
            side_effect=mock_read_obs,
        ),
        patch(
            "scripts.run_benchmark_lag_audit.extract_and_aggregate_benchmark_era5",
            side_effect=mock_extract_era5,
        ),
        patch("subprocess.run") as mock_subproc,
    ):
        mock_subproc.return_value = MagicMock(stdout="", returncode=0)

        audit_benchmark_station(
            station_id="21205710",
            mode="validation",
            output_dir=tmp_path,
            confirm_written_authorization=True,
            replicas=10,
        )

        assert mock_ledger.called
        call_kwargs = mock_ledger.call_args[1]
        assert call_kwargs["open_validation"] is True
        assert call_kwargs["purpose"] == "lag_audit"
        assert call_kwargs["station_id"] == "21205710"
        assert call_kwargs["variable"] == "temperature"

        # Assert strict check order
        assert "static_verification" in call_sequence
        assert "era5_completeness" in call_sequence
        assert "source_equivalence" in call_sequence
        assert "obs_hashes" in call_sequence
        assert "ledger_access" in call_sequence
        assert "read_obs" in call_sequence

        static_idx = call_sequence.index("static_verification")
        era5_idx = call_sequence.index("era5_completeness")
        equiv_idx = call_sequence.index("source_equivalence")
        obs_hash_idx = call_sequence.index("obs_hashes")
        ledger_idx = call_sequence.index("ledger_access")
        first_read_idx = call_sequence.index("read_obs")

        assert (
            static_idx
            < era5_idx
            < equiv_idx
            < obs_hash_idx
            < ledger_idx
            < first_read_idx
        ), (
            "Expected strict check order: static < era5 < equiv < "
            f"obs_hash < ledger < read. Observed sequence: {call_sequence}"
        )


def test_validation_mode_preflight_failure_on_second_station_blocks_first_station(
    tmp_path: Path,
) -> None:
    """Preflight failure on second station must block ledger access on first."""
    mock_stations = _make_mock_benchmark_stations(tmp_path)

    def mock_preflight(
        station_id: str,
        meta: Optional[Dict[str, Any]] = None,
        confirm_written_authorization: bool = False,
    ) -> Dict[str, Any]:
        if station_id == "21205710":
            return {
                "era5_dir": tmp_path,
                "era5_files": [Path("nc1")],
                "era5_hashes": {"nc1": "hash1"},
                "obs_tmax_hash": "hash_tx",
                "obs_tmin_hash": "hash_tn",
                "required_era5_years": [2004, 2005],
                "source_equiv": {"equivalent": True, "max_abs_diff_k": 0.0},
                "start_year": 2005,
                "end_year": 2006,
                "static_info": {"static_sha256": "static_sha"},
                "station_id": station_id,
            }
        raise FileNotFoundError(
            f"Preflight failure simulated for second station {station_id}"
        )

    with (
        patch.dict(
            "scripts.run_benchmark_lag_audit.BENCHMARK_STATIONS",
            mock_stations,
        ),
        patch(
            "scripts.run_benchmark_lag_audit.preflight_validation_checks",
            side_effect=mock_preflight,
        ),
        patch(
            "scripts.run_benchmark_lag_audit.validate_validation_access"
        ) as mock_ledger,
        patch(
            "sys.argv",
            [
                "run_benchmark_lag_audit.py",
                "--mode",
                "validation",
                "--station",
                "both",
                "--confirm-written-authorization",
                "--output-dir",
                str(tmp_path),
            ],
        ),
    ):
        with pytest.raises(
            FileNotFoundError,
            match="Preflight failure simulated for second station 23215030",
        ):
            main()

        # Crucial invariant: ledger access was NEVER attempted for station 1
        assert not mock_ledger.called


def test_validation_mode_missing_era5_year_aborts_without_ledger(
    tmp_path: Path,
) -> None:
    """Missing required ERA5 year must raise FileNotFoundError and not touch ledger."""
    with (
        patch("subprocess.run") as mock_subproc,
        patch(
            "scripts.run_benchmark_lag_audit.verify_station_static_parameters",
            return_value={"static_sha256": "mock_sha", "verified": True},
        ),
        patch(
            "scripts.run_benchmark_lag_audit.check_era5_completeness",
            side_effect=FileNotFoundError(
                "Missing required ERA5 file(s) for station 21205710"
            ),
        ),
        patch(
            "scripts.run_benchmark_lag_audit.validate_validation_access"
        ) as mock_ledger,
        patch(
            "scripts.run_benchmark_lag_audit.read_and_slice_benchmark_observations"
        ) as mock_read,
    ):
        mock_subproc.return_value = MagicMock(stdout="", returncode=0)

        with pytest.raises(FileNotFoundError, match="Missing required ERA5 file"):
            audit_benchmark_station(
                station_id="21205710",
                mode="validation",
                output_dir=tmp_path,
                confirm_written_authorization=True,
            )

        assert not mock_ledger.called
        assert not mock_read.called


def test_validation_mode_source_equivalence_failure_aborts_without_ledger(
    tmp_path: Path,
) -> None:
    """ERA5 source equivalence failure must abort and never touch the ledger."""
    with (
        patch("subprocess.run") as mock_subproc,
        patch(
            "scripts.run_benchmark_lag_audit.verify_station_static_parameters",
            return_value={"static_sha256": "mock_sha", "verified": True},
        ),
        patch(
            "scripts.run_benchmark_lag_audit.check_era5_completeness",
            return_value=([Path("mock.nc")], {"mock.nc": "hash"}),
        ),
        patch(
            "scripts.run_benchmark_lag_audit.verify_era5_source_equivalence",
            side_effect=ValueError(
                "ERA5 source equivalence failed: max |diff| > 1e-3 K"
            ),
        ),
        patch(
            "scripts.run_benchmark_lag_audit.validate_validation_access"
        ) as mock_ledger,
        patch(
            "scripts.run_benchmark_lag_audit.read_and_slice_benchmark_observations"
        ) as mock_read,
    ):
        mock_subproc.return_value = MagicMock(stdout="", returncode=0)

        with pytest.raises(ValueError, match="ERA5 source equivalence failed"):
            audit_benchmark_station(
                station_id="21205710",
                mode="validation",
                output_dir=tmp_path,
                confirm_written_authorization=True,
            )

        assert not mock_ledger.called
        assert not mock_read.called


def test_validation_mode_without_confirm_authorization_raises_permission_error(
    tmp_path: Path,
) -> None:
    """Validation without --confirm-written-authorization must abort immediately."""
    with (
        patch(
            "scripts.run_benchmark_lag_audit.validate_validation_access"
        ) as mock_ledger,
        patch(
            "scripts.run_benchmark_lag_audit.read_and_slice_benchmark_observations"
        ) as mock_read,
    ):
        with pytest.raises(PermissionError, match="--confirm-written-authorization"):
            audit_benchmark_station(
                station_id="21205710",
                mode="validation",
                output_dir=tmp_path,
                confirm_written_authorization=False,
            )

        assert not mock_ledger.called
        assert not mock_read.called


def test_validation_mode_aborts_on_dirty_working_tree(tmp_path: Path) -> None:
    """Validation mode aborts if target scripts or tropicor have uncommitted changes."""
    with (
        patch("subprocess.run") as mock_subproc,
        patch(
            "scripts.run_benchmark_lag_audit.validate_validation_access"
        ) as mock_ledger,
        patch(
            "scripts.run_benchmark_lag_audit.read_and_slice_benchmark_observations"
        ) as mock_read,
    ):
        mock_subproc.return_value = MagicMock(
            stdout=" M scripts/run_benchmark_lag_audit.py\n", returncode=0
        )

        with pytest.raises(
            RuntimeError, match="Uncommitted or untracked changes detected"
        ):
            audit_benchmark_station(
                station_id="21205710",
                mode="validation",
                output_dir=tmp_path,
                confirm_written_authorization=True,
            )

        assert not mock_ledger.called
        assert not mock_read.called


def test_calibration_mode_never_calls_ledger(tmp_path: Path) -> None:
    """Calibration mode must never call validate_validation_access or touch ledger."""
    obs_tx, obs_tn, mod_tx_dict, mod_tn_dict = _make_synthetic_benchmark_audit_data(
        1990, 1992
    )
    mock_stations = _make_mock_benchmark_stations(tmp_path)

    with (
        patch.dict(
            "scripts.run_benchmark_lag_audit.BENCHMARK_STATIONS",
            mock_stations,
        ),
        patch(
            "scripts.run_benchmark_lag_audit.validate_validation_access"
        ) as mock_ledger,
        patch(
            "scripts.run_benchmark_lag_audit.read_and_slice_benchmark_observations",
            side_effect=[obs_tx, obs_tn],
        ),
        patch(
            "scripts.run_benchmark_lag_audit.load_calibration_temperature_pair",
            return_value=(obs_tx, obs_tn, None, None),
        ),
        patch(
            "scripts.run_benchmark_lag_audit.inspect_observation_file_quality",
            return_value={
                "duplicate_dates_count": 0,
                "non_numeric_valor_count": 0,
            },
        ),
        patch(
            "scripts.run_benchmark_lag_audit.check_era5_completeness",
            return_value=([Path("nc1")], {"nc1": "hash1"}),
        ),
        patch(
            "scripts.run_benchmark_lag_audit.verify_era5_source_equivalence",
            return_value={
                "calibration_years": [1990, 1992],
                "equivalent": True,
                "era5_hourly_hashes": {},
                "era5_ts_hashes": {},
                "max_abs_diff_k": 1e-4,
                "n_hours": 1000,
                "station_id": "21205710",
            },
        ),
        patch(
            "scripts.run_benchmark_lag_audit.extract_and_aggregate_benchmark_era5",
            return_value=(mod_tx_dict, mod_tn_dict),
        ),
        patch(
            "scripts.run_benchmark_lag_audit.verify_station_static_parameters",
            return_value={
                "static_file": "local/era5_hourly/21205710/era5_static.nc",
                "static_sha256": "mock_sha",
                "verified": True,
            },
        ),
        patch(
            "scripts.run_benchmark_lag_audit.compute_file_sha256",
            return_value="fake_sha",
        ),
    ):
        res = audit_benchmark_station(
            station_id="21205710",
            mode="calibration",
            output_dir=tmp_path,
            replicas=10,
        )

        assert not mock_ledger.called
        assert res["station_id"] == "21205710"
        assert res["config"]["mode"] == "calibration"
        assert "validation_opening_record" not in res
        assert "observation_data_quality" in res
        assert (
            res["observation_data_quality"]["tmax"]["matches_official_loader"] is True
        )

        # Assert no path field in payload is absolute or
        # contains a backslash or a drive letter
        for key in ["era5_dir"]:
            val = res["config"][key]
            assert "\\" not in val, f"Path field config.{key} contains backslash: {val}"
            assert not (len(val) > 1 and val[1] == ":" and val[0].isalpha()), (
                f"Path field config.{key} contains drive letter: {val}"
            )
            assert not Path(val).is_absolute(), (
                f"Path field config.{key} is absolute: {val}"
            )

        for key in ["static_file"]:
            val = res["static_verification"][key]
            assert "\\" not in val, (
                f"Path field static_verification.{key} contains backslash: {val}"
            )
            assert not (len(val) > 1 and val[1] == ":" and val[0].isalpha()), (
                f"Path field static_verification.{key} contains drive letter: {val}"
            )
            assert not Path(val).is_absolute(), (
                f"Path field static_verification.{key} is absolute: {val}"
            )


def test_calibration_mode_observation_equivalence_and_data_quality(
    tmp_path: Path,
) -> None:
    """Test observation quality and equivalence with synthetic data (value+NaN)."""
    csv_data = """CodigoEstacion,Fecha,Valor
21205710,1990-01-01,20.5
21205710,1990-01-02,22.0
21205710,1990-01-02,
21205710,1990-01-03,21.5
21205710,1990-01-04,bad_str
"""
    obs_file = tmp_path / "obs_test.csv"
    obs_file.write_text(csv_data, encoding="utf-8")

    # Verify quality inspection counts
    quality = inspect_observation_file_quality(obs_file, "21205710", 1990, 1990)
    assert quality["duplicate_dates_count"] == 1  # 1990-01-02 appears twice
    assert quality["non_numeric_valor_count"] == 1  # 'bad_str' is non-numeric

    # Sliced observations test: mixed duplicate raises loud ValueError
    with pytest.raises(
        ValueError,
        match=(
            "Duplicate date group\\(s\\) mixing missing "
            "and non-missing raw Valor entries"
        ),
    ) as exc_info:
        read_and_slice_benchmark_observations(
            obs_file, "21205710", 1990, 1990, "obs_tmax"
        )
    assert "21205710" in str(exc_info.value)
    assert "obs_test.csv" in str(exc_info.value)
    assert "1990-01-02" in str(exc_info.value)

    # Test that mismatch against official loader raises ValueError
    mock_stations = _make_mock_benchmark_stations(tmp_path)
    obs_tx, obs_tn, mod_tx_dict, mod_tn_dict = _make_synthetic_benchmark_audit_data(
        1990, 1992
    )
    mismatched_tx = obs_tx.copy()
    mismatched_tx.iloc[0] = mismatched_tx.iloc[0] + 5.0  # cause deliberate mismatch

    with (
        patch.dict(
            "scripts.run_benchmark_lag_audit.BENCHMARK_STATIONS",
            mock_stations,
        ),
        patch(
            "scripts.run_benchmark_lag_audit.read_and_slice_benchmark_observations",
            side_effect=[obs_tx, obs_tn],
        ),
        patch(
            "scripts.run_benchmark_lag_audit.load_calibration_temperature_pair",
            return_value=(mismatched_tx, obs_tn, None, None),
        ),
        patch(
            "scripts.run_benchmark_lag_audit.check_era5_completeness",
            return_value=([Path("nc1")], {"nc1": "hash1"}),
        ),
        patch(
            "scripts.run_benchmark_lag_audit.verify_era5_source_equivalence",
            return_value={
                "calibration_years": [1990, 1992],
                "equivalent": True,
                "era5_hourly_hashes": {},
                "era5_ts_hashes": {},
                "max_abs_diff_k": 1e-4,
                "n_hours": 1000,
                "station_id": "21205710",
            },
        ),
        patch(
            "scripts.run_benchmark_lag_audit.verify_station_static_parameters",
            return_value={"static_sha256": "mock_sha", "verified": True},
        ),
        patch(
            "scripts.run_benchmark_lag_audit.compute_file_sha256",
            return_value="fake_sha",
        ),
    ):
        with pytest.raises(
            ValueError,
            match="Value mismatch between read_and_slice_benchmark_observations",
        ):
            audit_benchmark_station(
                station_id="21205710",
                mode="calibration",
                output_dir=tmp_path,
                replicas=10,
            )


def test_read_and_slice_benchmark_observations_duplicate_handling(
    tmp_path: Path,
) -> None:
    """Verify duplicate date resolution in read_and_slice_benchmark_observations."""
    # (i) Mixed value+NaN raises in both orders:
    # Value first
    csv_mix_val_first = """CodigoEstacion,Fecha,Valor
21205710,1990-01-01,20.5
21205710,1990-01-02,22.0
21205710,1990-01-02,
21205710,1990-01-03,21.5
"""
    f_mix_a = tmp_path / "obs_mix_a.csv"
    f_mix_a.write_text(csv_mix_val_first, encoding="utf-8")
    with pytest.raises(
        ValueError,
        match=(
            "Duplicate date group\\(s\\) mixing missing "
            "and non-missing raw Valor entries"
        ),
    ) as exc_a:
        read_and_slice_benchmark_observations(
            f_mix_a, "21205710", 1990, 1990, "obs_tmax"
        )
    assert "21205710" in str(exc_a.value)
    assert "obs_mix_a.csv" in str(exc_a.value)
    assert "1990-01-02" in str(exc_a.value)

    # NaN first
    csv_mix_nan_first = """CodigoEstacion,Fecha,Valor
21205710,1990-01-01,20.5
21205710,1990-01-02,
21205710,1990-01-02,22.0
21205710,1990-01-03,21.5
"""
    f_mix_b = tmp_path / "obs_mix_b.csv"
    f_mix_b.write_text(csv_mix_nan_first, encoding="utf-8")
    with pytest.raises(
        ValueError,
        match=(
            "Duplicate date group\\(s\\) mixing missing "
            "and non-missing raw Valor entries"
        ),
    ) as exc_b:
        read_and_slice_benchmark_observations(
            f_mix_b, "21205710", 1990, 1990, "obs_tmax"
        )
    assert "21205710" in str(exc_b.value)
    assert "obs_mix_b.csv" in str(exc_b.value)
    assert "1990-01-02" in str(exc_b.value)

    # (ii) Identical-value duplicates accepted and keep value; all-missing allowed
    csv_ident = """CodigoEstacion,Fecha,Valor
21205710,1990-01-01,20.5
21205710,1990-01-02,22.0
21205710,1990-01-02,22.0
21205710,1990-01-03,
21205710,1990-01-03,
"""
    f_ident = tmp_path / "obs_ident.csv"
    f_ident.write_text(csv_ident, encoding="utf-8")
    s_ident = read_and_slice_benchmark_observations(
        f_ident, "21205710", 1990, 1990, "obs_tmax"
    )
    assert s_ident.loc["1990-01-01"] == 20.5
    assert s_ident.loc["1990-01-02"] == 22.0
    assert pd.isna(s_ident.loc["1990-01-03"])

    # (iii) Distinct-value duplicates still raise
    csv_distinct = """CodigoEstacion,Fecha,Valor
21205710,1990-01-01,20.5
21205710,1990-01-02,22.0
21205710,1990-01-02,23.5
21205710,1990-01-03,21.5
"""
    f_distinct = tmp_path / "obs_distinct.csv"
    f_distinct.write_text(csv_distinct, encoding="utf-8")
    with pytest.raises(
        ValueError, match="Duplicate date with distinct values found"
    ) as exc_dist:
        read_and_slice_benchmark_observations(
            f_distinct, "21205710", 1990, 1990, "obs_tmax"
        )
    assert "obs_distinct.csv" in str(exc_dist.value)
    assert "21205710" in str(exc_dist.value)


def test_preflight_validation_checks_unit(tmp_path: Path) -> None:
    """Verify preflight_validation_checks runs verification gates and returns ctx."""
    mock_meta = {
        "name": "Mock Station",
        "latitude": 4.667867,
        "longitude": -74.101034,
        "val_years": (2005, 2006),
        "era5_ts_dir": tmp_path / "ts",
        "obs_tmax_csv": tmp_path / "mock_tx.csv",
        "obs_tmin_csv": tmp_path / "mock_tn.csv",
    }
    mock_meta["obs_tmax_csv"].write_text("tmax", encoding="utf-8")
    mock_meta["obs_tmin_csv"].write_text("tmin", encoding="utf-8")

    with (
        patch("subprocess.run") as mock_subproc,
        patch(
            "scripts.run_benchmark_lag_audit.verify_station_static_parameters",
            return_value={"static_sha256": "static_sha"},
        ),
        patch(
            "scripts.run_benchmark_lag_audit.check_era5_completeness",
            return_value=([Path("nc1")], {"nc1": "hash1"}),
        ),
        patch(
            "scripts.run_benchmark_lag_audit.verify_era5_source_equivalence",
            return_value={"equivalent": True, "max_abs_diff_k": 0.0},
        ),
        patch(
            "scripts.run_benchmark_lag_audit.compute_file_sha256",
            return_value="obs_sha",
        ),
    ):
        mock_subproc.return_value = MagicMock(stdout="", returncode=0)

        ctx = preflight_validation_checks(
            "21205710",
            meta=mock_meta,
            confirm_written_authorization=True,
        )
        assert ctx["station_id"] == "21205710"
        assert ctx["start_year"] == 2005
        assert ctx["end_year"] == 2006
        assert ctx["obs_tmax_hash"] == "obs_sha"
        assert ctx["obs_tmin_hash"] == "obs_sha"
        assert ctx["source_equiv"]["equivalent"] is True


def test_check_era5_completeness(tmp_path: Path) -> None:
    """Verify check_era5_completeness raises on missing files or duplicate files."""
    era_dir = tmp_path / "era5"
    era_dir.mkdir()

    # Create dummy netcdfs for 1990 and 1991
    f1990 = era_dir / "era5_2m_temperature_1990.nc"
    f1991 = era_dir / "era5_2m_temperature_1991.nc"
    f1990.write_text("dummy 1990", encoding="utf-8")
    f1991.write_text("dummy 1991", encoding="utf-8")

    # Success case
    files, hashes = check_era5_completeness(era_dir, "test_st", [1990, 1991])
    assert len(files) == 2
    assert f1990.name in hashes
    assert f1991.name in hashes

    # Missing year 1992
    with pytest.raises(FileNotFoundError, match=r"Missing years: \[1992\]"):
        check_era5_completeness(era_dir, "test_st", [1990, 1991, 1992])

    # Multiple files for year 1990
    dup_1990 = era_dir / "duplicate_era5_2m_temperature_1990.nc"
    dup_1990.write_text("dummy 1990 dup", encoding="utf-8")
    with pytest.raises(ValueError, match="Multiple ERA5 files found for year 1990"):
        check_era5_completeness(era_dir, "test_st", [1990, 1991])


def test_verify_era5_source_equivalence_unit(tmp_path: Path) -> None:
    """Verify verify_era5_source_equivalence unit contract."""
    mock_meta = {
        "name": "Mock Station",
        "latitude": 4.667867,
        "longitude": -74.101034,
        "cal_years": (1990, 1990),
        "era5_hourly_dir": tmp_path / "hourly",
        "era5_ts_dir": tmp_path / "ts",
    }
    mock_meta["era5_hourly_dir"].mkdir()
    mock_meta["era5_ts_dir"].mkdir()

    (mock_meta["era5_hourly_dir"] / "era5_2m_temperature_1990.nc").write_text(
        "dummy_h", encoding="utf-8"
    )
    (mock_meta["era5_ts_dir"] / "era5_2m_temperature_1990.nc").write_text(
        "dummy_ts", encoding="utf-8"
    )

    times = pd.date_range("1990-01-01", periods=24, freq="h")

    # Case 1: Equivalent within tolerance (<= 1e-3 K)
    da_h = xr.DataArray(
        np.full(24, 290.0),
        coords={"time": times},
        dims=["time"],
        attrs={"units": "K"},
    )
    da_ts_close = xr.DataArray(
        np.full(24, 290.0002),
        coords={"time": times},
        dims=["time"],
        attrs={"units": "K"},
    )

    mock_adapt_h = MagicMock()
    mock_adapt_h.extract_point.return_value = {"t2m": da_h}
    mock_adapt_h._resolve_variable_name.return_value = "t2m"

    mock_adapt_ts = MagicMock()
    mock_adapt_ts.extract_point.return_value = {"t2m": da_ts_close}
    mock_adapt_ts._resolve_variable_name.return_value = "t2m"

    with patch(
        "scripts.run_benchmark_lag_audit.ERA5Adapter",
        side_effect=[mock_adapt_h, mock_adapt_ts],
    ):
        res = verify_era5_source_equivalence("21205710", mock_meta)
        assert res["equivalent"] is True
        assert res["n_hours"] == 24
        assert abs(res["max_abs_diff_k"] - 0.0002) < 1e-7

    # Case 2: Exceeding tolerance (> 1e-3 K)
    da_ts_far = xr.DataArray(
        np.full(24, 290.005),
        coords={"time": times},
        dims=["time"],
        attrs={"units": "K"},
    )
    mock_adapt_ts_far = MagicMock()
    mock_adapt_ts_far.extract_point.return_value = {"t2m": da_ts_far}
    mock_adapt_ts_far._resolve_variable_name.return_value = "t2m"

    with patch(
        "scripts.run_benchmark_lag_audit.ERA5Adapter",
        side_effect=[mock_adapt_h, mock_adapt_ts_far],
    ):
        with pytest.raises(ValueError, match="max |difference| = 5.000000e-03 K"):
            verify_era5_source_equivalence("21205710", mock_meta)

    # Case 3: Timestamp mismatch
    times_mismatch = pd.date_range("1990-01-02", periods=24, freq="h")
    da_ts_shift = xr.DataArray(
        np.full(24, 290.0),
        coords={"time": times_mismatch},
        dims=["time"],
        attrs={"units": "K"},
    )
    mock_adapt_ts_shift = MagicMock()
    mock_adapt_ts_shift.extract_point.return_value = {"t2m": da_ts_shift}
    mock_adapt_ts_shift._resolve_variable_name.return_value = "t2m"

    with patch(
        "scripts.run_benchmark_lag_audit.ERA5Adapter",
        side_effect=[mock_adapt_h, mock_adapt_ts_shift],
    ):
        with pytest.raises(ValueError, match="Timestamps differ"):
            verify_era5_source_equivalence("21205710", mock_meta)

    # Case 4: Series containing NaN raises ValueError
    da_ts_nan = xr.DataArray(
        np.array([np.nan] + [290.0] * 23),
        coords={"time": times},
        dims=["time"],
        attrs={"units": "K"},
    )
    mock_adapt_ts_nan = MagicMock()
    mock_adapt_ts_nan.extract_point.return_value = {"t2m": da_ts_nan}
    mock_adapt_ts_nan._resolve_variable_name.return_value = "t2m"

    with patch(
        "scripts.run_benchmark_lag_audit.ERA5Adapter",
        side_effect=[mock_adapt_h, mock_adapt_ts_nan],
    ):
        with pytest.raises(
            ValueError,
            match="NaN values detected in ERA5 series during source equivalence",
        ):
            verify_era5_source_equivalence("21205710", mock_meta)

    # Case 5: Non-finite max difference raises ValueError
    da_ts_inf = xr.DataArray(
        np.array([np.inf] + [290.0] * 23),
        coords={"time": times},
        dims=["time"],
        attrs={"units": "K"},
    )
    mock_adapt_ts_inf = MagicMock()
    mock_adapt_ts_inf.extract_point.return_value = {"t2m": da_ts_inf}
    mock_adapt_ts_inf._resolve_variable_name.return_value = "t2m"

    with patch(
        "scripts.run_benchmark_lag_audit.ERA5Adapter",
        side_effect=[mock_adapt_h, mock_adapt_ts_inf],
    ):
        with pytest.raises(
            ValueError,
            match="Non-finite max absolute difference",
        ):
            verify_era5_source_equivalence("21205710", mock_meta)


def test_static_verification_rejects_cell_mismatch(tmp_path: Path) -> None:
    """Static verification must abort if extracted cell does not match Table J.2."""
    mock_meta = {
        "latitude": 4.667867,
        "longitude": -74.101034,
        "elevation_m": 2552.0,
        "expected_cell_lat": 4.75,
        "expected_cell_lon": -74.00,
        "expected_z_era5_m": 2753.4,
        "expected_delta_z_m": -201.4,
        "static_file": tmp_path / "mock_static.nc",
    }
    ds = xr.Dataset(
        data_vars={"z": (("latitude", "longitude"), np.array([[27000.0]]))},
        coords={"latitude": [5.00], "longitude": [-74.00]},
    )
    ds.to_netcdf(mock_meta["static_file"])

    with pytest.raises(ValueError, match="does not match Table J.2 expected cell"):
        verify_station_static_parameters("21205710", mock_meta)


def test_static_verification_rejects_elevation_mismatch(tmp_path: Path) -> None:
    """Static verification must abort if z_era5 differs by >0.1 m from Table J.2."""
    mock_meta = {
        "latitude": 4.667867,
        "longitude": -74.101034,
        "elevation_m": 2552.0,
        "expected_cell_lat": 4.75,
        "expected_cell_lon": -74.00,
        "expected_z_era5_m": 2753.4,
        "expected_delta_z_m": -201.4,
        "static_file": tmp_path / "mock_static.nc",
    }
    z_mismatch = 2755.0 * 9.80665
    ds = xr.Dataset(
        data_vars={"z": (("latitude", "longitude"), np.array([[z_mismatch]]))},
        coords={"latitude": [4.75], "longitude": [-74.00]},
    )
    ds.to_netcdf(mock_meta["static_file"])

    with pytest.raises(ValueError, match="differs from Table J.2 value"):
        verify_station_static_parameters("21205710", mock_meta)


def test_no_bias_or_dispersion_statistics_in_audit_output(tmp_path: Path) -> None:
    """Verify Section 8.2 boundary: output contains zero bias or dispersion metrics."""
    obs_tx, obs_tn, mod_tx_dict, mod_tn_dict = _make_synthetic_benchmark_audit_data(
        1990, 1992
    )
    mock_stations = _make_mock_benchmark_stations(tmp_path)

    with (
        patch.dict(
            "scripts.run_benchmark_lag_audit.BENCHMARK_STATIONS",
            mock_stations,
        ),
        patch(
            "scripts.run_benchmark_lag_audit.read_and_slice_benchmark_observations",
            side_effect=[obs_tx, obs_tn],
        ),
        patch(
            "scripts.run_benchmark_lag_audit.load_calibration_temperature_pair",
            return_value=(obs_tx, obs_tn, None, None),
        ),
        patch(
            "scripts.run_benchmark_lag_audit.inspect_observation_file_quality",
            return_value={
                "duplicate_dates_count": 0,
                "non_numeric_valor_count": 0,
            },
        ),
        patch(
            "scripts.run_benchmark_lag_audit.check_era5_completeness",
            return_value=([Path("nc1")], {"nc1": "hash1"}),
        ),
        patch(
            "scripts.run_benchmark_lag_audit.verify_era5_source_equivalence",
            return_value={
                "calibration_years": [1990, 1992],
                "equivalent": True,
                "era5_hourly_hashes": {},
                "era5_ts_hashes": {},
                "max_abs_diff_k": 1e-4,
                "n_hours": 1000,
                "station_id": "21205710",
            },
        ),
        patch(
            "scripts.run_benchmark_lag_audit.extract_and_aggregate_benchmark_era5",
            return_value=(mod_tx_dict, mod_tn_dict),
        ),
        patch(
            "scripts.run_benchmark_lag_audit.verify_station_static_parameters",
            return_value={
                "static_file": "local/era5_hourly/21205710/era5_static.nc",
                "static_sha256": "mock_sha",
                "verified": True,
            },
        ),
        patch(
            "scripts.run_benchmark_lag_audit.compute_file_sha256",
            return_value="fake_sha",
        ),
    ):
        res = audit_benchmark_station(
            station_id="21205710",
            mode="calibration",
            output_dir=tmp_path,
            replicas=10,
        )

        serialized = json.dumps(res)
        forbidden_terms = [
            "bias",
            "mae",
            "rmse",
            "variance",
            "standard_deviation",
            "dispersion",
            "mean_state",
            "percentile_loss",
        ]
        for term in forbidden_terms:
            assert term not in serialized.lower(), (
                f"Forbidden metric term '{term}' found in lag audit output payload!"
            )

        # Assert no path field in the payload is absolute or
        # contains a backslash or a drive letter
        def _assert_no_absolute_or_backslash_paths(
            obj: Any, key_path: str = ""
        ) -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    _assert_no_absolute_or_backslash_paths(
                        v, f"{key_path}.{k}" if key_path else k
                    )
            elif isinstance(obj, list):
                for item in obj:
                    _assert_no_absolute_or_backslash_paths(item, key_path)
            elif isinstance(obj, str):
                if any(term in key_path.lower() for term in ["path", "file", "dir"]):
                    assert "\\" not in obj, (
                        f"Path field '{key_path}' contains backslash: {obj}"
                    )
                    assert not (len(obj) > 1 and obj[1] == ":" and obj[0].isalpha()), (
                        f"Path field '{key_path}' contains drive letter: {obj}"
                    )
                    assert not Path(obj).is_absolute(), (
                        f"Path field '{key_path}' is absolute: {obj}"
                    )

        _assert_no_absolute_or_backslash_paths(res)


def test_validation_mode_resolves_relative_ledger_path_against_repo_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify relative ledger_path resolves against REPO_ROOT and leaves cwd clean."""
    mock_data_dir = tmp_path / "mock_files"
    mock_data_dir.mkdir()
    mock_stations = _make_mock_benchmark_stations(mock_data_dir)

    obs_tx, obs_tn, mod_tx_dict, mod_tn_dict = _make_synthetic_benchmark_audit_data(
        2005, 2006
    )

    ledger_received: List[Path] = []

    def mock_validate_access(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        lp = kwargs.get("ledger_path")
        if lp is not None:
            ledger_received.append(Path(lp))
        return {
            "timestamp": "2026-10-30T00:00:00Z",
            "commit_hash": "mock_commit",
            "purpose": kwargs.get("purpose", "lag_audit"),
            "station_id": kwargs.get("station_id", "21205710"),
            "variable": kwargs.get("variable", "temperature"),
            "seed": kwargs.get("seed"),
            "config": kwargs.get("config", {}),
        }

    work_dir = tmp_path / "isolated_cwd"
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)

    with (
        patch.dict("scripts.run_benchmark_lag_audit.BENCHMARK_STATIONS", mock_stations),
        patch(
            "scripts.run_benchmark_lag_audit.verify_station_static_parameters",
            return_value={"static_sha256": "mock_sha", "verified": True},
        ),
        patch(
            "scripts.run_benchmark_lag_audit.check_era5_completeness",
            return_value=([Path("mock.nc")], {"mock.nc": "hash"}),
        ),
        patch(
            "scripts.run_benchmark_lag_audit.verify_era5_source_equivalence",
            return_value={
                "calibration_years": [1990, 2004],
                "equivalent": True,
                "era5_hourly_hashes": {},
                "era5_ts_hashes": {},
                "max_abs_diff_k": 1e-4,
                "n_hours": 1000,
                "station_id": "21205710",
            },
        ),
        patch(
            "scripts.run_benchmark_lag_audit.compute_file_sha256",
            return_value="fake_sha",
        ),
        patch(
            "scripts.run_benchmark_lag_audit.validate_validation_access",
            side_effect=mock_validate_access,
        ),
        patch(
            "scripts.run_benchmark_lag_audit.read_and_slice_benchmark_observations",
            side_effect=[obs_tx, obs_tn],
        ),
        patch(
            "scripts.run_benchmark_lag_audit.extract_and_aggregate_benchmark_era5",
            return_value=(mod_tx_dict, mod_tn_dict),
        ),
        patch("subprocess.run", return_value=MagicMock(stdout="", returncode=0)),
    ):
        out_dir = tmp_path / "custom_out"
        audit_benchmark_station(
            station_id="21205710",
            mode="validation",
            output_dir=out_dir,
            confirm_written_authorization=True,
            replicas=10,
        )

    assert len(ledger_received) == 1
    rec = ledger_received[0]
    assert rec.is_absolute()
    assert rec == (REPO_ROOT / DEFAULT_LEDGER_PATH).resolve()
    # Confirm no file or directory is created under cwd (work_dir)
    assert list(work_dir.iterdir()) == []
