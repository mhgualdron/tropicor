"""Unit tests for Phase 3 Evaluator (Protocol v1.7, Appendix I).

Tests the pre-opening verification, single opening write-ahead ledger semantics,
validation data loading with strict out-of-sample masking, metrics and scorecards,
paired annual bootstrap progression, and deterministic JSON outputs.

CRITICAL RULE: Evaluates ONLY synthetic data in temporary test directories.
The real audit ledger (local/reports/validation_openings.jsonl) is verified to NEVER
exist before or after any test.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Generator, cast

import netCDF4 as nc
import numpy as np
import pandas as pd
import pytest

import tropicor.core.phase3_evaluator as phase3_evaluator
from tropicor.core.andean_registry import get_andean_station
from tropicor.core.phase3_evaluator import (
    ANDEAN_PHASE3_STATION_IDS,
    EXPECTED_INPUTS_REL_PATH,
    README_APPENDIX_G_REL_PATH,
    _check_authorization_tag,
    _check_git_working_tree_and_head,
    compute_daily_mae,
    compute_daily_pearson,
    compute_daily_rmse,
    compute_monthly_biases,
    compute_paired_annual_block_bootstrap_progression,
    compute_sha256,
    evaluate_progression_decision,
    evaluate_single_station,
    load_validation_temperature_pair,
    read_ledger_records,
    run_phase3_evaluation,
    verify_pre_opening,
    write_json_deterministic,
)
from tropicor.core.protocol import (
    DEFAULT_LEDGER_PATH,
    BootstrapConfidenceInterval,
)
from tropicor.downscale.temperature import (
    LEVEL3_TEMPERATURE_CONFIGS,
    Level1MonthlyAdditiveTemperature,
    Level2MeanStateDTRScalingTemperature,
    Level3QuantileMappingTemperature,
)

REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent

_PRODUCTION_REPLICAS: int = phase3_evaluator.PHASE3_BOOTSTRAP_REPLICAS

pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore::scipy.stats.ConstantInputWarning"),
]


# ==============================================================================
# Fixtures and Synthetic Generators
# ==============================================================================


@pytest.fixture(autouse=True)
def ensure_real_ledger_never_exists() -> Generator[None, None, None]:
    """Verify that the formal audit ledger never exists before or after tests."""
    real_ledger = REPO_ROOT / DEFAULT_LEDGER_PATH
    assert not real_ledger.exists(), (
        f"CRITICAL SAFETY VIOLATION: Real ledger exists before test: {real_ledger}"
    )
    yield
    assert not real_ledger.exists(), (
        f"CRITICAL SAFETY VIOLATION: Real ledger was created by test: {real_ledger}"
    )


@pytest.fixture(autouse=True)
def set_fast_bootstrap_replicas(monkeypatch: pytest.MonkeyPatch) -> None:
    """Monkeypatch PHASE3_BOOTSTRAP_REPLICAS to 20 for fast test suite execution."""
    monkeypatch.setattr(
        "tropicor.core.phase3_evaluator.PHASE3_BOOTSTRAP_REPLICAS",
        20,
    )



def create_synthetic_era5_nc(
    path: Path,
    year: int,
    lat: float = 6.25,
    lon: float = -75.50,
    base_k: float = 293.15,
) -> None:
    """Create a minimal synthetic hourly ERA5 NetCDF file for a single year."""
    start_dt = f"{year}-01-01 00:00:00"
    end_dt = f"{year}-12-31 23:00:00"
    times = pd.date_range(start_dt, end_dt, freq="h")
    n_hours = len(times)

    path.parent.mkdir(parents=True, exist_ok=True)
    with nc.Dataset(path, "w", format="NETCDF4_CLASSIC") as ds:
        ds.createDimension("time", n_hours)
        ds.createDimension("latitude", 1)
        ds.createDimension("longitude", 1)

        t_var = ds.createVariable("time", "f8", ("time",))
        t_var.units = f"hours since {year}-01-01 00:00:00"
        t_var.calendar = "standard"
        t_var[:] = np.arange(n_hours)

        lat_var = ds.createVariable("latitude", "f4", ("latitude",))
        lat_var[:] = [lat]

        lon_var = ds.createVariable("longitude", "f4", ("longitude",))
        lon_var[:] = [lon]

        t2m_var = ds.createVariable("t2m", "f4", ("time", "latitude", "longitude"))
        t2m_var.units = "K"

        # Diurnal cycle: 20 K daily range
        hours = times.hour.to_numpy()
        diurnal = 10.0 * np.sin((hours - 9) * 2 * np.pi / 24)
        t2m_data = (base_k + diurnal)[:, np.newaxis, np.newaxis].astype(np.float32)
        t2m_var[:] = t2m_data


def create_synthetic_dhime_csv(
    path: Path,
    station_id: str,
    start_date: str,
    end_date: str,
    base_val: float = 20.0,
    noise_amplitude: float = 0.5,
) -> None:
    """Create a synthetic DHIME daily observation CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dt_range = pd.date_range(start_date, end_date, freq="D")
    n_days = len(dt_range)
    # deterministic seasonal variation
    seasonal = 3.0 * np.sin(np.arange(n_days) * 2 * np.pi / 365)
    noise = noise_amplitude * np.cos(np.arange(n_days) * 2 * np.pi / 7)
    vals = base_val + seasonal + noise

    df = pd.DataFrame(
        {
            "CodigoEstacion": station_id,
            "Fecha": dt_range.strftime("%Y-%m-%d"),
            "Valor": np.round(vals, 2),
        }
    )
    df.to_csv(path, index=False)


@pytest.fixture(scope="session")
def base_synthetic_repo_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a baseline self-consistent synthetic git repository once per session."""
    repo = tmp_path_factory.mktemp("base_repo")

    # 1. Registry
    reg_path = repo / "tropicor" / "data" / "andean_station_registry.json"
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    stations_reg = {}
    for st in ANDEAN_PHASE3_STATION_IDS:
        andean_meta = get_andean_station(st)
        stations_reg[st] = {
            "codigo": st,
            "nombre": f"Station {st}",
            "latitud_cne": andean_meta.latitud_cne,
            "longitud_cne": andean_meta.longitud_cne,
            "altitud_cne": andean_meta.altitud_cne,
            "celda_era5": andean_meta.celda_era5,
            "cal_block": [2000, 2002],
            "val_block": [2003, 2005],
            "valid_years": [2000, 2001, 2002, 2003, 2004, 2005],
        }
    registry_data = {"version": "1.4", "stations": stations_reg}
    with open(reg_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(registry_data, f, indent=2)
    reg_sha256 = compute_sha256(reg_path)

    import tropicor.core.andean_registry as andean_reg

    orig_reg = andean_reg.DEFAULT_ANDEAN_REGISTRY_PATH
    try:
        andean_reg.DEFAULT_ANDEAN_REGISTRY_PATH = reg_path

        # 2. Lag audit report
        lag_path = (
            repo
            / "docs"
            / "science"
            / "appendix_g"
            / "lag_audit_andean_calibration.json"
        )
        lag_path.parent.mkdir(parents=True, exist_ok=True)
        lag_stations = {
            st: {
                "activated_lags": [],
                "lag_sensitivity_activated": False,
                "station_code": st,
            }
            for st in ANDEAN_PHASE3_STATION_IDS
        }
        lag_data = {"stations": lag_stations}
        with open(lag_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(lag_data, f, indent=2)
        lag_sha256 = compute_sha256(lag_path)

        # 3. Station observation CSVs and ERA5 NetCDFs
        ts_base = repo / "local" / "era5_ts"
        for st in ANDEAN_PHASE3_STATION_IDS:
            st_dir = ts_base / st
            st_dir.mkdir(parents=True, exist_ok=True)
            create_synthetic_dhime_csv(
                st_dir / f"TempMaxima_{st}.csv",
                st,
                "2000-01-01",
                "2005-12-31",
                base_val=25.0,
            )
            create_synthetic_dhime_csv(
                st_dir / f"TempMinima_{st}.csv",
                st,
                "2000-01-01",
                "2005-12-31",
                base_val=15.0,
            )
            andean_meta = get_andean_station(st)
            era5_lat = andean_meta.celda_era5["lat"]
            era5_lon = andean_meta.celda_era5["lon"]
            # NetCDFs for calibration (2000..2002) + border (2003)
            # + validation (2003..2005) + border (2006)
            for y in range(2000, 2007):
                create_synthetic_era5_nc(
                    st_dir / f"era5_2m_temperature_{y}.nc",
                    year=y,
                    lat=era5_lat,
                    lon=era5_lon,
                    base_k=293.15,
                )

        # 4. Calibration reports with pre-calculated raw biases
        cal_dir = repo / "docs" / "science" / "appendix_g" / "calibration"
        cal_dir.mkdir(parents=True, exist_ok=True)
        cal_hashes = {}
        cal_configs = {
            "24025040": 3,  # monthly, 100, linear
            "26135040": 2,  # monthly, 100, constant
            "27015070": 2,  # monthly, 100, constant
            "37015020": 3,  # monthly, 100, linear
            "52045010": 3,  # monthly, 100, linear
        }

        from tropicor.downscale.temperature import load_calibration_temperature_pair

        for st in ANDEAN_PHASE3_STATION_IDS:
            cfg_id = cal_configs[st]
            ref_cfg = LEVEL3_TEMPERATURE_CONFIGS[cfg_id]
            st_dir = ts_base / st
            obs_tmax, obs_tmin, mod_tmax, mod_tmin = load_calibration_temperature_pair(
                station_id=st,
                obs_tmax_path=st_dir / f"TempMaxima_{st}.csv",
                obs_tmin_path=st_dir / f"TempMinima_{st}.csv",
                era5_target=st_dir,
                window_start_utc=5,
                lag_days=0,
                min_hours=24,
            )
            rb_tmax = round(float((mod_tmax - obs_tmax).dropna().mean()), 4)
            rb_tmin = round(float((mod_tmin - obs_tmin).dropna().mean()), 4)

            c_report = {
                "station_id": st,
                "station_name": f"Station {st}",
                "raw_bias": {"tmax": rb_tmax, "tmin": rb_tmin},
                "level3": {
                    "winner_config_id": cfg_id,
                    "winner_grouping": ref_cfg["grouping"],
                    "winner_n_quantiles": ref_cfg["n_quantiles"],
                    "winner_extrapolation": ref_cfg["extrapolation"],
                },
            }
            cf_path = cal_dir / f"andes_calibration_{st}.json"
            with open(cf_path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(c_report, f, indent=2)
            cal_hashes[st] = compute_sha256(cf_path)
    finally:
        andean_reg.DEFAULT_ANDEAN_REGISTRY_PATH = orig_reg

    # 5. Expected inputs configuration
    exp_inputs_path = repo / EXPECTED_INPUTS_REL_PATH
    exp_inputs_path.parent.mkdir(parents=True, exist_ok=True)
    exp_inputs = {
        "registry": reg_sha256,
        "lag_audit": lag_sha256,
        "calibration": cal_hashes,
        "archived_paths": {
            "registry": "tropicor/data/andean_station_registry.json",
            "lag_audit": "docs/science/appendix_g/lag_audit_andean_calibration.json",
            "calibration": {
                st: f"docs/science/appendix_g/calibration/andes_calibration_{st}.json"
                for st in ANDEAN_PHASE3_STATION_IDS
            },
        },
    }
    with open(exp_inputs_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(exp_inputs, f, indent=2)

    # 6. README.md table in appendix_g
    readme_path = repo / README_APPENDIX_G_REL_PATH
    readme_lines = [
        "# Appendix G pre-validation archive",
        "",
        "## Files",
        "",
        "| File | SHA-256 |",
        "|---|---|",
        f"| `lag_audit_andean_calibration.json` | `{lag_sha256.upper()}` |",
    ]
    for st in ANDEAN_PHASE3_STATION_IDS:
        h_str = cal_hashes[st].upper()
        readme_lines.append(
            f"| `calibration/andes_calibration_{st}.json` | `{h_str}` |"
        )
    readme_path.write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    # 7. Initialize real git repository
    (repo / ".gitignore").write_text("local/reports/\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=str(repo),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=str(repo),
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "add", "."], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial test commit"],
        cwd=str(repo),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "tag", "-a", "phase3-authorization", "-m", "Authorized"],
        cwd=str(repo),
        check=True,
        capture_output=True,
    )

    return repo


@pytest.fixture
def synthetic_mock_repo(
    tmp_path: Path,
    base_synthetic_repo_template: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Build an isolated copy of the synthetic git repo environment for each test."""
    repo = tmp_path / "repo"
    shutil.copytree(base_synthetic_repo_template, repo)

    # Fast bootstrap replicas for testing
    monkeypatch.setattr("tropicor.core.phase3_evaluator.PHASE3_BOOTSTRAP_REPLICAS", 20)

    # Point registry path to this repo's registry
    monkeypatch.setattr(
        "tropicor.core.andean_registry.DEFAULT_ANDEAN_REGISTRY_PATH",
        repo / "tropicor" / "data" / "andean_station_registry.json",
    )

    # Ensure protocol.get_current_git_commit resolves the synthetic repo HEAD
    def get_repo_head() -> str:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout.strip()

    monkeypatch.setattr(
        "tropicor.core.phase3_evaluator.get_current_git_commit",
        get_repo_head,
    )
    monkeypatch.setattr(
        "tropicor.core.protocol.get_current_git_commit",
        get_repo_head,
    )

    return repo


# ==============================================================================
# C7: Real Git Tests (Working Tree & Authorization Tag)
# ==============================================================================


def test_check_git_working_tree_and_head_real_git(tmp_path: Path) -> None:
    """Test _check_git_working_tree_and_head across git states in real repo."""
    repo = tmp_path / "git_test_repo"
    repo.mkdir()

    # 1. Directory that is not a git repository
    is_clean, head_c, err = _check_git_working_tree_and_head(repo)
    assert not is_clean
    assert head_c == "unknown_or_uncommitted"
    assert err is not None

    # Initialize real repo
    subprocess.run(["git", "init"], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.name", "Tester"],
        cwd=str(repo),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "tester@test.com"],
        cwd=str(repo),
        check=True,
        capture_output=True,
    )
    (repo / "f.txt").write_text("hello", encoding="utf-8")
    subprocess.run(
        ["git", "add", "f.txt"], cwd=str(repo), check=True, capture_output=True
    )
    subprocess.run(
        ["git", "commit", "-m", "c1"], cwd=str(repo), check=True, capture_output=True
    )

    # 2. Clean repository
    is_clean, head_c, err = _check_git_working_tree_and_head(repo)
    assert is_clean
    assert len(head_c) == 40
    assert err is None

    # 3. Tracked modified file
    (repo / "f.txt").write_text("modified", encoding="utf-8")
    is_clean, head_c, err = _check_git_working_tree_and_head(repo)
    assert not is_clean
    assert len(head_c) == 40

    # Reset
    subprocess.run(
        ["git", "checkout", "f.txt"], cwd=str(repo), check=True, capture_output=True
    )

    # 4. Untracked file
    (repo / "untracked.txt").write_text("untracked", encoding="utf-8")
    is_clean, head_c, err = _check_git_working_tree_and_head(repo)
    assert not is_clean
    assert len(head_c) == 40


def test_check_authorization_tag_real_git(tmp_path: Path) -> None:
    """Test _check_authorization_tag in a real git repo."""
    repo = tmp_path / "tag_test_repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.name", "Tester"],
        cwd=str(repo),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "tester@test.com"],
        cwd=str(repo),
        check=True,
        capture_output=True,
    )
    (repo / "f.txt").write_text("c1", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "commit 1"],
        cwd=str(repo),
        check=True,
        capture_output=True,
    )
    proc_c1 = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=True,
    )
    commit1 = proc_c1.stdout.strip()

    # 1. Tag does not exist
    ok, target, err = _check_authorization_tag(repo, "phase3-authorization", commit1)
    assert not ok
    assert target == "missing"

    # 2. Lightweight tag (not annotated)
    subprocess.run(
        ["git", "tag", "phase3-authorization"],
        cwd=str(repo),
        check=True,
        capture_output=True,
    )
    ok, target, err = _check_authorization_tag(repo, "phase3-authorization", commit1)
    assert not ok
    assert target == "commit"

    # Delete lightweight tag
    subprocess.run(
        ["git", "tag", "-d", "phase3-authorization"],
        cwd=str(repo),
        check=True,
        capture_output=True,
    )

    # 3. Annotated tag on commit 1, but HEAD moved to commit 2
    subprocess.run(
        ["git", "tag", "-a", "phase3-authorization", "-m", "auth"],
        cwd=str(repo),
        check=True,
        capture_output=True,
    )
    (repo / "f.txt").write_text("c2", encoding="utf-8")
    subprocess.run(
        ["git", "commit", "-am", "commit 2"],
        cwd=str(repo),
        check=True,
        capture_output=True,
    )
    proc_c2 = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=True,
    )
    commit2 = proc_c2.stdout.strip()
    ok, target, err = _check_authorization_tag(repo, "phase3-authorization", commit2)
    assert not ok
    assert target == commit1  # targets commit 1, not HEAD commit 2

    # 4. Annotated tag targeting HEAD (valid)
    subprocess.run(
        ["git", "tag", "-d", "phase3-authorization"],
        cwd=str(repo),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "tag", "-a", "phase3-authorization", "-m", "auth on head"],
        cwd=str(repo),
        check=True,
        capture_output=True,
    )
    ok, target, err = _check_authorization_tag(repo, "phase3-authorization", commit2)
    assert ok
    assert target == commit2
    assert err is None


# ==============================================================================
# C1: Path and Replicas Locks Tests
# ==============================================================================


def test_c1_locks_non_canonical_paths_rejected(synthetic_mock_repo: Path) -> None:
    """Non-canonical ledger or reports paths must raise ValueError and write nothing."""
    repo = synthetic_mock_repo
    non_canonical_ledger = repo / "other_dir" / "ledger.jsonl"
    non_canonical_reports = repo / "other_reports"

    # Non-canonical ledger path
    with pytest.raises(ValueError, match="Non-canonical ledger path"):
        run_phase3_evaluation(
            repo_root=repo,
            dry_run=False,
            open_validation=True,
            ledger_path=non_canonical_ledger,
        )
    assert not non_canonical_ledger.exists()
    assert not (repo / DEFAULT_LEDGER_PATH).exists()

    # Non-canonical reports directory
    with pytest.raises(ValueError, match="Non-canonical reports directory"):
        run_phase3_evaluation(
            repo_root=repo,
            dry_run=False,
            open_validation=True,
            reports_dir=non_canonical_reports,
        )
    assert not non_canonical_reports.exists()


def test_c1_cli_formal_with_ledger_path_exits_nonzero(
    synthetic_mock_repo: Path,
) -> None:
    """CLI formal and reproduction modes reject path overrides with non-zero exit."""
    from scripts.run_andean_phase3 import main

    custom_ledger = str(synthetic_mock_repo / "custom_ledger.jsonl")
    exit_code = main(["--open-validation", "--ledger-path", custom_ledger])
    assert exit_code != 0
    assert not Path(custom_ledger).exists()
    assert not (synthetic_mock_repo / DEFAULT_LEDGER_PATH).exists()


# ==============================================================================
# C2: Opening Order Tests (Phase 0, Phase A, Phase B)
# ==============================================================================


def test_c2_station_3_prior_record_blocks_stations_1_2(
    synthetic_mock_repo: Path,
) -> None:
    """If station 3 has a prior record, formal mode aborts in Phase 0

    without writing.
    """
    repo = synthetic_mock_repo
    canonical_ledger = repo / DEFAULT_LEDGER_PATH
    canonical_ledger.parent.mkdir(parents=True, exist_ok=True)

    # Pre-populate ledger with opening for station 3 (27015070)
    proc_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=True,
    )
    head_sha = proc_head.stdout.strip()

    prior_rec = {
        "station_id": "27015070",
        "variable": "temperature",
        "commit_hash": head_sha,
        "timestamp_utc": "2026-10-24T00:00:00Z",
        "config": {"run_kind": "formal"},
    }
    canonical_ledger.write_text(json.dumps(prior_rec) + "\n", encoding="utf-8")

    with pytest.raises(
        PermissionError, match="Station 27015070 has already been opened"
    ):
        run_phase3_evaluation(
            repo_root=repo,
            run_kind="formal",
            open_validation=True,
            dry_run=False,
        )

    # Verify no records for stations 1 (24025040) or 2 (26135040) were written
    recs = read_ledger_records(canonical_ledger)
    assert len(recs) == 1
    assert recs[0]["station_id"] == "27015070"


def test_c2_eval_failure_on_station_2_leaves_5_records_and_reproduce_flags_interrupted(
    synthetic_mock_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Phase A writes all 5 records; evaluation failure on station 2 leaves 5 records;

    subsequent --reproduce succeeds and flags interrupted_formal_opening on station 2.
    """
    repo = synthetic_mock_repo
    canonical_ledger = repo / DEFAULT_LEDGER_PATH
    base_reports = repo / "local" / "reports" / "phase3"

    import tropicor.core.phase3_evaluator as p3e

    orig_load = p3e.load_validation_temperature_pair

    def fast_load_validation(*args: Any, **kwargs: Any) -> Any:
        tmax_obs, tmin_obs, tmax_mod, tmin_mod = orig_load(*args, **kwargs)
        return tmax_obs[:15], tmin_obs[:15], tmax_mod[:15], tmin_mod[:15]

    monkeypatch.setattr(p3e, "load_validation_temperature_pair", fast_load_validation)

    orig_eval = p3e.evaluate_single_station

    def mock_eval_failing_station_2(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        opening_rec = kwargs.get("opening_record", {})
        if (
            kwargs.get("station_id") == "26135040"
            and opening_rec.get("purpose") != "preflight_pipeline_smoke"
        ):
            raise RuntimeError("Simulated failure during evaluation of station 2")
        return orig_eval(*args, **kwargs)

    monkeypatch.setattr(p3e, "evaluate_single_station", mock_eval_failing_station_2)

    with pytest.raises(
        RuntimeError, match="Simulated failure during evaluation of station 2"
    ):
        run_phase3_evaluation(
            repo_root=repo,
            run_kind="formal",
            open_validation=True,
            dry_run=False,
        )

    # Verify that ALL 5 records were written in Phase A before the failure
    recs = read_ledger_records(canonical_ledger)
    assert len(recs) == 5
    written_ids = [r["station_id"] for r in recs]
    assert written_ids == list(ANDEAN_PHASE3_STATION_IDS)

    # Station 1 (24025040) scorecard was written; Station 2 (26135040) was NOT
    sc_st1 = base_reports / "24025040" / "phase3_scorecard_24025040.json"
    sc_st2 = base_reports / "26135040" / "phase3_scorecard_26135040.json"
    assert sc_st1.is_file()
    assert not sc_st2.is_file()

    # Now restore evaluator and execute reproduction run
    monkeypatch.setattr(p3e, "evaluate_single_station", orig_eval)

    reproduce_summary = run_phase3_evaluation(
        repo_root=repo,
        run_kind="reproduction",
        open_validation=True,
        dry_run=False,
    )
    assert reproduce_summary["run_kind"] == "reproduction"

    # Verify interrupted_formal_opening flag in reproduction scorecards
    rep_st1_path = (
        base_reports / "24025040" / "reproduction" / "phase3_scorecard_24025040.json"
    )
    rep_st2_path = (
        base_reports / "26135040" / "reproduction" / "phase3_scorecard_26135040.json"
    )
    assert rep_st1_path.is_file()
    assert rep_st2_path.is_file()

    with open(rep_st1_path, "r", encoding="utf-8") as f:
        rep_st1 = json.load(f)
    with open(rep_st2_path, "r", encoding="utf-8") as f:
        rep_st2 = json.load(f)

    assert rep_st1["interrupted_formal_opening"] is False
    assert rep_st2["interrupted_formal_opening"] is True


def test_c2_reproduce_missing_station_record_fails_before_writing(
    synthetic_mock_repo: Path,
) -> None:
    """Reproduction with missing station record fails in Phase 0 before writing."""
    repo = synthetic_mock_repo
    canonical_ledger = repo / DEFAULT_LEDGER_PATH
    canonical_ledger.parent.mkdir(parents=True, exist_ok=True)

    proc_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=True,
    )
    head_sha = proc_head.stdout.strip()

    # Write records for only first 4 stations (omit 52045010)
    lines = []
    for st in ANDEAN_PHASE3_STATION_IDS[:4]:
        lines.append(
            json.dumps(
                {
                    "station_id": st,
                    "variable": "temperature",
                    "commit_hash": head_sha,
                    "timestamp_utc": "2026-10-24T00:00:00Z",
                    "config": {"run_kind": "formal"},
                }
            )
        )
    canonical_ledger.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="station 52045010 requested, but no prior opening record exists",
    ):
        run_phase3_evaluation(
            repo_root=repo,
            run_kind="reproduction",
            open_validation=True,
            dry_run=False,
        )

    # Check no new records written
    recs = read_ledger_records(canonical_ledger)
    assert len(recs) == 4


def test_c2_corrupt_ledger_line_fails(synthetic_mock_repo: Path) -> None:
    """Corrupt ledger line raises ValueError."""
    repo = synthetic_mock_repo
    canonical_ledger = repo / DEFAULT_LEDGER_PATH
    canonical_ledger.parent.mkdir(parents=True, exist_ok=True)
    canonical_ledger.write_text("corrupted non json line\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Corrupt ledger line"):
        run_phase3_evaluation(
            repo_root=repo,
            run_kind="formal",
            open_validation=True,
            dry_run=False,
        )


# ==============================================================================
# C3: Preflight Smoke Tests
# ==============================================================================


def test_c3_preflight_smoke_executes_and_does_not_write_ledger(
    synthetic_mock_repo: Path,
) -> None:
    """Preflight executes smoke test, reports PASS, and does not write ledger."""
    repo = synthetic_mock_repo
    canonical_ledger = repo / DEFAULT_LEDGER_PATH

    res = run_phase3_evaluation(repo_root=repo, dry_run=True)
    assert res["status"] == "PASS"
    checks = res["pre_opening_checks"]
    assert checks["raw_bias_checks_passed"] is True
    assert checks["pipeline_smoke_passed"] is True
    for st in ANDEAN_PHASE3_STATION_IDS:
        assert checks["details"].get(f"smoke_{st}") == "PASS"

    assert not canonical_ledger.exists()


def test_c3_preflight_does_not_call_read_obs_block_with_validation_block(
    synthetic_mock_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preflight never reads observation files for the validation block."""
    repo = synthetic_mock_repo
    import tropicor.core.phase3_evaluator as p3e

    obs_block_calls = []
    orig_read = p3e._read_obs_block

    def spy_read_obs(
        path: Path,
        station_id: str,
        start_year: int,
        end_year: int,
        valid_years: Any = None,
    ) -> pd.Series:
        obs_block_calls.append((station_id, start_year, end_year))
        return orig_read(path, station_id, start_year, end_year, valid_years)

    monkeypatch.setattr(p3e, "_read_obs_block", spy_read_obs)

    res = run_phase3_evaluation(
        repo_root=repo, dry_run=True, station_ids=["24025040"]
    )
    assert res["status"] == "PASS"

    # All calls to _read_obs_block in preflight must be calibration block (2000, 2002)
    assert len(obs_block_calls) > 0
    for st, start_y, end_y in obs_block_calls:
        assert (start_y, end_y) == (2000, 2002), (
            f"Preflight read validation obs for station {st} ({start_y}-{end_y})"
        )


# ==============================================================================
# C4: Calibration Integrity Tests (Hashes and Raw Bias)
# ==============================================================================


def test_c4_modifying_cal_netcdf_changes_hash(synthetic_mock_repo: Path) -> None:
    """Altering 1 byte in a calibration NetCDF alters its input_file_hashes value."""
    repo = synthetic_mock_repo
    nc_cal = repo / "local" / "era5_ts" / "24025040" / "era5_2m_temperature_2000.nc"
    orig_file_sha = compute_sha256(nc_cal)

    report1, _ = verify_pre_opening(repo_root=repo, dry_run=True)
    orig_reported_sha = report1.input_file_hashes["era5_calibration"]["24025040"][
        "era5_2m_temperature_2000.nc"
    ]
    assert orig_reported_sha == orig_file_sha

    # Alter 1 byte in the calibration NetCDF file
    raw_bytes = bytearray(nc_cal.read_bytes())
    raw_bytes[-1] ^= 0x01
    nc_cal.write_bytes(raw_bytes)

    new_file_sha = compute_sha256(nc_cal)
    assert new_file_sha != orig_file_sha

    report2, _ = verify_pre_opening(repo_root=repo, dry_run=True)
    new_reported_sha = report2.input_file_hashes["era5_calibration"]["24025040"][
        "era5_2m_temperature_2000.nc"
    ]
    assert new_reported_sha != orig_reported_sha
    assert new_reported_sha == new_file_sha


def test_c4_altered_archived_raw_bias_raises_error(
    synthetic_mock_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Discrepancy between calculated and archived raw bias raises explicit error."""
    repo = synthetic_mock_repo
    proc_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=True,
    )
    head_sha = proc_head.stdout.strip()
    monkeypatch.setattr(
        phase3_evaluator,
        "_check_git_working_tree_and_head",
        lambda r: (True, head_sha, None),
    )

    cal_json = (
        repo
        / "docs"
        / "science"
        / "appendix_g"
        / "calibration"
        / "andes_calibration_24025040.json"
    )
    with open(cal_json, "r", encoding="utf-8") as f:
        cdata = json.load(f)
    cdata["raw_bias"]["tmax"] += 0.5
    with open(cal_json, "w", encoding="utf-8") as f:
        json.dump(cdata, f, indent=2)

    new_h = compute_sha256(cal_json)
    exp_file = repo / EXPECTED_INPUTS_REL_PATH
    with open(exp_file, "r+", encoding="utf-8") as f:
        exp = json.load(f)
        exp["calibration"]["24025040"] = new_h
        f.seek(0)
        json.dump(exp, f, indent=2)
        f.truncate()

    readme = repo / README_APPENDIX_G_REL_PATH
    content = readme.read_text(encoding="utf-8")
    content = re.sub(
        r"andes_calibration_24025040\.json` \| `[0-9a-fA-F]+`",
        f"andes_calibration_24025040.json` | `{new_h.upper()}`",
        content,
    )
    readme.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="Calibration raw bias mismatch"):
        verify_pre_opening(
            repo_root=repo, dry_run=False, station_ids=["24025040"]
        )


# ==============================================================================
# C5: Loader Strict Dictionary Equality Tests
# ==============================================================================


def test_c5_loader_rejects_modified_opening_record_field(
    synthetic_mock_repo: Path,
) -> None:
    """Loader rejects opening_record when any field does not match the ledger entry."""
    repo = synthetic_mock_repo
    canonical_ledger = repo / DEFAULT_LEDGER_PATH
    canonical_ledger.parent.mkdir(parents=True, exist_ok=True)

    proc_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=True,
    )
    head_sha = proc_head.stdout.strip()

    valid_rec = {
        "station_id": "24025040",
        "variable": "temperature",
        "commit_hash": head_sha,
        "timestamp_utc": "2026-10-24T00:00:00Z",
        "seed": 42,
        "purpose": "evaluation",
        "config": {"run_kind": "formal"},
    }
    canonical_ledger.write_text(json.dumps(valid_rec) + "\n", encoding="utf-8")

    st_dir = repo / "local" / "era5_ts" / "24025040"

    # Valid record succeeds
    load_validation_temperature_pair(
        station_id="24025040",
        obs_tmax_path=st_dir / "TempMaxima_24025040.csv",
        obs_tmin_path=st_dir / "TempMinima_24025040.csv",
        era5_target=st_dir,
        window_start_utc=5,
        lag_days=0,
        min_hours=24,
        opening_record=valid_rec,
        ledger_path=canonical_ledger,
    )

    # Altering seed from 42 to 99 must be rejected
    altered_rec = dict(valid_rec)
    altered_rec["seed"] = 99
    with pytest.raises(
        PermissionError,
        match="Exact opening record for station 24025040 not verified in ledger",
    ):
        load_validation_temperature_pair(
            station_id="24025040",
            obs_tmax_path=st_dir / "TempMaxima_24025040.csv",
            obs_tmin_path=st_dir / "TempMinima_24025040.csv",
            era5_target=st_dir,
            window_start_utc=5,
            lag_days=0,
            min_hours=24,
            opening_record=altered_rec,
            ledger_path=canonical_ledger,
        )


def test_c5_loader_rejects_wrong_station(synthetic_mock_repo: Path) -> None:
    """Loader rejects opening_record specifying another station."""
    repo = synthetic_mock_repo
    canonical_ledger = repo / DEFAULT_LEDGER_PATH
    canonical_ledger.parent.mkdir(parents=True, exist_ok=True)
    st_dir = repo / "local" / "era5_ts" / "24025040"

    rec = {
        "station_id": "26135040",
        "variable": "temperature",
        "commit_hash": "a" * 40,
        "timestamp_utc": "2026-10-24T00:00:00Z",
    }
    canonical_ledger.write_text(json.dumps(rec) + "\n", encoding="utf-8")

    with pytest.raises(
        PermissionError, match="does not match target station '24025040'"
    ):
        load_validation_temperature_pair(
            station_id="24025040",
            obs_tmax_path=st_dir / "TempMaxima_24025040.csv",
            obs_tmin_path=st_dir / "TempMinima_24025040.csv",
            era5_target=st_dir,
            window_start_utc=5,
            lag_days=0,
            min_hours=24,
            opening_record=rec,
            ledger_path=canonical_ledger,
        )


# ==============================================================================
# C7: Expanded Anti-Leakage Test
# ==============================================================================


def test_anti_leakage_transform_outputs_identical_with_contaminated_validation(
    synthetic_mock_repo: Path,
) -> None:
    """Fitted calibration models produce bit-for-bit identical outputs on a test series

    regardless of extreme contamination in the validation partition.
    """
    repo = synthetic_mock_repo
    st = "24025040"
    st_dir = repo / "local" / "era5_ts" / st
    tmax_csv = st_dir / f"TempMaxima_{st}.csv"
    tmin_csv = st_dir / f"TempMinima_{st}.csv"

    # Step 1: Fit models on clean data
    _, fitted_clean = verify_pre_opening(
        repo_root=repo, dry_run=True, station_ids=[st]
    )
    m1_clean = fitted_clean[st]["level1"]
    m2_clean = fitted_clean[st]["level2"]
    m3w_clean = fitted_clean[st]["level3_winner"]
    m3c_clean = fitted_clean[st]["level3_constant"]
    mod_tmax_cal_clean = fitted_clean[st]["mod_tmax_cal"].copy()
    mod_tmin_cal_clean = fitted_clean[st]["mod_tmin_cal"].copy()

    # Create fixed test series for transformation evaluation
    dt_idx = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    test_tmax = pd.Series(28.0 + 5.0 * np.sin(np.arange(len(dt_idx))), index=dt_idx)
    test_tmin = pd.Series(14.0 + 3.0 * np.cos(np.arange(len(dt_idx))), index=dt_idx)

    pred_clean_l1 = m1_clean.transform(test_tmax, test_tmin)
    pred_clean_l2 = m2_clean.transform(test_tmax, test_tmin)
    pred_clean_l3w = m3w_clean.transform(test_tmax, test_tmin)
    pred_clean_l3c = m3c_clean.transform(test_tmax, test_tmin)

    # Step 2: Severely contaminate the validation partition (years 2003..2005)
    for csv_file in (tmax_csv, tmin_csv):
        df = pd.read_csv(csv_file)
        dt = pd.to_datetime(df["Fecha"])
        val_mask = dt.dt.year >= 2003
        df.loc[val_mask, "Valor"] = 999.0  # extreme invalid observation value
        df.to_csv(csv_file, index=False)

    # Step 3: Re-fit calibration models on contaminated data
    _, fitted_contaminated = verify_pre_opening(
        repo_root=repo, dry_run=True, station_ids=[st]
    )
    m1_cont = fitted_contaminated[st]["level1"]
    m2_cont = fitted_contaminated[st]["level2"]
    m3w_cont = fitted_contaminated[st]["level3_winner"]
    m3c_cont = fitted_contaminated[st]["level3_constant"]
    mod_tmax_cal_cont = fitted_contaminated[st]["mod_tmax_cal"]
    mod_tmin_cal_cont = fitted_contaminated[st]["mod_tmin_cal"]

    # Verify calibration ERA5 is identical
    pd.testing.assert_series_equal(mod_tmax_cal_clean, mod_tmax_cal_cont)
    pd.testing.assert_series_equal(mod_tmin_cal_clean, mod_tmin_cal_cont)

    # Transform same fixed test series
    pred_cont_l1 = m1_cont.transform(test_tmax, test_tmin)
    pred_cont_l2 = m2_cont.transform(test_tmax, test_tmin)
    pred_cont_l3w = m3w_cont.transform(test_tmax, test_tmin)
    pred_cont_l3c = m3c_cont.transform(test_tmax, test_tmin)

    # Outputs must be bit-for-bit identical
    pd.testing.assert_series_equal(pred_clean_l1[0], pred_cont_l1[0])
    pd.testing.assert_series_equal(pred_clean_l1[1], pred_cont_l1[1])
    pd.testing.assert_series_equal(pred_clean_l2[0], pred_cont_l2[0])
    pd.testing.assert_series_equal(pred_clean_l2[1], pred_cont_l2[1])
    pd.testing.assert_series_equal(pred_clean_l3w[0], pred_cont_l3w[0])
    pd.testing.assert_series_equal(pred_clean_l3w[1], pred_cont_l3w[1])
    pd.testing.assert_series_equal(pred_clean_l3c[0], pred_cont_l3c[0])
    pd.testing.assert_series_equal(pred_clean_l3c[1], pred_cont_l3c[1])


# ==============================================================================
# C7: Order Spy Across All 5 Stations
# ==============================================================================


def test_order_spy_all_5_records_exist_before_first_validation_read(
    synthetic_mock_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that all 5 ledger opening records exist in the ledger before the

    FIRST validation observation read is executed.
    """
    repo = synthetic_mock_repo
    canonical_ledger = repo / DEFAULT_LEDGER_PATH
    import tropicor.core.phase3_evaluator as p3e

    first_load_executed = False
    records_at_first_load = -1

    orig_load = p3e.load_validation_temperature_pair

    def spy_load_validation(*args: Any, **kwargs: Any) -> Any:
        nonlocal first_load_executed, records_at_first_load
        if not first_load_executed:
            first_load_executed = True
            recs = read_ledger_records(canonical_ledger)
            records_at_first_load = len(recs)
        tmax_obs, tmin_obs, tmax_mod, tmin_mod = orig_load(*args, **kwargs)
        return tmax_obs[:10], tmin_obs[:10], tmax_mod[:10], tmin_mod[:10]

    monkeypatch.setattr(p3e, "load_validation_temperature_pair", spy_load_validation)

    summary = run_phase3_evaluation(
        repo_root=repo,
        run_kind="formal",
        open_validation=True,
        dry_run=False,
    )
    assert summary["run_kind"] == "formal"
    assert first_load_executed is True
    assert records_at_first_load == 5, (
        f"Expected all 5 ledger entries to exist before first validation read, "
        f"found {records_at_first_load}"
    )


# ==============================================================================
# CLI Argument Parsing and Execution Tests
# ==============================================================================


def test_cli_parse_args() -> None:
    """Verify CLI argument parsing behavior for valid and invalid combinations."""
    from scripts.run_andean_phase3 import parse_args

    # Valid preflight
    p_pref = parse_args(["--preflight"])
    assert p_pref.preflight is True
    assert p_pref.open_validation is False
    assert p_pref.reproduce is False

    # Valid formal without overrides
    p_formal = parse_args(["--open-validation"])
    assert p_formal.open_validation is True

    # Valid reproduce without overrides
    p_rep = parse_args(["--reproduce"])
    assert p_rep.reproduce is True

    # Forbidden overrides on formal mode
    with pytest.raises(SystemExit):
        parse_args(["--open-validation", "--ledger-path", "foo.jsonl"])

    with pytest.raises(SystemExit):
        parse_args(["--reproduce", "--reports-dir", "foo_dir"])

    with pytest.raises(SystemExit):
        parse_args(["--open-validation", "--era5-base-dir", "foo_era5"])

    with pytest.raises(SystemExit):
        parse_args(["--reproduce", "--repo-root", "foo_root"])


def test_cli_preflight_exit_code_1_with_broken_checks(
    synthetic_mock_repo: Path,
) -> None:
    """Preflight exits with return code 1 when Checks 3-6 fail."""
    from scripts.run_andean_phase3 import main

    repo = synthetic_mock_repo
    # Break Check 3 by altering registry hash in expected config
    exp_path = repo / EXPECTED_INPUTS_REL_PATH
    with open(exp_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["registry"] = "0" * 64
    with open(exp_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    exit_code = main(["--preflight", "--repo-root", str(repo)])
    assert exit_code == 1


# ==============================================================================
# Descriptive and Metric Calculation Tests
# ==============================================================================


def test_descriptive_metrics_hand_calculation() -> None:
    """Verify hand calculation of MAE, RMSE, Pearson r, and monthly biases."""
    idx = pd.date_range("2020-01-01", "2020-02-10", freq="D")
    obs = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    mod = obs + 2.0  # constant +2 bias

    assert compute_daily_mae(obs, mod) == pytest.approx(2.0, abs=1e-9)
    assert compute_daily_rmse(obs, mod) == pytest.approx(2.0, abs=1e-9)
    assert compute_daily_pearson(obs, mod) == pytest.approx(1.0, abs=1e-9)

    biases = compute_monthly_biases(obs, mod)
    assert biases["1"] == pytest.approx(2.0, abs=1e-9)
    assert biases["2"] == pytest.approx(2.0, abs=1e-9)
    assert biases["3"] is None


def test_paired_bootstrap_properties() -> None:
    """Verify paired annual bootstrap produces expected delta error distribution."""
    idx = pd.date_range("2010-01-01", "2019-12-31", freq="D")
    obs_tmax = pd.Series(25.0, index=idx)
    obs_tmin = pd.Series(15.0, index=idx)

    mod_base_tmax = obs_tmax + 3.0
    mod_base_tmin = obs_tmin - 3.0

    mod_cand_tmax = obs_tmax + 1.0
    mod_cand_tmin = obs_tmin - 1.0

    prog = compute_paired_annual_block_bootstrap_progression(
        tmax_obs=obs_tmax,
        tmin_obs=obs_tmin,
        tmax_mod_base=mod_base_tmax,
        tmin_mod_base=mod_base_tmin,
        tmax_mod_cand=mod_cand_tmax,
        tmin_mod_cand=mod_cand_tmin,
        n_replicas=50,
        seed=42,
    )

    for q in ("max_monthly_bias", "dtr_mean_bias", "delta_p5_tmin", "delta_p95_tmax"):
        ci = prog[q]
        assert not ci.insufficient_blocks
        assert ci.point_estimate > 0
        assert ci.ci_lower > 0


def test_progression_decision_table() -> None:
    """Verify dual promotion logic: primary improvement + no decisive regression."""
    # Case 1: Promoted
    cis_promoted: Dict[str, BootstrapConfidenceInterval] = {
        "dtr_mean_bias": BootstrapConfidenceInterval(1.0, 0.2, 1.8, 100, 42, 0, False),
        "delta_p5_tmin": BootstrapConfidenceInterval(0.5, -0.1, 1.1, 100, 42, 0, False),
        "delta_p95_tmax": BootstrapConfidenceInterval(
            0.3, -0.2, 0.8, 100, 42, 0, False
        ),
        "max_monthly_bias": BootstrapConfidenceInterval(
            0.8, 0.1, 1.5, 100, 42, 0, False
        ),
        "extreme_percentiles_bias": BootstrapConfidenceInterval(
            0.4, 0.0, 0.8, 100, 42, 0, False
        ),
    }
    dec1 = evaluate_progression_decision(cis_promoted)
    assert dec1["promoted"] is True
    assert "dtr_mean_bias" in dec1["primary_passes"]
    assert len(dec1["decisive_regressions"]) == 0

    # Case 2: Rejected due to significant decisive regression
    cis_regressed = dict(cis_promoted)
    cis_regressed["max_monthly_bias"] = BootstrapConfidenceInterval(
        -0.5, -1.0, -0.1, 100, 42, 0, False
    )
    dec2 = evaluate_progression_decision(cis_regressed)
    assert dec2["promoted"] is False
    assert "max_monthly_bias" in dec2["decisive_regressions"]


def test_deterministic_json_serialization(tmp_path: Path) -> None:
    """Verify atomic write, sorted keys, newline LF, and NaN conversion to None."""
    out_file = tmp_path / "deterministic.json"
    data = {
        "z_key": float("nan"),
        "a_key": 42,
        "nested": {"b": True, "a": [1, np.nan, float("inf")]},
    }
    write_json_deterministic(out_file, data)
    assert out_file.is_file()

    with open(out_file, "rb") as f:
        raw_bytes = f.read()

    assert b"\r\n" not in raw_bytes
    assert raw_bytes.endswith(b"\n")

    loaded = json.loads(raw_bytes.decode("utf-8"))
    assert loaded["z_key"] is None
    assert loaded["nested"]["a"] == [1, None, None]
    keys = list(loaded.keys())
    assert keys == ["a_key", "nested", "z_key"]


# ==============================================================================
# F5b: Bootstrap Replicas Production & Ledger Verification
# ==============================================================================


def test_default_production_bootstrap_replicas_is_1000() -> None:
    """Verify that default production bootstrap replicas is exactly 1000 upon import."""
    assert _PRODUCTION_REPLICAS == 1000


def test_formal_opening_ledger_records_1000_bootstrap_replicas(
    synthetic_mock_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify formal ledger record stores bootstrap.replicas == 1000

    when fixture is undone.
    """
    monkeypatch.setattr(
        "tropicor.core.phase3_evaluator.PHASE3_BOOTSTRAP_REPLICAS",
        _PRODUCTION_REPLICAS,
    )

    def abort_after_phase_a(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            "Aborting immediately after Phase A to avoid 1000 replicas run"
        )

    monkeypatch.setattr(
        "tropicor.core.phase3_evaluator.load_validation_temperature_pair",
        abort_after_phase_a,
    )

    with pytest.raises(
        RuntimeError, match="Aborting immediately after Phase A"
    ):
        run_phase3_evaluation(
            repo_root=synthetic_mock_repo,
            run_kind="formal",
            open_validation=True,
            dry_run=False,
        )

    records = read_ledger_records(synthetic_mock_repo / DEFAULT_LEDGER_PATH)
    assert len(records) == 5
    for rec in records:
        assert rec["config"]["bootstrap"]["replicas"] == 1000


# ==============================================================================
# F4c: Pipeline Smoke Abort in Formal Mode
# ==============================================================================


def test_formal_mode_aborts_without_creating_ledger_if_smoke_fails(
    synthetic_mock_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Forced failure in pipeline smoke test during pre-opening aborts formal mode

    without creating or touching the audit ledger.
    """
    repo = synthetic_mock_repo
    canonical_ledger = repo / DEFAULT_LEDGER_PATH
    assert not canonical_ledger.exists()

    orig_eval = phase3_evaluator.evaluate_single_station

    def failing_eval(*args: Any, **kwargs: Any) -> Any:
        opening_rec = kwargs.get("opening_record", {})
        if opening_rec.get("purpose") == "preflight_pipeline_smoke":
            raise RuntimeError("Forced pipeline smoke failure")
        return orig_eval(*args, **kwargs)

    monkeypatch.setattr(phase3_evaluator, "evaluate_single_station", failing_eval)

    with pytest.raises(RuntimeError, match="Forced pipeline smoke failure"):
        run_phase3_evaluation(
            repo_root=repo,
            run_kind="formal",
            open_validation=True,
            dry_run=False,
        )

    assert not canonical_ledger.exists(), (
        "Ledger must not exist after aborted formal run."
    )


# ==============================================================================
# F3b: Lag Audit Hash Discrepancy in README
# ==============================================================================


def test_pre_opening_aborts_if_readme_lag_audit_hash_mismatches(
    synthetic_mock_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-opening aborts with ValueError if lag audit SHA-256 in README is altered."""
    repo = synthetic_mock_repo
    proc_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=True,
    )
    head_sha = proc_head.stdout.strip()
    monkeypatch.setattr(
        phase3_evaluator,
        "_check_git_working_tree_and_head",
        lambda r: (True, head_sha, None),
    )

    readme = repo / README_APPENDIX_G_REL_PATH
    content = readme.read_text(encoding="utf-8")
    corrupt_content = re.sub(
        r"lag_audit_andean_calibration\.json` \| `[0-9a-fA-F]+`",
        "lag_audit_andean_calibration.json` | "
        "`0000000000000000000000000000000000000000000000000000000000000000`",
        content,
    )
    readme.write_text(corrupt_content, encoding="utf-8")

    with pytest.raises(ValueError, match="Archive README hashes do not match"):
        verify_pre_opening(repo_root=repo, dry_run=False)


# ==============================================================================
# F1: 13 Pre-Opening Abort Tests
# ==============================================================================


def test_r1_abort_hash_mismatch_registry(
    synthetic_mock_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-opening aborts if registry SHA-256 in expected inputs is altered."""
    repo = synthetic_mock_repo
    proc_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=True,
    )
    head_sha = proc_head.stdout.strip()
    monkeypatch.setattr(
        phase3_evaluator,
        "_check_git_working_tree_and_head",
        lambda r: (True, head_sha, None),
    )

    exp_file = repo / EXPECTED_INPUTS_REL_PATH
    with open(exp_file, "r+", encoding="utf-8") as f:
        data = json.load(f)
        data["registry"] = "0" * 64
        f.seek(0)
        json.dump(data, f, indent=2)
        f.truncate()

    with pytest.raises(
        ValueError, match="SHA-256 hashes of inputs do not match expected config"
    ):
        verify_pre_opening(repo_root=repo, dry_run=False)


def test_r1_abort_hash_mismatch_lag(
    synthetic_mock_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-opening aborts if lag audit SHA-256 in expected inputs is altered."""
    repo = synthetic_mock_repo
    proc_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=True,
    )
    head_sha = proc_head.stdout.strip()
    monkeypatch.setattr(
        phase3_evaluator,
        "_check_git_working_tree_and_head",
        lambda r: (True, head_sha, None),
    )

    exp_file = repo / EXPECTED_INPUTS_REL_PATH
    with open(exp_file, "r+", encoding="utf-8") as f:
        data = json.load(f)
        data["lag_audit"] = "0" * 64
        f.seek(0)
        json.dump(data, f, indent=2)
        f.truncate()

    with pytest.raises(
        ValueError, match="SHA-256 hashes of inputs do not match expected config"
    ):
        verify_pre_opening(repo_root=repo, dry_run=False)


def test_r1_abort_hash_mismatch_calibration_report(
    synthetic_mock_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-opening aborts if calibration report SHA-256 in expected inputs
    is altered.
    """
    repo = synthetic_mock_repo
    proc_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=True,
    )
    head_sha = proc_head.stdout.strip()
    monkeypatch.setattr(
        phase3_evaluator,
        "_check_git_working_tree_and_head",
        lambda r: (True, head_sha, None),
    )

    exp_file = repo / EXPECTED_INPUTS_REL_PATH
    with open(exp_file, "r+", encoding="utf-8") as f:
        data = json.load(f)
        data["calibration"]["24025040"] = "0" * 64
        f.seek(0)
        json.dump(data, f, indent=2)
        f.truncate()

    with pytest.raises(
        ValueError, match="SHA-256 hashes of inputs do not match expected config"
    ):
        verify_pre_opening(repo_root=repo, dry_run=False)


def test_r1_abort_readme_mismatch_cal(
    synthetic_mock_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-opening aborts if calibration report row in README table is altered."""
    repo = synthetic_mock_repo
    proc_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=True,
    )
    head_sha = proc_head.stdout.strip()
    monkeypatch.setattr(
        phase3_evaluator,
        "_check_git_working_tree_and_head",
        lambda r: (True, head_sha, None),
    )

    readme = repo / README_APPENDIX_G_REL_PATH
    content = readme.read_text(encoding="utf-8")
    corrupt_content = re.sub(
        r"andes_calibration_24025040\.json` \| `[0-9a-fA-F]+`",
        "andes_calibration_24025040.json` | "
        "`0000000000000000000000000000000000000000000000000000000000000000`",
        content,
    )
    readme.write_text(corrupt_content, encoding="utf-8")

    with pytest.raises(ValueError, match="Archive README hashes do not match"):
        verify_pre_opening(repo_root=repo, dry_run=False)


def test_r1_abort_activated_lag(
    synthetic_mock_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-opening aborts if any station has activated lags."""
    repo = synthetic_mock_repo
    proc_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=True,
    )
    head_sha = proc_head.stdout.strip()
    monkeypatch.setattr(
        phase3_evaluator,
        "_check_git_working_tree_and_head",
        lambda r: (True, head_sha, None),
    )

    lag_file = (
        repo
        / "docs"
        / "science"
        / "appendix_g"
        / "lag_audit_andean_calibration.json"
    )
    with open(lag_file, "r+", encoding="utf-8") as f:
        d = json.load(f)
        d["stations"]["24025040"]["activated_lags"] = [1]
        f.seek(0)
        json.dump(d, f, indent=2)
        f.truncate()

    new_h = compute_sha256(lag_file)
    exp_file = repo / EXPECTED_INPUTS_REL_PATH
    with open(exp_file, "r+", encoding="utf-8") as f:
        exp = json.load(f)
        exp["lag_audit"] = new_h
        f.seek(0)
        json.dump(exp, f, indent=2)
        f.truncate()

    readme = repo / README_APPENDIX_G_REL_PATH
    content = readme.read_text(encoding="utf-8")
    content = re.sub(
        r"lag_audit_andean_calibration\.json` \| `[0-9a-fA-F]+`",
        f"lag_audit_andean_calibration.json` | `{new_h.upper()}`",
        content,
    )
    readme.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="lag sensitivity not implemented"):
        verify_pre_opening(repo_root=repo, dry_run=False)


def test_r1_abort_winner_config_mismatch(
    synthetic_mock_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-opening aborts if calibration report winner config is invalid."""
    repo = synthetic_mock_repo
    proc_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=True,
    )
    head_sha = proc_head.stdout.strip()
    monkeypatch.setattr(
        phase3_evaluator,
        "_check_git_working_tree_and_head",
        lambda r: (True, head_sha, None),
    )

    cal_file = (
        repo
        / "docs"
        / "science"
        / "appendix_g"
        / "calibration"
        / "andes_calibration_24025040.json"
    )
    with open(cal_file, "r+", encoding="utf-8") as f:
        d = json.load(f)
        d["level3"]["winner_n_quantiles"] = 999
        f.seek(0)
        json.dump(d, f, indent=2)
        f.truncate()

    new_h = compute_sha256(cal_file)
    exp_file = repo / EXPECTED_INPUTS_REL_PATH
    with open(exp_file, "r+", encoding="utf-8") as f:
        exp = json.load(f)
        exp["calibration"]["24025040"] = new_h
        f.seek(0)
        json.dump(exp, f, indent=2)
        f.truncate()

    readme = repo / README_APPENDIX_G_REL_PATH
    content = readme.read_text(encoding="utf-8")
    content = re.sub(
        r"andes_calibration_24025040\.json` \| `[0-9a-fA-F]+`",
        f"andes_calibration_24025040.json` | `{new_h.upper()}`",
        content,
    )
    readme.write_text(content, encoding="utf-8")

    with pytest.raises(
        ValueError, match="winner config does not match LEVEL3_TEMPERATURE_CONFIGS"
    ):
        verify_pre_opening(repo_root=repo, dry_run=False)


def test_r1_abort_missing_border_era5_file(
    synthetic_mock_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-opening aborts if boundary ERA5 NetCDF (val_end + 1) is missing."""
    repo = synthetic_mock_repo
    proc_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=True,
    )
    head_sha = proc_head.stdout.strip()
    monkeypatch.setattr(
        phase3_evaluator,
        "_check_git_working_tree_and_head",
        lambda r: (True, head_sha, None),
    )

    border_file = (
        repo / "local" / "era5_ts" / "24025040" / "era5_2m_temperature_2006.nc"
    )
    border_file.unlink()

    with pytest.raises(FileNotFoundError, match="validation files missing"):
        verify_pre_opening(repo_root=repo, dry_run=False)


def test_r1_abort_missing_cal_era5_file(
    synthetic_mock_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-opening aborts if calibration ERA5 NetCDF (cal_start) is missing."""
    repo = synthetic_mock_repo
    proc_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=True,
    )
    head_sha = proc_head.stdout.strip()
    monkeypatch.setattr(
        phase3_evaluator,
        "_check_git_working_tree_and_head",
        lambda r: (True, head_sha, None),
    )

    cal_nc = (
        repo / "local" / "era5_ts" / "24025040" / "era5_2m_temperature_2000.nc"
    )
    cal_nc.unlink()

    with pytest.raises(FileNotFoundError, match="validation files missing"):
        verify_pre_opening(repo_root=repo, dry_run=False)


def test_r1_abort_git_fails_unknown_uncommitted(
    synthetic_mock_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-opening aborts if git check returns unknown_or_uncommitted error."""
    repo = synthetic_mock_repo
    monkeypatch.setattr(
        phase3_evaluator,
        "_check_git_working_tree_and_head",
        lambda r: (True, "unknown_or_uncommitted", "git command failed"),
    )

    with pytest.raises(RuntimeError, match="git error"):
        verify_pre_opening(repo_root=repo, dry_run=False)


def test_r1_abort_dirty_tree(synthetic_mock_repo: Path) -> None:
    """Pre-opening aborts if git working tree contains uncommitted changes."""
    repo = synthetic_mock_repo
    (repo / "uncommitted_file.txt").write_text("dirty content", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Working tree is dirty"):
        verify_pre_opening(repo_root=repo, dry_run=False)


def test_r1_abort_missing_tag(synthetic_mock_repo: Path) -> None:
    """Pre-opening aborts if authorization tag phase3-authorization is missing."""
    repo = synthetic_mock_repo
    subprocess.run(
        ["git", "tag", "-d", "phase3-authorization"],
        cwd=str(repo),
        check=True,
        capture_output=True,
    )

    with pytest.raises(RuntimeError, match="phase3-authorization"):
        verify_pre_opening(repo_root=repo, dry_run=False)


def test_r1_abort_lightweight_tag(synthetic_mock_repo: Path) -> None:
    """Pre-opening aborts if authorization tag is a lightweight (unannotated) tag."""
    repo = synthetic_mock_repo
    subprocess.run(
        ["git", "tag", "-d", "phase3-authorization"],
        cwd=str(repo),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "tag", "phase3-authorization"],
        cwd=str(repo),
        check=True,
        capture_output=True,
    )

    with pytest.raises(RuntimeError, match="phase3-authorization"):
        verify_pre_opening(repo_root=repo, dry_run=False)


def test_r1_abort_tag_target_differs_from_head(synthetic_mock_repo: Path) -> None:
    """Pre-opening aborts if authorization tag targets a commit different from HEAD."""
    repo = synthetic_mock_repo
    (repo / "new_file.txt").write_text("advance head", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "advance head"],
        cwd=str(repo),
        check=True,
        capture_output=True,
    )

    with pytest.raises(RuntimeError, match="phase3-authorization"):
        verify_pre_opening(repo_root=repo, dry_run=False)


# ==============================================================================
# F1: Station 5 Atomicity & Lifecycle
# ==============================================================================


def test_failure_on_station_5_leaves_zero_records(
    synthetic_mock_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failure during pre-opening on station 5 leaves zero records for stations 1-4."""
    repo = synthetic_mock_repo
    proc_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=True,
    )
    head_sha = proc_head.stdout.strip()
    monkeypatch.setattr(
        phase3_evaluator,
        "_check_git_working_tree_and_head",
        lambda r: (True, head_sha, None),
    )

    canonical_ledger = repo / DEFAULT_LEDGER_PATH
    st5_file = (
        repo / "local" / "era5_ts" / "52045010" / "TempMaxima_52045010.csv"
    )
    st5_file.unlink()

    with pytest.raises(FileNotFoundError, match="validation files missing"):
        run_phase3_evaluation(
            repo_root=repo,
            run_kind="formal",
            open_validation=True,
            dry_run=False,
        )

    assert not canonical_ledger.exists()


def test_formal_opening_writes_all_g10_fields(
    synthetic_mock_repo: Path,
) -> None:
    """Formal opening writes all protocol G.10 required fields to ledger."""
    repo = synthetic_mock_repo
    canonical_ledger = repo / DEFAULT_LEDGER_PATH

    summary = run_phase3_evaluation(
        repo_root=repo,
        run_kind="formal",
        open_validation=True,
        dry_run=False,
    )

    assert canonical_ledger.is_file()
    lines = (
        canonical_ledger.read_text(encoding="utf-8").strip().splitlines()
    )
    assert len(lines) == 5

    proc_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=True,
    )
    head_hex = proc_head.stdout.strip()

    for line in lines:
        rec = json.loads(line)
        assert rec["commit_hash"] == head_hex
        assert rec["variable"] == "temperature"
        assert rec["seed"] == 42
        assert "timestamp" in rec or "timestamp_utc" in rec
        assert "purpose" in rec
        cfg = rec["config"]
        assert cfg["run_kind"] == "formal"
        assert cfg["evaluator_commit"] == head_hex
        assert "calibration_report_sha256" in cfg
        assert cfg["methods"] == [
            "level0",
            "level1",
            "level2",
            "level3_winner",
            "level3_constant",
        ]
        assert cfg["bootstrap"]["seed"] == 42
        assert cfg["bootstrap"]["block"] == "annual"

    assert summary["run_kind"] == "formal"


def test_second_formal_opening_fails(
    synthetic_mock_repo: Path,
) -> None:
    """Second formal opening on an already opened station raises PermissionError."""
    repo = synthetic_mock_repo

    # First formal opening succeeds
    run_phase3_evaluation(
        repo_root=repo,
        run_kind="formal",
        open_validation=True,
        dry_run=False,
    )

    # Second formal opening must fail
    with pytest.raises(PermissionError, match="already been opened in ledger"):
        run_phase3_evaluation(
            repo_root=repo,
            run_kind="formal",
            open_validation=True,
            dry_run=False,
        )


def test_reproduction_requires_same_commit(
    synthetic_mock_repo: Path,
) -> None:
    """Reproduction requires identical git commit to prior formal opening."""
    repo = synthetic_mock_repo

    run_phase3_evaluation(
        repo_root=repo,
        run_kind="formal",
        open_validation=True,
        dry_run=False,
    )

    # Advance HEAD with a new commit
    (repo / "new_change.txt").write_text("another commit", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "advance head"],
        cwd=str(repo),
        check=True,
        capture_output=True,
    )

    with pytest.raises(ValueError, match="requires git commit"):
        run_phase3_evaluation(
            repo_root=repo,
            run_kind="reproduction",
            open_validation=True,
            dry_run=False,
        )


def test_loader_rejects_without_record_or_wrong_station(
    synthetic_mock_repo: Path,
) -> None:
    """Validation loader rejects if opening record is missing or wrong station."""
    repo = synthetic_mock_repo
    ledger_file = repo / DEFAULT_LEDGER_PATH
    st_files = repo / "local" / "era5_ts" / "24025040"

    fake_rec = {
        "station_id": "24025040",
        "commit_hash": "a" * 40,
        "timestamp_utc": "2026-10-24T00:00:00Z",
        "variable": "temperature",
    }
    with pytest.raises(PermissionError):
        load_validation_temperature_pair(
            station_id="24025040",
            obs_tmax_path=st_files / "TempMaxima_24025040.csv",
            obs_tmin_path=st_files / "TempMinima_24025040.csv",
            era5_target=st_files,
            window_start_utc=5,
            lag_days=0,
            min_hours=24,
            opening_record=fake_rec,
            ledger_path=ledger_file,
            registry_path=repo / "tropicor" / "data" / "andean_station_registry.json",
        )


def test_loader_excludes_calibration_and_masks_invalid_years(
    synthetic_mock_repo: Path,
) -> None:
    """Validation loader strictly returns validation block and masks excluded years."""
    repo = synthetic_mock_repo
    ledger_file = repo / DEFAULT_LEDGER_PATH

    # Execute formal evaluation to populate valid ledger records
    _ = run_phase3_evaluation(
        repo_root=repo,
        run_kind="formal",
        open_validation=True,
        dry_run=False,
    )
    records = read_ledger_records(ledger_file)
    rec_2402 = [r for r in records if r["station_id"] == "24025040"][0]

    # Exclude year 2004 from valid_years in registry
    reg_path = repo / "tropicor" / "data" / "andean_station_registry.json"
    with open(reg_path, "r+", encoding="utf-8") as f:
        d = json.load(f)
        d["stations"]["24025040"]["valid_years"] = [2000, 2001, 2002, 2003, 2005]
        f.seek(0)
        json.dump(d, f, indent=2)
        f.truncate()

    st_files = repo / "local" / "era5_ts" / "24025040"
    obs_max, obs_min, mod_max, mod_min = load_validation_temperature_pair(
        station_id="24025040",
        obs_tmax_path=st_files / "TempMaxima_24025040.csv",
        obs_tmin_path=st_files / "TempMinima_24025040.csv",
        era5_target=st_files,
        window_start_utc=5,
        lag_days=0,
        min_hours=24,
        opening_record=rec_2402,
        ledger_path=ledger_file,
        registry_path=reg_path,
    )

    dt_index = cast(pd.DatetimeIndex, obs_max.index)
    assert dt_index.min().year == 2003
    assert dt_index.max().year == 2005

    # Year 2004 must be masked to NaN per Protocol G.5.2
    mask_2004 = dt_index.year == 2004
    assert obs_max[mask_2004].isna().all()
    assert obs_min[mask_2004].isna().all()

    # Years 2003 and 2005 must be valid
    mask_2003 = dt_index.year == 2003
    assert obs_max[mask_2003].notna().all()
    assert obs_min[mask_2003].notna().all()


def test_synthetic_monthly_bias_scenario_l0_fails_l1_passes() -> None:
    """Synthetic scenario: L0 fails max_monthly_bias (> 0.5 C), L1 corrects it."""
    idx = pd.date_range("2000-01-01", "2009-12-31", freq="D")
    obs_tmax = pd.Series(25.0, index=idx)
    obs_tmin = pd.Series(15.0, index=idx)
    # Modeled series with +3 C monthly bias
    mod_tmax = obs_tmax + 3.0
    mod_tmin = obs_tmin + 3.0

    # L0
    m_biases_l0 = compute_monthly_biases(obs_tmax, mod_tmax)
    assert all(b == 3.0 for b in m_biases_l0.values())
    max_b_l0 = max(abs(b) for b in m_biases_l0.values() if b is not None)
    assert max_b_l0 > 0.5

    # L1
    l1 = Level1MonthlyAdditiveTemperature()
    l1.fit(obs_tmax, obs_tmin, mod_tmax, mod_tmin)
    pred_tmax, pred_tmin = l1.transform(mod_tmax, mod_tmin)

    m_biases_l1 = compute_monthly_biases(obs_tmax, pred_tmax)
    assert all(abs(b) < 1e-9 for b in m_biases_l1.values() if b is not None)
    max_b_l1 = max(abs(b) for b in m_biases_l1.values() if b is not None)
    assert max_b_l1 <= 0.5


def test_l3_constant_identical_to_winner_when_constant() -> None:
    """When winner uses constant extrapolation, Level 3 Constant matches winner."""
    idx = pd.date_range("2000-01-01", "2005-12-31", freq="D")
    obs_max = pd.Series(25.0 + np.sin(np.arange(len(idx))), index=idx)
    obs_min = pd.Series(15.0 + np.sin(np.arange(len(idx))), index=idx)
    mod_max = obs_max + 1.0
    mod_min = obs_min - 1.0

    w_model = Level3QuantileMappingTemperature(
        grouping="monthly", n_quantiles=50, extrapolation="constant"
    )
    w_model.fit(obs_max, obs_min, mod_max, mod_min)

    l1_model = Level1MonthlyAdditiveTemperature()
    l1_model.fit(obs_max, obs_min, mod_max, mod_min)

    l2_model = Level2MeanStateDTRScalingTemperature()
    l2_model.fit(obs_max, obs_min, mod_max, mod_min)

    fitted_info = {
        "level1": l1_model,
        "level2": l2_model,
        "level3_winner": w_model,
        "level3_constant": w_model,
        "identical_to_winner": True,
        "winner_config": {
            "config_id": 0,
            "grouping": "monthly",
            "n_quantiles": 50,
            "extrapolation": "constant",
        },
    }

    opening_rec = {
        "station_id": "99999",
        "commit_hash": "a" * 40,
        "variable": "temperature",
        "timestamp_utc": "2026-10-24T00:00:00Z",
    }
    res = evaluate_single_station(
        station_id="99999",
        fitted_info=fitted_info,
        obs_tmax_val=obs_max,
        obs_tmin_val=obs_min,
        mod_tmax_val=mod_max,
        mod_tmin_val=mod_min,
        opening_record=opening_rec,
        evaluator_commit="a" * 40,
        tag_target="a" * 40,
        run_kind="formal",
        interrupted_formal_opening=False,
        input_hashes={},
        bootstrap_replicas=10,
    )
    assert res["identical_to_winner"] is True
    sc_const = dict(res["scorecards"]["level3_constant"])
    sc_winner = dict(res["scorecards"]["level3_winner"])
    assert sc_const.pop("level") == "level3_constant"
    assert sc_winner.pop("level") == "level3_winner"
    assert sc_const == sc_winner


def test_ledger_path_independent_of_cwd(
    synthetic_mock_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ledger path resolves to repository path even when cwd is changed."""
    outside_dir = tmp_path / "outside_dir"
    outside_dir.mkdir()

    monkeypatch.chdir(outside_dir)
    report, _ = verify_pre_opening(
        repo_root=synthetic_mock_repo,
        station_ids=["24025040"],
        dry_run=True,
    )
    assert report.ledger_path_resolved_from_repo is True
