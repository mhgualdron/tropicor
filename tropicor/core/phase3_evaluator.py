"""Phase 3 Evaluator for Andean Batch 1 Temperature Opening (Protocol v1.7, App I).

Implements the pre-specified temperature evaluation hierarchy for the five Andean
stations (37015020, 24025040, 26135040, 27015070, 52045010) under Protocol v1.7:
- I.3: Strict pre-opening verification of repository state, authorization tag,
  frozen input hashes, winning configurations, validation file existence, and
  single-pass fitting of Levels 1, 2, 3-winner, and 3-constant on calibration data.
- I.5: Write-ahead single opening via audit ledger, with
  reproduction and interruption guards.
- I.6: Strict validation series loading with out-of-sample
  masking and no calibration leakage.
- I.7: Decisive metrics (3 decisive, 1,000 replicas, annual blocks, seed 42),
  physical consistency gate, crossing diagnostics, and descriptive metrics
  (Section G.9).
- I.8: Paired annual block bootstrap progression analysis
  across levels (0 -> 1 -> 2 -> 3).
- I.9: Deterministic JSON output generation (sorted keys, LF, no NaN, atomic write).
- G.3: Descriptive counts by level with no global synthesis verdict.

Methodological note on observation file reading:
Observation CSV files contain both calibration and validation blocks in a single
file; the frozen calibration loader and _read_obs_block parse the entire file and
slice by block prior to any use. No validation values are used prior to ledger
recording.
 
This module has NO side effects upon import.
"""

from __future__ import annotations

import dataclasses
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import uuid
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Collection,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    cast,
)

import numpy as np
import pandas as pd

from tropicor.core.andean_registry import AndeanStation, get_andean_station
from tropicor.core.protocol import (
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_LEDGER_PATH,
    MIN_BOOTSTRAP_BLOCKS,
    BootstrapConfidenceInterval,
    CrossingDiagnostic,
    GlobalOutcome,
    check_temperature_physical_gate,
    compute_daily_pearson,
    compute_temperature_decisive_metrics,
    evaluate_temperature_scorecard,
    get_current_git_commit,
    validate_validation_access,
)
from tropicor.downscale.temperature import (
    LEVEL3_TEMPERATURE_CONFIGS,
    Level1MonthlyAdditiveTemperature,
    Level2MeanStateDTRScalingTemperature,
    Level3QuantileMappingTemperature,
    _validate_temperature_range,
    load_calibration_temperature_pair,
)

# Canonical 5 Andean generalization stations
ANDEAN_PHASE3_STATION_IDS: Tuple[str, ...] = (
    "24025040",
    "26135040",
    "27015070",
    "37015020",
    "52045010",
)

EXPECTED_INPUTS_REL_PATH: Path = Path("tropicor/data/phase3_expected_inputs.json")
README_APPENDIX_G_REL_PATH: Path = Path("docs/science/appendix_g/README.md")
AUTHORIZATION_TAG_NAME: str = "phase3-authorization"

# Module-level constant for evaluation replicas (overridden in tests via monkeypatch)
PHASE3_BOOTSTRAP_REPLICAS: int = 1000


# ==============================================================================
# 1. Hashing and Verification Helpers (Appendix I.3)
# ==============================================================================


def compute_sha256(file_path: Path) -> str:
    """Compute the lowercase SHA-256 hexadecimal digest of a file.

    Reads in binary chunks without decoding or parsing data content.
    """
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found for hashing: {file_path}")
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def parse_appendix_g_readme_hashes(readme_path: Path) -> Dict[str, str]:
    """Parse file-to-SHA-256 table from docs/science/appendix_g/README.md.

    Returns mapping of file relative path string to lowercase SHA-256 hex digest.
    """
    if not readme_path.is_file():
        raise FileNotFoundError(f"Appendix G README not found: {readme_path}")
    hashes: Dict[str, str] = {}
    with open(readme_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            # Markdown table row: | `filename` | `hash` |
            m = re.match(
                r"^\|\s*`([^`]+)`\s*\|\s*`([a-fA-F0-9]{64})`\s*\|",
                stripped,
            )
            if m:
                rel_file = m.group(1).strip()
                digest = m.group(2).strip().lower()
                hashes[rel_file] = digest
    return hashes


def load_expected_inputs_config(config_path: Path) -> Dict[str, Any]:
    """Load the frozen Phase 3 expected inputs configuration."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Expected inputs config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return cast(Dict[str, Any], json.load(f))


# ==============================================================================
# 2. Pre-Opening Verification and Fitting (Appendix I.3, I.4)
# ==============================================================================


@dataclass(frozen=True)
class PreOpeningChecksReport:
    """Diagnostic report from Phase 3 pre-opening verification checks."""

    git_clean: bool
    git_commit: str
    tag_exists_and_annotated: bool
    tag_target_commit: str
    sha256_checks_passed: bool
    readme_hashes_matched: bool
    lag_sensitivity_not_activated: bool
    winner_configs_matched: bool
    validation_files_exist_and_hashed: bool
    ledger_path_resolved_from_repo: bool
    protocol_commit_equals_head: bool
    raw_bias_checks_passed: bool
    pipeline_smoke_passed: bool
    input_file_hashes: Dict[str, Any]
    details: Dict[str, Any]

    @property
    def all_checks_passed(self) -> bool:
        """Return True if all formal pre-opening checks passed."""
        return (
            self.git_clean
            and self.tag_exists_and_annotated
            and (self.tag_target_commit == self.git_commit)
            and self.sha256_checks_passed
            and self.readme_hashes_matched
            and self.lag_sensitivity_not_activated
            and self.winner_configs_matched
            and self.validation_files_exist_and_hashed
            and self.ledger_path_resolved_from_repo
            and self.protocol_commit_equals_head
            and self.raw_bias_checks_passed
            and self.pipeline_smoke_passed
        )


def _check_git_working_tree_and_head(
    repo_root: Path,
) -> Tuple[bool, str, Optional[str]]:
    """Verify git working tree status and retrieve HEAD commit hash.

    Returns:
        (is_clean, head_commit, error_message)
    """
    try:
        proc_status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc_status.returncode != 0:
            return (
                False,
                "unknown_or_uncommitted",
                f"git status failed with exit code {proc_status.returncode}: "
                f"{proc_status.stderr.strip()}",
            )
        is_clean = len(proc_status.stdout.strip()) == 0

        proc_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc_head.returncode != 0:
            return (
                False,
                "unknown_or_uncommitted",
                f"git rev-parse HEAD failed with exit code {proc_head.returncode}: "
                f"{proc_head.stderr.strip()}",
            )
        head_commit = proc_head.stdout.strip()
        if not re.match(r"^[0-9a-fA-F]{40}$", head_commit):
            return (
                False,
                head_commit,
                f"Invalid commit hash returned from git: {head_commit}",
            )

        return is_clean, head_commit, None
    except Exception as err:
        return False, "unknown_or_uncommitted", str(err)


def _check_authorization_tag(
    repo_root: Path, tag_name: str, expected_commit: str
) -> Tuple[bool, str, Optional[str]]:
    """Verify that tag exists, is annotated, and targets expected_commit.

    Returns:
        (is_valid, target_commit, error_message)
    """
    try:
        proc_type = subprocess.run(
            ["git", "cat-file", "-t", tag_name],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc_type.returncode != 0:
            return (
                False,
                "missing",
                f"Tag '{tag_name}' does not exist or git cat-file failed.",
            )
        obj_type = proc_type.stdout.strip()
        if obj_type != "tag":
            return (
                False,
                obj_type,
                f"Tag '{tag_name}' is not an annotated tag (type is '{obj_type}').",
            )

        proc_target = subprocess.run(
            ["git", "rev-parse", f"{tag_name}^{{commit}}"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc_target.returncode != 0:
            return (
                False,
                "unknown",
                f"Failed to resolve commit target for tag '{tag_name}'.",
            )
        target_commit = proc_target.stdout.strip()
        if target_commit != expected_commit:
            return (
                False,
                target_commit,
                f"Tag '{tag_name}' targets commit {target_commit}, "
                f"expected HEAD {expected_commit}.",
            )

        return True, target_commit, None
    except Exception as err:
        return False, "error", str(err)


def verify_pre_opening(
    repo_root: Path,
    expected_inputs_path: Optional[Path] = None,
    registry_path: Optional[Path] = None,
    era5_base_dir: Optional[Path] = None,
    ledger_path: Optional[Path] = None,
    station_ids: Sequence[str] = ANDEAN_PHASE3_STATION_IDS,
    dry_run: bool = False,
) -> Tuple[PreOpeningChecksReport, Dict[str, Any]]:
    """Execute pre-opening verification checks (I.3) and calibration fitting (I.4).

    Verifies Checks 1 through 6, performs calibration raw bias recalculation,
    fits calibration models, and when dry_run=True, executes a smoke test
    evaluating pipeline integrity without reading any validation observations.

    Note:
        During preflight (dry_run=True), ERA5 model data for the validation period
        (val_start..val_end+1) is loaded to verify complete temporal coverage and
        physical temperature bounds [-60, 60] °C, but NO validation observations
        are read or accessed. In-sample calibration series are used for pipeline
        smoke testing.

    Args:
        repo_root: Path to repository root.
        expected_inputs_path: Path to phase3_expected_inputs.json.
        registry_path: Path to andean_station_registry.json.
        era5_base_dir: Directory containing station timeseries NetCDF/CSV files.
        ledger_path: Path to audit ledger (used for Check 6 validation).
        station_ids: Sequence of 5 station identifiers to verify.
        dry_run: If True, reports checks without aborting on Checks 1-2,
                 and runs the preflight pipeline smoke test.

    Returns:
        (report, fitted_models)
    """
    repo_root = repo_root.resolve()
    details: Dict[str, Any] = {}
    input_file_hashes: Dict[str, Any] = {}

    # Check 1: Git working tree clean & HEAD commit
    is_clean, verified_head, git_err = _check_git_working_tree_and_head(repo_root)
    git_check_ok = bool(
        is_clean
        and verified_head != "unknown_or_uncommitted"
        and len(verified_head) == 40
    )
    details["git_clean"] = is_clean
    details["git_head"] = verified_head
    if git_err:
        details["git_error"] = git_err

    # Check protocol.get_current_git_commit() matches verified_head
    protocol_commit = get_current_git_commit()
    details["protocol_commit"] = protocol_commit
    protocol_commit_equals_head = bool(
        git_check_ok and (protocol_commit == verified_head)
    )

    # Check 2: Annotated authorization tag 'phase3-authorization'
    tag_ok, tag_target, tag_err = _check_authorization_tag(
        repo_root, AUTHORIZATION_TAG_NAME, verified_head
    )
    details["tag_valid"] = tag_ok
    details["tag_target"] = tag_target
    if tag_err:
        details["tag_error"] = tag_err

    # Check 3: SHA-256 hashes of registry, lag audit, calibration reports
    cfg_path = expected_inputs_path or (repo_root / EXPECTED_INPUTS_REL_PATH)
    expected_cfg = load_expected_inputs_config(cfg_path)
    archived_paths = expected_cfg.get("archived_paths", {})

    reg_file = registry_path or (
        repo_root
        / archived_paths.get("registry", "tropicor/data/andean_station_registry.json")
    )
    reg_hash = compute_sha256(reg_file)
    input_file_hashes["registry"] = reg_hash

    lag_file = repo_root / archived_paths.get(
        "lag_audit", "docs/science/appendix_g/lag_audit_andean_calibration.json"
    )
    lag_hash = compute_sha256(lag_file)
    input_file_hashes["lag_audit"] = lag_hash

    cal_report_hashes: Dict[str, str] = {}
    cal_file_map: Dict[str, Path] = {}
    for st in station_ids:
        cf_rel = archived_paths.get("calibration", {}).get(st)
        if not cf_rel:
            raise ValueError(
                f"Missing archived path for station {st} in expected config."
            )
        cf = repo_root / cf_rel
        cal_file_map[st] = cf
        cal_report_hashes[st] = compute_sha256(cf)
    input_file_hashes["calibration"] = cal_report_hashes

    sha256_checks_passed = bool(
        reg_hash == expected_cfg.get("registry")
        and lag_hash == expected_cfg.get("lag_audit")
        and all(
            cal_report_hashes[st] == expected_cfg.get("calibration", {}).get(st)
            for st in station_ids
        )
    )

    # Check 3b: Appendix G README table hashes match
    readme_path = repo_root / README_APPENDIX_G_REL_PATH
    readme_table = parse_appendix_g_readme_hashes(readme_path)
    readme_hashes_matched = True

    # (i) Lag audit row in README
    expected_lag_hash = str(expected_cfg.get("lag_audit", "")).lower()
    table_lag_hash = (
        readme_table.get("lag_audit_andean_calibration.json")
        or readme_table.get("docs/science/appendix_g/lag_audit_andean_calibration.json")
    )
    if table_lag_hash != expected_lag_hash:
        readme_hashes_matched = False
        details["readme_mismatch_lag"] = {
            "expected": expected_lag_hash,
            "found": table_lag_hash,
        }

    # (ii) Calibration report rows in README
    for st in station_ids:
        cal_rel = archived_paths.get("calibration", {}).get(st)
        if not cal_rel:
            readme_hashes_matched = False
            break
        rel_sub = f"calibration/{Path(cal_rel).name}"
        table_hash = readme_table.get(rel_sub) or readme_table.get(
            Path(cal_rel).name
        )
        if table_hash != cal_report_hashes[st]:
            readme_hashes_matched = False
            details[f"readme_mismatch_{st}"] = {
                "expected": cal_report_hashes[st],
                "found": table_hash,
            }

    # Check 3c: Lag sensitivity not activated
    with open(lag_file, "r", encoding="utf-8") as f:
        lag_data = json.load(f)
    lag_stations = lag_data.get("stations", {})
    lag_sensitivity_not_activated = True
    for st in station_ids:
        st_lag_info = lag_stations.get(st, {})
        activated = st_lag_info.get("activated_lags", [])
        is_act = bool(st_lag_info.get("lag_sensitivity_activated", False))
        if len(activated) > 0 or is_act:
            lag_sensitivity_not_activated = False
            raise ValueError("lag sensitivity not implemented")

    # Check 4: winner configuration == LEVEL3_TEMPERATURE_CONFIGS[id]
    winner_configs_matched = True
    winner_configs: Dict[str, Dict[str, Any]] = {}
    calibration_reports_data: Dict[str, Dict[str, Any]] = {}
    for st in station_ids:
        cf = cal_file_map[st]
        with open(cf, "r", encoding="utf-8") as f:
            cdata = json.load(f)
        calibration_reports_data[st] = cdata
        l3_info = cdata.get("level3", {})
        win_id = int(l3_info["winner_config_id"])
        win_group = str(l3_info["winner_grouping"])
        win_nq = int(l3_info["winner_n_quantiles"])
        win_extrap = str(l3_info["winner_extrapolation"])

        if win_id < 0 or win_id >= len(LEVEL3_TEMPERATURE_CONFIGS):
            winner_configs_matched = False
            break
        ref_cfg = LEVEL3_TEMPERATURE_CONFIGS[win_id]
        if not (
            ref_cfg["grouping"] == win_group
            and ref_cfg["n_quantiles"] == win_nq
            and ref_cfg["extrapolation"] == win_extrap
        ):
            winner_configs_matched = False

        winner_configs[st] = {
            "config_id": win_id,
            "grouping": win_group,
            "n_quantiles": win_nq,
            "extrapolation": win_extrap,
        }
    details["winner_configs"] = winner_configs

    # Check 5: Observation CSVs and ERA5 NetCDFs exist and hashed
    base_ts = era5_base_dir or (repo_root / "local" / "era5_ts")
    validation_files_exist = True
    station_files: Dict[str, Dict[str, Any]] = {}

    for st in station_ids:
        st_dir = base_ts / st
        if not st_dir.is_dir():
            validation_files_exist = False
            details[f"station_{st}_dir_missing"] = str(st_dir)
            break

        # Observation files: exactly 1 TempMaxima_*.csv and 1 TempMinima_*.csv
        tmax_list = sorted(list(st_dir.glob("TempMaxima_*.csv")))
        tmin_list = sorted(list(st_dir.glob("TempMinima_*.csv")))

        if len(tmax_list) != 1 or len(tmin_list) != 1:
            validation_files_exist = False
            details[f"station_{st}_csv_count_error"] = {
                "tmax_count": len(tmax_list),
                "tmin_count": len(tmin_list),
            }
            break

        tmax_csv = tmax_list[0]
        tmin_csv = tmin_list[0]
        tmax_h = compute_sha256(tmax_csv)
        tmin_h = compute_sha256(tmin_csv)

        # Retrieve station metadata from registry
        st_meta = get_andean_station(st, registry_path=reg_file)
        cal_start, cal_end = st_meta.cal_block
        val_start, val_end = st_meta.val_block

        # Calibration NetCDFs for cal_start..cal_end+1
        era5_cal_hashes: Dict[str, str] = {}
        for y in range(cal_start, cal_end + 2):
            nc_name = f"era5_2m_temperature_{y}.nc"
            nc_path = st_dir / nc_name
            if not nc_path.is_file():
                validation_files_exist = False
                details[f"station_{st}_missing_era5_cal_year"] = y
                break
            era5_cal_hashes[nc_name] = compute_sha256(nc_path)

        # Validation NetCDFs for val_start..val_end+1
        era5_val_hashes: Dict[str, str] = {}
        for y in range(val_start, val_end + 2):
            nc_name = f"era5_2m_temperature_{y}.nc"
            nc_path = st_dir / nc_name
            if not nc_path.is_file():
                validation_files_exist = False
                details[f"station_{st}_missing_era5_val_year"] = y
                break
            era5_val_hashes[nc_name] = compute_sha256(nc_path)

        if not validation_files_exist:
            break

        station_files[st] = {
            "tmax_csv": tmax_csv,
            "tmin_csv": tmin_csv,
            "tmax_sha256": tmax_h,
            "tmin_sha256": tmin_h,
            "era5_dir": st_dir,
            "era5_cal_files": [st_dir / name for name in era5_cal_hashes],
            "era5_val_files": [st_dir / name for name in era5_val_hashes],
            "era5_cal_hashes": era5_cal_hashes,
            "era5_val_hashes": era5_val_hashes,
            "cal_block": st_meta.cal_block,
            "val_block": st_meta.val_block,
            "valid_years": st_meta.valid_years,
        }

    input_file_hashes["era5_calibration"] = {
        st: info["era5_cal_hashes"] for st, info in station_files.items()
    }
    input_file_hashes["station_files"] = {
        st: {
            "tmax_csv": info["tmax_sha256"],
            "tmin_csv": info["tmin_sha256"],
            "era5_calibration_files": info["era5_cal_hashes"],
            "era5_validation_files": info["era5_val_hashes"],
        }
        for st, info in station_files.items()
    }

    # Relative paths mapping for preflight display (C4c)
    station_relative_paths: Dict[str, Dict[str, Any]] = {}
    for st, info in station_files.items():
        try:
            rel_tmax = str(
                info["tmax_csv"].resolve().relative_to(repo_root.resolve())
            ).replace("\\", "/")
        except ValueError:
            rel_tmax = f"local/era5_ts/{st}/{info['tmax_csv'].name}"
        try:
            rel_tmin = str(
                info["tmin_csv"].resolve().relative_to(repo_root.resolve())
            ).replace("\\", "/")
        except ValueError:
            rel_tmin = f"local/era5_ts/{st}/{info['tmin_csv'].name}"
        station_relative_paths[st] = {
            "tmax_csv": rel_tmax,
            "tmin_csv": rel_tmin,
        }
    details["station_relative_paths"] = station_relative_paths

    # Check 6: Ledger path resolution (C1d: demands exact canonical path equality)
    canonical_ledger = (repo_root / DEFAULT_LEDGER_PATH).resolve()
    resolved_ledger = ledger_path or canonical_ledger
    if not resolved_ledger.is_absolute():
        resolved_ledger = (repo_root / resolved_ledger).resolve()
    ledger_path_resolved_from_repo = bool(resolved_ledger.resolve() == canonical_ledger)

    # In formal mode (not dry_run), any pre-opening check failure aborts immediately
    if not dry_run:
        if not is_clean or not git_check_ok:
            raise RuntimeError(
                "Pre-opening check failed: Working tree is dirty or git error: "
                f"{git_err}"
            )
        if not tag_ok or tag_target != verified_head:
            raise RuntimeError(
                f"Pre-opening check failed: Tag '{AUTHORIZATION_TAG_NAME}' "
                f"missing, unannotated, or does not target HEAD ({tag_err})"
            )
        if not sha256_checks_passed:
            raise ValueError(
                "Pre-opening check failed: SHA-256 hashes of inputs "
                "do not match expected config."
            )
        if not readme_hashes_matched:
            raise ValueError(
                "Pre-opening check failed: Archive README hashes "
                "do not match expected config."
            )
        if not lag_sensitivity_not_activated:
            raise ValueError("lag sensitivity not implemented")
        if not winner_configs_matched:
            raise ValueError(
                "winner config does not match LEVEL3_TEMPERATURE_CONFIGS"
            )
        if not validation_files_exist:
            raise FileNotFoundError(
                "Pre-opening check failed: One or more validation files missing."
            )
        if not ledger_path_resolved_from_repo:
            raise ValueError(
                f"Pre-opening check failed: Ledger path {resolved_ledger} "
                f"must resolve to canonical repository path {canonical_ledger}."
            )

    # I.4 Single-pass fitting on calibration block & C4b raw bias verification
    fitted_models: Dict[str, Any] = {}
    raw_bias_checks_passed = True
    raw_bias_details: Dict[str, Any] = {}

    if validation_files_exist and sha256_checks_passed:
        for st in station_ids:
            if st not in station_files:
                continue
            st_info = station_files[st]
            w_cfg = winner_configs[st]

            # Load calibration paired series via frozen calibration loader
            try:
                obs_tmax_cal, obs_tmin_cal, mod_tmax_cal, mod_tmin_cal = (
                    load_calibration_temperature_pair(
                        station_id=st,
                        obs_tmax_path=st_info["tmax_csv"],
                        obs_tmin_path=st_info["tmin_csv"],
                        era5_target=st_info["era5_dir"],
                        window_start_utc=5,
                        lag_days=0,
                        min_hours=24,
                    )
                )

                # C4b: Raw bias recalculation and verification against archived
                # calibration report (definition in
                # scripts/run_andean_temperature_calibration.py:
                # raw_bias = round(float((mod - obs).dropna().mean()), 4))
                calc_raw_bias_tmax = round(
                    float((mod_tmax_cal - obs_tmax_cal).dropna().mean()), 4
                )
                calc_raw_bias_tmin = round(
                    float((mod_tmin_cal - obs_tmin_cal).dropna().mean()), 4
                )
                cdata = calibration_reports_data[st]
                arch_bias = cdata.get("raw_bias", {})
                arch_tmax = float(arch_bias.get("tmax", 0.0))
                arch_tmin = float(arch_bias.get("tmin", 0.0))
                diff_tmax = abs(calc_raw_bias_tmax - arch_tmax)
                diff_tmin = abs(calc_raw_bias_tmin - arch_tmin)
                bias_ok = bool(diff_tmax <= 1e-6 and diff_tmin <= 1e-6)
                if not bias_ok:
                    raw_bias_checks_passed = False
                    msg = (
                        f"Calibration raw bias mismatch for station {st}: "
                        f"recalculated (tmax={calc_raw_bias_tmax:.6f}, "
                        f"tmin={calc_raw_bias_tmin:.6f}) vs "
                        f"archived (tmax={arch_tmax:.6f}, tmin={arch_tmin:.6f}); "
                        f"diffs: tmax={diff_tmax:.2e}, tmin={diff_tmin:.2e}."
                    )
                    if not dry_run:
                        raise ValueError(msg)
                    else:
                        details[f"station_{st}_raw_bias_error"] = msg

                raw_bias_details[st] = {
                    "recalculated_tmax": calc_raw_bias_tmax,
                    "recalculated_tmin": calc_raw_bias_tmin,
                    "archived_tmax": arch_tmax,
                    "archived_tmin": arch_tmin,
                    "matched": bias_ok,
                }

                # Fit Level 1
                m1 = Level1MonthlyAdditiveTemperature()
                m1.fit(obs_tmax_cal, obs_tmin_cal, mod_tmax_cal, mod_tmin_cal)

                # Fit Level 2
                m2 = Level2MeanStateDTRScalingTemperature()
                m2.fit(obs_tmax_cal, obs_tmin_cal, mod_tmax_cal, mod_tmin_cal)

                # Fit Level 3 Winner
                grouping_val = cast(
                    Literal["monthly", "moving_window"], w_cfg["grouping"]
                )
                extrap_val = cast(
                    Literal["constant", "linear"], w_cfg["extrapolation"]
                )
                m3_winner = Level3QuantileMappingTemperature(
                    grouping=grouping_val,
                    n_quantiles=w_cfg["n_quantiles"],
                    extrapolation=extrap_val,
                )
                m3_winner.fit(obs_tmax_cal, obs_tmin_cal, mod_tmax_cal, mod_tmin_cal)

                # Fit Level 3 Constant sensitivity
                if w_cfg["extrapolation"] == "constant":
                    m3_constant = m3_winner
                    identical_to_winner = True
                else:
                    m3_constant = Level3QuantileMappingTemperature(
                        grouping=grouping_val,
                        n_quantiles=w_cfg["n_quantiles"],
                        extrapolation="constant",
                    )
                    m3_constant.fit(
                        obs_tmax_cal, obs_tmin_cal, mod_tmax_cal, mod_tmin_cal
                    )
                    identical_to_winner = False

                fitted_models[st] = {
                    "level1": m1,
                    "level2": m2,
                    "level3_winner": m3_winner,
                    "level3_constant": m3_constant,
                    "identical_to_winner": identical_to_winner,
                    "winner_config": w_cfg,
                    "station_files": st_info,
                    "cal_report_data": calibration_reports_data[st],
                    "mod_tmax_cal": mod_tmax_cal,
                    "mod_tmin_cal": mod_tmin_cal,
                }
            except Exception as cal_err:
                raw_bias_checks_passed = False
                details[f"station_{st}_calibration_error"] = str(cal_err)
                if not dry_run:
                    raise

    details["raw_bias_details"] = raw_bias_details

    # Pipeline smoke execution (C3/F4) runs across ALL modes
    # (formal, reproduction, dry_run). Must run before write-ahead
    # ledger writes in run_phase3_evaluation.
    pipeline_smoke_passed = False
    prior_integrity_ok = bool(
        sha256_checks_passed
        and readme_hashes_matched
        and validation_files_exist
        and raw_bias_checks_passed
    )
    if prior_integrity_ok:
        try:
            for st in station_ids:
                if st not in station_files or st not in fitted_models:
                    continue
                st_info = station_files[st]
                st_meta = get_andean_station(st, registry_path=reg_file)
                val_start, val_end = st_meta.val_block
                cal_start, cal_end = st_meta.cal_block

                # (i) ERA5 validation model block (val_start..val_end+1)
                # Verify 0 NaNs and physical temperature range [-60, 60] °C
                mod_tmax_val_smoke, mod_tmin_val_smoke = _load_era5_daily_block(
                    era5_dir=st_info["era5_dir"],
                    andean_station=st_meta,
                    years=list(range(val_start, val_end + 2)),
                    window_start_utc=5,
                    lag_days=0,
                    min_hours=24,
                )
                if (
                    mod_tmax_val_smoke.isna().any()
                    or mod_tmin_val_smoke.isna().any()
                ):
                    raise ValueError(
                        f"ERA5 daily validation series contains missing values "
                        f"for station {st}."
                    )
                _validate_temperature_range(
                    mod_tmax_val_smoke, f"mod_tmax_val_smoke ({st})"
                )
                _validate_temperature_range(
                    mod_tmin_val_smoke, f"mod_tmin_val_smoke ({st})"
                )

                # (ii) Calibration obs block via _read_obs_block
                obs_tmax_cal_smoke = _read_obs_block(
                    path=st_info["tmax_csv"],
                    station_id=st,
                    start_year=cal_start,
                    end_year=cal_end,
                    valid_years=st_meta.valid_years,
                )
                obs_tmin_cal_smoke = _read_obs_block(
                    path=st_info["tmin_csv"],
                    station_id=st,
                    start_year=cal_start,
                    end_year=cal_end,
                    valid_years=st_meta.valid_years,
                )
                _validate_temperature_range(
                    obs_tmax_cal_smoke, f"obs_tmax_cal_smoke ({st})"
                )
                _validate_temperature_range(
                    obs_tmin_cal_smoke, f"obs_tmin_cal_smoke ({st})"
                )

                # (iii) In-sample calibration evaluate_single_station smoke run
                # (20 replicas)
                smoke_opening_rec = {
                    "station_id": st,
                    "variable": "temperature",
                    "timestamp_utc": "1970-01-01T00:00:00Z",
                    "commit_hash": verified_head,
                    "purpose": "preflight_pipeline_smoke",
                    "config": {"run_kind": "preflight_smoke"},
                }
                smoke_hashes = {
                    "registry": input_file_hashes["registry"],
                    "lag_audit": input_file_hashes["lag_audit"],
                    "calibration_report": input_file_hashes["calibration"][st],
                    "obs_tmax_csv": st_info["tmax_sha256"],
                    "obs_tmin_csv": st_info["tmin_sha256"],
                    "era5_calibration_files": st_info["era5_cal_hashes"],
                    "era5_validation_files": st_info["era5_val_hashes"],
                }
                _ = evaluate_single_station(
                    station_id=st,
                    fitted_info=fitted_models[st],
                    obs_tmax_val=obs_tmax_cal_smoke,
                    obs_tmin_val=obs_tmin_cal_smoke,
                    mod_tmax_val=fitted_models[st]["mod_tmax_cal"],
                    mod_tmin_val=fitted_models[st]["mod_tmin_cal"],
                    opening_record=smoke_opening_rec,
                    evaluator_commit=verified_head,
                    tag_target=tag_target,
                    run_kind="formal",
                    interrupted_formal_opening=False,
                    input_hashes=smoke_hashes,
                    bootstrap_replicas=20,
                )
                details[f"smoke_{st}"] = "PASS"
            pipeline_smoke_passed = True
        except Exception as smoke_err:
            pipeline_smoke_passed = False
            details["pipeline_smoke_error"] = str(smoke_err)
            if not dry_run:
                raise


    report = PreOpeningChecksReport(
        git_clean=is_clean,
        git_commit=verified_head,
        tag_exists_and_annotated=tag_ok,
        tag_target_commit=tag_target,
        sha256_checks_passed=sha256_checks_passed,
        readme_hashes_matched=readme_hashes_matched,
        lag_sensitivity_not_activated=lag_sensitivity_not_activated,
        winner_configs_matched=winner_configs_matched,
        validation_files_exist_and_hashed=validation_files_exist,
        ledger_path_resolved_from_repo=ledger_path_resolved_from_repo,
        protocol_commit_equals_head=protocol_commit_equals_head,
        raw_bias_checks_passed=raw_bias_checks_passed,
        pipeline_smoke_passed=pipeline_smoke_passed,
        input_file_hashes=input_file_hashes,
        details=details,
    )

    # In formal mode, any failed check raises an error immediately
    if not dry_run:
        if not report.git_clean or not git_check_ok:
            raise RuntimeError(
                "Pre-opening check failed: Working tree is dirty or git error: "
                f"{git_err}"
            )
        if (
            not report.tag_exists_and_annotated
            or report.tag_target_commit != report.git_commit
        ):
            raise RuntimeError(
                f"Pre-opening check failed: Tag '{AUTHORIZATION_TAG_NAME}' "
                f"missing, unannotated, or does not target HEAD ({tag_err})"
            )
        if not report.sha256_checks_passed:
            raise ValueError(
                "Pre-opening check failed: SHA-256 hashes of inputs "
                "do not match expected config."
            )
        if not report.readme_hashes_matched:
            raise ValueError(
                "Pre-opening check failed: Archive README hashes "
                "do not match expected config."
            )
        if not report.lag_sensitivity_not_activated:
            raise ValueError("lag sensitivity not implemented")
        if not report.winner_configs_matched:
            raise ValueError(
                "Pre-opening check failed: Calibration report winner config "
                "does not match LEVEL3_TEMPERATURE_CONFIGS."
            )
        if not report.validation_files_exist_and_hashed:
            raise FileNotFoundError(
                "Pre-opening check failed: Required observation or ERA5 "
                "validation files missing."
            )
        if not report.ledger_path_resolved_from_repo:
            raise ValueError(
                f"Pre-opening check failed: Ledger path {resolved_ledger} "
                f"must resolve to canonical repository path {canonical_ledger}."
            )
        if not report.protocol_commit_equals_head:
            raise RuntimeError(
                "Pre-opening check failed: protocol.get_current_git_commit() "
                "does not match verified HEAD."
            )
        if not report.raw_bias_checks_passed:
            raise ValueError("Pre-opening check failed: Calibration raw bias mismatch.")
        if not report.pipeline_smoke_passed:
            raise RuntimeError("Pre-opening check failed: Pipeline smoke test failed.")

    return report, fitted_models


# ==============================================================================
# 3. Data Extraction Helpers and Validation Loader (Appendix I.5, I.6, G.5.2)
# ==============================================================================


def _read_obs_block(
    path: Path,
    station_id: str,
    start_year: int,
    end_year: int,
    valid_years: Optional[Collection[int]] = None,
) -> pd.Series:
    """Read observation extremes from CSV, parse dates, and slice to year block.

    Args:
        path: CSV path containing columns CodigoEstacion, Fecha, Valor.
        station_id: Target station identifier.
        start_year: Start year of target block (inclusive).
        end_year: End year of target block (inclusive).
        valid_years: Optional collection of calendar years with acceptable data quality.
                     Years in the block not in valid_years are masked to NaN.

    Returns:
        pd.Series indexed by daily pd.DatetimeIndex with values in °C.
    """
    clean_id = re.sub(r"\.0$", "", str(station_id).strip())
    if not path.is_file():
        raise FileNotFoundError(f"Observation file not found: {path}")

    df_raw = pd.read_csv(path)
    for req_col in ("CodigoEstacion", "Fecha", "Valor"):
        if req_col not in df_raw.columns:
            raise ValueError(f"Required column '{req_col}' not found in '{path}'.")

    st_codes = (
        df_raw["CodigoEstacion"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    if clean_id not in st_codes.values:
        raise ValueError(
            f"Station ID '{clean_id}' not found in observation file '{path}'."
        )

    df_st = df_raw[st_codes == clean_id].copy()

    # Strict ISO8601 parsing
    parsed_dt = pd.to_datetime(df_st["Fecha"], format="ISO8601", errors="coerce")
    unparseable_mask = (
        parsed_dt.isna()
        & df_st["Fecha"].notna()
        & (df_st["Fecha"].astype(str).str.strip() != "")
    )
    if unparseable_mask.sum() > 0:
        raise ValueError(f"Found unparseable date(s) in '{path}'.")

    # Validate numeric
    raw_val_str = df_st["Valor"].astype(str).str.strip().str.replace(",", ".")
    num_vals = pd.to_numeric(raw_val_str, errors="coerce")
    non_numeric_mask = (
        num_vals.isna()
        & df_st["Valor"].notna()
        & ~raw_val_str.str.upper().isin(["", "NAN", "NONE", "NULL"])
    )
    if non_numeric_mask.sum() > 0:
        raise ValueError(f"Found non-numeric value(s) in '{path}'.")

    # Check duplicate dates with distinct values
    df_st["dt_norm"] = parsed_dt.dt.normalize()
    dup_mask = df_st["dt_norm"].duplicated(keep=False)
    if dup_mask.any():
        dup_rows = df_st[dup_mask]
        for _, group in dup_rows.groupby("dt_norm"):
            valid_distinct = group["Valor"].dropna().unique()
            if len(valid_distinct) > 1:
                raise ValueError(
                    f"Found duplicate date with distinct values in '{path}'."
                )

    df_st = df_st.dropna(subset=["dt_norm"]).sort_values("dt_norm")
    df_st = df_st.drop_duplicates(subset=["dt_norm"], keep="first")

    clean_num = pd.to_numeric(
        df_st["Valor"].astype(str).str.strip().str.replace(",", "."),
        errors="coerce",
    )
    s_full = pd.Series(clean_num.values, index=pd.DatetimeIndex(df_st["dt_norm"]))
    s_idx = cast(pd.DatetimeIndex, s_full.index)

    full_block_index = pd.date_range(
        f"{start_year}-01-01", f"{end_year}-12-31", freq="D"
    )
    block_mask = (s_idx.year >= start_year) & (s_idx.year <= end_year)
    s_block = s_full[block_mask].reindex(full_block_index)

    if valid_years is not None:
        valid_set = set(valid_years)
        block_dt_idx = cast(pd.DatetimeIndex, s_block.index)
        invalid_years_mask = ~block_dt_idx.year.isin(valid_set)
        s_block.loc[invalid_years_mask] = np.nan

    return s_block





def _load_era5_daily_block(
    era5_dir: Path,
    andean_station: AndeanStation,
    years: Sequence[int],
    window_start_utc: int = 5,
    lag_days: int = 0,
    min_hours: int = 24,
) -> Tuple[pd.Series, pd.Series]:
    """Load ERA5 hourly NetCDFs, verify metadata, and aggregate to daily Tmax/Tmin.

    Logic aligned with load_calibration_temperature_pair in temperature.py:
    - Exact verification of registered ERA5 grid cell celda_era5 (atol=1e-3).
    - Dimensions: squeeze('step', 'surface'), isel(number=0).
    - Unit validation: requires 'K' (Kelvin).
    - Explicit Celsius conversion: - 273.15 with aggregate(convert_to_celsius=False).
    - Verification of 0 NaNs in daily partition.
    - Physical range validation [-60, 60] °C.

    Args:
        era5_dir: Directory containing station ERA5 NetCDFs (or single file).
        andean_station: AndeanStation registry metadata.
        years: NetCDF years to load (e.g. val_start..val_end+1).
        window_start_utc: Start hour for 24h aggregation (default 5 UTC).
        lag_days: Shift in days (default 0).
        min_hours: Minimum hours required for valid daily extreme (default 24).

    Returns:
        (mod_tmax, mod_tmin) Series sliced to the target partition.
    """
    from tropicor.io.era5 import ERA5Adapter, aggregate_hourly_to_daily_temperature

    clean_id = andean_station.codigo
    needed_years = sorted(list(set(years)))
    start_year = needed_years[0]
    # If more than one year provided and max year is boundary year (e.g. val_end + 1),
    # target slice is start_year .. max_year - 1
    if len(needed_years) > 1 and needed_years[-1] == andean_station.val_block[1] + 1:
        target_end_year = andean_station.val_block[1]
    elif len(needed_years) > 1 and needed_years[-1] == andean_station.cal_block[1] + 1:
        target_end_year = andean_station.cal_block[1]
    else:
        target_end_year = needed_years[-1]

    if era5_dir.is_dir():
        nc_files = sorted(glob.glob(str(era5_dir / "*2m_temperature*.nc")))
        val_nc_files = [
            f
            for f in nc_files
            if any(re.search(rf"(?<!\d){y}(?!\d)", Path(f).name) for y in needed_years)
        ]
        if len(val_nc_files) < len(needed_years):
            raise ValueError(
                f"Missing ERA5 NetCDF files for validation years "
                f"{needed_years[0]}-{needed_years[-1]} in directory {era5_dir}."
            )


    if era5_dir.is_dir():
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*Dask is not installed.*", category=UserWarning
            )
            adapter = ERA5Adapter(val_nc_files)
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*Dask is not installed.*", category=UserWarning
            )
            adapter = ERA5Adapter(era5_dir)

    lat, lon = andean_station.latitud_cne, andean_station.longitud_cne
    var_name = adapter._resolve_variable_name("t2m")
    ds_pt = adapter.extract_point(lat, lon, method="nearest")

    # Aligned with temperature.py lines 1281-1295
    grid_lat = float(ds_pt["latitude"].values)
    grid_lon = float(ds_pt["longitude"].values)
    exp_lat = andean_station.celda_era5["lat"]
    exp_lon = andean_station.celda_era5["lon"]
    if not (
        np.isclose(grid_lat, exp_lat, atol=1e-3)
        and np.isclose(grid_lon, exp_lon, atol=1e-3)
    ):
        raise ValueError(
            f"ERA5 grid cell ({grid_lat:.2f}, {grid_lon:.2f}) does not match "
            f"registered cell ({exp_lat:.2f}, {exp_lon:.2f}) "
            f"for Andean station '{clean_id}'."
        )

    da = ds_pt[var_name]
    if "step" in da.dims:
        da = da.squeeze("step")
    if "surface" in da.dims:
        da = da.squeeze("surface")
    if "number" in da.dims:
        da = da.isel(number=0)

    # Aligned with temperature.py lines 1307-1311
    units = str(da.attrs.get("units", "")).strip()
    if units != "K":
        raise ValueError(
            f"ERA5 variable '{var_name}' must have units 'K' (Kelvin), found '{units}'."
        )

    s_hourly_raw = da.to_series()
    s_hourly_raw.index = pd.DatetimeIndex(s_hourly_raw.index)
    s_hourly_c = s_hourly_raw - 273.15

    df_daily_mod = aggregate_hourly_to_daily_temperature(
        s_hourly_c,
        window_start_utc=window_start_utc,
        min_hours=min_hours,
        lag_days=lag_days,
        convert_to_celsius=False,
    )

    # Slice strictly to target block
    mod_idx = cast(pd.DatetimeIndex, df_daily_mod.index)
    target_mask = (mod_idx.year >= start_year) & (mod_idx.year <= target_end_year)
    full_target_index = pd.date_range(
        f"{start_year}-01-01", f"{target_end_year}-12-31", freq="D"
    )
    df_daily_mod_sliced = df_daily_mod.loc[target_mask].reindex(full_target_index)

    mod_tmax = df_daily_mod_sliced["tmax"]
    mod_tmin = df_daily_mod_sliced["tmin"]

    # Prohibit missing values (ERA5 has complete coverage)
    if mod_tmax.isna().any() or mod_tmin.isna().any():
        raise ValueError(
            f"ERA5 daily validation series contains missing values "
            f"for station {clean_id}."
        )

    _validate_temperature_range(mod_tmax, f"mod_tmax ({clean_id})")
    _validate_temperature_range(mod_tmin, f"mod_tmin ({clean_id})")

    mod_tmax.name = "mod_tmax"
    mod_tmin.name = "mod_tmin"

    return mod_tmax, mod_tmin


def load_validation_temperature_pair(
    station_id: str,
    obs_tmax_path: Path,
    obs_tmin_path: Path,
    era5_target: Path,
    window_start_utc: int,
    lag_days: int,
    min_hours: int,
    opening_record: Dict[str, Any],
    ledger_path: Path,
    registry_path: Optional[Path] = None,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Load validation partition paired series for daily temperature evaluation.

    Protocol Appendix I.5, I.6, G.5.2, G.10:
    - Demands opening record and verifies it matches station_id
      and is present in ledger via exact dictionary equality (C5).
    - Resolves validation partition range strictly from AndeanStation registry.
    - Yields ONLY validation years (never calibration years).
    - Sets observations of validation years not in registry valid_years to NaN.
    - Loads ERA5 hourly NetCDFs for val_start..val_end+1 and aggregates with 05-05 UTC.
    - Enforces physical temperature validity [-60, 60] °C and zero NaNs in ERA5.

    Returns:
        Tuple of (obs_tmax_val, obs_tmin_val, mod_tmax_val, mod_tmin_val).
    """
    clean_id = re.sub(r"\.0$", "", str(station_id).strip())

    # 1. Verify opening authorization record
    if not opening_record or not isinstance(opening_record, dict):
        raise PermissionError(
            "Validation loading blocked: opening_record must be provided."
        )
    rec_station = re.sub(r"\.0$", "", str(opening_record.get("station_id", "")).strip())
    if rec_station != clean_id:
        raise PermissionError(
            f"Validation loading blocked: opening record station '{rec_station}' "
            f"does not match target station '{clean_id}'."
        )

    # 2. Verify exact record presence in append-only ledger via full dict equality (C5)
    if not ledger_path.is_file():
        raise PermissionError(f"Validation ledger not found: {ledger_path}")
    record_found = False
    with open(ledger_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                entry = json.loads(line_str)
            except Exception as err:
                raise ValueError(
                    f"Corrupt ledger line {idx} in {ledger_path}: {line_str}"
                ) from err
            if entry == opening_record:
                record_found = True
                break

    if not record_found:
        raise PermissionError(
            f"Exact opening record for station {clean_id} "
            f"not verified in ledger {ledger_path}."
        )

    # 3. Retrieve metadata and validation bounds from registry
    andean_st = get_andean_station(clean_id, registry_path=registry_path)
    val_start, val_end = andean_st.val_block
    val_years_needed = list(range(val_start, val_end + 2))

    # 4. Load observation extremes using _read_obs_block helper
    obs_tmax_val = _read_obs_block(
        path=obs_tmax_path,
        station_id=clean_id,
        start_year=val_start,
        end_year=val_end,
        valid_years=andean_st.valid_years,
    )
    obs_tmin_val = _read_obs_block(
        path=obs_tmin_path,
        station_id=clean_id,
        start_year=val_start,
        end_year=val_end,
        valid_years=andean_st.valid_years,
    )

    _validate_temperature_range(obs_tmax_val, f"obs_tmax_val ({clean_id})")
    _validate_temperature_range(obs_tmin_val, f"obs_tmin_val ({clean_id})")

    obs_tmax_val.name = "obs_tmax"
    obs_tmin_val.name = "obs_tmin"

    # 5. Load ERA5 validation daily series using _load_era5_daily_block helper
    mod_tmax_val, mod_tmin_val = _load_era5_daily_block(
        era5_dir=era5_target,
        andean_station=andean_st,
        years=val_years_needed,
        window_start_utc=window_start_utc,
        lag_days=lag_days,
        min_hours=min_hours,
    )

    return obs_tmax_val, obs_tmin_val, mod_tmax_val, mod_tmin_val


# ==============================================================================
# 4. Descriptive Continuous Metrics (Section G.9, 3.1)
# ==============================================================================


def compute_daily_mae(obs: pd.Series, mod: pd.Series) -> float:
    """Compute Mean Absolute Error (MAE) between paired daily series."""
    valid_mask = obs.notna() & mod.notna()
    if not valid_mask.any():
        return float("nan")
    return float(
        np.mean(np.abs(mod[valid_mask].to_numpy() - obs[valid_mask].to_numpy()))
    )


def compute_daily_rmse(obs: pd.Series, mod: pd.Series) -> float:
    """Compute Root Mean Squared Error (RMSE) between paired daily series."""
    valid_mask = obs.notna() & mod.notna()
    if not valid_mask.any():
        return float("nan")
    diff = mod[valid_mask].to_numpy() - obs[valid_mask].to_numpy()
    return float(np.sqrt(np.mean(diff**2)))


def compute_monthly_biases(
    obs: pd.Series, mod: pd.Series
) -> Dict[str, Optional[float]]:
    """Compute 12-month calendar mean biases: Bias_m = mean(Mod) - mean(Obs)
    (Section 3.1).

    Returns mapping of month string ('1'..'12') to mean bias in °C.
    """
    valid_mask = obs.notna() & mod.notna()
    clean_obs = obs[valid_mask]
    clean_mod = mod[valid_mask]
    if clean_obs.empty:
        return {str(m): None for m in range(1, 13)}

    res: Dict[str, Optional[float]] = {}
    clean_obs_idx = cast(pd.DatetimeIndex, clean_obs.index)
    for m in range(1, 13):
        m_mask = clean_obs_idx.month == m
        if m_mask.any():
            b = float(clean_mod[m_mask].mean() - clean_obs[m_mask].mean())
            res[str(m)] = b
        else:
            res[str(m)] = None
    return res


# ==============================================================================
# 5. Paired Annual Block Bootstrap Progression (Section 6.4, Appendix I.8)
# ==============================================================================


PROGRESSION_METRICS: Tuple[str, ...] = (
    "max_monthly_bias",
    "dtr_mean_bias",
    "extreme_percentiles_bias",
    "delta_p5_tmin",
    "delta_p95_tmax",
)

PRIMARY_DISCRIMINATING_FAMILY_TEMP: Tuple[str, ...] = (
    "dtr_mean_bias",
    "delta_p5_tmin",
    "delta_p95_tmax",
)

DECISIVE_METRICS_TEMP: Tuple[str, ...] = (
    "max_monthly_bias",
    "dtr_mean_bias",
    "extreme_percentiles_bias",
)


def compute_paired_annual_block_bootstrap_progression(
    tmax_obs: pd.Series,
    tmin_obs: pd.Series,
    tmax_mod_base: pd.Series,
    tmin_mod_base: pd.Series,
    tmax_mod_cand: pd.Series,
    tmin_mod_cand: pd.Series,
    n_replicas: int = PHASE3_BOOTSTRAP_REPLICAS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Dict[str, BootstrapConfidenceInterval]:
    """Compute 95% CIs for Delta Error = |Error_base| - |Error_cand|
    via paired annual bootstrap.

    Protocol Section 6.4 & Appendix I.8:
    - Every replica uses the identical resampled years for
      baseline and candidate methods.
    - Synchronous random number generation matching
      compute_annual_block_bootstrap_temperature.
    - If valid years < MIN_BOOTSTRAP_BLOCKS (7), returns NaN intervals
      with insufficient_blocks=True.
    - Discards and counts non-finite replicas.
    """
    df = pd.DataFrame(
        {
            "tmax_obs": tmax_obs,
            "tmin_obs": tmin_obs,
            "tmax_base": tmax_mod_base,
            "tmin_base": tmin_mod_base,
            "tmax_cand": tmax_mod_cand,
            "tmin_cand": tmin_mod_cand,
        }
    ).sort_index()

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be a pd.DatetimeIndex.")

    # Identify calendar years with valid observations for both extremes
    valid_years = np.array(
        [
            y
            for y in sorted(df.index.year.unique())
            if (
                df.loc[df.index.year == y, "tmax_obs"].notna().any()
                and df.loc[df.index.year == y, "tmin_obs"].notna().any()
            )
        ]
    )
    n_years = len(valid_years)

    # Point estimates on the complete un-resampled series
    pt_base = compute_temperature_decisive_metrics(
        df["tmax_obs"], df["tmax_base"], df["tmin_obs"], df["tmin_base"]
    )
    pt_cand = compute_temperature_decisive_metrics(
        df["tmax_obs"], df["tmax_cand"], df["tmin_obs"], df["tmin_cand"]
    )

    point_deltas = {
        q: float(abs(pt_base[q]) - abs(pt_cand[q])) for q in PROGRESSION_METRICS
    }

    if n_years < MIN_BOOTSTRAP_BLOCKS:
        return {
            q: BootstrapConfidenceInterval(
                point_estimate=point_deltas[q],
                ci_lower=float("nan"),
                ci_upper=float("nan"),
                n_replicas=0,
                seed=seed,
                n_discarded_replicas=0,
                insufficient_blocks=True,
            )
            for q in PROGRESSION_METRICS
        }

    year_dfs = {y: df[df.index.year == y] for y in valid_years}
    rng = np.random.default_rng(seed)
    sampled_deltas: Dict[str, List[float]] = {q: [] for q in PROGRESSION_METRICS}
    discarded_counts: Dict[str, int] = {q: 0 for q in PROGRESSION_METRICS}

    for _ in range(n_replicas):
        resampled_years = rng.choice(valid_years, size=n_years, replace=True)
        sampled_df = pd.concat([year_dfs[y] for y in resampled_years], axis=0)

        rep_base = compute_temperature_decisive_metrics(
            sampled_df["tmax_obs"],
            sampled_df["tmax_base"],
            sampled_df["tmin_obs"],
            sampled_df["tmin_base"],
        )
        rep_cand = compute_temperature_decisive_metrics(
            sampled_df["tmax_obs"],
            sampled_df["tmax_cand"],
            sampled_df["tmin_obs"],
            sampled_df["tmin_cand"],
        )

        for q in PROGRESSION_METRICS:
            v_b = rep_base.get(q)
            v_c = rep_cand.get(q)
            if (
                v_b is not None
                and v_c is not None
                and np.isfinite(v_b)
                and np.isfinite(v_c)
            ):
                sampled_deltas[q].append(float(abs(v_b) - abs(v_c)))
            else:
                discarded_counts[q] += 1

    result: Dict[str, BootstrapConfidenceInterval] = {}
    for q in PROGRESSION_METRICS:
        vals = sampled_deltas[q]
        if len(vals) == 0:
            result[q] = BootstrapConfidenceInterval(
                point_estimate=point_deltas[q],
                ci_lower=float("nan"),
                ci_upper=float("nan"),
                n_replicas=0,
                seed=seed,
                n_discarded_replicas=discarded_counts[q],
                insufficient_blocks=True,
            )
        else:
            low = float(np.percentile(vals, 2.5))
            high = float(np.percentile(vals, 97.5))
            result[q] = BootstrapConfidenceInterval(
                point_estimate=point_deltas[q],
                ci_lower=low,
                ci_upper=high,
                n_replicas=len(vals),
                seed=seed,
                n_discarded_replicas=discarded_counts[q],
                insufficient_blocks=False,
            )
    return result


def evaluate_progression_decision(
    cis: Dict[str, BootstrapConfidenceInterval],
) -> Dict[str, Any]:
    """Apply the dual progression rule (Section 6.4, Appendix I.8).

    Promotion requires:
    1. At least one metric in the Primary Discriminating Family has CI lower bound > 0.
    2. None of the decisive metrics has CI upper bound < 0 (no significant regression).
    """
    primary_passes = []
    for q in PRIMARY_DISCRIMINATING_FAMILY_TEMP:
        ci = cis[q]
        if (
            not ci.insufficient_blocks
            and np.isfinite(ci.ci_lower)
            and ci.ci_lower > 0.0
        ):
            primary_passes.append(q)

    decisive_regressions = []
    for q in DECISIVE_METRICS_TEMP:
        ci = cis[q]
        if (
            not ci.insufficient_blocks
            and np.isfinite(ci.ci_upper)
            and ci.ci_upper < 0.0
        ):
            decisive_regressions.append(q)

    promoted = bool(len(primary_passes) > 0 and len(decisive_regressions) == 0)

    return {
        "promoted": promoted,
        "primary_passes": primary_passes,
        "decisive_regressions": decisive_regressions,
        "cis": {k: dataclasses.asdict(v) for k, v in cis.items()},
    }


# ==============================================================================
# 6. Serialization and Hygiene Helpers (Appendix I.9)
# ==============================================================================


def sanitize_for_json(obj: Any) -> Any:
    """Recursively sanitize data structures for RFC 8259 JSON compliance.

    Converts NaN, +inf, -inf to None.
    Preserves exact bool types before integer casting.
    """
    if obj is None:
        return None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (str, bytes)):
        return str(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_for_json(x) for x in obj]
    if isinstance(obj, pd.Series):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return sanitize_for_json(dataclasses.asdict(obj))
    return obj


def write_json_deterministic(path: Path, data: Any) -> None:
    """Write sanitized JSON deterministically with sorted keys, LF,
    and atomic replace.
    """
    sanitized = sanitize_for_json(data)
    json_text = json.dumps(sanitized, indent=2, sort_keys=True, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f".tmp_{os.getpid()}_{uuid.uuid4().hex[:8]}")
    with open(temp_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json_text + "\n")
    os.replace(temp_path, path)


# ==============================================================================
# 7. Station Evaluation and Orchestrator (Appendix I.5, I.7, I.8, I.9)
# ==============================================================================


def evaluate_single_station(
    station_id: str,
    fitted_info: Dict[str, Any],
    obs_tmax_val: pd.Series,
    obs_tmin_val: pd.Series,
    mod_tmax_val: pd.Series,
    mod_tmin_val: pd.Series,
    opening_record: Dict[str, Any],
    evaluator_commit: str,
    tag_target: str,
    run_kind: str,
    interrupted_formal_opening: bool,
    input_hashes: Dict[str, Any],
    bootstrap_replicas: int = PHASE3_BOOTSTRAP_REPLICAS,
) -> Dict[str, Any]:
    """Execute complete Phase 3 evaluation for a single authorized station.

    Evaluates L0, L1, L2, L3-winner, L3-constant (I.7), progression (I.8),
    descriptive metrics (G.9), and crossing diagnostics (Section 3.5).

    Crossing Rule Diagnostics:
        Evaluates physical consistency rule Tmin <= Tmax.
        The denominator for crossing fraction is the number of valid paired days
        (days where both Tmax and Tmin are non-NaN).
    """
    m1: Level1MonthlyAdditiveTemperature = fitted_info["level1"]
    m2: Level2MeanStateDTRScalingTemperature = fitted_info["level2"]
    m3_win: Level3QuantileMappingTemperature = fitted_info["level3_winner"]
    m3_const: Level3QuantileMappingTemperature = fitted_info["level3_constant"]
    identical_to_winner: bool = fitted_info["identical_to_winner"]

    # 1. Level 0: Raw ERA5 daily extremes (no crossing rule applied)
    l0_tmax = mod_tmax_val
    l0_tmin = mod_tmin_val
    l0_gate = check_temperature_physical_gate(l0_tmax, l0_tmin)
    valid_l0 = l0_tmax.notna() & l0_tmin.notna()
    n_cr_l0 = int((valid_l0 & (l0_tmin > l0_tmax)).sum())
    n_days_l0 = int(valid_l0.sum())
    frac_cr_l0 = float(n_cr_l0 / n_days_l0) if n_days_l0 > 0 else 0.0
    l0_diag = CrossingDiagnostic(
        n_crossings=n_cr_l0,
        n_total_days=n_days_l0,
        fraction_crossings=frac_cr_l0,
        alert_triggered=bool(frac_cr_l0 > 0.01),
    )
    l0_boot = compute_temperature_decisive_metrics_bootstrap(
        obs_tmax_val, l0_tmax, obs_tmin_val, l0_tmin, n_replicas=bootstrap_replicas
    )
    l0_scorecard = evaluate_temperature_scorecard(
        station_id=station_id,
        level="level0",
        boot_results=l0_boot,
        physical_gate_passed=l0_gate,
        crossing_diag=l0_diag,
    )

    # 2. Level 1: Monthly Additive
    l1_tmax, l1_tmin = m1.transform(mod_tmax_val, mod_tmin_val)
    l1_gate = check_temperature_physical_gate(l1_tmax, l1_tmin)
    l1_diag = m1.last_crossing_diagnostic_
    l1_boot = compute_temperature_decisive_metrics_bootstrap(
        obs_tmax_val, l1_tmax, obs_tmin_val, l1_tmin, n_replicas=bootstrap_replicas
    )
    l1_scorecard = evaluate_temperature_scorecard(
        station_id=station_id,
        level="level1",
        boot_results=l1_boot,
        physical_gate_passed=l1_gate,
        crossing_diag=l1_diag,
    )

    # 3. Level 2: Coupled Mean State + DTR Scaling
    l2_tmax, l2_tmin = m2.transform(mod_tmax_val, mod_tmin_val)
    l2_gate = check_temperature_physical_gate(l2_tmax, l2_tmin)
    l2_diag = m2.last_crossing_diagnostic_
    l2_boot = compute_temperature_decisive_metrics_bootstrap(
        obs_tmax_val, l2_tmax, obs_tmin_val, l2_tmin, n_replicas=bootstrap_replicas
    )
    l2_scorecard = evaluate_temperature_scorecard(
        station_id=station_id,
        level="level2",
        boot_results=l2_boot,
        physical_gate_passed=l2_gate,
        crossing_diag=l2_diag,
    )

    # 4. Level 3 Winner
    l3w_tmax, l3w_tmin = m3_win.transform(mod_tmax_val, mod_tmin_val)
    l3w_gate = check_temperature_physical_gate(l3w_tmax, l3w_tmin)
    l3w_diag = m3_win.last_crossing_diagnostic_
    l3w_boot = compute_temperature_decisive_metrics_bootstrap(
        obs_tmax_val, l3w_tmax, obs_tmin_val, l3w_tmin, n_replicas=bootstrap_replicas
    )
    l3w_scorecard = evaluate_temperature_scorecard(
        station_id=station_id,
        level="level3_winner",
        boot_results=l3w_boot,
        physical_gate_passed=l3w_gate,
        crossing_diag=l3w_diag,
    )

    # 5. Level 3 Constant Sensitivity
    if identical_to_winner:
        l3c_tmax, l3c_tmin = l3w_tmax, l3w_tmin
        l3c_gate = l3w_gate
        l3c_diag = l3w_diag
        l3c_boot = l3w_boot
        l3c_scorecard = evaluate_temperature_scorecard(
            station_id=station_id,
            level="level3_constant",
            boot_results=l3c_boot,
            physical_gate_passed=l3c_gate,
            crossing_diag=l3c_diag,
        )
    else:
        l3c_tmax, l3c_tmin = m3_const.transform(mod_tmax_val, mod_tmin_val)
        l3c_gate = check_temperature_physical_gate(l3c_tmax, l3c_tmin)
        l3c_diag = m3_const.last_crossing_diagnostic_
        l3c_boot = compute_temperature_decisive_metrics_bootstrap(
            obs_tmax_val,
            l3c_tmax,
            obs_tmin_val,
            l3c_tmin,
            n_replicas=bootstrap_replicas,
        )
        l3c_scorecard = evaluate_temperature_scorecard(
            station_id=station_id,
            level="level3_constant",
            boot_results=l3c_boot,
            physical_gate_passed=l3c_gate,
            crossing_diag=l3c_diag,
        )

    # Descriptive continuous metrics (Section G.9)
    def _calc_descriptives(m_tmax: pd.Series, m_tmin: pd.Series) -> Dict[str, Any]:
        return {
            "mae_tmax": compute_daily_mae(obs_tmax_val, m_tmax),
            "mae_tmin": compute_daily_mae(obs_tmin_val, m_tmin),
            "rmse_tmax": compute_daily_rmse(obs_tmax_val, m_tmax),
            "rmse_tmin": compute_daily_rmse(obs_tmin_val, m_tmin),
            "pearson_r_tmax": compute_daily_pearson(obs_tmax_val, m_tmax),
            "pearson_r_tmin": compute_daily_pearson(obs_tmin_val, m_tmin),
            "monthly_biases_tmax": compute_monthly_biases(obs_tmax_val, m_tmax),
            "monthly_biases_tmin": compute_monthly_biases(obs_tmin_val, m_tmin),
        }

    descriptives = {
        "level0": _calc_descriptives(l0_tmax, l0_tmin),
        "level1": _calc_descriptives(l1_tmax, l1_tmin),
        "level2": _calc_descriptives(l2_tmax, l2_tmin),
        "level3_winner": _calc_descriptives(l3w_tmax, l3w_tmin),
        "level3_constant": (
            _calc_descriptives(l3w_tmax, l3w_tmin)
            if identical_to_winner
            else _calc_descriptives(l3c_tmax, l3c_tmin)
        ),
    }

    # Progression analysis (I.8, Section 6.4): chain 0 -> 1 -> 2 -> 3
    prog_0_1 = compute_paired_annual_block_bootstrap_progression(
        obs_tmax_val,
        obs_tmin_val,
        l0_tmax,
        l0_tmin,
        l1_tmax,
        l1_tmin,
        n_replicas=bootstrap_replicas,
    )
    prog_1_2 = compute_paired_annual_block_bootstrap_progression(
        obs_tmax_val,
        obs_tmin_val,
        l1_tmax,
        l1_tmin,
        l2_tmax,
        l2_tmin,
        n_replicas=bootstrap_replicas,
    )
    prog_2_3 = compute_paired_annual_block_bootstrap_progression(
        obs_tmax_val,
        obs_tmin_val,
        l2_tmax,
        l2_tmin,
        l3w_tmax,
        l3w_tmin,
        n_replicas=bootstrap_replicas,
    )

    progression_report = {
        "0_to_1": evaluate_progression_decision(prog_0_1),
        "1_to_2": evaluate_progression_decision(prog_1_2),
        "2_to_3": evaluate_progression_decision(prog_2_3),
    }

    crossing_diagnostics = {
        "level0": dataclasses.asdict(l0_diag),
        "level1": dataclasses.asdict(l1_diag) if l1_diag else None,
        "level2": dataclasses.asdict(l2_diag) if l2_diag else None,
        "level3_winner": dataclasses.asdict(l3w_diag) if l3w_diag else None,
        "level3_constant": dataclasses.asdict(l3c_diag) if l3c_diag else None,
    }

    st_name = str(
        fitted_info.get("cal_report_data", {}).get(
            "station_name", f"Station {station_id}"
        )
    )

    st_files_meta = fitted_info.get("station_files", {})
    val_blk = list(st_files_meta.get("val_block", []))
    v_years_set = set(st_files_meta.get("valid_years", ()))
    if len(val_blk) == 2:
        val_years_used = sorted(
            list(set(range(val_blk[0], val_blk[1] + 1)) & v_years_set)
        )
    else:
        val_years_used = []

    valid_paired_days_tmax = int((obs_tmax_val.notna() & mod_tmax_val.notna()).sum())
    valid_paired_days_tmin = int((obs_tmin_val.notna() & mod_tmin_val.notna()).sum())

    return {
        "station_id": station_id,
        "station_name": st_name,
        "run_kind": run_kind,
        "interrupted_formal_opening": interrupted_formal_opening,
        "evaluator_commit": evaluator_commit,
        "tag_target": tag_target,
        "versions": {
            "python": sys.version,
            "numpy": str(np.__version__),
            "pandas": str(pd.__version__),
        },
        "input_hashes": input_hashes,
        "winner_config": fitted_info["winner_config"],
        "identical_to_winner": identical_to_winner,
        "level3_constant_identical_to_winner": identical_to_winner,
        "validation_block": val_blk,
        "valid_years_used": val_years_used,
        "valid_paired_days": {
            "tmax": valid_paired_days_tmax,
            "tmin": valid_paired_days_tmin,
        },
        "lag_sensitivity": "not_activated",
        "ledger_record": opening_record,
        "scorecards": {
            "level0": dataclasses.asdict(l0_scorecard),
            "level1": dataclasses.asdict(l1_scorecard),
            "level2": dataclasses.asdict(l2_scorecard),
            "level3_winner": dataclasses.asdict(l3w_scorecard),
            "level3_constant": dataclasses.asdict(l3c_scorecard),
        },
        "descriptive_metrics": descriptives,
        "progression": progression_report,
        "crossing_diagnostics": crossing_diagnostics,
        "disclosure_csv_parsing": (
            "Observation CSV files contain both calibration and validation blocks in "
            "a single file; the frozen calibration loader and _read_obs_block parse "
            "the entire file and slice by block prior to any use. No validation "
            "values are used prior to ledger recording."
        ),
    }


def compute_temperature_decisive_metrics_bootstrap(
    obs_tmax: pd.Series,
    mod_tmax: pd.Series,
    obs_tmin: pd.Series,
    mod_tmin: pd.Series,
    n_replicas: int = PHASE3_BOOTSTRAP_REPLICAS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Dict[str, BootstrapConfidenceInterval]:
    """Helper wrapper around protocol.compute_annual_block_bootstrap_temperature."""
    from tropicor.core.protocol import compute_annual_block_bootstrap_temperature

    return compute_annual_block_bootstrap_temperature(
        tmax_obs=obs_tmax,
        tmax_mod=mod_tmax,
        tmin_obs=obs_tmin,
        tmin_mod=mod_tmin,
        n_replicas=n_replicas,
        seed=seed,
    )


def read_ledger_records(ledger_path: Path) -> List[Dict[str, Any]]:
    """Read all entries from audit ledger, validating JSON structure."""
    if not ledger_path.is_file():
        return []
    records: List[Dict[str, Any]] = []
    with open(ledger_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except Exception as err:
                raise ValueError(
                    f"Corrupt ledger line {idx} in {ledger_path}: {s}"
                ) from err
            if (
                not isinstance(rec, dict)
                or "station_id" not in rec
                or "commit_hash" not in rec
                or "variable" not in rec
                or ("timestamp" not in rec and "timestamp_utc" not in rec)
            ):
                raise ValueError(
                    f"Corrupt ledger record at line {idx} in {ledger_path}: {s}"
                )
            records.append(rec)
    return records


def run_phase3_evaluation(
    repo_root: Path,
    run_kind: str = "formal",
    open_validation: bool = False,
    ledger_path: Optional[Path] = None,
    reports_dir: Optional[Path] = None,
    expected_inputs_path: Optional[Path] = None,
    registry_path: Optional[Path] = None,
    era5_base_dir: Optional[Path] = None,
    station_ids: Sequence[str] = ANDEAN_PHASE3_STATION_IDS,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Execute complete Phase 3 Andean Temperature Evaluation (Protocol v1.7).

    Follows the 3-phase opening protocol (Appendix I.5):
    - Phase 0 (Read-only): reads the audit ledger and validates the status of all
      5 stations before writing any record. Formal mode requires zero prior
      temperature records; reproduction mode requires prior records matching HEAD.
    - Phase A (Write-ahead): verifies git commit against verified HEAD and writes
      all 5 opening records via validate_validation_access before accessing validation.
    - Phase B (Evaluation): sequentially loads validation paired series, evaluates
      scorecards, and writes reports per station.

    Args:
        repo_root: Path to repository root.
        run_kind: 'formal' or 'reproduction'.
        open_validation: Mandatory boolean (must be True for non-dry-run).
        ledger_path: Audit ledger path (must resolve to canonical path in non-dry-run).
        reports_dir: Output directory (must resolve to canonical path in non-dry-run).
        expected_inputs_path: Path to phase3_expected_inputs.json.
        registry_path: Path to andean_station_registry.json.
        era5_base_dir: Base directory of station ERA5 folders.
        station_ids: 5 Andean stations to process.
        dry_run: If True, executes pre-opening verification and calibration fitting
                 without ledger opening or validation reading.

    Returns:
        Summary dictionary of the evaluation.
    """
    repo_root = repo_root.resolve()
    canonical_ledger = (repo_root / DEFAULT_LEDGER_PATH).resolve()
    canonical_reports = (repo_root / "local" / "reports" / "phase3").resolve()

    if not dry_run:
        if ledger_path is not None and ledger_path.resolve() != canonical_ledger:
            raise ValueError(
                f"Non-canonical ledger path '{ledger_path.resolve()}' "
                f"forbidden when dry_run=False. "
                f"Must be exactly canonical path '{canonical_ledger}'."
            )
        if reports_dir is not None and reports_dir.resolve() != canonical_reports:
            raise ValueError(
                f"Non-canonical reports directory '{reports_dir.resolve()}' "
                f"forbidden when dry_run=False. "
                f"Must be exactly canonical path '{canonical_reports}'."
            )
        resolved_ledger = canonical_ledger
        base_reports = canonical_reports
    else:
        resolved_ledger = (ledger_path or canonical_ledger).resolve()
        base_reports = (reports_dir or canonical_reports).resolve()

    # In non-dry-run mode, execute Phase 0 before calibration fitting (I.5)
    if not dry_run:
        if not open_validation:
            raise PermissionError(
                "Formal evaluation requires open_validation=True. "
                "Use dry-run / --preflight to verify inputs without opening."
            )

        # ----------------------------------------------------------------------
        # Phase 0: Read ledger (if exists) and validate all 5 stations without writing
        # ----------------------------------------------------------------------
        existing_records = read_ledger_records(resolved_ledger)

        if run_kind == "formal":
            for st in station_ids:
                clean_st = re.sub(r"\.0$", "", str(st).strip())
                st_records = [
                    r
                    for r in existing_records
                    if r.get("station_id") == clean_st
                    and r.get("variable") == "temperature"
                ]
                if len(st_records) > 0:
                    raise PermissionError(
                        f"Station {clean_st} has already been opened in ledger "
                        f"{resolved_ledger}. Protocol mandates exactly one formal "
                        f"opening per station. "
                        "Use --reproduce for reproduction runs."
                    )
        elif run_kind == "reproduction":
            current_head = get_current_git_commit()
            for st in station_ids:
                clean_st = re.sub(r"\.0$", "", str(st).strip())
                st_records = [
                    r
                    for r in existing_records
                    if r.get("station_id") == clean_st
                    and r.get("variable") == "temperature"
                ]
                if len(st_records) == 0:
                    raise ValueError(
                        f"Reproduction run for station {clean_st} requested, "
                        f"but no prior opening record exists in {resolved_ledger}."
                    )
                formal_records = [
                    r
                    for r in st_records
                    if r.get("config", {}).get("run_kind") == "formal"
                ]
                prior_record = (
                    formal_records[-1] if len(formal_records) > 0 else st_records[-1]
                )
                prior_commit = prior_record.get("commit_hash", "")
                if current_head != prior_commit:
                    raise ValueError(
                        f"Reproduction run for station {clean_st} requires git commit "
                        f"'{prior_commit}' matching prior formal opening, but HEAD is "
                        f"'{current_head}'."
                    )
        else:
            raise ValueError(
                f"Unknown run_kind '{run_kind}'. Expected 'formal' or 'reproduction'."
            )

    # 1. Pre-Opening Verification and Single-Pass Fitting (I.3, I.4)
    pre_report, fitted_models = verify_pre_opening(
        repo_root=repo_root,
        expected_inputs_path=expected_inputs_path,
        registry_path=registry_path,
        era5_base_dir=era5_base_dir,
        ledger_path=resolved_ledger,
        station_ids=station_ids,
        dry_run=dry_run,
    )

    if dry_run:
        return {
            "mode": "dry_run",
            "pre_opening_checks": dataclasses.asdict(pre_report),
            "n_fitted_stations": len(fitted_models),
            "status": "PASS" if pre_report.all_checks_passed else "FAIL",
        }

    # --------------------------------------------------------------------------
    # Phase A: Write-ahead authorization records across all 5 stations
    # (Verifies get_current_git_commit() == pre_report.git_commit before writing;
    #  writes all 5 records WITHOUT loading or reading ANY validation data)
    # --------------------------------------------------------------------------
    if get_current_git_commit() != pre_report.git_commit:
        raise RuntimeError(
            "Git commit mismatch before calling validate_validation_access. "
            "Aborting without writing."
        )

    opening_records: Dict[str, Dict[str, Any]] = {}
    for st in station_ids:
        clean_st = re.sub(r"\.0$", "", str(st).strip())
        cal_sha256 = pre_report.input_file_hashes["calibration"][clean_st]
        ledger_config = {
            "evaluator_commit": pre_report.git_commit,
            "run_kind": run_kind,
            "calibration_report_sha256": cal_sha256,
            "methods": [
                "level0",
                "level1",
                "level2",
                "level3_winner",
                "level3_constant",
            ],
            "bootstrap": {
                "replicas": PHASE3_BOOTSTRAP_REPLICAS,
                "seed": 42,
                "block": "annual",
            },
        }

        rec = validate_validation_access(
            open_validation=True,
            purpose=(
                f"Phase 3 Andean temperature evaluation for station "
                f"{clean_st} ({run_kind})"
            ),
            station_id=clean_st,
            variable="temperature",
            seed=42,
            config=ledger_config,
            ledger_path=resolved_ledger,
        )
        opening_records[clean_st] = rec

    # --------------------------------------------------------------------------
    # Phase B: Out-of-sample evaluation per station in sequence
    # (Validation data loading, model evaluation, and scorecard generation.
    #  Note: Stations are evaluated independently; there is no cross-station logic.)
    # --------------------------------------------------------------------------
    station_reports: Dict[str, Any] = {}
    success_counts_by_level: Dict[str, int] = {
        "level0": 0,
        "level1": 0,
        "level2": 0,
        "level3_winner": 0,
        "level3_constant": 0,
    }
    station_summaries: Dict[str, Any] = {}

    for st in station_ids:
        clean_st = re.sub(r"\.0$", "", str(st).strip())
        opening_record = opening_records[clean_st]
        formal_scorecard_path = (
            base_reports / clean_st / f"phase3_scorecard_{clean_st}.json"
        )
        interrupted_opening = bool(
            run_kind == "reproduction" and not formal_scorecard_path.is_file()
        )

        st_fitted = fitted_models[clean_st]
        st_files = st_fitted["station_files"]

        # Load validation paired extremes strictly after Phase A
        obs_tmax_val, obs_tmin_val, mod_tmax_val, mod_tmin_val = (
            load_validation_temperature_pair(
                station_id=clean_st,
                obs_tmax_path=st_files["tmax_csv"],
                obs_tmin_path=st_files["tmin_csv"],
                era5_target=st_files["era5_dir"],
                window_start_utc=5,
                lag_days=0,
                min_hours=24,
                opening_record=opening_record,
                ledger_path=resolved_ledger,
                registry_path=registry_path,
            )
        )

        st_hashes = {
            "registry": pre_report.input_file_hashes["registry"],
            "lag_audit": pre_report.input_file_hashes["lag_audit"],
            "calibration_report": pre_report.input_file_hashes["calibration"][clean_st],
            "obs_tmax_csv": st_files["tmax_sha256"],
            "obs_tmin_csv": st_files["tmin_sha256"],
            "era5_calibration_files": st_files["era5_cal_hashes"],
            "era5_validation_files": st_files["era5_val_hashes"],
        }

        st_eval_result = evaluate_single_station(
            station_id=clean_st,
            fitted_info=st_fitted,
            obs_tmax_val=obs_tmax_val,
            obs_tmin_val=obs_tmin_val,
            mod_tmax_val=mod_tmax_val,
            mod_tmin_val=mod_tmin_val,
            opening_record=opening_record,
            evaluator_commit=pre_report.git_commit,
            tag_target=pre_report.tag_target_commit,
            run_kind=run_kind,
            interrupted_formal_opening=interrupted_opening,
            input_hashes=st_hashes,
            bootstrap_replicas=PHASE3_BOOTSTRAP_REPLICAS,
        )

        station_reports[clean_st] = st_eval_result

        # Determine target output path
        if run_kind == "formal":
            out_file = base_reports / clean_st / f"phase3_scorecard_{clean_st}.json"
        else:
            out_file = (
                base_reports
                / clean_st
                / "reproduction"
                / f"phase3_scorecard_{clean_st}.json"
            )

        write_json_deterministic(out_file, st_eval_result)

        # Track success counts
        scorecards = st_eval_result["scorecards"]
        st_outcomes: Dict[str, str] = {}
        for lvl in ("level0", "level1", "level2", "level3_winner", "level3_constant"):
            sc = scorecards[lvl]
            out_val = sc["global_outcome"]
            st_outcomes[lvl] = out_val
            if out_val == GlobalOutcome.SUCCESS.value:
                success_counts_by_level[lvl] += 1

        prog = st_eval_result["progression"]
        station_summaries[clean_st] = {
            "station_name": st_eval_result["station_name"],
            "identical_to_winner": st_eval_result["identical_to_winner"],
            "validation_block": st_eval_result["validation_block"],
            "valid_years_used": st_eval_result["valid_years_used"],
            "valid_paired_days": st_eval_result["valid_paired_days"],
            "outcomes": st_outcomes,
            "promotions": {
                "0_to_1": prog["0_to_1"]["promoted"],
                "1_to_2": prog["1_to_2"]["promoted"],
                "2_to_3": prog["2_to_3"]["promoted"],
            },
        }

    # Summary report (I.9, G.3)
    summary_report = {
        "report_type": "Phase 3 Andean Temperature Summary",
        "protocol_version": "1.7",
        "evaluator_commit": pre_report.git_commit,
        "tag_target": pre_report.tag_target_commit,
        "run_kind": run_kind,
        "station_ids": list(station_ids),
        "identical_to_winner": {
            st: station_summaries[st]["identical_to_winner"]
            for st in station_ids
            if st in station_summaries
        },
        "success_counts_by_level": success_counts_by_level,
        "station_summaries": station_summaries,
        "governance_disclosure": (
            "No global pass/fail verdict, composite acceptance threshold, or synthetic "
            "transferability claim is applied to the Andean generalization "
            "set as a whole "
            "(Protocol Section G.3 / I.7.3)."
        ),
        "independence_note": (
            "Level 3 Constant extrapolation is identical to the winning Level 3 "
            "configuration for stations whose winner uses constant extrapolation; "
            "for those stations, Level 3 Constant does not constitute an "
            "independent method."
        ),
        "disclosure_csv_parsing": (
            "Observation CSV files contain both calibration and validation blocks in "
            "a single file; the frozen calibration loader and _read_obs_block parse "
            "the entire file and slice by block prior to any use. No validation "
            "values are used prior to ledger recording."
        ),
    }

    if run_kind == "formal":
        summary_path = base_reports / "phase3_summary.json"
    else:
        summary_path = base_reports / "reproduction" / "phase3_summary.json"

    write_json_deterministic(summary_path, summary_report)

    return summary_report
