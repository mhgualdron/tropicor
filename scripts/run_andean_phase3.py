#!/usr/bin/env python3
"""CLI runner for Andean Phase 3 Daily Temperature Evaluation (Protocol v1.7, App I).

Supported modes:
- --preflight: Dry-run verification (I.9). Executes pre-opening checks, calibration
  raw bias validation, calibration model fitting, and pipeline smoke test.
  Checks 1 and 2 are reported PASS/FAIL without aborting. Checks 3-6, raw bias,
  pipeline smoke, and all 5 station fittings must pass. Writes no ledger record
  and reads no validation observations. Exits non-zero if checks 3-6 or fittings fail.
- Formal mode (--open-validation): Mandatory write-ahead opening and out-of-sample evaluation
  across all 5 stations in a single invocation. Path overrides are strictly forbidden.
- --reproduce: Reproduction run for previously opened stations at identical commit.
  Path overrides are strictly forbidden.

No station selector is supported (all 5 Andean generalization stations are evaluated together).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add repository root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tropicor.core.phase3_evaluator import (  # noqa: E402
    ANDEAN_PHASE3_STATION_IDS,
    run_phase3_evaluation,
    verify_pre_opening,
)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the Phase 3 evaluator.

    Enforces Candado C1a: --open-validation and --reproduce forbid
    path override flags (--ledger-path, --reports-dir, --era5-base-dir, --repo-root).
    """
    parser = argparse.ArgumentParser(
        description="Phase 3 Evaluator for Andean Batch 1 Temperature Opening (Protocol v1.7)."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--preflight",
        action="store_true",
        help="Execute pre-opening verification (I.3) and calibration fitting (I.4) dry-run.",
    )
    group.add_argument(
        "--open-validation",
        action="store_true",
        help="Authorize formal validation opening and execute protocol evaluation.",
    )
    group.add_argument(
        "--reproduce",
        action="store_true",
        help="Execute labeled reproduction run for previously opened stations.",
    )

    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root directory (allowed in --preflight only).",
    )
    parser.add_argument(
        "--ledger-path",
        type=Path,
        default=None,
        help="Path to audit ledger (allowed in --preflight only).",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help="Path to output directory (allowed in --preflight only).",
    )
    parser.add_argument(
        "--era5-base-dir",
        type=Path,
        default=None,
        help="Base directory containing station timeseries NetCDF/CSV files (allowed in --preflight only).",
    )

    parsed = parser.parse_args(args)

    # C1a: If formal or reproduction, reject any path overrides
    if (parsed.open_validation or parsed.reproduce) and any(
        x is not None
        for x in (
            parsed.repo_root,
            parsed.ledger_path,
            parsed.reports_dir,
            parsed.era5_base_dir,
        )
    ):
        parser.error(
            "Formal (--open-validation) and reproduction (--reproduce) modes do not "
            "accept path override flags (--ledger-path, --reports-dir, --era5-base-dir, --repo-root)."
        )

    return parsed


def run_preflight_cli(
    repo_root: Path,
    era5_base_dir: Path | None = None,
    ledger_path: Path | None = None,
) -> int:
    """Execute preflight dry-run and print detailed check outputs."""
    print("=" * 80)
    print(
        "PHASE 3 ANDEAN TEMPERATURE EVALUATOR — PREFLIGHT VERIFICATION (Appendix I.3, I.4)"
    )
    print("=" * 80)
    print(f"Repository Root: {repo_root}")
    print(f"Target Stations (n=5): {', '.join(ANDEAN_PHASE3_STATION_IDS)}")
    print("-" * 80)

    try:
        report, fitted_models = verify_pre_opening(
            repo_root=repo_root,
            era5_base_dir=era5_base_dir,
            ledger_path=ledger_path,
            station_ids=ANDEAN_PHASE3_STATION_IDS,
            dry_run=True,
        )
    except Exception as err:
        print(f"\n[ABORT] Preflight verification encountered fatal error: {err}")
        return 1

    # Report Check 1
    chk1_status = "PASS" if report.git_clean else "FAIL"
    print(f"Check 1 (Working tree clean & Git HEAD): [{chk1_status}]")
    print(f"  - Clean working tree: {report.git_clean}")
    print(f"  - Verified Git HEAD commit: {report.git_commit}")
    print(
        f"  - protocol.get_current_git_commit(): {report.details.get('protocol_commit')}"
    )
    if report.details.get("git_error"):
        print(f"  - Git error detail: {report.details['git_error']}")

    # Report Check 2
    chk2_status = (
        "PASS"
        if (
            report.tag_exists_and_annotated
            and report.tag_target_commit == report.git_commit
        )
        else "FAIL"
    )
    print(f"Check 2 (Annotated authorization tag): [{chk2_status}]")
    print(
        f"  - Tag 'phase3-authorization' exists and is annotated: {report.tag_exists_and_annotated}"
    )
    print(f"  - Tag target commit: {report.tag_target_commit}")
    if report.details.get("tag_error"):
        print(f"  - Tag detail: {report.details['tag_error']}")

    # Report Check 3
    chk3_status = (
        "PASS"
        if (
            report.sha256_checks_passed
            and report.readme_hashes_matched
            and report.lag_sensitivity_not_activated
        )
        else "FAIL"
    )
    print(
        f"Check 3 (Frozen input hashes, README table, lag sensitivity): [{chk3_status}]"
    )
    print(
        f"  - Registry & calibration hashes matched expected config: {report.sha256_checks_passed}"
    )
    print(f"  - Appendix G README hashes matched: {report.readme_hashes_matched}")
    print(
        f"  - Lag sensitivity not activated across 5 stations: {report.lag_sensitivity_not_activated}"
    )

    # Report Check 4
    chk4_status = "PASS" if report.winner_configs_matched else "FAIL"
    print(
        f"Check 4 (Winning Level 3 configs match LEVEL3_TEMPERATURE_CONFIGS): [{chk4_status}]"
    )
    for st, wcfg in report.details.get("winner_configs", {}).items():
        print(
            f"  - Station {st}: config_id={wcfg['config_id']}, grouping={wcfg['grouping']}, "
            f"n_quantiles={wcfg['n_quantiles']}, extrapolation={wcfg['extrapolation']}"
        )

    # Report Check 5
    chk5_status = "PASS" if report.validation_files_exist_and_hashed else "FAIL"
    print(
        f"Check 5 (Validation CSVs and ERA5 NetCDFs exist and hashed): [{chk5_status}]"
    )
    print(
        f"  - Required observation CSVs and ERA5 NetCDFs verified: {report.validation_files_exist_and_hashed}"
    )

    # Report Check 6
    chk6_status = "PASS" if report.ledger_path_resolved_from_repo else "FAIL"
    print(
        f"Check 6 (Ledger path resolution to canonical repository path): [{chk6_status}]"
    )
    print(
        f"  - Ledger path resolved to canonical path: {report.ledger_path_resolved_from_repo}"
    )

    # Report Check 7 (C4b Raw Bias recalculation)
    chk_bias_status = "PASS" if report.raw_bias_checks_passed else "FAIL"
    print(
        f"Check 7 (Calibration raw bias verified against archived reports): [{chk_bias_status}]"
    )
    for st, binfo in report.details.get("raw_bias_details", {}).items():
        st_ok = "PASS" if binfo["matched"] else "FAIL"
        print(
            f"  - Station {st}: recalculated (tmax={binfo['recalculated_tmax']:.4f}, "
            f"tmin={binfo['recalculated_tmin']:.4f}) vs archived "
            f"(tmax={binfo['archived_tmax']:.4f}, tmin={binfo['archived_tmin']:.4f}) [{st_ok}]"
        )

    # Report Pipeline Smoke Test (C3)
    smoke_status = "PASS" if report.pipeline_smoke_passed else "FAIL"
    print(f"Pipeline Smoke Test (C3 in-sample calibration check): [{smoke_status}]")
    for st in ANDEAN_PHASE3_STATION_IDS:
        smk = report.details.get(f"smoke_{st}", "N/A")
        print(f"  - Station {st}: PIPELINE SMOKE: {smk}")

    # Report Calibration Fitting
    print("-" * 80)
    fit_status = (
        "PASS" if len(fitted_models) == len(ANDEAN_PHASE3_STATION_IDS) else "FAIL"
    )
    print(
        f"Fitted Calibration Models (I.4): [{fit_status}] ({len(fitted_models)} stations successfully fitted)"
    )
    for st in ANDEAN_PHASE3_STATION_IDS:
        if st in fitted_models:
            f_info = fitted_models[st]
            ident = f_info["identical_to_winner"]
            print(
                f"  - Station {st}: L1 fitted, L2 fitted, L3-winner fitted, L3-constant (identical_to_winner={ident})"
            )
        else:
            print(f"  - Station {st}: Fitting not performed")

    # C4c: Print relative-path SHA-256 digests for all inputs
    print("-" * 80)
    print("INPUT FILE SHA-256 HASHES (C4c):")
    print(
        f"tropicor/data/andean_station_registry.json: {report.input_file_hashes['registry']}"
    )
    print(
        f"docs/science/appendix_g/lag_audit_andean_calibration.json: {report.input_file_hashes['lag_audit']}"
    )
    for st in ANDEAN_PHASE3_STATION_IDS:
        rel_rep = f"docs/science/appendix_g/calibration/andes_calibration_{st}.json"
        print(f"{rel_rep}: {report.input_file_hashes['calibration'][st]}")

    st_paths = report.details.get("station_relative_paths", {})
    for st in ANDEAN_PHASE3_STATION_IDS:
        st_entry = st_paths.get(st, {})
        if "tmax_csv" in st_entry:
            tmax_rel = st_entry["tmax_csv"]
            print(
                f"{tmax_rel}: {report.input_file_hashes['station_files'][st]['tmax_csv']}"
            )
        if "tmin_csv" in st_entry:
            tmin_rel = st_entry["tmin_csv"]
            print(
                f"{tmin_rel}: {report.input_file_hashes['station_files'][st]['tmin_csv']}"
            )
        cal_ncs = report.input_file_hashes["station_files"][st].get(
            "era5_calibration_files", {}
        )
        for nc_name, nc_h in sorted(cal_ncs.items()):
            rel_nc = f"local/era5_ts/{st}/{nc_name}"
            print(f"{rel_nc}: {nc_h}")
        val_ncs = report.input_file_hashes["station_files"][st].get(
            "era5_validation_files", {}
        )
        for nc_name, nc_h in sorted(val_ncs.items()):
            # Avoid printing duplicate lines if a file is both in cal and val (boundary year)
            if nc_name in cal_ncs:
                continue
            rel_nc = f"local/era5_ts/{st}/{nc_name}"
            print(f"{rel_nc}: {nc_h}")

    print("-" * 80)
    # Checks 1 and 2 are reported without aborting. Checks 3-7, smoke, and fittings determine exit code.
    hard_checks_passed = (
        report.sha256_checks_passed
        and report.readme_hashes_matched
        and report.lag_sensitivity_not_activated
        and report.winner_configs_matched
        and report.validation_files_exist_and_hashed
        and report.ledger_path_resolved_from_repo
        and report.raw_bias_checks_passed
        and report.pipeline_smoke_passed
        and len(fitted_models) == 5
    )

    if hard_checks_passed:
        print(
            "PREFLIGHT STATUS: PASS (Required checks 3-7, smoke tests, and fittings passed; Checks 1-2 reported)"
        )
        return 0
    else:
        print("PREFLIGHT STATUS: FAIL (One or more required checks failed)")
        return 1


def main(args: list[str] | None = None) -> int:
    """Main execution entry point."""
    try:
        parsed = parse_args(args)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1

    if parsed.preflight:
        repo_root = (parsed.repo_root or REPO_ROOT).resolve()
        return run_preflight_cli(
            repo_root=repo_root,
            era5_base_dir=parsed.era5_base_dir,
            ledger_path=parsed.ledger_path,
        )

    # In formal and reproduction modes, repo_root is strictly REPO_ROOT
    repo_root = REPO_ROOT
    run_kind = "reproduction" if parsed.reproduce else "formal"

    print("=" * 80)
    print(
        f"PHASE 3 ANDEAN TEMPERATURE EVALUATOR — {run_kind.upper()} RUN (Protocol v1.7)"
    )
    print("=" * 80)

    try:
        summary = run_phase3_evaluation(
            repo_root=repo_root,
            run_kind=run_kind,
            open_validation=True,
            ledger_path=None,
            reports_dir=None,
            era5_base_dir=None,
            station_ids=ANDEAN_PHASE3_STATION_IDS,
            dry_run=False,
        )
        print("\nEvaluation successfully completed.")
        print(f"Evaluator Commit: {summary.get('evaluator_commit')}")
        print(f"Success Counts by Level: {summary.get('success_counts_by_level')}")
        return 0
    except Exception as err:
        print(f"\n[FATAL ERROR] Phase 3 evaluation aborted: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
