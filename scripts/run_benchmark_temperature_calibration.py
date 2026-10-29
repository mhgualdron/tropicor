"""Benchmark station temperature calibration runner (Protocol v1.8, Appendix J).

Executes daily temperature calibration hierarchy (Levels 1-3) on the primary benchmark stations:
- Jardín Botánico (21205710): Calibration partition 1990-01-01 to 2004-12-31 (15 years)
- Aguas Claras (23215030): Calibration partition 1990-01-01 to 2000-12-31 (11 years)

Protocol Safety Constraints:
- Accepts only station ids 21205710 and 23215030.
- Calibration partitions strictly resolved via get_benchmark_partitions('temperature', station_id)[0].
- Aggregation window parameters fixed at window_start_utc=5, lag_days=0, min_hours=24.
- Does NOT access, read, or evaluate any observation in validation windows.
- Does NOT create or modify the validation opening ledger.
- Deterministic JSON output saved to docs/science/appendix_j/calibration/benchmark_calibration_<id>.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tropicor.core.protocol import (  # noqa: E402
    get_benchmark_partitions,
    get_current_git_commit,
    temperature_tuning_loss,
)
from tropicor.downscale.temperature import (  # noqa: E402
    LEVEL3_TEMPERATURE_CONFIGS,
    Level1MonthlyAdditiveTemperature,
    Level2MeanStateDTRScalingTemperature,
    Level3QuantileMappingTemperature,
    load_calibration_temperature_pair,
    tune_level3_temperature_hyperparameters,
)

BENCHMARK_STATIONS: Dict[str, Dict[str, Any]] = {
    "21205710": {
        "name": "JARDIN BOTANICO - AUT",
        "default_obs_tmax": Path("local/newFiles/TempMaxima_JardinBotanico.csv"),
        "default_obs_tmin": Path("local/newFiles/TempMinima_JardinBotanico.csv"),
        "default_era5_dir": Path("local/era5_hourly/21205710"),
    },
    "23215030": {
        "name": "AGUAS CLARAS",
        "default_obs_tmax": Path("local/newFiles/TempMaxima_AguasClaras.csv"),
        "default_obs_tmin": Path("local/newFiles/TempMinima_AguasClaras.csv"),
        "default_era5_dir": Path("local/era5_hourly/23215030"),
    },
}


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_benchmark_calibration(
    station_id: str,
    obs_tmax_path: Path,
    obs_tmin_path: Path,
    era5_dir: Path,
    output_dir: Path = Path("docs/science/appendix_j/calibration"),
) -> Dict[str, Any]:
    """Run calibration hierarchy and produce deterministic JSON report."""
    git_check = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            "scripts/run_benchmark_temperature_calibration.py",
            "tropicor/",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    if git_check.stdout.strip():
        raise RuntimeError(
            "Uncommitted or untracked changes detected in scripts/run_benchmark_temperature_calibration.py or tropicor/:\n"
            f"{git_check.stdout}"
        )

    clean_id = re.sub(r"\.0$", "", str(station_id).strip())
    if clean_id not in BENCHMARK_STATIONS:
        raise ValueError(
            f"Unsupported station ID '{clean_id}'. This runner strictly accepts "
            f"benchmark station IDs: {list(BENCHMARK_STATIONS.keys())}."
        )

    st_meta = BENCHMARK_STATIONS[clean_id]
    cal_partitions = get_benchmark_partitions("temperature", clean_id)
    cal_start, cal_end = cal_partitions[0]

    print(
        f"=== Calibrating Primary Benchmark Station {clean_id} ({st_meta['name']}) ==="
    )
    print(f"Calibration block: {cal_start} to {cal_end}")
    print(f"Observation Tmax: {obs_tmax_path}")
    print(f"Observation Tmin: {obs_tmin_path}")
    print(f"ERA5 directory:   {era5_dir}")

    # Verify input paths exist
    if not obs_tmax_path.exists():
        raise FileNotFoundError(f"Tmax observation file not found: {obs_tmax_path}")
    if not obs_tmin_path.exists():
        raise FileNotFoundError(f"Tmin observation file not found: {obs_tmin_path}")
    if not era5_dir.exists():
        raise FileNotFoundError(f"ERA5 directory not found: {era5_dir}")

    # Collect and hash all calibration ERA5 NetCDF files
    needed_era5_years = list(range(cal_start, cal_end + 2))
    era5_file_hashes: Dict[str, str] = {}
    for y in needed_era5_years:
        nc_file = era5_dir / f"era5_2m_temperature_{y}.nc"
        if not nc_file.exists():
            raise FileNotFoundError(f"Required ERA5 NetCDF file not found: {nc_file}")
        era5_file_hashes[nc_file.name] = compute_sha256(nc_file)

    input_file_hashes = {
        "obs_tmax_csv": compute_sha256(obs_tmax_path),
        "obs_tmin_csv": compute_sha256(obs_tmin_path),
        "era5_calibration_files": era5_file_hashes,
    }

    # Load calibration paired series
    obs_tmax, obs_tmin, mod_tmax, mod_tmin = load_calibration_temperature_pair(
        station_id=clean_id,
        obs_tmax_path=obs_tmax_path,
        obs_tmin_path=obs_tmin_path,
        era5_target=era5_dir,
        window_start_utc=5,
        lag_days=0,
        min_hours=24,
    )

    # Raw biases
    raw_bias_tmax = float((mod_tmax - obs_tmax).mean())
    raw_bias_tmin = float((mod_tmin - obs_tmin).mean())

    # Level 1: Monthly Additive Bias Correction
    l1 = Level1MonthlyAdditiveTemperature()
    l1.fit(obs_tmax, obs_tmin, mod_tmax, mod_tmin)
    l1_pred_tmax, l1_pred_tmin = l1.transform(mod_tmax, mod_tmin)
    l1_loss = temperature_tuning_loss(obs_tmax, l1_pred_tmax, obs_tmin, l1_pred_tmin)

    # Level 2: Mean State & DTR Scaling
    l2 = Level2MeanStateDTRScalingTemperature()
    l2.fit(obs_tmax, obs_tmin, mod_tmax, mod_tmin)
    l2_pred_tmax, l2_pred_tmin = l2.transform(mod_tmax, mod_tmin)
    l2_loss = temperature_tuning_loss(obs_tmax, l2_pred_tmax, obs_tmin, l2_pred_tmin)

    # Level 3: Cross-validation over pre-specified 8-configuration grid
    best_res, l3_model, all_cv_results = tune_level3_temperature_hyperparameters(
        obs_tmax=obs_tmax,
        obs_tmin=obs_tmin,
        mod_tmax=mod_tmax,
        mod_tmin=mod_tmin,
        configs=LEVEL3_TEMPERATURE_CONFIGS,
    )

    # In-sample Level 3 evaluation of winner model
    l3_pred_tmax, l3_pred_tmin = l3_model.transform(mod_tmax, mod_tmin)
    l3_in_sample_loss = temperature_tuning_loss(
        obs_tmax, l3_pred_tmax, obs_tmin, l3_pred_tmin
    )

    # Level 3 Constant Sensitivity Model (same grouping and quantiles as winner)
    if best_res.extrapolation == "constant":
        l3_const_model = l3_model
        identical_to_winner = True
        l3_const_in_sample_loss = l3_in_sample_loss
    else:
        l3_const_model = Level3QuantileMappingTemperature(
            grouping=best_res.grouping,
            n_quantiles=best_res.n_quantiles,
            extrapolation="constant",
        )
        l3_const_model.fit(obs_tmax, obs_tmin, mod_tmax, mod_tmin)
        const_pred_tmax, const_pred_tmin = l3_const_model.transform(mod_tmax, mod_tmin)
        l3_const_in_sample_loss = temperature_tuning_loss(
            obs_tmax, const_pred_tmax, obs_tmin, const_pred_tmin
        )
        identical_to_winner = False

    cv_grid = [
        {
            "config_id": r.config_id,
            "grouping": r.grouping,
            "n_quantiles": r.n_quantiles,
            "extrapolation": r.extrapolation,
            "cv_loss": round(float(r.overall_tuning_loss), 4),
            "min_n_per_group": r.min_n_per_group,
            "n_extrapolated_days": r.n_extrapolated_days,
            "max_abs_extrap_corr": round(float(r.max_abs_extrapolation_correction), 4),
            "crossings_before": r.crossings_before_rule,
            "crossings_after": r.crossings_after_rule,
        }
        for r in all_cv_results
    ]

    report: Dict[str, Any] = {
        "report_type": "benchmark_temperature_calibration",
        "protocol_version": "1.8",
        "git_commit": get_current_git_commit(),
        "station_id": clean_id,
        "station_name": st_meta["name"],
        "calibration_block": [cal_start, cal_end],
        "input_hashes": input_file_hashes,
        "raw_bias": {
            "tmax": round(raw_bias_tmax, 4),
            "tmin": round(raw_bias_tmin, 4),
        },
        "level1": {
            "calibration_loss": round(float(l1_loss), 4),
            "crossings_before": l1.n_crossings_before_rule_,
            "crossings_after": l1.n_crossings_after_rule_,
        },
        "level2": {
            "calibration_loss": round(float(l2_loss), 4),
            "crossings_before": l2.n_crossings_before_rule_,
            "crossings_after": l2.n_crossings_after_rule_,
        },
        "level3": {
            "winner_config_id": best_res.config_id,
            "winner_grouping": best_res.grouping,
            "winner_n_quantiles": best_res.n_quantiles,
            "winner_extrapolation": best_res.extrapolation,
            "cv_loss": round(float(best_res.overall_tuning_loss), 4),
            "calibration_loss": round(float(l3_in_sample_loss), 4),
            "crossings_before": l3_model.n_crossings_before_rule_,
            "crossings_after": l3_model.n_crossings_after_rule_,
            "n_extrapolated_days": best_res.n_extrapolated_days,
            "max_abs_extrapolation_correction": round(
                float(best_res.max_abs_extrapolation_correction), 4
            ),
            "min_n_per_group": best_res.min_n_per_group,
            "cv_configurations": cv_grid,
        },
        "level3_constant": {
            "identical_to_winner": identical_to_winner,
            "grouping": best_res.grouping,
            "n_quantiles": best_res.n_quantiles,
            "extrapolation": "constant",
            "calibration_loss": round(float(l3_const_in_sample_loss), 4),
            "crossings_before": l3_const_model.n_crossings_before_rule_,
            "crossings_after": l3_const_model.n_crossings_after_rule_,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"benchmark_calibration_{clean_id}.json"

    # Deterministic JSON with LF endings and trailing newline
    json_bytes = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    out_file.write_bytes(json_bytes)
    print(f"Calibration report saved to: {out_file}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run daily temperature calibration for primary benchmark stations."
    )
    parser.add_argument(
        "--station",
        required=True,
        choices=["21205710", "23215030", "both"],
        help="Station code (21205710, 23215030, or both).",
    )
    parser.add_argument(
        "--obs-tmax",
        type=Path,
        default=None,
        help="Custom path to Tmax CSV (for single station mode).",
    )
    parser.add_argument(
        "--obs-tmin",
        type=Path,
        default=None,
        help="Custom path to Tmin CSV (for single station mode).",
    )
    parser.add_argument(
        "--era5-dir",
        type=Path,
        default=None,
        help="Custom path to ERA5 directory (for single station mode).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/science/appendix_j/calibration"),
        help="Output directory for calibration reports.",
    )

    args = parser.parse_args()

    if args.station == "both" and any(
        x is not None for x in (args.obs_tmax, args.obs_tmin, args.era5_dir)
    ):
        raise ValueError(
            "Custom path arguments (--obs-tmax, --obs-tmin, --era5-dir) cannot be used with --station both."
        )

    station_list = (
        ["21205710", "23215030"] if args.station == "both" else [args.station]
    )

    for st_id in station_list:
        defaults = BENCHMARK_STATIONS[st_id]
        tmax_p = args.obs_tmax or defaults["default_obs_tmax"]
        tmin_p = args.obs_tmin or defaults["default_obs_tmin"]
        era5_p = args.era5_dir or defaults["default_era5_dir"]

        run_benchmark_calibration(
            station_id=st_id,
            obs_tmax_path=tmax_p,
            obs_tmin_path=tmin_p,
            era5_dir=era5_p,
            output_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()
