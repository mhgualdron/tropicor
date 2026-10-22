"""Calibration executor for Andean generalization stations (Protocol v1.4, Appendix F).

Supports:
- Execution of daily temperature calibration hierarchy (Levels 1-3) on candidate stations
  from the Andean registry.
- Strict Protocol constraints: window_start_utc=5, lag_days=0, min_hours=24 explicit;
  minimizes temperature_tuning_loss with pre-specified tie-breaking.
- Output written to local/reports/andes_calibration_<codigo>.json with CV losses, winning
  config, extrapolated days, crossings, max tail correction, raw biases, protocol version,
  and git hash.
- Mode --dry-run: verifies observation paths and ERA5 NetCDF files (years, names, cell coords,
  units) WITHOUT reading or computing observation values.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import netCDF4

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tropicor.core.andean_registry import get_andean_station  # noqa: E402
from tropicor.core.protocol import (  # noqa: E402
    get_current_git_commit,
    temperature_tuning_loss,
)
from tropicor.downscale.temperature import (  # noqa: E402
    Level1MonthlyAdditiveTemperature,
    Level2MeanStateDTRScalingTemperature,
    load_calibration_temperature_pair,
    tune_level3_temperature_hyperparameters,
)


def run_dry_run(
    station_id: str,
    obs_tmax_path: Path,
    obs_tmin_path: Path,
    era5_dir: Path,
) -> Dict[str, Any]:
    """Verify presence of observation paths and ERA5 NetCDFs without reading observation values."""
    st = get_andean_station(station_id)
    cal_start, cal_end = st.cal_block

    # 1. Verify existence of observation files WITHOUT reading their values
    if obs_tmax_path.is_dir():
        tmax_files = list(obs_tmax_path.glob("TempMaxima_*.csv"))
        if len(tmax_files) != 1:
            raise ValueError(
                f"Expected exactly 1 file matching 'TempMaxima_*.csv' in {obs_tmax_path}, found {len(tmax_files)}."
            )
        obs_tmax_path = tmax_files[0]
    elif not obs_tmax_path.exists():
        raise ValueError(f"Tmax observation file not found: {obs_tmax_path}")
    elif not re.match(r"^TempMaxima_.*\.csv$", obs_tmax_path.name):
        raise ValueError(
            f"Tmax observation file '{obs_tmax_path.name}' does not match pattern 'TempMaxima_*.csv'."
        )

    if obs_tmin_path.is_dir():
        tmin_files = list(obs_tmin_path.glob("TempMinima_*.csv"))
        if len(tmin_files) != 1:
            raise ValueError(
                f"Expected exactly 1 file matching 'TempMinima_*.csv' in {obs_tmin_path}, found {len(tmin_files)}."
            )
        obs_tmin_path = tmin_files[0]
    elif not obs_tmin_path.exists():
        raise ValueError(f"Tmin observation file not found: {obs_tmin_path}")
    elif not re.match(r"^TempMinima_.*\.csv$", obs_tmin_path.name):
        raise ValueError(
            f"Tmin observation file '{obs_tmin_path.name}' does not match pattern 'TempMinima_*.csv'."
        )

    # 2. Verify ERA5 directory and files for calibration span [cal_start, cal_end + 1]
    if not era5_dir.exists():
        raise ValueError(f"ERA5 directory not found: {era5_dir}")

    era5_years_verified: List[int] = []
    for y in range(cal_start, cal_end + 2):
        matches = list(era5_dir.glob(f"era5_2m_temperature_{y}.nc"))
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly 1 NetCDF file matching 'era5_2m_temperature_{y}.nc' in {era5_dir}, found {len(matches)}."
            )
        f_nc = matches[0]
        # Inspect NetCDF coordinates and units WITHOUT touching observation values
        with netCDF4.Dataset(str(f_nc), "r") as nc:
            if "latitude" in nc.variables:
                lat_var = nc.variables["latitude"]
            elif "lat" in nc.variables:
                lat_var = nc.variables["lat"]
            else:
                lat_var = None

            if "longitude" in nc.variables:
                lon_var = nc.variables["longitude"]
            elif "lon" in nc.variables:
                lon_var = nc.variables["lon"]
            else:
                lon_var = None

            if lat_var is None or lon_var is None:
                raise ValueError(
                    f"NetCDF {f_nc.name} missing latitude/longitude coordinates"
                )
            lat_val = float(lat_var[0]) if lat_var.ndim > 0 else float(lat_var[:])
            lon_val = float(lon_var[0]) if lon_var.ndim > 0 else float(lon_var[:])

            if abs(lat_val - st.celda_lat) > 1e-4 or abs(lon_val - st.celda_lon) > 1e-4:
                raise ValueError(
                    f"NetCDF {f_nc.name} cell coordinates ({lat_val}, {lon_val}) do not match "
                    f"registry cell ({st.celda_lat}, {st.celda_lon}) within 1e-4."
                )

            if "t2m" in nc.variables:
                t2m_var = nc.variables["t2m"]
            elif "2m_temperature" in nc.variables:
                t2m_var = nc.variables["2m_temperature"]
            else:
                t2m_var = None
            if t2m_var is None:
                raise ValueError(
                    f"NetCDF {f_nc.name} missing temperature variable 't2m'"
                )
            units = getattr(t2m_var, "units", None)
            if units != "K":
                raise ValueError(
                    f"NetCDF {f_nc.name} temperature variable must have units 'K', found '{units}'"
                )

        era5_years_verified.append(y)

    print(f"[DRY-RUN OK] Station {st.codigo} ({st.nombre}):")
    print(
        f"  Observations present: Tmax={obs_tmax_path.name}, Tmin={obs_tmin_path.name} "
        "(values NOT read)"
    )
    print(
        f"  ERA5 files verified ({len(era5_years_verified)} files, {cal_start}-{cal_end + 1}): "
        f"all match cell ({st.celda_lat}, {st.celda_lon}) and units='K'"
    )
    return {
        "station_id": st.codigo,
        "station_name": st.nombre,
        "cal_block": [cal_start, cal_end],
        "era5_years_verified": era5_years_verified,
        "dry_run": True,
    }


def calibrate_station(
    station_id: str,
    obs_tmax_path: Path,
    obs_tmin_path: Path,
    era5_dir: Path,
    output_dir: Path = Path("local/reports"),
) -> Dict[str, Any]:
    """Execute calibration for an Andean station across Levels 1-3."""
    st = get_andean_station(station_id)

    # 1. Load paired calibration series using extended loader
    obs_tmax, obs_tmin, mod_tmax, mod_tmin = load_calibration_temperature_pair(
        station_id=station_id,
        obs_tmax_path=obs_tmax_path,
        obs_tmin_path=obs_tmin_path,
        era5_target=era5_dir,
        window_start_utc=5,
        lag_days=0,
        min_hours=24,
    )

    # Raw bias in calibration block
    raw_bias_tmax = float((mod_tmax - obs_tmax).dropna().mean())
    raw_bias_tmin = float((mod_tmin - obs_tmin).dropna().mean())

    # 2. Level 1 (Monthly Additive Bias)
    l1 = Level1MonthlyAdditiveTemperature()
    l1.fit(obs_tmax, obs_tmin, mod_tmax, mod_tmin)
    l1_tmax, l1_tmin = l1.transform(mod_tmax, mod_tmin)
    l1_loss = temperature_tuning_loss(obs_tmax, l1_tmax, obs_tmin, l1_tmin)

    # 3. Level 2 (Mean State + DTR Scaling)
    l2 = Level2MeanStateDTRScalingTemperature()
    l2.fit(obs_tmax, obs_tmin, mod_tmax, mod_tmin)
    l2_tmax, l2_tmin = l2.transform(mod_tmax, mod_tmin)
    l2_loss = temperature_tuning_loss(obs_tmax, l2_tmax, obs_tmin, l2_tmin)

    # 4. Level 3 CV Tuning (8 configs)
    best_res, l3_model, all_cv_results = tune_level3_temperature_hyperparameters(
        obs_tmax=obs_tmax,
        obs_tmin=obs_tmin,
        mod_tmax=mod_tmax,
        mod_tmin=mod_tmin,
    )
    l3_tmax, l3_tmin = l3_model.transform(mod_tmax, mod_tmin)
    l3_in_sample_loss = temperature_tuning_loss(obs_tmax, l3_tmax, obs_tmin, l3_tmin)

    cv_grid: List[Dict[str, Any]] = [
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
        "report_type": "andes_temperature_calibration",
        "protocol_version": "1.4",
        "git_commit": get_current_git_commit(),
        "station_id": st.codigo,
        "station_name": st.nombre,
        "calibration_block": [st.cal_block[0], st.cal_block[1]],
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
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"andes_calibration_{st.codigo}.json"
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Calibration report saved to: {out_file}")
    return report


def resolve_default_paths(station_id: str) -> Tuple[Path, Path, Path]:
    """Resolve default observation and ERA5 paths for a station code.

    Requires exactly 1 file matching 'TempMaxima_*.csv' and exactly 1 file
    matching 'TempMinima_*.csv'. Raises ValueError if 0 or >1 files found.
    No fallback path is permitted.
    """
    st_dir = Path("local/era5_ts") / station_id
    if not st_dir.exists():
        raise ValueError(f"Station directory not found: {st_dir}")
    tmax_files = list(st_dir.glob("TempMaxima_*.csv"))
    if len(tmax_files) != 1:
        raise ValueError(
            f"Expected exactly 1 file matching 'TempMaxima_*.csv' in {st_dir}, found {len(tmax_files)}."
        )
    tmin_files = list(st_dir.glob("TempMinima_*.csv"))
    if len(tmin_files) != 1:
        raise ValueError(
            f"Expected exactly 1 file matching 'TempMinima_*.csv' in {st_dir}, found {len(tmin_files)}."
        )
    return tmax_files[0], tmin_files[0], st_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run daily temperature calibration for an Andean station."
    )
    parser.add_argument(
        "--station",
        required=True,
        help="Station code from Andean registry (e.g. 37015020).",
    )
    parser.add_argument(
        "--obs-tmax",
        type=Path,
        default=None,
        help="Path to Tmax CSV. Defaults to local/era5_ts/<station>/TempMaxima_*.csv",
    )
    parser.add_argument(
        "--obs-tmin",
        type=Path,
        default=None,
        help="Path to Tmin CSV. Defaults to local/era5_ts/<station>/TempMinima_*.csv",
    )
    parser.add_argument(
        "--era5-dir",
        type=Path,
        default=None,
        help="Path to ERA5 directory. Defaults to local/era5_ts/<station>/",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify paths and ERA5 files without reading observation values.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("local/reports"),
        help="Output directory for calibration report.",
    )

    args = parser.parse_args()

    def_tmax, def_tmin, def_era5 = resolve_default_paths(args.station)
    tmax_p = args.obs_tmax or def_tmax
    tmin_p = args.obs_tmin or def_tmin
    era5_p = args.era5_dir or def_era5

    if args.dry_run:
        run_dry_run(
            station_id=args.station,
            obs_tmax_path=tmax_p,
            obs_tmin_path=tmin_p,
            era5_dir=era5_p,
        )
        return

    calibrate_station(
        station_id=args.station,
        obs_tmax_path=tmax_p,
        obs_tmin_path=tmin_p,
        era5_dir=era5_p,
        output_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
