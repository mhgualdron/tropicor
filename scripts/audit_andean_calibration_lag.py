#!/usr/bin/env python3
"""Pre-validation calibration lag audit for Andean generalization stations (Protocol Appendix G.6 / H.5).

Implements the observational lag audit strictly on the calibration partitions of the
five Andean generalization stations defined in Table F.4 and andean_station_registry.json:
- G.6.1: Executed strictly on the CALIBRATION partitions of the five Andean stations.
- G.6.2: Evaluates cross-correlation curves, lag correlations (lags -2, -1, 0, +1, +2 days),
  and extrema timing without computing any bias, error, or dispersion metrics (Section 8.2).
- G.6.3: Primary validation remains permanently fixed at lag 0 (05:00-05:00 UTC civil day).
- G.6.4 / H.5: Lag detection criterion evaluates paired difference Delta r(k) = r(k) - r(0)
  for candidate lags k in {-1, +1} using a paired 30 calendar-day block bootstrap (1,000 replicas,
  seed 42). If the 95% bootstrap CI strictly excludes zero in favor of k (Delta r(k) > 0,
  i.e. ci_lower > 0), labeled lag sensitivity at lag k is activated for validation.

Outputs:
    local/reports/lag_audit_andean_calibration.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tropicor.core.protocol import (  # noqa: E402
    DEFAULT_BOOTSTRAP_SEED,
    get_current_git_commit,
)
from tropicor.downscale.temperature import (  # noqa: E402
    load_calibration_temperature_pair,
)

AUDIT_LAGS = [-2, -1, 0, 1, 2]
CANDIDATE_LAGS = [-1, 1]
DEFAULT_BLOCK_SIZE = 30
DEFAULT_REPLICAS = 1000
DEFAULT_REGISTRY_PATH = REPO_ROOT / "tropicor" / "data" / "andean_station_registry.json"
DEFAULT_BASE_DIR = REPO_ROOT / "local" / "era5_ts"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "local" / "reports" / "lag_audit_andean_calibration.json"


def compute_file_sha256(filepath: Path) -> str:
    """Compute cryptographic SHA-256 hash of a file."""
    if not filepath.exists() or not filepath.is_file():
        raise FileNotFoundError(f"Input file not found for hashing: {filepath}")
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation coefficient returning NaN for n < 2 or zero variance.

    Args:
        x: First 1D array of values.
        y: Second 1D array of values.

    Returns:
        Pearson correlation r as float, or float('nan') if len < 2 or variance < 1e-12.
    """
    if len(x) < 2:
        return float("nan")
    std_x = float(np.std(x))
    std_y = float(np.std(y))
    if std_x < 1e-12 or std_y < 1e-12:
        return float("nan")
    r = float(np.corrcoef(x, y)[0, 1])
    return r


def generate_calendar_block_resample_indices(
    n: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate 1D resampled array of indices of length n using calendar moving blocks.

    Samples ceil(n / block_size) uniform start positions in 0..n-block_size,
    concatenates contiguous blocks of size block_size, and truncates to n.

    Args:
        n: Total number of days in the continuous calendar series.
        block_size: Size of contiguous calendar blocks in days.
        rng: NumPy random Generator.

    Returns:
        1D integer numpy array of length n containing resampled calendar indices.
    """
    if n < block_size:
        raise ValueError(f"Series length n={n} must be >= block_size={block_size}.")
    n_blocks = int(np.ceil(n / block_size))
    max_start = max(1, n - block_size + 1)
    starts = rng.integers(0, max_start, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
    return idx


def build_station_input_hashes(
    station_code: str,
    tmax_csv: Path,
    tmin_csv: Path,
    era5_dir: Path,
    cal_block: Tuple[int, int] | Sequence[int],
) -> Dict[str, str]:
    """Construct cryptographic SHA-256 hashes of all station inputs keyed by '<código>/<archivo>'.

    Args:
        station_code: Station identifier.
        tmax_csv: Path to maximum temperature observation CSV.
        tmin_csv: Path to minimum temperature observation CSV.
        era5_dir: Directory containing hourly ERA5 NetCDF files.
        cal_block: [cal_start, cal_end] calibration year range.

    Returns:
        Dictionary mapping '<código>/<archivo>' to SHA-256 hex digest.

    Raises:
        ValueError: If any required ERA5 file in cal_start..cal_end+1 is missing.
    """
    hashes: Dict[str, str] = {
        f"{station_code}/{tmax_csv.name}": compute_file_sha256(tmax_csv),
        f"{station_code}/{tmin_csv.name}": compute_file_sha256(tmin_csv),
    }
    cal_start, cal_end = cal_block[0], cal_block[1]
    cal_years = list(range(cal_start, cal_end + 2))
    for y in cal_years:
        f_era = era5_dir / f"era5_2m_temperature_{y}.nc"
        if not f_era.exists() or not f_era.is_file():
            raise ValueError(
                f"Required ERA5 file missing for station {station_code}: {f_era.name} "
                f"in {era5_dir}."
            )
        hashes[f"{station_code}/{f_era.name}"] = compute_file_sha256(f_era)
    return hashes


def calendar_block_bootstrap_lag_difference(
    obs: np.ndarray,
    mod_0: np.ndarray,
    mod_k: np.ndarray,
    block_size: int = DEFAULT_BLOCK_SIZE,
    n_replicas: int = DEFAULT_REPLICAS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Tuple[float, float, float, float, float, bool, int]:
    """Compute paired correlation difference Delta r(k) = r(k) - r(0) with 30-day calendar block bootstrap CI.

    Protocol Reference:
        Appendix G.6.4 / H.5: Evaluates Delta r(k) = r(k) - r(0) using paired 30 calendar-day
        moving block bootstrap with 1,000 replicas and seed fixed at 42. Resampling operates
        over the continuous daily index of the calibration period. Within each replica,
        correlations are computed on the days that are valid for the observation and for both
        modeled series being compared. Replicas with < 2 valid pairs or zero variance are
        discarded and counted in n_discarded_replicas.

    Args:
        obs: 1D array of observed daily temperature over the continuous calibration span (with NaNs).
        mod_0: 1D array of ERA5 daily temperature at lag 0 on matching calendar days.
        mod_k: 1D array of ERA5 daily temperature at lag k on matching calendar days.
        block_size: Size of contiguous calendar blocks in days (default 30).
        n_replicas: Number of bootstrap replicas (default 1000).
        seed: Random number generator seed (default 42).

    Returns:
        Tuple of (r_k_paired, r_0_paired, point_delta_r, ci_lower, ci_upper,
        excludes_zero_in_favor_of_k, n_discarded_replicas).
    """
    n = len(obs)
    if n == 0 or len(mod_0) != n or len(mod_k) != n:
        raise ValueError("Series lengths must match and be non-empty.")

    # 1. Full paired sample mask and point estimates
    valid_mask = ~np.isnan(obs) & ~np.isnan(mod_0) & ~np.isnan(mod_k)
    n_paired_days = int(np.sum(valid_mask))

    if n_paired_days < 2:
        raise ValueError(
            f"Insufficient valid paired days (n={n_paired_days}) to compute calibration correlation."
        )

    o_paired = obs[valid_mask]
    m0_paired = mod_0[valid_mask]
    mk_paired = mod_k[valid_mask]

    r_0_paired = compute_safe_pearson(o_paired, m0_paired)
    r_k_paired = compute_safe_pearson(o_paired, mk_paired)

    if np.isnan(r_0_paired) or np.isnan(r_k_paired):
        raise ValueError(
            f"Point correlation estimate produced NaN on calibration partition: "
            f"r_0={r_0_paired}, r_k={r_k_paired}."
        )

    point_delta_r = r_k_paired - r_0_paired

    if n < block_size:
        raise ValueError(
            f"Continuous series length n={n} is less than block_size={block_size}."
        )

    # 2. Calendar-day moving block bootstrap
    rng = np.random.default_rng(seed)
    diffs: List[float] = []
    n_discarded = 0

    for _ in range(n_replicas):
        idx = generate_calendar_block_resample_indices(n, block_size, rng)

        o_b = obs[idx]
        m0_b = mod_0[idx]
        mk_b = mod_k[idx]

        mask_b = ~np.isnan(o_b) & ~np.isnan(m0_b) & ~np.isnan(mk_b)
        n_valid_b = int(np.sum(mask_b))

        if n_valid_b < 2:
            n_discarded += 1
            continue

        o_v = o_b[mask_b]
        m0_v = m0_b[mask_b]
        mk_v = mk_b[mask_b]

        std_o = float(np.std(o_v))
        std_m0 = float(np.std(m0_v))
        std_mk = float(np.std(mk_v))

        if std_o < 1e-12 or std_m0 < 1e-12 or std_mk < 1e-12:
            n_discarded += 1
            continue

        r0_b = float(np.corrcoef(o_v, m0_v)[0, 1])
        rk_b = float(np.corrcoef(o_v, mk_v)[0, 1])

        if np.isnan(r0_b) or np.isnan(rk_b):
            n_discarded += 1
            continue

        diffs.append(rk_b - r0_b)

    if len(diffs) == 0:
        raise ValueError(
            "All bootstrap replicas were discarded; cannot compute confidence interval."
        )

    ci_lower = float(np.percentile(diffs, 2.5))
    ci_upper = float(np.percentile(diffs, 97.5))
    excludes_zero_in_favor_of_k = bool(ci_lower > 0.0)

    return (
        r_k_paired,
        r_0_paired,
        point_delta_r,
        ci_lower,
        ci_upper,
        excludes_zero_in_favor_of_k,
        n_discarded,
    )


def audit_variable_lag_calibration(
    obs: pd.Series,
    mod_by_lag: Dict[int, pd.Series],
    candidate_lags: Sequence[int] = CANDIDATE_LAGS,
    audit_lags: Sequence[int] = AUDIT_LAGS,
    block_size: int = DEFAULT_BLOCK_SIZE,
    n_replicas: int = DEFAULT_REPLICAS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    """Execute lag audit for a single temperature variable across calibration partition.

    Protocol References:
        - G.6.2: Lag correlations across lags -2, -1, 0, +1, +2 and extrema timing,
          evaluated on each individual lag's pairwise valid mask.
        - G.6.4 / H.5: Paired difference Delta r(k) = r(k) - r(0) for candidate lags k in {-1, +1}
          evaluated with 30 calendar-day moving block bootstrap on the 3-way paired mask.
    """
    # 1. Diagnostic cross-correlation curve across all audit lags (G.6.2)
    lag_correlations: Dict[str, float] = {}
    for k in audit_lags:
        if k not in mod_by_lag:
            raise ValueError(f"Lag {k} missing from modeled series dictionary.")
        mod_k = mod_by_lag[k]
        valid_mask_k = obs.notna() & mod_k.notna()
        r_k = compute_safe_pearson(
            obs[valid_mask_k].to_numpy(), mod_k[valid_mask_k].to_numpy()
        )
        if np.isnan(r_k):
            raise ValueError(
                f"Diagnostic lag {k} correlation produced NaN (insufficient samples or zero variance)."
            )
        lag_correlations[str(k)] = round(r_k, 6)

    # 2. Extrema timing (G.6.2)
    best_lag_str = max(
        lag_correlations.keys(), key=lambda k_str: lag_correlations[k_str]
    )
    extrema_timing = {
        "optimal_lag": int(best_lag_str),
        "max_r": lag_correlations[best_lag_str],
    }

    # 3. Candidate lag evaluations against lag 0 (G.6.4 / H.5)
    mod_0 = mod_by_lag[0]
    candidate_evaluations: Dict[str, Any] = {}
    triggered_lags: List[int] = []

    for k in candidate_lags:
        mod_k = mod_by_lag[k]
        r_k_p, r_0_p, point_delta, ci_l, ci_u, excludes_zero, n_disc = (
            calendar_block_bootstrap_lag_difference(
                obs=obs.to_numpy(),
                mod_0=mod_0.to_numpy(),
                mod_k=mod_k.to_numpy(),
                block_size=block_size,
                n_replicas=n_replicas,
                seed=seed,
            )
        )
        if np.isnan(ci_l) or np.isnan(ci_u):
            raise ValueError(
                f"Confidence interval for lag {k} produced NaN values: [{ci_l}, {ci_u}]."
            )
        r_0_p_rounded = round(r_0_p, 6)
        r_k_p_rounded = round(r_k_p, 6)
        point_delta_rounded = round(r_k_p_rounded - r_0_p_rounded, 6)
        candidate_evaluations[str(k)] = {
            "n_paired_days": int((obs.notna() & mod_0.notna() & mod_k.notna()).sum()),
            "r_0_paired": r_0_p_rounded,
            "r_k_paired": r_k_p_rounded,
            "point_delta_r": point_delta_rounded,
            "ci_95_lower": round(ci_l, 6),
            "ci_95_upper": round(ci_u, 6),
            "excludes_zero_in_favor_of_k": excludes_zero,
            "n_discarded_replicas": n_disc,
        }
        if excludes_zero:
            triggered_lags.append(k)

    return {
        "lag_correlations": lag_correlations,
        "extrema_timing": extrema_timing,
        "candidate_lags": candidate_evaluations,
        "triggered_lags": sorted(triggered_lags),
    }


def audit_station_lag_calibration(
    station_code: str,
    station_info: Dict[str, Any],
    base_dir: Path,
    block_size: int = DEFAULT_BLOCK_SIZE,
    n_replicas: int = DEFAULT_REPLICAS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Audit calibration lag for a registered Andean station using load_calibration_temperature_pair."""
    st_dir = base_dir / station_code
    if not st_dir.exists():
        raise FileNotFoundError(f"Station directory not found: {st_dir}")

    # Strictly locate unique CSV files (no silent fallback)
    tmax_files = list(st_dir.glob("TempMaxima_*.csv"))
    if len(tmax_files) != 1:
        raise ValueError(
            f"Expected exactly 1 TempMaxima CSV in {st_dir}, found {len(tmax_files)}."
        )
    tmin_files = list(st_dir.glob("TempMinima_*.csv"))
    if len(tmin_files) != 1:
        raise ValueError(
            f"Expected exactly 1 TempMinima CSV in {st_dir}, found {len(tmin_files)}."
        )

    tmax_csv = tmax_files[0]
    tmin_csv = tmin_files[0]

    # Collect hashes of observation CSV files and required ERA5 NetCDFs
    station_hashes = build_station_input_hashes(
        station_code=station_code,
        tmax_csv=tmax_csv,
        tmin_csv=tmin_csv,
        era5_dir=st_dir,
        cal_block=station_info["cal_block"],
    )

    # Load paired series strictly via load_calibration_temperature_pair (G.6.1 / H.5)
    obs_tmax, obs_tmin, mod_tmax_0, mod_tmin_0 = load_calibration_temperature_pair(
        station_id=station_code,
        obs_tmax_path=tmax_csv,
        obs_tmin_path=tmin_csv,
        era5_target=st_dir,
        window_start_utc=5,
        lag_days=0,
        min_hours=24,
    )

    # Derive ERA5 lag series across all audit lags (G.6.2, G.6.4 / H.5)
    # Under fixed window_start_utc=5, daily aggregation at lag k corresponds
    # identically to df.shift(k) in aggregate_hourly_to_daily_temperature.
    mod_tmax_by_lag: Dict[int, pd.Series] = {
        k: mod_tmax_0.shift(k) for k in AUDIT_LAGS
    }
    mod_tmin_by_lag: Dict[int, pd.Series] = {
        k: mod_tmin_0.shift(k) for k in AUDIT_LAGS
    }

    # Execute variable-level evaluations
    tmax_result = audit_variable_lag_calibration(
        obs=obs_tmax,
        mod_by_lag=mod_tmax_by_lag,
        candidate_lags=CANDIDATE_LAGS,
        audit_lags=AUDIT_LAGS,
        block_size=block_size,
        n_replicas=n_replicas,
        seed=seed,
    )
    tmin_result = audit_variable_lag_calibration(
        obs=obs_tmin,
        mod_by_lag=mod_tmin_by_lag,
        candidate_lags=CANDIDATE_LAGS,
        audit_lags=AUDIT_LAGS,
        block_size=block_size,
        n_replicas=n_replicas,
        seed=seed,
    )

    # Station-level sensitivity activation rule (G.6.4)
    all_triggered = sorted(
        list(set(tmax_result["triggered_lags"] + tmin_result["triggered_lags"]))
    )
    lag_sensitivity_activated = bool(len(all_triggered) > 0)

    cal_valid_years = [
        y
        for y in station_info["valid_years"]
        if station_info["cal_block"][0] <= y <= station_info["cal_block"][1]
    ]

    station_summary = {
        "station_code": station_code,
        "station_name": station_info["nombre"],
        "cal_block": station_info["cal_block"],
        "n_cal_years": len(cal_valid_years),
        "valid_cal_years": cal_valid_years,
        "tmax": tmax_result,
        "tmin": tmin_result,
        "lag_sensitivity_activated": lag_sensitivity_activated,
        "activated_lags": all_triggered,
    }

    return station_summary, station_hashes


def write_report_json(payload: Dict[str, Any], path: Path) -> None:
    """Serialize payload to JSON in memory with allow_nan=False, then write to path.

    Args:
        payload: Report data dictionary to serialize.
        path: Target file path to write.

    Raises:
        ValueError: If payload contains NaN, Infinity, or non-JSON compliant floats.
    """
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(serialized)
        f.write("\n")


def run_andean_calibration_lag_audit(
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    base_dir: Path = DEFAULT_BASE_DIR,
    output_path: Path = DEFAULT_OUTPUT_JSON,
    block_size: int = DEFAULT_BLOCK_SIZE,
    n_replicas: int = DEFAULT_REPLICAS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    """Execute complete Appendix G.6 lag audit across the five Andean stations."""
    if not registry_path.exists():
        raise FileNotFoundError(f"Andean registry not found: {registry_path}")

    with open(registry_path, encoding="utf-8") as f:
        registry_data = json.load(f)

    stations = registry_data["stations"]
    all_station_results: Dict[str, Any] = {}
    all_hashes: Dict[str, str] = {
        "registry": compute_file_sha256(registry_path)
    }

    # Verify working tree cleanliness without writing timestamp
    git_commit = get_current_git_commit()
    is_dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    )

    for code in sorted(stations.keys()):
        st_summary, st_hashes = audit_station_lag_calibration(
            station_code=code,
            station_info=stations[code],
            base_dir=base_dir,
            block_size=block_size,
            n_replicas=n_replicas,
            seed=seed,
        )
        all_station_results[code] = st_summary
        all_hashes.update(st_hashes)

    report_payload = {
        "commit_hash": git_commit,
        "config": {
            "protocol_section": "Appendix G.6 / H.5",
            "bootstrap_unit": "calendar_30d_moving_block",
            "block_size_days": block_size,
            "bootstrap_replicas": n_replicas,
            "bootstrap_seed": seed,
            "invalid_years_masked": True,
            "lag_convention": "model[t-k] paired with obs[t]",
            "window_start_utc": 5,
            "min_hours": 24,
            "audit_lags": AUDIT_LAGS,
            "candidate_lags": CANDIDATE_LAGS,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "numpy_version": str(np.__version__),
            "pandas_version": str(pd.__version__),
        },
        "input_hashes": {k: all_hashes[k] for k in sorted(all_hashes.keys())},
        "is_dirty": is_dirty,
        "stations": all_station_results,
    }

    write_report_json(report_payload, output_path)

    return report_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Andean temperature calibration lag under Protocol Appendix G.6 / H.5."
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY_PATH,
        help="Path to andean_station_registry.json.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Base directory containing local/era5_ts/<station> directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Path to output JSON report.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help="Block size in days for block bootstrap (default: 30).",
    )
    parser.add_argument(
        "--replicas",
        type=int,
        default=DEFAULT_REPLICAS,
        help="Number of bootstrap replicas (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
        help="Bootstrap pseudo-random seed (default: 42).",
    )
    args = parser.parse_args()

    result = run_andean_calibration_lag_audit(
        registry_path=args.registry,
        base_dir=args.base_dir,
        output_path=args.output,
        block_size=args.block_size,
        n_replicas=args.replicas,
        seed=args.seed,
    )
    print(f"Lag audit report written to: {args.output}")
    print(f"Stations evaluated: {len(result['stations'])}")
    for code, s in result["stations"].items():
        act = s["lag_sensitivity_activated"]
        lags = s["activated_lags"]
        print(f"  Station {code} ({s['station_name']}): sensitivity_activated={act}, lags={lags}")


if __name__ == "__main__":
    main()
