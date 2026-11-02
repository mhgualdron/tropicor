"""Calibration-years-only diagnostics for JB (21205710) and AC (23215030).

Covers:
  (1) Valid-year inventory under the J.7 criterion and diagnostic inclusion rule.
  (2) JB noise floor recomputed on included years only.
  (3) Step-versus-linear description of calibration residual series r_y.
  (4) DTR annual series and step-versus-linear description.
  (5) Valid-year criterion analysis: protocol text versus evaluator code.
  (6) Lists of included and excluded years with explicit reasons.

Rules:
  - Commit NOTHING.
  - Slice observations to calibration years only at load time.
  - Do NOT touch validation openings ledger.
  - Do NOT import phase3_evaluator or run audit scripts.
"""

from __future__ import annotations

import glob
import hashlib
import itertools
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from tropicor.downscale.temperature import load_calibration_temperature_pair


def compute_calendar_monthly_biases(
    obs: pd.Series, mod: pd.Series
) -> Dict[int, Optional[float]]:
    """Compute 12-month calendar mean biases: Bias_m = mean(Mod) - mean(Obs)."""
    valid_mask = obs.notna() & mod.notna()
    c_obs = obs[valid_mask]
    c_mod = mod[valid_mask]
    res: Dict[int, Optional[float]] = {}
    for m in range(1, 13):
        m_mask = c_obs.index.month == m
        if m_mask.any():
            res[m] = float(c_mod[m_mask].mean() - c_obs[m_mask].mean())
        else:
            res[m] = None
    return res

MIN_DAYS_PER_MONTH = 10
MIN_MONTHS_PER_YEAR = 9
BOOTSTRAP_SEED = 42

BENCHMARK_STATIONS = {
    "21205710": {
        "name": "JARDIN BOTANICO [21205710]",
        "tmax_csv": Path("local/newFiles/TempMaxima_JardinBotanico.csv"),
        "tmin_csv": Path("local/newFiles/TempMinima_JardinBotanico.csv"),
        "era5_dir": Path("local/era5_ts/21205710"),
        "cal_start": 1990,
        "cal_end": 2004,
    },
    "23215030": {
        "name": "AGUAS CLARAS [23215030]",
        "tmax_csv": Path("local/newFiles/TempMaxima_AguasClaras.csv"),
        "tmin_csv": Path("local/newFiles/TempMinima_AguasClaras.csv"),
        "era5_dir": Path("local/era5_ts/23215030"),
        "cal_start": 1990,
        "cal_end": 2000,
    },
}


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_noise_floor(
    inc_years: Sequence[int],
    data_by_ym: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]],
    k_cal: int,
    k_val: int,
    n_draws: int = 999,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    """Run random reassignments or full enumeration into pseudo-cal and pseudo-val partitions."""
    n_total = len(inc_years)
    comb_count = math.comb(n_total, k_val)
    enum_all = comb_count <= 999

    if enum_all:
        all_combinations = list(itertools.combinations(inc_years, k_val))
        mode_str = f"Enumerated ALL {len(all_combinations)} distinct assignments"
        assignments = []
        for p_val in all_combinations:
            p_val_set = set(p_val)
            p_cal_set = set(inc_years) - p_val_set
            assignments.append((p_cal_set, p_val_set))
    else:
        rng = np.random.default_rng(seed)
        mode_str = f"Random {n_draws} draws (seed {seed})"
        assignments = []
        for _ in range(n_draws):
            perm = rng.permutation(inc_years)
            assignments.append((set(perm[:k_cal]), set(perm[k_cal:k_cal + k_val])))

    noise_max_b: List[float] = []
    noise_abs_off: List[float] = []
    noise_seas_amp: List[float] = []

    for p_cal_yrs, p_val_yrs in assignments:
        b_pcal: Dict[int, Optional[float]] = {}
        for m in range(1, 13):
            m_mods = [data_by_ym[(int(y), m)][0] for y in p_cal_yrs if len(data_by_ym[(int(y), m)][0]) > 0]
            m_obss = [data_by_ym[(int(y), m)][1] for y in p_cal_yrs if len(data_by_ym[(int(y), m)][1]) > 0]
            if len(m_mods) > 0:
                b_pcal[m] = float(np.mean(np.concatenate(m_mods)) - np.mean(np.concatenate(m_obss)))
            else:
                b_pcal[m] = None

        b_pval_resids: List[float] = []
        for m in range(1, 13):
            if b_pcal[m] is None:
                continue
            m_mods = [data_by_ym[(int(y), m)][0] for y in p_val_yrs if len(data_by_ym[(int(y), m)][0]) > 0]
            m_obss = [data_by_ym[(int(y), m)][1] for y in p_val_yrs if len(data_by_ym[(int(y), m)][1]) > 0]
            if len(m_mods) > 0:
                val_m_bias = float(np.mean(np.concatenate(m_mods)) - np.mean(np.concatenate(m_obss)))
                b_pval_resids.append(val_m_bias - b_pcal[m])

        if len(b_pval_resids) > 0:
            arr = np.array(b_pval_resids)
            noise_max_b.append(float(np.max(np.abs(arr))))
            off = float(np.mean(arr))
            noise_abs_off.append(float(abs(off)))
            noise_seas_amp.append(float(np.max(np.abs(arr - off))))

    return {
        "n_used": n_total,
        "k_cal": k_cal,
        "k_val": k_val,
        "mode": mode_str,
        "max_monthly_bias_k": {
            "median": float(np.median(noise_max_b)),
            "p5": float(np.percentile(noise_max_b, 5)),
            "p95": float(np.percentile(noise_max_b, 95)),
        },
        "abs_offset_k": {
            "median": float(np.median(noise_abs_off)),
            "p95": float(np.percentile(noise_abs_off, 95)),
        },
        "seasonal_amplitude_k": {
            "median": float(np.median(noise_seas_amp)),
            "p95": float(np.percentile(noise_seas_amp, 95)),
        },
    }


def fit_step_vs_linear(
    years: np.ndarray,
    values: np.ndarray,
    n_permutations: int = 999,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    """Fit constant, linear OLS, and best single step model, with permutation test."""
    n = len(years)
    mu_0 = float(np.mean(values))
    sse_0 = float(np.sum((values - mu_0) ** 2))

    # Linear OLS
    poly = np.polyfit(years, values, 1)
    slope_dec = float(poly[0] * 10.0)
    pred_lin = np.polyval(poly, years)
    sse_lin = float(np.sum((values - pred_lin) ** 2))

    # Best step leaving at least 3 included years on each side
    best_break = None
    best_sse = float("inf")
    best_mu1, best_mu2 = None, None
    best_idx = None

    for k in range(3, n - 2):
        y_b = int(years[k - 1])
        g1, g2 = values[:k], values[k:]
        m1, m2 = float(np.mean(g1)), float(np.mean(g2))
        sse = float(np.sum((g1 - m1) ** 2) + np.sum((g2 - m2) ** 2))
        if sse < best_sse:
            best_sse = sse
            best_break = y_b
            best_mu1, best_mu2 = m1, m2
            best_idx = k

    step_size = float(best_mu2 - best_mu1)
    red_step = float(sse_0 - best_sse)

    # Permutation test
    rng_p = np.random.default_rng(seed)
    p_reds: List[float] = []
    for _ in range(n_permutations):
        p_vals = rng_p.permutation(values)
        p_best = float("inf")
        for k in range(3, n - 2):
            g1, g2 = p_vals[:k], p_vals[k:]
            sse_p = float(np.sum((g1 - np.mean(g1)) ** 2) + np.sum((g2 - np.mean(g2)) ** 2))
            if sse_p < p_best:
                p_best = sse_p
        p_reds.append(float(sse_0 - p_best))

    pct_rank = float(np.mean(np.array(p_reds) <= red_step) * 100.0)

    return {
        "n_years": n,
        "constant": {
            "mean": mu_0,
            "sse": sse_0,
        },
        "linear_ols": {
            "slope_k_per_decade": slope_dec,
            "intercept": float(poly[1]),
            "sse": sse_lin,
            "sse_reduction_vs_constant": sse_0 - sse_lin,
        },
        "best_step": {
            "break_year": best_break,
            "left_partition_end_year": best_break,
            "n_left": best_idx,
            "n_right": n - best_idx,
            "mean_left": best_mu1,
            "mean_right": best_mu2,
            "step_size": step_size,
            "sse": best_sse,
            "sse_reduction_vs_constant": red_step,
            "pct_sse_reduction": (red_step / sse_0) * 100.0,
            "permutation_percentile_rank": pct_rank,
            "median_perm_sse_reduction": float(np.median(p_reds)),
        },
        "sse_diff_linear_minus_step": sse_lin - best_sse,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    out_dir = Path("local/analysis_k/diagnostic2")
    out_dir.mkdir(parents=True, exist_ok=True)

    script_path = Path("local/analysis_k/diagnostic2/diagnostic_calibration_jb_ac.py")

    summary_data: Dict[str, Any] = {
        "script_path": str(script_path),
        "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "input_hashes": {},
    }

    # Hash input observation CSVs and ERA5 files
    for st, cfg in BENCHMARK_STATIONS.items():
        summary_data["input_hashes"][f"obs_tmax_{st}"] = compute_file_sha256(cfg["tmax_csv"])
        summary_data["input_hashes"][f"obs_tmin_{st}"] = compute_file_sha256(cfg["tmin_csv"])
        era5_files = sorted(glob.glob(str(cfg["era5_dir"] / "era5_2m_temperature_*.nc")))
        for f in era5_files:
            rel = Path(f).name
            summary_data["input_hashes"][f"era5_{st}_{rel}"] = compute_file_sha256(Path(f))

    # =========================================================================
    # ITEM 1: J.7 Inventory (Calibration Years Only)
    # =========================================================================
    print("=" * 100)
    print("ITEM 1: J.7 INVENTORY AND DIAGNOSTIC INCLUSION RULE (CALIBRATION YEARS ONLY)")
    print("=" * 100)

    inventory_rows: List[Dict[str, Any]] = []
    station_data: Dict[str, Any] = {}

    for st, cfg in BENCHMARK_STATIONS.items():
        obs_tx, obs_tn, mod_tx, mod_tn = load_calibration_temperature_pair(
            station_id=st,
            obs_tmax_path=cfg["tmax_csv"],
            obs_tmin_path=cfg["tmin_csv"],
            era5_target=cfg["era5_dir"],
            window_start_utc=5,
            lag_days=0,
            min_hours=24,
        )

        station_data[st] = {
            "obs_tx": obs_tx,
            "obs_tn": obs_tn,
            "mod_tx": mod_tx,
            "mod_tn": mod_tn,
            "cfg": cfg,
        }

        v4 = obs_tx.notna() & mod_tx.notna() & obs_tn.notna() & mod_tn.notna()
        cal_years = list(range(cfg["cal_start"], cfg["cal_end"] + 1))

        print(f"\n>>> Station {st} ({cfg['name']}) - Calibration Span: {cfg['cal_start']}-{cfg['cal_end']} <<<")
        print(f"{'Year':6s} | {'4-Way Days':10s} | {'J.7 Valid':9s} | {'Tmax Valid Mo':13s} | {'Tmax Inc':8s} | {'Tmin Valid Mo':13s} | {'Tmin Inc':8s} | {'DTR Valid Mo':12s} | {'DTR Inc':7s}")
        print("-" * 105)

        for y in cal_years:
            mask_y = obs_tx.index.year == y
            n_4way = int((mask_y & v4).sum())
            j7_valid = bool(n_4way >= 1)

            # Monthly counts under diagnostic rule (>= 10 days/mo, >= 9 mo/yr)
            mo_tx, mo_tn, mo_4way = 0, 0, 0
            for m in range(1, 13):
                m_mask = mask_y & (obs_tx.index.month == m)
                if (m_mask & obs_tx.notna() & mod_tx.notna()).sum() >= MIN_DAYS_PER_MONTH:
                    mo_tx += 1
                if (m_mask & obs_tn.notna() & mod_tn.notna()).sum() >= MIN_DAYS_PER_MONTH:
                    mo_tn += 1
                if (m_mask & v4).sum() >= MIN_DAYS_PER_MONTH:
                    mo_4way += 1

            inc_tx = bool(mo_tx >= MIN_MONTHS_PER_YEAR)
            inc_tn = bool(mo_tn >= MIN_MONTHS_PER_YEAR)
            inc_dtr = bool(mo_4way >= MIN_MONTHS_PER_YEAR)

            print(f"{y:6d} | {n_4way:10d} | {str(j7_valid):9s} | {mo_tx:13d} | {str(inc_tx):8s} | {mo_tn:13d} | {str(inc_tn):8s} | {mo_4way:12d} | {str(inc_dtr):7s}")

            inventory_rows.append({
                "station_id": st,
                "station_name": cfg["name"],
                "year": y,
                "partition": "calibration",
                "n_days_all_four_valid": n_4way,
                "valid_under_j7": j7_valid,
                "tmax_valid_months": mo_tx,
                "tmax_included": inc_tx,
                "tmin_valid_months": mo_tn,
                "tmin_included": inc_tn,
                "dtr_valid_months": mo_4way,
                "dtr_included": inc_dtr,
            })

    df_inv = pd.DataFrame(inventory_rows)
    inv_csv_path = out_dir / "jb_ac_valid_years_inventory.csv"
    df_inv.to_csv(inv_csv_path, index=False)
    summary_data["item1_inventory"] = inventory_rows

    # Summary counts per station and variable
    print("\nSummary Counts of Included Years (Diagnostic Rule: >= 10 days/mo in >= 9 mo/yr):")
    for st, cfg in BENCHMARK_STATIONS.items():
        sub = df_inv[df_inv["station_id"] == st]
        n_j7 = int(sub["valid_under_j7"].sum())
        n_tx = int(sub["tmax_included"].sum())
        n_tn = int(sub["tmin_included"].sum())
        n_dtr = int(sub["dtr_included"].sum())
        total_yrs = len(sub)
        print(f"  Station {st} ({cfg['name']}): Total {total_yrs} nominal calibration years")
        print(f"    - J.7 Valid Years (>= 1 four-way day): {n_j7} / {total_yrs}")
        print(f"    - Tmax Included Years (>= 9 valid months): {n_tx} / {total_yrs}")
        print(f"    - Tmin Included Years (>= 9 valid months): {n_tn} / {total_yrs}")
        print(f"    - DTR Included Years (>= 9 valid months):  {n_dtr} / {total_yrs}")

    # =========================================================================
    # ITEM 2: JB Noise Floor on Included Years Only
    # =========================================================================
    print("\n" + "=" * 100)
    print("ITEM 2: JB NOISE FLOOR ON INCLUDED YEARS ONLY (COMPARED WITH PREVIOUS 15-YR RUN)")
    print("=" * 100)

    jb_cfg = BENCHMARK_STATIONS["21205710"]
    obs_tx = station_data["21205710"]["obs_tx"]
    obs_tn = station_data["21205710"]["obs_tn"]
    mod_tx = station_data["21205710"]["mod_tx"]
    mod_tn = station_data["21205710"]["mod_tn"]

    sub_jb = df_inv[df_inv["station_id"] == "21205710"]
    inc_years_tx = sub_jb[sub_jb["tmax_included"]]["year"].values.tolist()
    inc_years_tn = sub_jb[sub_jb["tmin_included"]]["year"].values.tolist()

    noise_floor_results: Dict[str, Any] = {}
    noise_floor_csv_rows: List[Dict[str, Any]] = []

    # Previous 15-year values from summary.json
    prev_15yr = {
        "Tmax": {
            "balanced": {"max_b_med": 0.6636, "max_b_p5": 0.4321, "max_b_p95": 0.9899, "off_med": 0.0910, "off_p95": 0.2683, "sea_med": 0.6280, "sea_p95": 0.8950},
            "pseudo_val_5yr": {"max_b_med": 0.7630, "max_b_p5": 0.5042, "max_b_p95": 1.2242, "off_med": 0.1125, "off_p95": 0.3064, "sea_med": 0.7256, "sea_p95": 1.0996},
        },
        "Tmin": {
            "balanced": {"max_b_med": 0.8119, "max_b_p5": 0.5272, "max_b_p95": 1.2586, "off_med": 0.1221, "off_p95": 0.3282, "sea_med": 0.7586, "sea_p95": 1.1326},
            "pseudo_val_5yr": {"max_b_med": 1.0599, "max_b_p5": 0.6603, "max_b_p95": 1.6378, "off_med": 0.1437, "off_p95": 0.3864, "sea_med": 0.9662, "sea_p95": 1.6020},
        },
    }

    # AC values (11 complete years, unchanged)
    ac_prev = {
        "Tmax": {
            "balanced": {"n_used": 11, "mode": "Enumerated ALL 462 distinct assignments", "max_b_med": 0.7529, "max_b_p5": 0.4310, "max_b_p95": 1.1328, "off_med": 0.1333, "off_p95": 0.3637, "sea_med": 0.6387, "sea_p95": 0.9647},
            "pseudo_val_5yr": {"n_used": 11, "mode": "Enumerated ALL 462 distinct assignments", "max_b_med": 0.7529, "max_b_p5": 0.4310, "max_b_p95": 1.1328, "off_med": 0.1333, "off_p95": 0.3637, "sea_med": 0.6387, "sea_p95": 0.9647},
        },
        "Tmin": {
            "balanced": {"n_used": 11, "mode": "Enumerated ALL 462 distinct assignments", "max_b_med": 0.4290, "max_b_p5": 0.2724, "max_b_p95": 0.6491, "off_med": 0.0960, "off_p95": 0.2794, "sea_med": 0.3437, "sea_p95": 0.5306},
            "pseudo_val_5yr": {"n_used": 11, "mode": "Enumerated ALL 462 distinct assignments", "max_b_med": 0.4290, "max_b_p5": 0.2724, "max_b_p95": 0.6491, "off_med": 0.0960, "off_p95": 0.2794, "sea_med": 0.3437, "sea_p95": 0.5306},
        },
    }

    for var_name, obs_s, mod_s, inc_yrs in [
        ("Tmax", obs_tx, mod_tx, inc_years_tx),
        ("Tmin", obs_tn, mod_tn, inc_years_tn),
    ]:
        data_ym: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
        for y in inc_yrs:
            mask_y = obs_s.index.year == y
            for m in range(1, 13):
                m_mask = mask_y & (obs_s.index.month == m)
                v = obs_s[m_mask].notna() & mod_s[m_mask].notna()
                if v.sum() > 0:
                    data_ym[(y, m)] = (mod_s[m_mask][v].values, obs_s[m_mask][v].values)
                else:
                    data_ym[(y, m)] = (np.array([]), np.array([]))

        n = len(inc_yrs)
        k_cal_bal = int(math.ceil(n / 2.0))
        k_val_bal = int(math.floor(n / 2.0))
        res_bal = run_noise_floor(inc_yrs, data_ym, k_cal_bal, k_val_bal)

        k_val_5 = 5
        k_cal_5 = n - 5
        res_5 = run_noise_floor(inc_yrs, data_ym, k_cal_5, k_val_5)

        noise_floor_results[var_name] = {
            "n_included_years": n,
            "included_years": inc_yrs,
            "balanced": res_bal,
            "pseudo_val_5yr": res_5,
            "previous_15yr": prev_15yr[var_name],
        }

        for cfg_name, res_c in [("balanced", res_bal), ("pseudo_val_5yr", res_5)]:
            prev_c = prev_15yr[var_name][cfg_name]
            noise_floor_csv_rows.append({
                "station_id": "21205710",
                "station_name": jb_cfg["name"],
                "variable": var_name,
                "configuration": cfg_name,
                "n_years_used": n,
                "mode": res_c["mode"],
                "max_b_median": res_c["max_monthly_bias_k"]["median"],
                "max_b_p5": res_c["max_monthly_bias_k"]["p5"],
                "max_b_p95": res_c["max_monthly_bias_k"]["p95"],
                "abs_off_median": res_c["abs_offset_k"]["median"],
                "abs_off_p95": res_c["abs_offset_k"]["p95"],
                "seas_amp_median": res_c["seasonal_amplitude_k"]["median"],
                "seas_amp_p95": res_c["seasonal_amplitude_k"]["p95"],
                "prev_15yr_max_b_median": prev_c["max_b_med"],
                "prev_15yr_max_b_p5": prev_c["max_b_p5"],
                "prev_15yr_max_b_p95": prev_c["max_b_p95"],
                "prev_15yr_abs_off_median": prev_c["off_med"],
                "prev_15yr_abs_off_p95": prev_c["off_p95"],
                "prev_15yr_seas_amp_median": prev_c["sea_med"],
                "prev_15yr_seas_amp_p95": prev_c["sea_p95"],
            })

    # Print Table comparing new (included years) with previous 15yr
    print(f"{'Station & Var':18s} | {'Config':14s} | {'n':2s} | {'Mode':44s} | {'max_b med':9s} | {'P5':6s} | {'P95':6s} | {'|off| med':9s} | {'P95':6s} | {'seas med':8s} | {'P95':6s} | {'Prev max_b med (15yr)':22s}")
    print("-" * 175)
    for r in noise_floor_csv_rows:
        lbl = f"{r['station_id']} {r['variable']}"
        print(f"{lbl:18s} | {r['configuration']:14s} | {r['n_years_used']:2d} | {r['mode']:44s} | {r['max_b_median']:9.4f} | {r['max_b_p5']:6.4f} | {r['max_b_p95']:6.4f} | {r['abs_off_median']:9.4f} | {r['abs_off_p95']:6.4f} | {r['seas_amp_median']:8.4f} | {r['seas_amp_p95']:6.4f} | {r['prev_15yr_max_b_median']:22.4f}")

    # Re-print AC values unchanged
    print("\nAguas Claras (23215030, 11 complete calibration years) - Re-printed Unchanged:")
    for var_name in ["Tmax", "Tmin"]:
        for cfg_name in ["balanced", "pseudo_val_5yr"]:
            ac_c = ac_prev[var_name][cfg_name]
            print(f"  23215030 {var_name:4s} | {cfg_name:14s} | n={ac_c['n_used']:2d} | {ac_c['mode']:44s} | max_b med={ac_c['max_b_med']:.4f}, P5={ac_c['max_b_p5']:.4f}, P95={ac_c['max_b_p95']:.4f} | |off| med={ac_c['off_med']:.4f}, P95={ac_c['off_p95']:.4f} | seas med={ac_c['sea_med']:.4f}, P95={ac_c['sea_p95']:.4f}")

    df_nf = pd.DataFrame(noise_floor_csv_rows)
    nf_csv_path = out_dir / "jb_noise_floor_included_years.csv"
    df_nf.to_csv(nf_csv_path, index=False)
    summary_data["item2_jb_noise_floor_included_years"] = noise_floor_results
    summary_data["item2_ac_noise_floor_unchanged"] = ac_prev

    # =========================================================================
    # ITEM 3: Step versus Linear, Calibration Residuals r_y
    # =========================================================================
    print("\n" + "=" * 100)
    print("ITEM 3: STEP VERSUS LINEAR DESCRIPTION OF CALIBRATION RESIDUALS r_y (INCLUDED YEARS)")
    print("=" * 100)

    step_vs_lin_results: Dict[str, Any] = {}
    step_vs_lin_rows: List[Dict[str, Any]] = []

    for st in ["21205710", "23215030"]:
        df_ann = pd.read_csv(f"local/analysis_k/diagnostic2/calibration_annual_{st}.csv")
        st_name = df_ann["station_name"].iloc[0]
        step_vs_lin_results[st] = {"station_name": st_name, "variables": {}}

        print(f"\n>>> Station {st} ({st_name}) <<<")
        for var in ["Tmax", "Tmin"]:
            sub = df_ann[(df_ann["variable"] == var) & (df_ann["included_in_stats"] == True)]
            years_arr = sub["year"].values.astype(int)
            r_arr = sub["level1_residual_r_y"].values.astype(float)

            res_step = fit_step_vs_linear(years_arr, r_arr, n_permutations=999, seed=BOOTSTRAP_SEED)
            step_vs_lin_results[st]["variables"][var] = res_step

            c = res_step["constant"]
            l = res_step["linear_ols"]
            s = res_step["best_step"]

            print(f"\n--- Variable: {var} (n={res_step['n_years']} included years: {years_arr[0]}-{years_arr[-1]}) ---")
            print(f"  Constant Model: SSE_0 = {c['sse']:.6f} (mean = {c['mean']:+.4f} K)")
            print(f"  Linear OLS:     Slope = {l['slope_k_per_decade']:+.4f} K/decade | SSE_lin = {l['sse']:.6f} | Reduction = {l['sse_reduction_vs_constant']:.6f}")
            print(f"  Best Step:      Break after {s['break_year']} (<= {s['break_year']} [n={s['n_left']}] vs > {s['break_year']} [n={s['n_right']}])")
            print(f"                  Left mean mu1 = {s['mean_left']:+.4f} K | Right mean mu2 = {s['mean_right']:+.4f} K | Step size = {s['step_size']:+.4f} K")
            print(f"                  SSE_step = {s['sse']:.6f} | Reduction = {s['sse_reduction_vs_constant']:.6f} ({s['pct_sse_reduction']:.2f}%)")
            print(f"                  Permutation Rank: {s['permutation_percentile_rank']:.2f}% (median perm red = {s['median_perm_sse_reduction']:.6f})")
            print(f"  Comparison:     SSE_lin - SSE_step = {res_step['sse_diff_linear_minus_step']:+.6f}")

            step_vs_lin_rows.append({
                "station_id": st,
                "station_name": st_name,
                "variable": var,
                "n_years": res_step["n_years"],
                "span": f"{years_arr[0]}-{years_arr[-1]}",
                "constant_sse": c["sse"],
                "linear_slope_k_per_decade": l["slope_k_per_decade"],
                "linear_sse": l["sse"],
                "linear_sse_reduction": l["sse_reduction_vs_constant"],
                "step_break_year": s["break_year"],
                "step_n_left": s["n_left"],
                "step_n_right": s["n_right"],
                "step_mean_left": s["mean_left"],
                "step_mean_right": s["mean_right"],
                "step_size_k": s["step_size"],
                "step_sse": s["sse"],
                "step_sse_reduction": s["sse_reduction_vs_constant"],
                "step_pct_sse_reduction": s["pct_sse_reduction"],
                "step_perm_percentile_rank": s["permutation_percentile_rank"],
                "sse_diff_linear_minus_step": res_step["sse_diff_linear_minus_step"],
            })

    df_step_csv = pd.DataFrame(step_vs_lin_rows)
    step_csv_path = out_dir / "jb_ac_step_vs_linear.csv"
    df_step_csv.to_csv(step_csv_path, index=False)
    summary_data["item3_step_vs_linear"] = step_vs_lin_results

    # =========================================================================
    # ITEM 4: DTR Annual Series and Step vs Linear
    # =========================================================================
    print("\n" + "=" * 100)
    print("ITEM 4: DTR ANNUAL SERIES AND STEP-VERSUS-LINEAR DESCRIPTION (CALIBRATION YEARS ONLY)")
    print("=" * 100)

    dtr_results: Dict[str, Any] = {}
    dtr_annual_rows: List[Dict[str, Any]] = []

    for st, cfg in BENCHMARK_STATIONS.items():
        obs_tx = station_data[st]["obs_tx"]
        obs_tn = station_data[st]["obs_tn"]
        mod_tx = station_data[st]["mod_tx"]
        mod_tn = station_data[st]["mod_tn"]

        v4 = obs_tx.notna() & mod_tx.notna() & obs_tn.notna() & mod_tn.notna()
        dtr_obs = obs_tx - obs_tn
        dtr_mod = mod_tx - mod_tn
        dtr_bias_daily = dtr_mod - dtr_obs

        cal_years = list(range(cfg["cal_start"], cfg["cal_end"] + 1))
        ann_dtr_station: List[Dict[str, Any]] = []

        print(f"\n>>> Station {st} ({cfg['name']}) - DTR Calibration Annual Series <<<")
        print(f"{'Year':6s} | {'Partition':11s} | {'4-Way Valid Days':16s} | {'Valid Months':12s} | {'d_y DTR Bias (K)':18s} | {'Included':8s}")
        print("-" * 80)

        for y in cal_years:
            mask_y = obs_tx.index.year == y
            n_days = int((mask_y & v4).sum())
            m_biases = []
            for m in range(1, 13):
                m_mask = mask_y & (obs_tx.index.month == m) & v4
                if m_mask.sum() >= MIN_DAYS_PER_MONTH:
                    m_biases.append(float(dtr_bias_daily[m_mask].mean()))
            n_months = len(m_biases)
            inc = bool(n_months >= MIN_MONTHS_PER_YEAR)
            d_y_dtr = float(np.mean(m_biases)) if inc else np.nan

            d_str = f"{d_y_dtr:+.4f}" if inc else "       NaN        "
            print(f"{y:6d} | {'calibration':11s} | {n_days:16d} | {n_months:12d} | {d_str:18s} | {str(inc):8s}")

            row_rec = {
                "station_id": st,
                "station_name": cfg["name"],
                "year": y,
                "partition": "calibration",
                "n_valid_days_four_way": n_days,
                "n_valid_months": n_months,
                "d_y_dtr_bias_k": d_y_dtr,
                "included_in_stats": inc,
            }
            ann_dtr_station.append(row_rec)
            dtr_annual_rows.append(row_rec)

        df_st_dtr = pd.DataFrame(ann_dtr_station)
        sub_dtr = df_st_dtr[df_st_dtr["included_in_stats"] == True]
        years_dtr = sub_dtr["year"].values.astype(int)
        vals_dtr = sub_dtr["d_y_dtr_bias_k"].values.astype(float)
        n_dtr = len(years_dtr)

        # OLS slope & 95% bootstrap CI (1000 resamples, seed 42)
        poly = np.polyfit(years_dtr, vals_dtr, 1)
        slope_dec = float(poly[0] * 10.0)

        rng_b = np.random.default_rng(BOOTSTRAP_SEED)
        b_slopes: List[float] = []
        for _ in range(1000):
            idx = rng_b.choice(n_dtr, size=n_dtr, replace=True)
            b_slopes.append(float(np.polyfit(years_dtr[idx], vals_dtr[idx], 1)[0] * 10.0))
        ci_slope = [float(np.percentile(b_slopes, 2.5)), float(np.percentile(b_slopes, 97.5))]

        # Step vs linear
        res_step_dtr = fit_step_vs_linear(years_dtr, vals_dtr, n_permutations=999, seed=BOOTSTRAP_SEED)

        c = res_step_dtr["constant"]
        l = res_step_dtr["linear_ols"]
        s = res_step_dtr["best_step"]

        print(f"\nDTR Statistics ({st_name}):")
        print(f"  OLS Slope: {slope_dec:+.4f} K/decade | 95% CI: [{ci_slope[0]:+.4f}, {ci_slope[1]:+.4f}] K/decade")
        print(f"  Constant Model: SSE_0 = {c['sse']:.6f}")
        print(f"  Linear OLS:     SSE_lin = {l['sse']:.6f} | Reduction = {l['sse_reduction_vs_constant']:.6f}")
        print(f"  Best Step:      Break after {s['break_year']} (<= {s['break_year']} [n={s['n_left']}] vs > {s['break_year']} [n={s['n_right']}])")
        print(f"                  Left mean mu1 = {s['mean_left']:+.4f} K | Right mean mu2 = {s['mean_right']:+.4f} K | Step size = {s['step_size']:+.4f} K")
        print(f"                  SSE_step = {s['sse']:.6f} | Reduction = {s['sse_reduction_vs_constant']:.6f} ({s['pct_sse_reduction']:.2f}%)")
        print(f"                  Permutation Rank: {s['permutation_percentile_rank']:.2f}% (median perm red = {s['median_perm_sse_reduction']:.6f})")
        print(f"  Comparison:     SSE_lin - SSE_step = {res_step_dtr['sse_diff_linear_minus_step']:+.6f}")

        dtr_results[st] = {
            "station_name": cfg["name"],
            "n_included_years": n_dtr,
            "included_years": years_dtr.tolist(),
            "ols_slope_k_per_decade": slope_dec,
            "ols_slope_95ci_k_per_decade": ci_slope,
            "annual_series": ann_dtr_station,
            "step_vs_linear": res_step_dtr,
        }

    df_dtr_csv = pd.DataFrame(dtr_annual_rows)
    dtr_csv_path = out_dir / "jb_ac_dtr_annual_series.csv"
    df_dtr_csv.to_csv(dtr_csv_path, index=False)
    summary_data["item4_dtr"] = dtr_results

    # =========================================================================
    # ITEM 6: Lists of Included and Excluded Years with Reasons
    # =========================================================================
    print("\n" + "=" * 100)
    print("ITEM 6: LISTS OF INCLUDED AND EXCLUDED CALIBRATION YEARS WITH REASONS")
    print("=" * 100)

    item6_records: Dict[str, Any] = {}

    for st, cfg in BENCHMARK_STATIONS.items():
        sub = df_inv[df_inv["station_id"] == st]
        item6_records[st] = {"station_name": cfg["name"], "variables": {}}
        print(f"\n>>> Station {st} ({cfg['name']}) <<<")

        for var, inc_col, mo_col in [
            ("Tmax", "tmax_included", "tmax_valid_months"),
            ("Tmin", "tmin_included", "tmin_valid_months"),
            ("DTR", "dtr_included", "dtr_valid_months"),
        ]:
            inc_sub = sub[sub[inc_col] == True]
            exc_sub = sub[sub[inc_col] == False]

            inc_list = inc_sub["year"].tolist()
            exc_dict = {}

            for _, r in exc_sub.iterrows():
                y = int(r["year"])
                n4 = int(r["n_days_all_four_valid"])
                nm = int(r[mo_col])
                if n4 == 0:
                    reason = f"zero days with all four values valid (0 valid days, 0 valid months)"
                else:
                    reason = f"fewer than 9 valid months ({nm} valid months with >= 10 valid days; {n4} valid days)"
                exc_dict[y] = reason

            print(f"  {var:4s}: Included ({len(inc_list)} years): {inc_list}")
            print(f"        Excluded ({len(exc_dict)} years):")
            for y, rsn in exc_dict.items():
                print(f"          - Year {y}: {rsn}")

            item6_records[st]["variables"][var] = {
                "n_included": len(inc_list),
                "included_years": inc_list,
                "n_excluded": len(exc_dict),
                "excluded_years_with_reasons": exc_dict,
            }

    summary_data["item6_year_lists"] = item6_records

    # Save summary JSON
    summary_path = out_dir / "calibration_jb_ac_summary.json"
    summary_data["script_sha256"] = compute_file_sha256(script_path)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print("\n" + "=" * 100)
    print("OUTPUT MANIFEST AND SHA-256 HASHES")
    print("=" * 100)
    outputs = [
        script_path,
        inv_csv_path,
        nf_csv_path,
        step_csv_path,
        dtr_csv_path,
        summary_path,
    ]
    for p in outputs:
        print(f"  {str(p):65s} | SHA-256: {compute_file_sha256(p)}")


if __name__ == "__main__":
    main()
