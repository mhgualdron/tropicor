#!/usr/bin/env python3
"""Smoke test for daily precipitation downscaling hierarchy on calibration data.

Evaluates Level 0 (Raw ERA5), Level 1 (Linear Scaling), Level 2 (Threshold Matching),
and Level 3 (EQM) strictly on in-sample calibration partitions:
- Jardín Botánico (21205710): 1975–1994 (20 years)
- Aguas Claras (23215030): 1975–1987 (13 years)

Zero Validation Access:
Ingests data exclusively through `load_calibration_precipitation_pair`.
Validation years are discarded immediately after reading without computing anything with them.
"""

from __future__ import annotations

import sys
from typing import Dict

import numpy as np
import pandas as pd

from tropicor.core.protocol import (
    DEFAULT_LEDGER_PATH,
    apply_paired_mask,
    compute_seasonal_wet_day_frequency,
    compute_wet_day_frequency,
    compute_wet_day_percentiles,
)
from tropicor.downscale.precipitation import (
    Level1LinearScalingPrecipitation,
    Level2ThresholdMatchingPrecipitation,
    load_calibration_precipitation_pair,
    tune_level3_precipitation_hyperparameters,
)


def run_smoke_test_station(station_id: str, station_name: str) -> None:
    print("=" * 80)
    print(f"STATION: {station_name} [{station_id}] - CALIBRATION DIAGNOSTICS")
    print("=" * 80)

    # 1. Load data strictly through load_calibration_precipitation_pair
    obs_cal, mod_cal = load_calibration_precipitation_pair(station_id)

    # Paired synchronous mask
    df_paired = pd.DataFrame({"obs": obs_cal, "mod": mod_cal}).dropna()
    dt_idx = pd.DatetimeIndex(df_paired.index)
    valid_years = sorted(dt_idx.year.unique())
    n_valid_years = len(valid_years)
    n_paired_days = len(df_paired)
    n_wet_obs = int((df_paired["obs"] >= 1.0).sum())
    f_wet_obs_total = n_wet_obs / n_paired_days if n_paired_days > 0 else 0.0

    print("\n--- 1. Data Integrity & Sample Size ---")
    print(
        f"Calibration Span:         {obs_cal.index.min().strftime('%Y-%m-%d')} to {obs_cal.index.max().strftime('%Y-%m-%d')}"
    )
    print(f"Total Calendar Days:      {len(obs_cal)}")
    print(f"Valid Paired Days:        {n_paired_days}")
    print(
        f"Valid Years (>=1 pair):   {n_valid_years} ({valid_years[0]}–{valid_years[-1]})"
    )
    print(f"Observed Wet Days (>=1mm):{n_wet_obs}")
    print(
        f"Observed Wet Frequency:   {f_wet_obs_total:.4f} ({f_wet_obs_total * 100:.2f}%)"
    )

    # 2. Fit Downscalers
    # Level 1
    l1 = Level1LinearScalingPrecipitation()
    l1.fit(obs_cal, mod_cal)
    pred_l1 = l1.transform(mod_cal)

    # Level 2
    l2 = Level2ThresholdMatchingPrecipitation()
    l2.fit(obs_cal, mod_cal)
    pred_l2 = l2.transform(mod_cal)

    # Level 3 Grid Search & Winner
    print(
        "\n--- 2. Level 3 Hyperparameter Grid Optimization (Nested Annual Block CV) ---"
    )
    l3_winner, l3_results, l3_best = tune_level3_precipitation_hyperparameters(
        obs_cal, mod_cal
    )
    pred_l3 = l3_winner.transform(mod_cal)

    print(
        f"{'ID':<3} | {'Window':<17} | {'Bins':<4} | {'Extrap':<8} | {'Loss':<10} | {'Std_Fold':<10} | {'MinWet':<6} | {'Clipped'}"
    )
    print("-" * 80)
    for r in l3_results:
        marker = " <-- WINNER" if r.config_id == l3_best.config_id else ""
        print(
            f"{r.config_id:<3} | {r.window:<17} | {r.n_quantiles:<4} | {r.extrapolation:<8} | "
            f"{r.overall_tuning_loss:<10.4f} | {r.std_fold_loss:<10.4f} | {r.min_n_wet_per_group:<6} | "
            f"{r.n_clipped_below_zero}{marker}"
        )

    print(
        f"\nWinning Level 3 Configuration: ID {l3_best.config_id} ({l3_best.window}, {l3_best.n_quantiles} bins, {l3_best.extrapolation})"
    )
    print(
        f"Winning Concatenated Loss:    {l3_best.overall_tuning_loss:.4f} (std across annual folds: {l3_best.std_fold_loss:.4f})"
    )

    # 3. Parameters of Level 1 and Level 2
    print("\n--- 3. Monthly Parameters (Levels 1 & 2) ---")
    print(
        f"{'Month':<6} | {'s_m (L1)':<10} | {'f_wet_obs':<10} | {'P_th,m (mm)':<12} | {'s_m (L2)':<10}"
    )
    print("-" * 55)
    for m in range(1, 13):
        print(
            f"{m:<6} | {l1.monthly_factors.get(m, 0.0):<10.4f} | "
            f"{l2.monthly_observed_f_wet.get(m, 0.0):<10.4f} | "
            f"{l2.monthly_thresholds.get(m, 0.0):<12.4f} | "
            f"{l2.monthly_scaling_factors.get(m, 0.0):<10.4f}"
        )

    # 4. Calibration Mass Balance Check
    print("\n--- 4. Calibration Mass Balance Verification ---")
    sum_obs = float(df_paired["obs"].sum())
    sum_raw = float(df_paired["mod"].sum())
    _, c_l1 = apply_paired_mask(obs_cal, pred_l1)
    _, c_l2 = apply_paired_mask(obs_cal, pred_l2)
    _, c_l3 = apply_paired_mask(obs_cal, pred_l3)

    sum_l1 = float(c_l1.sum())
    sum_l2 = float(c_l2.sum())
    sum_l3 = float(c_l3.sum())

    print(f"Observed Total Accumulation:  {sum_obs:10.2f} mm")
    print(
        f"Raw Model (L0) Accumulation:  {sum_raw:10.2f} mm (Ratio: {sum_raw / sum_obs:.4f})"
    )
    print(
        f"Level 1 Corrected:            {sum_l1:10.2f} mm (Diff: {sum_l1 - sum_obs:+.6f} mm)"
    )
    print(
        f"Level 2 Corrected:            {sum_l2:10.2f} mm (Diff: {sum_l2 - sum_obs:+.6f} mm)"
    )
    print(
        f"Level 3 Corrected:            {sum_l3:10.2f} mm (Diff: {sum_l3 - sum_obs:+.2f} mm, Ratio: {sum_l3 / sum_obs:.4f})"
    )

    # 5. In-Sample Decisive & Descriptive Metrics by Level (0..3)
    levels: Dict[str, pd.Series] = {
        "Level 0 (Raw)": mod_cal,
        "Level 1 (LS)": pred_l1,
        "Level 2 (TMS)": pred_l2,
        "Level 3 (EQM)": pred_l3,
    }

    obs_seasonal = compute_seasonal_wet_day_frequency(df_paired["obs"])
    pct_obs = compute_wet_day_percentiles(df_paired["obs"], percentiles=(95.0, 99.0))
    p95_obs = pct_obs[95.0]
    p99_obs = pct_obs[99.0]

    print("\n--- 5. In-Sample Metrics (Levels 0 to 3) ---")
    print(
        f"{'Metric':<25} | {'Observed':<10} | {'Level 0 (Raw)':<13} | {'Level 1 (LS)':<13} | {'Level 2 (TMS)':<13} | {'Level 3 (EQM)':<13}"
    )
    print("-" * 105)

    # f_wet total
    f_wet_vals = {}
    for name, s in levels.items():
        _, clean_s = apply_paired_mask(obs_cal, s)
        f_wet_vals[name] = compute_wet_day_frequency(clean_s)
    print(
        f"{'f_wet (Total)':<25} | {f_wet_obs_total:<10.4f} | "
        f"{f_wet_vals['Level 0 (Raw)']:<13.4f} | {f_wet_vals['Level 1 (LS)']:<13.4f} | "
        f"{f_wet_vals['Level 2 (TMS)']:<13.4f} | {f_wet_vals['Level 3 (EQM)']:<13.4f}"
    )

    # Seasonal f_wet
    for season in ["DJF", "MAM", "JJA", "SON"]:
        s_obs_val = obs_seasonal[season]
        s_vals = {}
        for name, s in levels.items():
            _, clean_s = apply_paired_mask(obs_cal, s)
            s_vals[name] = compute_seasonal_wet_day_frequency(clean_s)[season]
        print(
            f"{f'f_wet ({season})':<25} | {s_obs_val:<10.4f} | "
            f"{s_vals['Level 0 (Raw)']:<13.4f} | {s_vals['Level 1 (LS)']:<13.4f} | "
            f"{s_vals['Level 2 (TMS)']:<13.4f} | {s_vals['Level 3 (EQM)']:<13.4f}"
        )

    # P95 and P95 PBIAS
    p95_vals = {}
    p95_pbias = {}
    for name, s in levels.items():
        _, clean_s = apply_paired_mask(obs_cal, s)
        pct = compute_wet_day_percentiles(clean_s, percentiles=(95.0, 99.0))
        p95_vals[name] = pct[95.0]
        p95_pbias[name] = (
            (pct[95.0] - p95_obs) / p95_obs * 100.0 if p95_obs > 0 else np.nan
        )

    print(
        f"{'P95 (mm)':<25} | {p95_obs:<10.2f} | "
        f"{p95_vals['Level 0 (Raw)']:<13.2f} | {p95_vals['Level 1 (LS)']:<13.2f} | "
        f"{p95_vals['Level 2 (TMS)']:<13.2f} | {p95_vals['Level 3 (EQM)']:<13.2f}"
    )
    print(
        f"{'PBIAS(P95) (%)':<25} | {'0.00%':<10} | "
        f"{p95_pbias['Level 0 (Raw)']:<+12.2f}% | {p95_pbias['Level 1 (LS)']:<+12.2f}% | "
        f"{p95_pbias['Level 2 (TMS)']:<+12.2f}% | {p95_pbias['Level 3 (EQM)']:<+12.2f}%"
    )

    # P99 and P99 PBIAS
    p99_vals = {}
    p99_pbias = {}
    for name, s in levels.items():
        _, clean_s = apply_paired_mask(obs_cal, s)
        pct = compute_wet_day_percentiles(clean_s, percentiles=(95.0, 99.0))
        p99_vals[name] = pct[99.0]
        p99_pbias[name] = (
            (pct[99.0] - p99_obs) / p99_obs * 100.0 if p99_obs > 0 else np.nan
        )

    print(
        f"{'P99 (mm)':<25} | {p99_obs:<10.2f} | "
        f"{p99_vals['Level 0 (Raw)']:<13.2f} | {p99_vals['Level 1 (LS)']:<13.2f} | "
        f"{p99_vals['Level 2 (TMS)']:<13.2f} | {p99_vals['Level 3 (EQM)']:<13.2f}"
    )
    print(
        f"{'PBIAS(P99) (%)':<25} | {'0.00%':<10} | "
        f"{p99_pbias['Level 0 (Raw)']:<+12.2f}% | {p99_pbias['Level 1 (LS)']:<+12.2f}% | "
        f"{p99_pbias['Level 2 (TMS)']:<+12.2f}% | {p99_pbias['Level 3 (EQM)']:<+12.2f}%"
    )

    # 6. Fraction of values in (0, 1.0) mm (Trace drizzle)
    print(
        "\n--- 6. Fraction of Corrected Values in (0, 1.0) mm (Trace/Drizzle Spectrum) ---"
    )
    n_obs_trace = ((df_paired["obs"] > 0.0) & (df_paired["obs"] < 1.0)).sum()
    frac_obs_trace = n_obs_trace / n_paired_days if n_paired_days > 0 else 0.0

    trace_fractions = {}
    for name, s in levels.items():
        _, clean_s = apply_paired_mask(obs_cal, s)
        n_trace = ((clean_s > 0.0) & (clean_s < 1.0)).sum()
        trace_fractions[name] = n_trace / len(clean_s) if len(clean_s) > 0 else 0.0

    print(
        f"Observed Trace Fraction (0, 1mm): {frac_obs_trace:.4f} ({frac_obs_trace * 100:.2f}%) [{n_obs_trace}/{n_paired_days} days]"
    )
    for name, f_tr in trace_fractions.items():
        print(f"  - {name:<18}: {f_tr:.4f} ({f_tr * 100:.2f}%)")


def main() -> None:
    print(
        "================================================================================"
    )
    print("TROPICOR: SMOKE TEST PRECIPITATION CALIBRATION (PHASE 2a)")
    print(
        "================================================================================"
    )

    # Ledger check: re-verify no validation openings exist
    ledger_exists = DEFAULT_LEDGER_PATH.exists()
    print(f"Ledger check: '{DEFAULT_LEDGER_PATH}' exists: {ledger_exists}")
    if ledger_exists:
        print("WARNING: Validation ledger exists! Check contents before continuing.")
    else:
        print("CONFIRMED: Zero validation access. Ledger does not exist.")

    stations = [
        ("21205710", "Jardín Botánico"),
        ("23215030", "Aguas Claras"),
    ]

    for st_id, st_name in stations:
        try:
            run_smoke_test_station(st_id, st_name)
        except Exception as exc:
            print(
                f"\nERROR running smoke test for {st_name} [{st_id}]: {exc}",
                file=sys.stderr,
            )
            raise


if __name__ == "__main__":
    main()
