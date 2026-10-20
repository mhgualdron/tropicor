"""Calibration-only smoke test for daily temperature downscaling (Levels 1-3).

Executes daily temperature downscaling models on the calibration partitions of
the two benchmark stations:
- Jardín Botánico (21205710): 1990-01-01 to 2004-12-31 (15 years)
- Aguas Claras (23215030): 1990-01-01 to 2000-12-31 (11 years)

Protocol Safety Constraints:
1. Operates strictly within the pre-specified calibration partitions.
2. Does NOT access, read, or evaluate validation partition records.
3. Does NOT create or write to the validation opening ledger.
4. Uses fixed window_start_utc=5 (05:00-05:00 UTC Colombian civil day, Section 8)
   and lag_days=0. Does NOT compare aggregation windows by calibration loss.
5. Verifies NaN propagation, unit plausibility, and universal crossing rule diagnostics.
"""

from pathlib import Path
from typing import Any, Dict, List, cast

import numpy as np
import pandas as pd

from tropicor.core.protocol import temperature_tuning_loss
from tropicor.downscale.temperature import (
    LEVEL3_TEMPERATURE_CONFIGS,
    Level1MonthlyAdditiveTemperature,
    Level2MeanStateDTRScalingTemperature,
    Level3QuantileMappingTemperature,
    date_to_leap_doy,
    load_calibration_temperature_pair,
    tune_level3_temperature_hyperparameters,
)


def run_station_smoke_test(
    station_id: str,
    obs_tmax_path: Path,
    obs_tmin_path: Path,
    era5_dir: Path,
    window_start_utc: int = 5,
    lag_days: int = 0,
) -> Dict[str, Any]:
    """Execute calibration-only smoke test for a single station across 8 Level 3 configs."""
    print(f"\n{'=' * 70}")
    print(f"Executing Calibration Smoke Test for Station {station_id}")
    print(f"{'=' * 70}")

    obs_tmax, obs_tmin, mod_tmax, mod_tmin = load_calibration_temperature_pair(
        station_id=station_id,
        obs_tmax_path=obs_tmax_path,
        obs_tmin_path=obs_tmin_path,
        era5_target=era5_dir,
        window_start_utc=window_start_utc,
        lag_days=lag_days,
        min_hours=24,
    )

    idx = pd.DatetimeIndex(obs_tmax.index)
    valid_tmax = obs_tmax.notna() & mod_tmax.notna()
    valid_tmin = obs_tmin.notna() & mod_tmin.notna()
    valid_both = valid_tmax & valid_tmin

    # 1. Valid pairs by variable by year and in total
    df_pairs = pd.DataFrame(
        {
            "year": idx.year,
            "tmax_pairs": valid_tmax.astype(int),
            "tmin_pairs": valid_tmin.astype(int),
            "both_pairs": valid_both.astype(int),
        }
    )
    annual_pairs = df_pairs.groupby("year").sum()
    total_pairs = annual_pairs.sum()

    print("\nValid Observation/Model Pairs by Year:")
    print(annual_pairs.to_string())
    print(
        f"TOTAL Valid Pairs: Tmax={total_pairs['tmax_pairs']}, "
        f"Tmin={total_pairs['tmin_pairs']}, Ambos={total_pairs['both_pairs']}"
    )

    # Group counts and minimum N identity
    monthly_ns_tmax: Dict[int, int] = {}
    monthly_ns_tmin: Dict[int, int] = {}
    for m in range(1, 13):
        monthly_ns_tmax[m] = int(((idx.month == m) & valid_tmax).sum())
        monthly_ns_tmin[m] = int(((idx.month == m) & valid_tmin).sum())

    obs_leap_doys = np.array([date_to_leap_doy(ts) for ts in idx])
    mw_ns_tmax: Dict[int, int] = {}
    mw_ns_tmin: Dict[int, int] = {}
    for d in range(1, 367):
        diff = np.abs(obs_leap_doys - d)
        circ_dist = np.minimum(diff, 366 - diff)
        w_mask = circ_dist <= 15
        mw_ns_tmax[d] = int((w_mask & valid_tmax).sum())
        mw_ns_tmin[d] = int((w_mask & valid_tmin).sum())

    min_m_tmax_val = min(monthly_ns_tmax.values())
    min_m_tmax_grp = [k for k, v in monthly_ns_tmax.items() if v == min_m_tmax_val]
    min_m_tmin_val = min(monthly_ns_tmin.values())
    min_m_tmin_grp = [k for k, v in monthly_ns_tmin.items() if v == min_m_tmin_val]

    min_mw_tmax_val = min(mw_ns_tmax.values())
    min_mw_tmax_grp = [k for k, v in mw_ns_tmax.items() if v == min_mw_tmax_val]
    min_mw_tmin_val = min(mw_ns_tmin.values())
    min_mw_tmin_grp = [k for k, v in mw_ns_tmin.items() if v == min_mw_tmin_val]

    print("\nGroup Identity Yielding min_n_per_group:")
    print(
        f"  Monthly: Tmax min N={min_m_tmax_val} in Month(s) {min_m_tmax_grp}; "
        f"Tmin min N={min_m_tmin_val} in Month(s) {min_m_tmin_grp}"
    )
    print(
        f"  Moving Window: Tmax min N={min_mw_tmax_val} in DOY(s) {min_mw_tmax_grp}; "
        f"Tmin min N={min_mw_tmin_val} in DOY(s) {min_mw_tmin_grp}"
    )

    # 2. Level 1
    print("\nFitting Level 1 (Monthly Additive Bias)...")
    l1 = Level1MonthlyAdditiveTemperature()
    l1.fit(obs_tmax, obs_tmin, mod_tmax, mod_tmin)
    l1_tmax, l1_tmin = l1.transform(mod_tmax, mod_tmin)
    l1_loss = temperature_tuning_loss(obs_tmax, l1_tmax, obs_tmin, l1_tmin)
    print(f"  Level 1 Calibration Loss: {l1_loss:.4f} °C")
    print(
        f"  Level 1 Crossings before rule: {l1.n_crossings_before_rule_}, "
        f"after rule: {l1.n_crossings_after_rule_}"
    )

    # 3. Level 2
    print("\nFitting Level 2 (Mean State + DTR Scaling)...")
    l2 = Level2MeanStateDTRScalingTemperature()
    l2.fit(obs_tmax, obs_tmin, mod_tmax, mod_tmin)
    l2_tmax, l2_tmin = l2.transform(mod_tmax, mod_tmin)
    l2_loss = temperature_tuning_loss(obs_tmax, l2_tmax, obs_tmin, l2_tmin)
    print(f"  Level 2 Calibration Loss: {l2_loss:.4f} °C")
    print(
        f"  Level 2 Crossings before rule: {l2.n_crossings_before_rule_}, "
        f"after rule: {l2.n_crossings_after_rule_}"
    )

    # 4. Level 3 CV Tuning (All 8 configurations)
    print(
        "\nTuning Level 3 EQM (Nested Annual Block CV within Calibration, 8 configs)..."
    )
    best_res, l3_model, all_cv_results = tune_level3_temperature_hyperparameters(
        obs_tmax=obs_tmax,
        obs_tmin=obs_tmin,
        mod_tmax=mod_tmax,
        mod_tmin=mod_tmin,
    )
    l3_tmax, l3_tmin = l3_model.transform(mod_tmax, mod_tmin)
    l3_in_sample_loss = temperature_tuning_loss(obs_tmax, l3_tmax, obs_tmin, l3_tmin)

    winner_desc = (
        f"{best_res.grouping}, {best_res.n_quantiles} bins, {best_res.extrapolation}"
    )
    print(f"  Winner Config ID: {best_res.config_id} ({winner_desc})")
    print(f"  Level 3 Out-of-Sample CV Loss: {best_res.overall_tuning_loss:.4f} °C")
    print(f"  Level 3 In-Sample Calibration Loss: {l3_in_sample_loss:.4f} °C")
    print(
        f"  Level 3 In-Sample Crossings before rule: {l3_model.n_crossings_before_rule_}, "
        f"after rule: {l3_model.n_crossings_after_rule_}"
    )
    print(f"  Level 3 Min Samples/Group: {best_res.min_n_per_group}")
    print(f"  Level 3 Extrapolated Days (OOS CV total): {best_res.n_extrapolated_days}")
    print(
        f"  Level 3 Max Abs Extrapolation Correction: "
        f"{best_res.max_abs_extrapolation_correction:.4f} °C"
    )

    # 5. Full-calibration slope diagnostics across all 8 configurations
    slope_diagnostics: List[Dict[str, Any]] = []
    for cfg in LEVEL3_TEMPERATURE_CONFIGS:
        cfg_id = cfg["config_id"]
        grouping = cfg["grouping"]
        n_q = cfg["n_quantiles"]
        extrap = cfg["extrapolation"]

        m_full = Level3QuantileMappingTemperature(
            grouping=grouping,
            n_quantiles=n_q,
            extrapolation=extrap,
        )
        m_full.fit(obs_tmax, obs_tmin, mod_tmax, mod_tmin)
        tmax_c, tmin_c = m_full.transform(mod_tmax, mod_tmin)

        keys = sorted(list(m_full.tmax_slopes_upper_.keys()))

        tmax_s_up = [m_full.tmax_slopes_upper_[k] for k in keys]
        tmax_sep_up = [
            float(m_full.tmax_quantiles_mod_[k][-1] - m_full.tmax_quantiles_mod_[k][-2])
            for k in keys
        ]
        tmax_ns = [
            monthly_ns_tmax[k] if grouping == "monthly" else mw_ns_tmax[k] for k in keys
        ]

        tmin_s_low = [m_full.tmin_slopes_lower_[k] for k in keys]
        tmin_sep_low = [
            float(m_full.tmin_quantiles_mod_[k][1] - m_full.tmin_quantiles_mod_[k][0])
            for k in keys
        ]
        tmin_ns = [
            monthly_ns_tmin[k] if grouping == "monthly" else mw_ns_tmin[k] for k in keys
        ]

        corr_tmax: pd.Series = (tmax_c - mod_tmax).abs()
        corr_tmin: pd.Series = (tmin_c - mod_tmin).abs()

        n_gt5_tmax = int((corr_tmax > 5.0).sum())
        n_gt10_tmax = int((corr_tmax > 10.0).sum())
        max_idx_tmax = cast(pd.Timestamp, corr_tmax.idxmax())
        max_corr_tmax = float(corr_tmax.loc[max_idx_tmax])
        t_mod_max = float(mod_tmax.loc[max_idx_tmax])

        if grouping == "monthly":
            grp_tmax = max_idx_tmax.month
        else:
            d = max_idx_tmax.dayofyear
            if not max_idx_tmax.is_leap_year and max_idx_tmax.month > 2:
                d += 1
            grp_tmax = d
        q_mod_max = m_full.tmax_quantiles_mod_[grp_tmax]
        if t_mod_max > q_mod_max[-1]:
            slope_tmax = m_full.tmax_slopes_upper_[grp_tmax]
        elif t_mod_max < q_mod_max[0]:
            slope_tmax = m_full.tmax_slopes_lower_[grp_tmax]
        else:
            slope_tmax = 1.0

        n_gt5_tmin = int((corr_tmin > 5.0).sum())
        n_gt10_tmin = int((corr_tmin > 10.0).sum())
        max_idx_tmin = cast(pd.Timestamp, corr_tmin.idxmax())
        max_corr_tmin = float(corr_tmin.loc[max_idx_tmin])
        t_mod_min = float(mod_tmin.loc[max_idx_tmin])

        if grouping == "monthly":
            grp_tmin = max_idx_tmin.month
        else:
            d = max_idx_tmin.dayofyear
            if not max_idx_tmin.is_leap_year and max_idx_tmin.month > 2:
                d += 1
            grp_tmin = d
        q_mod_min = m_full.tmin_quantiles_mod_[grp_tmin]
        if t_mod_min > q_mod_min[-1]:
            slope_tmin = m_full.tmin_slopes_upper_[grp_tmin]
        elif t_mod_min < q_mod_min[0]:
            slope_tmin = m_full.tmin_slopes_lower_[grp_tmin]
        else:
            slope_tmin = 1.0

        slope_diagnostics.append(
            {
                "config_id": cfg_id,
                "grouping": grouping,
                "n_quantiles": n_q,
                "extrapolation": extrap,
                "tmax_s_up_min": min(tmax_s_up),
                "tmax_s_up_med": float(np.median(tmax_s_up)),
                "tmax_s_up_max": max(tmax_s_up),
                "tmax_sep_up_min": min(tmax_sep_up),
                "tmax_sep_up_med": float(np.median(tmax_sep_up)),
                "tmax_sep_up_max": max(tmax_sep_up),
                "tmax_n_min": min(tmax_ns),
                "tmax_n_med": float(np.median(tmax_ns)),
                "tmax_n_max": max(tmax_ns),
                "tmax_gt5": n_gt5_tmax,
                "tmax_gt10": n_gt10_tmax,
                "tmax_max_corr": max_corr_tmax,
                "tmax_max_date": str(max_idx_tmax.date()),
                "tmax_max_tmod": t_mod_max,
                "tmax_max_grp": grp_tmax,
                "tmax_max_slope": slope_tmax,
                "tmin_s_low_min": min(tmin_s_low),
                "tmin_s_low_med": float(np.median(tmin_s_low)),
                "tmin_s_low_max": max(tmin_s_low),
                "tmin_sep_low_min": min(tmin_sep_low),
                "tmin_sep_low_med": float(np.median(tmin_sep_low)),
                "tmin_sep_low_max": max(tmin_sep_low),
                "tmin_n_min": min(tmin_ns),
                "tmin_n_med": float(np.median(tmin_ns)),
                "tmin_n_max": max(tmin_ns),
                "tmin_gt5": n_gt5_tmin,
                "tmin_gt10": n_gt10_tmin,
                "tmin_max_corr": max_corr_tmin,
                "tmin_max_date": str(max_idx_tmin.date()),
                "tmin_max_tmod": t_mod_min,
                "tmin_max_grp": grp_tmin,
                "tmin_max_slope": slope_tmin,
            }
        )

    return {
        "station_id": station_id,
        "annual_pairs": annual_pairs,
        "total_pairs": total_pairs,
        "min_m_tmax": (min_m_tmax_val, min_m_tmax_grp),
        "min_m_tmin": (min_m_tmin_val, min_m_tmin_grp),
        "min_mw_tmax": (min_mw_tmax_val, min_mw_tmax_grp),
        "min_mw_tmin": (min_mw_tmin_val, min_mw_tmin_grp),
        "l1_loss": l1_loss,
        "l1_crossings_before": l1.n_crossings_before_rule_,
        "l1_crossings_after": l1.n_crossings_after_rule_,
        "l2_loss": l2_loss,
        "l2_crossings_before": l2.n_crossings_before_rule_,
        "l2_crossings_after": l2.n_crossings_after_rule_,
        "l3_best_config_id": best_res.config_id,
        "l3_grouping": best_res.grouping,
        "l3_n_quantiles": best_res.n_quantiles,
        "l3_extrapolation": best_res.extrapolation,
        "l3_cv_loss": best_res.overall_tuning_loss,
        "l3_cal_loss": l3_in_sample_loss,
        "l3_crossings_before": l3_model.n_crossings_before_rule_,
        "l3_crossings_after": l3_model.n_crossings_after_rule_,
        "l3_min_n_per_group": best_res.min_n_per_group,
        "l3_extrapolated_days": best_res.n_extrapolated_days,
        "l3_max_extrap_corr": best_res.max_abs_extrapolation_correction,
        "cv_grid_results": [
            {
                "config_id": r.config_id,
                "grouping": r.grouping,
                "n_quantiles": r.n_quantiles,
                "extrapolation": r.extrapolation,
                "cv_loss": r.overall_tuning_loss,
                "min_n_per_group": r.min_n_per_group,
                "n_extrapolated_days": r.n_extrapolated_days,
                "max_abs_extrap_corr": r.max_abs_extrapolation_correction,
                "crossings_before": r.crossings_before_rule,
                "crossings_after": r.crossings_after_rule,
            }
            for r in all_cv_results
        ],
        "slope_diagnostics": slope_diagnostics,
    }


def main() -> None:
    """Run calibration smoke test for both benchmark stations."""
    stations = [
        {
            "id": "21205710",
            "name": "Jardín Botánico",
            "tmax": Path("local/newFiles/TempMaxima_JardinBotanico.csv"),
            "tmin": Path("local/newFiles/TempMinima_JardinBotanico.csv"),
            "era5": Path("local/era5_hourly/21205710"),
        },
        {
            "id": "23215030",
            "name": "Aguas Claras",
            "tmax": Path("local/newFiles/TempMaxima_AguasClaras.csv"),
            "tmin": Path("local/newFiles/TempMinima_AguasClaras.csv"),
            "era5": Path("local/era5_hourly/23215030"),
        },
    ]

    results: List[Dict[str, Any]] = []
    for st in stations:
        tmax_p = cast(Path, st["tmax"])
        tmin_p = cast(Path, st["tmin"])
        era5_p = cast(Path, st["era5"])
        st_id = str(st["id"])
        if not tmax_p.exists() or not tmin_p.exists():
            print(f"Skipping {st['name']}: observation files not found.")
            continue
        res = run_station_smoke_test(
            station_id=st_id,
            obs_tmax_path=tmax_p,
            obs_tmin_path=tmin_p,
            era5_dir=era5_p,
            window_start_utc=5,
            lag_days=0,
        )
        results.append(res)

    # Format Markdown Report
    lines: List[str] = [
        "# Smoke Test Report: Daily Temperature Calibration (Levels 1-3)",
        "",
        "- **Protocol Scope**: Calibration partition strictly isolated (no validation access, no ledger created).",
        "- **Aggregation Window**: Fixed at `05:00-05:00 UTC` (Colombian local civil day, Section 8), lag 0.",
        "- **Tuning Loss**: Mean absolute error of percentiles 5 to 95 for Tmax, Tmin, and DTR (`temperature_tuning_loss`, Section 4.1).",
        "- **Grid Size**: 8 configurations (grouping x bins x extrapolation scheme).",
        "",
        "## 1. Summary of Calibration Results",
        "",
        "| Station ID | Station Name | Level 1 Cal Loss (°C) | Level 2 Cal Loss (°C) | Level 3 Best Config | Level 3 CV Loss (°C) | Level 3 Cal Loss (°C) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in results:
        cfg_str = f"ID {r['l3_best_config_id']} ({r['l3_grouping']}, {r['l3_n_quantiles']}, {r['l3_extrapolation']})"
        st_name = "Jardín Botánico" if r["station_id"] == "21205710" else "Aguas Claras"
        lines.append(
            f"| `{r['station_id']}` | {st_name} | {r['l1_loss']:.4f} | "
            f"{r['l2_loss']:.4f} | {cfg_str} | {r['l3_cv_loss']:.4f} | "
            f"{r['l3_cal_loss']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 2. Valid Observation/Model Pairs in Calibration",
            "",
        ]
    )

    for r in results:
        st_name = "Jardín Botánico" if r["station_id"] == "21205710" else "Aguas Claras"
        lines.append(f"### {st_name} (`{r['station_id']}`)")
        lines.append("")
        lines.append(
            "| Year | Tmax Valid Pairs | Tmin Valid Pairs | Both Valid Pairs |"
        )
        lines.append("| :--- | :---: | :---: | :---: |")
        ap = r["annual_pairs"]
        for yr, row in ap.iterrows():
            lines.append(
                f"| {yr} | {row['tmax_pairs']} | {row['tmin_pairs']} | {row['both_pairs']} |"
            )
        tp = r["total_pairs"]
        lines.append(
            f"| **TOTAL** | **{tp['tmax_pairs']}** | **{tp['tmin_pairs']}** | **{tp['both_pairs']}** |"
        )
        lines.append("")
        lines.append(
            f"- **Monthly min N identity**: Tmax min N = {r['min_m_tmax'][0]} (Month {r['min_m_tmax'][1]}); "
            f"Tmin min N = {r['min_m_tmin'][0]} (Month {r['min_m_tmin'][1]})."
        )
        lines.append(
            f"- **Moving Window min N identity**: Tmax min N = {r['min_mw_tmax'][0]} (DOY {r['min_mw_tmax'][1]}); "
            f"Tmin min N = {r['min_mw_tmin'][0]} (DOY {r['min_mw_tmin'][1]})."
        )
        lines.append("")

    lines.extend(
        [
            "## 3. Level 3 Hyperparameter Grid (8 Configs) CV Loss & Diagnostics",
            "",
            "| Station | Config ID | Grouping | Quantiles | Extrapolation | CV Loss (°C) | Min N/Group | Extrap Days | Max Extrap Corr (°C) | Crossings (Before / After) |",
            "| :--- | :---: | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    for r in results:
        st_name = "Jardín Botánico" if r["station_id"] == "21205710" else "Aguas Claras"
        for g in r["cv_grid_results"]:
            cr_str = f"{g['crossings_before']} / {g['crossings_after']}"
            lines.append(
                f"| {st_name} | {g['config_id']} | {g['grouping']} | {g['n_quantiles']} | "
                f"{g['extrapolation']} | {g['cv_loss']:.4f} | {g['min_n_per_group']} | "
                f"{g['n_extrapolated_days']} | {g['max_abs_extrap_corr']:.4f} | {cr_str} |"
            )

    lines.extend(
        [
            "",
            "## 4. Full Calibration Slope Diagnostics by Configuration",
            "",
            "| Station | Cfg | Variable | s (Min, Med, Max) | Mod Extreme Sep (Min, Med, Max) | Group N (Min, Med, Max) | Abs Corr > 5°C | Abs Corr > 10°C | Max Corr (°C) | Date (Mod T, Group, Slope) |",
            "| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
        ]
    )

    for r in results:
        st_name = "Jardín Botánico" if r["station_id"] == "21205710" else "Aguas Claras"
        for d in r["slope_diagnostics"]:
            cfg_id = d["config_id"]
            # Tmax row
            s_up_str = f"[{d['tmax_s_up_min']:.2f}, {d['tmax_s_up_med']:.2f}, {d['tmax_s_up_max']:.2f}]"
            sep_up_str = f"[{d['tmax_sep_up_min']:.2f}, {d['tmax_sep_up_med']:.2f}, {d['tmax_sep_up_max']:.2f}]"
            n_up_str = f"[{d['tmax_n_min']}, {d['tmax_n_med']:.0f}, {d['tmax_n_max']}]"
            max_tmax_info = f"{d['tmax_max_date']} ({d['tmax_max_tmod']:.1f}°C, g={d['tmax_max_grp']}, s={d['tmax_max_slope']:.2f})"
            lines.append(
                f"| {st_name} | {cfg_id} | Tmax (Upper) | {s_up_str} | {sep_up_str} | {n_up_str} | "
                f"{d['tmax_gt5']} | {d['tmax_gt10']} | {d['tmax_max_corr']:.2f} | {max_tmax_info} |"
            )
            # Tmin row
            s_low_str = f"[{d['tmin_s_low_min']:.2f}, {d['tmin_s_low_med']:.2f}, {d['tmin_s_low_max']:.2f}]"
            sep_low_str = f"[{d['tmin_sep_low_min']:.2f}, {d['tmin_sep_low_med']:.2f}, {d['tmin_sep_low_max']:.2f}]"
            n_low_str = f"[{d['tmin_n_min']}, {d['tmin_n_med']:.0f}, {d['tmin_n_max']}]"
            max_tmin_info = f"{d['tmin_max_date']} ({d['tmin_max_tmod']:.1f}°C, g={d['tmin_max_grp']}, s={d['tmin_max_slope']:.2f})"
            lines.append(
                f"| {st_name} | {cfg_id} | Tmin (Lower) | {s_low_str} | {sep_low_str} | {n_low_str} | "
                f"{d['tmin_gt5']} | {d['tmin_gt10']} | {d['tmin_max_corr']:.2f} | {max_tmin_info} |"
            )

    report_content = "\n".join(lines) + "\n"
    out_path = Path("local/reports/smoke_test_temperature_calibration.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_content, encoding="utf-8")
    print(f"\nSmoke test completed successfully. Report written to: {out_path}")


if __name__ == "__main__":
    main()
