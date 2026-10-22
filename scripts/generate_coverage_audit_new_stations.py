"""Generate coverage audit report for candidate stations in local/reports/coverage_audit_new_stations.json.

Reads candidate slots from local/reports/slot_plan.csv, evaluates coverage under
pre-specified protocol windows, computes optimal partitions, and resolves slot status.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_station_data import (  # noqa: E402
    audit_completeness,
    audit_file_structure,
    audit_simultaneous_temperature,
    extract_clean_numeric_series,
)
from tropicor.core.protocol import get_current_git_commit  # noqa: E402
from tropicor.core.station_selection import (  # noqa: E402
    compute_optimal_partition,
    evaluate_partition_eligibility,
    evaluate_slot_status,
)

KNOWN_SENTINELS = {
    "-999",
    "-999.0",
    "9999",
    "9999.0",
    "-9999",
    "-9999.0",
    "S/D",
    "s/d",
    "Nulo",
    "nulo",
    "nan",
    "NaN",
    "NAN",
}


def load_slot_plan(csv_path: Path) -> List[Dict[str, str]]:
    """Load slot plan mapping from CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Slot plan file not found: {csv_path}")

    rows: List[Dict[str, str]] = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "ranura": row["ranura"].strip(),
                    "plan": row["plan"].strip(),
                    "codigo": row["codigo"].strip(),
                }
            )
    return rows


def audit_variable_coverage(
    csv_path: Path,
    station_code: str,
    var_label: str,
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    """Audit coverage and diagnostic counts for a single variable file."""
    # Run structure audit with restrict_range
    struct_res, df_raw = audit_file_structure(
        filepath=csv_path,
        expected_station_id=station_code,
        expected_start=start_date,
        expected_end=end_date,
        restrict_to_expected_range=True,
    )

    if not struct_res.station_code_matches_expected:
        raise ValueError(
            f"Station code mismatch in '{csv_path}'. Expected '{station_code}', "
            f"found {struct_res.station_codes}."
        )

    if struct_res.unparseable_dates_count > 0:
        raise ValueError(
            f"Found {struct_res.unparseable_dates_count} unparseable date(s) in '{csv_path}': "
            f"{struct_res.unparseable_samples}."
        )

    clean_series, _ = extract_clean_numeric_series(df_raw)
    clean_series_eval = clean_series.loc[start_date:end_date]

    comp_res = audit_completeness(
        series=clean_series_eval,
        variable_label=var_label,
        expected_start=start_date,
        expected_end=end_date,
        evaluation_scope=f"Rango Protocolo ({start_date[:4]}-{end_date[:4]})",
    )

    # Diagnostics: (i) duplicate dates, (iii) out of extreme physical bounds
    dup_dates_count = int(df_raw["Fecha"].duplicated().sum())

    val_str = df_raw["Valor"].fillna("").astype(str).str.strip().str.replace(",", ".")
    is_sentinel = val_str.isin(KNOWN_SENTINELS)
    num_vals = pd.to_numeric(val_str, errors="coerce")
    valid_nums = num_vals[(~is_sentinel) & (~num_vals.isna())]

    if "Precipitaci" in var_label:
        out_bounds_count = int(((valid_nums < 0.0) | (valid_nums > 1000.0)).sum())
    else:
        out_bounds_count = int(((valid_nums < -30.0) | (valid_nums > 50.0)).sum())

    # Quality flag proportions by decade
    flags_by_decade: Dict[str, Dict[str, Any]] = {}
    if "NivelAprobacion" in df_raw.columns and "Fecha" in df_raw.columns:
        df_eval = df_raw.copy()
        df_eval["dt"] = pd.to_datetime(df_eval["Fecha"], errors="coerce")
        df_eval = df_eval[
            (df_eval["dt"] >= pd.Timestamp(start_date))
            & (df_eval["dt"] <= pd.Timestamp(end_date))
        ]
        df_eval["decade"] = (df_eval["dt"].dt.year // 10) * 10

        for decade, group in df_eval.groupby("decade"):
            counts = (
                group["NivelAprobacion"]
                .fillna("MISSING")
                .astype(str)
                .str.strip()
                .value_counts()
            )
            total_d = len(group)
            flags_by_decade[f"{decade}s"] = {
                "total_records": int(total_d),
                "counts": {str(k): int(v) for k, v in counts.items()},
                "proportions": {
                    str(k): round(float(v) / total_d, 4) for k, v in counts.items()
                },
            }

    return {
        "file_relative_path": Path(csv_path).as_posix(),
        "file_sha256": struct_res.file_sha256,
        "window_evaluated": f"{start_date} to {end_date}",
        "total_records_in_window": int(struct_res.total_raw_rows),
        "valid_days_count": int(comp_res.total_valid_days),
        "total_calendar_days": int(comp_res.total_expected_days),
        "valid_days_pct": round(
            float(comp_res.total_valid_days) / comp_res.total_expected_days * 100.0,
            2,
        ),
        "valid_years_count": int(comp_res.years_ge_330_count),
        "valid_years_list": [int(y) for y in comp_res.years_ge_330_list],
        "quality_flags_by_decade": flags_by_decade,
        "diagnostics": {
            "duplicate_dates_count": dup_dates_count,
            "out_of_bounds_count": out_bounds_count,
        },
        "clean_series": clean_series_eval,
    }


def compute_thermal_inversions_count(s_tmax: pd.Series, s_tmin: pd.Series) -> int:
    """Count occurrences where Tmax < Tmin on days where both are valid."""
    common_idx = s_tmax.dropna().index.intersection(s_tmin.dropna().index)
    if len(common_idx) == 0:
        return 0
    inversions = (s_tmax.loc[common_idx] < s_tmin.loc[common_idx]).sum()
    return int(inversions)


def generate_coverage_audit(
    base_dir: Path = Path("local/era5_ts"),
    slot_plan_path: Path = Path("local/reports/slot_plan.csv"),
    output_path: Path = Path("local/reports/coverage_audit_new_stations.json"),
) -> Dict[str, Any]:
    """Run coverage audit, resolve slots, and generate unified JSON report."""
    slot_rows = load_slot_plan(slot_plan_path)

    # Discover candidate stations that have local directories
    candidate_stations = sorted(
        list(set(r["codigo"] for r in slot_rows if (base_dir / r["codigo"]).exists()))
    )

    results: Dict[str, Any] = {
        "report_type": "coverage_audit_new_stations",
        "description": "Coverage and completeness audit under restricted protocol ranges for new candidate stations",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_current_git_commit(),
        "windows": {
            "temperature": "1990-01-01 to 2020-12-31 (31 calendar years)",
            "precipitation": "1975-01-01 to 2020-12-31 (46 calendar years)",
        },
        "stations": {},
        "slot_selection": {},
    }

    station_eval_map: Dict[str, Dict[str, Any]] = {}

    for st in candidate_stations:
        st_dir = base_dir / st
        pr_files = list(st_dir.glob("DiaPluviometrico_*.csv"))
        tmax_files = list(st_dir.glob("TempMaxima_*.csv"))
        tmin_files = list(st_dir.glob("TempMinima_*.csv"))

        if len(pr_files) > 1:
            raise ValueError(
                f"Station {st}: multiple files found for P: {[f.name for f in pr_files]}"
            )
        if len(tmax_files) > 1:
            raise ValueError(
                f"Station {st}: multiple files found for Tmax: {[f.name for f in tmax_files]}"
            )
        if len(tmin_files) > 1:
            raise ValueError(
                f"Station {st}: multiple files found for Tmin: {[f.name for f in tmin_files]}"
            )

        missing_files: List[str] = []
        if len(pr_files) == 0:
            missing_files.append("DiaPluviometrico_*.csv")
        if len(tmax_files) == 0:
            missing_files.append("TempMaxima_*.csv")
        if len(tmin_files) == 0:
            missing_files.append("TempMinima_*.csv")

        if missing_files:
            st_data = {
                "station_code": st,
                "station_directory": Path(st_dir).as_posix(),
                "status": "INCOMPLETO",
                "missing_files": missing_files,
                "temperature_evaluation": {
                    "eligible": False,
                    "ineligibility_reasons": [
                        f"Missing required CSV files: {missing_files}"
                    ],
                    "partition": None,
                },
                "precipitation_evaluation": {
                    "eligible": False,
                    "ineligibility_reasons": [
                        f"Missing required CSV files: {missing_files}"
                    ],
                    "partition": None,
                },
            }
            results["stations"][st] = st_data
            station_eval_map[st] = st_data
            continue

        pr_f = pr_files[0]
        tmax_f = tmax_files[0]
        tmin_f = tmin_files[0]

        # Audit variables
        pr_audit = audit_variable_coverage(
            pr_f, st, "Precipitación", "1975-01-01", "2020-12-31"
        )
        tmax_audit = audit_variable_coverage(
            tmax_f, st, "Temperatura Máxima", "1990-01-01", "2020-12-31"
        )
        tmin_audit = audit_variable_coverage(
            tmin_f, st, "Temperatura Mínima", "1990-01-01", "2020-12-31"
        )

        s_tmax = tmax_audit.pop("clean_series")
        s_tmin = tmin_audit.pop("clean_series")
        _ = pr_audit.pop("clean_series")

        # Simultaneous temperature audit (1990-2020)
        sim_temp = audit_simultaneous_temperature(
            tmax_series=s_tmax,
            tmin_series=s_tmin,
            expected_start="1990-01-01",
            expected_end="2020-12-31",
        )

        # Thermal inversions count (Tmax < Tmin)
        inversions_count = compute_thermal_inversions_count(s_tmax, s_tmin)

        joint_valid_years = [int(y) for y in sim_temp.years_simultaneous_ge_330_list]
        temp_part = compute_optimal_partition(joint_valid_years)
        temp_eligible, reasons_temp = evaluate_partition_eligibility(
            temp_part, min_joint_years=20, min_block_years=8
        )

        # Precipitation partition (1975-2020)
        pr_valid_years = [int(y) for y in pr_audit["valid_years_list"]]
        pr_part = compute_optimal_partition(pr_valid_years)
        pr_eligible, reasons_pr = evaluate_partition_eligibility(
            pr_part, min_joint_years=20, min_block_years=8
        )

        st_data = {
            "station_code": st,
            "station_directory": Path(st_dir).as_posix(),
            "variables": {
                "P": pr_audit,
                "Tmax": tmax_audit,
                "Tmin": tmin_audit,
            },
            "diagnostics": {
                "duplicate_dates": {
                    "P": pr_audit["diagnostics"]["duplicate_dates_count"],
                    "Tmax": tmax_audit["diagnostics"]["duplicate_dates_count"],
                    "Tmin": tmin_audit["diagnostics"]["duplicate_dates_count"],
                },
                "thermal_inversions_tmax_lt_tmin": inversions_count,
                "physical_outliers": {
                    "P_negative_or_gt_1000mm": pr_audit["diagnostics"][
                        "out_of_bounds_count"
                    ],
                    "Tmax_outside_neg30_pos50": tmax_audit["diagnostics"][
                        "out_of_bounds_count"
                    ],
                    "Tmin_outside_neg30_pos50": tmin_audit["diagnostics"][
                        "out_of_bounds_count"
                    ],
                },
            },
            "joint_temperature": {
                "total_simultaneous_valid_days": int(
                    sim_temp.total_simultaneous_valid_days
                ),
                "joint_valid_years_count": len(joint_valid_years),
                "joint_valid_years_list": joint_valid_years,
            },
            "temperature_evaluation": {
                "eligible": temp_eligible,
                "ineligibility_reasons": reasons_temp,
                "partition": temp_part,
            },
            "precipitation_evaluation": {
                "eligible": pr_eligible,
                "ineligibility_reasons": reasons_pr,
                "partition": pr_part,
            },
        }

        results["stations"][st] = st_data
        station_eval_map[st] = st_data

    # Evaluate Slot Selection across S1 to S10
    slots_grouped: Dict[str, List[Tuple[str, str, str, List[str]]]] = {}
    for r in slot_rows:
        slot = r["ranura"]
        plan = r["plan"]
        code = r["codigo"]

        if code in station_eval_map:
            st_info = station_eval_map[code]
            if st_info.get("status") == "INCOMPLETO":
                status = "INCOMPLETO"
                missing = st_info.get("missing_files", [])
            else:
                is_elig = st_info["temperature_evaluation"]["eligible"]
                status = "ELEGIBLE" if is_elig else "NO_ELEGIBLE"
                missing = []
        else:
            status = "SIN_DATOS"
            missing = []

        if slot not in slots_grouped:
            slots_grouped[slot] = []
        slots_grouped[slot].append((plan, code, status, missing))

    for slot, plan_items in slots_grouped.items():
        plan_tuples = [(p, c, s) for p, c, s, _ in plan_items]
        slot_status, selected_info = evaluate_slot_status(plan_tuples)
        results["slot_selection"][slot] = {
            "slot": slot,
            "status": slot_status,
            "selected_plan": selected_info[0] if selected_info else None,
            "selected_station": selected_info[1] if selected_info else None,
            "plans": [
                {
                    "plan": p,
                    "station_code": c,
                    "status": s,
                    **({"missing_files": m} if s == "INCOMPLETO" else {}),
                }
                for p, c, s, m in plan_items
            ],
        }

    # Write output JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return results


def print_unified_report(report_data: Dict[str, Any]) -> None:
    """Print console summary and markdown tables from the in-memory report object."""
    stations = report_data["stations"]

    print("=" * 110)
    print(
        "TABLA 1: AUDITORÍA DE COBERTURA, PARTICIÓN Y ELEGIBILIDAD (7 ESTACIONES CANDIDATAS)"
    )
    print("=" * 110)
    header = (
        f"{'Estación':<10} | {'Tmax (Y)':<8} | {'Tmin (Y)':<8} | {'Joint Y':<8} | "
        f"{'Cal Block':<12} | {'Val Block':<12} | {'Cal Y':<6} | {'Val Y':<6} | "
        f"{'P (Y)':<6} | {'Elegible Temp':<13} | {'Diagnósticos (dup/inv/out)':<26}"
    )
    print(header)
    print("-" * 110)

    for st, data in sorted(stations.items()):
        if data.get("status") == "INCOMPLETO":
            print(
                f"{st:<10} | INCOMPLETO - Archivos faltantes: {data.get('missing_files')}"
            )
            continue

        tx_y = data["variables"]["Tmax"]["valid_years_count"]
        tn_y = data["variables"]["Tmin"]["valid_years_count"]
        p_y = data["variables"]["P"]["valid_years_count"]
        j_y = data["joint_temperature"]["joint_valid_years_count"]

        t_part = data["temperature_evaluation"]["partition"]
        cal_str = (
            f"{t_part['cal_block'][0]}-{t_part['cal_block'][1]}"
            if t_part["cal_block"]
            else "N/A"
        )
        val_str = (
            f"{t_part['val_block'][0]}-{t_part['val_block'][1]}"
            if t_part["val_block"]
            else "N/A"
        )
        cal_n = t_part["n_cal_valid"]
        val_n = t_part["n_val_valid"]

        temp_elig = "SI" if data["temperature_evaluation"]["eligible"] else "NO"

        diag = data["diagnostics"]
        dup_sum = sum(diag["duplicate_dates"].values())
        inv_sum = diag["thermal_inversions_tmax_lt_tmin"]
        out_sum = sum(diag["physical_outliers"].values())
        diag_str = f"dup={dup_sum}, inv={inv_sum}, out={out_sum}"

        row_str = (
            f"{st:<10} | {tx_y:<8} | {tn_y:<8} | {j_y:<8} | "
            f"{cal_str:<12} | {val_str:<12} | {cal_n:<6} | {val_n:<6} | "
            f"{p_y:<6} | {temp_elig:<13} | {diag_str:<26}"
        )
        print(row_str)

    print("=" * 110)
    print("\n" + "=" * 80)
    print("TABLA 2: SELECCIÓN Y RESOLUCIÓN DE RANURAS (S1 - S10)")
    print("=" * 80)
    print(
        f"{'Ranura':<8} | {'Estado Ranura':<14} | {'Seleccionada':<16} | {'Planes y Estado':<38}"
    )
    print("-" * 80)

    for slot, sdata in report_data["slot_selection"].items():
        s_status = sdata["status"]
        sel = (
            f"Plan {sdata['selected_plan']} ({sdata['selected_station']})"
            if sdata["selected_plan"]
            else "PENDIENTE"
        )
        plan_details = ", ".join(
            f"{p['plan']}:{p['station_code']} ({p['status']})" for p in sdata["plans"]
        )
        print(f"{slot:<8} | {s_status:<14} | {sel:<16} | {plan_details}")

    print("=" * 80)


def verify_console_matches_json(report_data: Dict[str, Any], json_path: Path) -> None:
    """Verify strictly that in-memory counts match JSON counts for all candidate stations."""
    with open(json_path, mode="r", encoding="utf-8") as f:
        disk_data = json.load(f)

    for st, mem_st in report_data["stations"].items():
        assert st in disk_data["stations"], f"Station {st} missing from JSON file!"
        disk_st = disk_data["stations"][st]

        if mem_st.get("status") == "INCOMPLETO":
            assert disk_st.get("status") == "INCOMPLETO"
            assert mem_st.get("missing_files") == disk_st.get("missing_files")
            continue

        # Check Tmax valid years count
        assert (
            mem_st["variables"]["Tmax"]["valid_years_count"]
            == disk_st["variables"]["Tmax"]["valid_years_count"]
        ), f"Mismatch in Tmax valid years count for station {st}"

        # Check Tmin valid years count
        assert (
            mem_st["variables"]["Tmin"]["valid_years_count"]
            == disk_st["variables"]["Tmin"]["valid_years_count"]
        ), f"Mismatch in Tmin valid years count for station {st}"

        # Check Joint valid years count
        assert (
            mem_st["joint_temperature"]["joint_valid_years_count"]
            == disk_st["joint_temperature"]["joint_valid_years_count"]
        ), f"Mismatch in Joint valid years count for station {st}"

        # Check Calibration valid years count
        assert (
            mem_st["temperature_evaluation"]["partition"]["n_cal_valid"]
            == disk_st["temperature_evaluation"]["partition"]["n_cal_valid"]
        ), f"Mismatch in Calibration valid years count for station {st}"

        # Check Validation valid years count
        assert (
            mem_st["temperature_evaluation"]["partition"]["n_val_valid"]
            == disk_st["temperature_evaluation"]["partition"]["n_val_valid"]
        ), f"Mismatch in Validation valid years count for station {st}"

        # Check Eligibility boolean
        assert (
            mem_st["temperature_evaluation"]["eligible"]
            == disk_st["temperature_evaluation"]["eligible"]
        ), f"Mismatch in Temperature eligibility for station {st}"

        # Check Diagnostic counts
        assert (
            mem_st["diagnostics"]["thermal_inversions_tmax_lt_tmin"]
            == disk_st["diagnostics"]["thermal_inversions_tmax_lt_tmin"]
        ), f"Mismatch in thermal inversions count for station {st}"

        assert (
            mem_st["diagnostics"]["duplicate_dates"]
            == disk_st["diagnostics"]["duplicate_dates"]
        ), f"Mismatch in duplicate dates counts for station {st}"

        assert (
            mem_st["diagnostics"]["physical_outliers"]
            == disk_st["diagnostics"]["physical_outliers"]
        ), f"Mismatch in physical outliers counts for station {st}"

    print(
        f"\n[VERIFICATION OK] Verified {len(report_data['stations'])} stations: "
        f"In-memory console counts are 100% IDENTICAL to JSON counts."
    )


if __name__ == "__main__":
    report_out = generate_coverage_audit()
    print_unified_report(report_out)
    verify_console_matches_json(
        report_out, Path("local/reports/coverage_audit_new_stations.json")
    )
