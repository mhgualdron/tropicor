"""Audit script to evaluate feasibility of CDD_max completeness rules.

Evaluates ground observation records ONLY on calibration partitions:
  - Jardín Botánico (21205710): 1975-1994 (20 years)
  - Aguas Claras (23215030): 1975-1987 (13 years)

No model evaluation, no out-of-sample validation data, and no ledger logging.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from tropicor.core.protocol import (
    CDD_MAX_MONTHLY_MISSING_FRAC,
    CDD_MIN_ANNUAL_VALID_DAYS,
    PRECIP_DRY_THRESHOLD,
    compute_annual_cdd_max,
)
from tropicor.io.ideam import read_dhime_daily_precipitation


def analyze_station_cdd(
    station_name: str,
    station_code: str,
    filepath: Path,
    cal_start_year: int,
    cal_end_year: int,
) -> None:
    print(f"\n{'=' * 78}")
    print(f"ESTACIÓN: {station_name} ({station_code}) - CALIBRACIÓN {cal_start_year}–{cal_end_year}")
    print(f"{'=' * 78}")

    ds = read_dhime_daily_precipitation(filepath)
    series_full = ds["pr"].squeeze().to_pandas().sort_index()
    if "quality_flag" in ds:
        qf = ds["quality_flag"].squeeze().to_pandas().sort_index()
        is_faltante = qf.astype(str).str.strip().str.lower() == "faltante"
        series_full.loc[is_faltante] = np.nan

    # Filter strictly within calibration partition
    start_date = f"{cal_start_year}-01-01"
    end_date = f"{cal_end_year}-12-31"
    s_cal = series_full.loc[start_date:end_date]
    assert isinstance(s_cal, pd.Series)

    # Run protocol CDD engine passing observation as obs and mod
    cdd_res = compute_annual_cdd_max(s_cal, s_cal)
    diag = cdd_res.diagnostic

    total_years = cal_end_year - cal_start_year + 1
    discarded_years = diag.n_discarded_years_obs
    evaluated_years = total_years - discarded_years

    # Yearly breakdown
    years = list(range(cal_start_year, cal_end_year + 1))
    df = pd.DataFrame({"obs": s_cal}).sort_index()
    assert isinstance(df.index, pd.DatetimeIndex)

    excluded_months_list: List[str] = []
    years_with_valid_streaks = 0
    total_candidate_streaks = 0
    total_valid_streaks = 0

    print("\n--- DESGLOSE ANUAL ---")
    print(f"{'Año':<6} | {'Días Válidos':<13} | {'Estado Año':<12} | {'Meses Excluidos (>20% faltante)':<32} | {'CDD_max':<8}")
    print(f"{'-' * 88}")

    for year in years:
        start_dt = pd.Timestamp(year, 1, 1)
        end_dt = pd.Timestamp(year, 12, 31)
        full_idx = pd.date_range(start_dt, end_dt, freq="D")
        sub = df.loc[df.index.year == year]
        if sub.index.has_duplicates:
            sub = sub[~sub.index.duplicated(keep="first")]
        df_year = sub.reindex(full_idx)
        assert isinstance(df_year.index, pd.DatetimeIndex)

        valid_days = int(df_year["obs"].notna().sum())
        if valid_days < CDD_MIN_ANNUAL_VALID_DAYS:
            print(f"{year:<6} | {valid_days:<13} | {'DESCARTADO':<12} | {'N/A (<330 días)':<32} | {'N/A':<8}")
            continue

        invalid_m: List[int] = []
        for m in range(1, 13):
            df_m = df_year[df_year.index.month == m]
            if len(df_m) == 0:
                invalid_m.append(m)
            else:
                missing_frac = df_m["obs"].isna().sum() / len(df_m)
                if missing_frac > CDD_MAX_MONTHLY_MISSING_FRAC:
                    invalid_m.append(m)
                    excluded_months_list.append(f"{year}-{m:02d} ({missing_frac*100:.1f}%)")

        # Count streaks in year
        curr_len = 0
        touches_inv = False
        has_dry = False
        valid_in_year = 0
        cand_in_year = 0

        for idx, val in df_year["obs"].items():
            is_inv_m = idx.month in invalid_m  # type: ignore
            is_nan = pd.isna(val)
            is_inv_day = is_inv_m or is_nan
            is_wet = (not is_inv_day) and (val >= PRECIP_DRY_THRESHOLD)

            if is_wet:
                if has_dry:
                    cand_in_year += 1
                    if not touches_inv:
                        valid_in_year += 1
                curr_len = 0
                touches_inv = False
                has_dry = False
            else:
                if is_inv_day:
                    touches_inv = True
                else:
                    has_dry = True
                    curr_len += 1

        if has_dry:
            cand_in_year += 1
            if not touches_inv:
                valid_in_year += 1

        total_candidate_streaks += cand_in_year
        total_valid_streaks += valid_in_year
        if valid_in_year > 0:
            years_with_valid_streaks += 1

        cdd_val = cdd_res.annual_obs.get(year, float("nan"))
        cdd_str = f"{cdd_val:.1f} d" if np.isfinite(cdd_val) else "0.0 d"
        inv_str = ", ".join(str(m) for m in invalid_m) if invalid_m else "Ninguno"
        print(f"{year:<6} | {valid_days:<13} | {'VÁLIDO':<12} | {inv_str:<32} | {cdd_str:<8}")

    print("\n--- DIAGNÓSTICO DE FACTIBILIDAD CDD ---")
    print(f"1. Años totales en calibración:           {total_years}")
    print(f"2. Años descartados (<330 días válidos):  {discarded_years}")
    print(f"3. Años evaluados (válidos):              {evaluated_years}")
    print(f"4. Meses excluidos (>20% faltante):       {len(excluded_months_list)} meses en total")
    if excluded_months_list:
        print(f"   Detalle: {', '.join(excluded_months_list)}")
    print(f"5. Rachas candidatas totales:             {total_candidate_streaks}")
    print(f"6. Rachas descartadas (NaN / mes excl.):  {diag.n_discarded_streaks_obs}")
    print(f"7. Rachas válidas restantes:              {total_valid_streaks}")
    print(f"8. Años con >= 1 racha válida:            {years_with_valid_streaks} de {evaluated_years}")
    print(f"9. Media CDD_max observación:             {cdd_res.mean_obs:.2f} días")


def main() -> None:
    jb_path = Path("local/newFiles/DiaPluviometrico_JardinBotanico_21205710.csv")
    ac_path = Path("local/newFiles/DiaPluviometrico_AguasClaras.csv")

    if not jb_path.exists():
        print(f"Error: no existe {jb_path}", file=sys.stderr)
        sys.exit(1)
    if not ac_path.exists():
        print(f"Error: no existe {ac_path}", file=sys.stderr)
        sys.exit(1)

    analyze_station_cdd("Jardín Botánico", "21205710", jb_path, 1975, 1994)
    analyze_station_cdd("Aguas Claras (Aguachica)", "23215030", ac_path, 1975, 1987)


if __name__ == "__main__":
    main()
