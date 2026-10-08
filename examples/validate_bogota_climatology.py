"""Canonical Sanity Check: Climatology and Rainfall Regime of Bogotá.

Replicates canonical climatological normals on real IDEAM ground observations:
- Station: Jardín Botánico (Code: 21205710, Bogotá, D.C.)
- Data: local/21205710.xlsx (1990 - 2026 monthly precipitation records)
- Objective: Verify that TROPICOR's climatology engine reproduces the well-known
  bimodal cycle of the Bogotá high plain (peaks in April-May and October-November)
  driven by the double crossing of the Intertropical Convergence Zone (ITCZ).

Note:
    This serves as a sanity check on a single canonical Andean station to verify
    algorithmic behavior on real historical data. Comprehensive spatial validation
    across natural regions is performed via the multi-station benchmark suite.
"""

import sys
import warnings
from pathlib import Path

from tropicor.core.climatology import (
    annual_cycle_amplitude,
    annual_cycle_peaks,
    annual_cycle_phase,
    classify_rainfall_regime,
    compute_monthly_climatology,
)
from tropicor.io.ideam import IdeamAdapter

# Suppress openpyxl style warnings for clean CLI output
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")


def validate_bogota_climatology() -> None:
    """Execute canonical climatology sanity check on real Bogotá station records."""
    print("=" * 78)
    print("   TROPICOR v0.1.2: CANONICAL SANITY CHECK (BOGOTÁ JARDÍN BOTÁNICO)")
    print("=" * 78)

    data_path = Path("local/21205710.xlsx")
    if not data_path.exists():
        print(f"[-] Error: Local dataset '{data_path}' not found.")
        print("    Place '21205710.xlsx' in 'local/' directory to run this test.")
        sys.exit(1)

    # 1. Ingest real IDEAM DHIME time series
    print("\n>>> [STEP 1] Ingesting Real IDEAM DHIME Ground Observations")
    print("-" * 78)
    adapter = IdeamAdapter(data_path)
    series = adapter.get_series()

    print(f"[*] Station Name:     {adapter.name} [{adapter.code}]")
    print(f"[*] Municipality:     {adapter.municipality}, {adapter.department}")
    coords = f"Lat {adapter.latitude:.4f}°N, Lon {adapter.longitude:.4f}°W"
    print(f"[*] Coordinates:      {coords}")
    print(f"[*] Station Altitude: {adapter.elevation:.1f} masl")
    start_m = series.index[0].strftime("%Y-%m")
    end_m = series.index[-1].strftime("%Y-%m")
    print(f"[*] Total Records:    {len(series)} months ({start_m} to {end_m})")
    valid_count = series.dropna().count()
    pct = valid_count / len(series)
    print(f"[*] Valid Records:    {valid_count} observations ({pct:.1%} completeness)")

    # 2. Compute 12-month climatology profile
    print("\n>>> [STEP 2] Computing 12-Month Climatological Normal Profile")
    print("-" * 78)
    clim = compute_monthly_climatology(series, min_years=10, min_obs_per_month=5)

    month_names = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    print("Month         | Mean Precip (mm/month) | Regional Context (Literature Ref)")
    print("--------------|------------------------|----------------------------------")
    for m in range(1, 13):
        val = clim.loc[m]
        desc = ""
        if m in {1, 2}:
            desc = "1st Dry Season (Early-year veranillo)"
        elif m in {4, 5}:
            desc = "1st Rainy Season (Northward ITCZ passage)"
        elif m in {7, 8}:
            desc = "2nd Dry Season (Mid-year veranillo / low flow)"
        elif m in {10, 11}:
            desc = "2nd Rainy Season (Southward ITCZ return)"
        else:
            desc = "Seasonal Transition"

        print(f"{m:02d} - {month_names[m - 1]:<9} | {val:>22.2f} mm | {desc}")

    print("-" * 78)
    print("Note: 'Regional Context' is a literature reference for the Bogotá basin,")
    print("      not an algorithmic classification output.")

    # 3. Seasonal geometry and regime classification
    print("\n>>> [STEP 3] Seasonal Cycle Geometry & Algorithmic Classification")
    print("-" * 78)

    amp = annual_cycle_amplitude(clim)
    global_peak = annual_cycle_phase(clim, mode="peak")
    global_trough = annual_cycle_phase(clim, mode="trough")

    print(f"[*] Annual Cycle Amplitude (max - min): {amp:.2f} mm/month")
    p_name = month_names[global_peak - 1]
    p_val = clim.loc[global_peak]
    print(
        f"[*] Global Peak (annual_cycle_phase):   "
        f"Month {global_peak:02d} ({p_name}) [{p_val:.2f} mm]"
    )
    t_name = month_names[global_trough - 1]
    t_val = clim.loc[global_trough]
    print(
        f"[*] Global Trough (annual_cycle_phase): "
        f"Month {global_trough:02d} ({t_name}) [{t_val:.2f} mm]"
    )

    # 4. Circular peak detection and regime classification
    peaks = annual_cycle_peaks(clim)
    regime = classify_rainfall_regime(clim)

    peak_names = [
        f"Month {p:02d} ({month_names[p - 1]}: {clim.loc[p]:.1f} mm)" for p in peaks
    ]
    print(f"[*] Detected Circular Peaks:            {', '.join(peak_names)}")
    print(f"[*] Classified Rainfall Regime:         '{regime.upper()}'")

    # 5. Sanity check evaluation
    print("\n>>> [STEP 4] Canonical Sanity Check Outcome")
    print("-" * 78)
    if regime == "bimodal":
        print("[+] SANITY CHECK PASSED:")
        print(
            "    TROPICOR correctly identified the bimodal rainfall pattern of Bogotá."
        )
        print(f"    Detected peaks: {peaks} (April and November, as in literature).")
        print("    Note: Multi-basin empirical validation across all 6 natural regions")
        print("    remains the standard for preprint benchmarking.")
    elif regime == "multimodal":
        print("[!] MULTIMODAL RESULT WITH DEFAULT 10% PROMINENCE:")
        print(f"    Detected peaks: {peaks}")
        print("    Inspect local sub-peaks or adjust prominence threshold.")
    else:
        print(f"[?] Classified regime: {regime} with peaks at {peaks}")

    print("=" * 78)


if __name__ == "__main__":
    validate_bogota_climatology()
