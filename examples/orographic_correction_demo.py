"""Synthetic mathematical benchmark for physical orographic lapse-rate correction.

This script demonstrates the algebraic and statistical mechanics of vertical
downscaling using synthetic temperature series modeled after the Bucaramanga UIS
station topography (Delta z = -1220 m -> +7.93 deg C warming adjustment).

SCIENTIFIC DISCLOSURE:
The series evaluated here are synthetically generated to verify unit and formula
consistency. In real empirical observations, complex boundary layer processes
(such as nocturnal inversions and cold-air pooling) introduce non-linear noise,
so RMSE will not collapse to 0.25 deg C in real-world applications.

Mathematical & Statistical Properties Verified:
- Affine Shift Invariance: Because constant lapse rate adjustment is a uniform
  scalar offset Delta T = -Gamma * Delta z, Pearson r is mathematically invariant.
- Metric Reduction: Mean Bias, RMSE, MAE, and Euclidean distance drop dramatically,
  demonstrating that distance and error metrics are essential for reanalysis audits.
"""

import numpy as np
import pandas as pd

from tropicor.core.metrics import compute_validation_metrics
from tropicor.downscale import (
    STANDARD_LAPSE_RATE,
    compute_elevation_offset,
    lapse_rate_temperature_correction,
)
from tropicor.io.stations import StationCatalog


def run_benchmark_demonstration() -> None:
    """Execute the Bucaramanga UIS benchmark orographic correction audit."""
    print("=" * 80)
    print("TROPICOR: Synthetic Benchmark — Orographic Lapse-Rate Correction")
    print("Benchmark Case: Bucaramanga UIS Station Topography (Code: 23195040)")
    print("=" * 80)
    print("  [!] SCIENTIFIC DISCLOSURE:")
    print("      This script evaluates SYNTHETIC temperature data generated to")
    print("      verify the mathematical plumping of the lapse-rate correction.")
    print("      In real field data, nocturnal inversions and boundary layer noise")
    print("      prevent RMSE from collapsing to near zero.")
    print("-" * 80)

    # 1. Retrieve canonical benchmark station metadata
    catalog = StationCatalog.from_benchmark()
    station = catalog.get("23195040")

    station_elevation = station.elevation  # 898.0 masl
    era5_elevation = 2118.0  # Smoothed Cordillera Oriental grid node

    delta_z = compute_elevation_offset(station_elevation, era5_elevation)
    expected_delta_t = -1.0 * STANDARD_LAPSE_RATE * delta_z

    print(f"\nStation Name:           {station.name}")
    print(f"Department / Region:    {station.department} ({station.region.value})")
    print(f"Station Elevation:      {station_elevation:.1f} masl")
    print(f"ERA5 Model Elevation:   {era5_elevation:.1f} masl")
    print(f"Elevation Discrepancy:  {delta_z:.1f} m (station is in lower valley)")
    print(f"Physical Adjustment:    {expected_delta_t:+.2f} deg C (expected warming)")

    # 2. Synthesize 10-year monthly temperature series (1990-1999, 120 months)
    # Ground-truth: Bucaramanga tropical plateau temperature (~24.0 deg C mean)
    dates = pd.date_range("1990-01-01", periods=120, freq="MS")
    np.random.seed(42)

    seasonal_cycle = 1.8 * np.sin(np.linspace(0, 10 * 2 * np.pi, 120))
    enso_noise = np.random.normal(0, 0.4, 120)
    obs_temp = pd.Series(
        24.0 + seasonal_cycle + enso_noise,
        index=dates,
        name="ideam_synthetic",
    )

    # Raw ERA5 at smoothed mountain elevation (2,118 masl) is ~7.93 deg C colder
    model_noise = np.random.normal(0, 0.25, 120)
    raw_era5_temp = pd.Series(
        obs_temp - expected_delta_t + model_noise,
        index=dates,
        name="era5_raw_synthetic",
    )

    # 3. Compute baseline metrics before correction
    report_before = compute_validation_metrics(obs_temp, raw_era5_temp)

    print("\n" + "-" * 40)
    print("1. BEFORE CORRECTION (Synthetic Raw ERA5):")
    print("-" * 40)
    print(f"  Observed Mean:     {obs_temp.mean():.2f} deg C")
    print(f"  Model Mean:        {raw_era5_temp.mean():.2f} deg C")
    print(f"  Mean Bias:         {report_before.bias:+.2f} deg C (severe cold bias)")
    print(f"  RMSE:              {report_before.rmse:.2f} deg C")
    print(f"  MAE:               {report_before.mae:.2f} deg C")
    print(f"  Pearson r:         {report_before.pearson_r:.4f}")
    print(f"  Euclidean Dist d:  {report_before.euclidean_distance:.2f}")

    # 4. Apply physical orographic lapse-rate correction
    corrected_era5_temp = lapse_rate_temperature_correction(
        temperature=raw_era5_temp,
        station_elevation=station_elevation,
        model_elevation=era5_elevation,
        lapse_rate=STANDARD_LAPSE_RATE,
    )

    # 5. Compute validation metrics after correction
    report_after = compute_validation_metrics(obs_temp, corrected_era5_temp)

    print("\n" + "-" * 40)
    print("2. AFTER OROGRAPHIC LAPSE-RATE CORRECTION:")
    print("-" * 40)
    print(f"  Observed Mean:     {obs_temp.mean():.2f} deg C")
    print(f"  Model Mean:        {corrected_era5_temp.mean():.2f} deg C")
    print(f"  Mean Bias:         {report_after.bias:+.2f} deg C (virtually zero)")
    print(f"  RMSE:              {report_after.rmse:.2f} deg C")
    print(f"  MAE:               {report_after.mae:.2f} deg C")
    print(f"  Pearson r:         {report_after.pearson_r:.4f} (invariant)")
    print(f"  Euclidean Dist d:  {report_after.euclidean_distance:.2f}")

    # 6. Verify mathematical invariant and error reduction
    print("\n" + "=" * 80)
    print("SCIENTIFIC SUMMARY & VERIFICATION:")
    print("=" * 80)
    print(
        f"[OK] Bias reduction:     {report_before.bias:+.2f} deg C -> "
        f"{report_after.bias:+.2f} deg C"
    )
    print(
        f"[OK] RMSE reduction:     {report_before.rmse:.2f} deg C -> "
        f"{report_after.rmse:.2f} deg C"
    )
    print(
        f"[OK] Pearson invariance: r_raw = {report_before.pearson_r:.4f} == "
        f"r_corrected = {report_after.pearson_r:.4f}"
    )
    print(
        "[OK] Pedagogical insight: Pearson r is invariant under constant offset, "
        "proving why RMSE and Bias are mandatory for orographic evaluations."
    )
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark_demonstration()
