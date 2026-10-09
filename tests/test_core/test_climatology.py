"""Unit tests for annual climatological curves and regime classification."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tropicor.core.climatology import (
    annual_cycle_amplitude,
    annual_cycle_peaks,
    annual_cycle_phase,
    classify_rainfall_regime,
    climatological_curve_distance,
    compute_monthly_climatology,
)
from tropicor.io.era5 import ERA5Adapter


def test_compute_monthly_climatology_synthetic_cycle() -> None:
    """Test 12-month climatology computation from known analytical harmonic."""
    # 15 years of monthly data (180 months)
    dates = pd.date_range("2000-01-01", periods=180, freq="MS")
    # T(m) = 22.0 + 3.0 * sin(2*pi * (m - 1) / 12)
    months = dates.month
    values = 22.0 + 3.0 * np.sin(2.0 * np.pi * (months - 1) / 12.0)
    series = pd.Series(values, index=dates, name="synthetic_temp")

    clim = compute_monthly_climatology(series, min_years=10)

    assert isinstance(clim, pd.Series)
    assert len(clim) == 12
    assert list(clim.index) == list(range(1, 13))
    assert clim.index.name == "month"
    assert clim.name == "synthetic_temp"

    for m in range(1, 13):
        expected_val = 22.0 + 3.0 * np.sin(2.0 * np.pi * (m - 1) / 12.0)
        assert math.isclose(clim.loc[m], expected_val, rel_tol=1e-5)


def test_compute_monthly_climatology_default_min_years() -> None:
    """Test default min_years=10 threshold raises ValueError for short series."""
    # Only 4 years of data
    dates = pd.date_range("2010-01-01", periods=48, freq="MS")
    series = pd.Series(np.ones(48), index=dates)

    with pytest.raises(ValueError, match="min_years=10 is required"):
        compute_monthly_climatology(series)

    # Overriding min_years succeeds
    clim = compute_monthly_climatology(series, min_years=3, min_obs_per_month=3)
    assert len(clim) == 12
    assert not clim.isna().any()


def test_compute_monthly_climatology_min_obs_per_month() -> None:
    """Test observational completeness filter assigns NaN to sparse months."""
    dates = pd.date_range("2000-01-01", periods=144, freq="MS")  # 12 years
    values = np.full(144, 25.0)

    # Artificially remove June observations except for 2 valid records
    series = pd.Series(values, index=dates)
    june_mask = dates.month == 6
    # Leave only first 2 Junes valid, set others to NaN (10 NaNs)
    june_indices = series[june_mask].index
    series.loc[june_indices[2:]] = np.nan

    warn_match = "have fewer than 5 valid observations"
    with pytest.warns(UserWarning, match=warn_match):
        clim = compute_monthly_climatology(series, min_years=10, min_obs_per_month=5)

    assert np.isnan(clim.loc[6])
    assert not np.isnan(clim.loc[1])
    assert clim.loc[1] == 25.0


def test_compute_monthly_climatology_invalid_index() -> None:
    """Test compute_monthly_climatology rejects non-DatetimeIndex and empty inputs."""
    s_int = pd.Series([1.0, 2.0, 3.0], index=[1, 2, 3])
    with pytest.raises(TypeError, match="must be a pandas DatetimeIndex"):
        compute_monthly_climatology(s_int)

    s_empty = pd.Series(dtype=np.float64, index=pd.DatetimeIndex([]))
    with pytest.raises(ValueError, match="contains no valid non-NaN observations"):
        compute_monthly_climatology(s_empty)


def test_climatological_curve_distance_identical() -> None:
    """Test Euclidean distance between identical 12-month curves is 0.0."""
    clim = pd.Series(np.linspace(10.0, 20.0, 12), index=range(1, 13))
    dist = climatological_curve_distance(clim, clim)
    assert math.isclose(dist, 0.0, abs_tol=1e-7)


def test_climatological_curve_distance_constant_shift() -> None:
    """Test constant uniform shift Delta produces d = Delta * sqrt(12)."""
    delta = 3.5
    c1 = pd.Series(np.full(12, 20.0), index=range(1, 13))
    c2 = pd.Series(np.full(12, 20.0 + delta), index=range(1, 13))

    dist = climatological_curve_distance(c1, c2)
    expected_dist = delta * math.sqrt(12.0)
    assert math.isclose(dist, expected_dist, rel_tol=1e-6)


def test_climatological_curve_distance_normalized() -> None:
    """Test normalized Euclidean distance modes: 'mean', 'amplitude', 'std'."""
    # Observed: linear ramp from 10 to 32 (amplitude 22, mean 21.0)
    obs = pd.Series(np.linspace(10.0, 32.0, 12), index=range(1, 13))
    shift = 2.0
    mod = obs + shift
    raw_dist = shift * math.sqrt(12.0)

    # 1. Normalize by mean
    dist_mean = climatological_curve_distance(obs, mod, normalize="mean")
    assert math.isclose(dist_mean, raw_dist / float(np.mean(obs.values)), rel_tol=1e-5)

    # 2. Normalize by amplitude
    dist_amp = climatological_curve_distance(obs, mod, normalize="amplitude")
    assert math.isclose(dist_amp, raw_dist / 22.0, rel_tol=1e-5)

    # 3. Normalize by std
    dist_std = climatological_curve_distance(obs, mod, normalize="std")
    assert math.isclose(
        dist_std, raw_dist / float(np.std(obs.values, ddof=1)), rel_tol=1e-5
    )

    # 4. Invalid mode raises ValueError
    with pytest.raises(ValueError, match="Invalid normalize mode"):
        climatological_curve_distance(
            obs,
            mod,
            normalize="invalid",  # type: ignore[arg-type]
        )


def test_climatological_curve_distance_thesis_dissociation() -> None:
    """Test UNAL thesis finding: curves with r=1 have non-zero distance."""
    # Annual cycle shape
    t = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    y1 = pd.Series(20.0 + 5.0 * np.sin(t), index=range(1, 13))
    y2 = pd.Series(20.0 + 10.0 * np.sin(t), index=range(1, 13))  # 2x amplitude

    dist = climatological_curve_distance(y1, y2)
    assert dist > 0.0

    # Verify Pearson r is 1.0 despite substantial curve separation
    r = float(np.corrcoef(y1.values, y2.values)[0, 1])
    assert math.isclose(r, 1.0, rel_tol=1e-5)


def test_climatological_curve_distance_missing_months_raises_clear_error() -> None:
    """Test ValueError is raised when either curve contains incomplete months."""
    c1 = pd.Series(np.ones(12), index=range(1, 13))
    c2 = pd.Series(np.ones(12), index=range(1, 13))
    c2.loc[6] = np.nan  # June missing

    with pytest.raises(ValueError, match="Found only 11 valid months"):
        climatological_curve_distance(c1, c2)


def test_annual_cycle_amplitude() -> None:
    """Test annual cycle amplitude calculation."""
    clim = pd.Series([10.0, 15.0, 25.0, 5.0, 12.0], index=range(1, 6))
    amp = annual_cycle_amplitude(clim)
    assert math.isclose(amp, 20.0, abs_tol=1e-5)  # 25 - 5

    # Flat curve amplitude is 0.0
    flat = pd.Series(np.full(12, 18.0), index=range(1, 13))
    assert math.isclose(annual_cycle_amplitude(flat), 0.0, abs_tol=1e-5)

    # Empty raises ValueError
    empty = pd.Series([np.nan, np.nan])
    with pytest.raises(ValueError, match="contains no valid non-NaN values"):
        annual_cycle_amplitude(empty)


def test_annual_cycle_phase_global() -> None:
    """Test global annual cycle phase detection for peak and trough."""
    # Profile with peak in October (month 10) and trough in January (month 1)
    vals = [5.0, 10.0, 15.0, 20.0, 18.0, 12.0, 10.0, 14.0, 22.0, 35.0, 25.0, 10.0]
    clim = pd.Series(vals, index=range(1, 13))

    assert annual_cycle_phase(clim, mode="peak") == 10
    assert annual_cycle_phase(clim, mode="max") == 10
    assert annual_cycle_phase(clim, mode="trough") == 1
    assert annual_cycle_phase(clim, mode="valley") == 1
    assert annual_cycle_phase(clim, mode="min") == 1

    with pytest.raises(ValueError, match="Invalid mode"):
        annual_cycle_phase(clim, mode="invalid")  # type: ignore[arg-type]


def test_annual_cycle_peaks_bimodal_andean() -> None:
    """Test circular peak detection for bimodal Andean precipitation cycle."""
    # Synthetic Andean regime (e.g. Bogotá / Medellín):
    # Peaks in April (m=4) and October (m=10); troughs in January (m=1) and July (m=7)
    # Month:  1   2   3   4   5   6   7   8   9  10  11  12
    vals = [30, 45, 80, 140, 110, 55, 40, 50, 75, 160, 130, 50]
    clim = pd.Series(vals, index=range(1, 13))

    peaks = annual_cycle_peaks(clim)
    assert peaks == [4, 10]

    regime = classify_rainfall_regime(clim)
    assert regime == "bimodal"


def test_annual_cycle_peaks_unimodal_orinoquia() -> None:
    """Test circular peak detection for unimodal Orinoquía / Amazonía regime."""
    # Synthetic Orinoquía regime (e.g. Villavicencio):
    # Single peak in June (m=6); dry season in January (m=1)
    # Month:  1   2   3   4    5    6    7    8    9   10  11  12
    vals = [20, 40, 90, 200, 320, 410, 350, 290, 240, 190, 90, 35]
    clim = pd.Series(vals, index=range(1, 13))

    peaks = annual_cycle_peaks(clim)
    assert peaks == [6]

    regime = classify_rainfall_regime(clim)
    assert regime == "unimodal"


def test_annual_cycle_peaks_circular_boundary() -> None:
    """Test circular boundary wrap-around when a peak sits at December (m=12)."""
    # Peaks in May (m=5) and December (m=12)
    # December has November (m=11) and January (m=1) as lower neighbors
    # Month:  1   2   3   4   5    6   7   8   9   10  11  12
    vals = [40, 35, 50, 90, 130, 60, 30, 45, 60, 75, 90, 150]
    clim = pd.Series(vals, index=range(1, 13))

    peaks = annual_cycle_peaks(clim)
    assert peaks == [5, 12]

    regime = classify_rainfall_regime(clim)
    assert regime == "bimodal"


def test_classify_rainfall_regime_flat_and_multimodal() -> None:
    """Test classify_rainfall_regime handles indeterminate and multimodal cycles."""
    # Completely flat precipitation cycle (zero prominence peaks)
    flat = pd.Series(np.full(12, 100.0), index=range(1, 13))
    assert classify_rainfall_regime(flat) == "indeterminate"

    # Multimodal profile (3 peaks: m=2, m=6, m=10)
    # Month:  1   2   3  4  5   6   7  8  9  10  11 12
    vals = [10, 50, 15, 10, 20, 60, 20, 15, 25, 55, 20, 10]
    multi = pd.Series(vals, index=range(1, 13))
    peaks = annual_cycle_peaks(multi, min_prominence=10.0)
    assert len(peaks) == 3
    assert classify_rainfall_regime(multi, min_prominence=10.0) == "multimodal"


# Explicit bimodal profile used for hand-computed prominences.
# Month:        1   2    3    4    5   6   7   8   9   10   11  12
_BIMODAL = [55, 70, 110, 140, 125, 75, 50, 50, 70, 125, 145, 80]


def test_annual_cycle_peaks_default_return_type_unchanged() -> None:
    """Default call still returns a plain list of months."""
    clim = pd.Series(_BIMODAL, index=range(1, 13), dtype=float)
    peaks = annual_cycle_peaks(clim)
    assert isinstance(peaks, list)
    assert peaks == [4, 11]


def test_annual_cycle_peaks_return_prominences_hand_computed() -> None:
    """Prominences match values derived by hand on the circular cycle.

    April (140): lowest base on the lower side is Jan (55), the other side
    reaches Jul/Aug (50), so prominence = 140 - 55 = 85.
    November (145, global max): both bases reach 50, so prominence = 95.
    """
    clim = pd.Series(_BIMODAL, index=range(1, 13), dtype=float)
    months, proms = annual_cycle_peaks(clim, return_prominences=True)
    assert months == [4, 11]
    assert proms == pytest.approx([85.0, 95.0])


def test_annual_cycle_peaks_prominences_match_months_of_default_call() -> None:
    """Months with and without prominences are identical."""
    clim = pd.Series(_BIMODAL, index=range(1, 13), dtype=float)
    months, _ = annual_cycle_peaks(clim, return_prominences=True)
    assert months == annual_cycle_peaks(clim)


def test_annual_cycle_peaks_prominences_are_in_data_units() -> None:
    """Scaling the climatology scales prominences, they are not fractions."""
    clim = pd.Series(_BIMODAL, index=range(1, 13), dtype=float)
    _, proms = annual_cycle_peaks(clim, return_prominences=True)
    _, proms_x10 = annual_cycle_peaks(clim * 10.0, return_prominences=True)
    assert proms_x10 == pytest.approx([p * 10.0 for p in proms])


def test_annual_cycle_peaks_prominences_respect_threshold() -> None:
    """Every returned prominence is at least the requested minimum."""
    clim = pd.Series(_BIMODAL, index=range(1, 13), dtype=float)
    months, proms = annual_cycle_peaks(
        clim, min_prominence=90.0, return_prominences=True
    )
    assert months == [11]
    assert proms == pytest.approx([95.0])
    assert all(p >= 90.0 for p in proms)


def test_annual_cycle_peaks_prominences_wrap_december_january() -> None:
    """A peak at the calendar boundary keeps a circular prominence."""
    vals = [100, 40, 20, 20, 20, 20, 20, 20, 20, 20, 40, 90]
    clim = pd.Series(vals, index=range(1, 13), dtype=float)
    months, proms = annual_cycle_peaks(clim, return_prominences=True)
    assert months == [1]
    assert proms == pytest.approx([80.0])


def test_annual_cycle_peaks_prominences_empty_for_flat_cycle() -> None:
    """A flat cycle has no peaks and returns two empty lists."""
    flat = pd.Series(np.full(12, 100.0), index=range(1, 13))
    months, proms = annual_cycle_peaks(flat, return_prominences=True)
    assert months == [] and proms == []


def test_annual_cycle_peaks_prominences_still_require_12_months() -> None:
    """The completeness check applies to the new option as well."""
    clim = pd.Series(_BIMODAL, index=range(1, 13), dtype=float)
    clim.loc[3] = np.nan
    with pytest.raises(ValueError, match="12 valid"):
        annual_cycle_peaks(clim, return_prominences=True)


def test_era5_adapter_get_point_series_alias(synthetic_era5_dataset: Path) -> None:
    """Test that ERA5Adapter.get_point_series matches get_series output."""
    adapter = ERA5Adapter(synthetic_era5_dataset)

    lat = 4.6
    lon = -74.1

    s_orig = adapter.get_series(lat, lon, "t2m")
    s_alias = adapter.get_point_series(lat, lon, "t2m")

    pd.testing.assert_series_equal(s_orig, s_alias)
    adapter.close()
