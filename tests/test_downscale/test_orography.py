"""Unit tests for orographic and lapse-rate vertical temperature correction."""

import numpy as np
import pandas as pd
import pytest

from tropicor.core.metrics import compute_validation_metrics
from tropicor.downscale.orography import (
    DRY_ADIABATIC_LAPSE_RATE,
    MOIST_ADIABATIC_LAPSE_RATE,
    STANDARD_LAPSE_RATE,
    compute_elevation_offset,
    lapse_rate_temperature_correction,
)


class TestElevationOffset:
    """Test suite for compute_elevation_offset."""

    def test_uis_bucaramanga_offset(self) -> None:
        """Verify elevation discrepancy for the Bucaramanga UIS benchmark station."""
        station_z = 898.0
        model_z = 2118.0
        offset = compute_elevation_offset(station_z, model_z)
        assert offset == pytest.approx(-1220.0)

    def test_paramo_high_elevation_offset(self) -> None:
        """Verify positive elevation discrepancy for mountain peak/paramo."""
        station_z = 3600.0
        model_z = 2400.0
        offset = compute_elevation_offset(station_z, model_z)
        assert offset == pytest.approx(1200.0)

    def test_zero_elevation_offset(self) -> None:
        """Verify zero discrepancy when elevations match exactly."""
        offset = compute_elevation_offset(500.0, 500.0)
        assert offset == pytest.approx(0.0)

    def test_invalid_non_finite_elevations(self) -> None:
        """Verify ValueError is raised when either elevation is NaN or infinite."""
        with pytest.raises(ValueError, match="Elevations must be finite numbers"):
            compute_elevation_offset(float("nan"), 1000.0)

        with pytest.raises(ValueError, match="Elevations must be finite numbers"):
            compute_elevation_offset(500.0, float("inf"))


class TestLapseRateTemperatureCorrection:
    """Test suite for lapse_rate_temperature_correction."""

    def test_uis_scalar_benchmark(self) -> None:
        """Verify +7.93 °C adjustment at UIS Bucaramanga with standard lapse rate."""
        raw_temp = 16.0
        corrected = lapse_rate_temperature_correction(
            temperature=raw_temp,
            station_elevation=898.0,
            model_elevation=2118.0,
            lapse_rate=STANDARD_LAPSE_RATE,
        )
        # delta_t = -0.0065 * (898 - 2118) = -0.0065 * (-1220) = +7.93 °C
        assert corrected == pytest.approx(16.0 + 7.93, rel=1e-5)

    def test_dry_and_moist_lapse_rates(self) -> None:
        """Verify dry and moist adiabatic adjustments."""
        raw_temp = 20.0
        z_station = 1000.0
        z_model = 2000.0  # delta_z = -1000.0

        # Dry: -0.0098 * (-1000) = +9.8 °C
        corrected_dry = lapse_rate_temperature_correction(
            temperature=raw_temp,
            station_elevation=z_station,
            model_elevation=z_model,
            lapse_rate=DRY_ADIABATIC_LAPSE_RATE,
        )
        assert corrected_dry == pytest.approx(29.8, rel=1e-5)

        # Moist: -0.0050 * (-1000) = +5.0 °C
        corrected_moist = lapse_rate_temperature_correction(
            temperature=raw_temp,
            station_elevation=z_station,
            model_elevation=z_model,
            lapse_rate=MOIST_ADIABATIC_LAPSE_RATE,
        )
        assert corrected_moist == pytest.approx(25.0, rel=1e-5)

    def test_elevation_invariance_when_equal(self) -> None:
        """Verify that temperature is unchanged when elevations are identical."""
        raw_temp = 18.5
        corrected = lapse_rate_temperature_correction(
            temperature=raw_temp,
            station_elevation=1500.0,
            model_elevation=1500.0,
        )
        assert corrected == pytest.approx(raw_temp)

    def test_paramo_cooling_adjustment(self) -> None:
        """Verify temperature decreases when station is higher than model grid."""
        raw_temp = 12.0
        z_station = 3500.0
        z_model = 2500.0  # delta_z = +1000.0

        # delta_t = -0.0065 * (+1000) = -6.5 °C
        corrected = lapse_rate_temperature_correction(
            temperature=raw_temp,
            station_elevation=z_station,
            model_elevation=z_model,
        )
        assert corrected == pytest.approx(5.5, rel=1e-5)

    def test_pandas_series_preservation(self) -> None:
        """Verify index, name, and NaNs are preserved on pandas Series."""
        dates = pd.date_range("2010-01-01", periods=12, freq="MS")
        raw_series = pd.Series(
            [15.0, 15.5, 16.0, np.nan, 16.5, 17.0, 16.8, 16.2, 15.9, 15.4, 15.1, 14.8],
            index=dates,
            name="t2m_era5",
        )

        corrected_series = lapse_rate_temperature_correction(
            temperature=raw_series,
            station_elevation=898.0,
            model_elevation=2118.0,
        )

        assert isinstance(corrected_series, pd.Series)
        assert corrected_series.name == "t2m_era5"
        pd.testing.assert_index_equal(corrected_series.index, raw_series.index)
        assert np.isnan(corrected_series.iloc[3])
        assert corrected_series.iloc[0] == pytest.approx(15.0 + 7.93, rel=1e-5)

    def test_pandas_dataframe_preservation(self) -> None:
        """Verify temperature columns are adjusted concurrently in DataFrame."""
        dates = pd.date_range("2010-01-01", periods=6, freq="MS")
        raw_df = pd.DataFrame(
            {
                "t_min": [10.0, 11.0, 12.0, 11.5, 10.5, 9.8],
                "t_max": [22.0, 23.0, 24.0, 23.5, 22.5, 21.8],
            },
            index=dates,
        )

        corrected_df = lapse_rate_temperature_correction(
            temperature=raw_df,
            station_elevation=898.0,
            model_elevation=2118.0,
        )

        assert isinstance(corrected_df, pd.DataFrame)
        pd.testing.assert_index_equal(corrected_df.index, raw_df.index)
        assert list(corrected_df.columns) == ["t_min", "t_max"]
        assert corrected_df["t_min"].iloc[0] == pytest.approx(17.93, rel=1e-5)
        assert corrected_df["t_max"].iloc[0] == pytest.approx(29.93, rel=1e-5)

    def test_numpy_ndarray_preservation(self) -> None:
        """Verify 1D and 2D numpy arrays are adjusted preserving exact shape."""
        arr_1d = np.array([14.0, 15.0, 16.0])
        corr_1d = lapse_rate_temperature_correction(arr_1d, 898.0, 2118.0)
        assert isinstance(corr_1d, np.ndarray)
        assert corr_1d.shape == (3,)
        np.testing.assert_allclose(corr_1d, arr_1d + 7.93, rtol=1e-5)

        arr_2d = np.array([[14.0, 15.0], [16.0, 17.0]])
        corr_2d = lapse_rate_temperature_correction(arr_2d, 898.0, 2118.0)
        assert corr_2d.shape == (2, 2)
        np.testing.assert_allclose(corr_2d, arr_2d + 7.93, rtol=1e-5)

    def test_time_varying_lapse_rate_series(self) -> None:
        """Verify applying a monthly time-varying lapse rate series."""
        dates = pd.date_range("2010-01-01", periods=3, freq="MS")
        raw_series = pd.Series([15.0, 16.0, 17.0], index=dates)
        varying_rates = pd.Series([0.0060, 0.0065, 0.0070], index=dates)

        # delta_z = -1000
        # adjustments: -(-1000) * [0.0060, 0.0065, 0.0070] = [+6.0, +6.5, +7.0]
        corrected = lapse_rate_temperature_correction(
            temperature=raw_series,
            station_elevation=1000.0,
            model_elevation=2000.0,
            lapse_rate=varying_rates,
        )

        expected = pd.Series([21.0, 22.5, 24.0], index=dates)
        pd.testing.assert_series_equal(corrected, expected)

    def test_pearson_invariance_and_error_reduction(self) -> None:
        """Verify Pearson correlation shift-invariance and drop in RMSE and Bias."""
        dates = pd.date_range("1990-01-01", periods=120, freq="MS")
        np.random.seed(42)

        # Ground-truth station temperatures around 24 °C (Bucaramanga plateau)
        true_station = pd.Series(
            24.0
            + 2.0 * np.sin(np.linspace(0, 10 * np.pi, 120))
            + np.random.normal(0, 0.5, 120),
            index=dates,
        )

        # Raw ERA5 at smoothed model elevation (2,118 vs 898 masl) is ~8 °C colder
        raw_era5 = true_station - 7.93 + np.random.normal(0, 0.2, 120)

        # Report before correction
        report_before = compute_validation_metrics(true_station, raw_era5)

        # Correct raw ERA5 via lapse rate
        corrected_era5 = lapse_rate_temperature_correction(
            temperature=raw_era5,
            station_elevation=898.0,
            model_elevation=2118.0,
            lapse_rate=STANDARD_LAPSE_RATE,
        )

        # Report after correction
        report_after = compute_validation_metrics(true_station, corrected_era5)

        # 1. Pearson r MUST remain mathematically invariant
        assert report_before.pearson_r == pytest.approx(
            report_after.pearson_r, rel=1e-12
        )

        # 2. Bias drops toward near zero
        assert abs(report_after.bias) < 0.1
        assert abs(report_before.bias) > 7.5

        # 3. RMSE and MAE drop by orders of magnitude
        assert report_after.rmse < 1.0
        assert report_before.rmse > 7.5
        assert report_after.mae < 1.0
        assert report_before.mae > 7.5

        # 4. Euclidean distance contracts significantly
        assert report_after.euclidean_distance < report_before.euclidean_distance

    def test_negative_lapse_rate_guard(self) -> None:
        """Verify ValueError is raised on negative rate unless allow_inversion=True."""
        with pytest.raises(ValueError, match="Negative lapse rate"):
            lapse_rate_temperature_correction(
                temperature=20.0,
                station_elevation=1000.0,
                model_elevation=1500.0,
                lapse_rate=-0.002,
                allow_inversion=False,
            )

        # Allowed when explicitly requested
        res = lapse_rate_temperature_correction(
            temperature=20.0,
            station_elevation=1000.0,
            model_elevation=1500.0,
            lapse_rate=-0.002,
            allow_inversion=True,
        )
        # delta_z = -500. delta_t = -(-0.002) * (-500) = -1.0
        assert res == pytest.approx(19.0)

    def test_invalid_lapse_rate_and_elevation(self) -> None:
        """Verify exceptions for non-finite inputs and unsupported types."""
        with pytest.raises(ValueError, match="Lapse rate must be finite"):
            lapse_rate_temperature_correction(
                20.0, 1000.0, 1500.0, lapse_rate=float("inf")
            )

        with pytest.raises(TypeError, match="Unsupported temperature type"):
            lapse_rate_temperature_correction("invalid_str", 1000.0, 1500.0)  # type: ignore[arg-type]

        with pytest.raises(
            ValueError, match="Cannot apply time-varying vector lapse rate"
        ):
            lapse_rate_temperature_correction(
                temperature=20.0,
                station_elevation=1000.0,
                model_elevation=1500.0,
                lapse_rate=pd.Series([0.006, 0.007]),
            )
