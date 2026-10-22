"""Synthetic tests for daily temperature downscaling with empty (100% NaN) years.

Verifies Protocol requirements:
- Item 8: tune_level3_temperature_hyperparameters handles 12-year series with 5
  intercalated 100% NaN observation years, creating exactly 7 folds with zero NaNs,
  and selects the identical winning configuration as the filtered series.
- Item 9: Level1 and Level2 fit with empty observation years produce identical
  biases and scaling factors as the filtered series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tropicor.downscale.temperature import (
    Level1MonthlyAdditiveTemperature,
    Level2MeanStateDTRScalingTemperature,
    tune_level3_temperature_hyperparameters,
)


def _generate_12yr_series():
    """Generate deterministic 12-year daily series (2000-2011)."""
    dates = pd.date_range("2000-01-01", "2011-12-31", freq="D")
    n = len(dates)
    rng = np.random.RandomState(42)

    doy = dates.dayofyear.values
    mod_tmax = pd.Series(
        25.0 + 3.0 * np.sin(2 * np.pi * doy / 365) + rng.normal(0, 1.0, n),
        index=dates,
    )
    mod_tmin = pd.Series(
        15.0 + 2.0 * np.sin(2 * np.pi * doy / 365) + rng.normal(0, 1.0, n),
        index=dates,
    )

    obs_tmax = mod_tmax - 1.5 + rng.normal(0, 0.5, n)
    obs_tmin = mod_tmin - 0.8 + rng.normal(0, 0.5, n)

    empty_years = [2001, 2003, 2005, 2007, 2009]
    valid_years = [y for y in range(2000, 2012) if y not in empty_years]

    obs_tmax_masked = obs_tmax.copy()
    obs_tmin_masked = obs_tmin.copy()
    for y in empty_years:
        obs_tmax_masked[dates.year == y] = np.nan
        obs_tmin_masked[dates.year == y] = np.nan

    valid_mask = dates.year.isin(valid_years)
    clean_obs_tmax = obs_tmax[valid_mask]
    clean_obs_tmin = obs_tmin[valid_mask]
    clean_mod_tmax = mod_tmax[valid_mask]
    clean_mod_tmin = mod_tmin[valid_mask]

    return {
        "dates": dates,
        "empty_years": empty_years,
        "valid_years": valid_years,
        "mod_tmax": mod_tmax,
        "mod_tmin": mod_tmin,
        "obs_tmax_masked": obs_tmax_masked,
        "obs_tmin_masked": obs_tmin_masked,
        "clean_obs_tmax": clean_obs_tmax,
        "clean_obs_tmin": clean_obs_tmin,
        "clean_mod_tmax": clean_mod_tmax,
        "clean_mod_tmin": clean_mod_tmin,
    }


def test_tune_level3_temperature_with_5_intercalated_empty_years():
    """Item 8: 12-yr series with 5 empty years -> 7 folds, 0 NaNs, matching winner."""
    data = _generate_12yr_series()

    # 1. Tuning on 12-year series with 5 intercalated 100% NaN observation years
    res_masked, model_masked, all_res_masked = tune_level3_temperature_hyperparameters(
        obs_tmax=data["obs_tmax_masked"],
        obs_tmin=data["obs_tmin_masked"],
        mod_tmax=data["mod_tmax"],
        mod_tmin=data["mod_tmin"],
    )

    # Must produce exactly len(valid_years) = 7 folds
    expected_folds = len(data["valid_years"])
    assert len(res_masked.annual_fold_losses) == expected_folds, (
        f"Expected {expected_folds} folds, got {len(res_masked.annual_fold_losses)}"
    )

    # Must not have NaNs in any fold loss
    for i, fold_l in enumerate(res_masked.annual_fold_losses):
        assert np.isfinite(fold_l), f"Fold {i} loss is not finite: {fold_l}"

    # Overall tuning loss must be finite
    assert np.isfinite(res_masked.overall_tuning_loss)

    # Verify each configuration in grid also has 7 finite folds
    for r in all_res_masked:
        assert len(r.annual_fold_losses) == expected_folds
        assert all(np.isfinite(fl) for fl in r.annual_fold_losses)
        assert np.isfinite(r.overall_tuning_loss)

    # 2. Tuning on the same series without the empty years
    res_clean, model_clean, all_res_clean = tune_level3_temperature_hyperparameters(
        obs_tmax=data["clean_obs_tmax"],
        obs_tmin=data["clean_obs_tmin"],
        mod_tmax=data["clean_mod_tmax"],
        mod_tmin=data["clean_mod_tmin"],
    )

    # Winning configuration must be identical
    assert res_masked.config_id == res_clean.config_id, (
        f"Winning config mismatch: "
        f"masked={res_masked.config_id} vs clean={res_clean.config_id}"
    )

    # All 8 configurations must have identical overall CV losses within 1e-12
    assert len(all_res_masked) == 8
    assert len(all_res_clean) == 8
    for rm, rc in zip(all_res_masked, all_res_clean):
        assert rm.config_id == rc.config_id
        diff = abs(rm.overall_tuning_loss - rc.overall_tuning_loss)
        assert diff <= 1e-12, (
            f"Config {rm.config_id} CV loss mismatch: "
            f"masked={rm.overall_tuning_loss:.15f} vs "
            f"clean={rc.overall_tuning_loss:.15f} (diff={diff:.2e})"
        )
        for i, (fm, fc) in enumerate(zip(rm.annual_fold_losses, rc.annual_fold_losses)):
            fold_diff = abs(fm - fc)
            assert fold_diff <= 1e-12, (
                f"Config {rm.config_id} fold {i} loss mismatch: "
                f"masked={fm:.15f} vs clean={fc:.15f} (diff={fold_diff:.2e})"
            )


def test_level1_and_level2_fit_with_intercalated_empty_years():
    """Item 9: Level 1 and 2 fit with empty observation years match clean series."""
    data = _generate_12yr_series()

    # Level 1 fit
    l1_masked = Level1MonthlyAdditiveTemperature().fit(
        data["obs_tmax_masked"],
        data["obs_tmin_masked"],
        data["mod_tmax"],
        data["mod_tmin"],
    )
    l1_clean = Level1MonthlyAdditiveTemperature().fit(
        data["clean_obs_tmax"],
        data["clean_obs_tmin"],
        data["clean_mod_tmax"],
        data["clean_mod_tmin"],
    )

    for m in range(1, 13):
        assert np.isclose(
            l1_masked.tmax_biases_[m], l1_clean.tmax_biases_[m], atol=1e-12
        )
        assert np.isclose(
            l1_masked.tmin_biases_[m], l1_clean.tmin_biases_[m], atol=1e-12
        )

    # Level 2 fit
    l2_masked = Level2MeanStateDTRScalingTemperature().fit(
        data["obs_tmax_masked"],
        data["obs_tmin_masked"],
        data["mod_tmax"],
        data["mod_tmin"],
    )
    l2_clean = Level2MeanStateDTRScalingTemperature().fit(
        data["clean_obs_tmax"],
        data["clean_obs_tmin"],
        data["clean_mod_tmax"],
        data["clean_mod_tmin"],
    )

    for m in range(1, 13):
        assert np.isclose(
            l2_masked.tmean_biases_[m], l2_clean.tmean_biases_[m], atol=1e-12
        )
        assert np.isclose(
            l2_masked.dtr_scaling_factors_[m],
            l2_clean.dtr_scaling_factors_[m],
            atol=1e-12,
        )
