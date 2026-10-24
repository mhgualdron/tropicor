"""Unit tests for Andean calibration lag audit (Protocol Appendix G.6 / H.5).

All tests use purely synthetic data (no real observations or online ERA5 files).
Verifies:
1. Known lag detected with gaps in the daily calendar index.
2. Bootstrap block never joins non-contiguous calendar dates.
3. Station filename collisions avoided by '<station_code>/<filename>' keys.
4. Missing required ERA5 file raises ValueError.
5. point_delta_r == r_k_paired - r_0_paired holds exactly.
6. Degenerate replica (<2 valid pairs or zero variance) is discarded and counted.
7. Equivalence of shift(k) vs lag_days=k in loader for k=-1 and k=+1 with trend.
8. Simultaneous series without lag produce no activated sensitivity.
9. Bootstrap reproducibility with fixed DEFAULT_BOOTSTRAP_SEED.
10. Safe Pearson correlation returns NaN for zero variance or n < 2.
11. Loader rejects observations outside calibration and validation-only ERA5.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import cast

import netCDF4
import numpy as np
import pandas as pd
import pytest

from scripts.audit_andean_calibration_lag import (
    audit_station_lag_calibration,
    audit_variable_lag_calibration,
    build_station_input_hashes,
    calendar_block_bootstrap_lag_difference,
    compute_safe_pearson,
    generate_calendar_block_resample_indices,
    run_andean_calibration_lag_audit,
    write_report_json,
)
from tropicor.core.andean_registry import DEFAULT_ANDEAN_REGISTRY_PATH
from tropicor.core.protocol import DEFAULT_BOOTSTRAP_SEED
from tropicor.downscale.temperature import load_calibration_temperature_pair

TEST_STATION_ID = "37015020"
CAL_START = 1991
CAL_END = 2009


def _create_synthetic_era5_hourly_nc(
    path: Path,
    start_date: str,
    end_date: str,
    t2m_k: float = 293.15,
    lat: float = 7.25,
    lon: float = -72.75,
    daily_trend: float = 0.0,
) -> None:
    """Helper to create a minimal synthetic ERA5 NetCDF file for loader testing."""
    hourly_dates = pd.date_range(start_date, end_date, freq="h")
    n_hours = len(hourly_dates)
    with netCDF4.Dataset(str(path), "w", format="NETCDF4") as nc:
        nc.createDimension("time", n_hours)
        nc.createDimension("latitude", 1)
        nc.createDimension("longitude", 1)
        time_var = nc.createVariable("time", "f8", ("time",))
        time_var.units = f"hours since {start_date} 00:00:00"
        lat_var = nc.createVariable("latitude", "f4", ("latitude",))
        lon_var = nc.createVariable("longitude", "f4", ("longitude",))
        t2m_var = nc.createVariable("t2m", "f4", ("time", "latitude", "longitude"))
        t2m_var.units = "K"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            time_var[:] = np.arange(n_hours)
            lat_var[:] = [lat]
            lon_var[:] = [lon]
            hours = hourly_dates.hour.values
            day_offsets = (
                hourly_dates.normalize() - hourly_dates[0].normalize()
            ).days.values
            diurnal = 5.0 * np.sin(2 * np.pi * (hours - 9) / 24.0)
            t2m_vals = t2m_k + daily_trend * day_offsets + diurnal
            t2m_var[:] = t2m_vals.astype(np.float32).reshape(n_hours, 1, 1)


def test_known_lag_with_gaps_in_calendar_index() -> None:
    """Injected lag k=1 is detected even when daily calendar index contains gaps."""
    dates = pd.date_range("2000-01-01", periods=900, freq="D")
    rng = np.random.default_rng(123)

    t = np.linspace(0, 45, len(dates))
    signal = 20.0 + 5.0 * np.sin(t) + rng.normal(0, 0.5, len(dates))
    mod_0 = pd.Series(signal, index=dates)

    # Injected lag +1 day
    mod_1 = mod_0.shift(1)
    obs = mod_1 + rng.normal(0, 0.1, len(dates))

    # Introduce periodic and random gaps (NaNs) in the continuous index
    obs.iloc[100:200] = np.nan
    obs.iloc[400:550] = np.nan
    obs.iloc[rng.choice(len(dates), size=100, replace=False)] = np.nan

    mod_by_lag = {k: mod_0.shift(k) for k in [-2, -1, 0, 1, 2]}

    res = audit_variable_lag_calibration(
        obs=obs,
        mod_by_lag=mod_by_lag,
        candidate_lags=[-1, 1],
        audit_lags=[-2, -1, 0, 1, 2],
        block_size=30,
        n_replicas=1000,
        seed=DEFAULT_BOOTSTRAP_SEED,
    )

    eval_1 = res["candidate_lags"]["1"]
    assert eval_1["point_delta_r"] > 0.0
    assert eval_1["ci_95_lower"] > 0.0
    assert eval_1["excludes_zero_in_favor_of_k"] is True

    eval_neg1 = res["candidate_lags"]["-1"]
    assert eval_neg1["excludes_zero_in_favor_of_k"] is False
    assert res["triggered_lags"] == [1]


def test_bootstrap_block_never_joins_non_contiguous_dates() -> None:
    """In calendar block bootstrap, every resampled block is internally contiguous."""
    n = 365
    block_size = 30
    rng = np.random.default_rng(42)
    idx = generate_calendar_block_resample_indices(n, block_size, rng)

    assert len(idx) == n
    n_blocks = int(np.ceil(n / block_size))
    for b in range(n_blocks):
        start = b * block_size
        end = min((b + 1) * block_size, n)
        chunk = idx[start:end]
        assert len(chunk) > 0
        assert np.all(np.diff(chunk) == 1)


def test_station_filename_collision_avoided_in_input_hashes(tmp_path: Path) -> None:
    """Distinct stations with identical filenames never collide in input_hashes."""
    st1 = "11111110"
    st2 = "22222220"
    dir1 = tmp_path / st1
    dir2 = tmp_path / st2
    dir1.mkdir()
    dir2.mkdir()

    # Create identical filenames in both folders with different content
    (dir1 / "TempMaxima_Test.csv").write_text("st1_max", encoding="utf-8")
    (dir1 / "TempMinima_Test.csv").write_text("st1_min", encoding="utf-8")
    (dir1 / "era5_2m_temperature_1990.nc").write_bytes(b"st1_nc_1990")
    (dir1 / "era5_2m_temperature_1991.nc").write_bytes(b"st1_nc_1991")

    (dir2 / "TempMaxima_Test.csv").write_text("st2_max", encoding="utf-8")
    (dir2 / "TempMinima_Test.csv").write_text("st2_min", encoding="utf-8")
    (dir2 / "era5_2m_temperature_1990.nc").write_bytes(b"st2_nc_1990")
    (dir2 / "era5_2m_temperature_1991.nc").write_bytes(b"st2_nc_1991")

    h1 = build_station_input_hashes(
        station_code=st1,
        tmax_csv=dir1 / "TempMaxima_Test.csv",
        tmin_csv=dir1 / "TempMinima_Test.csv",
        era5_dir=dir1,
        cal_block=(1990, 1990),
    )
    h2 = build_station_input_hashes(
        station_code=st2,
        tmax_csv=dir2 / "TempMaxima_Test.csv",
        tmin_csv=dir2 / "TempMinima_Test.csv",
        era5_dir=dir2,
        cal_block=(1990, 1990),
    )

    combined = {**h1, **h2}
    assert len(combined) == len(h1) + len(h2)
    assert len(combined) == 8

    for k in h1:
        assert k.startswith(f"{st1}/")
    for k in h2:
        assert k.startswith(f"{st2}/")


def test_missing_era5_file_raises_value_error(tmp_path: Path) -> None:
    """Missing required ERA5 NetCDF file in cal_start..cal_end+1 raises ValueError."""
    st_code = "37015020"
    st_dir = tmp_path / st_code
    st_dir.mkdir()

    (st_dir / "TempMaxima_Test.csv").write_text("dummy", encoding="utf-8")
    (st_dir / "TempMinima_Test.csv").write_text("dummy", encoding="utf-8")
    (st_dir / "era5_2m_temperature_1991.nc").write_bytes(b"dummy_nc")

    station_info = {
        "nombre": "BERLIN",
        "cal_block": [1991, 2009],
        "valid_years": list(range(1991, 2010)),
    }

    with pytest.raises(ValueError, match="Required ERA5 file missing"):
        audit_station_lag_calibration(
            station_code=st_code,
            station_info=station_info,
            base_dir=tmp_path,
        )


def test_point_delta_r_equals_difference_exactly() -> None:
    """point_delta_r == r_k_paired - r_0_paired holds exactly."""
    rng = np.random.default_rng(999)
    n = 600
    obs = rng.normal(15, 3, n)
    mod_0 = obs + rng.normal(0, 1, n)
    mod_k = obs + rng.normal(0, 1.5, n)

    rk_p, r0_p, delta_r, ci_l, ci_u, exc, disc = (
        calendar_block_bootstrap_lag_difference(
            obs=obs,
            mod_0=mod_0,
            mod_k=mod_k,
            block_size=30,
            n_replicas=100,
            seed=42,
        )
    )
    assert delta_r == rk_p - r0_p

    dates = pd.date_range("2000-01-01", periods=n, freq="D")
    obs_s = pd.Series(obs, index=dates)
    mod_0_s = pd.Series(mod_0, index=dates)
    mod_k_s = pd.Series(mod_k, index=dates)
    mod_by_lag = {-1: mod_k_s, 0: mod_0_s, 1: mod_k_s}

    res = audit_variable_lag_calibration(
        obs=obs_s,
        mod_by_lag=mod_by_lag,
        candidate_lags=[-1, 1],
        audit_lags=[0],
        block_size=30,
        n_replicas=100,
        seed=42,
    )
    for k in ["-1", "1"]:
        c = res["candidate_lags"][k]
        assert c["point_delta_r"] == round(c["r_k_paired"] - c["r_0_paired"], 6)


def test_degenerate_replica_discarded_and_counted() -> None:
    """Degenerate replicas (< 2 valid pairs or zero var) are discarded and counted."""
    n = 120
    obs = np.full(n, np.nan)
    mod_0 = np.full(n, np.nan)
    mod_k = np.full(n, np.nan)

    # Only 3 valid points in window 0..30
    obs[5:8] = [10.0, 12.0, 14.0]
    mod_0[5:8] = [10.1, 11.9, 14.2]
    mod_k[5:8] = [9.8, 12.2, 13.9]

    rk_p, r0_p, delta_r, ci_l, ci_u, exc, n_disc = (
        calendar_block_bootstrap_lag_difference(
            obs=obs,
            mod_0=mod_0,
            mod_k=mod_k,
            block_size=30,
            n_replicas=100,
            seed=42,
        )
    )
    assert n_disc > 0


def test_shift_k_matches_loader_lag_days(tmp_path: Path) -> None:
    """Daily trend ensures shift(-1) != shift(0) != shift(+1), and matches loader."""
    obs_dates = pd.date_range("1991-01-01", "1993-12-31", freq="D")
    df_obs = pd.DataFrame(
        {
            "CodigoEstacion": [TEST_STATION_ID] * len(obs_dates),
            "Fecha": obs_dates.strftime("%Y-%m-%d"),
            "Valor": [18.0] * len(obs_dates),
        }
    )
    tmax_path = tmp_path / "TempMaxima_Test.csv"
    tmin_path = tmp_path / "TempMinima_Test.csv"
    df_obs.to_csv(tmax_path, index=False)
    df_obs.to_csv(tmin_path, index=False)

    # NetCDF covering from prior day 1990-12-31 to 2010-01-05 with daily trend
    era5_file = tmp_path / "era5_cal.nc"
    _create_synthetic_era5_hourly_nc(
        era5_file,
        "1990-12-31",
        f"{CAL_END + 1}-01-05",
        t2m_k=290.0,
        daily_trend=0.004,
    )

    o_tx_0, o_tn_0, m_tx_0, m_tn_0 = load_calibration_temperature_pair(
        station_id=TEST_STATION_ID,
        obs_tmax_path=tmax_path,
        obs_tmin_path=tmin_path,
        era5_target=era5_file,
        window_start_utc=5,
        lag_days=0,
        min_hours=24,
    )

    o_tx_neg1, o_tn_neg1, m_tx_neg1, m_tn_neg1 = (
        load_calibration_temperature_pair(
            station_id=TEST_STATION_ID,
            obs_tmax_path=tmax_path,
            obs_tmin_path=tmin_path,
            era5_target=era5_file,
            window_start_utc=5,
            lag_days=-1,
            min_hours=24,
        )
    )

    o_tx_pos1, o_tn_pos1, m_tx_pos1, m_tn_pos1 = (
        load_calibration_temperature_pair(
            station_id=TEST_STATION_ID,
            obs_tmax_path=tmax_path,
            obs_tmin_path=tmin_path,
            era5_target=era5_file,
            window_start_utc=5,
            lag_days=1,
            min_hours=24,
        )
    )

    m_tx_shift_neg1 = m_tx_0.shift(-1)
    m_tx_shift_pos1 = m_tx_0.shift(1)
    common_all = m_tx_shift_neg1.dropna().index.intersection(
        m_tx_shift_pos1.dropna().index
    )

    # Assert shift(-1), shift(0), and shift(+1) are all distinct
    assert not np.allclose(m_tx_0.loc[common_all], m_tx_shift_neg1.loc[common_all])
    assert not np.allclose(m_tx_0.loc[common_all], m_tx_shift_pos1.loc[common_all])
    assert not np.allclose(
        m_tx_shift_neg1.loc[common_all], m_tx_shift_pos1.loc[common_all]
    )

    # Verify lag_days=-1 == shift(-1) and differs from shift(+1)
    common_neg1 = m_tx_shift_neg1.dropna().index.intersection(
        m_tx_neg1.dropna().index
    )
    np.testing.assert_allclose(
        m_tx_shift_neg1.loc[common_neg1].to_numpy(),
        m_tx_neg1.loc[common_neg1].to_numpy(),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        m_tn_0.shift(-1).loc[common_neg1].to_numpy(),
        m_tn_neg1.loc[common_neg1].to_numpy(),
        atol=1e-5,
    )
    assert not np.allclose(
        m_tx_neg1.loc[common_all].to_numpy(),
        m_tx_shift_pos1.loc[common_all].to_numpy(),
    )

    # Verify lag_days=+1 == shift(+1) on common valid days
    common_pos1 = m_tx_shift_pos1.dropna().index.intersection(
        m_tx_pos1.dropna().index
    )
    np.testing.assert_allclose(
        m_tx_shift_pos1.loc[common_pos1].to_numpy(),
        m_tx_pos1.loc[common_pos1].to_numpy(),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        m_tn_0.shift(1).loc[common_pos1].to_numpy(),
        m_tn_pos1.loc[common_pos1].to_numpy(),
        atol=1e-5,
    )


def test_simultaneous_series_no_lag() -> None:
    """Simultaneous series without lag produces Delta r <= 0 and no sensitivity."""
    dates = pd.date_range("2000-01-01", periods=600, freq="D")
    rng = np.random.default_rng(456)

    t = np.linspace(0, 30, len(dates))
    signal = 20.0 + 5.0 * np.sin(t) + rng.normal(0, 0.5, len(dates))
    mod_0 = pd.Series(signal, index=dates)
    obs = mod_0 + rng.normal(0, 0.2, len(dates))

    mod_by_lag = {k: mod_0.shift(k) for k in [-2, -1, 0, 1, 2]}

    res = audit_variable_lag_calibration(
        obs=obs,
        mod_by_lag=mod_by_lag,
        candidate_lags=[-1, 1],
        audit_lags=[-2, -1, 0, 1, 2],
        block_size=30,
        n_replicas=1000,
        seed=DEFAULT_BOOTSTRAP_SEED,
    )

    for k in ["-1", "1"]:
        eval_k = res["candidate_lags"][k]
        assert eval_k["point_delta_r"] < 0.0
        assert eval_k["ci_95_lower"] < 0.0
        assert eval_k["excludes_zero_in_favor_of_k"] is False

    assert res["triggered_lags"] == []
    assert res["extrema_timing"]["optimal_lag"] == 0


def test_bootstrap_reproducibility() -> None:
    """Two calls with fixed DEFAULT_BOOTSTRAP_SEED produce identical CIs."""
    rng = np.random.default_rng(789)
    n = 400
    obs = rng.normal(20, 2, n)
    mod_0 = obs + rng.normal(0, 0.5, n)
    mod_k = np.roll(mod_0, 1)

    res1 = calendar_block_bootstrap_lag_difference(
        obs=obs,
        mod_0=mod_0,
        mod_k=mod_k,
        block_size=30,
        n_replicas=500,
        seed=DEFAULT_BOOTSTRAP_SEED,
    )

    res2 = calendar_block_bootstrap_lag_difference(
        obs=obs,
        mod_0=mod_0,
        mod_k=mod_k,
        block_size=30,
        n_replicas=500,
        seed=DEFAULT_BOOTSTRAP_SEED,
    )

    assert res1 == res2


def test_safe_pearson_constant_and_small_sample() -> None:
    """Safe Pearson correlation returns NaN for zero variance or n < 2."""
    c = np.full(50, 15.0)
    v = np.linspace(10, 20, 50)
    assert np.isnan(compute_safe_pearson(c, v))
    assert np.isnan(compute_safe_pearson(v, c))
    assert np.isnan(compute_safe_pearson(c, c))
    assert np.isnan(compute_safe_pearson(np.array([1.0]), np.array([2.0])))


def test_load_calibration_temperature_pair_rejects_validation(
    tmp_path: Path,
) -> None:
    """Observations outside calibration are never returned, and val ERA5 raises."""
    obs_dates = pd.date_range("1991-01-01", "2020-12-31", freq="D")
    df_obs = pd.DataFrame(
        {
            "CodigoEstacion": [TEST_STATION_ID] * len(obs_dates),
            "Fecha": obs_dates.strftime("%Y-%m-%d"),
            "Valor": [18.0] * len(obs_dates),
        }
    )
    tmax_path = tmp_path / "tmax.csv"
    tmin_path = tmp_path / "tmin.csv"
    df_obs.to_csv(tmax_path, index=False)
    df_obs.to_csv(tmin_path, index=False)

    era5_cal_file = tmp_path / "era5_cal.nc"
    _create_synthetic_era5_hourly_nc(
        era5_cal_file,
        f"{CAL_START}-01-01",
        f"{CAL_END + 1}-01-05",
        t2m_k=291.15,
    )

    o_tmax, o_tmin, m_tmax, m_tmin = load_calibration_temperature_pair(
        station_id=TEST_STATION_ID,
        obs_tmax_path=tmax_path,
        obs_tmin_path=tmin_path,
        era5_target=era5_cal_file,
        window_start_utc=5,
        lag_days=0,
        min_hours=24,
    )

    assert o_tmax.index.max() == pd.Timestamp(f"{CAL_END}-12-31")
    assert o_tmin.index.max() == pd.Timestamp(f"{CAL_END}-12-31")
    assert o_tmax.index.min() == pd.Timestamp(f"{CAL_START}-01-01")
    dt_idx = cast(pd.DatetimeIndex, o_tmax.index)
    assert (dt_idx.year > CAL_END).sum() == 0

    val_era_dir = tmp_path / "era5_val_only"
    val_era_dir.mkdir()
    val_era_file = val_era_dir / "era5_2m_temperature_2015.nc"
    _create_synthetic_era5_hourly_nc(
        val_era_file,
        "2015-01-01",
        "2015-12-31 23:00:00",
        t2m_k=291.15,
    )

    with pytest.raises(
        ValueError, match="No ERA5 NetCDF files matching calibration years"
    ):
        load_calibration_temperature_pair(
            station_id=TEST_STATION_ID,
            obs_tmax_path=tmax_path,
            obs_tmin_path=tmin_path,
            era5_target=val_era_dir,
            window_start_utc=5,
            lag_days=0,
            min_hours=24,
        )


def test_calendar_block_bootstrap_insufficient_pairs_raises_value_error() -> None:
    """Less than 2 valid paired days raises explicit ValueError."""
    obs = np.array([10.0, np.nan, np.nan])
    mod_0 = np.array([10.0, np.nan, np.nan])
    mod_k = np.array([10.0, np.nan, np.nan])
    with pytest.raises(ValueError, match="Insufficient valid paired days"):
        calendar_block_bootstrap_lag_difference(
            obs=obs,
            mod_0=mod_0,
            mod_k=mod_k,
            block_size=30,
            n_replicas=10,
        )


def test_calendar_block_bootstrap_point_nan_zero_variance_raises_value_error() -> None:
    """Zero variance in observations produces NaN point estimate, raising ValueError."""
    n = 40
    obs = np.full(n, 20.0)
    mod_0 = np.linspace(15.0, 25.0, n)
    mod_k = np.linspace(14.0, 24.0, n)
    with pytest.raises(
        ValueError, match="Point correlation estimate produced NaN"
    ):
        calendar_block_bootstrap_lag_difference(
            obs=obs,
            mod_0=mod_0,
            mod_k=mod_k,
            block_size=30,
            n_replicas=10,
        )


def test_calendar_block_bootstrap_shorter_than_block_size_raises_value_error() -> None:
    """Continuous series length n < block_size raises explicit ValueError."""
    n = 20
    obs = np.linspace(10.0, 20.0, n)
    mod_0 = np.linspace(10.0, 20.0, n)
    mod_k = np.linspace(10.0, 20.0, n)
    with pytest.raises(ValueError, match="less than block_size"):
        calendar_block_bootstrap_lag_difference(
            obs=obs,
            mod_0=mod_0,
            mod_k=mod_k,
            block_size=30,
            n_replicas=10,
        )


def test_calendar_block_bootstrap_all_replicas_discarded_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Monkeypatched resample containing only 1 valid pair discards all replicas."""
    mock_indices = np.concatenate([np.arange(0, 30), np.arange(1, 11)])
    monkeypatch.setattr(
        "scripts.audit_andean_calibration_lag.generate_calendar_block_resample_indices",
        lambda n, block_size, rng: mock_indices,
    )
    n = 40
    obs = np.full(n, np.nan)
    mod_0 = np.full(n, np.nan)
    mod_k = np.full(n, np.nan)
    obs[0] = 10.0
    mod_0[0] = 10.0
    mod_k[0] = 10.0
    obs[39] = 20.0
    mod_0[39] = 20.0
    mod_k[39] = 20.0
    with pytest.raises(ValueError, match="All bootstrap replicas were discarded"):
        calendar_block_bootstrap_lag_difference(
            obs=obs,
            mod_0=mod_0,
            mod_k=mod_k,
            block_size=30,
            n_replicas=50,
        )


def test_write_report_json_nan_raises_value_error_and_file_not_created(
    tmp_path: Path,
) -> None:
    """write_report_json rejects NaN with ValueError and does not create output file."""
    out_file = tmp_path / "test_report.json"
    payload = {"station": "37015020", "invalid_value": float("nan")}
    with pytest.raises(ValueError):
        write_report_json(payload, out_file)
    assert not out_file.exists()


def test_degenerate_replica_zero_variance_counted_in_discarded() -> None:
    """Replicas sampling zero-variance sub-intervals are discarded and counted."""
    n = 90
    obs = np.empty(n, dtype=float)
    mod_0 = np.empty(n, dtype=float)
    mod_k = np.empty(n, dtype=float)
    obs[:80] = 15.0
    mod_0[:80] = 15.0
    mod_k[:80] = 15.0
    obs[80:] = np.linspace(10.0, 20.0, 10)
    mod_0[80:] = np.linspace(10.0, 20.0, 10)
    mod_k[80:] = np.linspace(10.0, 20.0, 10)
    _, _, _, _, _, _, n_disc = calendar_block_bootstrap_lag_difference(
        obs=obs,
        mod_0=mod_0,
        mod_k=mod_k,
        block_size=30,
        n_replicas=100,
        seed=42,
    )
    assert n_disc > 0


def test_end_to_end_andean_calibration_lag_audit_synthetic_37015020(
    tmp_path: Path,
) -> None:
    """End-to-end audit for 37015020 masks invalid years and writes LF-only JSON."""
    st_dir = tmp_path / TEST_STATION_ID
    st_dir.mkdir(parents=True)

    dates = pd.date_range("1991-01-01", "2010-12-31", freq="D")
    df_obs = pd.DataFrame(
        {
            "CodigoEstacion": [TEST_STATION_ID] * len(dates),
            "Fecha": dates.strftime("%Y-%m-%d"),
            "Valor": (
                15.0 + 3.0 * np.sin(np.linspace(0, 50, len(dates)))
            ).round(2),
        }
    )
    (st_dir / "TempMaxima_Berlin.csv").write_text(
        df_obs.to_csv(index=False), encoding="utf-8"
    )
    (st_dir / "TempMinima_Berlin.csv").write_text(
        df_obs.assign(Valor=df_obs["Valor"] - 5.0).to_csv(index=False),
        encoding="utf-8",
    )

    for y in range(1991, 2011):
        nc_p = st_dir / f"era5_2m_temperature_{y}.nc"
        h_dates = pd.date_range(f"{y}-01-01", f"{y}-12-31 23:00:00", freq="h")
        n_h = len(h_dates)
        with netCDF4.Dataset(str(nc_p), "w", format="NETCDF4") as nc:
            nc.createDimension("time", n_h)
            nc.createDimension("latitude", 1)
            nc.createDimension("longitude", 1)
            tv = nc.createVariable("time", "f8", ("time",))
            tv.units = f"hours since {y}-01-01 00:00:00"
            latv = nc.createVariable("latitude", "f4", ("latitude",))
            lonv = nc.createVariable("longitude", "f4", ("longitude",))
            t2m = nc.createVariable("t2m", "f4", ("time", "latitude", "longitude"))
            t2m.units = "K"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                tv[:] = np.arange(n_h)
                latv[:] = [7.25]
                lonv[:] = [-72.75]
                hours = h_dates.hour.values
                day_off = (
                    h_dates.normalize() - h_dates[0].normalize()
                ).days.values
                t2m[:] = (
                    290.0
                    + 0.004 * day_off
                    + 5.0 * np.sin(2 * np.pi * (hours - 9) / 24.0)
                ).astype(np.float32).reshape(n_h, 1, 1)

    with open(DEFAULT_ANDEAN_REGISTRY_PATH, encoding="utf-8") as f:
        reg_full = json.load(f)
    reg_single = {
        "version": reg_full["version"],
        "stations": {TEST_STATION_ID: reg_full["stations"][TEST_STATION_ID]},
    }
    reg_path = tmp_path / "single_registry.json"
    reg_path.write_text(json.dumps(reg_single, indent=2), encoding="utf-8")

    out_json = tmp_path / "audit_output.json"
    rep = run_andean_calibration_lag_audit(
        registry_path=reg_path,
        base_dir=tmp_path,
        output_path=out_json,
        n_replicas=100,
    )

    st_res = rep["stations"][TEST_STATION_ID]
    assert st_res["n_cal_years"] == 11
    # Total calendar days in 1991-2009 is 6940;
    # invalid years (1998-2004, 2006) are masked
    tmax_eval_1 = st_res["tmax"]["candidate_lags"]["1"]
    assert tmax_eval_1["n_paired_days"] == 4017
    assert tmax_eval_1["n_paired_days"] < 6940

    st_hashes = {k: v for k, v in rep["input_hashes"].items() if k != "registry"}
    assert len(st_hashes) == 22
    for k in st_hashes:
        assert k.startswith(f"{TEST_STATION_ID}/")

    raw_bytes = out_json.read_bytes()
    assert b"\r\n" not in raw_bytes
    text = raw_bytes.decode("utf-8")
    assert "NaN" not in text
    assert "nan" not in text
    parsed = json.loads(text)
    assert parsed["config"]["protocol_section"] == "Appendix G.6 / H.5"
    for k in ["python_version", "numpy_version", "pandas_version"]:
        assert k in parsed["config"]
        assert isinstance(parsed["config"][k], str)
        assert len(parsed["config"][k].strip()) > 0
