"""Unit tests for audit_station_data.py with synthetic hand-calculated series."""

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_station_data import (
    audit_completeness,
    audit_file_structure,
    audit_physical_rules,
    audit_sentinels_and_flags,
    audit_simultaneous_temperature,
    audit_temperature_inversions,
    audit_triplet_overlap,
    compute_haversine_distance_km,
    detect_streaks,
    group_contiguous_dates,
    inspect_quality_flags_only,
)

# ==============================================================================
# Control 1: Estructura, Fechas, Duplicados, Orden y Rango Esperado
# ==============================================================================


def test_control_1_file_structure_and_calendar(tmp_path: Path) -> None:
    """Verify Control 1 with synthetic hand-crafted records."""
    # Synthetic dataset with known anomalies:
    # 1. One unparseable date: 'fecha_invalida'
    # 2. One duplicate timestamp: '1990-01-02' appears twice
    # 3. Non-monotonic ordering: '1990-01-01' appears after '1990-01-02'
    # 4. Expected range: 1990-01-01 to 1990-01-05 (5 calendar days)
    #    Present in range: 1990-01-01, 1990-01-02 (2 calendar days)
    #    Missing calendar days: 1990-01-03, 1990-01-04, 1990-01-05 (3 days)
    csv_file = tmp_path / "synthetic_station.csv"
    content = (
        "CodigoEstacion,NombreEstacion,Fecha,Valor,NivelAprobacion\n"
        "99999999,ESTACION SINTETICA,1990-01-01 0:00,10.5,Preliminar\n"
        "99999999,ESTACION SINTETICA,1990-01-02 0:00,20.0,Preliminar\n"
        "99999999,ESTACION SINTETICA,1990-01-02 0:00,20.0,Definitivo\n"
        "99999999,ESTACION SINTETICA,fecha_invalida,15.0,Preliminar\n"
        "99999999,ESTACION SINTETICA,1990-01-01 0:00,10.5,Preliminar\n"
    )
    csv_file.write_text(content, encoding="utf-8")

    res, _ = audit_file_structure(
        csv_file,
        expected_station_id="99999999",
        expected_start="1990-01-01",
        expected_end="1990-01-05",
    )

    # 1. Station code and name
    assert res.station_codes == ["99999999"]
    assert res.station_names == ["ESTACION SINTETICA"]
    assert res.station_code_matches_expected is True
    assert res.total_raw_rows == 5

    # 2. Date parseability
    assert res.parseable_dates_count == 4
    assert res.unparseable_dates_count == 1
    assert "fecha_invalida" in res.unparseable_samples[0]

    # 3. Duplicates: 1990-01-01 and 1990-01-02 both appear twice
    assert res.duplicate_dates_count == 4

    # 4. Monotonic ordering: violated by 1990-01-01 after 1990-01-02
    assert res.is_monotonic_increasing is False
    assert res.date_inversion_count >= 1

    # 5. Expected calendar range: 5 expected days (Jan 1..5)
    assert res.expected_calendar_days == 5
    assert res.present_calendar_days_in_range == 2
    assert res.missing_calendar_days_count == 3
    assert len(res.missing_calendar_ranges) == 1
    assert res.missing_calendar_ranges[0] == {
        "start": "1990-01-03",
        "end": "1990-01-05",
        "days": 3,
    }

    # 6. SHA-256 calculation exists and is 64 hex characters
    assert len(res.file_sha256) == 64


def test_group_contiguous_dates_empty() -> None:
    """Verify date grouping on empty list."""
    assert group_contiguous_dates([]) == []


def test_group_contiguous_dates_isolated_and_multi_day() -> None:
    """Verify date grouping with isolated days and multi-day spans."""
    dates = [
        pd.Timestamp("1995-01-01"),
        pd.Timestamp("1995-01-02"),
        pd.Timestamp("1995-01-05"),
        pd.Timestamp("1995-01-10"),
        pd.Timestamp("1995-01-11"),
        pd.Timestamp("1995-01-12"),
    ]
    ranges = group_contiguous_dates(dates)
    assert len(ranges) == 3
    assert ranges[0] == {"start": "1995-01-01", "end": "1995-01-02", "days": 2}
    assert ranges[1] == {"start": "1995-01-05", "end": "1995-01-05", "days": 1}
    assert ranges[2] == {"start": "1995-01-10", "end": "1995-01-12", "days": 3}


# ==============================================================================
# Control 2: Valores no numéricos, centinelas y banderas de calidad
# ==============================================================================


def test_control_2_sentinels_and_quality_flags() -> None:
    """Verify Control 2 detection of sentinels and quality flags."""
    # Hand-crafted synthetic DataFrame
    # Sentinels: -999 (2 times), 9999 (1 time), S/D (1 time), -9999 (1 time)
    # Non-numeric unhandled: 'CORRUPTO' (2 times)
    # Quality flags: 'Definitivo' (3), 'Preliminar' (2), 'Faltante' (2)
    df = pd.DataFrame(
        {
            "Valor": [
                "12.5",
                "-999",
                "9999",
                "S/D",
                "-9999",
                "-999",
                "CORRUPTO",
                "CORRUPTO",
            ],
            "NivelAprobacion": [
                "Definitivo",
                "Definitivo",
                "Preliminar",
                "Faltante",
                "Definitivo",
                "Preliminar",
                "Faltante",
                "Desconocido",
            ],
        }
    )

    res = audit_sentinels_and_flags(
        df_raw=df,
        variable_label="Precipitación",
        filepath="synthetic.csv",
    )

    assert res.sentinel_counts["-999"] == 2
    assert res.sentinel_counts["9999"] == 1
    assert res.sentinel_counts["S/D"] == 1
    assert res.sentinel_counts["-9999"] == 1

    assert res.non_numeric_counts == {"CORRUPTO": 2}

    assert res.quality_flag_counts["Definitivo"] == 3
    assert res.quality_flag_counts["Preliminar"] == 2
    assert res.quality_flag_counts["Faltante"] == 2
    assert res.quality_flag_counts["Desconocido"] == 1


def test_inspect_quality_flags_only(tmp_path: Path) -> None:
    """Verify reading ONLY the NivelAprobacion column."""
    csv_file = tmp_path / "test_flags.csv"
    csv_file.write_text(
        "Codigo,Fecha,Valor,NivelAprobacion\n"
        "1,1990-01-01,10,Definitivo\n"
        "1,1990-01-02,20,Definitivo\n"
        "1,1990-01-03,30,Preliminar\n",
        encoding="utf-8",
    )
    res = inspect_quality_flags_only(csv_file)
    assert res == {"Definitivo": 2, "Preliminar": 1}


# ==============================================================================
# Control 3: Reglas Físicas, Rachas y Resolución Observada
# ==============================================================================


def test_control_3_precipitation_physical_rules_and_streaks() -> None:
    """Verify Control 3 physical rules on precipitation series."""
    dates = pd.date_range("1990-01-01", periods=16, freq="D")
    # Values:
    # Day 0 (1990-01-01): 10.0 (valid)
    # Day 1 (1990-01-02): -2.5 (anomalous < 0)
    # Day 2 (1990-01-03): 550.0 (anomalous > 500)
    # Days 3..7 (5 days): [14.2, 14.2, 14.2, 14.2, 14.2] (streak >= 5 days, P != 0)
    # Days 8..13 (6 days): [0.0]*6 (zero streak, MUST NOT be flagged for precip)
    # Days 14..15 (2 days): [8.0, 8.0] (< 5 days, not a streak)
    raw_vals = [
        "10",
        "-2.5",
        "550.0",
        "14.2",
        "14.2",
        "14.2",
        "14.2",
        "14.2",
        "0.0",
        "0.0",
        "0.0",
        "0.0",
        "0.0",
        "0.0",
        "8.0",
        "8.0",
    ]
    numeric_vals = [float(v) for v in raw_vals]

    series = pd.Series(numeric_vals, index=dates)
    raw_strings = pd.Series(raw_vals, index=dates)

    res = audit_physical_rules(
        series=series,
        raw_strings=raw_strings,
        variable_label="Precipitación (mm)",
        lower_bound=0.0,
        upper_bound=500.0,
        min_streak_days=5,
        ignore_zero_streaks=True,
    )

    # 1. Bounds violations
    assert res.below_min_count == 1
    assert res.below_min_records == [{"date": "1990-01-02", "value": -2.5}]

    assert res.above_max_count == 1
    assert res.above_max_records == [{"date": "1990-01-03", "value": 550.0}]

    # 2. Streaks: exactly 1 non-zero streak (14.2 mm from 1990-01-04 to 1990-01-08)
    assert res.streaks_count == 1
    stk = res.streaks[0]
    assert stk["start_date"] == "1990-01-04"
    assert stk["end_date"] == "1990-01-08"
    assert stk["length"] == 5
    assert stk["value"] == 14.2

    # 3. Decimal resolution:
    # "10" has 0 decimals (1)
    # All other 15 have 1 decimal (15)
    assert res.decimal_places_counts == {0: 1, 1: 15}


def test_control_3_temperature_bounds_and_inversions() -> None:
    """Verify Control 3 temperature physical bounds and inversions (Tmax < Tmin)."""
    dates = pd.date_range("1990-01-01", periods=6, freq="D")
    # Day 0: Tmax = 4.0 (< 5 °C -> flagged), Tmin = 1.0
    # Day 1: Tmax = 46.0 (> 45 °C -> flagged), Tmin = 20.0
    # Day 2: Tmax = 25.0, Tmin = -6.0 (< -5 °C -> flagged)
    # Day 3: Tmax = 30.0, Tmin = 36.0 (Tmin > 35 °C AND inversion Tmax < Tmin)
    # Day 4: Tmax = 22.0, Tmin = 22.0 (equal, not inversion)
    # Day 5: Tmax = 28.0, Tmin = 18.0 (valid)
    tmax = pd.Series([4.0, 46.0, 25.0, 30.0, 22.0, 28.0], index=dates)
    tmin = pd.Series([1.0, 20.0, -6.0, 36.0, 22.0, 18.0], index=dates)

    res_tmax = audit_physical_rules(
        series=tmax,
        raw_strings=tmax.astype(str),
        variable_label="Temperatura Máxima (°C)",
        lower_bound=5.0,
        upper_bound=45.0,
        min_streak_days=10,
        ignore_zero_streaks=False,
    )
    assert res_tmax.below_min_count == 1
    assert res_tmax.below_min_records == [{"date": "1990-01-01", "value": 4.0}]
    assert res_tmax.above_max_count == 1
    assert res_tmax.above_max_records == [{"date": "1990-01-02", "value": 46.0}]

    res_tmin = audit_physical_rules(
        series=tmin,
        raw_strings=tmin.astype(str),
        variable_label="Temperatura Mínima (°C)",
        lower_bound=-5.0,
        upper_bound=35.0,
        min_streak_days=10,
        ignore_zero_streaks=False,
    )
    assert res_tmin.below_min_count == 1
    assert res_tmin.below_min_records == [{"date": "1990-01-03", "value": -6.0}]
    assert res_tmin.above_max_count == 1
    assert res_tmin.above_max_records == [{"date": "1990-01-04", "value": 36.0}]

    # Inversion check
    res_inv = audit_temperature_inversions(tmax, tmin)
    assert res_inv.inversion_count == 1
    assert res_inv.inversion_records == [
        {"date": "1990-01-04", "tmax": 30.0, "tmin": 36.0}
    ]


def test_detect_streaks_temperature_10_days() -> None:
    """Verify temperature streak detection with threshold >= 10 days."""
    dates = pd.date_range("1990-01-01", periods=12, freq="D")
    # 10 days of constant 24.5 °C followed by 2 days of 25.0 °C
    vals = np.array([24.5] * 10 + [25.0, 25.0])
    streaks = detect_streaks(dates, vals, min_length=10, ignore_zero=False)
    assert len(streaks) == 1
    assert streaks[0]["start_date"] == "1990-01-01"
    assert streaks[0]["end_date"] == "1990-01-10"
    assert streaks[0]["length"] == 10
    assert streaks[0]["value"] == 24.5


# ==============================================================================
# Control 4: Completitud Anual y Mensual, > 20% Faltante, Años >= 330 Días
# ==============================================================================


def test_control_4_completeness_annual_and_monthly() -> None:
    """Verify Control 4 completeness calculations with synthetic counts."""
    # 2 years: 1990 (365 days) and 1991 (365 days)
    # Total expected days = 730
    full_dates = pd.date_range("1990-01-01", "1991-12-31", freq="D")
    values = pd.Series(np.ones(len(full_dates)), index=full_dates)

    # In 1990: artificially set 25 days in June to NaN
    # June has 30 days: 30 - 25 = 5 valid days -> 25/30 = 83.33% missing (> 20%)
    # Total valid days in 1990 = 365 - 25 = 340 days (340 >= 330 -> ge_330 is True)
    june_1990_missing = pd.date_range("1990-06-01", "1990-06-25", freq="D")
    values.loc[june_1990_missing] = np.nan

    # In 1991: set 65 days in Jan-Feb-March to NaN
    # Total valid days in 1991 = 365 - 65 = 300 days (300 < 330 -> ge_330 is False)
    spring_1991_missing = pd.date_range("1991-01-01", "1991-03-06", freq="D")
    values.loc[spring_1991_missing] = np.nan

    res = audit_completeness(
        series=values,
        variable_label="Precipitación",
        expected_start="1990-01-01",
        expected_end="1991-12-31",
    )

    assert res.total_expected_days == 730
    assert res.total_valid_days == 730 - 25 - 65
    assert res.total_missing_days == 25 + 65

    # Annual metrics
    ann_1990 = res.annual_summary[0]
    assert ann_1990["year"] == 1990
    assert ann_1990["total_days"] == 365
    assert ann_1990["valid_days"] == 340
    assert ann_1990["missing_days"] == 25
    assert ann_1990["ge_330"] is True

    ann_1991 = res.annual_summary[1]
    assert ann_1991["year"] == 1991
    assert ann_1991["total_days"] == 365
    assert ann_1991["valid_days"] == 300
    assert ann_1991["missing_days"] == 65
    assert ann_1991["ge_330"] is False

    assert res.years_ge_330_count == 1
    assert res.years_ge_330_list == [1990]

    # Monthly metrics: 1990-06 must be in months_gt_20pct_missing_list
    gt_20_months = [m["year_month"] for m in res.months_gt_20pct_missing_list]
    assert "1990-06" in gt_20_months
    june_entry = [
        m for m in res.months_gt_20pct_missing_list if m["year_month"] == "1990-06"
    ][0]
    assert june_entry["total_days"] == 30
    assert june_entry["valid_days"] == 5
    assert june_entry["missing_days"] == 25
    assert np.isclose(june_entry["pct_missing"], 83.33, atol=0.01)


def test_control_4_simultaneous_temperature() -> None:
    """Verify simultaneous temperature completeness (Tmax and Tmin valid)."""
    dates = pd.date_range("1990-01-01", "1991-12-31", freq="D")
    tmax = pd.Series(20.0, index=dates)
    tmin = pd.Series(10.0, index=dates)

    # In 1990: drop 10 days from tmax and 20 different days from tmin
    # simultaneous valid = 365 - 30 = 335 days (335 >= 330 -> True)
    tmax.loc["1990-01-01":"1990-01-10"] = np.nan
    tmin.loc["1990-02-01":"1990-02-20"] = np.nan

    # In 1991: drop 50 days from tmax
    # simultaneous valid = 365 - 50 = 315 days (< 330 -> False)
    tmax.loc["1991-01-01":"1991-02-19"] = np.nan

    res = audit_simultaneous_temperature(
        tmax, tmin, expected_start="1990-01-01", expected_end="1991-12-31"
    )

    assert res.annual_simultaneous_valid[0]["year"] == 1990
    assert res.annual_simultaneous_valid[0]["simultaneous_valid_days"] == 335
    assert res.annual_simultaneous_valid[0]["ge_330"] is True

    assert res.annual_simultaneous_valid[1]["year"] == 1991
    assert res.annual_simultaneous_valid[1]["simultaneous_valid_days"] == 315
    assert res.annual_simultaneous_valid[1]["ge_330"] is False

    assert res.years_simultaneous_ge_330_count == 1
    assert res.years_simultaneous_ge_330_list == [1990]


def test_control_4_triplet_overlap() -> None:
    """Verify 3-variable simultaneous overlap (PR + Tmax + Tmin)."""
    dates = pd.date_range("1990-01-01", "1991-12-31", freq="D")
    pr = pd.Series(5.0, index=dates)
    tmax = pd.Series(28.0, index=dates)
    tmin = pd.Series(18.0, index=dates)

    # In 1990: drop 15 days from tmax and 15 different days from pr -> 335 simultaneous
    pr.loc["1990-01-01":"1990-01-15"] = np.nan
    tmax.loc["1990-02-01":"1990-02-15"] = np.nan

    # In 1991: drop 100 days from tmin -> 265 simultaneous
    tmin.loc["1991-01-01":"1991-04-10"] = np.nan

    res = audit_triplet_overlap(
        pr_series=pr,
        tmax_series=tmax,
        tmin_series=tmin,
        start_date="1990-01-01",
        end_date="1991-12-31",
    )

    assert res.annual_triplet_summary[0]["year"] == 1990
    assert res.annual_triplet_summary[0]["triplet_valid_days"] == 335
    assert res.annual_triplet_summary[0]["ge_330"] is True

    assert res.annual_triplet_summary[1]["year"] == 1991
    assert res.annual_triplet_summary[1]["triplet_valid_days"] == 265
    assert res.annual_triplet_summary[1]["ge_330"] is False

    assert res.years_triplet_ge_330_count == 1
    assert res.years_triplet_ge_330_list == [1990]
    assert res.years_triplet_any_count == 2
    assert res.years_triplet_any_list == [1990, 1991]


def test_haversine_distance() -> None:
    """Verify Haversine distance calculation against known hand-calculated points."""
    # 1 degree of latitude along meridian:
    # d = 6371.0088 * (pi / 180) = 111.1949 km
    d_lat = compute_haversine_distance_km(0.0, 0.0, 1.0, 0.0)
    assert np.isclose(d_lat, 111.1949, atol=0.01)

    # Same point distance is 0.0
    d_zero = compute_haversine_distance_km(4.5, -76.5, 4.5, -76.5)
    assert np.isclose(d_zero, 0.0, atol=1e-5)
