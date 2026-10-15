"""Contract tests for IDEAM DHIME daily precipitation adapter (Fase 3).

Uses small, purely synthetic fixtures (never real files) to validate:
1. CF-1.8 convention compliance (pr variable, mm/day units, (time, station) dims).
2. Complete calendar reconstruction with NaNs (never 0 mm) on missing days.
3. Row deduplication with quality priority (Definitivo > Preliminar).
4. Preservation of NivelAprobacion as quality_flag.
5. Repair of Excel Spanish-locale datetime auto-formatting (d.m pattern).
6. Coordinate attribution from CNE or explicit overrides.
7. 07:00 to 07:00 local time observation window metadata.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tropicor.io.ideam import (
    DhimeDailyPrecipitationAdapter,
    read_dhime_daily_precipitation,
)


@pytest.fixture
def synthetic_dhime_df() -> pd.DataFrame:
    """Generate small synthetic DHIME daily precipitation DataFrame with edge cases."""
    # Dates: 2020-01-01 to 2020-01-06 (with 2020-01-04 intentionally missing)
    # Includes:
    # - 0 mm integer
    # - Excel d.m datetime (e.g. 3.4 mm turned into 3-Apr datetime)
    # - String decimal '12.5'
    # - Duplicate on 2020-01-03 (one Preliminar, one Definitivo)
    # - Missing day 2020-01-04 (must become NaN)
    records = [
        {
            "CodigoEstacion": 21205710,
            "NombreEstacion": "JARDIN BOTANICO - AUT",
            "Variable": "PRECIPITACION",
            "Parametro": "Día pluviométrico (convencional)",
            "Fecha": "2020-01-01",
            "Unidad": "mm",
            "Valor": 0,
            "NivelAprobacion": "Definitivo",
        },
        {
            "CodigoEstacion": 21205710,
            "NombreEstacion": "JARDIN BOTANICO - AUT",
            "Variable": "PRECIPITACION",
            "Parametro": "Día pluviométrico (convencional)",
            "Fecha": "2020-01-02",
            "Unidad": "mm",
            # Excel d.m format: 3.4 mm coerced to 3-Apr
            "Valor": datetime(2026, 4, 3, 0, 0),
            "NivelAprobacion": "Definitivo",
        },
        # Duplicate pair on 2020-01-03:
        {
            "CodigoEstacion": 21205710,
            "NombreEstacion": "JARDIN BOTANICO - AUT",
            "Variable": "PRECIPITACION",
            "Parametro": "Día pluviométrico (convencional)",
            "Fecha": "2020-01-03",
            "Unidad": "mm",
            "Valor": "10.0",
            "NivelAprobacion": "Preliminar",
        },
        {
            "CodigoEstacion": 21205710,
            "NombreEstacion": "JARDIN BOTANICO - AUT",
            "Variable": "PRECIPITACION",
            "Parametro": "Día pluviométrico (convencional)",
            "Fecha": "2020-01-03",
            "Unidad": "mm",
            "Valor": "12.5",
            "NivelAprobacion": "Definitivo",
        },
        # 2020-01-04 is MISSING
        {
            "CodigoEstacion": 21205710,
            "NombreEstacion": "JARDIN BOTANICO - AUT",
            "Variable": "PRECIPITACION",
            "Parametro": "Día pluviométrico (convencional)",
            "Fecha": "2020-01-05",
            "Unidad": "mm",
            # Excel d.m format: 21.8 mm coerced to 21-Aug
            "Valor": datetime(2026, 8, 21, 0, 0),
            "NivelAprobacion": "Definitivo",
        },
        {
            "CodigoEstacion": 21205710,
            "NombreEstacion": "JARDIN BOTANICO - AUT",
            "Variable": "PRECIPITACION",
            "Parametro": "Día pluviométrico (convencional)",
            "Fecha": "2020-01-06",
            "Unidad": "mm",
            "Valor": "0.4",
            "NivelAprobacion": "Preliminar",
        },
    ]
    return pd.DataFrame(records)


def test_calendar_reconstruction_missing_days_are_nan(
    synthetic_dhime_df: pd.DataFrame,
) -> None:
    """Missing calendar dates must be represented as NaN, never 0.0 mm."""
    adapter = DhimeDailyPrecipitationAdapter(
        synthetic_dhime_df, lat=4.668, lon=-74.101, elevation=2552.0
    )
    ds = adapter.to_xarray()

    # Total days: 2020-01-01 to 2020-01-06 = 6 days
    assert len(ds.time) == 6
    expected_dates = pd.date_range("2020-01-01", "2020-01-06", freq="D")
    pd.testing.assert_index_equal(ds.indexes["time"], expected_dates, check_names=False)

    # Day 4 (2020-01-04) was missing: MUST be NaN, NOT 0.0
    day4_pr = float(ds["pr"].sel(time="2020-01-04").values[0])
    assert np.isnan(day4_pr)

    day4_flag = str(ds["quality_flag"].sel(time="2020-01-04").values[0])
    assert day4_flag == "Faltante"


def test_excel_date_autoformat_repair(
    synthetic_dhime_df: pd.DataFrame,
) -> None:
    """Excel Spanish-locale coerced dates (e.g. 3.4 mm -> 3-Apr) must be repaired."""
    adapter = DhimeDailyPrecipitationAdapter(
        synthetic_dhime_df, lat=4.668, lon=-74.101, elevation=2552.0
    )
    ds = adapter.to_xarray()

    # Day 2: 3-Apr datetime -> 3.4 mm
    day2_pr = float(ds["pr"].sel(time="2020-01-02").values[0])
    assert np.isclose(day2_pr, 3.4)

    # Day 5: 21-Aug datetime -> 21.8 mm
    day5_pr = float(ds["pr"].sel(time="2020-01-05").values[0])
    assert np.isclose(day5_pr, 21.8)

    # Day 1: int 0 -> 0.0 mm
    day1_pr = float(ds["pr"].sel(time="2020-01-01").values[0])
    assert np.isclose(day1_pr, 0.0)

    # Day 6: string '0.4' -> 0.4 mm
    day6_pr = float(ds["pr"].sel(time="2020-01-06").values[0])
    assert np.isclose(day6_pr, 0.4)


def test_deduplication_quality_priority(
    synthetic_dhime_df: pd.DataFrame,
) -> None:
    """Duplicate dates must keep Definitivo over Preliminar."""
    adapter = DhimeDailyPrecipitationAdapter(
        synthetic_dhime_df, lat=4.668, lon=-74.101, elevation=2552.0
    )
    ds = adapter.to_xarray()

    # Day 3 had two records: 10.0 (Preliminar) and 12.5 (Definitivo)
    day3_pr = float(ds["pr"].sel(time="2020-01-03").values[0])
    day3_flag = str(ds["quality_flag"].sel(time="2020-01-03").values[0])

    assert np.isclose(day3_pr, 12.5)
    assert day3_flag == "Definitivo"


def test_cf_conventions_and_metadata(
    synthetic_dhime_df: pd.DataFrame,
) -> None:
    """Verify CF-1.8 attributes, observation window, and coordinate dimensions."""
    ds = read_dhime_daily_precipitation(
        synthetic_dhime_df, lat=4.667867, lon=-74.101034, elevation=2552.0
    )

    # Dimensions
    assert set(ds.dims) == {"time", "station"}
    assert ds["pr"].dims == ("time", "station")
    assert ds["quality_flag"].dims == ("time", "station")

    # Variable attributes
    assert ds["pr"].attrs["standard_name"] == "precipitation_flux"
    assert ds["pr"].attrs["units"] == "mm/day"
    assert "07:00 to 07:00" in ds["pr"].attrs["accumulation_window"]
    assert "12:00 to 12:00 UTC" in ds["pr"].attrs["accumulation_window"]

    # Coordinates
    assert np.isclose(float(ds["lat"].values[0]), 4.667867)
    assert np.isclose(float(ds["lon"].values[0]), -74.101034)
    assert np.isclose(float(ds["altitud"].values[0]), 2552.0)
    assert np.isclose(float(ds["elevation"].values[0]), 2552.0)
    assert ds["station"].values[0] == "21205710"

    # Global attributes
    assert ds.attrs["Conventions"] == "CF-1.8"
    assert "07:00 to 07:00" in ds.attrs["observation_window"]
    assert "date_convention" in ds.attrs


def test_read_from_csv_file(tmp_path: Path, synthetic_dhime_df: pd.DataFrame) -> None:
    """Verify reading from an actual CSV file path on disk."""
    csv_file = tmp_path / "test_dhime.csv"
    # Convert datetime object to ISO string for CSV export
    df_copy = synthetic_dhime_df.copy()
    is_dt = df_copy["Valor"].apply(lambda x: isinstance(x, datetime))
    df_copy.loc[is_dt, "Valor"] = "3.4"
    df_copy.to_csv(csv_file, index=False)

    ds = read_dhime_daily_precipitation(
        csv_file, lat=4.668, lon=-74.101, elevation=2552.0
    )
    assert isinstance(ds, xr.Dataset)
    assert len(ds.time) == 6
    assert ds["station"].values[0] == "21205710"
