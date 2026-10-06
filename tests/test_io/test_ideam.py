"""Unit tests for native IDEAM DHIME IdeamAdapter."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tropicor.io.ideam import IdeamAdapter


def test_ideam_adapter_file_not_found() -> None:
    """Test IdeamAdapter raises FileNotFoundError for missing path."""
    with pytest.raises(FileNotFoundError, match="IDEAM file not found"):
        IdeamAdapter("non_existent_file.xlsx")


def test_ideam_adapter_synthetic_metadata(synthetic_dhime_excel: Path) -> None:
    """Test metadata parsing from synthetic DHIME excel fixture."""
    adapter = IdeamAdapter(synthetic_dhime_excel)

    assert adapter.code == "21205710"
    assert adapter.name == "JARDIN BOTANICO"
    assert adapter.latitude == 4.667867
    assert adapter.longitude == -74.101034
    assert adapter.elevation == 2552.0
    assert adapter.department == "Bogotá"
    assert adapter.municipality == "Bogotá, D.C"
    assert adapter.variable == "PRECIPITACION"
    assert adapter.frequency == "Mensual"
    assert adapter.unit == "mm"


def test_ideam_adapter_synthetic_series(synthetic_dhime_excel: Path) -> None:
    """Test get_series() extraction from synthetic DHIME excel fixture."""
    adapter = IdeamAdapter(synthetic_dhime_excel)
    series = adapter.get_series()

    assert isinstance(series, pd.Series)
    assert isinstance(series.index, pd.DatetimeIndex)
    assert len(series) == 12
    assert series.name == "precipitacion"

    # Verify first value (Jan 1990)
    assert series.iloc[0] == 62.5
    assert series.index[0] == pd.Timestamp("1990-01-01")

    # Verify missing sentinel -9999.0 (Oct 1990, index 9) mapped to NaN
    oct_val = series.loc[pd.Timestamp("1990-10-01")]
    assert np.isnan(oct_val)


def test_ideam_adapter_real_local_file() -> None:
    """Test parsing real DHIME file local/21205710.xlsx if present."""
    local_file = Path("local/21205710.xlsx")
    if not local_file.exists():
        pytest.skip("Local test file local/21205710.xlsx not found.")

    adapter = IdeamAdapter(local_file)
    assert adapter.code == "21205710"
    assert adapter.name == "JARDIN BOTANICO"
    assert adapter.latitude == 4.667867
    assert adapter.longitude == -74.101034
    assert adapter.elevation == 2552.0
    assert adapter.variable == "PRECIPITACION"

    series = adapter.get_series()
    assert isinstance(series, pd.Series)
    assert isinstance(series.index, pd.DatetimeIndex)
    assert len(series) > 300  # 1990 to 2026 ~ 400+ monthly records
    assert series.index[0] == pd.Timestamp("1990-01-01")
    assert series.iloc[0] == 62.5
