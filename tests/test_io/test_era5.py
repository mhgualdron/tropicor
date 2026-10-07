"""Unit tests for ERA5 reanalysis ingestion engine."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tropicor.io.era5 import ERA5Adapter
from tropicor.io.stations import StationMetadata


def test_era5_adapter_initialization(synthetic_era5_dataset: Path) -> None:
    """Test initialization and coordinate monotonic ascending sorting."""
    adapter = ERA5Adapter(synthetic_era5_dataset)
    ds = adapter.dataset

    assert "time" in ds.coords
    assert "latitude" in ds.coords
    assert "longitude" in ds.coords

    # Verify latitudes and longitudes are sorted monotonically ascending
    lats = ds["latitude"].values
    lons = ds["longitude"].values
    assert np.all(np.diff(lats) > 0)
    assert np.all(np.diff(lons) > 0)

    adapter.close()


def test_era5_adapter_expver_and_valid_time(
    synthetic_era5_expver_dataset: Path,
) -> None:
    """Test Copernicus CDS resolution of valid_time and expver dimension."""
    with ERA5Adapter(synthetic_era5_expver_dataset) as adapter:
        ds = adapter.dataset
        assert "time" in ds.coords
        assert "valid_time" not in ds.coords
        assert "expver" not in ds.dims
        assert "expver" not in ds.coords

        # Extract series and verify expver=1 was chosen (295.15 K -> 22.0 °C)
        series = adapter.get_series(latitude=6.0, longitude=-73.0, variable="t2m")
        assert len(series) == 6
        assert np.isclose(series.iloc[0], 22.0, atol=1e-3)


def test_era5_adapter_extract_point_nearest(synthetic_era5_dataset: Path) -> None:
    """Test point extraction with nearest neighbor interpolation."""
    with ERA5Adapter(synthetic_era5_dataset) as adapter:
        pt = adapter.extract_point(latitude=7.1, longitude=-73.1, method="nearest")
        assert "t2m" in pt.data_vars
        assert "tp" in pt.data_vars
        assert pt["t2m"].ndim == 1
        assert len(pt["time"]) == 360


def test_era5_adapter_extract_point_bilinear(synthetic_era5_dataset: Path) -> None:
    """Test point extraction with bilinear spatial interpolation."""
    with ERA5Adapter(synthetic_era5_dataset) as adapter:
        pt = adapter.extract_point(latitude=7.125, longitude=-73.125, method="bilinear")
        assert "t2m" in pt.data_vars
        assert pt["t2m"].ndim == 1
        assert len(pt["time"]) == 360


def test_era5_adapter_unit_conversion_metadata(synthetic_era5_dataset: Path) -> None:
    """Test deterministic unit conversions reading NetCDF attrs['units']."""
    with ERA5Adapter(synthetic_era5_dataset) as adapter:
        # t2m has units='K' -> converted to °C (295.15 K base -> ~22°C)
        t_series = adapter.get_series(latitude=7.0, longitude=-73.0, variable="t2m")
        assert isinstance(t_series, pd.Series)
        assert len(t_series) == 360
        assert 15.0 < t_series.mean() < 28.0

        # tp has units='m' -> converted to mm (* 1000)
        p_series = adapter.get_series(latitude=7.0, longitude=-73.0, variable="tp")
        assert isinstance(p_series, pd.Series)
        assert len(p_series) == 360
        # Synthetic raw tp was 0.05 to 0.35 m -> in mm should be 50 to 350 mm
        assert p_series.min() >= 45.0
        assert p_series.max() <= 355.0


def test_era5_adapter_missing_units_warning_and_passthrough(
    synthetic_era5_missing_units_dataset: Path,
) -> None:
    """Test that missing units attribute triggers UserWarning and passes through."""
    with ERA5Adapter(synthetic_era5_missing_units_dataset) as adapter:
        with pytest.warns(UserWarning, match="missing 'units' attribute"):
            t_series = adapter.get_series(latitude=7.5, longitude=-72.5, variable="t2m")

        # Values should NOT be modified by heuristics (should stay 22.5)
        assert np.isclose(t_series.iloc[0], 22.5)

        with pytest.warns(UserWarning, match="missing 'units' attribute"):
            p_series = adapter.get_series(latitude=7.5, longitude=-72.5, variable="tp")

        # Values should NOT be modified by heuristics (should stay 100.0)
        assert np.isclose(p_series.iloc[0], 100.0)


def test_era5_adapter_model_elevation(synthetic_era5_dataset: Path) -> None:
    """Test model elevation extraction from surface geopotential z / 9.80665."""
    with ERA5Adapter(synthetic_era5_dataset) as adapter:
        elev = adapter.get_model_elevation(latitude=7.0, longitude=-73.0)
        assert elev is not None
        assert np.isclose(elev, 2118.0, atol=0.1)


def test_era5_adapter_model_elevation_missing(
    synthetic_era5_expver_dataset: Path,
) -> None:
    """Test model elevation returns None when variable z is missing."""
    with ERA5Adapter(synthetic_era5_expver_dataset) as adapter:
        elev = adapter.get_model_elevation(latitude=6.0, longitude=-73.0)
        assert elev is None


def test_era5_adapter_get_station_series(
    synthetic_era5_dataset: Path,
    synthetic_station_dict: dict,
) -> None:
    """Test extracting series directly from StationMetadata object."""
    station = StationMetadata(**synthetic_station_dict)
    with ERA5Adapter(synthetic_era5_dataset) as adapter:
        series = adapter.get_station_series(station, variable="tp", method="nearest")
        assert isinstance(series, pd.Series)
        assert len(series) == 360
        assert series.name == "tp"


def test_era5_adapter_context_manager(synthetic_era5_dataset: Path) -> None:
    """Test context manager correctly closes file handles."""
    with ERA5Adapter(synthetic_era5_dataset) as adapter:
        assert adapter.dataset is not None
    # Dataset handle is closed; accessing or closing again should not fail
    adapter.close()


def test_era5_adapter_invalid_method(synthetic_era5_dataset: Path) -> None:
    """Test that unsupported spatial interpolation method raises ValueError."""
    with ERA5Adapter(synthetic_era5_dataset) as adapter:
        with pytest.raises(ValueError, match="Invalid extraction method"):
            adapter.extract_point(latitude=7.0, longitude=-73.0, method="cubic")


def test_era5_adapter_invalid_variable(synthetic_era5_dataset: Path) -> None:
    """Test that requesting nonexistent variable raises KeyError."""
    with ERA5Adapter(synthetic_era5_dataset) as adapter:
        with pytest.raises(KeyError, match="Variable 'nonexistent_var' not found"):
            adapter.get_series(
                latitude=7.0, longitude=-73.0, variable="nonexistent_var"
            )


def test_era5_adapter_file_not_found(tmp_path: Path) -> None:
    """Test FileNotFoundError when opening nonexistent file path."""
    nonexistent = tmp_path / "does_not_exist.nc"
    with pytest.raises(FileNotFoundError, match="ERA5 file not found"):
        ERA5Adapter(nonexistent)
