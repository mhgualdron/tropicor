"""Unit tests for station metadata taxonomy and StationCatalog."""

import pandas as pd
import pytest

from tropicor.io.stations import (
    BENCHMARK_STATIONS,
    NaturalRegion,
    StationCatalog,
    StationMetadata,
)


def test_station_metadata_initialization() -> None:
    """Test creating a valid StationMetadata record."""
    station = StationMetadata(
        code="21205710",
        name="JARDIN BOTANICO",
        latitude=4.667867,
        longitude=-74.101034,
        elevation=2552.0,
        region=NaturalRegion.ANDINA,
        department="Bogotá",
        municipality="Bogotá, D.C",
    )
    assert station.code == "21205710"
    assert station.name == "JARDIN BOTANICO"
    assert station.elevation == 2552.0
    assert station.region == NaturalRegion.ANDINA


def test_station_metadata_bounds_validation() -> None:
    """Test spatial coordinate bounds validation."""
    # Invalid latitude (> 13.5)
    with pytest.raises(ValueError, match="Latitude 15.0 outside Colombia"):
        StationMetadata(
            code="99999999",
            name="OUT OF BOUNDS",
            latitude=15.0,
            longitude=-74.0,
            elevation=100.0,
            region=NaturalRegion.CARIBE,
            department="Test",
        )

    # Invalid longitude (< -82.0)
    with pytest.raises(ValueError, match="Longitude -85.0 outside Colombia"):
        StationMetadata(
            code="99999999",
            name="OUT OF BOUNDS",
            latitude=5.0,
            longitude=-85.0,
            elevation=100.0,
            region=NaturalRegion.PACIFICO,
            department="Test",
        )


def test_station_catalog_default_initialization() -> None:
    """Test StationCatalog initializes with benchmark stations."""
    catalog = StationCatalog()
    assert len(catalog) == len(BENCHMARK_STATIONS)
    assert len(catalog) > 0


def test_station_catalog_get_by_code() -> None:
    """Test fetching station by code."""
    catalog = StationCatalog()
    st = catalog.get_by_code("21205710")
    assert st.name == "JARDIN BOTANICO - AUT"
    assert st.elevation == 2553.0

    # Test dictionary-style access (UIS Bucaramanga)
    st_dict_access = catalog["23195040"]
    assert st_dict_access.name == "UNIVERSIDAD INDUSTRIAL SANTANDER"
    assert st_dict_access.elevation == 898.0

    # Test key error for unknown station
    with pytest.raises(KeyError, match="Station code '99999999' not found"):
        catalog.get_by_code("99999999")


def test_station_catalog_filter_by_region() -> None:
    """Test filtering catalog by natural region."""
    catalog = StationCatalog()
    andina_stations = catalog.filter_by_region(NaturalRegion.ANDINA)
    assert len(andina_stations) > 0
    assert all(st.region == NaturalRegion.ANDINA for st in andina_stations)

    # Filter by string name
    caribe_stations = catalog.filter_by_region("CARIBE")
    assert len(caribe_stations) > 0
    assert all(st.region == NaturalRegion.CARIBE for st in caribe_stations)


def test_station_catalog_bounding_box() -> None:
    """Test computing spatial bounding box of catalog."""
    catalog = StationCatalog()
    min_lat, max_lat, min_lon, max_lon = catalog.bounding_box()

    assert -4.5 <= min_lat < max_lat <= 13.5
    assert -82.0 <= min_lon < max_lon <= -66.5


def test_station_catalog_to_dataframe() -> None:
    """Test exporting catalog to pandas DataFrame."""
    catalog = StationCatalog()
    df = catalog.to_dataframe()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(catalog)
    expected_cols = {
        "code",
        "name",
        "latitude",
        "longitude",
        "elevation",
        "region",
        "department",
        "municipality",
        "variable",
    }
    assert expected_cols.issubset(df.columns)
    assert (df["region"] == "ANDINA").any()
