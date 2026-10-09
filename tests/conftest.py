"""Pytest fixtures for TROPICOR unit and integration tests."""

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import pytest
import xarray as xr

# Ensure matplotlib uses headless Agg backend in test suite
try:
    import matplotlib

    matplotlib.use("Agg")
except ImportError:
    pass


@pytest.fixture
def synthetic_station_dict() -> Dict[str, Any]:
    """Return synthetic station metadata dictionary for unit testing."""
    return {
        "code": "21205710",
        "name": "JARDIN BOTANICO",
        "latitude": 4.667867,
        "longitude": -74.101034,
        "elevation": 2552.0,
        "region": "ANDINA",
        "department": "Bogotá",
        "municipality": "Bogotá, D.C",
        "variable": "PRECIPITACION",
    }


@pytest.fixture
def synthetic_dhime_excel(tmp_path: Path) -> Path:
    """Generate a synthetic native IDEAM DHIME Excel file (.xlsx)."""
    file_path = tmp_path / "synthetic_dhime_21205710.xlsx"

    # Raw metadata grid matching real DHIME header structure
    header_report = (
        "Reporte de información Hidrometeorológica de DHIME generado (06/10/2026 09:33)"
    )
    raw_rows = [
        [None, header_report, None, None, None, None],
        [
            "Nombre estacion:",
            "JARDIN BOTANICO [21205710]",
            "Corriente:",
            None,
            "Categoría de estación:",
            "Climatológica Principal",
        ],
        ["Latitud:", 4.667867, "Longitud:", -74.101034, "Elevación:", 2552],
        [
            "Entidad:",
            "INSTITUTO DE HIDROLOGIA METEOROLOGIA Y ESTUDIOS AMBIENTALES",
            "Area Operativa:",
            "Area Operativa 11 - Cundinamarca-Amazonas",
            "Departamento:",
            "Bogotá",
        ],
        [
            "Municipio:",
            "Bogotá, D.C",
            "Fecha instalacïon:",
            "15/09/1974 00:00",
            "Fecha suspensión:",
            None,
        ],
        [
            "Variable:",
            "PRECIPITACION",
            "Frecuencia:",
            "Mensual",
            "Fecha consulta:",
            "01/01/1990 00:00-01/12/1990 00:00",
        ],
        [
            "Parametro:",
            "Precipitación total mensual",
            "Unidad medida:",
            "mm",
            None,
            None,
        ],
        ["Fecha", None, "Valor:", None, "Nivel de Aprobación", None],
    ]

    # Add 12 monthly data rows for 1990 (including missing sentinel -9999.0)
    dates = pd.date_range("1990-01-01", "1990-12-01", freq="MS")
    values = [
        62.5,
        44.9,
        58.2,
        143.3,
        104.4,
        10.4,
        33.2,
        45.0,
        78.1,
        -9999.0,
        92.4,
        50.0,
    ]

    for dt, val in zip(dates, values):
        raw_rows.append(
            [dt.strftime("%Y-%m-%d %H:%M"), None, val, None, "Preliminar", None]
        )

    df = pd.DataFrame(raw_rows)
    df.to_excel(file_path, index=False, header=False, engine="openpyxl")
    return file_path


@pytest.fixture
def synthetic_era5_dataset(tmp_path: Path) -> Path:
    """Generate a synthetic 30-year monthly NetCDF dataset (1990-2019, 360 months)."""
    nc_path = tmp_path / "synthetic_era5_1990_2019.nc"

    times = pd.date_range("1990-01-01", periods=360, freq="MS")
    # ECMWF descending latitudes convention (15.0N down to -5.0S)
    lats = np.linspace(15.0, -5.0, 81, dtype=np.float32)
    lons = np.linspace(-85.0, -65.0, 81, dtype=np.float32)

    # Synthetic temperature in Kelvin (~22°C base + annual cycle + noise)
    t2m_data = np.zeros((len(times), len(lats), len(lons)), dtype=np.float32)
    for t_idx, t in enumerate(times):
        seasonal = 2.0 * np.sin(2.0 * np.pi * t.month / 12.0)
        t2m_data[t_idx, :, :] = 295.15 + seasonal

    # Synthetic precipitation in meters (0.05m to 0.35m)
    rng = np.random.default_rng(42)
    tp_data = rng.uniform(0.05, 0.35, size=(len(times), len(lats), len(lons))).astype(
        np.float32
    )

    # Synthetic geopotential in m^2/s^2 (2118m * 9.80665 near Bucaramanga / Santander)
    g0 = 9.80665
    z_data = np.full((len(lats), len(lons)), 2118.0 * g0, dtype=np.float32)

    ds = xr.Dataset(
        data_vars={
            "t2m": (
                ("time", "latitude", "longitude"),
                t2m_data,
                {"units": "K", "long_name": "2 metre temperature"},
            ),
            "tp": (
                ("time", "latitude", "longitude"),
                tp_data,
                {"units": "m", "long_name": "Total precipitation"},
            ),
            "z": (
                ("latitude", "longitude"),
                z_data,
                {"units": "m**2 s**-2", "long_name": "Geopotential"},
            ),
        },
        coords={
            "time": times,
            "latitude": lats,
            "longitude": lons,
        },
        attrs={
            "Conventions": "CF-1.6",
            "history": "Synthetic ERA5 generated for TROPICOR unit tests",
        },
    )

    ds.to_netcdf(nc_path)
    return nc_path


@pytest.fixture
def synthetic_era5_missing_units_dataset(tmp_path: Path) -> Path:
    """Generate a small NetCDF dataset with missing 'units' metadata attributes."""
    nc_path = tmp_path / "synthetic_era5_missing_units.nc"

    times = pd.date_range("1990-01-01", periods=12, freq="MS")
    lats = np.linspace(10.0, 5.0, 5, dtype=np.float32)
    lons = np.linspace(-75.0, -70.0, 5, dtype=np.float32)

    t2m_data = np.full((len(times), len(lats), len(lons)), 22.5, dtype=np.float32)
    tp_data = np.full((len(times), len(lats), len(lons)), 100.0, dtype=np.float32)

    ds = xr.Dataset(
        data_vars={
            "t2m": (("time", "latitude", "longitude"), t2m_data),  # No attrs
            "tp": (("time", "latitude", "longitude"), tp_data),  # No attrs
        },
        coords={
            "time": times,
            "latitude": lats,
            "longitude": lons,
        },
    )

    ds.to_netcdf(nc_path)
    return nc_path


@pytest.fixture
def synthetic_era5_expver_dataset(tmp_path: Path) -> Path:
    """Generate a NetCDF dataset with Copernicus CDS 'expver' and valid_time."""
    nc_path = tmp_path / "synthetic_era5_expver.nc"

    valid_times = pd.date_range("1990-01-01", periods=6, freq="MS")
    lats = np.linspace(8.0, 4.0, 3, dtype=np.float32)
    lons = np.linspace(-74.0, -72.0, 3, dtype=np.float32)
    expvers = np.array([1, 5], dtype=np.int32)

    t2m_data = np.zeros(
        (len(valid_times), len(expvers), len(lats), len(lons)), dtype=np.float32
    )
    # expver=1 has valid consolidated data (295.15 K)
    t2m_data[:, 0, :, :] = 295.15
    # expver=5 has preliminary data / NaNs
    t2m_data[:, 1, :, :] = np.nan

    ds = xr.Dataset(
        data_vars={
            "t2m": (
                ("valid_time", "expver", "latitude", "longitude"),
                t2m_data,
                {"units": "K", "long_name": "2 metre temperature"},
            ),
        },
        coords={
            "valid_time": valid_times,
            "expver": expvers,
            "latitude": lats,
            "longitude": lons,
        },
    )

    ds.to_netcdf(nc_path)
    return nc_path


@pytest.fixture
def synthetic_paired_series() -> Tuple[pd.Series, pd.Series]:
    """Generate paired observed vs modeled 30-year monthly time series."""
    idx = pd.date_range("1990-01-01", periods=360, freq="MS")
    t = np.linspace(0, 30 * 2 * np.pi, 360)
    rng = np.random.default_rng(42)

    observed = pd.Series(
        25.0 + 5.0 * np.sin(t) + rng.normal(0, 0.2, 360),
        index=idx,
        name="observed",
    )
    # Model with high correlation but -8°C Andean elevation bias
    modeled = pd.Series(
        17.0 + 5.0 * np.sin(t) + rng.normal(0, 0.2, 360),
        index=idx,
        name="modeled",
    )
    return observed, modeled
