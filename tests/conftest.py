"""Pytest fixtures for TROPICOR unit and integration tests."""

from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest


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
