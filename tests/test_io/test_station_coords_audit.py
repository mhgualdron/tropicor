"""Tests for station coordinates audit against CNE catalog (Fase 1)."""

from pathlib import Path

import pytest

from scripts.audit_station_coords import haversine_km, run_audit


def test_haversine_km_known_points() -> None:
    """Verify haversine formula against known distance between Bogotá and Medellín."""
    # Bogotá (approx 4.7110, -74.0721) to Medellín (approx 6.2442, -75.5812)
    dist = haversine_km(4.7110, -74.0721, 6.2442, -75.5812)
    assert 235.0 < dist < 245.0


def test_haversine_identical_point() -> None:
    """Distance between identical coordinates must be 0.0."""
    assert haversine_km(4.5, -74.0, 4.5, -74.0) == 0.0


@pytest.mark.skipif(
    not Path("local/CNE_IDEAM.xls").exists()
    or not Path("local/Libro_estaciones.xlsx").exists(),
    reason="Local audit files not present",
)
def test_station_coords_audit_execution() -> None:
    """Verify audit runs, generates CSV, and produces expected columns and rows."""
    df_res = run_audit()
    assert len(df_res) == 48

    expected_cols = [
        "codigo",
        "nombre_actual",
        "nombre_cne",
        "coord_source",
        "elevation_source",
        "altitud_final",
        "lat_actual",
        "lon_actual",
        "altitud_actual",
        "lat_cne",
        "lon_cne",
        "altitud_cne",
        "altitud_dem_cne",
        "altitud_dem_tesis",
        "dif_dem_vs_actual",
        "dif_dem_vs_cne",
        "distancia_km",
        "dif_altitud_m",
        "status",
        "motivo",
    ]
    for col in expected_cols:
        assert col in df_res.columns

    # Check valid status flags
    assert set(df_res["status"].unique()).issubset(
        {"OK", "REVIEW", "UNRESOLVED", "not_in_CNE"}
    )

    # Check output file exists
    out_csv = Path("local/station_coords_audit.csv")
    assert out_csv.exists()
    assert out_csv.stat().st_size > 0
