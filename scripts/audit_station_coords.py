"""Audit station coordinates against the IDEAM National Station Catalog (CNE) and DEM.

Cross-references stations from Libro_estaciones.xlsx / tropicor against
local/CNE_IDEAM.xls. Normalizes coordinates to WGS84 decimal degrees,
validates all elevations against SRTM DEM (NASA/USGS), and applies user-approved
decisions for coord_source, elevation_source, and final benchmark inclusion.

Outputs: local/station_coords_audit.csv
"""

from __future__ import annotations

import json
import math
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from tropicor.io.stations import BENCHMARK_STATIONS

CNE_FILE = Path("local/CNE_IDEAM.xls")
LIBRO_FILE = Path("local/Libro_estaciones.xlsx")
OUT_CSV = Path("local/station_coords_audit.csv")
DEM_CACHE_FILE = Path("local/.dem_cache.json")

EARTH_RADIUS_KM = 6371.0088
DIST_THRESHOLD_KM = 1.0
ALT_THRESHOLD_M = 50.0

# User-approved decisions for Phase 1
APPROVED_DECISIONS: Dict[str, Dict[str, str]] = {
    # 4 stations with coordinate changes to CNE and DEM evaluated at CNE
    "32155010": {
        "coord_source": "CNE",
        "elevation_source": "DEM",
        "status": "OK",
        "motivo": "Aprobado: CNE coords, DEM evaluado en CNE",
    },
    "37045010": {
        "coord_source": "CNE",
        "elevation_source": "DEM",
        "status": "OK",
        "motivo": "Aprobado: CNE coords, DEM evaluado en CNE",
    },
    "17015010": {
        "coord_source": "CNE",
        "elevation_source": "DEM",
        "status": "OK",
        "motivo": "Aprobado: CNE coords, DEM evaluado en CNE",
    },
    "15075030": {
        "coord_source": "CNE",
        "elevation_source": "DEM",
        "status": "OK",
        "motivo": "Aprobado: CNE coords, DEM evaluado en CNE",
    },
    # 3 stations with elevation discrepancy resolved via DEM
    "23215030": {
        "coord_source": "CNE",
        "elevation_source": "DEM",
        "status": "OK",
        "motivo": "Aprobado: CNE coords, DEM elevation",
    },
    "15015060": {
        "coord_source": "CNE",
        "elevation_source": "DEM",
        "status": "OK",
        "motivo": "Aprobado: CNE coords, DEM elevation",
    },
    "23195040": {
        "coord_source": "CNE",
        "elevation_source": "DEM",
        "status": "OK",
        "motivo": "Aprobado: CNE coords, DEM (CNE erróneo: 118 m)",
    },
    # Unresolved station excluded from benchmark
    "54025010": {
        "coord_source": "—",
        "elevation_source": "—",
        "status": "UNRESOLVED",
        "motivo": "Excluido: coordenada en tierras bajas (DEM 44 m) vs CNE 1115 m",
    },
    # Island station not in CNE catalog
    "57015010": {
        "coord_source": "tesis",
        "elevation_source": "tesis",
        "status": "not_in_CNE",
        "motivo": "No aparece en CNE_IDEAM; preserva metadato de tesis",
    },
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance between two points in km."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * c


def fetch_dem_elevations(
    coords: List[Dict[str, float]],
    cache_path: Path = DEM_CACHE_FILE,
) -> Dict[str, float]:
    """Fetch ground elevation from SRTM DEM with local disk caching."""
    cache: Dict[str, float] = {}
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    to_fetch: List[Dict[str, float]] = []
    for c in coords:
        key = f"{c['latitude']:.6f},{c['longitude']:.6f}"
        if key not in cache:
            to_fetch.append({"latitude": c["latitude"], "longitude": c["longitude"]})

    if to_fetch:
        url = "https://api.open-elevation.com/api/v1/lookup"
        payload = json.dumps({"locations": to_fetch}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "TROPICOR/0.1",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
                for item in data.get("results", []):
                    k = f"{item['latitude']:.6f},{item['longitude']:.6f}"
                    cache[k] = round(float(item["elevation"]), 2)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to fetch DEM elevations from API: {e}")

    return cache


def run_audit(
    decisions: Dict[str, Dict[str, str]] | None = None,
) -> pd.DataFrame:
    """Run cross-reference audit and write local/station_coords_audit.csv."""
    if not CNE_FILE.exists():
        raise FileNotFoundError(f"CNE file not found: {CNE_FILE}")
    if not LIBRO_FILE.exists():
        raise FileNotFoundError(f"Libro file not found: {LIBRO_FILE}")

    active_decisions = decisions if decisions is not None else APPROVED_DECISIONS

    # Load CNE
    df_cne = pd.read_excel(CNE_FILE)
    df_cne["codigo_clean"] = (
        df_cne["CODIGO"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    )

    df_cne["lat_cne"] = pd.to_numeric(
        df_cne["LATITUD"].astype(str).str.strip().str.replace(",", "."),
        errors="coerce",
    )
    df_cne["lon_cne"] = pd.to_numeric(
        df_cne["LONGITUD"].astype(str).str.strip().str.replace(",", "."),
        errors="coerce",
    )
    df_cne["alt_cne"] = pd.to_numeric(df_cne["ALTITUD"], errors="coerce")

    cne_indexed = df_cne.set_index("codigo_clean")

    # Load current stations from Libro_estaciones.xlsx
    df_libro = pd.read_excel(LIBRO_FILE, sheet_name="Estaciones")
    df_libro["codigo_clean"] = (
        df_libro["Codigo"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    )

    benchmark_map = {s.code: s for s in BENCHMARK_STATIONS}

    # Fetch DEM for thesis coordinates and CNE coordinates
    coords_for_dem: List[Dict[str, float]] = []
    for _, row in df_libro.iterrows():
        coords_for_dem.append(
            {"latitude": float(row["lat"]), "longitude": float(row["lon"])}
        )
        code = str(row["codigo_clean"])
        if code in cne_indexed.index:
            cne_match = cne_indexed.loc[code]
            if isinstance(cne_match, pd.DataFrame):
                cne_match = cne_match.iloc[0]
            lat_c = float(cne_match["lat_cne"])
            lon_c = float(cne_match["lon_cne"])
            if not math.isnan(lat_c) and not math.isnan(lon_c):
                coords_for_dem.append({"latitude": lat_c, "longitude": lon_c})

    dem_cache = fetch_dem_elevations(coords_for_dem)

    audit_records: List[Dict[str, Any]] = []

    for _, row in df_libro.iterrows():
        code = str(row["codigo_clean"])
        name = str(row["Estacion"]).strip()
        lat_act = float(row["lat"])
        lon_act = float(row["lon"])

        meta = benchmark_map.get(code)
        alt_act = meta.elevation if meta else float("nan")

        key_dem_act = f"{lat_act:.6f},{lon_act:.6f}"
        alt_dem_act = dem_cache.get(key_dem_act, float("nan"))
        dif_dem_act = (
            alt_dem_act - alt_act if not math.isnan(alt_dem_act) else float("nan")
        )

        if code in cne_indexed.index:
            cne_match = cne_indexed.loc[code]
            if isinstance(cne_match, pd.DataFrame):
                cne_match = cne_match.iloc[0]

            lat_cne = float(cne_match["lat_cne"])
            lon_cne = float(cne_match["lon_cne"])
            alt_cne = float(cne_match["alt_cne"])
            cne_name = str(cne_match["NOMBRE"]).strip()

            dist_km = haversine_km(lat_act, lon_act, lat_cne, lon_cne)
            dif_alt = alt_cne - alt_act

            # DEM at CNE coordinate
            key_dem_cne = f"{lat_cne:.6f},{lon_cne:.6f}"
            alt_dem_cne = dem_cache.get(key_dem_cne, alt_dem_act)
            dif_dem_cne = (
                alt_dem_cne - alt_cne if not math.isnan(alt_dem_cne) else float("nan")
            )
        else:
            lat_cne = float("nan")
            lon_cne = float("nan")
            alt_cne = float("nan")
            cne_name = ""
            dist_km = float("nan")
            dif_alt = float("nan")
            alt_dem_cne = float("nan")
            dif_dem_cne = float("nan")

        # Determine decision parameters
        if code in active_decisions:
            dec = active_decisions[code]
            coord_source = dec["coord_source"]
            elev_source = dec["elevation_source"]
            status = dec["status"]
            motivo = dec["motivo"]
        else:
            coord_source = "CNE"
            elev_source = "CNE"
            status = "OK"
            motivo = "OK"

        # Chosen altitude based on elevation_source
        if elev_source == "DEM":
            # If coordinates are from CNE, use DEM at CNE coordinates
            alt_final = alt_dem_cne if coord_source == "CNE" else alt_dem_act
        elif elev_source == "CNE":
            alt_final = alt_cne
        elif elev_source == "tesis":
            alt_final = alt_act
        else:
            alt_final = float("nan")

        audit_records.append(
            {
                "codigo": code,
                "nombre_actual": name,
                "nombre_cne": cne_name,
                "coord_source": coord_source,
                "elevation_source": elev_source,
                "altitud_final": (
                    round(alt_final, 1) if not math.isnan(alt_final) else float("nan")
                ),
                "lat_actual": lat_act,
                "lon_actual": lon_act,
                "altitud_actual": alt_act,
                "lat_cne": lat_cne,
                "lon_cne": lon_cne,
                "altitud_cne": alt_cne,
                "altitud_dem_cne": (
                    round(alt_dem_cne, 1)
                    if not math.isnan(alt_dem_cne)
                    else float("nan")
                ),
                "altitud_dem_tesis": (
                    round(alt_dem_act, 1)
                    if not math.isnan(alt_dem_act)
                    else float("nan")
                ),
                "dif_dem_vs_actual": (
                    round(dif_dem_act, 1)
                    if not math.isnan(dif_dem_act)
                    else float("nan")
                ),
                "dif_dem_vs_cne": (
                    round(dif_dem_cne, 1)
                    if not math.isnan(dif_dem_cne)
                    else float("nan")
                ),
                "distancia_km": (
                    round(dist_km, 3) if not math.isnan(dist_km) else float("nan")
                ),
                "dif_altitud_m": (
                    round(dif_alt, 1) if not math.isnan(dif_alt) else float("nan")
                ),
                "status": status,
                "motivo": motivo,
            }
        )

    df_audit = pd.DataFrame(audit_records)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_audit.to_csv(OUT_CSV, index=False)
    return df_audit


if __name__ == "__main__":
    df_res = run_audit()
    print("=" * 80)
    print("AUDITORÍA DE COORDENADAS CON DECISIONES APROBADAS Y VALIDACIÓN DEM")
    print("=" * 80)
    print(f"Total estaciones: {len(df_res)}")
    print(f"Estado OK: {(df_res['status'] == 'OK').sum()}")
    print(f"Estado UNRESOLVED: {(df_res['status'] == 'UNRESOLVED').sum()}")
    print(f"Estado not_in_CNE: {(df_res['status'] == 'not_in_CNE').sum()}")
