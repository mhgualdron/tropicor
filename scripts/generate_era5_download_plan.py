"""Generate ERA5 download plan for candidate stations based on CNE_IDEAM.xls."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.download_era5_years import compute_nearest_grid_center  # noqa: E402


def get_complete_stations_from_coverage(
    coverage_path: Path = Path("local/reports/coverage_audit_new_stations.json"),
) -> List[str]:
    """Extract candidate station codes from slots marked COMPLETA in coverage JSON."""
    if not coverage_path.exists():
        raise FileNotFoundError(f"Coverage audit JSON not found: {coverage_path}")

    with open(coverage_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    complete_stations: List[str] = []
    for _slot, sdata in data.get("slot_selection", {}).items():
        if sdata.get("status") == "COMPLETA" and sdata.get("selected_station"):
            complete_stations.append(str(sdata["selected_station"]))

    return sorted(complete_stations)


def generate_download_plan(
    cne_path: Path = Path("local/CNE_IDEAM.xls"),
    coverage_path: Path = Path("local/reports/coverage_audit_new_stations.json"),
    target_stations: Optional[List[str]] = None,
    output_path: Path = Path("local/reports/era5_download_plan.json"),
) -> Dict[str, Any]:
    """Generate download plan strictly from CNE_IDEAM.xls for complete slots."""
    if not cne_path.exists():
        raise FileNotFoundError(f"Catalog file not found: {cne_path}")

    if target_stations is None:
        target_stations = get_complete_stations_from_coverage(coverage_path)

    df = pd.read_excel(cne_path)
    df["CODIGO_STR"] = (
        df["CODIGO"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    )

    plan_entries: Dict[str, Dict[str, Any]] = {}
    cell_mapping: Dict[tuple[float, float], List[str]] = {}

    for code in target_stations:
        match = df[df["CODIGO_STR"] == code]
        if match.empty:
            raise ValueError(f"Station {code} not found in catalog {cne_path}")

        row = match.iloc[0]
        name = str(row["NOMBRE"]).strip()

        lat_raw = str(row["LATITUD"]).strip().replace(",", ".")
        lon_raw = str(row["LONGITUD"]).strip().replace(",", ".")

        lat = float(lat_raw)
        lon = float(lon_raw)

        # Bounds verification
        if not (-5.0 <= lat <= 14.0):
            raise ValueError(
                f"Station {code} latitude {lat} outside valid bounds [-5, 14]"
            )
        if not (-80.0 <= lon <= -66.0):
            raise ValueError(
                f"Station {code} longitude {lon} outside valid bounds [-80, -66]"
            )

        lat_center, lon_center = compute_nearest_grid_center(lat, lon)

        margin_lat = 0.125 - abs(lat - lat_center)
        margin_lon = 0.125 - abs(lon - lon_center)
        margen = float(min(margin_lat, margin_lon))

        grid_cell = (float(lat_center), float(lon_center))
        if grid_cell not in cell_mapping:
            cell_mapping[grid_cell] = []
        cell_mapping[grid_cell].append(code)

        plan_entries[code] = {
            "codigo": code,
            "nombre": name,
            "catalogo_lat": lat,
            "catalogo_lon": lon,
            "celda_lat": float(lat_center),
            "celda_lon": float(lon_center),
            "margen": margen,
            "margen_menor_0_01": bool(margen < 0.01),
            "shared_cell_with": [],
        }

    # Add shared_cell_with
    for code, entry in plan_entries.items():
        cell = (entry["celda_lat"], entry["celda_lon"])
        shared = [c for c in cell_mapping[cell] if c != code]
        entry["shared_cell_with"] = shared

    result = {
        "description": "ERA5 timeseries download plan for candidate stations",
        "catalog_source": "local/CNE_IDEAM.xls",
        "stations": plan_entries,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    res = generate_download_plan()
    print("Download plan generated successfully:")
    for code, data in res["stations"].items():
        print(
            f"  {code} ({data['nombre']}): coord=({data['catalogo_lat']}, {data['catalogo_lon']}) "
            f"-> celda=({data['celda_lat']}, {data['celda_lon']}), margen={data['margen']:.6f}°, "
            f"margen < 0.01°: {data['margen_menor_0_01']}, shared={data['shared_cell_with']}"
        )
