"""Generate the Andean station registry (tropicor/data/andean_station_registry.json).

Reads:
- docs/science/appendix_f/coverage_audit_new_stations.json
- docs/science/appendix_f/era5_download_plan.json
- local/CNE_IDEAM.xls

Computes SHA-256 hashes of both input reports and embeds them into the registry.
Contains zero observation values.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def generate_andean_registry(
    coverage_path: Path = Path(
        "docs/science/appendix_f/coverage_audit_new_stations.json"
    ),
    plan_path: Path = Path("docs/science/appendix_f/era5_download_plan.json"),
    cne_path: Path = Path("local/CNE_IDEAM.xls"),
    output_path: Path = Path("tropicor/data/andean_station_registry.json"),
) -> Dict[str, Any]:
    """Generate canonical andean_station_registry.json from coverage audit and download plan."""
    if not coverage_path.exists():
        raise FileNotFoundError(f"Coverage audit not found: {coverage_path}")
    if not plan_path.exists():
        raise FileNotFoundError(f"Download plan not found: {plan_path}")
    if not cne_path.exists():
        raise FileNotFoundError(f"CNE catalog not found: {cne_path}")

    cov_sha = compute_sha256(coverage_path)
    plan_sha = compute_sha256(plan_path)

    with open(coverage_path, "r", encoding="utf-8") as f:
        cov_data = json.load(f)

    with open(plan_path, "r", encoding="utf-8") as f:
        plan_data = json.load(f)

    # Read altitudes from CNE
    df_cne = pd.read_excel(cne_path)
    df_cne["CODIGO_STR"] = (
        df_cne["CODIGO"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    )
    cne_alt_map = {}
    for _, row in df_cne.iterrows():
        cne_alt_map[row["CODIGO_STR"]] = float(row["ALTITUD"])

    # Find complete slots in slot order
    slots_selection = cov_data.get("slot_selection", {})
    stations_out: Dict[str, Any] = {}

    for slot_name, slot_info in sorted(slots_selection.items()):
        if slot_info.get("status") == "COMPLETA" and slot_info.get("selected_station"):
            st_code = str(slot_info["selected_station"])
            st_cov = cov_data["stations"][st_code]
            st_plan = plan_data["stations"][st_code]

            joint_valid = st_cov["joint_temperature"]["joint_valid_years_list"]
            cal_b = st_cov["temperature_evaluation"]["partition"]["cal_block"]
            val_b = st_cov["temperature_evaluation"]["partition"]["val_block"]

            stations_out[st_code] = {
                "codigo": st_code,
                "nombre": st_plan["nombre"],
                "latitud_cne": float(st_plan["catalogo_lat"]),
                "longitud_cne": float(st_plan["catalogo_lon"]),
                "altitud_cne": float(cne_alt_map.get(st_code, 0.0)),
                "celda_era5": {
                    "lat": float(st_plan["celda_lat"]),
                    "lon": float(st_plan["celda_lon"]),
                },
                "cal_block": [int(cal_b[0]), int(cal_b[1])],
                "val_block": [int(val_b[0]), int(val_b[1])],
                "valid_years": [int(y) for y in joint_valid],
            }

    registry: Dict[str, Any] = {
        "description": "Registry of Andean generalization stations (Protocol v1.4, Appendix F)",
        "version": "1.4",
        "source_coverage_audit_sha256": cov_sha,
        "source_download_plan_sha256": plan_sha,
        "stations": stations_out,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Registry written to {output_path} with {len(stations_out)} stations.")
    return registry


if __name__ == "__main__":
    generate_andean_registry()
