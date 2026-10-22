"""Generate Table F.4 for docs/science/evaluation-protocol.md from canonical data files."""

from __future__ import annotations

import json
from pathlib import Path


def generate_table_f4() -> str:
    cov_path = Path("docs/science/appendix_f/coverage_audit_new_stations.json")
    reg_path = Path("tropicor/data/andean_station_registry.json")

    with open(cov_path, "r", encoding="utf-8") as f:
        cov = json.load(f)
    with open(reg_path, "r", encoding="utf-8") as f:
        reg = json.load(f)

    regions = {
        "37015020": "Oriental (Páramo)",
        "24025040": "Oriental (Premontane)",
        "26135040": "Central (Premontane)",
        "27015070": "Central (Aburrá Valley)",
        "52045010": "Nudo de los Pastos / Nariño (Andes del sur)",
    }
    slots = {
        "37015020": "S2",
        "24025040": "S3",
        "26135040": "S4",
        "27015070": "S6",
        "52045010": "S8",
    }

    lines = [
        (
            "| Slot | Code | Station Name | Cordillera / Region | CNE Elev. (m) | "
            "Joint Valid Yrs | Cal. Block ($n_{\\text{cal}}$) | "
            "Val. Block ($n_{\\text{val}}$) | "
            "SHA-256 (12-char prefix) `DiaPluviometrico` | "
            "SHA-256 (12-char prefix) `TempMaxima` | "
            "SHA-256 (12-char prefix) `TempMinima` |"
        ),
        "|:---:|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for st_id in ["37015020", "24025040", "26135040", "27015070", "52045010"]:
        st = reg["stations"][st_id]
        st_cov = cov["stations"][st_id]
        v = st_cov["variables"]
        p_sha = v["P"]["file_sha256"][:12]
        tx_sha = v["Tmax"]["file_sha256"][:12]
        tn_sha = v["Tmin"]["file_sha256"][:12]

        cal_b = st["cal_block"]
        val_b = st["val_block"]
        n_cal = st_cov["temperature_evaluation"]["partition"]["n_cal_valid"]
        n_val = st_cov["temperature_evaluation"]["partition"]["n_val_valid"]
        n_total = st_cov["joint_temperature"]["joint_valid_years_count"]
        name = st["nombre"].split("[")[0].strip()
        elev = f"{st['altitud_cne']:,.1f}"

        line = (
            f"| **{slots[st_id]}** | `{st_id}` | {name} | {regions[st_id]} | "
            f"{elev} | {n_total} | {cal_b[0]}–{cal_b[1]} ({n_cal}) | "
            f"{val_b[0]}–{val_b[1]} ({n_val}) | `{p_sha}` | `{tx_sha}` | `{tn_sha}` |"
        )
        lines.append(line)

    return "\n".join(lines)


if __name__ == "__main__":
    print(generate_table_f4())
