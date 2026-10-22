"""Read-only data-driven registry of Andean stations (Protocol v1.4, App F).

Provides strictly validated metadata and temporal partition ranges for the Andean
evaluation set, guaranteeing:
- Validation partition integrity (strict block separation, cal_end < val_start).
- Data-driven configuration with zero embedded observation values.
- Strict ValueError guards against missing metadata keys or overlapping blocks.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Set, Tuple

from tropicor.core.station_selection import compute_optimal_partition

DEFAULT_ANDEAN_REGISTRY_PATH: Path = (
    Path(__file__).resolve().parent.parent / "data" / "andean_station_registry.json"
)

REQUIRED_REGISTRY_VERSION: str = "1.4"
ERA5_GRID_STEP_DEG: float = 0.25


def _is_grid_multiple(value: float, step: float = ERA5_GRID_STEP_DEG) -> bool:
    """Return True if value is an integer multiple of step (tolerance 1e-9)."""
    ratio = value / step
    return abs(ratio - round(ratio)) < 1e-9


REQUIRED_STATION_KEYS: Set[str] = {
    "codigo",
    "nombre",
    "latitud_cne",
    "longitud_cne",
    "altitud_cne",
    "celda_era5",
    "cal_block",
    "val_block",
    "valid_years",
}


@dataclass(frozen=True)
class AndeanStation:
    """Metadata and temporal partitions for an Andean generalization station."""

    codigo: str
    nombre: str
    latitud_cne: float
    longitud_cne: float
    altitud_cne: float
    celda_lat: float
    celda_lon: float
    cal_block: Tuple[int, int]
    val_block: Tuple[int, int]
    valid_years: Tuple[int, ...]

    @property
    def celda_era5(self) -> Dict[str, float]:
        """Return ERA5 grid cell as a dictionary with 'lat' and 'lon'."""
        return {"lat": self.celda_lat, "lon": self.celda_lon}


def load_andean_registry(
    registry_path: Optional[Path] = None,
) -> Dict[str, AndeanStation]:
    """Load and validate the Andean station registry from JSON.

    Args:
        registry_path: Optional custom path to registry JSON. If None, uses default.

    Returns:
        Mapping of clean station code to validated AndeanStation dataclass.

    Raises:
        FileNotFoundError: If the registry file does not exist.
        ValueError: If required keys missing, blocks overlap, or cal_end >= val_start.
    """
    path = registry_path or DEFAULT_ANDEAN_REGISTRY_PATH
    if not path.exists():
        raise FileNotFoundError(f"Andean station registry file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    version = data.get("version")
    if version != REQUIRED_REGISTRY_VERSION:
        raise ValueError(
            f"Registry version must be '{REQUIRED_REGISTRY_VERSION}', got {version!r}."
        )

    if "stations" not in data or not isinstance(data["stations"], dict):
        raise ValueError(
            "Registry JSON missing required top-level 'stations' dictionary."
        )

    stations: Dict[str, AndeanStation] = {}
    for raw_code, s_info in data["stations"].items():
        if not isinstance(s_info, dict):
            raise ValueError(f"Station entry for '{raw_code}' must be a dictionary.")

        missing_keys = REQUIRED_STATION_KEYS - set(s_info.keys())
        if missing_keys:
            raise ValueError(
                f"Station '{raw_code}' is missing required keys: {sorted(missing_keys)}"
            )

        code = re.sub(r"\.0$", "", str(s_info["codigo"]).strip())
        name = str(s_info["nombre"]).strip()

        try:
            lat_cne = float(s_info["latitud_cne"])
            lon_cne = float(s_info["longitud_cne"])
            alt_cne = float(s_info["altitud_cne"])
        except (TypeError, ValueError) as err:
            raise ValueError(
                f"Station '{code}' coordinates or altitude must be numeric: {err}"
            ) from err

        celda = s_info["celda_era5"]
        if not isinstance(celda, dict) or "lat" not in celda or "lon" not in celda:
            raise ValueError(
                f"Station '{code}' celda_era5 must be a dict with 'lat' and 'lon'."
            )
        try:
            celda_lat = float(celda["lat"])
            celda_lon = float(celda["lon"])
        except (TypeError, ValueError) as err:
            raise ValueError(
                f"Station '{code}' celda_era5 coordinates must be numeric: {err}"
            ) from err
        if not (_is_grid_multiple(celda_lat) and _is_grid_multiple(celda_lon)):
            raise ValueError(
                f"Station '{code}' celda_era5 ({celda_lat}, {celda_lon}) is not a "
                f"multiple of {ERA5_GRID_STEP_DEG} degrees."
            )

        # (e) Nearest grid node to CNE: round(coord * 4) / 4 (tolerance 1e-9)
        expected_lat = round(lat_cne * 4) / 4
        expected_lon = round(lon_cne * 4) / 4
        if abs(celda_lat - expected_lat) > 1e-9 or abs(celda_lon - expected_lon) > 1e-9:
            raise ValueError(
                f"Station '{code}' celda_era5 ({celda_lat}, {celda_lon}) is not "
                f"the nearest 0.25-degree grid node to CNE coordinates "
                f"({lat_cne}, {lon_cne}): expected ({expected_lat}, {expected_lon})."
            )

        cal_b = s_info["cal_block"]
        val_b = s_info["val_block"]
        if not (isinstance(cal_b, Sequence) and len(cal_b) == 2):
            raise ValueError(
                f"Station '{code}' 'cal_block' must be a sequence of 2 years."
            )
        if not (isinstance(val_b, Sequence) and len(val_b) == 2):
            raise ValueError(
                f"Station '{code}' 'val_block' must be a sequence of 2 years."
            )

        cal_start, cal_end = int(cal_b[0]), int(cal_b[1])
        val_start, val_end = int(val_b[0]), int(val_b[1])

        if cal_start > cal_end:
            raise ValueError(
                f"Station '{code}': invalid calibration block [{cal_start}, {cal_end}]."
            )
        if val_start > val_end:
            raise ValueError(
                f"Station '{code}': invalid validation block [{val_start}, {val_end}]."
            )

        # Strict ordering and non-overlap verification
        if cal_end >= val_start:
            raise ValueError(
                f"Station '{code}': calibration end ({cal_end}) must be strictly less "
                f"than validation start ({val_start})."
            )

        # Check for overlapping ranges
        if max(cal_start, val_start) <= min(cal_end, val_end):
            raise ValueError(
                f"Station '{code}': overlapping calibration [{cal_start}, {cal_end}] "
                f"and validation [{val_start}, {val_end}] blocks."
            )

        raw_years = s_info["valid_years"]
        if not isinstance(raw_years, Sequence):
            raise ValueError(f"Station '{code}' 'valid_years' must be a list/sequence.")
        valid_years = tuple(sorted(int(y) for y in raw_years))
        vy_set = set(valid_years)

        # (b) Block endpoints must be valid years; cal_end = last valid year
        # before the cut (val_start).
        if cal_start not in vy_set or val_end not in vy_set:
            raise ValueError(
                f"Station '{code}': cal_block start ({cal_start}) and val_block end "
                f"({val_end}) must both be in valid_years."
            )
        before_cut = [y for y in valid_years if y < val_start]
        if not before_cut or cal_end != before_cut[-1]:
            raise ValueError(
                f"Station '{code}': cal_block end ({cal_end}) must be the last valid "
                f"year before validation start ({val_start})."
            )

        # (a) Blocks must equal the protocol partition of valid_years.
        expected = compute_optimal_partition(valid_years)
        if expected["cal_block"] != [cal_start, cal_end] or expected["val_block"] != [
            val_start,
            val_end,
        ]:
            raise ValueError(
                f"Station '{code}': blocks cal={[cal_start, cal_end]} "
                f"val={[val_start, val_end]} do not match "
                f"compute_optimal_partition(valid_years): "
                f"cal={expected['cal_block']} val={expected['val_block']}."
            )

        station_obj = AndeanStation(
            codigo=code,
            nombre=name,
            latitud_cne=lat_cne,
            longitud_cne=lon_cne,
            altitud_cne=alt_cne,
            celda_lat=celda_lat,
            celda_lon=celda_lon,
            cal_block=(cal_start, cal_end),
            val_block=(val_start, val_end),
            valid_years=valid_years,
        )
        stations[code] = station_obj

    return stations


def get_andean_station(
    station_id: str,
    registry_path: Optional[Path] = None,
) -> AndeanStation:
    """Retrieve an Andean station by its station code.

    Args:
        station_id: Station identifier (e.g. '37015020').
        registry_path: Optional custom path to registry JSON.

    Returns:
        AndeanStation dataclass.

    Raises:
        ValueError: If station_id is not in the Andean registry.
    """
    clean_id = re.sub(r"\.0$", "", str(station_id).strip())
    stations = load_andean_registry(registry_path=registry_path)
    if clean_id not in stations:
        raise ValueError(f"Station '{clean_id}' not found in Andean registry.")
    return stations[clean_id]


def get_andean_partition(
    variable: str,
    station_id: str,
    registry_path: Optional[Path] = None,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Retrieve pre-specified cal and val partitions for an Andean station.

    Args:
        variable: Climate variable ('temperature').
        station_id: Station identifier.
        registry_path: Optional custom path to registry JSON.

    Returns:
        Tuple of ((cal_start, cal_end), (val_start, val_end)).

    Raises:
        ValueError: If variable is not 'temperature' or station is not registered.
    """
    if variable.lower() != "temperature":
        raise ValueError(
            f"Variable '{variable}' not currently supported for Andean registry "
            "(only 'temperature' registered in Protocol v1.4)."
        )
    st = get_andean_station(station_id, registry_path=registry_path)
    return st.cal_block, st.val_block


def request_andean_validation_partition(station_id: str) -> None:
    """Attempting to request validation data for Andean stations raises ValueError.

    Under Protocol v1.4 (Appendix F), validation partitions for Andean generalization
    stations are strictly closed to prevent data snooping prior to Phase 3.
    """
    clean_id = re.sub(r"\.0$", "", str(station_id).strip())
    raise ValueError(
        f"Validation partition for Andean station '{clean_id}' is strictly closed. "
        "Access prohibited prior to Phase 3 evaluation (Protocol v1.4 Appendix F.7)."
    )
