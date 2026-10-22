"""Download hourly ERA5 total precipitation year by year for a specified area.

CLI features:
- One Copernicus CDS request per year (avoiding large multi-year rejections).
- Skips years already downloaded (allows stopping and resuming anytime).
- Safe file writes: downloads to temporary ``.part`` files, renamed upon success.
- Exponential backoff retries for failed requests.
- Multithreaded downloads with configurable workers.
- Credentials loaded safely from ``.env`` via python-dotenv / CDS format;
  secret keys are never printed or logged.
- Progress and errors appended to ``download.log``.
- ``--dry-run`` to inspect planned requests without contacting CDS.
- ``--verify`` to inspect downloaded NetCDF files via xarray.

Usage:
    uv run python scripts/download_era5_years.py --source single-levels --start-year 1975 \\
        --end-year 2025 --dry-run
    uv run python scripts/download_era5_years.py --source single-levels --start-year 1975 \\
        --end-year 2025 --verify
"""

from __future__ import annotations

import argparse
import calendar
import datetime
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import cdsapi  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import xarray as xr
from dotenv import load_dotenv

DEFAULT_DATASET = "reanalysis-era5-single-levels"
DEFAULT_TIMESERIES_DATASET = "reanalysis-era5-single-levels-timeseries"
DEFAULT_VARIABLE = "total_precipitation"
DEFAULT_BASE_OUT_DIR = Path("local/era5_hourly")
DEFAULT_BASE_TS_OUT_DIR = Path("local/era5_ts")
DEFAULT_WORKERS = 3
MAX_RETRIES = 3

VARIABLE_CDS_MAP = {
    "2m_temperature": "t2m",
    "total_precipitation": "tp",
}
VARIABLE_UNITS_MAP = {
    "2m_temperature": "K",
    "total_precipitation": "m",
}
VARIABLE_LONG_NAME_MAP = {
    "2m_temperature": "2 metre temperature",
    "total_precipitation": "Total precipitation",
}


def compute_nearest_grid_center(lat: float, lon: float) -> Tuple[float, float]:
    """Calculate nearest 0.25 deg ERA5 grid cell center according to Section 2.1.1."""
    lat_center = round(lat * 4.0) / 4.0
    lon_center = round(lon * 4.0) / 4.0
    return float(lat_center), float(lon_center)


def compute_station_area(station_code: str) -> Tuple[List[float], Any]:
    """Calculate 1.0x1.0 deg bounding box centered on station snapped to 0.25 ERA5 grid."""
    from tropicor.io.stations import StationCatalog

    catalog = StationCatalog()
    station = catalog.get(str(station_code).strip())
    if station is None:
        raise ValueError(f"Station '{station_code}' not found in catalog.")
    lat_center = round(station.latitude * 4.0) / 4.0
    lon_center = round(station.longitude * 4.0) / 4.0
    north = round(lat_center + 0.50, 2)
    south = round(lat_center - 0.50, 2)
    west = round(lon_center - 0.50, 2)
    east = round(lon_center + 0.50, 2)
    return [north, west, south, east], station


DEFAULT_STATIC_VARIABLES = ["geopotential", "land_sea_mask"]


ALLOWED_TIMESERIES_VARS: Set[str] = {"2m_temperature", "total_precipitation"}


def check_or_update_manifest(
    out_dir: Path,
    requested_area: Optional[List[float]] = None,
    station: Optional[str] = None,
    variable: Optional[str] = None,
    dataset: str = DEFAULT_DATASET,
    static_downloaded: bool = False,
    source: str = "single-levels",
    location: Optional[Dict[str, float]] = None,
    variables: Optional[List[str]] = None,
    write: bool = True,
) -> None:
    """Verifica que out_dir no contenga datos de otra área/punto y actualiza manifest.json."""
    import json

    manifest_path = out_dir / "manifest.json"
    manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if requested_area is not None:
            existing_area = manifest.get("area")
            if existing_area is not None:
                diffs = [abs(a - b) for a, b in zip(existing_area, requested_area)]
                if any(d > 0.01 for d in diffs):
                    raise RuntimeError(
                        f"Conflicto de área geográfica en '{out_dir}':\n"
                        f"El directorio ya contiene datos descargados para el área {existing_area}, "
                        f"pero se solicitó {requested_area}.\n"
                        f"Operación abortada para evitar sobrescribir archivos de otra área."
                    )
        if location is not None:
            existing_loc = manifest.get("location")
            if existing_loc is not None:
                d_lat = abs(
                    existing_loc.get("latitude", 0.0) - location.get("latitude", 0.0)
                )
                d_lon = abs(
                    existing_loc.get("longitude", 0.0) - location.get("longitude", 0.0)
                )
                if d_lat > 1e-4 or d_lon > 1e-4:
                    raise RuntimeError(
                        f"Conflicto de ubicación puntual en '{out_dir}':\n"
                        f"El directorio ya contiene datos descargados para la ubicación {existing_loc}, "
                        f"pero se solicitó {location}.\n"
                        f"Operación abortada para evitar sobrescribir archivos de otra ubicación."
                    )

    # Verificar si hay archivos .nc existentes con área incompatible
    if requested_area is not None:
        check_var = variable or DEFAULT_VARIABLE
        nc_files = list(out_dir.glob(f"era5_{check_var}_*.nc"))
        if not nc_files:
            nc_files = [
                f for f in out_dir.glob("era5_*.nc") if not f.name.endswith(".part")
            ]
        if nc_files and not manifest_path.exists():
            try:
                with xr.open_dataset(nc_files[0]) as ds_test:
                    lat_c = "latitude" if "latitude" in ds_test.coords else "lat"
                    lon_c = "longitude" if "longitude" in ds_test.coords else "lon"
                    if lat_c in ds_test.coords and lon_c in ds_test.coords:
                        n_e = float(ds_test[lat_c].max())
                        s_e = float(ds_test[lat_c].min())
                        w_e = float(ds_test[lon_c].min())
                        e_e = float(ds_test[lon_c].max())
                        if w_e > 180.0:
                            w_e = ((w_e + 180.0) % 360.0) - 180.0
                        if e_e > 180.0:
                            e_e = ((e_e + 180.0) % 360.0) - 180.0
                        exist_area = [n_e, w_e, s_e, e_e]
                        diffs = [abs(a - b) for a, b in zip(exist_area, requested_area)]
                        if any(d > 0.05 for d in diffs):
                            raise RuntimeError(
                                f"Conflicto de área geográfica en archivo '{nc_files[0]}':\n"
                                f"El archivo existente cubre {exist_area}, pero se solicitó {requested_area}.\n"
                                f"Operación abortada para evitar sobrescribir archivos de otra área."
                            )
            except Exception as exc:
                if isinstance(exc, RuntimeError):
                    raise

    out_dir.mkdir(parents=True, exist_ok=True)

    # Mantener lista acumulativa de variables descargadas en el directorio
    existing_vars = list(manifest.get("variables", []))
    if not existing_vars and "variable" in manifest and manifest["variable"]:
        existing_vars = [manifest["variable"]]

    if source == "single-levels":
        var_to_add = variable if variable is not None else DEFAULT_VARIABLE
        if var_to_add not in existing_vars:
            existing_vars.append(var_to_add)
    else:
        if variable and variable not in existing_vars:
            existing_vars.append(variable)
        if variables:
            for v in variables:
                if v and v not in existing_vars:
                    existing_vars.append(v)

    manifest_data = {
        "source": source,
        "station": station or manifest.get("station"),
        "area": requested_area if requested_area is not None else manifest.get("area"),
        "location": location if location is not None else manifest.get("location"),
        "variable": variable
        if variable is not None
        else (existing_vars[0] if existing_vars else None),
        "variables": existing_vars,
        "dataset": dataset,
        "updated_at": datetime.datetime.now().isoformat(),
    }
    if "static_file" in manifest:
        manifest_data["static_file"] = manifest["static_file"]
    if "static_variables" in manifest:
        manifest_data["static_variables"] = manifest["static_variables"]
    if static_downloaded or (out_dir / "era5_static.nc").exists():
        manifest_data["static_file"] = "era5_static.nc"
        manifest_data["static_variables"] = DEFAULT_STATIC_VARIABLES

    if not write:
        return

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)


MONTHS = [f"{m:02d}" for m in range(1, 13)]
DAYS = [f"{d:02d}" for d in range(1, 32)]
HOURS = [f"{h:02d}:00" for h in range(24)]


def load_cds_credentials(env_file: Path = Path(".env")) -> Tuple[str, str]:
    """Load CDSAPI_URL and CDSAPI_KEY from .env or environment variables.

    Supports both standard KEY=VALUE syntax and CDS .cdsapirc colon syntax
    (url: ... / key: ...). Validates presence without ever printing the secret key.

    Args:
        env_file: Path to the .env file.

    Returns:
        Tuple of (cds_url, cds_key).

    Raises:
        RuntimeError: If either credential is missing.
    """
    if env_file.exists():
        import logging

        logging.getLogger("dotenv.main").setLevel(logging.ERROR)
        load_dotenv(env_file)

    url: Optional[str] = os.environ.get("CDSAPI_URL") or os.environ.get("url")
    key: Optional[str] = os.environ.get("CDSAPI_KEY") or os.environ.get("key")

    # If python-dotenv could not parse colon-separated .cdsapirc style in .env:
    if (not url or not key) and env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str or line_str.startswith("#"):
                    continue
                if ":" in line_str:
                    parts = line_str.split(":", 1)
                    k = parts[0].strip().upper()
                    v = parts[1].strip()
                    if k in ("URL", "CDSAPI_URL") and not url:
                        url = v
                    elif k in ("KEY", "CDSAPI_KEY") and not key:
                        key = v
                elif "=" in line_str:
                    parts = line_str.split("=", 1)
                    k = parts[0].strip().upper()
                    v = parts[1].strip()
                    if k in ("URL", "CDSAPI_URL") and not url:
                        url = v
                    elif k in ("KEY", "CDSAPI_KEY") and not key:
                        key = v

    if not url or not key:
        missing = []
        if not url:
            missing.append("CDSAPI_URL (o 'url')")
        if not key:
            missing.append("CDSAPI_KEY (o 'key')")
        raise RuntimeError(
            f"Faltan credenciales de Copernicus CDS ({', '.join(missing)}). "
            f"Asegúrese de definir CDSAPI_URL y CDSAPI_KEY en {env_file}."
        )

    # Set in os.environ so cdsapi picks them up automatically
    os.environ["CDSAPI_URL"] = url
    os.environ["CDSAPI_KEY"] = key

    return url, key


def _log_event(log_file: Path, message: str) -> None:
    """Append a timestamped log line to the download log."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {message}\n")


def build_cds_request(
    year: int,
    variable: str = DEFAULT_VARIABLE,
    area: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Construct the CDS API request dictionary for a single year."""
    if area is None:
        raise ValueError("El parámetro 'area' [N, W, S, E] es obligatorio.")
    return {
        "product_type": ["reanalysis"],
        "variable": [variable],
        "year": [str(year)],
        "month": MONTHS,
        "day": DAYS,
        "time": HOURS,
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": area,
    }


def download_year(
    year: int,
    out_dir: Path,
    variable: str = DEFAULT_VARIABLE,
    dataset: str = DEFAULT_DATASET,
    area: Optional[List[float]] = None,
    log_file: Optional[Path] = None,
) -> str:
    """Download one year of hourly data from CDS, skipping if already present."""
    target = out_dir / f"era5_{variable}_{year}.nc"
    log_path = log_file or (out_dir / "download.log")

    if target.exists() and target.stat().st_size > 0:
        msg = f"{year}: ya descargado ({target.stat().st_size / 1e6:.1f} MB), omitido"
        _log_event(log_path, f"SKIP {year}: {target.name} ya existe")
        return msg

    request = build_cds_request(year, variable=variable, area=area)
    client = cdsapi.Client(quiet=True)

    _log_event(log_path, f"START {year}: solicitando descarga a CDS")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            tmp = target.with_suffix(".part")
            client.retrieve(dataset, request).download(str(tmp))
            tmp.rename(target)
            size_mb = target.stat().st_size / 1e6
            _log_event(
                log_path,
                f"SUCCESS {year}: guardado en {target.name} ({size_mb:.1f} MB)",
            )
            return f"{year}: OK ({size_mb:.1f} MB)"
        except Exception as exc:  # noqa: BLE001
            _log_event(
                log_path,
                f"RETRY {year} (intento {attempt}/{MAX_RETRIES}): "
                f"{type(exc).__name__}: {exc}",
            )
            if attempt == MAX_RETRIES:
                _log_event(
                    log_path,
                    f"FAILED {year}: fallo definitivo tras {MAX_RETRIES} intentos",
                )
                return f"{year}: FAILED tras {MAX_RETRIES} intentos ({exc})"
            time.sleep(30 * attempt)

    return f"{year}: FAILED"


def build_static_cds_request(
    area: Optional[List[float]] = None,
    variables: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Construct the CDS API request dictionary for static/invariant surface fields."""
    if area is None:
        raise ValueError("El parámetro 'area' [N, W, S, E] es obligatorio.")
    vars_to_req = variables or DEFAULT_STATIC_VARIABLES
    return {
        "product_type": ["reanalysis"],
        "variable": vars_to_req,
        "year": ["2020"],
        "month": ["01"],
        "day": ["01"],
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": area,
    }


def download_static(
    out_dir: Path,
    area: Optional[List[float]] = None,
    dataset: str = DEFAULT_DATASET,
    variables: Optional[List[str]] = None,
    log_file: Optional[Path] = None,
) -> str:
    """Download static/invariant fields (geopotential, land_sea_mask) from CDS."""
    if area is None:
        raise ValueError("El parámetro 'area' [N, W, S, E] es obligatorio.")
    target = out_dir / "era5_static.nc"
    log_path = log_file or (out_dir / "download.log")

    if target.exists() and target.stat().st_size > 0:
        msg = f"Campos estáticos: ya descargados ({target.stat().st_size / 1e6:.2f} MB), omitido"
        _log_event(log_path, f"SKIP STATIC: {target.name} ya existe")
        return msg

    request = build_static_cds_request(area, variables=variables)
    client = cdsapi.Client(quiet=True)

    _log_event(
        log_path,
        "START STATIC: solicitando campos estáticos (geopotential, land_sea_mask) a CDS",
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            tmp = target.with_suffix(".part")
            client.retrieve(dataset, request).download(str(tmp))
            tmp.rename(target)
            size_mb = target.stat().st_size / 1e6
            _log_event(
                log_path,
                f"SUCCESS STATIC: guardado en {target.name} ({size_mb:.2f} MB)",
            )
            return f"Static: OK ({size_mb:.2f} MB)"
        except Exception as exc:  # noqa: BLE001
            _log_event(
                log_path,
                f"RETRY STATIC (intento {attempt}/{MAX_RETRIES}): "
                f"{type(exc).__name__}: {exc}",
            )
            if attempt == MAX_RETRIES:
                _log_event(
                    log_path,
                    f"FAILED STATIC: fallo definitivo tras {MAX_RETRIES} intentos",
                )
                return f"Static: FAILED tras {MAX_RETRIES} intentos ({exc})"
            time.sleep(10 * attempt)

    return "Static: FAILED"


def run_verify(
    out_dir: Path,
    start_year: int,
    end_year: int,
    variable: str = DEFAULT_VARIABLE,
) -> None:
    """Verify downloaded NetCDF files via xarray."""
    print("=" * 88)
    print(f"VERIFICACIÓN DE ARCHIVOS ERA5: {out_dir} ({start_year} a {end_year})")
    print("=" * 88)

    years = range(start_year, end_year + 1)
    missing: List[int] = []
    valid: List[Dict[str, Any]] = []
    corrupted: List[Tuple[int, str]] = []

    for y in years:
        nc_path = out_dir / f"era5_{variable}_{y}.nc"
        if not nc_path.exists():
            missing.append(y)
            continue

        try:
            with xr.open_dataset(nc_path) as ds:
                n_hours = int(ds.sizes.get("time", ds.sizes.get("valid_time", 0)))
                is_leap = calendar.isleap(y)
                expected_hours = 8784 if is_leap else 8760

                # Coordinate bounds
                lat_c = "latitude" if "latitude" in ds.coords else "lat"
                lon_c = "longitude" if "longitude" in ds.coords else "lon"

                lat_min = float(ds[lat_c].min()) if lat_c in ds.coords else float("nan")
                lat_max = float(ds[lat_c].max()) if lat_c in ds.coords else float("nan")
                lon_min = float(ds[lon_c].min()) if lon_c in ds.coords else float("nan")
                lon_max = float(ds[lon_c].max()) if lon_c in ds.coords else float("nan")

                size_mb = nc_path.stat().st_size / 1e6
                status = (
                    "OK"
                    if n_hours == expected_hours
                    else f"WARN: {n_hours}/{expected_hours} h"
                )

                valid.append(
                    {
                        "year": y,
                        "hours": n_hours,
                        "expected": expected_hours,
                        "lat_range": (lat_min, lat_max),
                        "lon_range": (lon_min, lon_max),
                        "size_mb": size_mb,
                        "status": status,
                    }
                )
        except Exception as exc:  # noqa: BLE001
            corrupted.append((y, str(exc)))

    # Inspect static file if present
    static_nc = out_dir / "era5_static.nc"
    if static_nc.exists():
        try:
            with xr.open_dataset(static_nc) as ds_s:
                s_vars = list(ds_s.data_vars.keys())
                s_mb = static_nc.stat().st_size / 1e6
                print(
                    f"\nArchivo estático detectado: {static_nc.name} ({s_mb:.2f} MB), "
                    f"Variables: {s_vars}"
                )
        except Exception as exc:  # noqa: BLE001
            print(f"\nArchivo estático {static_nc.name} ilegible: {exc}")

    # Print summary
    print(f"\nTotal años en rango: {len(years)}")
    print(f"Archivos encontrados: {len(valid) + len(corrupted)}")
    print(f"Años faltantes: {len(missing)} ({missing if missing else 'Ninguno'})")
    if corrupted:
        print(f"Archivos corruptos/ilegibles: {len(corrupted)}")
        for y, err in corrupted:
            print(f"  - {y}: {err}")

    if valid:
        hdr = (
            f"\n{'Año':<6} {'Horas':<12} {'Esperadas':<10} "
            f"{'Latitud':<18} {'Longitud':<18} {'MB':<8} {'Estado'}"
        )
        print(hdr)
        print("-" * 88)
        for v in valid:
            lat_str = f"[{v['lat_range'][0]:.2f}, {v['lat_range'][1]:.2f}]"
            lon_str = f"[{v['lon_range'][0]:.2f}, {v['lon_range'][1]:.2f}]"
            print(
                f"{v['year']:<6} {v['hours']:<12} {v['expected']:<10} "
                f"{lat_str:<18} {lon_str:<18} {v['size_mb']:<8.1f} {v['status']}"
            )


def run_dry_run(
    start_year: int,
    end_year: int,
    area: List[float],
    out_dir: Path,
    workers: int,
    variable: str,
    dataset: str,
    url: str,
    download_static: bool = False,
    static_only: bool = False,
) -> None:
    """Print the detailed dry-run execution plan without contacting CDS."""
    years = list(range(start_year, end_year + 1))
    out_dir.mkdir(parents=True, exist_ok=True)

    already_done = [y for y in years if (out_dir / f"era5_{variable}_{y}.nc").exists()]
    to_download = [y for y in years if y not in already_done]

    print("=" * 88)
    print("PLAN DE EJECUCIÓN (DRY RUN) - DESCARGA HORARIA ERA5")
    print("=" * 88)
    print(f"CDS API URL:          {url}")
    print("CDS API Key:          [CONFIGURADA EN .env / NO IMPRESA]")
    print(f"Dataset CDS:          {dataset}")
    print(f"Variable:             {variable}")
    print(
        f"Área [N, W, S, E]:    {area} (N={area[0]}, W={area[1]}, "
        f"S={area[2]}, E={area[3]})"
    )
    if not static_only:
        print(f"Período:              {start_year} a {end_year} ({len(years)} años)")
    print(f"Directorio salida:    {out_dir}")
    print(f"Archivo de log:       {out_dir / 'download.log'}")
    if not static_only:
        print(f"Hilos paralelos:      {workers}")
        print(f"Archivos existentes:  {len(already_done)} años")
        print(f"Años por solicitar:   {len(to_download)} años")

    if download_static or static_only:
        static_file = out_dir / "era5_static.nc"
        st_status = "ya existe" if static_file.exists() else "por descargar"
        print(
            f"Campos estáticos:     {DEFAULT_STATIC_VARIABLES} -> {static_file.name} "
            f"({st_status})"
        )
        sample_static = build_static_cds_request(area=area)
        print("  Ejemplo de solicitud para campos estáticos:")
        for k, v in sample_static.items():
            print(f"    {k}: {v}")

    if not static_only:
        print("-" * 88)
        print(f"Ejemplo de solicitud para el primer año ({start_year}):")
        sample_req = build_cds_request(start_year, variable=variable, area=area)
        print(f"  Target: {out_dir / f'era5_{variable}_{start_year}.nc'}")
        for k, v in sample_req.items():
            if isinstance(v, list) and len(v) > 4:
                print(f"  {k}: [{v[0]}, {v[1]}, ..., {v[-1]}] ({len(v)} elementos)")
            else:
                print(f"  {k}: {v}")
    print("=" * 88)
    print("MODO DRY-RUN COMPLETADO: No se realizaron peticiones de red ni descargas.")


def build_timeseries_cds_request(
    lat_center: float,
    lon_center: float,
    start_year: int,
    end_year: int,
    variables: List[str],
) -> Dict[str, Any]:
    """Construct the CDS API request dictionary for point timeseries dataset."""
    return {
        "variable": list(variables),
        "location": {
            "latitude": float(lat_center),
            "longitude": float(lon_center),
        },
        "date": f"{start_year}-01-01/{end_year + 1}-01-01",
        "data_format": "netcdf",
    }


def verify_timeseries_dataset(
    ds: xr.Dataset,
    lat_center: float,
    lon_center: float,
    start_year: int,
    end_year: int,
    requested_variables: List[str],
) -> None:
    """Verifies coordinates, units, time index, zero NaNs, and unexpected dimensions."""
    if "latitude" not in ds.coords and "lat" not in ds.coords:
        raise ValueError("Returned dataset missing latitude coordinate.")
    if "longitude" not in ds.coords and "lon" not in ds.coords:
        raise ValueError("Returned dataset missing longitude coordinate.")

    lat_val = float(
        ds["latitude"].values if "latitude" in ds.coords else ds["lat"].values
    )
    lon_val = float(
        ds["longitude"].values if "longitude" in ds.coords else ds["lon"].values
    )

    if abs(lat_val - lat_center) > 1e-6:
        raise ValueError(
            f"Latitude mismatch: requested {lat_center}, returned {lat_val} (tolerance 1e-6)."
        )
    if abs(lon_val - lon_center) > 1e-6:
        raise ValueError(
            f"Longitude mismatch: requested {lon_center}, returned {lon_val} (tolerance 1e-6)."
        )

    tdim = (
        "valid_time"
        if "valid_time" in ds.dims
        else "time"
        if "time" in ds.dims
        else None
    )
    if tdim is None:
        raise ValueError(
            "No valid time dimension ('valid_time' or 'time') found in dataset."
        )

    allowed_dims = {tdim}
    for coord in ("latitude", "lat", "longitude", "lon"):
        if coord in ds.dims:
            allowed_dims.add(coord)
    unexpected_dims = set(ds.dims) - allowed_dims
    if unexpected_dims:
        unexpected_sorted = sorted(str(d) for d in unexpected_dims)
        raise ValueError(
            f"Unexpected dimensions {unexpected_sorted} found in CDS timeseries dataset; "
            "expected only time/latitude/longitude without silent collapse."
        )

    for c in ("expver", "number"):
        if c in ds.coords:
            c_val = ds[c].values
            if hasattr(c_val, "__len__") and len(c_val) > 1:
                raise ValueError(
                    f"Unexpected multi-valued coordinate '{c}' found in dataset: {c_val}"
                )

    time_arr = ds[tdim].values
    dt_index = pd.DatetimeIndex(time_arr)
    if dt_index.tz is not None:
        dt_index = dt_index.tz_convert(None)

    if dt_index.has_duplicates:
        raise ValueError("Returned time series contains duplicate timestamps.")

    expected_index = pd.date_range(
        start=f"{start_year}-01-01 00:00:00",
        end=f"{end_year + 1}-01-01 23:00:00",
        freq="h",
    )
    if len(dt_index) != len(expected_index) or not dt_index.equals(expected_index):
        missing_steps = expected_index.difference(dt_index)
        extra_steps = dt_index.difference(expected_index)
        raise ValueError(
            f"Time index mismatch: expected {len(expected_index)} hourly steps "
            f"from {expected_index[0]} to {expected_index[-1]}, but got {len(dt_index)}. "
            f"Missing: {len(missing_steps)}, Extra: {len(extra_steps)}."
        )

    for req_var in requested_variables:
        var_name = VARIABLE_CDS_MAP.get(req_var, req_var)
        if var_name not in ds.data_vars:
            raise ValueError(
                f"Expected variable '{var_name}' (for '{req_var}') not found in dataset: "
                f"{list(ds.data_vars.keys())}"
            )
        da = ds[var_name]
        expected_unit = VARIABLE_UNITS_MAP.get(req_var)
        actual_unit = str(da.attrs.get("units", "")).strip()
        if expected_unit and actual_unit != expected_unit:
            raise ValueError(
                f"Unit mismatch for variable '{var_name}': expected '{expected_unit}', "
                f"got '{actual_unit}'."
            )

        nan_count = int(da.isnull().sum().values)
        if nan_count > 0:
            raise ValueError(
                f"Found {nan_count} NaN values in variable '{var_name}'; zero NaNs required."
            )


def download_timeseries_netcdf(
    client: Any,
    dataset: str,
    request: Dict[str, Any],
    target_nc_path: Path,
    log_file: Optional[Path] = None,
) -> Path:
    """Download timeseries from CDS and unpack ZIP if packaged inside ZIP."""
    target_nc_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_download = target_nc_path.with_suffix(".tmp_download")

    if log_file:
        _log_event(log_file, f"START TIMESERIES: requesting {request['date']} from CDS")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            client.retrieve(dataset, request).download(str(tmp_download))
            if zipfile.is_zipfile(tmp_download):
                with zipfile.ZipFile(tmp_download, "r") as z:
                    nc_files = [n for n in z.namelist() if n.endswith(".nc")]
                    if not nc_files:
                        raise ValueError(
                            f"Downloaded ZIP did not contain any .nc files: {z.namelist()}"
                        )
                    extracted = z.extract(nc_files[0], path=str(target_nc_path.parent))
                tmp_download.unlink(missing_ok=True)
                Path(extracted).replace(target_nc_path)
            else:
                tmp_download.replace(target_nc_path)

            if log_file:
                _log_event(
                    log_file,
                    f"SUCCESS TIMESERIES: downloaded to {target_nc_path.name} "
                    f"({target_nc_path.stat().st_size / 1e6:.2f} MB)",
                )
            return target_nc_path
        except Exception as exc:  # noqa: BLE001
            if log_file:
                _log_event(
                    log_file,
                    f"RETRY TIMESERIES (intento {attempt}/{MAX_RETRIES}): "
                    f"{type(exc).__name__}: {exc}",
                )
            if attempt == MAX_RETRIES:
                if log_file:
                    _log_event(
                        log_file,
                        f"FAILED TIMESERIES: fallo definitivo tras {MAX_RETRIES} intentos",
                    )
                raise
            time.sleep(15 * attempt)

    return target_nc_path


def split_and_save_annual_files(
    ds: xr.Dataset,
    lat_center: float,
    lon_center: float,
    start_year: int,
    end_year: int,
    requested_variables: List[str],
    out_dir: Path,
    log_file: Optional[Path] = None,
) -> List[Path]:
    """Splits full timeseries into annual NetCDF files matching existing ERA5 scheme."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tdim = "valid_time" if "valid_time" in ds.dims else "time"
    created_files: List[Path] = []

    for req_var in requested_variables:
        var_name = VARIABLE_CDS_MAP.get(req_var, req_var)
        da_full = ds[var_name]

        years = list(range(start_year, end_year + 1))
        for y in years:
            t_start = f"{y}-01-01 00:00:00"
            t_end = f"{y}-12-31 23:00:00"
            da_year = da_full.sel({tdim: slice(t_start, t_end)})

            times = pd.DatetimeIndex(da_year[tdim].values)
            expected_hours = 8784 if calendar.isleap(y) else 8760
            if len(times) != expected_hours:
                raise ValueError(
                    f"Year {y}: expected {expected_hours} hours, got {len(times)}."
                )

            data_values = da_year.values.astype(np.float32).reshape(-1, 1, 1)
            ds_annual = xr.Dataset(
                data_vars={
                    var_name: (
                        ("time", "latitude", "longitude"),
                        data_values,
                        {
                            "units": VARIABLE_UNITS_MAP.get(
                                req_var, str(da_year.attrs.get("units", ""))
                            ),
                            "long_name": VARIABLE_LONG_NAME_MAP.get(
                                req_var, str(da_year.attrs.get("long_name", ""))
                            ),
                        },
                    )
                },
                coords={
                    "time": times,
                    "latitude": np.array([lat_center], dtype=np.float64),
                    "longitude": np.array([lon_center], dtype=np.float64),
                },
            )

            target_file = out_dir / f"era5_{req_var}_{y}.nc"
            tmp_file = target_file.with_suffix(".nc.part")
            ds_annual.to_netcdf(tmp_file)
            tmp_file.replace(target_file)
            created_files.append(target_file)
            if log_file:
                _log_event(
                    log_file,
                    f"SAVED {target_file.name} ({target_file.stat().st_size / 1e6:.2f} MB)",
                )

        # Boundary year (end_year + 1): 24h of Jan 1
        y_b = end_year + 1
        t_b_start = f"{y_b}-01-01 00:00:00"
        t_b_end = f"{y_b}-01-01 23:00:00"
        da_b = da_full.sel({tdim: slice(t_b_start, t_b_end)})
        times_b = pd.DatetimeIndex(da_b[tdim].values)
        if len(times_b) != 24:
            raise ValueError(
                f"Boundary year {y_b}: expected 24 hours of Jan 1, got {len(times_b)}."
            )

        data_b_values = da_b.values.astype(np.float32).reshape(-1, 1, 1)
        ds_boundary = xr.Dataset(
            data_vars={
                var_name: (
                    ("time", "latitude", "longitude"),
                    data_b_values,
                    {
                        "units": VARIABLE_UNITS_MAP.get(
                            req_var, str(da_b.attrs.get("units", ""))
                        ),
                        "long_name": VARIABLE_LONG_NAME_MAP.get(
                            req_var, str(da_b.attrs.get("long_name", ""))
                        ),
                    },
                )
            },
            coords={
                "time": times_b,
                "latitude": np.array([lat_center], dtype=np.float64),
                "longitude": np.array([lon_center], dtype=np.float64),
            },
        )
        target_b_file = out_dir / f"era5_{req_var}_{y_b}.nc"
        tmp_b_file = target_b_file.with_suffix(".nc.part")
        ds_boundary.to_netcdf(tmp_b_file)
        tmp_b_file.replace(target_b_file)
        created_files.append(target_b_file)
        if log_file:
            _log_event(
                log_file,
                f"SAVED {target_b_file.name} ({target_b_file.stat().st_size / 1e6:.2f} MB)",
            )

    return created_files


def run_timeseries_dry_run(
    station: str,
    lat: float,
    lon: float,
    lat_center: float,
    lon_center: float,
    variables: List[str],
    start_year: int,
    end_year: int,
    out_dir: Path,
    dataset: str,
    url: str,
) -> None:
    """Print the detailed dry-run execution plan for timeseries mode."""
    req = build_timeseries_cds_request(
        lat_center=lat_center,
        lon_center=lon_center,
        start_year=start_year,
        end_year=end_year,
        variables=variables,
    )
    years = list(range(start_year, end_year + 1))
    boundary_year = end_year + 1

    print("=" * 88)
    print("PLAN DE EJECUCIÓN (DRY RUN) - DESCARGA TIMESERIES ERA5 PUNTUAL")
    print("=" * 88)
    print(f"CDS API URL:          {url}")
    print("CDS API Key:          [CONFIGURADA EN .env / NO IMPRESA]")
    print(f"Dataset CDS:          {dataset}")
    print(f"Estación:             {station} (lat={lat:.6f}, lon={lon:.6f})")
    print(f"Celda 0.25° (Centro): lat={lat_center:.2f}, lon={lon_center:.2f}")
    print(f"Variables:            {variables}")
    print(
        f"Período:              {start_year} a {end_year} ({len(years)} años) "
        f"+ boundary {boundary_year}-01-01 (24 h)"
    )
    print(f"Directorio salida:    {out_dir}")
    print(f"Archivo de log:       {out_dir / 'download.log'}")
    print("-" * 88)
    print("Solicitud CDS (1 petición única para el rango completo):")
    for k, v in req.items():
        print(f"  {k}: {v}")
    print("-" * 88)
    print(f"Archivos anuales a generar en {out_dir}:")
    for var in variables:
        for y in years:
            print(f"  - era5_{var}_{y}.nc")
        print(f"  - era5_{var}_{boundary_year}.nc (solo 24 h del 1 de enero)")
    print("=" * 88)
    print("MODO DRY-RUN COMPLETADO: No se realizaron peticiones de red ni descargas.")


def run_timeseries(
    station: str,
    lat: float,
    lon: float,
    variables: List[str],
    start_year: int,
    end_year: int,
    out_dir: Optional[Path] = None,
    dataset: str = DEFAULT_TIMESERIES_DATASET,
    env_file: Path = Path(".env"),
    dry_run: bool = False,
    client: Optional[Any] = None,
) -> List[Path]:
    """Execute ERA5 timeseries point retrieval, verification, and annual splitting."""
    lat_center, lon_center = compute_nearest_grid_center(lat, lon)
    target_out_dir = out_dir or (DEFAULT_BASE_TS_OUT_DIR / str(station))

    url, _ = load_cds_credentials(env_file)
    if dry_run:
        run_timeseries_dry_run(
            station=station,
            lat=lat,
            lon=lon,
            lat_center=lat_center,
            lon_center=lon_center,
            variables=variables,
            start_year=start_year,
            end_year=end_year,
            out_dir=target_out_dir,
            dataset=dataset,
            url=url,
        )
        return []

    target_out_dir.mkdir(parents=True, exist_ok=True)
    log_file = target_out_dir / "download.log"

    # Validate compatibility with existing manifest if it exists
    if (target_out_dir / "manifest.json").exists():
        check_or_update_manifest(
            out_dir=target_out_dir,
            station=station,
            dataset=dataset,
            source="timeseries",
            location={"latitude": lat_center, "longitude": lon_center},
            variables=variables,
            write=False,
        )

    # Check if all annual files already exist and are valid
    all_exist = True
    for var in variables:
        for y in range(start_year, end_year + 2):
            f = target_out_dir / f"era5_{var}_{y}.nc"
            if not f.exists() or f.stat().st_size == 0:
                all_exist = False
                break
        if not all_exist:
            break

    if all_exist:
        msg = (
            f"Estación {station}: todos los archivos ({start_year}-{end_year + 1}) "
            f"ya existen en {target_out_dir}, omitiendo descarga."
        )
        print(msg)
        _log_event(log_file, f"SKIP: {msg}")
        existing_files = [
            target_out_dir / f"era5_{v}_{y}.nc"
            for v in variables
            for y in range(start_year, end_year + 2)
        ]
        return existing_files

    cds_client = client or cdsapi.Client(quiet=True)
    req = build_timeseries_cds_request(
        lat_center=lat_center,
        lon_center=lon_center,
        start_year=start_year,
        end_year=end_year,
        variables=variables,
    )

    print(
        f"Iniciando descarga timeseries puntual ({dataset}) para {station} "
        f"({lat_center}, {lon_center}) años {start_year}-{end_year}..."
    )
    raw_nc_path = target_out_dir / f"_raw_timeseries_{start_year}_{end_year}.nc"
    download_timeseries_netcdf(
        client=cds_client,
        dataset=dataset,
        request=req,
        target_nc_path=raw_nc_path,
        log_file=log_file,
    )

    try:
        with xr.open_dataset(raw_nc_path) as ds:
            print("Verificando consistencia del dataset devuelto...")
            verify_timeseries_dataset(
                ds=ds,
                lat_center=lat_center,
                lon_center=lon_center,
                start_year=start_year,
                end_year=end_year,
                requested_variables=variables,
            )
            print("Dataset verificado OK. Partiendo en archivos anuales...")
            created = split_and_save_annual_files(
                ds=ds,
                lat_center=lat_center,
                lon_center=lon_center,
                start_year=start_year,
                end_year=end_year,
                requested_variables=variables,
                out_dir=target_out_dir,
                log_file=log_file,
            )
            check_or_update_manifest(
                out_dir=target_out_dir,
                station=station,
                dataset=dataset,
                source="timeseries",
                location={"latitude": lat_center, "longitude": lon_center},
                variables=variables,
                write=True,
            )
    finally:
        raw_nc_path.unlink(missing_ok=True)

    print(
        f"Descarga y partición completada: {len(created)} archivos anuales "
        f"guardados en {target_out_dir}."
    )
    _log_event(log_file, f"SESSION COMPLETED: {len(created)} archivos creados")
    return created


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Descarga horaria de ERA5 por años individuales o timeseries puntual."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        choices=["single-levels", "timeseries"],
        help="Modo/fuente CDS: 'single-levels' (por área/año) o 'timeseries' (punto específico multi-año).",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=1975,
        help="Año inicial de la descarga (por defecto: 1975).",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2025,
        help="Año final de la descarga (por defecto: 2025).",
    )
    parser.add_argument(
        "--station",
        type=str,
        default=None,
        help=(
            "Código de la estación IDEAM (ej. 21205710, 23215030). "
            "En modo timeseries, nombre de la carpeta de salida."
        ),
    )
    parser.add_argument(
        "--lat",
        type=float,
        default=None,
        help="Latitud explícita de la estación para modo timeseries.",
    )
    parser.add_argument(
        "--lon",
        type=float,
        default=None,
        help="Longitud explícita de la estación para modo timeseries.",
    )
    parser.add_argument(
        "--area",
        type=float,
        nargs=4,
        default=None,
        metavar=("N", "W", "S", "E"),
        help="Límites geográficos Norte Oeste Sur Este para modo single-levels.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directorio de destino (por defecto: local/era5_hourly/<codigo>/ o local/era5_ts/<station>/).",
    )
    parser.add_argument(
        "--base-out-dir",
        type=Path,
        default=None,
        help="Directorio base para modo timeseries (por defecto: local/era5_ts).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Número de solicitudes concurrentes al CDS en modo single-levels (por defecto: 3).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=f"Nombre del dataset en CDS (por defecto: {DEFAULT_DATASET} o {DEFAULT_TIMESERIES_DATASET}).",
    )
    parser.add_argument(
        "--variable",
        nargs="+",
        default=None,
        help=f"Nombre de la(s) variable(s) en CDS (por defecto: {DEFAULT_VARIABLE}).",
    )
    parser.add_argument(
        "--download-static",
        "--static",
        dest="download_static",
        action="store_true",
        help="Descargar campos estáticos/invariantes (geopotential, land_sea_mask) para la estación.",
    )
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Descargar únicamente campos estáticos (era5_static.nc) sin descargar series anuales.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Ruta al archivo .env con credenciales CDS (por defecto: .env).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Muestra el plan de descarga sin realizar solicitudes al CDS.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "Abre los archivos .nc existentes con xarray y reporta completitud horaria."
        ),
    )
    return parser.parse_args(args)


def validate_cli_args(args: argparse.Namespace) -> None:
    """Validate CLI argument combinations for single-levels and timeseries modes."""
    if not args.static_only and args.start_year > args.end_year:
        raise ValueError(
            f"--start-year ({args.start_year}) no puede ser mayor que "
            f"--end-year ({args.end_year})."
        )

    if args.source == "timeseries":
        if args.download_static or args.static_only:
            raise ValueError(
                "Los argumentos --download-static y --static-only son exclusivos del modo single-levels "
                "y no están permitidos en modo timeseries."
            )
        if args.area is not None:
            raise ValueError(
                "El argumento --area es exclusivo del modo single-levels y no debe "
                "especificarse en modo timeseries."
            )
        if args.station is None:
            raise ValueError("--station es obligatorio en modo timeseries.")
        if args.lat is None:
            raise ValueError("--lat es obligatorio en modo timeseries.")
        if args.lon is None:
            raise ValueError("--lon es obligatorio en modo timeseries.")
        if not args.variable:
            raise ValueError(
                "--variable (ej. 2m_temperature y/o total_precipitation) es obligatorio en modo timeseries."
            )
        for v in args.variable:
            if v not in ALLOWED_TIMESERIES_VARS:
                raise ValueError(
                    f"Variable '{v}' no es válida para el modo timeseries. "
                    f"Valores permitidos: {sorted(ALLOWED_TIMESERIES_VARS)}."
                )

    elif args.source == "single-levels":
        if args.base_out_dir is not None:
            raise ValueError(
                "El argumento --base-out-dir es exclusivo del modo timeseries y no debe "
                "especificarse en modo single-levels."
            )
        if args.variable and len(args.variable) > 1:
            raise ValueError(
                f"El modo single-levels solo admite exactamente una variable por ejecución. "
                f"Se especificaron {len(args.variable)}: {args.variable}."
            )
        if args.lat is not None or args.lon is not None:
            raise ValueError(
                "Los argumentos --lat y --lon son exclusivos del modo timeseries "
                "y no deben especificarse en modo single-levels."
            )


def main() -> None:
    """Main CLI entrypoint."""
    args = parse_args()

    try:
        validate_cli_args(args)
    except ValueError as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)

    # =========================================================================
    # MODO 1: TIMESERIES PUNTUAL
    # =========================================================================
    if args.source == "timeseries":
        variables = args.variable
        dataset = args.dataset or DEFAULT_TIMESERIES_DATASET
        base_ts_dir = args.base_out_dir or DEFAULT_BASE_TS_OUT_DIR
        out_dir = args.out_dir or (base_ts_dir / str(args.station))

        if args.verify:
            for v in variables:
                run_verify(
                    out_dir=out_dir,
                    start_year=args.start_year,
                    end_year=args.end_year,
                    variable=v,
                )
            return

        run_timeseries(
            station=args.station,
            lat=args.lat,
            lon=args.lon,
            variables=variables,
            start_year=args.start_year,
            end_year=args.end_year,
            out_dir=out_dir,
            dataset=dataset,
            env_file=args.env_file,
            dry_run=args.dry_run,
        )
        return

    # =========================================================================
    # MODO 2: SINGLE-LEVELS (POR ÁREA / AÑO)
    # =========================================================================
    var_single = (
        args.variable[0]
        if (args.variable and len(args.variable) > 0)
        else DEFAULT_VARIABLE
    )
    dataset_sl = args.dataset or DEFAULT_DATASET

    # Resolver área y estación
    area = args.area
    station_code = args.station
    if area is None:
        if station_code is None:
            print(
                "Error: Debe especificarse --station (ej. --station 21205710) o --area N W S E.",
                file=sys.stderr,
            )
            sys.exit(1)
        try:
            area, st_meta = compute_station_area(station_code)
            print(
                f"Estación: {st_meta.name} [{st_meta.code}] -> Área calculada: {area}"
            )
        except Exception as err:
            print(
                f"Error al calcular área desde estación '{station_code}': {err}",
                file=sys.stderr,
            )
            sys.exit(1)

    # Resolver out_dir: por defecto local/era5_hourly/<codigo>/
    if args.out_dir is not None:
        out_dir = args.out_dir
    elif station_code is not None:
        out_dir = DEFAULT_BASE_OUT_DIR / str(station_code)
    else:
        out_dir = DEFAULT_BASE_OUT_DIR

    # 1. Modo de verificación (no requiere credenciales CDS)
    if args.verify:
        run_verify(
            out_dir=out_dir,
            start_year=args.start_year,
            end_year=args.end_year,
            variable=var_single,
        )
        return

    # 2. Verificar credenciales CDS de manera segura
    try:
        url, _ = load_cds_credentials(args.env_file)
    except RuntimeError as err:
        print(f"Error de autenticación: {err}", file=sys.stderr)
        sys.exit(1)

    # 3. Dry-run mode (no escribe manifest.json ni crea directorios)
    if args.dry_run:
        if out_dir.exists() and (out_dir / "manifest.json").exists():
            check_or_update_manifest(
                out_dir=out_dir,
                requested_area=area,
                station=station_code,
                variable=var_single,
                dataset=dataset_sl,
                static_downloaded=args.download_static or args.static_only,
                write=False,
            )
        run_dry_run(
            start_year=args.start_year,
            end_year=args.end_year,
            area=area,
            out_dir=out_dir,
            workers=args.workers,
            variable=var_single,
            dataset=dataset_sl,
            url=url,
            download_static=args.download_static,
            static_only=args.static_only,
        )
        return

    # Validar compatibilidad si el directorio y manifiesto ya existen
    if out_dir.exists() and (out_dir / "manifest.json").exists():
        try:
            check_or_update_manifest(
                out_dir=out_dir,
                requested_area=area,
                station=station_code,
                variable=var_single,
                dataset=dataset_sl,
                static_downloaded=args.download_static or args.static_only,
                write=False,
            )
        except RuntimeError as r_err:
            print(f"ERROR: {r_err}", file=sys.stderr)
            sys.exit(1)

    # 4. Modo de descarga real
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "download.log"

    # 4a. Descargar campos estáticos si fueron solicitados
    if args.download_static or args.static_only:
        print(f"Solicitando descarga de campos estáticos en {out_dir}...")
        static_result = download_static(
            out_dir=out_dir,
            area=area,
            dataset=dataset_sl,
            log_file=log_file,
        )
        print(static_result, flush=True)
        check_or_update_manifest(
            out_dir=out_dir,
            requested_area=area,
            station=station_code,
            variable=var_single,
            dataset=dataset_sl,
            static_downloaded=True,
            write=True,
        )
        if args.static_only:
            _log_event(log_file, "SESSION COMPLETED (STATIC ONLY)")
            return

    # 4b. Descargar series anuales
    years = list(range(args.start_year, args.end_year + 1))
    print(
        f"Iniciando descarga de {len(years)} años en {out_dir} "
        f"con {args.workers} hilos..."
    )
    _log_event(
        log_file,
        f"SESSION START: años {args.start_year} a {args.end_year}, "
        f"workers={args.workers}, variable={var_single}",
    )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                download_year,
                year=y,
                out_dir=out_dir,
                variable=var_single,
                dataset=dataset_sl,
                area=area,
                log_file=log_file,
            ): y
            for y in years
        }
        for future in as_completed(futures):
            print(future.result(), flush=True)

    # El manifiesto se escribe solo tras una descarga exitosa
    check_or_update_manifest(
        out_dir=out_dir,
        requested_area=area,
        station=station_code,
        variable=var_single,
        dataset=dataset_sl,
        static_downloaded=args.download_static or args.static_only,
        write=True,
    )

    _log_event(log_file, "SESSION COMPLETED")


if __name__ == "__main__":
    main()
