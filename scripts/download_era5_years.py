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
    uv run python scripts/download_era5_years.py --start-year 1975 \\
        --end-year 2025 --dry-run
    uv run python scripts/download_era5_years.py --start-year 1975 \\
        --end-year 2025 --verify
"""

from __future__ import annotations

import argparse
import calendar
import datetime
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cdsapi
import xarray as xr
from dotenv import load_dotenv

DEFAULT_DATASET = "reanalysis-era5-single-levels"
DEFAULT_VARIABLE = "total_precipitation"
DEFAULT_BASE_OUT_DIR = Path("local/era5_hourly")
DEFAULT_WORKERS = 3
MAX_RETRIES = 3


def compute_station_area(station_code: str) -> Tuple[List[float], Any]:
    """Calculate 1.0x1.0 deg bounding box centered on station snapped to 0.25 ERA5 grid."""
    from tropicor.io.stations import StationCatalog

    catalog = StationCatalog()
    station = catalog.get(str(station_code).strip())
    lat_center = round(station.latitude * 4.0) / 4.0
    lon_center = round(station.longitude * 4.0) / 4.0
    north = round(lat_center + 0.50, 2)
    south = round(lat_center - 0.50, 2)
    west = round(lon_center - 0.50, 2)
    east = round(lon_center + 0.50, 2)
    return [north, west, south, east], station


def check_or_update_manifest(
    out_dir: Path,
    requested_area: List[float],
    station: Optional[str] = None,
    variable: str = DEFAULT_VARIABLE,
    dataset: str = DEFAULT_DATASET,
) -> None:
    """Verifica que out_dir no contenga datos descargados de otra área y actualiza manifest.json."""
    import json

    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
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

    # Verificar si hay archivos .nc existentes con área incompatible
    nc_files = list(out_dir.glob(f"era5_{variable}_*.nc"))
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
    manifest_data = {
        "station": station,
        "area": requested_area,
        "variable": variable,
        "dataset": dataset,
        "updated_at": datetime.datetime.now().isoformat(),
    }
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

    # Print summary
    print(f"Total años en rango: {len(years)}")
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
    print(f"Período:              {start_year} a {end_year} ({len(years)} años)")
    print(f"Directorio salida:    {out_dir}")
    print(f"Archivo de log:       {out_dir / 'download.log'}")
    print(f"Hilos paralelos:      {workers}")
    print(f"Archivos existentes:  {len(already_done)} años")
    print(f"Años por solicitar:   {len(to_download)} años")
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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Descarga horaria de ERA5 por años individuales para un dominio espacial."
        )
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
        help="Código de la estación IDEAM (ej. 21205710, 23215030). Si no se pasa --area, se calcula automáticamente.",
    )
    parser.add_argument(
        "--area",
        type=float,
        nargs=4,
        default=None,
        metavar=("N", "W", "S", "E"),
        help="Límites geográficos Norte Oeste Sur Este. Si se omite, se calcula automáticamente desde --station.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directorio de destino (por defecto: local/era5_hourly/<codigo>/).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Número de solicitudes concurrentes al CDS (por defecto: 3).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help=f"Nombre del dataset en CDS (por defecto: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--variable",
        type=str,
        default=DEFAULT_VARIABLE,
        help=f"Nombre de la variable en CDS (por defecto: {DEFAULT_VARIABLE}).",
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
    return parser.parse_args()


def main() -> None:
    """Main CLI entrypoint."""
    args = parse_args()

    if args.start_year > args.end_year:
        print(
            f"Error: --start-year ({args.start_year}) no puede ser mayor que "
            f"--end-year ({args.end_year}).",
            file=sys.stderr,
        )
        sys.exit(1)

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
            print(f"Estación: {st_meta.name} [{st_meta.code}] -> Área calculada: {area}")
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
            variable=args.variable,
        )
        return

    # Validar que no se sobrescriban datos de otra área geográfica
    try:
        check_or_update_manifest(
            out_dir=out_dir,
            requested_area=area,
            station=station_code,
            variable=args.variable,
            dataset=args.dataset,
        )
    except RuntimeError as r_err:
        print(f"ERROR: {r_err}", file=sys.stderr)
        sys.exit(1)

    # 2. Verificar credenciales CDS de manera segura
    try:
        url, _ = load_cds_credentials(args.env_file)
    except RuntimeError as err:
        print(f"Error de autenticación: {err}", file=sys.stderr)
        sys.exit(1)

    # 3. Dry-run mode
    if args.dry_run:
        run_dry_run(
            start_year=args.start_year,
            end_year=args.end_year,
            area=area,
            out_dir=out_dir,
            workers=args.workers,
            variable=args.variable,
            dataset=args.dataset,
            url=url,
        )
        return

    # 4. Modo de descarga real
    out_dir.mkdir(parents=True, exist_ok=True)
    years = list(range(args.start_year, args.end_year + 1))
    log_file = out_dir / "download.log"
    print(
        f"Iniciando descarga de {len(years)} años en {out_dir} "
        f"con {args.workers} hilos..."
    )
    _log_event(
        log_file,
        f"SESSION START: años {args.start_year} a {args.end_year}, "
        f"workers={args.workers}",
    )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                download_year,
                year=y,
                out_dir=out_dir,
                variable=args.variable,
                dataset=args.dataset,
                area=area,
                log_file=log_file,
            ): y
            for y in years
        }
        for future in as_completed(futures):
            print(future.result(), flush=True)

    _log_event(log_file, "SESSION COMPLETED")


if __name__ == "__main__":
    main()
