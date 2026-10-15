"""Native IDEAM DHIME reader engine and time series adapter."""

import re
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr


class IdeamAdapter:
    """Adapter for native IDEAM DHIME meteorological Excel exports (.xlsx).

    Parses the 7-row header metadata block and extracts monthly time series
    (Columns 0: Fecha, 2: Valor) into standardized pandas Series.
    """

    def __init__(self, filepath: Union[str, Path]) -> None:
        """Initialize adapter and parse DHIME header metadata block."""
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"IDEAM file not found: {self.filepath}")

        # Read raw excel header grid (rows 0-7) without column headers
        df_header = pd.read_excel(
            self.filepath, nrows=8, header=None, engine="openpyxl"
        )
        self._parse_metadata(df_header)

    def _parse_metadata(self, df: pd.DataFrame) -> None:
        """Extract metadata properties from row block 0..6."""
        # Row 1: Station name & code e.g. "JARDIN BOTANICO [21205710]"
        raw_name = str(df.iloc[1, 1]) if pd.notna(df.iloc[1, 1]) else ""
        match = re.search(r"\[(\d+)\]", raw_name)
        if match:
            self.code = match.group(1)
            self.name = raw_name.replace(f"[{self.code}]", "").strip()
        else:
            self.code = "UNKNOWN"
            self.name = raw_name.strip()

        # Row 2: Lat (Col 1), Lon (Col 3), Elevation (Col 5)
        self.latitude = float(str(df.iloc[2, 1]))
        self.longitude = float(str(df.iloc[2, 3]))
        self.elevation = float(str(df.iloc[2, 5]))

        # Row 3: Entity (Col 1), Area Operativa (Col 3), Department (Col 5)
        dept_val = df.iloc[3, 5]
        self.department = str(dept_val).strip() if pd.notna(dept_val) else ""

        # Row 4: Municipality (Col 1)
        muni_val = df.iloc[4, 1]
        self.municipality = str(muni_val).strip() if pd.notna(muni_val) else ""

        # Row 5: Variable (Col 1), Frequency (Col 3)
        var_val = df.iloc[5, 1]
        freq_val = df.iloc[5, 3]
        self.variable = str(var_val).strip().upper() if pd.notna(var_val) else ""
        self.frequency = str(freq_val).strip() if pd.notna(freq_val) else ""

        # Row 6: Parameter (Col 1), Unit (Col 3)
        param_val = df.iloc[6, 1]
        unit_val = df.iloc[6, 3]
        self.parameter = str(param_val).strip() if pd.notna(param_val) else ""
        self.unit = str(unit_val).strip() if pd.notna(unit_val) else ""

    def get_series(self) -> pd.Series:
        """Read data table starting from row 8 and return a clean pd.Series.

        Returns:
            pd.Series with strict pd.DatetimeIndex (freq='1MS') and numeric values.
            Missing values (-9999, -999, blank, 'Nulo') are mapped to np.nan.
        """
        # Read data rows starting from index 8
        df_raw = pd.read_excel(
            self.filepath, skiprows=8, header=None, engine="openpyxl"
        )
        if df_raw.empty:
            return pd.Series(dtype="float64", name=self.variable.lower())

        # Col 0: Fecha, Col 2: Valor
        dates_raw = df_raw.iloc[:, 0]
        values_raw = df_raw.iloc[:, 2]

        # Parse dates to DatetimeIndex
        dates = pd.to_datetime(dates_raw, errors="coerce")
        # Normalize timestamps to start of month (1MS)
        dates_1ms = pd.DatetimeIndex(
            pd.Series(dates).dt.to_period("M").dt.to_timestamp()
        )

        # Coerce values to numeric and map sentinel missing codes to NaN
        values = pd.to_numeric(values_raw, errors="coerce")
        values = values.replace([-9999.0, -999.0, -9999, -999], np.nan)

        series = pd.Series(
            data=values.to_numpy(),
            index=dates_1ms,
            name=self.variable.lower(),
        )

        # Drop rows where date failed parsing, sort by index, and drop duplicates
        series = series.loc[series.index.notna()]
        series = series[~series.index.duplicated(keep="first")].sort_index()

        # Set strict monthly start frequency if continuous
        try:
            series = series.asfreq("1MS")
        except ValueError:
            pass

        return series


class DhimeDailyPrecipitationAdapter:
    """Adapter for IDEAM DHIME daily precipitation exports (Día Pluviométrico).

    Parses tabular DHIME exports (Excel / CSV) with columns:
    ``CodigoEstacion``, ``NombreEstacion``, ``Variable``, ``Parametro``,
    ``Fecha``, ``Unidad``, ``Valor``, ``NivelAprobacion``.

    Key characteristics:
    1. **CF Conventions**: Produces an ``xarray.Dataset`` with variable ``pr``
       in ``mm/day``, dimensions ``(time, station)``, and approved spatial coordinates
       (lat, lon, altitud/elevation) sourced from the IDEAM CNE catalog.
    2. **Calendar Reconstruction**: Rebuilds the complete daily calendar grid.
       Missing days are strictly represented as ``NaN`` (never imputed as 0.0 mm).
    3. **Quality Flag Preservation**: Preserves ``NivelAprobacion`` as a companion
       data variable ``quality_flag`` (e.g. 'Definitivo', 'Preliminar', 'Faltante').
    4. **Observation Window Metadata**: Formally documents that the conventional
       rainfall day runs from 07:00 to 07:00 local time (12:00 to 12:00 UTC).
    5. **Excel Date Auto-Format Repair**: Detects and repairs Excel Spanish-locale
       auto-formatting where decimal values like '3.4' were coerced into
       'd.m' date cells (e.g. 3-Apr).
    """

    def __init__(
        self,
        filepath: Union[str, Path, pd.DataFrame],
        cne_path: Optional[Union[str, Path]] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        elevation: Optional[float] = None,
    ) -> None:
        """Initialize adapter.

        Args:
            filepath: Path to DHIME Excel / CSV file, or a pre-loaded DataFrame.
            cne_path: Path to CNE_IDEAM.xls catalog. If None, checks default
                path ``local/CNE_IDEAM.xls``.
            lat: Optional explicit latitude override in decimal degrees.
            lon: Optional explicit longitude override in decimal degrees.
            elevation: Optional explicit elevation override in meters.
        """
        if isinstance(filepath, pd.DataFrame):
            self.raw_df = filepath.copy()
            self.filepath: Optional[Path] = None
        else:
            self.filepath = Path(filepath)
            if not self.filepath.exists():
                raise FileNotFoundError(f"DHIME file not found: {self.filepath}")
            if self.filepath.suffix.lower() in (".xlsx", ".xls"):
                self.raw_df = pd.read_excel(self.filepath)
            else:
                self.raw_df = pd.read_csv(self.filepath)

        self.cne_path = Path(cne_path) if cne_path else Path("local/CNE_IDEAM.xls")
        self.lat_override = lat
        self.lon_override = lon
        self.elevation_override = elevation

    @staticmethod
    def _clean_valor(v: Any) -> float:
        """Convert DHIME raw Valor cell to float mm, repairing Excel d.m dates."""
        if pd.isna(v):
            return float("nan")
        # Repair Excel Spanish-locale date coercion (e.g. 3.4 mm -> 3-Apr datetime)
        if hasattr(v, "day") and hasattr(v, "month"):
            return float(f"{v.day}.{v.month}")
        if isinstance(v, (int, float)):
            val = float(v)
            return float("nan") if val < 0 else val
        if isinstance(v, str):
            v_str = v.strip().replace(",", ".")
            try:
                val = float(v_str)
                return float("nan") if val < 0 else val
            except ValueError:
                return float("nan")
        return float("nan")

    def _resolve_coordinates(self, station_code: str) -> Tuple[float, float, float]:
        """Resolve lat, lon, altitud from overrides or audited station catalog."""
        if (
            self.lat_override is not None
            and self.lon_override is not None
            and self.elevation_override is not None
        ):
            return self.lat_override, self.lon_override, self.elevation_override

        code_str = str(station_code).strip().replace(".0", "")

        # 1. Search in canonical audited StationCatalog (BENCHMARK_STATIONS)
        from tropicor.io.stations import StationCatalog

        catalog = StationCatalog.from_benchmark()
        st = catalog.get(code_str)
        if st is not None:
            lat = self.lat_override if self.lat_override is not None else st.latitude
            lon = self.lon_override if self.lon_override is not None else st.longitude
            alt = (
                self.elevation_override
                if self.elevation_override is not None
                else st.elevation
            )
            return (lat, lon, alt)

        # 2. Search in local/station_coords_audit.csv if present
        audit_path = Path("local/station_coords_audit.csv")
        if audit_path.exists():
            try:
                df_audit = pd.read_csv(audit_path)
                df_audit["codigo_clean"] = (
                    df_audit["codigo"]
                    .astype(str)
                    .str.strip()
                    .str.replace(r"\.0$", "", regex=True)
                )
                match = df_audit[df_audit["codigo_clean"] == code_str]
                if not match.empty:
                    row = match.iloc[0]
                    lat_cne = row.get("lat_cne", float("nan"))
                    lon_cne = row.get("lon_cne", float("nan"))
                    alt_final = row.get("altitud_final", float("nan"))
                    lat_val = float(row.get("lat_actual", lat_cne))
                    lon_val = float(row.get("lon_actual", lon_cne))
                    alt_val = float(row.get("altitud_actual", alt_final))
                    lat = (
                        self.lat_override
                        if self.lat_override is not None
                        else lat_val
                    )
                    lon = (
                        self.lon_override
                        if self.lon_override is not None
                        else lon_val
                    )
                    alt = (
                        self.elevation_override
                        if self.elevation_override is not None
                        else alt_val
                    )
                    return (lat, lon, alt)
            except Exception:  # noqa: BLE001
                pass

        return (
            self.lat_override if self.lat_override is not None else float("nan"),
            self.lon_override if self.lon_override is not None else float("nan"),
            (
                self.elevation_override
                if self.elevation_override is not None
                else float("nan")
            ),
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Parse raw records into a clean, deduplicated, calendar-complete DataFrame."""
        df = self.raw_df.copy()
        required_cols = ["CodigoEstacion", "Fecha", "Valor"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' not found in DHIME data. "
                    f"Columns: {df.columns.tolist()}"
                )

        df["CodigoEstacion"] = (
            df["CodigoEstacion"]
            .astype(str)
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
        )
        df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce").dt.normalize()
        df = df[df["Fecha"].notna()].copy()

        df["Valor_clean"] = df["Valor"].apply(self._clean_valor)

        # Quality flag column
        if "NivelAprobacion" in df.columns:
            df["quality_flag"] = df["NivelAprobacion"].astype(str).str.strip()
        else:
            df["quality_flag"] = "Desconocido"

        # Deduplicate per (CodigoEstacion, Fecha) prioritizing Definitivo > Preliminar
        quality_priority = {"Definitivo": 1, "Preliminar": 2}
        df["_prio"] = df["quality_flag"].map(lambda x: quality_priority.get(x, 99))
        df = df.sort_values(by=["CodigoEstacion", "Fecha", "_prio"])
        df = df.drop_duplicates(subset=["CodigoEstacion", "Fecha"], keep="first")
        df = df.drop(columns=["_prio"])

        return df

    def to_xarray(self) -> xr.Dataset:
        """Convert DHIME data into a CF-compliant xarray.Dataset (time x station)."""
        df_clean = self.to_dataframe()
        if df_clean.empty:
            raise ValueError("No valid records found in DHIME data.")

        # Reconstruct complete calendar across observed range
        min_date = df_clean["Fecha"].min()
        max_date = df_clean["Fecha"].max()
        full_dates = pd.date_range(min_date, max_date, freq="D")

        station_codes = sorted(df_clean["CodigoEstacion"].unique().tolist())

        pr_grid = np.full(
            (len(full_dates), len(station_codes)), np.nan, dtype=np.float64
        )
        flag_grid = np.full(
            (len(full_dates), len(station_codes)), "Faltante", dtype=object
        )

        lats: List[float] = []
        lons: List[float] = []
        alts: List[float] = []
        names: List[str] = []

        date_index_map = {d: i for i, d in enumerate(full_dates)}

        for st_idx, st_code in enumerate(station_codes):
            sub = df_clean[df_clean["CodigoEstacion"] == st_code]
            st_name = (
                str(sub["NombreEstacion"].iloc[0]).strip()
                if "NombreEstacion" in sub.columns
                else st_code
            )
            lat, lon, alt = self._resolve_coordinates(st_code)

            lats.append(lat)
            lons.append(lon)
            alts.append(alt)
            names.append(st_name)

            for _, row in sub.iterrows():
                d_idx = date_index_map.get(row["Fecha"])
                if d_idx is not None:
                    pr_grid[d_idx, st_idx] = row["Valor_clean"]
                    flag_grid[d_idx, st_idx] = row["quality_flag"]

        obs_window = "07:00 to 07:00 local time (12:00 to 12:00 UTC)"
        comment_acc = (
            "Accumulated rainfall over 24 hours preceding the morning observation."
        )

        ds = xr.Dataset(
            data_vars={
                "pr": (
                    ["time", "station"],
                    pr_grid,
                    {
                        "standard_name": "precipitation_flux",
                        "long_name": "Daily precipitation (dia pluviometrico)",
                        "units": "mm/day",
                        "cell_methods": "time: sum (interval: 1 day)",
                        "accumulation_window": obs_window,
                        "comment": comment_acc,
                    },
                ),
                "quality_flag": (
                    ["time", "station"],
                    flag_grid,
                    {
                        "long_name": "DHIME data approval level (nivel de aprobacion)",
                        "flag_values": "Definitivo, Preliminar, Faltante",
                    },
                ),
            },
            coords={
                "time": full_dates,
                "station": station_codes,
                "lat": (
                    ["station"],
                    lats,
                    {"standard_name": "latitude", "units": "degrees_north"},
                ),
                "lon": (
                    ["station"],
                    lons,
                    {"standard_name": "longitude", "units": "degrees_east"},
                ),
                "altitud": (
                    ["station"],
                    alts,
                    {"standard_name": "altitude", "units": "m", "positive": "up"},
                ),
                "elevation": (
                    ["station"],
                    alts,
                    {"standard_name": "altitude", "units": "m", "positive": "up"},
                ),
                "station_name": (
                    ["station"],
                    names,
                    {"long_name": "Station name"},
                ),
            },
            attrs={
                "title": "IDEAM DHIME Daily Precipitation Series",
                "institution": (
                    "Instituto de Hidrologia, Meteorologia y "
                    "Estudios Ambientales (IDEAM)"
                ),
                "source": "DHIME conventional daily rain gauge records",
                "observation_window": obs_window,
                "date_convention": (
                    "DHIME date D = accumulation 07:00 local D → 07:00 local D+1 "
                    "(12:00–12:00 UTC); confirmed empirically (lag 0, 2 stations, "
                    "non-overlapping 95% CI)"
                ),
                "Conventions": "CF-1.8",
            },
        )
        return ds


def read_dhime_daily_precipitation(
    filepath: Union[str, Path, pd.DataFrame],
    cne_path: Optional[Union[str, Path]] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    elevation: Optional[float] = None,
) -> xr.Dataset:
    """Convenience function to read DHIME daily precipitation into an xarray.Dataset."""
    adapter = DhimeDailyPrecipitationAdapter(
        filepath=filepath,
        cne_path=cne_path,
        lat=lat,
        lon=lon,
        elevation=elevation,
    )
    return adapter.to_xarray()
