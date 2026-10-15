"""ERA5 global atmospheric reanalysis NetCDF reader and point extraction engine."""

import importlib.util
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import pandas as pd
import xarray as xr

from tropicor.io.stations import StationMetadata

STANDARD_GRAVITY: float = 9.80665


@dataclass(frozen=True)
class VariableAggregationConvention:
    """Temporal aggregation convention for a meteorological variable."""

    variable: str
    window_start_utc: int
    lag_days: int
    description: str
    date_convention: str
    scope: str


VARIABLE_CONVENTIONS: Dict[str, VariableAggregationConvention] = {
    "precipitation": VariableAggregationConvention(
        variable="precipitation",
        window_start_utc=12,
        lag_days=0,
        description="12:00–12:00 UTC (07:00–07:00 local time), Lag 0",
        date_convention=(
            "DHIME date D = accumulation 07:00 local D → 07:00 local D+1 "
            "(12:00–12:00 UTC); confirmed empirically (lag 0, 2 stations, "
            "non-overlapping 95% CI)"
        ),
        scope="2 stations (21205710 1975–2015, 23215030 1975–2000)",
    ),
    "temperature": VariableAggregationConvention(
        variable="temperature",
        window_start_utc=5,
        lag_days=0,
        description="05:00–05:00 UTC (00:00–24:00 local time, local civil day), Lag 0",
        date_convention=(
            "DHIME date D = extreme temperatures 00:00–24:00 local D "
            "(05:00 UTC D → 05:00 UTC D+1); confirmed empirically (lag 0, "
            "2 stations, 1990–1994; paired |Δr| <= 0.0057 at lag 0 full year, "
            "05–05 UTC maintained as local civil day default)"
        ),
        scope=(
            "2 stations (21205710, 23215030), 1990–1994; "
            "paired |Δr| <= 0.0057 at lag 0 full year"
        ),
    ),
}


def get_variable_convention(variable: str) -> VariableAggregationConvention:
    """Retrieve temporal aggregation convention for a meteorological variable.

    Args:
        variable: Variable name or alias (e.g. 'precipitation', 'tp',
            'temperature', 't2m').

    Returns:
        VariableAggregationConvention with window_start_utc, lag_days,
        date_convention, and scope.
    """
    key = variable.lower().strip()
    if "precip" in key or key in ("tp", "rain", "pr"):
        return VARIABLE_CONVENTIONS["precipitation"]
    if "temp" in key or key in ("t2m", "tmax", "tmin"):
        return VARIABLE_CONVENTIONS["temperature"]
    raise KeyError(
        f"No temporal convention configured for variable '{variable}'. "
        f"Available: {list(VARIABLE_CONVENTIONS.keys())}"
    )


class ERA5Adapter:
    """Adapter for ingesting and spatially querying ERA5 NetCDF reanalysis datasets."""

    def __init__(
        self,
        filepath_or_pattern: Union[str, Path, Sequence[Union[str, Path]]],
        chunks: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize ERA5Adapter and sanitize dimensions and coordinates.

        Args:
            filepath_or_pattern: Path to NetCDF file, glob pattern, or list of paths.
            chunks: Optional Dask chunking dictionary for out-of-core evaluation.
        """
        open_kwargs: Dict[str, Any] = {}
        if chunks is not None:
            open_kwargs["chunks"] = chunks

        # Detect GRIB format and verify optional dependencies
        is_grib = False
        if isinstance(filepath_or_pattern, (str, Path)):
            path_str = str(filepath_or_pattern).lower()
            if any(path_str.endswith(ext) for ext in [".grib", ".grb", ".grib2"]):
                is_grib = True
        elif isinstance(filepath_or_pattern, Sequence) and len(filepath_or_pattern) > 0:
            first_path = str(filepath_or_pattern[0]).lower()
            if any(first_path.endswith(ext) for ext in [".grib", ".grb", ".grib2"]):
                is_grib = True

        if is_grib:
            if importlib.util.find_spec("cfgrib") is None:
                raise ImportError(
                    "GRIB format support requires the 'cfgrib' package. "
                    "Install it with: pip install 'tropicor[grib]'. "
                    "For optimal speed and stability, convert to NetCDF (.nc)."
                )
            open_kwargs["engine"] = "cfgrib"

        if isinstance(filepath_or_pattern, (str, Path)):
            path_str = str(filepath_or_pattern)
            # Check if glob pattern or single file
            if any(char in path_str for char in ["*", "?", "["]):
                try:
                    self._ds = xr.open_mfdataset(
                        path_str, combine="by_coords", **open_kwargs
                    )
                except ImportError as exc:
                    if "dask" in str(exc).lower():
                        warnings.warn(
                            "Dask is not installed. Multi-file ERA5 dataset is "
                            "being loaded sequentially into memory. For faster "
                            "chunked or parallel loading, install: "
                            "pip install 'tropicor[parallel]'",
                            UserWarning,
                            stacklevel=2,
                        )
                        import glob

                        matched = sorted(glob.glob(path_str))
                        if not matched:
                            raise FileNotFoundError(
                                f"No files matched pattern: {path_str}"
                            ) from exc
                        datasets = [xr.open_dataset(f, **open_kwargs) for f in matched]
                        tdim = (
                            "valid_time"
                            if "valid_time" in datasets[0].dims
                            else "time"
                        )
                        self._ds = xr.concat(datasets, dim=tdim)
                    else:
                        raise
            else:
                p = Path(path_str)
                if not p.exists():
                    raise FileNotFoundError(f"ERA5 file not found: {p}")
                self._ds = xr.open_dataset(p, **open_kwargs)
        else:
            try:
                self._ds = xr.open_mfdataset(
                    [str(f) for f in filepath_or_pattern],
                    combine="by_coords",
                    **open_kwargs,
                )
            except ImportError as exc:
                if "dask" in str(exc).lower():
                    warnings.warn(
                        "Dask is not installed. Multi-file ERA5 dataset is "
                        "being loaded sequentially into memory. For faster "
                        "chunked or parallel loading, install: "
                        "pip install 'tropicor[parallel]'",
                        UserWarning,
                        stacklevel=2,
                    )
                    datasets = [
                        xr.open_dataset(str(f), **open_kwargs)
                        for f in filepath_or_pattern
                    ]
                    tdim = (
                        "valid_time" if "valid_time" in datasets[0].dims else "time"
                    )
                    self._ds = xr.concat(datasets, dim=tdim)
                else:
                    raise

        self._sanitize_dataset()

    def _sanitize_dataset(self) -> None:
        """Normalize coordinates, Copernicus CDS dimensions, and coordinate sorting."""
        ds = self._ds

        # 1. Copernicus CDS Time Harmonization: valid_time -> time
        if "valid_time" in ds.coords and "time" not in ds.coords:
            ds = ds.rename({"valid_time": "time"})

        # 2. Copernicus CDS Expver Resolution: select consolidated expver=1
        if "expver" in ds.dims or "expver" in ds.coords:
            if "expver" in ds.dims:
                if 1 in ds["expver"].values:
                    ds = ds.sel(expver=1)
                else:
                    ds = ds.isel(expver=0)
            if "expver" in ds.coords:
                ds = ds.drop_vars("expver", errors="ignore")

        # 3. Coordinate name standardization
        rename_map = {}
        if "lat" in ds.coords and "latitude" not in ds.coords:
            rename_map["lat"] = "latitude"
        if "lon" in ds.coords and "longitude" not in ds.coords:
            rename_map["lon"] = "longitude"
        if rename_map:
            ds = ds.rename(rename_map)

        # 4. Longitude normalization: [0, 360] -> [-180, 180]
        if "longitude" in ds.coords:
            lons = ds["longitude"]
            if (lons > 180.0).any():
                norm_lons = ((lons + 180.0) % 360.0) - 180.0
                ds = ds.assign_coords(longitude=norm_lons)

        # 5. Monotonic sorting of spatial coordinates
        sort_dims = [dim for dim in ["latitude", "longitude"] if dim in ds.coords]
        if sort_dims:
            ds = ds.sortby(sort_dims)

        self._ds = ds

    @property
    def dataset(self) -> xr.Dataset:
        """Return the underlying sanitized xarray Dataset."""
        return self._ds

    def extract_point(
        self,
        latitude: float,
        longitude: float,
        method: str = "nearest",
    ) -> xr.Dataset:
        """Extract spatial point slice from dataset.

        Args:
            latitude: Target latitude in decimal degrees [-90, 90].
            longitude: Target longitude in decimal degrees [-180, 180].
            method: Interpolation method: 'nearest' or 'bilinear' ('linear').

        Returns:
            xarray.Dataset containing sliced variables.
        """
        valid_methods = {"nearest", "bilinear", "linear"}
        if method not in valid_methods:
            raise ValueError(
                f"Invalid extraction method '{method}'. Must be one of {valid_methods}."
            )

        if method in {"bilinear", "linear"}:
            return self._ds.interp(
                latitude=latitude, longitude=longitude, method="linear"
            )
        return self._ds.sel(latitude=latitude, longitude=longitude, method="nearest")

    def _resolve_variable_name(self, var: str) -> str:
        """Resolve requested variable name against dataset variables."""
        if var in self._ds.data_vars:
            return var

        var_lower = var.lower()
        candidates: List[str] = []
        if var_lower in {
            "t2m",
            "2m_temperature",
            "temp",
            "temperature",
            "temperatura",
        }:
            candidates = ["t2m", "2t", "2m_temperature", "temperature"]
        elif var_lower in {
            "tp",
            "precip",
            "precipitation",
            "precipitacion",
            "total_precipitation",
        }:
            candidates = ["tp", "total_precipitation", "precipitation"]
        elif var_lower in {"z", "geopotential", "elevation"}:
            candidates = ["z", "geopotential"]
        elif var_lower in {"lsm", "land_sea_mask", "mask"}:
            candidates = ["lsm", "land_sea_mask"]

        for cand in candidates:
            if cand in self._ds.data_vars:
                return cand

        avail_vars = list(self._ds.data_vars.keys())
        raise KeyError(
            f"Variable '{var}' not found in dataset. Available: {avail_vars}"
        )

    def get_series(
        self,
        latitude: float,
        longitude: float,
        variable: str,
        method: str = "nearest",
    ) -> pd.Series:
        """Extract a single time series at given coordinates with unit conversion.

        Args:
            latitude: Latitude coordinate.
            longitude: Longitude coordinate.
            variable: Variable name (e.g. 't2m', 'tp', 'temperature', 'precipitation').
            method: Spatial extraction method ('nearest' or 'bilinear').

        Returns:
            pd.Series with DatetimeIndex normalized to Start-of-Month (1MS).
        """
        var_name = self._resolve_variable_name(variable)
        ds_pt = self.extract_point(latitude, longitude, method=method)
        da = ds_pt[var_name]

        # Deterministic unit conversion based strictly on NetCDF metadata attribute
        units = da.attrs.get("units")
        if units is None:
            warnings.warn(
                f"Variable '{var_name}' in NetCDF dataset has missing 'units' "
                "attribute. Passing values through unconverted.",
                UserWarning,
                stacklevel=2,
            )
        else:
            units_str = str(units).strip().lower()
            # Temperature conversions
            if (
                var_name in {"t2m", "2t", "2m_temperature"}
                or "temp" in variable.lower()
            ):
                if units_str in {"k", "kelvin"}:
                    da = da - 273.15
                elif units_str in {"c", "°c", "celsius"}:
                    pass
                else:
                    warnings.warn(
                        f"Unrecognized temperature unit '{units}' for variable "
                        f"'{var_name}'. Passing values through unconverted.",
                        UserWarning,
                        stacklevel=2,
                    )
            # Precipitation conversions
            elif var_name == "tp" or "precip" in variable.lower():
                if units_str in {"m", "meter", "meters", "metre", "metres"}:
                    da = da * 1000.0
                elif units_str in {
                    "mm",
                    "millimeter",
                    "millimeters",
                    "millimetre",
                    "millimetres",
                }:
                    pass
                else:
                    warnings.warn(
                        f"Unrecognized precipitation unit '{units}' for variable "
                        f"'{var_name}'. Passing values through unconverted.",
                        UserWarning,
                        stacklevel=2,
                    )

        # Reduce auxiliary dimensions (e.g. GRIB steps, ensemble members,
        # surface levels)
        if "step" in da.dims:
            da = da.sum(dim="step")
        if "surface" in da.dims:
            da = da.squeeze("surface")
        if "number" in da.dims:
            da = da.isel(number=0)

        # Convert to pandas Series
        series_raw = da.to_series()
        time_index = pd.DatetimeIndex(series_raw.index)

        # Detect if series is sub-monthly (multiple timestamps per month)
        is_sub_monthly = len(time_index) > len(time_index.to_period("M").unique())

        if is_sub_monthly:
            if var_name == "tp" or "precip" in variable.lower():
                monthly = series_raw.resample("1MS").sum()
            else:
                monthly = series_raw.resample("1MS").mean()
            monthly.name = variable.lower()
            return monthly.sort_index()

        time_index_1ms = time_index.to_period("M").to_timestamp()
        series = pd.Series(
            data=series_raw.values,
            index=time_index_1ms,
            name=variable.lower(),
        )
        return series[~series.index.duplicated(keep="first")].sort_index()

    def get_point_series(
        self,
        latitude: float,
        longitude: float,
        variable: str,
        method: str = "nearest",
    ) -> pd.Series:
        """Extract a single time series at given coordinates with unit conversion.

        Ergonomic alias for :meth:`get_series`.

        Args:
            latitude: Latitude coordinate.
            longitude: Longitude coordinate.
            variable: Variable name (e.g. 't2m', 'tp', 'temperature', 'precipitation').
            method: Spatial extraction method ('nearest' or 'bilinear').

        Returns:
            pd.Series with DatetimeIndex normalized to Start-of-Month (1MS).
        """
        return self.get_series(
            latitude=latitude,
            longitude=longitude,
            variable=variable,
            method=method,
        )

    def get_station_series(
        self,
        station: StationMetadata,
        variable: str,
        method: str = "nearest",
    ) -> pd.Series:
        """Extract time series collocated at given station coordinates."""
        return self.get_series(
            latitude=station.latitude,
            longitude=station.longitude,
            variable=variable,
            method=method,
        )

    def get_daily_precipitation(
        self,
        latitude: float,
        longitude: float,
        method: str = "nearest",
        window_start_utc: int = 12,
        min_hours: int = 24,
        lag_days: int = 0,
    ) -> pd.Series:
        """Extract and aggregate daily precipitation at given coordinates.

        Accumulates hourly precipitation according to the WMO / IDEAM Día Pluviométrico
        convention (default: 12:00 UTC day D to 12:00 UTC day D+1, mapped to day D).

        Args:
            latitude: Target latitude in decimal degrees.
            longitude: Target longitude in decimal degrees.
            method: Spatial extraction method ('nearest' or 'bilinear').
            window_start_utc: Starting hour of the 24h accumulation window in UTC
                (default 12).
            min_hours: Minimum number of valid hours required per day (default 24).
            lag_days: Optional lag offset in days (default 0).

        Returns:
            pd.Series containing daily precipitation in mm, indexed by calendar date
            with daily frequency.
        """
        var_name = self._resolve_variable_name("tp")
        ds_pt = self.extract_point(latitude, longitude, method=method)
        da = ds_pt[var_name]

        # Unit conversion to mm
        units = da.attrs.get("units")
        if units is not None:
            units_str = str(units).strip().lower()
            if units_str in {"m", "meter", "meters", "metre", "metres"}:
                da = da * 1000.0
        else:
            # If units missing and mean is typical for meters (< 0.1 m/hr)
            if float(da.mean()) < 0.1:
                da = da * 1000.0

        if "step" in da.dims:
            da = da.sum(dim="step")
        if "surface" in da.dims:
            da = da.squeeze("surface")
        if "number" in da.dims:
            da = da.isel(number=0)

        s_hourly = da.to_series()
        s_hourly.index = pd.DatetimeIndex(s_hourly.index)

        return aggregate_hourly_to_daily_precipitation(
            s_hourly,
            window_start_utc=window_start_utc,
            min_hours=min_hours,
            lag_days=lag_days,
        )

    def get_station_daily_precipitation(
        self,
        station: StationMetadata,
        method: str = "nearest",
        window_start_utc: int = 12,
        min_hours: int = 24,
        lag_days: int = 0,
    ) -> pd.Series:
        """Extract daily precipitation series collocated at station coordinates."""
        return self.get_daily_precipitation(
            latitude=station.latitude,
            longitude=station.longitude,
            method=method,
            window_start_utc=window_start_utc,
            min_hours=min_hours,
            lag_days=lag_days,
        )

    def get_model_elevation(
        self,
        latitude: float,
        longitude: float,
        method: str = "nearest",
    ) -> Optional[float]:
        """Extract model ground elevation in meters above sea level (masl).

        Calculates elevation from surface geopotential z (z / 9.80665). Returns None
        if variable 'z' or 'geopotential' is not present in the dataset.
        """
        try:
            z_var = self._resolve_variable_name("z")
        except KeyError:
            return None

        ds_pt = self.extract_point(latitude, longitude, method=method)
        z_val = ds_pt[z_var]

        # If z contains time dimension, take temporal mean (topography is static)
        if "time" in z_val.dims:
            z_val = z_val.mean(dim="time")

        elevation_meters = float(z_val.values) / STANDARD_GRAVITY
        return round(elevation_meters, 2)

    def get_land_sea_mask(
        self,
        latitude: float,
        longitude: float,
        method: str = "nearest",
    ) -> Optional[float]:
        """Extract land-sea mask proportion [0, 1] at given coordinates.

        Returns None if variable 'lsm' or 'land_sea_mask' is not present in dataset.
        """
        try:
            lsm_var = self._resolve_variable_name("lsm")
        except KeyError:
            return None

        ds_pt = self.extract_point(latitude, longitude, method=method)
        lsm_val = ds_pt[lsm_var]
        if "time" in lsm_val.dims:
            lsm_val = lsm_val.mean(dim="time")
        return round(float(lsm_val.values), 4)

    def get_daily_temperature(
        self,
        latitude: float,
        longitude: float,
        method: str = "nearest",
        window_start_utc: int = 5,
        min_hours: int = 24,
        lag_days: int = 0,
        convert_to_celsius: bool = True,
        closed: Optional[Literal["left", "right"]] = None,
    ) -> pd.DataFrame:
        """Extract daily temperature series (tmax, tmin, tmean) at coordinates.

        Aggregates hourly temperature over a configurable 24h window (default:
        05:00 UTC D to 05:00 UTC D+1, corresponding to local civil day 00:00-24:00
        UTC-5 in Colombia), converting Kelvin to Celsius and masking incomplete
        days with NaN.

        Args:
            latitude: Target latitude in decimal degrees.
            longitude: Target longitude in decimal degrees.
            method: Spatial extraction method ('nearest' or 'bilinear').
            window_start_utc: Starting hour of the 24h window in UTC (default 5).
            min_hours: Minimum number of valid hours required per day (default 24).
            lag_days: Optional lag offset in days (default 0).
            convert_to_celsius: If True, converts Kelvin to Celsius (T - 273.15).
            closed: Resample boundary definition ('left' or 'right').

        Returns:
            pd.DataFrame with columns ['tmax', 'tmin', 'tmean'] indexed by calendar
            date (freq='D').
        """
        var_name = self._resolve_variable_name("t2m")
        ds_pt = self.extract_point(latitude, longitude, method=method)
        da = ds_pt[var_name]

        if "step" in da.dims:
            da = da.mean(dim="step")
        if "surface" in da.dims:
            da = da.squeeze("surface")
        if "number" in da.dims:
            da = da.isel(number=0)

        s_hourly = da.to_series()
        s_hourly.index = pd.DatetimeIndex(s_hourly.index)

        return aggregate_hourly_to_daily_temperature(
            s_hourly,
            window_start_utc=window_start_utc,
            min_hours=min_hours,
            lag_days=lag_days,
            convert_to_celsius=convert_to_celsius,
            closed=closed,
        )

    def get_station_daily_temperature(
        self,
        station: StationMetadata,
        method: str = "nearest",
        window_start_utc: int = 5,
        min_hours: int = 24,
        lag_days: int = 0,
        convert_to_celsius: bool = True,
        closed: Optional[Literal["left", "right"]] = None,
    ) -> pd.DataFrame:
        """Extract daily temperature DataFrame collocated at station coordinates."""
        return self.get_daily_temperature(
            latitude=station.latitude,
            longitude=station.longitude,
            method=method,
            window_start_utc=window_start_utc,
            min_hours=min_hours,
            lag_days=lag_days,
            convert_to_celsius=convert_to_celsius,
            closed=closed,
        )

    def close(self) -> None:
        """Close the underlying NetCDF dataset handles."""
        self._ds.close()

    def __enter__(self) -> "ERA5Adapter":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


def aggregate_hourly_to_daily_precipitation(
    hourly_series: pd.Series,
    window_start_utc: int = 12,
    min_hours: int = 24,
    lag_days: int = 0,
) -> pd.Series:
    """Aggregate hourly precipitation to daily accumulation matching WMO convention.

    By WMO and IDEAM standard (Día Pluviométrico), precipitation for day D
    corresponds to the 24-hour accumulation between 07:00 local time on day D
    (12:00 UTC) and 07:00 local time on day D+1 (12:00 UTC).

    Args:
        hourly_series: Hourly precipitation series with DatetimeIndex in UTC (in mm).
        window_start_utc: Starting hour of the 24h accumulation window in UTC
            (default 12 for 07:00 Colombia).
        min_hours: Minimum number of valid hours required in the 24h window
            (default 24). Days with fewer hours are masked with NaN.
        lag_days: Lag offset in days (default 0). Positive shifts values forward,
            negative backward.

    Returns:
        pd.Series with normalized daily DatetimeIndex (freq='D') in mm.
    """
    if not isinstance(hourly_series.index, pd.DatetimeIndex):
        raise ValueError("hourly_series must have a pandas DatetimeIndex.")

    # Resample 24h with window_start_utc offset
    # closed='right', label='left': interval (D 12:00 .. D+1 12:00] is labeled D 12:00
    resampled = hourly_series.resample(
        "24h",
        offset=f"{window_start_utc}h",
        closed="right",
        label="left",
    ).sum(min_count=min_hours)

    dt_index = pd.DatetimeIndex(resampled.index)
    resampled.index = dt_index.normalize()
    resampled = resampled.asfreq("D")

    if lag_days != 0:
        resampled = resampled.shift(lag_days)

    resampled.name = "pr"
    return resampled


def aggregate_hourly_to_daily_temperature(
    hourly_series: pd.Series,
    window_start_utc: int = 5,
    min_hours: int = 24,
    lag_days: int = 0,
    convert_to_celsius: bool = True,
    closed: Optional[Literal["left", "right"]] = None,
) -> pd.DataFrame:
    """Aggregate hourly temperature to daily Tmax, Tmin, and Tmean.

    Converts values from Kelvin to Celsius (T_C = T_K - 273.15) if needed, computes
    maximum, minimum, and mean temperature over the specified 24-hour window, and
    strictly masks incomplete days (< min_hours valid observations) with NaN.

    Args:
        hourly_series: Hourly temperature series with DatetimeIndex in UTC.
        window_start_utc: Starting hour of the 24h window in UTC (default 5 for
            05:00-05:00 UTC local civil day; 0 for 00:00-24:00 UTC;
            12 for 12:00-12:00 UTC).
        min_hours: Minimum number of valid hours required in the 24h window
            (default 24). Days with fewer valid hours have NaN for extremes and mean.
        lag_days: Lag offset in days (default 0). Positive shifts values forward,
            negative backward.
        convert_to_celsius: If True (default), converts from Kelvin to Celsius.
            Values are checked to avoid double conversion if already in Celsius.
        closed: Interval boundary definition for resample: 'left' or 'right'.
            If None, defaults to 'left' for window_start_utc in (0, 5) and 'right' for
            window_start_utc == 12.

    Returns:
        pd.DataFrame with columns ['tmax', 'tmin', 'tmean'] indexed by calendar
        date (freq='D') in °C.
    """
    if not isinstance(hourly_series.index, pd.DatetimeIndex):
        raise ValueError("hourly_series must have a pandas DatetimeIndex.")

    s = hourly_series.copy()

    # Convert K to °C
    if convert_to_celsius:
        valid_vals = s.dropna()
        if not valid_vals.empty:
            if float(valid_vals.median()) > 100.0:
                s = s - 273.15
            else:
                warnings.warn(
                    "Temperature series appears to be already in Celsius "
                    "(median <= 100). Skipping Kelvin-to-Celsius conversion.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            s = s - 273.15

    resample_closed: Literal["left", "right"] = (
        closed
        if closed is not None
        else ("left" if window_start_utc in (0, 5) else "right")
    )

    grouper = s.resample(
        "24h",
        offset=f"{window_start_utc}h",
        closed=resample_closed,
        label="left",
    )

    tmax = grouper.max()
    tmin = grouper.min()
    tmean = grouper.mean()
    counts = grouper.count()

    # Mask days with fewer than min_hours valid readings with NaN
    mask = counts < min_hours
    tmax = tmax.mask(mask)
    tmin = tmin.mask(mask)
    tmean = tmean.mask(mask)

    df = pd.DataFrame(
        {
            "tmax": tmax,
            "tmin": tmin,
            "tmean": tmean,
        }
    )

    dt_index = pd.DatetimeIndex(df.index)
    df.index = dt_index.normalize()
    df = df.asfreq("D")

    if lag_days != 0:
        df = df.shift(lag_days)

    return df

