"""ERA5 global atmospheric reanalysis NetCDF reader and point extraction engine."""

import importlib.util
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd
import xarray as xr

from tropicor.io.stations import StationMetadata

STANDARD_GRAVITY: float = 9.80665


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
                self._ds = xr.open_mfdataset(
                    path_str, combine="by_coords", **open_kwargs
                )
            else:
                p = Path(path_str)
                if not p.exists():
                    raise FileNotFoundError(f"ERA5 file not found: {p}")
                self._ds = xr.open_dataset(p, **open_kwargs)
        else:
            self._ds = xr.open_mfdataset(
                [str(f) for f in filepath_or_pattern],
                combine="by_coords",
                **open_kwargs,
            )

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
        if var_lower in {"t2m", "temp", "temperature", "temperatura"}:
            candidates = ["t2m", "2t", "temperature"]
        elif var_lower in {"tp", "precip", "precipitation", "precipitacion"}:
            candidates = ["tp", "total_precipitation", "precipitation"]
        elif var_lower in {"z", "geopotential", "elevation"}:
            candidates = ["z", "geopotential"]

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
            if var_name in {"t2m", "2t"} or "temp" in variable.lower():
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

    def close(self) -> None:
        """Close the underlying NetCDF dataset handles."""
        self._ds.close()

    def __enter__(self) -> "ERA5Adapter":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
