"""Native IDEAM DHIME reader engine and time series adapter."""

import re
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


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
