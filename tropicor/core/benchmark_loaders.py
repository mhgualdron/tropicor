"""Full-record observation loaders for benchmark evaluation.

This module performs file I/O and READS ALL ROWS OF THE STATION, including
validation years; it never reads the ledger file and never imports or calls
any ledger function; the orchestrator reads the ledger, verifies that the
records are in it, and passes them in.

The Section E.6.1 plausibility check is applied to the requested years (J.3.1),
and J.3.3 screening counts are computed elsewhere.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union, cast

import pandas as pd

from tropicor.downscale.temperature import (
    MAX_PLAUSIBLE_TEMP_C,
    MIN_PLAUSIBLE_TEMP_C,
)


@dataclass(frozen=True, eq=False)
class ObservationRecord:
    """Container for full-record station observation series and metadata.

    The series and approval attributes are excluded from representation to
    prevent exposure of observation values or data labels in strings or logs.
    Equality is identity-based (eq=False) to avoid ambiguous truth-value errors
    from underlying pandas Series.
    """

    station_id: str
    series: pd.Series = field(repr=False)
    n_station_rows: int
    n_identical_duplicates_dropped: int
    approval: Optional[pd.Series] = field(default=None, repr=False)


def read_observation_record(
    path: Union[str, Path],
    station_id: str,
    first_year: int,
    last_year: int,
    calibration_last_year: int,
    opening_records: Optional[Sequence[Mapping[str, Any]]] = None,
    opening_purpose: Optional[str] = None,
    read_approval: bool = False,
) -> ObservationRecord:
    """Read all rows of a station from an observation CSV, validate and slice.

    Validates date formats, numeric values, and duplicate dates over all rows
    of the target station before trimming to the requested year range and
    reindexing to a complete daily DatetimeIndex.

    When validation years are requested (last_year > calibration_last_year),
    an opening authorization record matching the target station and purpose
    must be supplied in opening_records before any file access takes place.
    The module never reads the ledger file directly.

    Args:
        path: Path to the raw observation CSV file.
        station_id: Station identifier (e.g. '21205710').
        first_year: First year of the requested evaluation period.
        last_year: Last year of the requested evaluation period.
        calibration_last_year: Last year of the station's calibration partition.
        opening_records: Sequence of opening authorization mappings from ledger.
        opening_purpose: Purpose string expected in the opening record.
        read_approval: If True, also extract and align NivelAprobacion flags.

    Returns:
        ObservationRecord containing daily series, row counts, and optional
        approval flags.

    Raises:
        ValueError: If first_year > last_year, opening_purpose is invalid,
            required columns are missing, station is absent, dates or values
            cannot be parsed, duplicate dates have distinct values, or values
            fall outside E.6.1 plausibility limits in the requested years.
        PermissionError: If validation years are requested without a matching
            opening record in opening_records.
        FileNotFoundError: If the observation file path does not exist.
    """
    if first_year > last_year:
        raise ValueError(
            f"Invalid year range: first_year ({first_year}) > last_year ({last_year})."
        )

    clean_id = re.sub(r"\.0$", "", str(station_id).strip())

    if last_year > calibration_last_year:
        if (
            opening_purpose is None
            or not isinstance(opening_purpose, str)
            or not opening_purpose.strip()
        ):
            raise ValueError(
                "opening_purpose must be a non-empty string when validation "
                "years are requested."
            )

        # OPEN ITEM v1.10: K.8.2 and J.10 do not name the ledger purpose label of the benchmark formal opening  # noqa: E501
        authorized = False
        if opening_records is not None:
            for rec in opening_records:
                if isinstance(rec, Mapping):
                    rec_st = re.sub(r"\.0$", "", str(rec.get("station_id", "")).strip())
                    rec_purpose = rec.get("purpose")
                    if rec_st == clean_id and rec_purpose == opening_purpose:
                        authorized = True
                        break

        if not authorized:
            raise PermissionError(
                f"Validation loading blocked for station '{clean_id}': "
                f"no matching opening record found for purpose '{opening_purpose}'."
            )

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Observation file not found: {path_obj}")

    read_category: Optional[str] = None
    try:
        df_raw = pd.read_csv(path_obj)
    except pd.errors.ParserError:
        read_category = "parser_error"
    except UnicodeDecodeError:
        read_category = "decode_error"
    except pd.errors.EmptyDataError:
        read_category = "empty_file"

    if read_category is not None:
        raise ValueError(f"Could not read observation file ({read_category}).")

    required_cols = ["CodigoEstacion", "Fecha", "Valor"]
    if read_approval:
        required_cols.append("NivelAprobacion")
    missing = [c for c in required_cols if c not in df_raw.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s) in observation file: {', '.join(missing)}"
        )

    st_codes = (
        df_raw["CodigoEstacion"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    if clean_id not in st_codes.values:
        raise ValueError(f"Station ID '{clean_id}' not found in observation file.")

    df_st = df_raw[st_codes == clean_id].copy()
    n_station_rows = int(len(df_st))

    parsed_dt = pd.to_datetime(df_st["Fecha"], format="ISO8601", errors="coerce")
    unparseable_mask = (
        parsed_dt.isna()
        & df_st["Fecha"].notna()
        & (df_st["Fecha"].astype(str).str.strip() != "")
    )
    unparseable_count = int(unparseable_mask.sum())
    if unparseable_count > 0:
        raise ValueError(
            f"Station '{clean_id}': found {unparseable_count} unparseable date(s)."
        )

    raw_val_str = df_st["Valor"].astype(str).str.strip().str.replace(",", ".")
    num_vals = pd.to_numeric(raw_val_str, errors="coerce")
    non_numeric_mask = (
        num_vals.isna()
        & df_st["Valor"].notna()
        & ~raw_val_str.str.upper().isin(["", "NAN", "NONE", "NULL"])
    )
    non_numeric_count = int(non_numeric_mask.sum())
    if non_numeric_count > 0:
        raise ValueError(
            f"Station '{clean_id}': found {non_numeric_count} non-numeric value(s) "
            "in column 'Valor'."
        )

    df_st["dt_norm"] = parsed_dt.dt.normalize()
    df_st["clean_val"] = num_vals
    dup_mask = df_st["dt_norm"].duplicated(keep=False)
    if dup_mask.any():
        dup_rows = df_st[dup_mask]
        affected_dates: list[pd.Timestamp] = []
        affected_rows_count = 0
        for dt_val, group in dup_rows.groupby("dt_norm"):
            valid_distinct = group["clean_val"].dropna().unique()
            if len(valid_distinct) > 1:
                affected_dates.append(pd.Timestamp(cast(Any, dt_val)))
                affected_rows_count += len(group)

        if affected_dates:
            affected_dates.sort()
            first_iso = affected_dates[0].strftime("%Y-%m-%d")
            raise ValueError(
                f"Station '{clean_id}': found {len(affected_dates)} duplicate "
                f"date(s) with distinct values; first affected date: {first_iso} "
                f"({affected_rows_count} rows involved)."
            )

    df_st = df_st.dropna(subset=["dt_norm"]).sort_values("dt_norm")
    rows_before_dedup = len(df_st)
    df_st = df_st.drop_duplicates(subset=["dt_norm"], keep="first")
    n_identical_duplicates_dropped = rows_before_dedup - len(df_st)

    s_full = pd.Series(
        df_st["clean_val"].values,
        index=pd.DatetimeIndex(df_st["dt_norm"]),
        dtype="float64",
    )

    s_approval: Optional[pd.Series] = None
    if read_approval:
        approval_vals = df_st["NivelAprobacion"].where(
            df_st["NivelAprobacion"].isna(),
            df_st["NivelAprobacion"].astype(str).str.strip(),
        )
        s_approval = pd.Series(
            approval_vals.values,
            index=pd.DatetimeIndex(df_st["dt_norm"]),
            dtype="object",
        )

    full_daily_index = pd.date_range(
        f"{first_year}-01-01", f"{last_year}-12-31", freq="D"
    )
    s_idx = cast(pd.DatetimeIndex, s_full.index)
    trim_mask = (s_idx.year >= first_year) & (s_idx.year <= last_year)

    s_trimmed = s_full[trim_mask]
    series_reindexed = s_trimmed.reindex(full_daily_index)

    # Section E.6.1 plausibility check on requested years (J.3.1; NaN ignored)
    valid_vals = series_reindexed.dropna()
    if not valid_vals.empty:
        implausible_mask = (valid_vals < MIN_PLAUSIBLE_TEMP_C) | (
            valid_vals > MAX_PLAUSIBLE_TEMP_C
        )
        n_implausible = int(implausible_mask.sum())
        if n_implausible > 0:
            raise ValueError(
                f"Station '{clean_id}': found {n_implausible} value(s) "
                "outside the E.6.1 plausibility limits in the requested years."
            )

    approval_reindexed: Optional[pd.Series] = None
    if s_approval is not None:
        s_app_trimmed = s_approval[trim_mask]
        approval_reindexed = s_app_trimmed.reindex(full_daily_index)

    return ObservationRecord(
        station_id=clean_id,
        series=series_reindexed,
        n_station_rows=n_station_rows,
        n_identical_duplicates_dropped=n_identical_duplicates_dropped,
        approval=approval_reindexed,
    )
