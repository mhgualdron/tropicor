"""Unit tests for tropicor.core.benchmark_loaders.

All tests operate strictly on synthetic in-memory or tmp_path CSV data.
No real observation files, ERA5 datasets, or audit ledgers are accessed.
"""

import inspect
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import tropicor.core.benchmark_loaders as bld
from tropicor.core.benchmark_loaders import (
    ObservationRecord,
    read_observation_record,
)


def _assert_clean_exception(exc: BaseException) -> None:
    """Assert exception chain contains no sentinel observation tokens."""
    sentinels = ["12345", "87654", "XYZ"]
    visited: set[int] = set()
    stack: list[BaseException] = [exc]
    while stack:
        cur = stack.pop()
        if id(cur) in visited:
            continue
        visited.add(id(cur))
        msg_str = str(cur)
        msg_repr = repr(cur)
        for s in sentinels:
            assert s not in msg_str, f"Sentinel '{s}' in str(exc): {msg_str}"
            assert s not in msg_repr, f"Sentinel '{s}' in repr(exc): {msg_repr}"
        if cur.__cause__ is not None:
            stack.append(cur.__cause__)
        if cur.__context__ is not None:
            stack.append(cur.__context__)

    tb_str = "".join(traceback.format_exception(exc))
    for s in sentinels:
        assert s not in tb_str, f"Sentinel '{s}' in traceback: {tb_str}"


def _create_csv(path: Path, rows: list[str]) -> Path:
    """Helper to write CSV lines with standard UTF-8 encoding."""
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_happy_path(tmp_path: Path) -> None:
    """Happy path: reads 2001-2002 from 2000-2003 rows without opening records."""
    csv_file = _create_csv(
        tmp_path / "obs_happy.csv",
        [
            "CodigoEstacion,Fecha,Valor",
            "21205710,2000-06-15,10.0",
            "21205710,2001-05-10,15.5",
            "21205710,2002-08-20,22.0",
            "21205710,2003-01-01,18.0",
        ],
    )
    rec = read_observation_record(
        csv_file,
        station_id="21205710",
        first_year=2001,
        last_year=2002,
        calibration_last_year=2002,
    )
    assert isinstance(rec, ObservationRecord)
    assert rec.station_id == "21205710"
    assert rec.n_station_rows == 4
    assert rec.n_identical_duplicates_dropped == 0
    assert rec.approval is None

    expected_index = pd.date_range("2001-01-01", "2002-12-31", freq="D")
    pd.testing.assert_index_equal(rec.series.index, expected_index)
    assert rec.series.dtype == float
    assert rec.series.loc["2001-05-10"] == 15.5
    assert rec.series.loc["2002-08-20"] == 22.0
    assert np.isnan(rec.series.loc["2001-01-01"])
    assert rec.series.count() == 2


def test_station_normalization(tmp_path: Path) -> None:
    """Station normalization handles .0 suffixes; ignores other stations."""
    csv_file = _create_csv(
        tmp_path / "obs_norm.csv",
        [
            "CodigoEstacion,Fecha,Valor",
            "21205710.0,2001-01-01,11.0",
            "99999999,INVALID_DATE,NOT_NUMERIC",
            "99999999,2001-01-01,99.9",
        ],
    )
    rec1 = read_observation_record(
        csv_file,
        station_id="21205710",
        first_year=2001,
        last_year=2001,
        calibration_last_year=2001,
    )
    assert rec1.series.loc["2001-01-01"] == 11.0
    assert rec1.n_station_rows == 1

    csv_file2 = _create_csv(
        tmp_path / "obs_norm2.csv",
        [
            "CodigoEstacion,Fecha,Valor",
            "21205710,2001-01-01,12.0",
        ],
    )
    rec2 = read_observation_record(
        csv_file2,
        station_id="21205710.0",
        first_year=2001,
        last_year=2001,
        calibration_last_year=2001,
    )
    assert rec2.series.loc["2001-01-01"] == 12.0


def test_validation_over_all_station_rows_before_trimming(tmp_path: Path) -> None:
    """Checks over all station rows trigger even for years outside range."""
    # Unparseable date in 2005 (outside requested 2001-2002)
    p_bad_date = _create_csv(
        tmp_path / "bad_date.csv",
        [
            "CodigoEstacion,Fecha,Valor",
            "21205710,2001-01-01,10.0",
            "21205710,2005-99-99,15.0",
        ],
    )
    with pytest.raises(ValueError, match="unparseable date"):
        read_observation_record(
            p_bad_date, "21205710", 2001, 2002, calibration_last_year=2002
        )

    # Non-numeric value in 2005 (outside requested 2001-2002)
    p_bad_val = _create_csv(
        tmp_path / "bad_val.csv",
        [
            "CodigoEstacion,Fecha,Valor",
            "21205710,2001-01-01,10.0",
            "21205710,2005-01-01,INVALID_VAL",
        ],
    )
    with pytest.raises(ValueError, match="non-numeric value"):
        read_observation_record(
            p_bad_val, "21205710", 2001, 2002, calibration_last_year=2002
        )

    # Duplicate dates with distinct values in 2005 (outside requested 2001-2002)
    p_dup = _create_csv(
        tmp_path / "bad_dup.csv",
        [
            "CodigoEstacion,Fecha,Valor",
            "21205710,2001-01-01,10.0",
            "21205710,2005-06-01,12.0",
            "21205710,2005-06-01,14.0",
        ],
    )
    with pytest.raises(ValueError, match="duplicate date.*distinct values"):
        read_observation_record(
            p_dup, "21205710", 2001, 2002, calibration_last_year=2002
        )


def test_identical_duplicates(tmp_path: Path) -> None:
    """Identical duplicates are dropped, first kept, and counted."""
    csv_file = _create_csv(
        tmp_path / "obs_dups.csv",
        [
            "CodigoEstacion,Fecha,Valor,NivelAprobacion",
            "21205710,2001-01-01,10.0,Definitivo",
            "21205710,2001-01-01,10.0,Preliminar",
            "21205710,2001-01-02,15.0,Definitivo",
        ],
    )
    rec = read_observation_record(
        csv_file,
        station_id="21205710",
        first_year=2001,
        last_year=2001,
        calibration_last_year=2001,
        read_approval=True,
    )
    assert rec.n_station_rows == 3
    assert rec.n_identical_duplicates_dropped == 1
    assert rec.series.loc["2001-01-01"] == 10.0
    assert rec.series.loc["2001-01-02"] == 15.0
    assert rec.approval is not None
    assert rec.approval.loc["2001-01-01"] == "Definitivo"


def test_comma_decimals_and_empty_null_nan_values(tmp_path: Path) -> None:
    """Comma decimals convert cleanly; empty/null/nan tokens become NaN."""
    csv_file = _create_csv(
        tmp_path / "obs_clean.csv",
        [
            "CodigoEstacion,Fecha,Valor",
            '21205710,2001-01-01,"12,5"',
            "21205710,2001-01-02,",
            "21205710,2001-01-03,NaN",
            "21205710,2001-01-04,NULL",
            "21205710,2001-01-05,None",
        ],
    )
    rec = read_observation_record(
        csv_file,
        station_id="21205710",
        first_year=2001,
        last_year=2001,
        calibration_last_year=2001,
    )
    assert rec.series.loc["2001-01-01"] == 12.5
    assert np.isnan(rec.series.loc["2001-01-02"])
    assert np.isnan(rec.series.loc["2001-01-03"])
    assert np.isnan(rec.series.loc["2001-01-04"])
    assert np.isnan(rec.series.loc["2001-01-05"])


def test_plausibility_check_e61(tmp_path: Path) -> None:
    """E.6.1 plausibility check [-60, 60] on requested years, ignoring NaN."""
    # a. Values exactly at limits [-60.0, 60.0] are accepted
    p_exact = _create_csv(
        tmp_path / "obs_exact_limits.csv",
        [
            "CodigoEstacion,Fecha,Valor",
            "21205710,2001-01-01,-60.0",
            "21205710,2001-01-02,60.0",
        ],
    )
    rec_exact = read_observation_record(
        p_exact,
        station_id="21205710",
        first_year=2001,
        last_year=2001,
        calibration_last_year=2001,
    )
    assert rec_exact.series.loc["2001-01-01"] == -60.0
    assert rec_exact.series.loc["2001-01-02"] == 60.0

    # a. Values just outside limits raise ValueError
    p_low = _create_csv(
        tmp_path / "obs_just_below.csv",
        [
            "CodigoEstacion,Fecha,Valor",
            "21205710,2001-01-01,-60.01",
        ],
    )
    with pytest.raises(ValueError, match="outside the E.6.1 plausibility limits"):
        read_observation_record(
            p_low,
            station_id="21205710",
            first_year=2001,
            last_year=2001,
            calibration_last_year=2001,
        )

    p_high = _create_csv(
        tmp_path / "obs_just_above.csv",
        [
            "CodigoEstacion,Fecha,Valor",
            "21205710,2001-01-01,60.01",
        ],
    )
    with pytest.raises(ValueError, match="outside the E.6.1 plausibility limits"):
        read_observation_record(
            p_high,
            station_id="21205710",
            first_year=2001,
            last_year=2001,
            calibration_last_year=2001,
        )

    # b. Implausible value outside requested range is ignored; inside raises
    p_outside = _create_csv(
        tmp_path / "obs_outside_range.csv",
        [
            "CodigoEstacion,Fecha,Valor",
            "21205710,2000-06-15,75.0",
            "21205710,2001-06-15,20.0",
        ],
    )
    # Requested 2001-2001: year 2000 is outside requested range, should NOT raise
    rec_out = read_observation_record(
        p_outside,
        station_id="21205710",
        first_year=2001,
        last_year=2001,
        calibration_last_year=2001,
    )
    assert rec_out.series.loc["2001-06-15"] == 20.0

    # Requested 2000-2001: year 2000 is inside requested range, MUST raise
    with pytest.raises(ValueError, match="outside the E.6.1 plausibility limits"):
        read_observation_record(
            p_outside,
            station_id="21205710",
            first_year=2000,
            last_year=2001,
            calibration_last_year=2001,
        )

    # c. Exception text contains count only, neither values nor dates; __cause__ and __context__ are None  # noqa: E501
    p_sentinel = _create_csv(
        tmp_path / "obs_plaus_sentinels.csv",
        [
            "CodigoEstacion,Fecha,Valor",
            "21205710,2001-06-15,-87654.321",
            "21205710,2001-07-20,12345.678",
            "21205710,2001-08-10,25.0",
        ],
    )
    with pytest.raises(ValueError) as exc_info:
        read_observation_record(
            p_sentinel,
            station_id="21205710",
            first_year=2001,
            last_year=2001,
            calibration_last_year=2001,
        )
    err = exc_info.value
    err_str = str(err)
    err_repr = repr(err)
    assert "found 2 value(s) outside the E.6.1 plausibility limits" in err_str
    assert "87654" not in err_str and "87654" not in err_repr
    assert "12345" not in err_str and "12345" not in err_repr
    assert "2001-06-15" not in err_str and "2001-06-15" not in err_repr
    assert "2001-07-20" not in err_str and "2001-07-20" not in err_repr
    assert err.__cause__ is None
    assert err.__context__ is None
    _assert_clean_exception(err)

    # d. NaN values are ignored by the check
    p_nans = _create_csv(
        tmp_path / "obs_nans.csv",
        [
            "CodigoEstacion,Fecha,Valor",
            "21205710,2001-01-01,NaN",
            "21205710,2001-01-02,NULL",
            "21205710,2001-01-03,",
            "21205710,2001-01-04,18.0",
        ],
    )
    rec_nans = read_observation_record(
        p_nans,
        station_id="21205710",
        first_year=2001,
        last_year=2001,
        calibration_last_year=2001,
    )
    assert rec_nans.series.loc["2001-01-04"] == 18.0


def test_approval_flags(tmp_path: Path) -> None:
    """Approval flags align to series; stripped string labels or NaN."""
    csv_with_app = _create_csv(
        tmp_path / "obs_app.csv",
        [
            "CodigoEstacion,Fecha,Valor,NivelAprobacion",
            "21205710,2001-01-01,15.0,  Definitivo  ",
            "21205710,2001-01-03,16.0,Preliminar",
        ],
    )
    rec_app = read_observation_record(
        csv_with_app,
        station_id="21205710",
        first_year=2001,
        last_year=2001,
        calibration_last_year=2001,
        read_approval=True,
    )
    assert rec_app.approval is not None
    assert rec_app.approval.loc["2001-01-01"] == "Definitivo"
    assert pd.isna(rec_app.approval.loc["2001-01-02"])
    assert rec_app.approval.loc["2001-01-03"] == "Preliminar"

    # Missing NivelAprobacion with read_approval=True raises ValueError
    csv_no_app = _create_csv(
        tmp_path / "obs_no_app.csv",
        [
            "CodigoEstacion,Fecha,Valor",
            "21205710,2001-01-01,15.0",
        ],
    )
    with pytest.raises(ValueError, match="NivelAprobacion"):
        read_observation_record(
            csv_no_app,
            station_id="21205710",
            first_year=2001,
            last_year=2001,
            calibration_last_year=2001,
            read_approval=True,
        )

    # Missing NivelAprobacion with read_approval=False returns approval=None
    rec_no_app = read_observation_record(
        csv_no_app,
        station_id="21205710",
        first_year=2001,
        last_year=2001,
        calibration_last_year=2001,
        read_approval=False,
    )
    assert rec_no_app.approval is None


def test_contract_and_file_errors(tmp_path: Path) -> None:
    """Verifies argument contract, missing file, missing columns, absent station."""
    # first_year > last_year
    with pytest.raises(ValueError, match="Invalid year range"):
        read_observation_record(
            tmp_path / "any.csv",
            station_id="21205710",
            first_year=2005,
            last_year=2002,
            calibration_last_year=2002,
        )

    # Missing file
    with pytest.raises(FileNotFoundError, match="Observation file not found"):
        read_observation_record(
            tmp_path / "nonexistent.csv",
            station_id="21205710",
            first_year=2001,
            last_year=2002,
            calibration_last_year=2002,
        )

    # Missing required columns
    csv_missing_col = _create_csv(
        tmp_path / "missing_col.csv",
        ["CodigoEstacion,Fecha", "21205710,2001-01-01"],
    )
    with pytest.raises(ValueError, match="Missing required column"):
        read_observation_record(
            csv_missing_col,
            station_id="21205710",
            first_year=2001,
            last_year=2001,
            calibration_last_year=2001,
        )

    # Station absent from file
    csv_absent = _create_csv(
        tmp_path / "absent_st.csv",
        ["CodigoEstacion,Fecha,Valor", "99999999,2001-01-01,10.0"],
    )
    with pytest.raises(ValueError, match="Station ID '21205710' not found"):
        read_observation_record(
            csv_absent,
            station_id="21205710",
            first_year=2001,
            last_year=2001,
            calibration_last_year=2001,
        )


def test_opening_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Opening guard blocks validation requests prior to any file I/O."""
    target_st = "21205710"
    purpose = "benchmark_formal_opening"
    cal_last = 2004
    val_last = 2005

    # 1. Validation years requested, opening_records empty/None -> PermissionError
    # Monkeypatch pd.read_csv to ensure no read occurs
    def _forbidden_read(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("read_csv must not be called")

    monkeypatch.setattr(pd, "read_csv", _forbidden_read)

    nonexistent_path = tmp_path / "nonexistent_file.csv"
    with pytest.raises(PermissionError, match="Validation loading blocked"):
        read_observation_record(
            nonexistent_path,
            target_st,
            2001,
            val_last,
            cal_last,
            opening_records=None,
            opening_purpose=purpose,
        )

    with pytest.raises(PermissionError, match="Validation loading blocked"):
        read_observation_record(
            nonexistent_path,
            target_st,
            2001,
            val_last,
            cal_last,
            opening_records=[],
            opening_purpose=purpose,
        )

    monkeypatch.undo()

    # 2. Record for another station -> PermissionError
    other_st_record: list[Mapping[str, Any]] = [
        {"station_id": "23215030", "purpose": purpose}
    ]
    with pytest.raises(PermissionError, match="Validation loading blocked"):
        read_observation_record(
            nonexistent_path,
            target_st,
            2001,
            val_last,
            cal_last,
            opening_records=other_st_record,
            opening_purpose=purpose,
        )

    # 3. Record for right station but wrong purpose -> PermissionError
    wrong_purpose_record: list[Mapping[str, Any]] = [
        {"station_id": target_st, "purpose": "other_purpose"}
    ]
    with pytest.raises(PermissionError, match="Validation loading blocked"):
        read_observation_record(
            nonexistent_path,
            target_st,
            2001,
            val_last,
            cal_last,
            opening_records=wrong_purpose_record,
            opening_purpose=purpose,
        )

    # 4. Correct opening record -> reads and returns successfully
    valid_csv = _create_csv(
        tmp_path / "val_obs.csv",
        ["CodigoEstacion,Fecha,Valor", "21205710,2005-01-01,16.0"],
    )
    correct_record: list[Mapping[str, Any]] = [
        {"station_id": target_st, "purpose": purpose}
    ]
    rec = read_observation_record(
        valid_csv,
        target_st,
        2005,
        val_last,
        cal_last,
        opening_records=correct_record,
        opening_purpose=purpose,
    )
    assert rec.series.loc["2005-01-01"] == 16.0

    # 5. last_year <= calibration_last_year with no records -> reads
    rec_cal = read_observation_record(
        valid_csv,
        target_st,
        2001,
        cal_last,
        cal_last,
        opening_records=None,
        opening_purpose=None,
    )
    assert rec_cal.station_id == target_st

    # 6. Validation requested with opening_purpose None or empty -> ValueError
    with pytest.raises(ValueError, match="opening_purpose must be a non-empty string"):
        read_observation_record(
            valid_csv,
            target_st,
            2001,
            val_last,
            cal_last,
            opening_records=correct_record,
            opening_purpose=None,
        )

    # 7. Guard raises PermissionError before file access even with implausible values
    implausible_csv = _create_csv(
        tmp_path / "implausible_val.csv",
        ["CodigoEstacion,Fecha,Valor", "21205710,2005-01-01,999.0"],
    )
    with pytest.raises(PermissionError, match="Validation loading blocked"):
        read_observation_record(
            implausible_csv,
            target_st,
            2001,
            val_last,
            cal_last,
            opening_records=None,
            opening_purpose=purpose,
        )


def _write_sentinel_csv(path: Path, case_index: int) -> Path:
    """Write synthetic sentinel CSVs without embedding tokens in test calls."""
    if case_index == 1:
        # Duplicate dates with distinct sentinel values
        content = (
            "CodigoEstacion,Fecha,Valor\n"
            "21205710,2001-01-01,12345.678\n"
            "21205710,2001-01-01,87654.321\n"
        )
        path.write_text(content, encoding="utf-8")
    elif case_index == 2:
        # Non-numeric cell
        content = "CodigoEstacion,Fecha,Valor\n21205710,2001-01-01,XYZ12345\n"
        path.write_text(content, encoding="utf-8")
    elif case_index == 3:
        # Unparseable date cell
        content = "CodigoEstacion,Fecha,Valor\n21205710,XYZ12345,15.0\n"
        path.write_text(content, encoding="utf-8")
    elif case_index == 4:
        # Malformed CSV triggering ParserError
        content = (
            "CodigoEstacion,Fecha,Valor\n"
            "21205710,2001-01-01,10.0\n"
            "21205710,2001-01-02,12345.678,87654.321,XYZ\n"
        )
        path.write_text(content, encoding="utf-8")
    elif case_index == 5:
        # Invalid UTF-8 bytes triggering UnicodeDecodeError
        raw_bytes = (
            b"CodigoEstacion,Fecha,Valor\n21205710,2001-01-01,\xff\xfeXYZ12345\n"
        )
        path.write_bytes(raw_bytes)
    elif case_index == 6:
        # Empty file triggering EmptyDataError
        path.write_text("", encoding="utf-8")
    return path


def test_value_free_exceptions_with_sentinels(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verifies all exception chains are completely free of cell values or tokens."""
    # Case 1: duplicate dates with distinct values
    p1 = _write_sentinel_csv(tmp_path / "c1.csv", 1)
    with pytest.raises(ValueError) as exc_info1:
        read_observation_record(p1, "21205710", 2001, 2001, 2001)
    _assert_clean_exception(exc_info1.value)

    # Case 2: non-numeric cell
    p2 = _write_sentinel_csv(tmp_path / "c2.csv", 2)
    with pytest.raises(ValueError) as exc_info2:
        read_observation_record(p2, "21205710", 2001, 2001, 2001)
    _assert_clean_exception(exc_info2.value)

    # Case 3: unparseable date cell
    p3 = _write_sentinel_csv(tmp_path / "c3.csv", 3)
    with pytest.raises(ValueError) as exc_info3:
        read_observation_record(p3, "21205710", 2001, 2001, 2001)
    _assert_clean_exception(exc_info3.value)

    # Case 4: malformed CSV (ParserError)
    p4 = _write_sentinel_csv(tmp_path / "c4.csv", 4)
    with pytest.raises(pd.errors.ParserError):
        pd.read_csv(p4)
    with pytest.raises(ValueError) as exc_info4:
        read_observation_record(p4, "21205710", 2001, 2001, 2001)
    assert str(exc_info4.value) == "Could not read observation file (parser_error)."
    assert exc_info4.value.__cause__ is None
    assert exc_info4.value.__context__ is None
    _assert_clean_exception(exc_info4.value)

    # Case 5: invalid UTF-8 bytes (UnicodeDecodeError)
    p5 = _write_sentinel_csv(tmp_path / "c5.csv", 5)
    with pytest.raises(UnicodeDecodeError):
        pd.read_csv(p5)
    with pytest.raises(ValueError) as exc_info5:
        read_observation_record(p5, "21205710", 2001, 2001, 2001)
    assert str(exc_info5.value) == "Could not read observation file (decode_error)."
    assert exc_info5.value.__cause__ is None
    assert exc_info5.value.__context__ is None
    _assert_clean_exception(exc_info5.value)

    # Case 6: empty file (EmptyDataError)
    p6 = _write_sentinel_csv(tmp_path / "c6.csv", 6)
    with pytest.raises(pd.errors.EmptyDataError):
        pd.read_csv(p6)
    with pytest.raises(ValueError) as exc_info6:
        read_observation_record(p6, "21205710", 2001, 2001, 2001)
    assert str(exc_info6.value) == "Could not read observation file (empty_file)."
    assert exc_info6.value.__cause__ is None
    assert exc_info6.value.__context__ is None
    _assert_clean_exception(exc_info6.value)

    # Successful read: stdout and stderr remain empty
    p_ok = _create_csv(
        tmp_path / "c_ok.csv",
        ["CodigoEstacion,Fecha,Valor", "21205710,2001-01-01,15.0"],
    )
    read_observation_record(p_ok, "21205710", 2001, 2001, 2001)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_observation_record_repr_and_equality(tmp_path: Path) -> None:
    """ObservationRecord repr conceals series and approval; eq=False uses identity."""
    csv_file = _create_csv(
        tmp_path / "rec_test.csv",
        [
            "CodigoEstacion,Fecha,Valor,NivelAprobacion",
            "21205710,2001-01-01,25.5,SENTINELLABEL",
        ],
    )
    rec = read_observation_record(
        csv_file,
        station_id="21205710",
        first_year=2001,
        last_year=2001,
        calibration_last_year=2001,
        read_approval=True,
    )
    rec_repr = repr(rec)
    rec_str = str(rec)
    assert "25.5" not in rec_repr
    assert "25.5" not in rec_str
    assert "SENTINELLABEL" not in rec_repr
    assert "SENTINELLABEL" not in rec_str

    # Identity equality does not evaluate series boolean truth value
    assert (rec == rec) is True
    assert not (rec != rec)


def test_source_guard_benchmark_loaders() -> None:
    """Source guard: no forbidden ledger identifiers or print calls."""
    src = inspect.getsource(bld)
    forbidden = [
        "validate_validation_access",
        "record_validation_opening",
        "validation_openings.jsonl",
        "print(",
    ]
    for term in forbidden:
        assert term not in src, f"Forbidden term '{term}' found in benchmark_loaders"
