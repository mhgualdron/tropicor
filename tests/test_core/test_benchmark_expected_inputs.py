"""Tests for Section J.11 frozen expected inputs configuration and loader."""

import json
import re
from pathlib import Path

import pytest

from tropicor.core.benchmark_loaders import (
    BENCHMARK_EXPECTED_INPUTS_PATH,
    BENCHMARK_STATION_CONSTANTS,
    load_benchmark_expected_inputs,
)

VALID_J11_DATA: dict[str, str] = {
    "local/newFiles/TempMaxima_JardinBotanico.csv": (
        "d5c5a1e78205299f1022071974653e47d5846b12f5e4de5bec13de0f1439936c"
    ),
    "local/newFiles/TempMinima_JardinBotanico.csv": (
        "41e6a8df65e5282c33d60c4a3a0907d0b0a034d98982fe4faef99ed3f9b550f1"
    ),
    "local/newFiles/TempMaxima_AguasClaras.csv": (
        "46a44e5b2f45068a5fafcd37688f21c2279aa70392ee79f682258953686c7056"
    ),
    "local/newFiles/TempMinima_AguasClaras.csv": (
        "5e0c3d22d63868e63e6e581557dfc03376c7da26ece5e8baa13631e1544d9ea3"
    ),
    "local/era5_hourly/21205710/era5_static.nc": (
        "f22a17f169c1897430faf5819f42f1958df24fd535ba731d90810f75b50d70a1"
    ),
    "local/era5_hourly/23215030/era5_static.nc": (
        "4b2ffbdf5c8dd674c00f405fed62288e2d4ffd34b8e228d5eda17f10aa69c7f9"
    ),
}


def _parse_j11_table() -> dict[str, str]:
    """Parse Section J.11 table from protocol markdown document."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    protocol_path = repo_root / "docs" / "science" / "evaluation-protocol.md"
    text = protocol_path.read_text(encoding="utf-8")
    match = re.search(r"### J\.11 Frozen Inputs.*?\n(.*?)\n###", text, re.DOTALL)
    assert match is not None, "J.11 section not found in protocol"

    rows: dict[str, str] = {}
    for line in match.group(1).splitlines():
        line = line.strip()
        if line.startswith("|") and "`" in line:
            parts = [p.strip().replace("`", "") for p in line.split("|")[1:-1]]
            if (
                len(parts) == 2
                and not parts[0].startswith("---")
                and parts[0] != "File"
            ):
                rows[parts[0]] = parts[1]
    return rows


def test_j11_json_equals_protocol_table_cell_by_cell():
    """Verify JSON matches Section J.11 protocol table cell-by-cell."""
    protocol_table = _parse_j11_table()
    assert len(protocol_table) == 6, (
        f"Expected exactly 6 rows in Section J.11 table, got {len(protocol_table)}"
    )

    loaded = load_benchmark_expected_inputs()
    assert set(loaded.keys()) == set(protocol_table.keys())
    for file_path, expected_hash in protocol_table.items():
        assert loaded[file_path] == expected_hash, (
            f"Hash mismatch for {file_path}: loaded {loaded[file_path]} != "
            f"protocol {expected_hash}"
        )


def test_static_paths_contain_valid_benchmark_station_ids():
    """Verify static paths contain registered benchmark station identifiers."""
    loaded = load_benchmark_expected_inputs()
    static_paths = [p for p in loaded.keys() if p.endswith("era5_static.nc")]
    assert len(static_paths) == 2

    for station_id in ("21205710", "23215030"):
        assert station_id in BENCHMARK_STATION_CONSTANTS
        matches = [p for p in static_paths if station_id in p]
        assert len(matches) == 1, (
            f"Expected exactly 1 static path for {station_id}, found {matches}"
        )


def test_j11_contains_no_hourly_era5_row():
    """Verify Section J.11 contains no hourly ERA5 entries."""
    # Note: Section J.11 states hourly files are hashed at pre-opening (I.3.5).
    protocol_table = _parse_j11_table()
    loaded = load_benchmark_expected_inputs()
    year_regex = re.compile(r"(19\d\d|20\d\d)")

    for path in protocol_table.keys():
        assert "2m_temperature" not in path, f"Hourly ERA5 file in protocol: {path}"
        assert not year_regex.search(Path(path).name), (
            f"Path contains calendar year in filename: {path}"
        )

    for path in loaded.keys():
        assert "2m_temperature" not in path, f"Hourly ERA5 file in config: {path}"
        assert not year_regex.search(Path(path).name), (
            f"Path contains calendar year in filename: {path}"
        )


parametrize = pytest.mark.parametrize


@parametrize(
    "tamper_action",
    [
        "five_entries",
        "extra_entry",
        "absolute_path",
        "parent_traversal",
        "uppercase_hash",
        "63_char_hash",
        "trailing_newline_hash",
        "wrong_csv_name",
    ],
)
def test_loader_validation_rejects_invalid_inputs(tmp_path: Path, tamper_action: str):
    """Verify loader strictly rejects tampered or non-compliant configurations."""
    data = dict(VALID_J11_DATA)
    if tamper_action == "five_entries":
        data.pop("local/era5_hourly/23215030/era5_static.nc")
    elif tamper_action == "extra_entry":
        data["local/extra_untracked_input.csv"] = "a" * 64
    elif tamper_action == "absolute_path":
        data.pop("local/newFiles/TempMaxima_JardinBotanico.csv")
        data["/local/newFiles/TempMaxima_JardinBotanico.csv"] = VALID_J11_DATA[
            "local/newFiles/TempMaxima_JardinBotanico.csv"
        ]
    elif tamper_action == "parent_traversal":
        data.pop("local/newFiles/TempMaxima_JardinBotanico.csv")
        data["local/../newFiles/TempMaxima_JardinBotanico.csv"] = VALID_J11_DATA[
            "local/newFiles/TempMaxima_JardinBotanico.csv"
        ]
    elif tamper_action == "uppercase_hash":
        data["local/newFiles/TempMaxima_JardinBotanico.csv"] = VALID_J11_DATA[
            "local/newFiles/TempMaxima_JardinBotanico.csv"
        ].upper()
    elif tamper_action == "63_char_hash":
        data["local/newFiles/TempMaxima_JardinBotanico.csv"] = "a" * 63
    elif tamper_action == "trailing_newline_hash":
        data["local/newFiles/TempMaxima_JardinBotanico.csv"] = (
            VALID_J11_DATA["local/newFiles/TempMaxima_JardinBotanico.csv"] + "\n"
        )
    elif tamper_action == "wrong_csv_name":
        data.pop("local/newFiles/TempMaxima_JardinBotanico.csv")
        data["local/newFiles/TempMaxima_Unknown.csv"] = VALID_J11_DATA[
            "local/newFiles/TempMaxima_JardinBotanico.csv"
        ]
    else:
        pytest.fail(f"Unknown tamper action: {tamper_action}")

    test_json_file = tmp_path / "tampered_expected_inputs.json"
    test_json_file.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError):
        load_benchmark_expected_inputs(test_json_file)


def test_json_source_guard_no_pii_no_uppercase():
    """Verify JSON source has no forbidden path prefixes or uppercase hex."""
    content = BENCHMARK_EXPECTED_INPUTS_PATH.read_text(encoding="utf-8")
    forbidden_prefixes = ["C" + ":", "/" + "home" + "/", "/" + "Users" + "/"]
    for pat in forbidden_prefixes:
        assert pat not in content, f"Forbidden path pattern {pat} found in JSON"

    # Verify no uppercase hex in values or anywhere in hash strings
    parsed = json.loads(content)
    for path, hex_hash in parsed.items():
        assert re.search(r"[A-F]", hex_hash) is None, (
            f"Uppercase hex found in hash for {path}: {hex_hash}"
        )
    hex_values = re.findall(r'"([0-9a-zA-Z]{64})"', content)
    assert len(hex_values) == 6
    for val in hex_values:
        assert val.islower(), f"Uppercase character in hex digest: {val}"


def test_json_packaging_default_path():
    """Verify default config path resolves to packaged file in tropicor."""
    assert BENCHMARK_EXPECTED_INPUTS_PATH.is_file(), (
        f"Default expected inputs file does not exist: {BENCHMARK_EXPECTED_INPUTS_PATH}"
    )
    pkg_dir = Path(__file__).resolve().parent.parent.parent / "tropicor"
    assert BENCHMARK_EXPECTED_INPUTS_PATH.is_relative_to(pkg_dir), (
        f"{BENCHMARK_EXPECTED_INPUTS_PATH} not relative to {pkg_dir}"
    )

    loaded = load_benchmark_expected_inputs(None)
    assert len(loaded) == 6
