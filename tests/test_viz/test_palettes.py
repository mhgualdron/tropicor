"""Unit tests for Okabe-Ito colorblind-safe palette and cartographic tokens."""

import re

import pytest

from tropicor.io.stations import NaturalRegion
from tropicor.viz.palettes import (
    OKABE_ITO,
    REGIME_MARKERS,
    REGION_COLORS,
    get_region_color,
)

HEX_PATTERN = re.compile(r"^#[0-9a-fA-F]{6}$")


def test_okabe_ito_palette_hex_format() -> None:
    """Verify all Okabe-Ito palette colors are valid 6-character hex strings."""
    assert len(OKABE_ITO) == 8
    for name, hex_code in OKABE_ITO.items():
        assert HEX_PATTERN.match(hex_code), f"{name}: {hex_code} is invalid hex"


def test_region_colors_distinctness() -> None:
    """Verify each of the 6 Colombian natural regions has a unique color."""
    assert len(REGION_COLORS) == 6
    for reg in NaturalRegion:
        assert reg in REGION_COLORS
        assert HEX_PATTERN.match(REGION_COLORS[reg])

    unique_colors = set(REGION_COLORS.values())
    assert len(unique_colors) == 6


def test_andina_pacifico_contrast() -> None:
    """Verify Andina and Pacífico have high contrast (Blue vs Yellow)."""
    andina_color = REGION_COLORS[NaturalRegion.ANDINA]
    pacifico_color = REGION_COLORS[NaturalRegion.PACIFICO]

    assert andina_color == OKABE_ITO["blue"]  # #0072B2
    assert pacifico_color == OKABE_ITO["yellow"]  # #F0E442
    assert andina_color != pacifico_color


def test_get_region_color_enum_and_string() -> None:
    """Verify get_region_color retrieves colors via enum and string."""
    assert get_region_color(NaturalRegion.CARIBE) == OKABE_ITO["orange"]
    assert get_region_color("CARIBE") == OKABE_ITO["orange"]
    assert get_region_color("caribe") == OKABE_ITO["orange"]
    assert get_region_color("Caribe") == OKABE_ITO["orange"]

    with pytest.raises(KeyError, match="Unknown natural region 'ATLANTICO'"):
        get_region_color("ATLANTICO")


def test_regime_markers_completeness() -> None:
    """Verify climatological regime markers support required regimes."""
    required_regimes = {"bimodal", "unimodal", "multimodal", "indeterminate"}
    assert required_regimes.issubset(REGIME_MARKERS.keys())
    assert REGIME_MARKERS["multimodal"] == "D"
    assert REGIME_MARKERS["bimodal"] == "o"
    assert REGIME_MARKERS["unimodal"] == "^"
