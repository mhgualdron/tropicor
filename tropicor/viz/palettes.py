"""Colorblind-safe palettes and cartographic styling tokens for TROPICOR.

Follows the Okabe & Ito (2008) universal color design palette, optimized for
all major forms of color vision deficiency (deuteranopia, protanopia, tritanopia).
"""

from typing import Dict, Union

from tropicor.io.stations import NaturalRegion

# Karl Okabe & Kei Ito (2008) Universal Color Palette
OKABE_ITO: Dict[str, str] = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermilion": "#D55E00",
    "reddish_purple": "#CC79A7",
}

# Regional mapping using Okabe-Ito hues.
# Pacifico uses Yellow (#F0E442) to provide maximum visual contrast against
# its geographic neighbor Andina (#0072B2, Blue).
REGION_COLORS: Dict[NaturalRegion, str] = {
    NaturalRegion.ANDINA: OKABE_ITO["blue"],  # #0072B2
    NaturalRegion.CARIBE: OKABE_ITO["orange"],  # #E69F00
    NaturalRegion.PACIFICO: OKABE_ITO["yellow"],  # #F0E442
    NaturalRegion.ORINOQUIA: OKABE_ITO["reddish_purple"],  # #CC79A7
    NaturalRegion.AMAZONIA: OKABE_ITO["bluish_green"],  # #009E73
    NaturalRegion.INSULAR: OKABE_ITO["vermilion"],  # #D55E00
}

# Alias for compatibility with user plan terminology
REGION_PALETTE = REGION_COLORS

# String lookup map supporting string keys (e.g., 'ANDINA', 'Andina')
REGION_COLORS_STR: Dict[str, str] = {k.value: v for k, v in REGION_COLORS.items()}

# Distinct marker styles for precipitation climatological regimes
REGIME_MARKERS: Dict[str, str] = {
    "bimodal": "o",
    "unimodal": "^",
    "multimodal": "D",
    "indeterminate": "s",
    "disagreement": "X",
    "no_data": "o",
}

# Standard graphical styling tokens for publication quality
FIGURE_DPI: int = 300
DEFAULT_FONT_FAMILY: str = "sans-serif"
DEFAULT_LINE_WIDTH: float = 1.8
GRID_LINE_COLOR: str = "#E0E0E0"
LAND_FILL_COLOR: str = "#F8F9FA"
NEIGHBOR_FILL_COLOR: str = "#E9ECEF"
BORDER_EDGE_COLOR: str = "#6C757D"
COASTLINE_COLOR: str = "#495057"


def get_region_color(region: Union[NaturalRegion, str]) -> str:
    """Retrieve color hex code for a natural region.

    Args:
        region: NaturalRegion enum instance or case-insensitive region name string.

    Returns:
        Hexadecimal color string corresponding to the region in the Okabe-Ito palette.

    Raises:
        KeyError: If region name does not match any recognized natural region.
    """
    if isinstance(region, NaturalRegion):
        return REGION_COLORS[region]

    reg_norm = str(region).strip().upper()
    if reg_norm in REGION_COLORS_STR:
        return REGION_COLORS_STR[reg_norm]

    # Try mapping by enum key if string doesn't match value directly
    try:
        return REGION_COLORS[NaturalRegion[reg_norm]]
    except KeyError as err:
        valid_regions = [r.value for r in NaturalRegion]
        msg = f"Unknown natural region '{region}'. Must be one of {valid_regions}."
        raise KeyError(msg) from err
