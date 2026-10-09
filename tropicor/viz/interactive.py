"""Interactive geospatial visualizations for Colombian station networks via Plotly."""

from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import pandas as pd

from tropicor.io.stations import NaturalRegion, StationCatalog, StationMetadata
from tropicor.viz._common import require_plotly
from tropicor.viz.palettes import get_region_color

if TYPE_CHECKING:
    import plotly.graph_objects as go


def _normalize_interactive_stations(
    stations: Union[StationCatalog, pd.DataFrame, List[StationMetadata]],
) -> pd.DataFrame:
    """Normalize station inputs to a DataFrame with complete metadata columns."""
    if isinstance(stations, StationCatalog):
        df = stations.to_dataframe()
    elif isinstance(stations, list):
        records = [
            asdict(s) if hasattr(s, "__dataclass_fields__") else dict(s)
            for s in stations
        ]
        df = pd.DataFrame(records)
    elif isinstance(stations, pd.DataFrame):
        df = stations.copy()
    else:
        msg = f"Unsupported stations type: {type(stations)}"
        raise ValueError(msg)

    required_cols = {"code", "latitude", "longitude"}
    if not required_cols.issubset(df.columns):
        msg = f"Stations missing mandatory columns: {required_cols - set(df.columns)}"
        raise ValueError(msg)

    # Fill defaults for optional metadata
    for col in ("name", "department", "municipality", "elevation", "region"):
        if col not in df.columns:
            df[col] = "N/A"

    df["code"] = df["code"].astype(str)
    return df


def plot_station_map_interactive(
    stations: Union[StationCatalog, pd.DataFrame, List[StationMetadata]],
    values: Optional[Union[pd.Series, Dict[str, float], str]] = None,
    color_by_region: bool = True,
    title: str = "TROPICOR Meteorological Station Network (Colombia)",
    map_style: str = "carto-positron",
    zoom: float = 4.8,
    center: Optional[Dict[str, float]] = None,
    colorscale: str = "Viridis",
    cbar_label: Optional[str] = None,
    height: int = 700,
    width: Optional[int] = None,
) -> "go.Figure":
    """Generate an interactive scatter map with rich station hover cards.

    Displays station markers over a clean geographic basemap (default Carto-Positron).
    Supports categorical coloring by natural region with interactive legend toggling,
    or continuous coloring by quantitative metrics (elevation, RMSE, KGE).

    Args:
        stations: StationCatalog, list of StationMetadata, or DataFrame.
        values: Optional continuous values (pd.Series, dict, or column name).
        color_by_region: Whether to color stations by NaturalRegion if values is None.
        title: Map title.
        map_style: Cartographic tile style ('carto-positron', 'open-street-map').
        zoom: Initial map zoom level.
        center: Initial map center dictionary with 'lat' and 'lon'.
        colorscale: Plotly continuous colorscale name.
        cbar_label: Colorbar title text for continuous values.
        height: Figure height in pixels.
        width: Optional figure width in pixels.

    Returns:
        Plotly Figure object ready for interactive display or HTML export.
    """
    go, _ = require_plotly()

    df = _normalize_interactive_stations(stations)

    # Resolve metric values
    has_values = False
    metric_label = cbar_label or "Value"
    if values is not None:
        has_values = True
        if isinstance(values, str):
            if values not in df.columns:
                msg = f"Values column '{values}' not found in stations DataFrame."
                raise ValueError(msg)
            df["_metric_val"] = pd.to_numeric(df[values], errors="coerce")
            metric_label = cbar_label or values
        elif isinstance(values, dict):
            df["_metric_val"] = df["code"].map(values).astype(float)
        elif isinstance(values, pd.Series):
            val_map = {str(k): float(v) for k, v in values.items()}
            df["_metric_val"] = df["code"].map(val_map).astype(float)

    traces = []

    if has_values:
        # Build custom hover columns
        custom_cols = [
            "name",
            "code",
            "region",
            "department",
            "municipality",
            "elevation",
            "_metric_val",
        ]
        customdata = df[custom_cols].to_numpy()

        hovertemplate = (
            "<b>%{customdata[0]}</b> (IDEAM: %{customdata[1]})<br>"
            "Region: %{customdata[2]}<br>"
            "Dept: %{customdata[3]} | Mun: %{customdata[4]}<br>"
            "Elevation: %{customdata[5]} masl<br>"
            "Coordinates: %{lat:.4f}°N, %{lon:.4f}°W<br>"
            f"<b>{metric_label}:</b> %{{customdata[6]:.3g}}"
            "<extra></extra>"
        )

        trace = go.Scattergeo(
            lat=df["latitude"],
            lon=df["longitude"],
            mode="markers",
            name="Stations",
            customdata=customdata,
            hovertemplate=hovertemplate,
            marker=dict(
                size=9,
                color=df["_metric_val"],
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(title=metric_label, len=0.7),
                line=dict(width=1, color="black"),
            ),
        )
        traces.append(trace)

    elif color_by_region and "region" in df.columns:
        # Sort regions according to standard NaturalRegion order
        present_regs = df["region"].unique()

        for reg in present_regs:
            sub = df[df["region"] == reg].copy()
            if sub.empty:
                continue

            reg_color = get_region_color(reg)
            reg_name = reg.value if isinstance(reg, NaturalRegion) else str(reg)

            custom_cols = [
                "name",
                "code",
                "region",
                "department",
                "municipality",
                "elevation",
            ]
            customdata = sub[custom_cols].to_numpy()

            hovertemplate = (
                "<b>%{customdata[0]}</b> (IDEAM: %{customdata[1]})<br>"
                "Region: %{customdata[2]}<br>"
                "Dept: %{customdata[3]} | Mun: %{customdata[4]}<br>"
                "Elevation: %{customdata[5]} masl<br>"
                "Coordinates: %{lat:.4f}°N, %{lon:.4f}°W"
                "<extra></extra>"
            )

            trace = go.Scattergeo(
                lat=sub["latitude"],
                lon=sub["longitude"],
                mode="markers",
                name=reg_name.capitalize(),
                customdata=customdata,
                hovertemplate=hovertemplate,
                marker=dict(
                    size=9,
                    color=reg_color,
                    line=dict(width=1, color="black"),
                ),
            )
            traces.append(trace)
    else:
        # Default single trace
        custom_cols = [
            "name",
            "code",
            "region",
            "department",
            "municipality",
            "elevation",
        ]
        customdata = df[custom_cols].to_numpy()
        hovertemplate = (
            "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
            "Dept: %{customdata[3]} | Mun: %{customdata[4]}<br>"
            "Coordinates: %{lat:.4f}°N, %{lon:.4f}°W"
            "<extra></extra>"
        )
        trace = go.Scattergeo(
            lat=df["latitude"],
            lon=df["longitude"],
            mode="markers",
            name="Stations",
            customdata=customdata,
            hovertemplate=hovertemplate,
            marker=dict(
                size=9,
                color="#0072B2",
                line=dict(width=1, color="black"),
            ),
        )
        traces.append(trace)

    fig = go.Figure(data=traces)

    layout_kwargs: Dict[str, Any] = {
        "title": dict(
            text=title,
            x=0.03,
            font=dict(size=14, family="sans-serif"),
        ),
        "margin": dict(l=10, r=10, t=45, b=10),
        "height": height,
        "legend": dict(
            title=dict(text="Natural Region"),
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.02,
            bgcolor="rgba(255, 255, 255, 0.85)",
            bordercolor="#CFD8DC",
            borderwidth=1,
        ),
        "geo": dict(
            scope="south america",
            resolution=50,
            showland=True,
            landcolor="#F8F9FA",
            showocean=True,
            oceancolor="#EBF4F9",
            showlakes=True,
            lakecolor="#EBF4F9",
            showcountries=True,
            countrycolor="#6C757D",
            showcoastlines=True,
            coastlinecolor="#455A64",
            projection_type="mercator",
            lataxis_range=[-5.0, 14.5],
            lonaxis_range=[-83.5, -66.0],
        ),
    }

    if width is not None:
        layout_kwargs["width"] = width

    fig.update_layout(**layout_kwargs)
    return fig
