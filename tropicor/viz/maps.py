"""Static cartographic map plotting engine for Colombian station networks.

Provides publication-grade spatial visualizations with bundled Natural Earth boundaries,
colorblind-safe styling, and dynamic coordinate-based insets for San Andrés and Malpelo.
"""

from dataclasses import asdict
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Union

import pandas as pd

from tropicor.io.stations import NaturalRegion, StationCatalog, StationMetadata
from tropicor.viz._boundaries import (
    MAINLAND_EXTENT,
    MALPELO_EXTENT,
    SAN_ANDRES_EXTENT,
    draw_cartographic_basemap,
)
from tropicor.viz._common import require_matplotlib
from tropicor.viz.palettes import (
    REGIME_MARKERS,
    REGION_COLORS,
    get_region_color,
)

if TYPE_CHECKING:
    import matplotlib.axes


def _normalize_stations_frame(
    stations: Union[StationCatalog, pd.DataFrame, List[StationMetadata]],
) -> pd.DataFrame:
    """Normalize station representations into a standard pandas DataFrame.

    Args:
        stations: StationCatalog, list of StationMetadata, or pandas DataFrame.

    Returns:
        DataFrame containing code, name, latitude, longitude, elevation, and region.

    Raises:
        ValueError: If required spatial coordinates or station code columns are missing.
    """
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
        msg = (
            f"Stations data missing mandatory columns {required_cols - set(df.columns)}"
        )
        raise ValueError(msg)

    # Ensure code is string for robust indexing
    df["code"] = df["code"].astype(str)
    return df


def _partition_by_bounding_box(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Dynamically partition stations into Mainland, San Andrés, and Malpelo subsets.

    Filters stations purely by geographic bounding box coordinates, avoiding hardcoded
    station identification codes.

    Args:
        df: DataFrame containing 'latitude' and 'longitude' columns.

    Returns:
        Tuple of (df_mainland, df_san_andres, df_malpelo).
    """
    # San Andrés & Providencia Archipelago bounding box
    sa_mask = (
        (df["latitude"] >= 12.0)
        & (df["latitude"] <= 14.0)
        & (df["longitude"] >= -82.2)
        & (df["longitude"] <= -80.5)
    )

    # Malpelo Island bounding box
    mal_mask = (
        (df["latitude"] >= 3.7)
        & (df["latitude"] <= 4.5)
        & (df["longitude"] >= -82.0)
        & (df["longitude"] <= -81.2)
    )

    df_sa = df[sa_mask].copy()
    df_mal = df[mal_mask].copy()
    df_main = df[~(sa_mask | mal_mask)].copy()

    return df_main, df_sa, df_mal


def plot_station_map(
    stations: Union[StationCatalog, pd.DataFrame, List[StationMetadata]],
    values: Optional[Union[pd.Series, Dict[str, float], str]] = None,
    color_by_region: bool = True,
    ax: Optional["matplotlib.axes.Axes"] = None,
    show_insets: bool = True,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: Optional[str] = "Colombian Meteorological Station Network",
    cbar_label: Optional[str] = None,
    figsize: Tuple[float, float] = (8.0, 9.0),
) -> "matplotlib.axes.Axes":
    """Plot station locations over bundled Natural Earth cartographic boundaries.

    Supports continuous metric coloring (e.g. elevation, RMSE, KGE) with a colorbar,
    or categorical coloring by natural region via the Okabe-Ito colorblind-safe palette.
    Automatically generates dual insets for San Andrés/Providencia and Isla de Malpelo.

    Args:
        stations: StationCatalog, list of StationMetadata, or pandas DataFrame.
        values: Optional numeric series, dict (keyed by station code), or DataFrame
            column name to color markers continuously.
        color_by_region: Whether to color stations by NaturalRegion if values is None.
        ax: Optional Matplotlib Axes. If None, a new Figure and Axes are created.
        show_insets: Whether to generate geographic insets for outlying islands.
        cmap: Colormap name for continuous values.
        vmin: Minimum value for colormap scaling.
        vmax: Maximum value for colormap scaling.
        title: Plot title text.
        cbar_label: Colorbar descriptive label.
        figsize: Figure dimension tuple in inches (width, height).

    Returns:
        The primary Matplotlib Axes containing the mainland map.
    """
    plt = require_matplotlib()

    if ax is None:
        fig, primary_ax = plt.subplots(figsize=figsize)
    else:
        primary_ax = ax
        fig = primary_ax.figure

    df = _normalize_stations_frame(stations)

    # Resolve continuous values if provided
    has_values = False
    if values is not None:
        has_values = True
        if isinstance(values, str):
            if values not in df.columns:
                msg = f"Values column '{values}' not found in stations DataFrame."
                raise ValueError(msg)
            df["_metric_val"] = pd.to_numeric(df[values], errors="coerce")
        elif isinstance(values, dict):
            df["_metric_val"] = df["code"].map(values).astype(float)
        elif isinstance(values, pd.Series):
            if values.index.dtype == object or str(values.index[0]).isdigit():
                val_map = {str(k): float(v) for k, v in values.items()}
                df["_metric_val"] = df["code"].map(val_map).astype(float)
            else:
                df["_metric_val"] = pd.to_numeric(values.values, errors="coerce")

        calc_vmin = vmin if vmin is not None else float(df["_metric_val"].min())
        calc_vmax = vmax if vmax is not None else float(df["_metric_val"].max())
    else:
        calc_vmin = None
        calc_vmax = None

    # Render primary mainland basemap
    draw_cartographic_basemap(
        primary_ax,
        extent=MAINLAND_EXTENT,
        draw_neighbors=True,
        draw_islands=False,
    )

    # Partition stations into mainland and outlying island subsets
    df_main, df_sa, df_mal = _partition_by_bounding_box(df)

    # Initialize insets if requested
    inset_sa = None
    inset_mal = None
    if show_insets:
        # San Andrés & Providencia inset in upper-left corner (open ocean)
        inset_sa = primary_ax.inset_axes([0.02, 0.70, 0.22, 0.24])
        draw_cartographic_basemap(
            inset_sa,
            extent=SAN_ANDRES_EXTENT,
            draw_neighbors=False,
            draw_islands=True,
        )
        inset_sa.set_title(
            "San Andrés & Prov.", fontsize=7.5, pad=3, weight="bold", color="#37474F"
        )
        inset_sa.set_xticks([])
        inset_sa.set_yticks([])

        # Malpelo Island inset in mid-left Pacific open water
        inset_mal = primary_ax.inset_axes([0.02, 0.38, 0.20, 0.22])
        draw_cartographic_basemap(
            inset_mal,
            extent=MALPELO_EXTENT,
            draw_neighbors=False,
            draw_islands=True,
        )
        inset_mal.set_title(
            "Isla Malpelo", fontsize=7.5, pad=3, weight="bold", color="#37474F"
        )
        inset_mal.set_xticks([])
        inset_mal.set_yticks([])

    # Plot stations
    scatter_handle = None
    if has_values:
        scatter_handle = primary_ax.scatter(
            df_main["longitude"],
            df_main["latitude"],
            c=df_main["_metric_val"],
            cmap=cmap,
            vmin=calc_vmin,
            vmax=calc_vmax,
            s=48,
            edgecolors="black",
            linewidths=0.6,
            zorder=5,
        )
        if show_insets and inset_sa is not None and not df_sa.empty:
            inset_sa.scatter(
                df_sa["longitude"],
                df_sa["latitude"],
                c=df_sa["_metric_val"],
                cmap=cmap,
                vmin=calc_vmin,
                vmax=calc_vmax,
                s=48,
                edgecolors="black",
                linewidths=0.6,
                zorder=5,
            )
        if show_insets and inset_mal is not None and not df_mal.empty:
            inset_mal.scatter(
                df_mal["longitude"],
                df_mal["latitude"],
                c=df_mal["_metric_val"],
                cmap=cmap,
                vmin=calc_vmin,
                vmax=calc_vmax,
                s=48,
                edgecolors="black",
                linewidths=0.6,
                zorder=5,
            )

        cbar = fig.colorbar(
            scatter_handle,
            ax=primary_ax,
            orientation="vertical",
            pad=0.02,
            shrink=0.7,
        )
        if cbar_label:
            cbar.set_label(cbar_label, fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    elif color_by_region and "region" in df.columns:
        regions_present = df["region"].unique()
        for reg in regions_present:
            reg_color = get_region_color(reg)
            reg_label = reg.value if isinstance(reg, NaturalRegion) else str(reg)

            # Mainland stations for this region
            sub_main = df_main[df_main["region"] == reg]
            if not sub_main.empty:
                primary_ax.scatter(
                    sub_main["longitude"],
                    sub_main["latitude"],
                    color=reg_color,
                    s=48,
                    label=reg_label.capitalize(),
                    edgecolors="black",
                    linewidths=0.6,
                    zorder=5,
                )

            # Inset stations
            if show_insets and inset_sa is not None:
                sub_sa = df_sa[df_sa["region"] == reg]
                if not sub_sa.empty:
                    inset_sa.scatter(
                        sub_sa["longitude"],
                        sub_sa["latitude"],
                        color=reg_color,
                        s=48,
                        edgecolors="black",
                        linewidths=0.6,
                        zorder=5,
                    )
            if show_insets and inset_mal is not None:
                sub_mal = df_mal[df_mal["region"] == reg]
                if not sub_mal.empty:
                    inset_mal.scatter(
                        sub_mal["longitude"],
                        sub_mal["latitude"],
                        color=reg_color,
                        s=48,
                        edgecolors="black",
                        linewidths=0.6,
                        zorder=5,
                    )

        primary_ax.legend(
            title="Natural Region",
            bbox_to_anchor=(1.02, 1.0),
            loc="upper left",
            frameon=True,
            fontsize=8,
            title_fontsize=9,
        )
    else:
        # Single default color fallback
        primary_ax.scatter(
            df_main["longitude"],
            df_main["latitude"],
            color="#0072B2",
            s=48,
            edgecolors="black",
            linewidths=0.6,
            zorder=5,
        )
        if show_insets and inset_sa is not None and not df_sa.empty:
            inset_sa.scatter(
                df_sa["longitude"],
                df_sa["latitude"],
                color="#0072B2",
                s=48,
                edgecolors="black",
                linewidths=0.6,
                zorder=5,
            )
        if show_insets and inset_mal is not None and not df_mal.empty:
            inset_mal.scatter(
                df_mal["longitude"],
                df_mal["latitude"],
                color="#0072B2",
                s=48,
                edgecolors="black",
                linewidths=0.6,
                zorder=5,
            )

    if title:
        primary_ax.set_title(title, fontsize=11, fontweight="bold", pad=12)

    primary_ax.set_xlabel("Longitude (°W)", fontsize=9)
    primary_ax.set_ylabel("Latitude (°N)", fontsize=9)

    return primary_ax


def plot_regime_map(
    stations: Union[StationCatalog, pd.DataFrame, List[StationMetadata]],
    regimes: Union[Dict[str, str], pd.Series],
    disagreements: Optional[Union[List[str], Set[str]]] = None,
    ax: Optional["matplotlib.axes.Axes"] = None,
    show_insets: bool = True,
    title: Optional[str] = "Colombian Precipitation Regimes by Station",
    figsize: Tuple[float, float] = (8.0, 9.0),
) -> "matplotlib.axes.Axes":
    """Plot station rainfall regimes with distinct markers and Okabe-Ito colors.

    Renders regime symbols (bimodal, unimodal, multimodal, indeterminate) with regional
    coloring, accompanied by decoupled dual legends placed outside the map.
    Stations without computed regimes are displayed in neutral grey as 'Sin datos'.
    Stations with model-observation disagreement are flagged with an 'X' marker.

    Args:
        stations: StationCatalog, list of StationMetadata, or pandas DataFrame.
        regimes: Mapping from station code to regime name (dict or pd.Series).
        disagreements: Optional collection of station codes exhibiting regime mismatch.
        ax: Optional Matplotlib Axes. If None, a new Figure and Axes are created.
        show_insets: Whether to generate geographic insets for outlying islands.
        title: Plot title text.
        figsize: Figure dimension tuple in inches.

    Returns:
        The primary Matplotlib Axes containing the regime map.
    """
    plt = require_matplotlib()
    import matplotlib.lines as mlines

    if ax is None:
        _, primary_ax = plt.subplots(figsize=figsize)
    else:
        primary_ax = ax

    df = _normalize_stations_frame(stations)

    # Attach regime classification
    if isinstance(regimes, pd.Series):
        reg_map = {str(k): str(v).strip().lower() for k, v in regimes.items()}
    elif isinstance(regimes, dict):
        reg_map = {str(k): str(v).strip().lower() for k, v in regimes.items()}
    else:
        msg = f"Regimes must be a dict or pd.Series, received {type(regimes)}"
        raise ValueError(msg)

    # Missing stations in catalog mapped to 'no_data'
    df["regime"] = df["code"].map(reg_map).fillna("no_data")

    # If explicit disagreements provided, track them as boolean flag
    dis_set = {str(d) for d in disagreements} if disagreements is not None else set()
    df["is_disagreement"] = df["code"].isin(dis_set)

    # Render primary mainland basemap
    draw_cartographic_basemap(
        primary_ax,
        extent=MAINLAND_EXTENT,
        draw_neighbors=True,
        draw_islands=False,
    )

    df_main, df_sa, df_mal = _partition_by_bounding_box(df)

    inset_sa = None
    inset_mal = None
    if show_insets:
        # San Andrés & Providencia inset in upper-left corner (open ocean)
        inset_sa = primary_ax.inset_axes([0.02, 0.70, 0.22, 0.24])
        draw_cartographic_basemap(
            inset_sa,
            extent=SAN_ANDRES_EXTENT,
            draw_neighbors=False,
            draw_islands=True,
        )
        inset_sa.set_title(
            "San Andrés & Prov.", fontsize=7.5, pad=3, weight="bold", color="#37474F"
        )
        inset_sa.set_xticks([])
        inset_sa.set_yticks([])

        # Malpelo Island inset in mid-left Pacific open water
        inset_mal = primary_ax.inset_axes([0.02, 0.38, 0.20, 0.22])
        draw_cartographic_basemap(
            inset_mal,
            extent=MALPELO_EXTENT,
            draw_neighbors=False,
            draw_islands=True,
        )
        inset_mal.set_title(
            "Isla Malpelo", fontsize=7.5, pad=3, weight="bold", color="#37474F"
        )
        inset_mal.set_xticks([])
        inset_mal.set_yticks([])

    # Plot stations by (regime, region) combinations
    present_regimes = set()
    present_regions = set()

    for _, row in df_main.iterrows():
        regime_name = row["regime"]
        region = row.get("region", NaturalRegion.ANDINA)
        present_regimes.add(regime_name)

        if regime_name == "no_data":
            primary_ax.scatter(
                row["longitude"],
                row["latitude"],
                marker="o",
                color="#B0BEC5",
                s=28,
                edgecolors="black",
                linewidths=0.5,
                zorder=4,
            )
        else:
            marker = REGIME_MARKERS.get(regime_name, "s")
            color = get_region_color(region)
            primary_ax.scatter(
                row["longitude"],
                row["latitude"],
                marker=marker,
                color=color,
                s=54,
                edgecolors="black",
                linewidths=0.6,
                zorder=5,
            )
            if row["is_disagreement"]:
                primary_ax.scatter(
                    row["longitude"],
                    row["latitude"],
                    marker="x",
                    color="black",
                    s=55,
                    linewidths=2.2,
                    zorder=7,
                )
            if isinstance(region, NaturalRegion):
                present_regions.add(region)
            else:
                present_regions.add(str(region))

    # Plot inset stations
    if show_insets and inset_sa is not None:
        for _, row in df_sa.iterrows():
            regime_name = row["regime"]
            region = row.get("region", NaturalRegion.INSULAR)
            present_regimes.add(regime_name)

            if regime_name == "no_data":
                inset_sa.scatter(
                    row["longitude"],
                    row["latitude"],
                    marker="o",
                    color="#B0BEC5",
                    s=28,
                    edgecolors="black",
                    linewidths=0.5,
                    zorder=4,
                )
            else:
                marker = REGIME_MARKERS.get(regime_name, "s")
                color = get_region_color(region)
                inset_sa.scatter(
                    row["longitude"],
                    row["latitude"],
                    marker=marker,
                    color=color,
                    s=54,
                    edgecolors="black",
                    linewidths=0.6,
                    zorder=5,
                )
                if row["is_disagreement"]:
                    inset_sa.scatter(
                        row["longitude"],
                        row["latitude"],
                        marker="x",
                        color="black",
                        s=55,
                        linewidths=2.2,
                        zorder=7,
                    )
                if isinstance(region, NaturalRegion):
                    present_regions.add(region)
                else:
                    present_regions.add(str(region))

    if show_insets and inset_mal is not None:
        for _, row in df_mal.iterrows():
            regime_name = row["regime"]
            region = row.get("region", NaturalRegion.INSULAR)
            present_regimes.add(regime_name)

            if regime_name == "no_data":
                inset_mal.scatter(
                    row["longitude"],
                    row["latitude"],
                    marker="o",
                    color="#B0BEC5",
                    s=28,
                    edgecolors="black",
                    linewidths=0.5,
                    zorder=4,
                )
            else:
                marker = REGIME_MARKERS.get(regime_name, "s")
                color = get_region_color(region)
                inset_mal.scatter(
                    row["longitude"],
                    row["latitude"],
                    marker=marker,
                    color=color,
                    s=54,
                    edgecolors="black",
                    linewidths=0.6,
                    zorder=5,
                )
                if row["is_disagreement"]:
                    inset_mal.scatter(
                        row["longitude"],
                        row["latitude"],
                        marker="x",
                        color="black",
                        s=55,
                        linewidths=2.2,
                        zorder=7,
                    )
                if isinstance(region, NaturalRegion):
                    present_regions.add(region)
                else:
                    present_regions.add(str(region))

    # 1. Legend for Regimes (Marker shapes)
    regime_order = [
        ("bimodal", "Bimodal", "o", "#455A64"),
        ("unimodal", "Unimodal", "^", "#455A64"),
        ("multimodal", "Multimodal", "D", "#455A64"),
        ("indeterminate", "Indeterminate", "s", "#455A64"),
        ("no_data", "No data", "o", "#B0BEC5"),
    ]
    regime_handles = [
        mlines.Line2D(
            [],
            [],
            color="none",
            marker=marker_shape,
            markerfacecolor=facecolor,
            markeredgecolor="black",
            markeredgewidth=0.6,
            markersize=7 if code == "no_data" else 8,
            label=label_text,
        )
        for code, label_text, marker_shape, facecolor in regime_order
        if code in present_regimes
    ]

    if df["is_disagreement"].any():
        regime_handles.append(
            mlines.Line2D(
                [],
                [],
                color="none",
                marker="x",
                markeredgecolor="black",
                markeredgewidth=2.2,
                markersize=8,
                label="Obs/ERA5 disagreement",
            )
        )

    leg_regime = primary_ax.legend(
        handles=regime_handles,
        title="Rainfall Regime",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        frameon=True,
        fontsize=8,
        title_fontsize=9,
    )
    primary_ax.add_artist(leg_regime)

    # 2. Legend for Natural Regions (Colors)
    region_handles = []
    sorted_regions = [r for r in NaturalRegion if r in present_regions]
    for r in sorted_regions:
        region_handles.append(
            mlines.Line2D(
                [],
                [],
                color="none",
                marker="o",
                markerfacecolor=REGION_COLORS[r],
                markeredgecolor="black",
                markeredgewidth=0.6,
                markersize=8,
                label=r.value.capitalize(),
            )
        )

    if region_handles:
        primary_ax.legend(
            handles=region_handles,
            title="Natural Region",
            bbox_to_anchor=(1.02, 0.65),
            loc="upper left",
            frameon=True,
            fontsize=8,
            title_fontsize=9,
        )

    if title:
        primary_ax.set_title(title, fontsize=11, fontweight="bold", pad=12)

    primary_ax.set_xlabel("Longitude (°W)", fontsize=9)
    primary_ax.set_ylabel("Latitude (°N)", fontsize=9)

    return primary_ax
