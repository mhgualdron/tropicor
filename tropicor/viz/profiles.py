"""Analytical profiles and diagnostic visualizations for climate validation.

Provides publication-grade climatological comparisons, Taylor diagrams,
diurnal temperature ranges, scatter diagnostics, and data completeness heatmaps.
"""

from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
from scipy import stats

from tropicor.core.climatology import (
    annual_cycle_peaks,
    classify_rainfall_regime,
    compute_monthly_climatology,
)
from tropicor.core.dtr import compute_double_difference, compute_dtr
from tropicor.core.metrics import (
    TaylorStatistics,
    compute_validation_metrics,
    taylor_statistics,
)
from tropicor.viz._common import require_matplotlib
from tropicor.viz.palettes import OKABE_ITO

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

MONTH_LABELS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]


def _extract_monthly_climatology_curve(series: pd.Series) -> pd.Series:
    """Extract or compute 12-month annual climatology curve from a pandas Series."""
    # If already a 12-month profile
    if len(series) == 12 and (
        list(series.index) == list(range(1, 13))
        or list(series.index) == list(range(12))
    ):
        s = series.copy()
        s.index = range(1, 13)
        return s

    # Multi-year DatetimeIndex series
    if isinstance(series.index, pd.DatetimeIndex):
        try:
            return compute_monthly_climatology(series, min_years=5)
        except ValueError:
            # Fallback to simple monthly grouping if shorter than threshold
            return series.groupby(series.index.month).mean()

    # Generic fallback
    if len(series) == 12:
        s = pd.Series(series.values, index=range(1, 13))
        return s

    msg = "Series must have DatetimeIndex or be pre-computed 12-month curve."
    raise ValueError(msg)


def plot_climatology_comparison(
    observed: pd.Series,
    modeled: pd.Series,
    variable: Literal["precipitation", "temperature"] = "precipitation",
    variable_label: Optional[str] = None,
    station_name: Optional[str] = None,
    model_name: str = "ERA5",
    ax: Optional["matplotlib.axes.Axes"] = None,
    show_regime: bool = True,
    figsize: Tuple[float, float] = (7.5, 4.5),
) -> "matplotlib.axes.Axes":
    """Plot observed vs modeled 12-month annual climatological cycle.

    When variable is 'precipitation', automatically classifies and displays rainfall
    regimes (bimodal, unimodal, multimodal, indeterminate) for both series.

    Args:
        observed: Reference observational time series or 12-month climatology.
        modeled: Modeled / reanalysis time series or 12-month climatology.
        variable: Climate variable type ('precipitation' or 'temperature').
        variable_label: Custom axis label text. If None, default unit is selected.
        station_name: Optional station name for plot title.
        model_name: Name of the reanalysis / model dataset.
        ax: Optional Matplotlib Axes. If None, creates a new Figure and Axes.
        show_regime: Whether to display regime badges when variable == 'precipitation'.
        figsize: Figure dimension tuple in inches.

    Returns:
        The Matplotlib Axes containing the climatology plot.
    """
    plt = require_matplotlib()

    if ax is None:
        _, primary_ax = plt.subplots(figsize=figsize)
    else:
        primary_ax = ax

    obs_clim = _extract_monthly_climatology_curve(observed)
    mod_clim = _extract_monthly_climatology_curve(modeled)

    months = np.arange(1, 13)
    model_color = (
        OKABE_ITO["blue"] if variable == "precipitation" else OKABE_ITO["vermilion"]
    )

    # Plot P10-P90 interannual variability bands if multi-year series provided
    if isinstance(observed.index, pd.DatetimeIndex):
        obs_p10 = observed.groupby(observed.index.month).quantile(0.10)
        obs_p90 = observed.groupby(observed.index.month).quantile(0.90)
        p10_vals = [obs_p10.get(m, np.nan) for m in months]
        p90_vals = [obs_p90.get(m, np.nan) for m in months]
        primary_ax.fill_between(
            months,
            p10_vals,
            p90_vals,
            color=OKABE_ITO["black"],
            alpha=0.15,
            label="Obs P10–P90",
            zorder=1,
        )

    if isinstance(modeled.index, pd.DatetimeIndex):
        mod_p10 = modeled.groupby(modeled.index.month).quantile(0.10)
        mod_p90 = modeled.groupby(modeled.index.month).quantile(0.90)
        p10_mod = [mod_p10.get(m, np.nan) for m in months]
        p90_mod = [mod_p90.get(m, np.nan) for m in months]
        primary_ax.fill_between(
            months,
            p10_mod,
            p90_mod,
            color=model_color,
            alpha=0.10,
            label=f"{model_name} P10–P90",
            zorder=1,
        )

    # Plot observed reference curve
    primary_ax.plot(
        months,
        obs_clim.values,
        color=OKABE_ITO["black"],
        marker="o",
        markersize=6,
        linewidth=2.0,
        label="Observed (IDEAM)",
        zorder=4,
    )

    # Plot modeled curve
    primary_ax.plot(
        months,
        mod_clim.values,
        color=model_color,
        marker="s",
        markersize=5,
        linewidth=1.8,
        linestyle="--",
        label=f"Modeled ({model_name})",
        zorder=3,
    )

    # Plot detected peaks if variable == "precipitation"
    if variable == "precipitation":
        try:
            obs_peaks = annual_cycle_peaks(obs_clim)
            for p in obs_peaks:
                primary_ax.scatter(
                    p,
                    obs_clim.loc[p],
                    marker="*",
                    s=120,
                    color=OKABE_ITO["black"],
                    edgecolors="white",
                    linewidths=1.0,
                    zorder=6,
                )
            mod_peaks = annual_cycle_peaks(mod_clim)
            for p in mod_peaks:
                primary_ax.scatter(
                    p,
                    mod_clim.loc[p],
                    marker="v",
                    s=70,
                    color=model_color,
                    edgecolors="white",
                    linewidths=0.8,
                    zorder=6,
                )
        except Exception:
            pass

    # Resolve y-axis label
    if variable_label is not None:
        y_text = variable_label
    elif variable == "precipitation":
        y_text = "Precipitation (mm/month)"
    else:
        y_text = "Temperature (°C)"
    primary_ax.set_ylabel(y_text, fontsize=9.5)

    # Build title with detected regimes
    if variable == "precipitation" and show_regime:
        try:
            reg_obs = classify_rainfall_regime(obs_clim).capitalize()
            reg_mod = classify_rainfall_regime(mod_clim).capitalize()
            prefix = f"{station_name}\n" if station_name else ""
            title_text = f"{prefix}Obs: {reg_obs} | {model_name}: {reg_mod}"
        except Exception:
            title_text = station_name or "Annual Climatological Cycle"
    elif station_name:
        title_text = f"{station_name} - Climatological Cycle"
    else:
        title_text = "Annual Climatological Cycle"

    primary_ax.set_title(title_text, fontsize=10.0, fontweight="bold", pad=8)
    primary_ax.set_xticks(months)
    primary_ax.set_xticklabels(MONTH_LABELS, fontsize=8.5)
    primary_ax.set_xlabel("Month", fontsize=9.5)
    primary_ax.grid(True, linestyle=":", alpha=0.6, color="#B0BEC5")
    primary_ax.legend(frameon=True, fontsize=8.0, loc="best")

    return primary_ax


def plot_climatology_multiples(
    stations_data: Dict[str, Tuple[pd.Series, pd.Series]],
    variable: Literal["precipitation", "temperature"] = "precipitation",
    variable_label: Optional[str] = None,
    model_name: str = "ERA5",
    ncols: int = 3,
    show_regime: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple["matplotlib.figure.Figure", np.ndarray]:
    """Plot small multiples grid of annual climatologies across multiple stations.

    Args:
        stations_data: Dict mapping station identifier to (observed, modeled).
        variable: Climate variable ('precipitation' or 'temperature').
        variable_label: Optional y-axis label.
        model_name: Name of reanalysis model.
        ncols: Number of columns in subplot grid.
        show_regime: Whether to display detected regime badges in titles.
        figsize: Optional figure size tuple. Defaults to dynamic calculation.

    Returns:
        Tuple of (Figure, array of Axes).
    """
    plt = require_matplotlib()

    n_stations = len(stations_data)
    nrows = int(np.ceil(n_stations / ncols))

    if figsize is None:
        fig_w = ncols * 3.5
        fig_h = nrows * 2.8
        figsize = (fig_w, fig_h)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
    axes_flat = np.atleast_1d(axes).flatten()

    for idx, (st_name, (obs_s, mod_s)) in enumerate(stations_data.items()):
        ax = axes_flat[idx]
        plot_climatology_comparison(
            observed=obs_s,
            modeled=mod_s,
            variable=variable,
            variable_label=variable_label if idx % ncols == 0 else "",
            station_name=st_name,
            model_name=model_name,
            ax=ax,
            show_regime=show_regime,
        )
        if idx % ncols != 0:
            ax.set_ylabel("")

    # Hide unused subplot slots
    for empty_idx in range(n_stations, len(axes_flat)):
        axes_flat[empty_idx].set_visible(False)

    fig.tight_layout()
    return fig, axes_flat


def plot_taylor_diagram(
    stats_list: Union[
        TaylorStatistics, List[TaylorStatistics], Dict[str, TaylorStatistics]
    ],
    labels: Optional[List[str]] = None,
    ref_std: Optional[float] = None,
    normalize: bool = False,
    ax: Optional["matplotlib.axes.Axes"] = None,
    title: Optional[str] = "Taylor Diagram",
    figsize: Tuple[float, float] = (7.0, 7.0),
) -> "matplotlib.axes.Axes":
    """Generate a Taylor diagram evaluating standard deviation, correlation, and CRMSE.

    Plots the geometric synthesis of model performance following Karl E. Taylor (2001)
    in Cartesian space with labeled correlation rays, standard deviation arcs,
    and centered root mean square error (CRMSE) contours.

    Args:
        stats_list: Single TaylorStatistics, list, or dict mapping label to stats.
        labels: Optional labels for each statistics record.
        ref_std: Reference standard deviation. Inferred from stats if None.
        normalize: Whether diagram represents normalized metrics (reference std = 1.0).
        ax: Optional Matplotlib Axes. If None, creates a new Figure and Axes.
        title: Plot title text.
        figsize: Figure dimension tuple in inches.

    Returns:
        The Matplotlib Axes containing the Taylor diagram.
    """
    plt = require_matplotlib()

    if ax is None:
        _, primary_ax = plt.subplots(figsize=figsize)
    else:
        primary_ax = ax

    # Normalize stats input into list of (label, TaylorStatistics)
    records: List[Tuple[str, TaylorStatistics]] = []
    if isinstance(stats_list, tuple) and len(stats_list) == 2:
        obs, mod = stats_list
        t_stat = taylor_statistics(obs, mod, normalize=normalize)
        lbl = labels[0] if (labels and len(labels) > 0) else "Model"
        records.append((lbl, t_stat))
    elif isinstance(stats_list, dict):
        for k, v in stats_list.items():
            records.append((str(k), v))
    elif isinstance(stats_list, list):
        for idx, item in enumerate(stats_list):
            lbl = labels[idx] if (labels and idx < len(labels)) else f"Model {idx + 1}"
            records.append((lbl, item))
    elif isinstance(stats_list, TaylorStatistics):
        lbl = labels[0] if (labels and len(labels) > 0) else "Model"
        records.append((lbl, stats_list))

    # Determine reference standard deviation
    if normalize:
        target_ref_std = 1.0
    elif ref_std is not None:
        target_ref_std = float(ref_std)
    elif records:
        target_ref_std = records[0][1].std_obs
    else:
        target_ref_std = 1.0

    # Determine max radius
    all_stds = [target_ref_std] + [
        r[1].std_model / (r[1].std_obs if normalize else 1.0) for r in records
    ]
    max_radius = float(np.max(all_stds) * 1.35)
    max_radius = max(max_radius, target_ref_std * 1.3)

    # 1. Outer boundary arc (first quadrant: 0 to pi/2)
    arc_theta = np.linspace(0, np.pi / 2, 200)
    primary_ax.plot(
        max_radius * np.cos(arc_theta),
        max_radius * np.sin(arc_theta),
        color="#37474F",
        linewidth=1.2,
    )

    # 2. Concentric standard deviation circles
    std_ticks = np.linspace(0, max_radius, 5)[1:]
    for s_val in std_ticks:
        primary_ax.plot(
            s_val * np.cos(arc_theta),
            s_val * np.sin(arc_theta),
            color="#CFD8DC",
            linestyle="--",
            linewidth=0.7,
        )
        primary_ax.text(
            s_val,
            -0.03 * max_radius,
            f"{s_val:.2g}",
            fontsize=7,
            ha="center",
            color="#546E7A",
        )

    # 3. Correlation rays and tick labels
    corr_ticks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    for c_val in corr_ticks:
        theta_c = np.arccos(c_val)
        primary_ax.plot(
            [0, max_radius * np.cos(theta_c)],
            [0, max_radius * np.sin(theta_c)],
            color="#ECEFF1",
            linestyle="-",
            linewidth=0.6,
        )
        # Position label just outside outer arc
        x_lbl = max_radius * 1.03 * np.cos(theta_c)
        y_lbl = max_radius * 1.03 * np.sin(theta_c)
        primary_ax.text(
            x_lbl,
            y_lbl,
            f"{c_val}",
            fontsize=7,
            ha="center",
            va="center",
            color="#37474F",
        )

    # Outer arc label
    primary_ax.text(
        max_radius * 0.75,
        max_radius * 0.75,
        "Correlation",
        rotation=-45,
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color="#37474F",
    )

    # 4. Centered RMS Error (CRMSE) circles centered at (target_ref_std, 0)
    phi_crmse = np.linspace(0, np.pi, 200)
    crmse_levels = np.linspace(0.25 * target_ref_std, 1.25 * target_ref_std, 4)
    for e_val in crmse_levels:
        cx = target_ref_std + e_val * np.cos(phi_crmse)
        cy = e_val * np.sin(phi_crmse)
        valid = (cx >= 0) & (cy >= 0) & (cx**2 + cy**2 <= max_radius**2)
        if np.any(valid):
            primary_ax.plot(
                cx[valid],
                cy[valid],
                color="#81C784",
                linestyle=":",
                linewidth=0.8,
            )
            mid_idx = np.where(valid)[0][len(np.where(valid)[0]) // 2]
            primary_ax.text(
                cx[mid_idx],
                cy[mid_idx],
                f"E'={e_val:.2g}",
                fontsize=6.5,
                color="#388E3C",
                ha="center",
            )

    # 5. Reference observation marker at (target_ref_std, 0)
    primary_ax.scatter(
        [target_ref_std],
        [0],
        color=OKABE_ITO["black"],
        marker="*",
        s=120,
        label="Reference (Obs)",
        zorder=10,
        edgecolors="white",
        linewidths=0.5,
    )

    # 6. Plot model test points
    palette_colors = [
        OKABE_ITO["blue"],
        OKABE_ITO["orange"],
        OKABE_ITO["bluish_green"],
        OKABE_ITO["reddish_purple"],
        OKABE_ITO["yellow"],
        OKABE_ITO["sky_blue"],
    ]

    for idx, (label_txt, t_stat) in enumerate(records):
        s_m = t_stat.std_model / (t_stat.std_obs if normalize else 1.0)
        c_m = np.clip(t_stat.correlation, 0.0, 1.0)
        theta_m = np.arccos(c_m)

        x_pt = s_m * np.cos(theta_m)
        y_pt = s_m * np.sin(theta_m)
        pt_color = palette_colors[idx % len(palette_colors)]

        primary_ax.scatter(
            [x_pt],
            [y_pt],
            color=pt_color,
            marker="o",
            s=64,
            label=label_txt,
            zorder=8,
            edgecolors="black",
            linewidths=0.5,
        )

    # Formatting axes
    primary_ax.set_aspect("equal", adjustable="box")
    primary_ax.set_xlim(0, max_radius * 1.15)
    primary_ax.set_ylim(0, max_radius * 1.15)
    primary_ax.set_xlabel(
        "Normalized Standard Deviation" if normalize else "Standard Deviation",
        fontsize=9,
    )
    primary_ax.set_ylabel(
        "Normalized Standard Deviation" if normalize else "Standard Deviation",
        fontsize=9,
    )

    # Clean borders
    primary_ax.spines["top"].set_visible(False)
    primary_ax.spines["right"].set_visible(False)
    primary_ax.spines["left"].set_position(("data", 0))
    primary_ax.spines["bottom"].set_position(("data", 0))

    if title:
        primary_ax.set_title(title, fontsize=11, fontweight="bold", pad=14)

    # Suppress duplicate '0' at origin on y-axis
    import matplotlib.ticker as ticker

    primary_ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda val, pos: "" if np.isclose(val, 0, atol=1e-3) else f"{val:g}"
        )
    )

    primary_ax.legend(
        bbox_to_anchor=(1.05, 1.0),
        loc="upper left",
        frameon=True,
        fontsize=8,
    )

    return primary_ax


def plot_obs_vs_model(
    observed: pd.Series,
    modeled: pd.Series,
    station_name: Optional[str] = None,
    variable_label: str = "Value",
    model_name: str = "ERA5",
    ax: Optional["matplotlib.axes.Axes"] = None,
    show_metrics: bool = True,
    figsize: Tuple[float, float] = (6.0, 6.0),
) -> "matplotlib.axes.Axes":
    """Scatter plot of observed vs modeled series with regression and stats box.

    Args:
        observed: Reference observation time series.
        modeled: Reanalysis or simulation time series.
        station_name: Optional station name for plot title.
        variable_label: Label and unit for axes (e.g. 'Precipitation (mm/month)').
        model_name: Name of modeled dataset.
        ax: Optional Matplotlib Axes.
        show_metrics: Whether to include verification metrics summary text box.
        figsize: Figure dimensions in inches.

    Returns:
        The Matplotlib Axes containing the scatter plot.
    """
    plt = require_matplotlib()

    if ax is None:
        _, primary_ax = plt.subplots(figsize=figsize)
    else:
        primary_ax = ax

    # Align series
    df = pd.concat([observed.rename("obs"), modeled.rename("mod")], axis=1).dropna()
    x = df["obs"].to_numpy(dtype=float)
    y = df["mod"].to_numpy(dtype=float)

    # Scatter points
    primary_ax.scatter(
        x,
        y,
        color=OKABE_ITO["blue"],
        alpha=0.6,
        s=36,
        edgecolors="none",
        zorder=3,
        label="Monthly samples",
    )

    # 1:1 Identity reference line
    min_val = min(float(x.min()), float(y.min()))
    max_val = max(float(x.max()), float(y.max()))
    padding = (max_val - min_val) * 0.05
    line_min = min_val - padding
    line_max = max_val + padding

    primary_ax.plot(
        [line_min, line_max],
        [line_min, line_max],
        color="#78909C",
        linestyle="--",
        linewidth=1.2,
        label="1:1 Identity",
        zorder=2,
    )

    # Linear regression line
    if len(x) >= 2:
        res = stats.linregress(x, y)
        x_fit = np.linspace(line_min, line_max, 100)
        y_fit = res.slope * x_fit + res.intercept
        primary_ax.plot(
            x_fit,
            y_fit,
            color=OKABE_ITO["vermilion"],
            linewidth=1.4,
            label=f"Fit (m={res.slope:.2f})",
            zorder=4,
        )

    # Validation metrics box
    if show_metrics and len(x) >= 3:
        report = compute_validation_metrics(df["obs"], df["mod"])
        text_lines = [
            f"Pearson r = {report.pearson_r:.2f}",
            f"RMSE = {report.rmse:.2f}",
            f"Mean Bias = {report.bias:+.2f}",
            f"KGE = {report.kge:.2f}",
            f"n = {report.n_samples}",
        ]
        box_text = "\n".join(text_lines)
        primary_ax.text(
            0.05,
            0.95,
            box_text,
            transform=primary_ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="#FAFAFA",
                edgecolor="#CFD8DC",
                alpha=0.9,
            ),
        )

    primary_ax.set_xlim(line_min, line_max)
    primary_ax.set_ylim(line_min, line_max)
    primary_ax.set_aspect("equal", adjustable="box")

    primary_ax.set_xlabel(f"Observed {variable_label}", fontsize=9)
    primary_ax.set_ylabel(f"{model_name} {variable_label}", fontsize=9)

    title_txt = (
        f"{station_name} - Validation Scatter" if station_name else "Validation Scatter"
    )
    primary_ax.set_title(title_txt, fontsize=10.5, fontweight="bold", pad=8)
    primary_ax.grid(True, linestyle=":", alpha=0.5, color="#CFD8DC")
    primary_ax.legend(frameon=True, fontsize=8, loc="lower right")

    return primary_ax


def plot_missing_data_heatmap(
    series_or_df: Union[pd.Series, pd.DataFrame],
    title: Optional[str] = "Observational Data Completeness",
    ax: Optional["matplotlib.axes.Axes"] = None,
    figsize: Tuple[float, float] = (9.0, 5.0),
) -> "matplotlib.axes.Axes":
    """Generate temporal completeness heatmap matrix across years and calendar months.

    Args:
        series_or_df: Time series with DatetimeIndex, or Year x Month matrix DataFrame.
        title: Plot title text.
        ax: Optional Matplotlib Axes.
        figsize: Figure dimension tuple in inches.

    Returns:
        The Matplotlib Axes containing the completeness heatmap.
    """
    plt = require_matplotlib()

    if ax is None:
        _, primary_ax = plt.subplots(figsize=figsize)
    else:
        primary_ax = ax

    if isinstance(series_or_df, pd.Series):
        if not isinstance(series_or_df.index, pd.DatetimeIndex):
            msg = "Series must have a pandas DatetimeIndex."
            raise TypeError(msg)
        df_temp = pd.DataFrame(
            {
                "year": series_or_df.index.year,
                "month": series_or_df.index.month,
                "valid": (~series_or_df.isna()).astype(float),
            }
        )
        matrix = df_temp.pivot_table(
            index="year", columns="month", values="valid", aggfunc="mean"
        ).fillna(0.0)
    elif isinstance(series_or_df, pd.DataFrame):
        matrix = series_or_df.copy()
    else:
        msg = f"Unsupported input type: {type(series_or_df)}"
        raise ValueError(msg)

    # Plot binary or percentage presence
    im = primary_ax.imshow(
        matrix.values,
        cmap="Blues",
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        origin="upper",
    )

    primary_ax.set_xticks(np.arange(12))
    primary_ax.set_xticklabels(MONTH_LABELS, fontsize=8.5)
    primary_ax.set_yticks(np.arange(len(matrix.index)))
    primary_ax.set_yticklabels(matrix.index, fontsize=7.5)

    primary_ax.set_xlabel("Month", fontsize=9)
    primary_ax.set_ylabel("Year", fontsize=9)

    cbar = primary_ax.figure.colorbar(im, ax=primary_ax, pad=0.02, shrink=0.8)
    cbar.set_label("Completeness Ratio", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5)

    if title:
        primary_ax.set_title(title, fontsize=10.5, fontweight="bold", pad=8)

    return primary_ax


def plot_thermal_range(
    tmax: pd.Series,
    tmin: pd.Series,
    tmax_model: Optional[pd.Series] = None,
    tmin_model: Optional[pd.Series] = None,
    station_name: Optional[str] = None,
    model_name: str = "ERA5",
    ax: Optional["matplotlib.axes.Axes"] = None,
    figsize: Tuple[float, float] = (8.0, 4.5),
) -> "matplotlib.axes.Axes":
    """Plot monthly diurnal temperature range (DTR) and double difference profiles.

    Args:
        tmax: Observed maximum temperature time series.
        tmin: Observed minimum temperature time series.
        tmax_model: Optional modeled maximum temperature series.
        tmin_model: Optional modeled minimum temperature series.
        station_name: Optional station name for plot title.
        model_name: Name of modeled dataset.
        ax: Optional Matplotlib Axes.
        figsize: Figure dimension tuple in inches.

    Returns:
        The Matplotlib Axes containing the thermal range profiles.
    """
    plt = require_matplotlib()

    if ax is None:
        _, primary_ax = plt.subplots(figsize=figsize)
    else:
        primary_ax = ax

    obs_dtr = compute_dtr(tmax, tmin)
    obs_dtr_clim = _extract_monthly_climatology_curve(obs_dtr)
    months = np.arange(1, 13)

    primary_ax.plot(
        months,
        obs_dtr_clim.values,
        color=OKABE_ITO["black"],
        marker="o",
        linewidth=2.0,
        label="Observed DTR",
        zorder=4,
    )

    if tmax_model is not None and tmin_model is not None:
        mod_dtr = compute_dtr(tmax_model, tmin_model)
        mod_dtr_clim = _extract_monthly_climatology_curve(mod_dtr)
        primary_ax.plot(
            months,
            mod_dtr_clim.values,
            color=OKABE_ITO["blue"],
            marker="s",
            linestyle="--",
            linewidth=1.8,
            label=f"{model_name} DTR",
            zorder=3,
        )

        # Double difference curve
        ddiff = compute_double_difference(obs_dtr, mod_dtr)
        ddiff_clim = _extract_monthly_climatology_curve(ddiff)
        primary_ax.plot(
            months,
            ddiff_clim.values,
            color=OKABE_ITO["reddish_purple"],
            marker="^",
            linestyle=":",
            linewidth=1.5,
            label="Double Difference (Bias)",
            zorder=3,
        )

    title_txt = (
        f"{station_name} - Diurnal Thermal Range (DTR)"
        if station_name
        else "Diurnal Thermal Range (DTR)"
    )
    primary_ax.set_title(title_txt, fontsize=10.5, fontweight="bold", pad=8)
    primary_ax.set_xticks(months)
    primary_ax.set_xticklabels(MONTH_LABELS, fontsize=8.5)
    primary_ax.set_xlabel("Month", fontsize=9)
    primary_ax.set_ylabel("Temperature Range (°C)", fontsize=9)
    primary_ax.grid(True, linestyle=":", alpha=0.6, color="#B0BEC5")
    primary_ax.legend(frameon=True, fontsize=8, loc="best")

    return primary_ax
