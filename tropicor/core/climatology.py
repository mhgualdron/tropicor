"""Annual climatological curves, seasonal cycle geometry, and curve metrics.

Replicates and extends the 12-month climatology cycle analysis, circular peak
detection, and geometric curve separation algorithms from UNAL thesis Entregas 7 & 10.
"""

import warnings
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

EPSILON: float = 1e-6


def compute_monthly_climatology(
    series: pd.Series,
    min_years: int = 10,
    min_obs_per_month: int = 5,
) -> pd.Series:
    """Compute the 12-month mean climatological annual cycle profile.

    Groups multi-year monthly observations by calendar month (1 to 12)
    and computes the arithmetic mean for each month, enforcing strict
    observational completeness thresholds.

    Note:
        If any calendar month has fewer than `min_obs_per_month` valid records,
        it will be assigned NaN and emit a warning. Downstream functions that
        require a complete 12-month profile (:func:`climatological_curve_distance`,
        :func:`annual_cycle_peaks`) will subsequently raise a ValueError
        signaling that the curve is incomplete. Adjust `min_obs_per_month` or inspect
        observational gaps if needed.

    Args:
        series: Time series with a pandas DatetimeIndex.
        min_years: Minimum number of unique calendar years required. Defaults to 10.
        min_obs_per_month: Minimum number of valid non-NaN observations required
            for a specific month to compute its mean. Defaults to 5. Months with
            fewer observations are assigned NaN.

    Returns:
        pd.Series with integer index 1..12 (months Jan-Dec) and index name 'month'.

    Raises:
        TypeError: If series index is not a pandas DatetimeIndex.
        ValueError: If series is empty or spans fewer than min_years.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("Series index must be a pandas DatetimeIndex.")

    valid_series = series.dropna()
    if valid_series.empty:
        raise ValueError("Series contains no valid non-NaN observations.")

    n_years = valid_series.index.year.nunique()
    if n_years < min_years:
        raise ValueError(
            f"Series spans {n_years} unique years with data, but min_years={min_years} "
            f"is required to establish a climatological baseline."
        )

    # Compute count and mean per calendar month
    counts = series.groupby(series.index.month).count()
    means = series.groupby(series.index.month).mean()

    # Reindex explicitly to ensure standard 1..12 index
    counts = counts.reindex(range(1, 13), fill_value=0)
    means = means.reindex(range(1, 13))

    # Apply completeness filter per month
    incomplete_mask = counts < min_obs_per_month
    if incomplete_mask.any():
        incomplete_months = list(counts[incomplete_mask].index)
        warnings.warn(
            f"Months {incomplete_months} have fewer than {min_obs_per_month} "
            "valid observations and have been assigned NaN. Downstream curve functions "
            "will require complete 12-month profiles.",
            UserWarning,
            stacklevel=2,
        )
        means[incomplete_mask] = np.nan

    means.index.name = "month"
    means.name = series.name or "climatology"
    return means


def climatological_curve_distance(
    obs_clim: pd.Series,
    mod_clim: pd.Series,
    normalize: Optional[Literal["mean", "amplitude", "std"]] = None,
) -> float:
    """Compute Euclidean distance between two 12-month climatological curves.

    Implements the geometric curve distance evaluated in UNAL thesis Entregas 7 & 10:
        d = sqrt( sum_{m=1}^{12} (mod_clim(m) - obs_clim(m))^2 )

    Optionally normalizes by observed mean, amplitude, or standard deviation
    to allow fair cross-basin comparisons between contrasting hydroclimatic regimes.

    Note:
        Both curves must have valid (non-NaN) values across all 12 calendar months.
        If either curve contains NaNs (e.g. due to observational gaps filtered by
        `min_obs_per_month`), this function raises a ValueError.

    Args:
        obs_clim: 12-month observed climatological series.
        mod_clim: 12-month modeled or reanalysis climatological series.
        normalize: Optional normalization strategy:
            - None: Raw dimensional Euclidean distance.
            - 'mean': Normalized by observed mean (|mean(obs)|).
            - 'amplitude': Normalized by observed cycle amplitude (max(obs) - min(obs)).
            - 'std': Normalized by sample standard deviation of observed cycle.

    Returns:
        Euclidean distance (dimensional or normalized) between annual cycles.

    Raises:
        ValueError: If curves do not contain 12 synchronized non-NaN monthly means,
            or if normalize mode is invalid.
    """
    df_sync = pd.concat(
        [obs_clim.rename("obs"), mod_clim.rename("mod")], axis=1
    ).dropna()

    if len(df_sync) < 12:
        missing_count = 12 - len(df_sync)
        raise ValueError(
            "Both climatological curves must have 12 synchronous non-NaN monthly "
            f"means. Found only {len(df_sync)} valid months ({missing_count} missing "
            "or NaN). Check for observational gaps or adjust `min_obs_per_month`."
        )

    y_obs = df_sync["obs"].to_numpy(dtype=np.float64)
    y_mod = df_sync["mod"].to_numpy(dtype=np.float64)

    raw_dist = float(np.sqrt(np.sum((y_mod - y_obs) ** 2)))

    if normalize is None:
        return raw_dist

    norm_mode = normalize.lower()
    if norm_mode == "mean":
        denom = float(np.abs(np.mean(y_obs)))
        if denom < EPSILON:
            warnings.warn(
                "Observed climatology mean is near zero; returning raw distance.",
                UserWarning,
                stacklevel=2,
            )
            return raw_dist
        return float(raw_dist / denom)

    elif norm_mode == "amplitude":
        denom = float(np.max(y_obs) - np.min(y_obs))
        if denom < EPSILON:
            warnings.warn(
                "Observed climatology amplitude is near zero; returning raw distance.",
                UserWarning,
                stacklevel=2,
            )
            return raw_dist
        return float(raw_dist / denom)

    elif norm_mode == "std":
        denom = float(np.std(y_obs, ddof=1))
        if denom < EPSILON:
            warnings.warn(
                "Observed climatology variance is near zero; returning raw distance.",
                UserWarning,
                stacklevel=2,
            )
            return raw_dist
        return float(raw_dist / denom)

    else:
        raise ValueError(
            f"Invalid normalize mode '{normalize}'. "
            "Allowed modes: None, 'mean', 'amplitude', 'std'."
        )


def annual_cycle_amplitude(clim: pd.Series) -> float:
    """Compute the amplitude of the annual climatological cycle (max - min).

    Args:
        clim: Climatological series (e.g. 12-month profile).

    Returns:
        Difference between maximum and minimum climatological values.

    Raises:
        ValueError: If the series contains no valid non-NaN values.
    """
    valid = clim.dropna()
    if valid.empty:
        raise ValueError("Climatology series contains no valid non-NaN values.")

    return float(valid.max() - valid.min())


def annual_cycle_phase(
    clim: pd.Series,
    mode: Literal["peak", "max", "trough", "valley", "min"] = "peak",
) -> int:
    """Identify the month of global peak or trough in the annual cycle.

    Note:
        For bimodal regimes (such as the Colombian Andes with two wet seasons),
        this function identifies ONLY the single global extremum. To detect both
        peaks and evaluate seasonal distortion, use :func:`annual_cycle_peaks`.

    Args:
        clim: 12-month climatological series with month index (1..12).
        mode: Extremum to detect: 'peak'/'max' or 'trough'/'valley'/'min'.

    Returns:
        Calendar month (integer 1..12) corresponding to the global extremum.

    Raises:
        ValueError: If series is empty or mode is invalid.
    """
    valid = clim.dropna()
    if valid.empty:
        raise ValueError("Climatology series contains no valid non-NaN values.")

    mode_lower = mode.lower()
    if mode_lower in {"peak", "max"}:
        return int(valid.idxmax())
    elif mode_lower in {"trough", "valley", "min"}:
        return int(valid.idxmin())
    else:
        raise ValueError(
            f"Invalid mode '{mode}'. "
            "Allowed modes: 'peak', 'max', 'trough', 'valley', 'min'."
        )


def annual_cycle_peaks(
    clim: pd.Series,
    min_prominence: Optional[float] = None,
) -> List[int]:
    """Detect all local maxima in a 12-month cycle using circular boundary padding.

    Treats the annual cycle as circular (December m=12 is adjacent to January m=1).
    Avoids edge clipping by evaluating peaks over a 3x tiled series (36 months)
    and extracting peaks located in the central 12-month replica.

    Args:
        clim: 12-month climatological profile with index 1..12.
        min_prominence: Minimum topographic prominence required for a peak.
            If None, defaults to 10% of cycle amplitude (0.1 * (max - min)).

    Returns:
        Sorted list of integer calendar months corresponding to local maxima.

    Raises:
        ValueError: If clim does not contain 12 non-NaN monthly means.
    """
    valid = clim.dropna()
    if len(valid) < 12:
        raise ValueError(
            "Annual cycle must contain 12 valid non-NaN monthly means. "
            f"Found {len(valid)}. Check for incomplete months or missing observations."
        )

    # Ensure aligned 1..12 vector
    vals = valid.reindex(range(1, 13)).to_numpy(dtype=np.float64)
    amp = float(np.max(vals) - np.min(vals))

    if min_prominence is None:
        prominence = 0.10 * amp if amp > EPSILON else None
    else:
        prominence = float(min_prominence)

    # 3x circular tiling: replica 0 (0..11), replica 1 (12..23), replica 2 (24..35)
    tiled_vals = np.tile(vals, 3)

    find_kwargs = {}
    if prominence is not None and prominence > 0:
        find_kwargs["prominence"] = prominence

    peaks, _ = find_peaks(tiled_vals, **find_kwargs)

    # Filter peaks that fall inside the middle replica [12, 23]
    # Map index i to calendar month (1..12): (i - 12) + 1
    central_peaks = [int((p - 12) + 1) for p in peaks if 12 <= p < 24]
    return sorted(central_peaks)


def classify_rainfall_regime(
    clim: pd.Series,
    min_prominence: Optional[float] = None,
) -> Literal["unimodal", "bimodal", "multimodal", "indeterminate"]:
    """Classify precipitation annual cycle regime based on circular peak count.

    Important:
        This classification is designed exclusively for precipitation series.
        In tropical latitudes, temperature cycles are extremely muted (1-3°C amplitude),
        meaning application of peak prominence classification to temperature is
        physically inappropriate and will yield noise or indeterminate outputs.

    Differentiates between classic tropical rainfall regimes:
        - 'unimodal': 1 significant peak (e.g. Orinoquía, Amazonía, Caribe).
        - 'bimodal': 2 significant peaks (e.g. Colombian Andes / ITCZ double passage).
        - 'multimodal': > 2 significant peaks (irregular / transition zones).
        - 'indeterminate': 0 significant peaks (uniform rainfall / low seasonality).

    Args:
        clim: 12-month precipitation climatology profile.
        min_prominence: Prominence threshold passed to :func:`annual_cycle_peaks`.

    Returns:
        One of 'unimodal', 'bimodal', 'multimodal', or 'indeterminate'.
    """
    peaks = annual_cycle_peaks(clim, min_prominence=min_prominence)
    n_peaks = len(peaks)

    if n_peaks == 1:
        return "unimodal"
    elif n_peaks == 2:
        return "bimodal"
    elif n_peaks > 2:
        return "multimodal"
    else:
        return "indeterminate"
