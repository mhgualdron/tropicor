"""Benchmark evaluator helpers under Protocol Appendix J and K.

Provides pure helpers for the benchmark evaluator:
- Valid-year rule of J.7.1
- Calendar-year masking to valid years
- Ledger-free 5-year block partitioning
- J.7.2 sample status classification
"""

from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    cast,
)

import numpy as np
import pandas as pd

from tropicor.core.protocol import (
    DEFAULT_BOOTSTRAP_REPLICAS,
    DEFAULT_BOOTSTRAP_SEED,
    MIN_BOOTSTRAP_BLOCKS,
    CrossingDiagnostic,
    EvaluationScorecard,
    GlobalOutcome,
    apply_weaker_of_two_concordance,
    check_temperature_physical_gate,
    compute_annual_block_bootstrap_temperature,
    evaluate_temperature_scorecard,
)
from tropicor.downscale.temperature import (
    Level1MonthlyAdditiveTemperature,
    Level2MeanStateDTRScalingTemperature,
    Level3QuantileMappingTemperature,
)

BENCHMARK_STATION_IDS: Tuple[str, str] = ("21205710", "23215030")

SAMPLE_STATUS_OK: str = "ok"
SAMPLE_STATUS_INSUFFICIENT: str = "not_established_insufficient_sample"
SAMPLE_STATUS_ABORTED: str = "aborted_safeguard"


def valid_years_j71(
    tmax_obs: pd.Series,
    tmin_obs: pd.Series,
    tmax_mod: pd.Series,
    tmin_mod: pd.Series,
) -> List[int]:
    """Return sorted calendar years containing at least one valid day.

    Under Protocol J.7.1, a calendar year is valid if and only if there is at least
    one calendar day on which all four temperature series (tmax_obs, tmin_obs,
    tmax_mod, tmin_mod) are simultaneously non-NaN.

    Parameters
    ----------
    tmax_obs : pd.Series
        Observed daily maximum temperature series.
    tmin_obs : pd.Series
        Observed daily minimum temperature series.
    tmax_mod : pd.Series
        Model/predicted daily maximum temperature series.
    tmin_mod : pd.Series
        Model/predicted daily minimum temperature series.

    Returns
    -------
    List[int]
        Sorted list of valid calendar years.

    Raises
    ------
    ValueError
        If series indexes are not pd.DatetimeIndex or are not identical.
    """
    for s, name in (
        (tmax_obs, "tmax_obs"),
        (tmin_obs, "tmin_obs"),
        (tmax_mod, "tmax_mod"),
        (tmin_mod, "tmin_mod"),
    ):
        if not isinstance(s.index, pd.DatetimeIndex):
            raise ValueError(f"{name} index must be a pd.DatetimeIndex.")

    base_index = tmax_obs.index
    if not (
        tmin_obs.index.equals(base_index)
        and tmax_mod.index.equals(base_index)
        and tmin_mod.index.equals(base_index)
    ):
        raise ValueError("All four series must have identical indexes.")
    assert isinstance(base_index, pd.DatetimeIndex)

    valid_mask = (
        tmax_obs.notna() & tmin_obs.notna() & tmax_mod.notna() & tmin_mod.notna()
    )
    if not valid_mask.any():
        return []

    valid_years = sorted(int(y) for y in set(base_index[valid_mask].year))
    return valid_years


def mask_to_valid_years(
    tmax_obs: pd.Series,
    tmin_obs: pd.Series,
    tmax_mod: pd.Series,
    tmin_mod: pd.Series,
    valid_years: Sequence[int],
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return copies of the 4 series with non-valid calendar years masked to NaN.

    Inputs are never modified.

    Parameters
    ----------
    tmax_obs : pd.Series
        Observed daily maximum temperature series.
    tmin_obs : pd.Series
        Observed daily minimum temperature series.
    tmax_mod : pd.Series
        Model/predicted daily maximum temperature series.
    tmin_mod : pd.Series
        Model/predicted daily minimum temperature series.
    valid_years : Sequence[int]
        Sequence of valid calendar years to preserve.

    Returns
    -------
    Tuple[pd.Series, pd.Series, pd.Series, pd.Series]
        Copies of the 4 series with values in non-valid years set to NaN.

    Raises
    ------
    ValueError
        If series indexes are not pd.DatetimeIndex or are not identical.
    """
    for s, name in (
        (tmax_obs, "tmax_obs"),
        (tmin_obs, "tmin_obs"),
        (tmax_mod, "tmax_mod"),
        (tmin_mod, "tmin_mod"),
    ):
        if not isinstance(s.index, pd.DatetimeIndex):
            raise ValueError(f"{name} index must be a pd.DatetimeIndex.")

    base_index = tmax_obs.index
    if not (
        tmin_obs.index.equals(base_index)
        and tmax_mod.index.equals(base_index)
        and tmin_mod.index.equals(base_index)
    ):
        raise ValueError("All four series must have identical indexes.")
    assert isinstance(base_index, pd.DatetimeIndex)

    valid_set = set(valid_years)
    invalid_mask = ~base_index.year.isin(valid_set)

    s1 = tmax_obs.copy()
    s2 = tmin_obs.copy()
    s3 = tmax_mod.copy()
    s4 = tmin_mod.copy()

    if invalid_mask.any():
        s1.loc[invalid_mask] = np.nan
        s2.loc[invalid_mask] = np.nan
        s3.loc[invalid_mask] = np.nan
        s4.loc[invalid_mask] = np.nan

    return s1, s2, s3, s4


def five_year_blocks(years: Sequence[int]) -> List[Tuple[int, int]]:
    """Partition calendar years into contiguous 5-year blocks.

    Follows the same block algorithm as contiguous_5year_block_cv in
    protocol.py (blocks of 5 from the first year; when remaining years
    number fewer than 10 and are not exactly 5, the last block takes all
    of them), but with no validation ledger access.

    Parameters
    ----------
    years : Sequence[int]
        Sequence of calendar years.

    Returns
    -------
    List[Tuple[int, int]]
        List of (first_year, last_year) tuples defining the blocks.

    Raises
    ------
    ValueError
        If fewer than 5 unique years are provided.
    """
    sorted_years = sorted(set(years))
    n_years = len(sorted_years)
    if n_years < 5:
        raise ValueError(
            f"5-year block partition requires at least 5 years, found {n_years}."
        )

    blocks: List[Tuple[int, int]] = []
    i = 0
    while i < n_years:
        rem = n_years - i
        if rem < 10 and rem != 5:
            block = sorted_years[i:]
            blocks.append((min(block), max(block)))
            break
        else:
            block = sorted_years[i : i + 5]
            blocks.append((min(block), max(block)))
            i += 5

    return blocks


def track_sample_status(n_valid_years: int) -> str:
    """Classify sample status under Protocol J.7.2.

    Parameters
    ----------
    n_valid_years : int
        Number of valid calendar years under J.7.1.

    Returns
    -------
    str
        SAMPLE_STATUS_OK if n_valid_years >= MIN_BOOTSTRAP_BLOCKS, else
        SAMPLE_STATUS_INSUFFICIENT.
    """
    if n_valid_years >= MIN_BOOTSTRAP_BLOCKS:
        return SAMPLE_STATUS_OK
    return SAMPLE_STATUS_INSUFFICIENT


@dataclass(frozen=True)
class LevelPrediction:
    """Predicted daily extremes and optional crossing diagnostic for a level.

    Parameters
    ----------
    tmax : pd.Series
        Predicted daily maximum temperature series.
    tmin : pd.Series
        Predicted daily minimum temperature series.
    crossing_diagnostic : Optional[CrossingDiagnostic]
        Diagnostic from applying the universal crossing rule, if available.
    """

    tmax: pd.Series
    tmin: pd.Series
    crossing_diagnostic: Optional[CrossingDiagnostic] = None


@dataclass(frozen=True)
class TrackEvaluation:
    """Evaluation summary for all levels on a single evaluation track.

    Parameters
    ----------
    station_id : str
        Station identifier.
    valid_years : List[int]
        Sorted list of valid calendar years under J.7.1.
    sample_status : str
        Sample status (SAMPLE_STATUS_OK, SAMPLE_STATUS_INSUFFICIENT, or
        SAMPLE_STATUS_ABORTED).
    scorecards : Dict[str, Optional[EvaluationScorecard]]
        Scorecards for levels ("level0", "level1", "level2", "level3_winner",
        "level3_constant"). None when sample_status != SAMPLE_STATUS_OK.
    outcomes : Dict[str, GlobalOutcome]
        Global outcomes for each level. Set to GlobalOutcome.NOT_ESTABLISHED
        when sample_status != SAMPLE_STATUS_OK.
    identical_to_winner : bool
        Whether Level 3 constant sensitivity is identical to the Level 3 winner.
    reason : Optional[str]
        Explanation when sample_status != SAMPLE_STATUS_OK, otherwise None.
    """

    station_id: str
    valid_years: List[int]
    sample_status: str
    scorecards: Dict[str, Optional[EvaluationScorecard]]
    outcomes: Dict[str, GlobalOutcome]
    identical_to_winner: bool
    reason: Optional[str] = None


def evaluate_track_levels(
    station_id: str,
    obs_tmax: pd.Series,
    obs_tmin: pd.Series,
    raw_tmax: pd.Series,
    raw_tmin: pd.Series,
    predictions: Mapping[str, LevelPrediction],
    identical_to_winner: bool,
    n_replicas: int = DEFAULT_BOOTSTRAP_REPLICAS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> TrackEvaluation:
    """Evaluate Levels 0 through 3 on a single benchmark evaluation track.

    Implements Protocol Appendix J evaluation logic with J.7 sample gating:
    - Checks required levels in predictions mapping and DatetimeIndex compatibility.
    - Computes valid calendar years under J.7.1.
    - Short-circuits with GlobalOutcome.NOT_ESTABLISHED and no bootstrap calls
      if fewer than MIN_BOOTSTRAP_BLOCKS (7) valid years are present (J.7.2).
    - Masks series to valid years via mask_to_valid_years.
    - Evaluates physical gate, crossing diagnostics, bootstrap confidence intervals,
      and three-tier scorecards for Levels 0, 1, 2, 3 winner, and 3 constant.
    - Reuses Level 3 winner results without re-running bootstrap when
      identical_to_winner is True.

    Parameters
    ----------
    station_id : str
        Station identifier.
    obs_tmax : pd.Series
        Observed daily maximum temperature series.
    obs_tmin : pd.Series
        Observed daily minimum temperature series.
    raw_tmax : pd.Series
        Raw model/reanalysis daily maximum temperature series.
    raw_tmin : pd.Series
        Raw model/reanalysis daily minimum temperature series.
    predictions : Mapping[str, LevelPrediction]
        Mapping of level predictions containing at least 'level1', 'level2',
        'level3_winner', and conditionally 'level3_constant' if identical_to_winner
        is False.
    identical_to_winner : bool
        Whether Level 3 constant configuration is identical to the Level 3 winner.
    n_replicas : int, default=DEFAULT_BOOTSTRAP_REPLICAS
        Number of bootstrap replicas (default: 1000).
    seed : int, default=DEFAULT_BOOTSTRAP_SEED
        Random seed for the bootstrap (default: 42).

    Returns
    -------
    TrackEvaluation
        Track evaluation result containing valid years, sample status, scorecards,
        and global outcomes.

    Raises
    ------
    ValueError
        If required prediction keys are missing or series indexes do not match.
    """
    required_keys = {"level1", "level2", "level3_winner"}
    if not identical_to_winner:
        required_keys.add("level3_constant")

    missing_keys = required_keys - set(predictions.keys())
    if missing_keys:
        raise ValueError(f"Missing required prediction levels: {sorted(missing_keys)}")

    for s, name in (
        (obs_tmax, "obs_tmax"),
        (obs_tmin, "obs_tmin"),
        (raw_tmax, "raw_tmax"),
        (raw_tmin, "raw_tmin"),
    ):
        if not isinstance(s.index, pd.DatetimeIndex):
            raise ValueError(f"{name} index must be a pd.DatetimeIndex.")

    base_index = obs_tmax.index
    if not (
        obs_tmin.index.equals(base_index)
        and raw_tmax.index.equals(base_index)
        and raw_tmin.index.equals(base_index)
    ):
        raise ValueError(
            "obs_tmax, obs_tmin, raw_tmax, and raw_tmin must have identical indexes."
        )

    for lvl_name, pred in predictions.items():
        if not isinstance(
            pred.tmax.index, pd.DatetimeIndex
        ) or not pred.tmax.index.equals(base_index):
            raise ValueError(
                f"Prediction '{lvl_name}' tmax index must be a pd.DatetimeIndex "
                "identical to obs_tmax."
            )
        if not isinstance(
            pred.tmin.index, pd.DatetimeIndex
        ) or not pred.tmin.index.equals(base_index):
            raise ValueError(
                f"Prediction '{lvl_name}' tmin index must be a pd.DatetimeIndex "
                "identical to obs_tmax."
            )

    valid_years = valid_years_j71(obs_tmax, obs_tmin, raw_tmax, raw_tmin)
    sample_status = track_sample_status(len(valid_years))

    levels_order = (
        "level0",
        "level1",
        "level2",
        "level3_winner",
        "level3_constant",
    )

    if sample_status != SAMPLE_STATUS_OK:
        return TrackEvaluation(
            station_id=station_id,
            valid_years=valid_years,
            sample_status=sample_status,
            scorecards={lvl: None for lvl in levels_order},
            outcomes={lvl: GlobalOutcome.NOT_ESTABLISHED for lvl in levels_order},
            identical_to_winner=identical_to_winner,
            reason=(
                f"insufficient sample: {len(valid_years)} valid years "
                "< MIN_BOOTSTRAP_BLOCKS"
            ),
        )

    m_obs_tmax, m_obs_tmin, m_raw_tmax, m_raw_tmin = mask_to_valid_years(
        obs_tmax, obs_tmin, raw_tmax, raw_tmin, valid_years
    )

    masked_preds: Dict[str, Tuple[pd.Series, pd.Series]] = {}
    for lvl, pred in predictions.items():
        _, _, m_tmax, m_tmin = mask_to_valid_years(
            obs_tmax, obs_tmin, pred.tmax, pred.tmin, valid_years
        )
        masked_preds[lvl] = (m_tmax, m_tmin)

    scorecards: Dict[str, Optional[EvaluationScorecard]] = {}

    # Level 0: Raw model series
    l0_gate = check_temperature_physical_gate(m_raw_tmax, m_raw_tmin)
    valid_l0 = m_raw_tmax.notna() & m_raw_tmin.notna()
    n_cr_l0 = int((valid_l0 & (m_raw_tmin > m_raw_tmax)).sum())
    n_days_l0 = int(valid_l0.sum())
    frac_cr_l0 = float(n_cr_l0 / n_days_l0) if n_days_l0 > 0 else 0.0
    l0_diag = CrossingDiagnostic(
        n_crossings=n_cr_l0,
        n_total_days=n_days_l0,
        fraction_crossings=frac_cr_l0,
        alert_triggered=bool(frac_cr_l0 > 0.01),
    )
    l0_boot = compute_annual_block_bootstrap_temperature(
        tmax_obs=m_obs_tmax,
        tmax_mod=m_raw_tmax,
        tmin_obs=m_obs_tmin,
        tmin_mod=m_raw_tmin,
        n_replicas=n_replicas,
        seed=seed,
    )
    scorecards["level0"] = evaluate_temperature_scorecard(
        station_id=station_id,
        level="level0",
        boot_results=l0_boot,
        physical_gate_passed=l0_gate,
        crossing_diag=l0_diag,
    )

    # Levels 1, 2, 3_winner
    l3w_boot = None
    l3w_gate = False
    l3w_diag: Optional[CrossingDiagnostic] = None

    for lvl in ("level1", "level2", "level3_winner"):
        pred = predictions[lvl]
        m_pred_tmax, m_pred_tmin = masked_preds[lvl]
        gate_passed = check_temperature_physical_gate(m_pred_tmax, m_pred_tmin)
        if pred.crossing_diagnostic is not None:
            crossing_diag = pred.crossing_diagnostic
        else:
            valid_p = m_pred_tmax.notna() & m_pred_tmin.notna()
            n_cr = int((valid_p & (m_pred_tmin > m_pred_tmax)).sum())
            n_days = int(valid_p.sum())
            frac = float(n_cr / n_days) if n_days > 0 else 0.0
            crossing_diag = CrossingDiagnostic(
                n_crossings=n_cr,
                n_total_days=n_days,
                fraction_crossings=frac,
                alert_triggered=bool(frac > 0.01),
            )
        boot = compute_annual_block_bootstrap_temperature(
            tmax_obs=m_obs_tmax,
            tmax_mod=m_pred_tmax,
            tmin_obs=m_obs_tmin,
            tmin_mod=m_pred_tmin,
            n_replicas=n_replicas,
            seed=seed,
        )
        if lvl == "level3_winner":
            l3w_boot = boot
            l3w_gate = gate_passed
            l3w_diag = crossing_diag

        scorecards[lvl] = evaluate_temperature_scorecard(
            station_id=station_id,
            level=lvl,
            boot_results=boot,
            physical_gate_passed=gate_passed,
            crossing_diag=crossing_diag,
        )

    # Level 3 Constant
    if identical_to_winner:
        assert l3w_boot is not None
        scorecards["level3_constant"] = evaluate_temperature_scorecard(
            station_id=station_id,
            level="level3_constant",
            boot_results=l3w_boot,
            physical_gate_passed=l3w_gate,
            crossing_diag=l3w_diag,
        )
    else:
        pred_l3c = predictions["level3_constant"]
        m_l3c_tmax, m_l3c_tmin = masked_preds["level3_constant"]
        l3c_gate = check_temperature_physical_gate(m_l3c_tmax, m_l3c_tmin)
        if pred_l3c.crossing_diagnostic is not None:
            l3c_diag = pred_l3c.crossing_diagnostic
        else:
            valid_c = m_l3c_tmax.notna() & m_l3c_tmin.notna()
            n_cr = int((valid_c & (m_l3c_tmin > m_l3c_tmax)).sum())
            n_days = int(valid_c.sum())
            frac = float(n_cr / n_days) if n_days > 0 else 0.0
            l3c_diag = CrossingDiagnostic(
                n_crossings=n_cr,
                n_total_days=n_days,
                fraction_crossings=frac,
                alert_triggered=bool(frac > 0.01),
            )
        l3c_boot = compute_annual_block_bootstrap_temperature(
            tmax_obs=m_obs_tmax,
            tmax_mod=m_l3c_tmax,
            tmin_obs=m_obs_tmin,
            tmin_mod=m_l3c_tmin,
            n_replicas=n_replicas,
            seed=seed,
        )
        scorecards["level3_constant"] = evaluate_temperature_scorecard(
            station_id=station_id,
            level="level3_constant",
            boot_results=l3c_boot,
            physical_gate_passed=l3c_gate,
            crossing_diag=l3c_diag,
        )

    outcomes: Dict[str, GlobalOutcome] = {}
    for lvl in levels_order:
        sc = scorecards[lvl]
        assert sc is not None
        outcomes[lvl] = sc.global_outcome

    return TrackEvaluation(
        station_id=station_id,
        valid_years=valid_years,
        sample_status=sample_status,
        scorecards=scorecards,
        outcomes=outcomes,
        identical_to_winner=identical_to_winner,
        reason=None,
    )


@dataclass(frozen=True)
class FittedLevels:
    """Container holding fitted downscaling models for Levels 1 through 3.

    Parameters
    ----------
    level1 : Level1MonthlyAdditiveTemperature
        Fitted Level 1 model.
    level2 : Level2MeanStateDTRScalingTemperature
        Fitted Level 2 model.
    level3_winner : Level3QuantileMappingTemperature
        Fitted Level 3 winner model.
    level3_constant : Level3QuantileMappingTemperature
        Fitted Level 3 constant model (identical instance if winner extrapolation
        is 'constant').
    identical_to_winner : bool
        Whether level3_constant is identical to level3_winner.
    winner_config_id : int
        Configuration ID of the Level 3 winner.
    """

    level1: Level1MonthlyAdditiveTemperature
    level2: Level2MeanStateDTRScalingTemperature
    level3_winner: Level3QuantileMappingTemperature
    level3_constant: Level3QuantileMappingTemperature
    identical_to_winner: bool
    winner_config_id: int


def fit_levels(
    obs_tmax: pd.Series,
    obs_tmin: pd.Series,
    mod_tmax: pd.Series,
    mod_tmin: pd.Series,
    winner_config: Mapping[str, Any],
) -> FittedLevels:
    """Fit Levels 1, 2, and 3 downscaling models on training/calibration series.

    Hyperparameters are fixed by winner_config (no re-tuning or internal CV).
    ValueError from any model's fit() propagates immediately.

    Parameters
    ----------
    obs_tmax : pd.Series
        Observed daily maximum temperature series.
    obs_tmin : pd.Series
        Observed daily minimum temperature series.
    mod_tmax : pd.Series
        Model/reanalysis daily maximum temperature series.
    mod_tmin : pd.Series
        Model/reanalysis daily minimum temperature series.
    winner_config : Mapping[str, Any]
        Configuration mapping containing 'config_id', 'grouping', 'n_quantiles',
        and 'extrapolation'.

    Returns
    -------
    FittedLevels
        Container holding fitted models and configuration metadata.
    """
    m1 = Level1MonthlyAdditiveTemperature()
    m1.fit(obs_tmax, obs_tmin, mod_tmax, mod_tmin)

    m2 = Level2MeanStateDTRScalingTemperature()
    m2.fit(obs_tmax, obs_tmin, mod_tmax, mod_tmin)

    grouping_val = cast(Literal["monthly", "moving_window"], winner_config["grouping"])
    extrap_val = cast(Literal["constant", "linear"], winner_config["extrapolation"])
    m3_winner = Level3QuantileMappingTemperature(
        grouping=grouping_val,
        n_quantiles=int(winner_config["n_quantiles"]),
        extrapolation=extrap_val,
    )
    m3_winner.fit(obs_tmax, obs_tmin, mod_tmax, mod_tmin)

    if winner_config["extrapolation"] == "constant":
        m3_constant = m3_winner
        identical_to_winner = True
    else:
        m3_constant = Level3QuantileMappingTemperature(
            grouping=grouping_val,
            n_quantiles=int(winner_config["n_quantiles"]),
            extrapolation="constant",
        )
        m3_constant.fit(obs_tmax, obs_tmin, mod_tmax, mod_tmin)
        identical_to_winner = False

    return FittedLevels(
        level1=m1,
        level2=m2,
        level3_winner=m3_winner,
        level3_constant=m3_constant,
        identical_to_winner=identical_to_winner,
        winner_config_id=int(winner_config["config_id"]),
    )


def predict_levels(
    fitted: FittedLevels,
    mod_tmax: pd.Series,
    mod_tmin: pd.Series,
) -> Dict[str, LevelPrediction]:
    """Generate predictions for Levels 1 through 3 using fitted models.

    Parameters
    ----------
    fitted : FittedLevels
        Container with fitted models.
    mod_tmax : pd.Series
        Model/reanalysis daily maximum temperature series to downscale.
    mod_tmin : pd.Series
        Model/reanalysis daily minimum temperature series to downscale.

    Returns
    -------
    Dict[str, LevelPrediction]
        Predictions for 'level1', 'level2', 'level3_winner', and (if not
        identical_to_winner) 'level3_constant', with crossing diagnostics
        read after transform.
    """
    l1_tmax, l1_tmin = fitted.level1.transform(mod_tmax, mod_tmin)
    pred_l1 = LevelPrediction(
        tmax=l1_tmax,
        tmin=l1_tmin,
        crossing_diagnostic=fitted.level1.last_crossing_diagnostic_,
    )

    l2_tmax, l2_tmin = fitted.level2.transform(mod_tmax, mod_tmin)
    pred_l2 = LevelPrediction(
        tmax=l2_tmax,
        tmin=l2_tmin,
        crossing_diagnostic=fitted.level2.last_crossing_diagnostic_,
    )

    l3w_tmax, l3w_tmin = fitted.level3_winner.transform(mod_tmax, mod_tmin)
    pred_l3w = LevelPrediction(
        tmax=l3w_tmax,
        tmin=l3w_tmin,
        crossing_diagnostic=fitted.level3_winner.last_crossing_diagnostic_,
    )

    preds: Dict[str, LevelPrediction] = {
        "level1": pred_l1,
        "level2": pred_l2,
        "level3_winner": pred_l3w,
    }

    if not fitted.identical_to_winner:
        l3c_tmax, l3c_tmin = fitted.level3_constant.transform(mod_tmax, mod_tmin)
        pred_l3c = LevelPrediction(
            tmax=l3c_tmax,
            tmin=l3c_tmin,
            crossing_diagnostic=fitted.level3_constant.last_crossing_diagnostic_,
        )
        preds["level3_constant"] = pred_l3c

    return preds


def evaluate_primary_split(
    station_id: str,
    obs_tmax: pd.Series,
    obs_tmin: pd.Series,
    raw_tmax: pd.Series,
    raw_tmin: pd.Series,
    cal_range: Tuple[int, int],
    val_range: Tuple[int, int],
    winner_config: Mapping[str, Any],
    n_replicas: int = DEFAULT_BOOTSTRAP_REPLICAS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> TrackEvaluation:
    """Evaluate primary split track (calibration fit, validation evaluation).

    Series cover the whole record; slices by calendar year; fit_levels ONLY on
    the calibration slice; predict_levels on the validation slice; evaluate_track_levels
    on the validation slice.
    Raises ValueError if calibration and validation year ranges overlap.

    Any ValueError from fit_levels or predict_levels propagates immediately.

    Parameters
    ----------
    station_id : str
        Station identifier.
    obs_tmax : pd.Series
        Observed daily maximum temperature series over full record.
    obs_tmin : pd.Series
        Observed daily minimum temperature series over full record.
    raw_tmax : pd.Series
        Raw model daily maximum temperature series over full record.
    raw_tmin : pd.Series
        Raw model daily minimum temperature series over full record.
    cal_range : Tuple[int, int]
        Calibration year range (start_year, end_year) inclusive.
    val_range : Tuple[int, int]
        Validation year range (start_year, end_year) inclusive.
    winner_config : Mapping[str, Any]
        Configuration mapping for the Level 3 winner.
    n_replicas : int, default=DEFAULT_BOOTSTRAP_REPLICAS
        Number of bootstrap replicas.
    seed : int, default=DEFAULT_BOOTSTRAP_SEED
        Random seed for the bootstrap.

    Returns
    -------
    TrackEvaluation
        Track evaluation on the validation slice.

    Raises
    ------
    ValueError
        If calibration and validation ranges overlap or inputs are invalid.
    """
    cal_start, cal_end = cal_range
    val_start, val_end = val_range

    cal_years = set(range(cal_start, cal_end + 1))
    val_years = set(range(val_start, val_end + 1))
    if cal_years & val_years:
        raise ValueError(
            f"Calibration range {cal_range} and validation range {val_range} overlap."
        )

    base_index = obs_tmax.index
    if not isinstance(base_index, pd.DatetimeIndex):
        raise ValueError("obs_tmax index must be a pd.DatetimeIndex.")

    cal_mask = base_index.year.isin(cal_years)
    val_mask = base_index.year.isin(val_years)

    # Primary split: safeguard errors propagate (not converted to a status).
    fitted = fit_levels(
        obs_tmax=obs_tmax[cal_mask],
        obs_tmin=obs_tmin[cal_mask],
        mod_tmax=raw_tmax[cal_mask],
        mod_tmin=raw_tmin[cal_mask],
        winner_config=winner_config,
    )

    preds = predict_levels(
        fitted=fitted,
        mod_tmax=raw_tmax[val_mask],
        mod_tmin=raw_tmin[val_mask],
    )

    return evaluate_track_levels(
        station_id=station_id,
        obs_tmax=obs_tmax[val_mask],
        obs_tmin=obs_tmin[val_mask],
        raw_tmax=raw_tmax[val_mask],
        raw_tmin=raw_tmin[val_mask],
        predictions=preds,
        identical_to_winner=fitted.identical_to_winner,
        n_replicas=n_replicas,
        seed=seed,
    )


def _block_cv_predictions(
    obs_tmax: pd.Series,
    obs_tmin: pd.Series,
    raw_tmax: pd.Series,
    raw_tmin: pd.Series,
    winner_config: Mapping[str, Any],
) -> Tuple[Dict[str, LevelPrediction], bool]:
    """Generate concatenated out-of-sample block CV predictions over full record.

    Parameters
    ----------
    obs_tmax : pd.Series
        Observed daily maximum temperature series over full record.
    obs_tmin : pd.Series
        Observed daily minimum temperature series over full record.
    raw_tmax : pd.Series
        Raw model daily maximum temperature series over full record.
    raw_tmin : pd.Series
        Raw model daily minimum temperature series over full record.
    winner_config : Mapping[str, Any]
        Configuration mapping for the Level 3 winner.

    Returns
    -------
    Tuple[Dict[str, LevelPrediction], bool]
        Concatenated out-of-sample predictions (with crossing_diagnostic=None)
        and whether level3_constant is identical to winner.

    Raises
    ------
    ValueError
        If a fold's fit or predict raises ValueError.
    """
    base_index = obs_tmax.index
    if not isinstance(base_index, pd.DatetimeIndex):
        raise ValueError("obs_tmax index must be a pd.DatetimeIndex.")

    years = sorted(set(base_index.year))
    blocks = five_year_blocks(years)

    all_fold_preds: List[Dict[str, LevelPrediction]] = []
    identical_to_winner = bool(winner_config.get("extrapolation") == "constant")

    for first_year, last_year in blocks:
        block_years = set(range(first_year, last_year + 1))
        val_mask = base_index.year.isin(block_years)
        cal_mask = ~val_mask

        try:
            fitted = fit_levels(
                obs_tmax=obs_tmax[cal_mask],
                obs_tmin=obs_tmin[cal_mask],
                mod_tmax=raw_tmax[cal_mask],
                mod_tmin=raw_tmin[cal_mask],
                winner_config=winner_config,
            )
            fold_preds = predict_levels(
                fitted=fitted,
                mod_tmax=raw_tmax[val_mask],
                mod_tmin=raw_tmin[val_mask],
            )
        except ValueError as e:
            raise ValueError(f"fold {first_year}-{last_year}: {e}") from e

        all_fold_preds.append(fold_preds)

    levels = list(all_fold_preds[0].keys())
    concat_preds: Dict[str, LevelPrediction] = {}
    for lvl in levels:
        concat_tmax = pd.concat(
            [fp[lvl].tmax for fp in all_fold_preds], axis=0
        ).sort_index()
        concat_tmin = pd.concat(
            [fp[lvl].tmin for fp in all_fold_preds], axis=0
        ).sort_index()
        concat_preds[lvl] = LevelPrediction(
            tmax=concat_tmax,
            tmin=concat_tmin,
            crossing_diagnostic=None,
        )

    return concat_preds, identical_to_winner


def evaluate_block_cv(
    station_id: str,
    obs_tmax: pd.Series,
    obs_tmin: pd.Series,
    raw_tmax: pd.Series,
    raw_tmin: pd.Series,
    winner_config: Mapping[str, Any],
    n_replicas: int = DEFAULT_BOOTSTRAP_REPLICAS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> TrackEvaluation:
    """Evaluate 5-year block cross-validation track.

    Partitions calendar years into contiguous 5-year blocks using five_year_blocks.
    Raises ValueError if calendar years have gaps.
    For each block:
      - Fits models on all rows outside the block.
      - Predicts on block rows.
    Concatenates out-of-sample predictions across folds and evaluates with
    evaluate_track_levels on the full record.

    If any fold raises ValueError (safeguard failure), returns an aborted
    TrackEvaluation with sample_status SAMPLE_STATUS_ABORTED.

    Parameters
    ----------
    station_id : str
        Station identifier.
    obs_tmax : pd.Series
        Observed daily maximum temperature series over full record.
    obs_tmin : pd.Series
        Observed daily minimum temperature series over full record.
    raw_tmax : pd.Series
        Raw model daily maximum temperature series over full record.
    raw_tmin : pd.Series
        Raw model daily minimum temperature series over full record.
    winner_config : Mapping[str, Any]
        Configuration mapping for the Level 3 winner.
    n_replicas : int, default=DEFAULT_BOOTSTRAP_REPLICAS
        Number of bootstrap replicas.
    seed : int, default=DEFAULT_BOOTSTRAP_SEED
        Random seed for the bootstrap.

    Returns
    -------
    TrackEvaluation
        Track evaluation on the full record.

    Raises
    ------
    ValueError
        If calendar years have gaps or series indexes are invalid.
    """
    base_index = obs_tmax.index
    if not isinstance(base_index, pd.DatetimeIndex):
        raise ValueError("obs_tmax index must be a pd.DatetimeIndex.")

    years = sorted(set(base_index.year))
    if len(years) > 1 and any(y2 != y1 + 1 for y1, y2 in zip(years[:-1], years[1:])):
        raise ValueError("Calendar years of the index have gaps.")

    levels_order = (
        "level0",
        "level1",
        "level2",
        "level3_winner",
        "level3_constant",
    )

    try:
        preds, identical_to_winner = _block_cv_predictions(
            obs_tmax=obs_tmax,
            obs_tmin=obs_tmin,
            raw_tmax=raw_tmax,
            raw_tmin=raw_tmin,
            winner_config=winner_config,
        )
    except ValueError as e:
        # OPEN ITEM v1.10: J.6.2 does not define the outcome of an aborted station; placeholder NOT_ESTABLISHED  # noqa: E501
        return TrackEvaluation(
            station_id=station_id,
            valid_years=[],
            sample_status=SAMPLE_STATUS_ABORTED,
            scorecards={lvl: None for lvl in levels_order},
            outcomes={lvl: GlobalOutcome.NOT_ESTABLISHED for lvl in levels_order},
            identical_to_winner=bool(winner_config.get("extrapolation") == "constant"),
            reason=str(e),
        )

    return evaluate_track_levels(
        station_id=station_id,
        obs_tmax=obs_tmax,
        obs_tmin=obs_tmin,
        raw_tmax=raw_tmax,
        raw_tmin=raw_tmin,
        predictions=preds,
        identical_to_winner=identical_to_winner,
        n_replicas=n_replicas,
        seed=seed,
    )


def combine_tracks(
    primary: TrackEvaluation,
    cv: TrackEvaluation,
) -> Dict[str, GlobalOutcome]:
    """Combine primary split and block CV tracks using weaker-of-the-two concordance.

    Under Protocol Section 4.3, the concordance outcome is the weaker of the
    primary and CV track outcomes (Falsified < Not Established < Success).

    Parameters
    ----------
    primary : TrackEvaluation
        Primary split track evaluation.
    cv : TrackEvaluation
        5-year block cross-validation track evaluation.

    Returns
    -------
    Dict[str, GlobalOutcome]
        Combined global outcome for each level.
    """
    levels_order = (
        "level0",
        "level1",
        "level2",
        "level3_winner",
        "level3_constant",
    )
    combined: Dict[str, GlobalOutcome] = {}
    for k in levels_order:
        out_p = primary.outcomes[k]
        out_cv = cv.outcomes[k]
        combined[k] = apply_weaker_of_two_concordance(out_p, out_cv)
    return combined


def levels_with_global_success(
    outcomes_by_station: Mapping[str, Mapping[str, GlobalOutcome]],
) -> List[str]:
    """Return downscaling levels that achieve Global Success across all stations.

    Filters candidate levels among ('level1', 'level2', 'level3_winner') to those
    with GlobalOutcome.SUCCESS at every station in outcomes_by_station.
    Under Protocol Section 5.4, Level 0 and Level 3 constant never qualify for
    operational designation.

    Note: Selecting the final operational level under Section 5.4 item 3
    ('highest promoted level') depends on the reading of 'promoted' (Section 6.4);
    final operational level selection is NOT implemented here.

    Parameters
    ----------
    outcomes_by_station : Mapping[str, Mapping[str, GlobalOutcome]]
        Mapping from station_id to mapping of level to GlobalOutcome.

    Returns
    -------
    List[str]
        List of candidate levels achieving Global Success across all stations.
    """
    # OPEN ITEM v1.10: Section 5.4 item 3 ('highest promoted level') depends on the reading of 'promoted' (Section 6.4); final operational level NOT selected here  # noqa: E501
    candidate_levels = ("level1", "level2", "level3_winner")
    if not outcomes_by_station:
        return []

    successful: List[str] = []
    for lvl in candidate_levels:
        if all(
            st_outcomes.get(lvl) == GlobalOutcome.SUCCESS
            for st_outcomes in outcomes_by_station.values()
        ):
            successful.append(lvl)

    return successful
