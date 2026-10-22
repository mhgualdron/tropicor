"""Station selection, partitioning, and slot evaluation engine.

Implements pre-specified protocol rules for partitioning candidate station records
into non-overlapping calibration and validation blocks, evaluating sample-size
eligibility thresholds, and resolving priority slots across candidate plans.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple


def compute_optimal_partition(valid_years: Sequence[int]) -> Dict[str, Any]:
    """Compute optimal partition minimizing |n_cal - n_val| with tie-breaking.

    Partitions valid years into two contiguous blocks: calibration (earlier years)
    and validation (later years). When |n_cal - n_val| is tied, calibration
    receives the larger block.

    Args:
        valid_years: Sequence of calendar years meeting the annual completeness
            threshold.

    Returns:
        Dictionary containing partition details:
            - valid_years: Sorted unique list of valid years.
            - n_valid_years: Count of valid years.
            - cal_block: [start_year, end_year] of calibration span, or None.
            - val_block: [start_year, end_year] of validation span, or None.
            - n_cal_valid: Number of valid years in calibration block.
            - n_val_valid: Number of valid years in validation block.
            - diff_cal_val: Absolute difference |n_cal - n_val|.
    """
    y_sorted = sorted(set(int(y) for y in valid_years))
    n_y = len(y_sorted)

    if n_y < 2:
        return {
            "valid_years": y_sorted,
            "n_valid_years": n_y,
            "cal_block": None,
            "val_block": None,
            "n_cal_valid": 0,
            "n_val_valid": 0,
            "diff_cal_val": 0,
        }

    best_cut_idx = 1
    min_diff = 999999
    best_n_cal = 0
    best_n_val = 0

    for cut_idx in range(1, n_y):
        n_cal = cut_idx
        n_val = n_y - cut_idx
        diff = abs(n_cal - n_val)
        if diff < min_diff:
            min_diff = diff
            best_cut_idx = cut_idx
            best_n_cal = n_cal
            best_n_val = n_val
        elif diff == min_diff:
            # Tie breaker: calibration receives the larger block
            if n_cal > n_val:
                min_diff = diff
                best_cut_idx = cut_idx
                best_n_cal = n_cal
                best_n_val = n_val

    cal_years = y_sorted[:best_cut_idx]
    val_years = y_sorted[best_cut_idx:]

    return {
        "valid_years": y_sorted,
        "n_valid_years": n_y,
        "cal_block": [cal_years[0], cal_years[-1]],
        "val_block": [val_years[0], val_years[-1]],
        "n_cal_valid": best_n_cal,
        "n_val_valid": best_n_val,
        "diff_cal_val": min_diff,
    }


def evaluate_partition_eligibility(
    partition: Dict[str, Any],
    min_joint_years: int = 20,
    min_block_years: int = 8,
) -> Tuple[bool, List[str]]:
    """Evaluate mechanical sample size eligibility criteria for a partition.

    Criteria:
    1. Total valid years >= min_joint_years (default 20).
    2. Calibration block valid years >= min_block_years (default 8).
    3. Validation block valid years >= min_block_years (default 8).

    Args:
        partition: Partition dictionary from compute_optimal_partition.
        min_joint_years: Minimum total valid years required.
        min_block_years: Minimum valid years required in each block.

    Returns:
        Tuple of (is_eligible, list_of_ineligibility_reasons).
    """
    reasons: List[str] = []
    n_y = int(partition.get("n_valid_years", 0))
    n_cal = int(partition.get("n_cal_valid", 0))
    n_val = int(partition.get("n_val_valid", 0))

    if n_y < min_joint_years:
        reasons.append(f"Insufficient joint valid years: {n_y} < {min_joint_years}")
    if n_cal < min_block_years:
        reasons.append(
            f"Insufficient calibration valid years: {n_cal} < {min_block_years}"
        )
    if n_val < min_block_years:
        reasons.append(
            f"Insufficient validation valid years: {n_val} < {min_block_years}"
        )

    return len(reasons) == 0, reasons


VALID_PLAN_STATUSES = {"ELEGIBLE", "NO_ELEGIBLE", "SIN_DATOS", "INCOMPLETO"}


def evaluate_slot_status(
    plans_status: Sequence[Tuple[str, str, str]],
) -> Tuple[str, Optional[Tuple[str, str]]]:
    """Evaluate the status of a single slot given its sequential plans (A, B, C).

    Rule:
    - Selects the first plan (A, B, C) with ELEGIBLE status.
    - A slot is PENDIENTE if any preceding plan is SIN_DATOS or INCOMPLETO,
      and none of the preceding plans are ELEGIBLE.
    - An INCOMPLETO plan indicates a candidate directory with missing variable
      files; it counts as SIN_DATOS for slot resolution (slot remains PENDIENTE).
    - A slot is COMPLETA if a plan is ELEGIBLE and all preceding plans are NO_ELEGIBLE.
    - If all plans are NO_ELEGIBLE, the slot has no eligible candidate.

    Args:
        plans_status: Sequence of (plan_letter, station_code, status) in order A, B, C.

    Returns:
        Tuple of (slot_status, selected_plan_info) where:
            - slot_status is 'COMPLETA', 'PENDIENTE', or 'NO_ELEGIBLE'.
            - selected_plan_info is (plan_letter, station_code) or None.

    Raises:
        ValueError: If plans_status is empty or contains an unknown status.
    """
    if not plans_status:
        raise ValueError("plans_status must not be empty.")

    for _, _, status in plans_status:
        if status not in VALID_PLAN_STATUSES:
            valid_list = sorted(VALID_PLAN_STATUSES)
            raise ValueError(
                f"Unknown status '{status}'. Expected one of {valid_list}."
            )

    for plan_letter, station_code, status in plans_status:
        if status == "ELEGIBLE":
            return "COMPLETA", (plan_letter, station_code)
        if status in ("SIN_DATOS", "INCOMPLETO"):
            return "PENDIENTE", None
        # If status == 'NO_ELEGIBLE', continue to next plan

    return "NO_ELEGIBLE", None
