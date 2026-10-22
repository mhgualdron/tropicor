"""Unit tests for station selection, optimal partitioning, and slot evaluation rules.

All tests use purely synthetic data (no real station observations).
"""

from __future__ import annotations

from tropicor.core.station_selection import (
    compute_optimal_partition,
    evaluate_partition_eligibility,
    evaluate_slot_status,
)


def test_partition_tie_breaking_calibration_larger() -> None:
    """Verify tie-breaking rule: calibration gets larger block in a tie."""
    # 21 consecutive years: candidate cuts at 10 (10/11, diff=1) and 11 (11/10, diff=1)
    years_21 = list(range(2000, 2021))
    part_21 = compute_optimal_partition(years_21)

    assert part_21["n_valid_years"] == 21
    assert part_21["n_cal_valid"] == 11
    assert part_21["n_val_valid"] == 10
    assert part_21["diff_cal_val"] == 1
    assert part_21["cal_block"] == [2000, 2010]
    assert part_21["val_block"] == [2011, 2020]

    # 17 consecutive years: cuts at 8 (8/9, diff=1) and 9 (9/8, diff=1) -> 9/8
    years_17 = list(range(1990, 2007))
    part_17 = compute_optimal_partition(years_17)
    assert part_17["n_valid_years"] == 17
    assert part_17["n_cal_valid"] == 9
    assert part_17["n_val_valid"] == 8
    assert part_17["diff_cal_val"] == 1


def test_partition_years_with_gaps_inside_block() -> None:
    """Verify partition behavior when years have gaps inside blocks."""
    # 20 years with gaps inside both blocks
    # Cal: 10 years spanning 1990 to 2004 (1991, 1993, 1997, 1999, 2001 missing)
    # Val: 10 years spanning 2006 to 2020 (2007, 2010, 2013, 2015, 2018 missing)
    years = [
        1990,
        1992,
        1994,
        1995,
        1996,
        1998,
        2000,
        2002,
        2003,
        2004,
        2006,
        2008,
        2009,
        2011,
        2012,
        2014,
        2016,
        2017,
        2019,
        2020,
    ]
    part = compute_optimal_partition(years)

    assert part["n_valid_years"] == 20
    assert part["n_cal_valid"] == 10
    assert part["n_val_valid"] == 10
    assert part["diff_cal_val"] == 0
    assert part["cal_block"] == [1990, 2004]
    assert part["val_block"] == [2006, 2020]


def test_eligibility_19_20_21_joint_years() -> None:
    """Verify eligibility gates for 19, 20, and 21 joint valid years."""
    # 19 joint years: strictly below the 20-year threshold
    years_19 = list(range(2000, 2019))
    part_19 = compute_optimal_partition(years_19)
    eligible_19, reasons_19 = evaluate_partition_eligibility(part_19)
    assert eligible_19 is False
    assert any("Insufficient joint valid years: 19 < 20" in r for r in reasons_19)

    # 20 joint years: meets the 20-year threshold with 10 cal / 10 val
    years_20 = list(range(2000, 2020))
    part_20 = compute_optimal_partition(years_20)
    eligible_20, reasons_20 = evaluate_partition_eligibility(part_20)
    assert eligible_20 is True
    assert len(reasons_20) == 0

    # 21 joint years: meets threshold with 11 cal / 10 val
    years_21 = list(range(2000, 2021))
    part_21 = compute_optimal_partition(years_21)
    eligible_21, reasons_21 = evaluate_partition_eligibility(part_21)
    assert eligible_21 is True
    assert len(reasons_21) == 0


def test_eligibility_block_with_7_or_8_years() -> None:
    """Verify block eligibility threshold: exactly 8 years passes, 7 years fails."""
    # 15 valid years: cut is 8 cal / 7 val.
    # Cal has 8 (meets >= 8), Val has 7 (< 8), and total 15 (< 20).
    years_15 = list(range(2000, 2015))
    part_15 = compute_optimal_partition(years_15)
    assert part_15["n_cal_valid"] == 8
    assert part_15["n_val_valid"] == 7
    eligible_15, reasons_15 = evaluate_partition_eligibility(part_15)
    assert eligible_15 is False
    assert any("Insufficient validation valid years: 7 < 8" in r for r in reasons_15)
    assert not any("calibration" in r for r in reasons_15)

    # Direct test of calibration block < 8
    part_custom = {
        "n_valid_years": 20,
        "n_cal_valid": 7,
        "n_val_valid": 13,
        "cal_block": [2000, 2006],
        "val_block": [2007, 2019],
    }
    eligible_c, reasons_c = evaluate_partition_eligibility(part_custom)
    assert eligible_c is False
    assert any("Insufficient calibration valid years: 7 < 8" in r for r in reasons_c)

    # Both blocks with exactly 8 years (e.g. 16 valid years)
    years_16 = list(range(2000, 2016))
    part_16 = compute_optimal_partition(years_16)
    assert part_16["n_cal_valid"] == 8
    assert part_16["n_val_valid"] == 8
    eligible_16, reasons_16 = evaluate_partition_eligibility(part_16)
    assert eligible_16 is False  # Fails overall total 16 < 20, but block sizes pass
    assert not any("calibration" in r for r in reasons_16)
    assert not any("validation" in r for r in reasons_16)


def test_berlin_synthetic_pattern_21_years_cut_11_10() -> None:
    """Verify synthetic pattern matching Berlin (37015020): 21 years, cut 11/10."""
    berlin_years = [
        1991,
        1992,
        1993,
        1994,
        1995,
        1996,
        1997,
        2005,
        2007,
        2008,
        2009,
        2010,
        2011,
        2012,
        2013,
        2014,
        2015,
        2016,
        2018,
        2019,
        2020,
    ]
    part = compute_optimal_partition(berlin_years)

    assert part["n_valid_years"] == 21
    assert part["n_cal_valid"] == 11
    assert part["n_val_valid"] == 10
    assert part["diff_cal_val"] == 1
    assert part["cal_block"] == [1991, 2009]
    assert part["val_block"] == [2010, 2020]

    eligible, reasons = evaluate_partition_eligibility(part)
    assert eligible is True
    assert len(reasons) == 0


def test_mogotes_synthetic_pattern_20_years_dispersed_validation() -> None:
    """Verify synthetic pattern matching Mogotes (24025040): 20 years, 10/10."""
    mogotes_years = [
        1990,
        1991,
        1993,
        1994,
        1995,
        1996,
        1997,
        1998,
        1999,
        2000,
        2001,
        2002,
        2003,
        2004,
        2005,
        2006,
        2007,
        2010,
        2019,
        2020,
    ]
    part = compute_optimal_partition(mogotes_years)

    assert part["n_valid_years"] == 20
    assert part["n_cal_valid"] == 10
    assert part["n_val_valid"] == 10
    assert part["diff_cal_val"] == 0
    assert part["cal_block"] == [1990, 2000]
    assert part["val_block"] == [2001, 2020]

    # Verify that validation span covers 2001 to 2020 with internal gaps
    assert (
        part["val_block"][1] - part["val_block"][0] + 1 == 20
    )  # 20 calendar years span
    assert part["n_val_valid"] == 10  # but only 10 valid years

    eligible, reasons = evaluate_partition_eligibility(part)
    assert eligible is True
    assert len(reasons) == 0


def test_slot_resolution_rules() -> None:
    """Verify slot evaluation states: COMPLETA, PENDIENTE, NO_ELEGIBLE."""
    # Case 1: Plan A is ELEGIBLE -> COMPLETA with Plan A
    s1, plan1 = evaluate_slot_status(
        [
            ("A", "37015020", "ELEGIBLE"),
            ("B", "24035240", "SIN_DATOS"),
            ("C", "35025060", "SIN_DATOS"),
        ]
    )
    assert s1 == "COMPLETA"
    assert plan1 == ("A", "37015020")

    # Case 2: Plan A is NO_ELEGIBLE, Plan B is SIN_DATOS -> PENDIENTE
    s2, plan2 = evaluate_slot_status(
        [
            ("A", "24035120", "NO_ELEGIBLE"),
            ("B", "24035130", "SIN_DATOS"),
            ("C", "35075010", "SIN_DATOS"),
        ]
    )
    assert s2 == "PENDIENTE"
    assert plan2 is None

    # Case 3: Plan A is NO_ELEGIBLE, Plan B is ELEGIBLE -> COMPLETA with Plan B
    s3, plan3 = evaluate_slot_status(
        [
            ("A", "24035120", "NO_ELEGIBLE"),
            ("B", "24035130", "ELEGIBLE"),
            ("C", "35075010", "SIN_DATOS"),
        ]
    )
    assert s3 == "COMPLETA"
    assert plan3 == ("B", "24035130")

    # Case 4: Plan A is SIN_DATOS -> PENDIENTE
    s4, plan4 = evaluate_slot_status(
        [
            ("A", "26150160", "SIN_DATOS"),
            ("B", "26155150", "SIN_DATOS"),
            ("C", "26135140", "SIN_DATOS"),
        ]
    )
    assert s4 == "PENDIENTE"
    assert plan4 is None

    # Case 5: All plans NO_ELEGIBLE -> NO_ELEGIBLE
    s5, plan5 = evaluate_slot_status(
        [
            ("A", "1001", "NO_ELEGIBLE"),
            ("B", "1002", "NO_ELEGIBLE"),
            ("C", "1003", "NO_ELEGIBLE"),
        ]
    )
    assert s5 == "NO_ELEGIBLE"
    assert plan5 is None


def test_evaluate_slot_status_empty_plans_raises_value_error() -> None:
    """Verify that an empty plans list raises ValueError."""
    import pytest

    with pytest.raises(ValueError, match="plans_status must not be empty"):
        evaluate_slot_status([])


def test_evaluate_slot_status_unknown_status_raises_value_error() -> None:
    """Verify that an unknown plan status raises ValueError."""
    import pytest

    with pytest.raises(ValueError, match="Unknown status 'DESCONOCIDO'"):
        evaluate_slot_status([("A", "37015020", "DESCONOCIDO")])


def test_evaluate_slot_status_incompleto_counts_as_sin_datos() -> None:
    """Verify that INCOMPLETO counts as SIN_DATOS (slot becomes PENDIENTE)."""
    # Case A: Plan A is INCOMPLETO -> PENDIENTE
    status, plan = evaluate_slot_status(
        [
            ("A", "37015020", "INCOMPLETO"),
            ("B", "24035240", "SIN_DATOS"),
            ("C", "35025060", "SIN_DATOS"),
        ]
    )
    assert status == "PENDIENTE"
    assert plan is None

    # Case B: Plan A is NO_ELEGIBLE, Plan B is INCOMPLETO -> PENDIENTE
    status2, plan2 = evaluate_slot_status(
        [
            ("A", "24035120", "NO_ELEGIBLE"),
            ("B", "24035130", "INCOMPLETO"),
            ("C", "35075010", "SIN_DATOS"),
        ]
    )
    assert status2 == "PENDIENTE"
    assert plan2 is None


def test_compute_optimal_partition_invalid_year_between_blocks_belongs_to_neither() -> (
    None
):
    """Verify partition when invalid year falls between cal and val blocks.

    Protocol Rule F.3 / 2b:
    The validation block starts at the first valid year after the cut point,
    and any invalid year between the two blocks belongs to neither block.
    """
    # 20 valid years: 10 calibration (1990-1999) and 10 validation (2001-2010).
    # Year 2000 is invalid (< 330 days) and omitted from valid_years.
    years = list(range(1990, 2000)) + list(range(2001, 2011))
    part = compute_optimal_partition(years)

    assert part["n_valid_years"] == 20
    assert part["n_cal_valid"] == 10
    assert part["n_val_valid"] == 10
    assert part["diff_cal_val"] == 0
    # Cal block ends in 1999
    assert part["cal_block"] == [1990, 1999]
    # Val block starts in 2001 (first valid year after cut)
    assert part["val_block"] == [2001, 2010]
    # Gap year 2000 is strictly outside both blocks:
    assert 2000 > part["cal_block"][1]
    assert 2000 < part["val_block"][0]
