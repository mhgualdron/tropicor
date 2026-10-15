"""Exploratory out-of-sample benchmark for statistical bias correction (Sprint 8).

EXPLORATORY ONLY. Inputs are thesis-processed station/ERA5 files in ``local/``
that have not yet been rebuilt from primary sources. Results must not be cited
as validation and are kept out of the public documentation.

Design (agreed in the Sprint 8 plan):

1. Only OUT-OF-SAMPLE metrics are reported. In-sample linear scaling matches the
   calibration mean by construction, so in-sample bias is never reported.
2. Two protocols per series:
   - split-sample: calibrate 1990-2009, evaluate 2010-2019;
   - leave-one-year-out (LOYO): fit on all other years, predict the held-out
     year, then score the concatenated out-of-sample predictions.
   LOYO is the PRIMARY protocol whenever the split leaves fewer than
   ``MIN_SPLIT_TEST_MONTHS`` test months (e.g. Las Flores ends in 2015).
3. Groupings ``monthly``, ``moving_window`` (3) and ``pooled`` are compared;
   the package default is chosen from these results, not assumed.
4. EQM is judged on distributional metrics (KS distance, P10/P50/P90 errors,
   annual-cycle amplitude error). RMSE and KGE are reported as context only.
5. Known data-quality issues are handled explicitly (see ``QC_EXCLUSIONS`` and
   ``QC_FLAGS``).

Output: ``local/bias_correction_exploratory_report.csv`` (git-ignored).
"""

import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from tropicor.core.alignment import align_common_period
from tropicor.core.bias import EmpiricalQuantileMapping, LinearScalingCorrection
from tropicor.core.metrics import compute_validation_metrics

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

LOCAL_DIR = Path("local")
OUT_CSV = LOCAL_DIR / "bias_correction_exploratory_report.csv"

CAL_PERIOD = ("1990-01-01", "2009-12-31")
TEST_PERIOD = ("2010-01-01", "2019-12-31")
MIN_SPLIT_TEST_MONTHS = 96  # < 8 test years -> LOYO is the primary protocol

GROUPINGS = ["monthly", "moving_window", "pooled"]

STATIONS = [
    {
        "name": "Puerto Carreño",
        "file": "APuertoCarreno_1990-2019.xlsx",
        "region": "Orinoquía",
    },
    {"name": "Las Flores", "file": "LasFlores_1990-2019.xlsx", "region": "Caribe"},
    {"name": "Noanamá", "file": "Noanama_1990-2019.xlsx", "region": "Pacífico"},
    {
        "name": "Gorgona",
        "file": "Gorgona_1990-2019.xlsx",
        "region": "Insular (Pacífico)",
    },
]

# (variable, sheet, obs date col, obs value col, mod date col, mod value col)
VARIABLES = [
    ("precipitation", "Precipitacion_Total_Mensual", "Fecha", "IDEAM", "Fera", "ERA5"),
    ("tmin", "TMin", "Fecha_Minima", "Minima", "Fecha_Minima", "Era5_Minima"),
    ("tmax", "TMax", "Fecha_Maxima", "Maxima", "Fecha_Maxima", "Era5_Maxima"),
]

# Records removed before any fitting. Keys: (station, variable).
QC_EXCLUSIONS: Dict[Tuple[str, str], List[Tuple[str, str]]] = {
    ("Las Flores", "tmax"): [
        (
            "2018-03-01",
            "Isolated record 32 months after station closure; "
            "TMax=21.0 °C (Z=+8.70, Sprint 7) and TMax < TMin in same month.",
        ),
    ],
    ("Las Flores", "tmin"): [
        (
            "2018-03-01",
            "Same isolated 2018-03 record; TMin=23.0 °C exceeds "
            "the TMax of that month (physically inconsistent pair).",
        ),
    ],
}

# Series kept in the benchmark but flagged for interpretation.
QC_FLAGS: Dict[Tuple[str, str], str] = {
    ("Noanamá", "tmax"): (
        "Residual drift (model - obs increasing, cal=-0.34 -> test=+1.06 C, "
        "drift=+1.40 C); possible station inhomogeneity (Sprint 7); out-of-sample "
        "results may reflect station drift rather than ERA5."
    ),
}


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


def _clean_series(df: pd.DataFrame, date_col: str, val_col: str) -> pd.Series:
    """Parse, sort and deduplicate a monthly series from a sheet."""
    sub = df[[date_col, val_col]].dropna()
    dates = pd.to_datetime(sub[date_col], errors="coerce")
    series = pd.Series(sub[val_col].to_numpy(), index=dates, dtype=float)
    series = series[series.index.notna()].dropna().sort_index()
    return series[~series.index.duplicated(keep="first")]


def load_pair(
    xl: pd.ExcelFile, station: str, spec: Tuple[str, ...]
) -> Tuple[pd.Series, pd.Series, List[str]]:
    """Load an aligned obs/mod pair and apply documented QC exclusions."""
    var, sheet, d_obs, v_obs, d_mod, v_mod = spec
    df = xl.parse(sheet)
    obs = _clean_series(df, d_obs, v_obs)
    mod = _clean_series(df, d_mod, v_mod)
    obs, mod = align_common_period(obs, mod, drop_na=True)

    notes: List[str] = []
    for date, reason in QC_EXCLUSIONS.get((station, var), []):
        ts = pd.Timestamp(date)
        if ts in obs.index:
            obs = obs.drop(ts)
            mod = mod.drop(ts)
            notes.append(f"EXCLUDED {date}: {reason}")
    if (station, var) in QC_FLAGS:
        notes.append(f"FLAG: {QC_FLAGS[(station, var)]}")
    return obs, mod, notes


# -----------------------------------------------------------------------------
# Correctors and protocols
# -----------------------------------------------------------------------------

Corrector = Callable[[pd.Series, pd.Series, pd.Series], pd.Series]


def make_corrector(method: str, grouping: str, var_type: str) -> Corrector:
    """Return f(obs_cal, mod_cal, mod_eval) -> corrected mod_eval."""

    def _apply(obs_c: pd.Series, mod_c: pd.Series, mod_e: pd.Series) -> pd.Series:
        est: Any
        if method == "LS":
            est = LinearScalingCorrection(variable=var_type, grouping=grouping)
        else:
            est = EmpiricalQuantileMapping(variable=var_type, grouping=grouping)
        return est.fit(obs_c, mod_c).transform(mod_e)

    return _apply


def run_split(
    obs: pd.Series, mod: pd.Series, corrector: Optional[Corrector]
) -> Tuple[pd.Series, pd.Series, int, int]:
    """Calibrate on CAL_PERIOD, return (obs_test, pred_test, n_warn, n_cal)."""
    cal = (obs.index >= CAL_PERIOD[0]) & (obs.index <= CAL_PERIOD[1])
    test = (obs.index >= TEST_PERIOD[0]) & (obs.index <= TEST_PERIOD[1])
    obs_t, mod_t = obs[test], mod[test]
    if corrector is None or test.sum() == 0:
        return obs_t, mod_t, 0, int(cal.sum())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pred = corrector(obs[cal], mod[cal], mod_t)
    return obs_t, pred, len(caught), int(cal.sum())


def run_loyo(
    obs: pd.Series, mod: pd.Series, corrector: Optional[Corrector]
) -> Tuple[pd.Series, pd.Series, int]:
    """Leave-one-year-out: concatenated out-of-sample predictions."""
    if corrector is None:
        return obs, mod, 0
    preds: List[pd.Series] = []
    n_warn = 0
    years = obs.index.year
    for year in np.unique(years):
        hold = years == year
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            preds.append(corrector(obs[~hold], mod[~hold], mod[hold]))
        n_warn += len(caught)
    pred = pd.concat(preds).sort_index()
    return obs.loc[pred.index], pred, n_warn


# -----------------------------------------------------------------------------
# Metrics (evaluated on out-of-sample predictions only)
# -----------------------------------------------------------------------------


def annual_cycle_amplitude(series: pd.Series) -> float:
    """Max-min of calendar-month means; NaN if any month is missing."""
    assert isinstance(series.index, pd.DatetimeIndex)
    clim = series.groupby(series.index.month).mean()
    if len(clim) < 12:
        return float("nan")
    return float(clim.max() - clim.min())


def evaluate(obs: pd.Series, pred: pd.Series) -> Dict[str, float]:
    """Distributional metrics (primary) and point metrics (context)."""
    out: Dict[str, float] = {"n_eval": float(len(obs))}
    if len(obs) < 3:
        return out
    o, p = obs.to_numpy(), pred.to_numpy()
    out["ks_pooled"] = float(stats.ks_2samp(p, o).statistic)

    # Fair per-calendar-month KS distance: evaluates whether monthly distributions match
    ks_m_list: List[float] = []
    assert isinstance(obs.index, pd.DatetimeIndex)
    for m in range(1, 13):
        mask_m = obs.index.month == m
        if mask_m.sum() >= 2:
            val_o = o[mask_m]
            val_p = p[mask_m]
            ks_m_list.append(float(stats.ks_2samp(val_p, val_o).statistic))
    out["ks"] = float(np.mean(ks_m_list)) if ks_m_list else float("nan")

    for q in (0.10, 0.50, 0.90):
        out[f"p{int(q * 100)}_err"] = float(np.quantile(p, q) - np.quantile(o, q))
    out["amp_err"] = annual_cycle_amplitude(pred) - annual_cycle_amplitude(obs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = compute_validation_metrics(obs, pred)
    out.update(bias=rep.bias, rmse=rep.rmse, kge=rep.kge, pearson_r=rep.pearson_r)
    return out


# -----------------------------------------------------------------------------
# Main benchmark
# -----------------------------------------------------------------------------


def run_benchmark() -> pd.DataFrame:
    """Run all stations x variables x methods x groupings x protocols."""
    print("=" * 92)
    print("  TROPICOR Sprint 8 - EXPLORATORY out-of-sample bias correction benchmark")
    print("  Inputs: thesis-processed files (not yet rebuilt from primary sources)")
    print("=" * 92)

    methods: List[Tuple[str, str]] = [("raw", "-")]
    methods += [(m, g) for m in ("LS", "EQM") for g in GROUPINGS]

    records: List[Dict[str, Any]] = []
    for st in STATIONS:
        fpath = LOCAL_DIR / st["file"]
        if not fpath.exists():
            print(f"[-] Skipping {st['name']}: '{fpath}' not found.")
            continue
        xl = pd.ExcelFile(fpath)

        for spec in VARIABLES:
            var = spec[0]
            var_type = "precipitation" if var == "precipitation" else "temperature"
            obs, mod, notes = load_pair(xl, st["name"], spec)

            test_mask = (obs.index >= TEST_PERIOD[0]) & (obs.index <= TEST_PERIOD[1])
            n_test_split = int(test_mask.sum())
            primary = "split" if n_test_split >= MIN_SPLIT_TEST_MONTHS else "loyo"

            # Non-stationarity context: raw bias drift between periods.
            cal_mask = (obs.index >= CAL_PERIOD[0]) & (obs.index <= CAL_PERIOD[1])
            raw_bias_cal = float((mod[cal_mask] - obs[cal_mask]).mean())
            raw_bias_test = (
                float((mod[test_mask] - obs[test_mask]).mean())
                if n_test_split
                else float("nan")
            )

            print(
                f"\n--- {st['name']} [{st['region']}] | {var} | "
                f"N={len(obs)} | N_test(split)={n_test_split} | "
                f"N_years={obs.index.year.nunique()} | primary={primary.upper()}"
            )
            print(
                f"    Raw bias drift: cal={raw_bias_cal:+.2f} -> "
                f"test={raw_bias_test:+.2f} "
                f"(drift={raw_bias_test - raw_bias_cal:+.2f})"
            )
            for note in notes:
                print(f"    {note}")

            for method, grouping in methods:
                corr = (
                    None
                    if method == "raw"
                    else make_corrector(method, grouping, var_type)
                )
                o_s, p_s, w_s, n_cal = run_split(obs, mod, corr)
                o_l, p_l, w_l = run_loyo(obs, mod, corr)

                for protocol, o_e, p_e, n_w in (
                    ("split", o_s, p_s, w_s),
                    ("loyo", o_l, p_l, w_l),
                ):
                    rec: Dict[str, Any] = {
                        "station": st["name"],
                        "region": st["region"],
                        "variable": var,
                        "method": method,
                        "grouping": grouping,
                        "protocol": protocol,
                        "is_primary": protocol == primary,
                        "n_cal_split": n_cal,
                        "n_test_split": n_test_split,
                        "n_years": int(obs.index.year.nunique()),
                        "n_fit_warnings": n_w,
                        "raw_bias_cal": raw_bias_cal,
                        "raw_bias_test": raw_bias_test,
                        "qc_notes": " | ".join(notes),
                    }
                    rec.update(evaluate(o_e, p_e))
                    records.append(rec)

            _print_series_table(records, st["name"], var, primary)

    df = pd.DataFrame(records)
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    _print_grouping_summary(df)
    print(f"\n[+] Exploratory report written to {OUT_CSV} (git-ignored)")
    return df


def _print_series_table(
    records: List[Dict[str, Any]], station: str, var: str, primary: str
) -> None:
    """Print primary-protocol metrics for one station/variable."""
    rows = [
        r
        for r in records
        if r["station"] == station and r["variable"] == var and r["protocol"] == primary
    ]
    print(
        f"    {'method':<5} {'grouping':<14} {'KS':>6} {'P10err':>8} "
        f"{'P50err':>8} {'P90err':>8} {'AmpErr':>8} | {'bias':>7} "
        f"{'RMSE':>7} {'KGE':>6} {'warn':>5}"
    )
    for r in rows:
        print(
            f"    {r['method']:<5} {r['grouping']:<14} "
            f"{r.get('ks', np.nan):>6.3f} {r.get('p10_err', np.nan):>8.2f} "
            f"{r.get('p50_err', np.nan):>8.2f} {r.get('p90_err', np.nan):>8.2f} "
            f"{r.get('amp_err', np.nan):>8.2f} | {r.get('bias', np.nan):>7.2f} "
            f"{r.get('rmse', np.nan):>7.2f} {r.get('kge', np.nan):>6.2f} "
            f"{r['n_fit_warnings']:>5d}"
        )


def _print_grouping_summary(df: pd.DataFrame) -> None:
    """Compare groupings by actual metric values (medians and ranges) and mean ranks.

    Evaluated strictly on the primary protocol.
    """
    prim = df[df["is_primary"]].copy()
    prim["abs_amp"] = prim["amp_err"].abs()
    prim["abs_p90"] = prim["p90_err"].abs()
    prim["abs_bias"] = prim["bias"].abs()
    prim["family"] = np.where(
        prim["variable"] == "precipitation", "precipitation", "temperature"
    )

    print("\n" + "=" * 92)
    print(
        "  GROUPING COMPARISON (primary protocol: actual values [median (min..max)]"
        " and mean ranks)"
    )
    print("=" * 92)

    for family in ["precipitation", "temperature"]:
        sub_fam = prim[prim["family"] == family]
        unit = "mm" if family == "precipitation" else "°C"
        n_series = sub_fam["variable"].nunique() * sub_fam["station"].nunique()

        print(f"\n{'-' * 92}")
        print(f"  VARIABLE FAMILY: {family.upper()} ({unit}) | {n_series} series")
        print(f"{'-' * 92}")

        # Baseline raw metrics
        raw_rows = sub_fam[sub_fam["method"] == "raw"]
        raw_ks = raw_rows["ks"].median()
        raw_amp = raw_rows["abs_amp"].median()
        raw_p90 = raw_rows["abs_p90"].median()
        raw_bias = raw_rows["abs_bias"].median()
        print(
            f"  [RAW BASELINE] Median across series: "
            f"KS={raw_ks:.3f} | |AmpErr|={raw_amp:.2f} {unit} | "
            f"|P90err|={raw_p90:.2f} {unit} | |bias|={raw_bias:.2f} {unit}\n"
        )

        for method in ["LS", "EQM"]:
            sub_m = sub_fam[sub_fam["method"] == method]
            if sub_m.empty:
                continue

            print(f"  Method: {method}")

            # 1. Actual Metric Values: Median (min .. max)
            val_rows = []
            for grp in GROUPINGS:
                gs = sub_m[sub_m["grouping"] == grp]
                ks_med, ks_lo, ks_hi = (
                    gs["ks"].median(),
                    gs["ks"].min(),
                    gs["ks"].max(),
                )
                amp_med, amp_lo, amp_hi = (
                    gs["abs_amp"].median(),
                    gs["abs_amp"].min(),
                    gs["abs_amp"].max(),
                )
                p90_med, p90_lo, p90_hi = (
                    gs["abs_p90"].median(),
                    gs["abs_p90"].min(),
                    gs["abs_p90"].max(),
                )
                b_med, b_lo, b_hi = (
                    gs["abs_bias"].median(),
                    gs["abs_bias"].min(),
                    gs["abs_bias"].max(),
                )
                val_rows.append(
                    {
                        "grouping": grp,
                        "KS (monthly)": f"{ks_med:.3f} ({ks_lo:.3f}..{ks_hi:.3f})",
                        "|AmpErr|": f"{amp_med:.2f} ({amp_lo:.2f}..{amp_hi:.2f})",
                        "|P90err|": f"{p90_med:.2f} ({p90_lo:.2f}..{p90_hi:.2f})",
                        "|bias|": f"{b_med:.2f} ({b_lo:.2f}..{b_hi:.2f})",
                    }
                )
            df_vals = pd.DataFrame(val_rows).set_index("grouping")
            print("    Actual metric values [median (min..max)]:")
            print("    " + df_vals.to_string().replace("\n", "\n    "))

            # 2. Mean Ranks (1 = best)
            ranks = []
            for _, s in sub_m.groupby(["station", "variable"]):
                r = s.set_index("grouping")[["ks", "abs_amp", "abs_p90", "abs_bias"]]
                ranks.append(r.rank(method="average"))
            if ranks:
                mean_rank = pd.concat(ranks).groupby(level=0).mean()
                print("\n    Mean rank within series (1=best):")
                print(
                    "    "
                    + mean_rank.rename(
                        columns={
                            "ks": "KS",
                            "abs_amp": "|AmpErr|",
                            "abs_p90": "|P90err|",
                            "abs_bias": "|bias|",
                        }
                    )
                    .round(2)
                    .to_string()
                    .replace("\n", "\n    ")
                )
            print()


if __name__ == "__main__":
    run_benchmark()
