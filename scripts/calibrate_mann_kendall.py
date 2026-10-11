"""Reproducible Monte Carlo Calibration & Power Benchmark for Mann-Kendall Engine.

Evaluates Type I error (false positive rate) and statistical power for:
1. Uncorrected Mann-Kendall test
2. Hamed-Rao modified test with max_lags=3 (TROPICOR empirical default)
3. Hamed-Rao modified test with all significant lags (unconstrained)

Simulations evaluate AR(1) processes across:
- Sample lengths: n in {120, 360}
- Autoregressive parameters: phi in {0.30, 0.65, 0.85}
- Seeds: [42, 101, 2024, 777, 9999] (5 independent runs)
- Iterations: 200 per seed (1,000 total iterations per parameter cell)
- Nominal significance: alpha = 0.05
- Power trend size: delta = 1.5 * sigma / n per timestep
  (total drift of 1.5 std across series)
"""

import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd

from tropicor.core.diagnostics import compute_mann_kendall_trend

SEEDS: List[int] = [42, 101, 2024, 777, 9999]
PHIS: List[float] = [0.30, 0.65, 0.85]
NS: List[int] = [120, 360]
N_SIM: int = 200
ALPHA: float = 0.05


def run_calibration_sweep() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute reproducible Monte Carlo calibration and power benchmark."""
    print("=" * 88)
    print("   TROPICOR: MANN-KENDALL REPRODUCIBILITY & CALIBRATION SWEEP")
    print("=" * 88)
    print(f"Seeds ({len(SEEDS)}):       {SEEDS}")
    print(f"Sample lengths n:   {NS}")
    print(f"Autoregressions:    {PHIS}")
    print(f"Iterations:         {N_SIM} per seed ({len(SEEDS) * N_SIM} per cell)")
    print(f"Significance level: alpha = {ALPHA}")
    print("=" * 88)

    type1_records = []
    power_records = []

    # Suppress UserWarnings emitted when phi=0.85 during simulation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)

        # ---------------------------------------------------------------------
        # 1. Type I Error (False Positive Rate) Sweep (H0: No Trend)
        # ---------------------------------------------------------------------
        print("\n[+] Running Part 1: Type I Error Sweep (H0: No Trend)...")
        for n_len in NS:
            dates = pd.date_range("2000-01-01", periods=n_len, freq="MS")
            for phi in PHIS:
                uncorr_rates: List[float] = []
                hr3_rates: List[float] = []
                hr_all_rates: List[float] = []

                for seed in SEEDS:
                    rng = np.random.default_rng(seed)
                    uncorr_rejs = 0
                    hr3_rejs = 0
                    hr_all_rejs = 0

                    for _ in range(N_SIM):
                        e = np.zeros(n_len)
                        innov = rng.normal(0, 1, n_len)
                        for t in range(1, n_len):
                            e[t] = phi * e[t - 1] + innov[t]

                        s = pd.Series(e, index=dates)

                        # 1. Uncorrected
                        tr_u = compute_mann_kendall_trend(
                            s, correction="none", alpha=ALPHA
                        )
                        if tr_u.p_value < ALPHA:
                            uncorr_rejs += 1

                        # 2. Hamed-Rao max_lags=3
                        tr_hr3 = compute_mann_kendall_trend(
                            s, correction="hamed_rao", max_lags=3, alpha=ALPHA
                        )
                        if tr_hr3.p_value < ALPHA:
                            hr3_rejs += 1

                        # 3. Hamed-Rao all significant lags
                        tr_hr_all = compute_mann_kendall_trend(
                            s, correction="hamed_rao", max_lags=None, alpha=ALPHA
                        )
                        if tr_hr_all.p_value < ALPHA:
                            hr_all_rejs += 1

                    uncorr_rates.append(uncorr_rejs / N_SIM)
                    hr3_rates.append(hr3_rejs / N_SIM)
                    hr_all_rates.append(hr_all_rejs / N_SIM)

                mean_u = float(np.mean(uncorr_rates))
                mean_hr3 = float(np.mean(hr3_rates))
                mean_hr_all = float(np.mean(hr_all_rates))

                type1_records.append(
                    {
                        "n": n_len,
                        "phi": phi,
                        "uncorrected": mean_u,
                        "hr_all_lags": mean_hr_all,
                        "hr_max_lags_3": mean_hr3,
                    }
                )

                print(
                    f"  n={n_len:>3} | phi={phi:.2f} | "
                    f"Uncorr: {mean_u:>5.1%} | "
                    f"HR(all): {mean_hr_all:>5.1%} | "
                    f"HR(max_lags=3): {mean_hr3:>5.1%}"
                )

        # ---------------------------------------------------------------------
        # 2. Statistical Power Sweep (H1: True Trend Added)
        # ---------------------------------------------------------------------
        print("\n[+] Running Part 2: Statistical Power Benchmark (H1: True Trend)...")
        for n_len in NS:
            dates = pd.date_range("2000-01-01", periods=n_len, freq="MS")
            # Decimal years step is 1/12 per month; trend = (1.5 / n_len) per month
            # in decimal years: annual_slope = (1.5 / n_len) * 12
            monthly_slope = 1.5 / n_len
            t_month = np.arange(n_len)

            for phi in PHIS:
                p_u_rates: List[float] = []
                p_hr3_rates: List[float] = []
                p_hr_all_rates: List[float] = []

                for seed in SEEDS:
                    rng = np.random.default_rng(seed)
                    p_u_rejs = 0
                    p_hr3_rejs = 0
                    p_hr_all_rejs = 0

                    for _ in range(N_SIM):
                        e = np.zeros(n_len)
                        innov = rng.normal(0, 1, n_len)
                        for t in range(1, n_len):
                            e[t] = phi * e[t - 1] + innov[t]

                        s_trend = pd.Series(e + monthly_slope * t_month, index=dates)

                        tr_u = compute_mann_kendall_trend(
                            s_trend, correction="none", alpha=ALPHA
                        )
                        if tr_u.p_value < ALPHA:
                            p_u_rejs += 1

                        tr_hr3 = compute_mann_kendall_trend(
                            s_trend, correction="hamed_rao", max_lags=3, alpha=ALPHA
                        )
                        if tr_hr3.p_value < ALPHA:
                            p_hr3_rejs += 1

                        tr_hr_all = compute_mann_kendall_trend(
                            s_trend, correction="hamed_rao", max_lags=None, alpha=ALPHA
                        )
                        if tr_hr_all.p_value < ALPHA:
                            p_hr_all_rejs += 1

                    p_u_rates.append(p_u_rejs / N_SIM)
                    p_hr3_rates.append(p_hr3_rejs / N_SIM)
                    p_hr_all_rates.append(p_hr_all_rejs / N_SIM)

                mean_pu = float(np.mean(p_u_rates))
                mean_phr3 = float(np.mean(p_hr3_rates))
                mean_phr_all = float(np.mean(p_hr_all_rates))

                power_records.append(
                    {
                        "n": n_len,
                        "phi": phi,
                        "monthly_slope": monthly_slope,
                        "uncorrected": mean_pu,
                        "hr_all_lags": mean_phr_all,
                        "hr_max_lags_3": mean_phr3,
                    }
                )

                print(
                    f"  n={n_len:>3} | phi={phi:.2f} | "
                    f"Uncorr: {mean_pu:>5.1%} | "
                    f"HR(all): {mean_phr_all:>5.1%} | "
                    f"HR(max_lags=3): {mean_phr3:>5.1%}"
                )

    df_t1 = pd.DataFrame(type1_records)
    df_pw = pd.DataFrame(power_records)

    print("\n" + "=" * 88)
    print("MARKDOWN TABLE: TYPE I ERROR (FALSE POSITIVE RATE)")
    print("=" * 88)
    header = (
        "| Record Length ($n$) | Autoregressive $\\phi$ | "
        "Uncorrected MK | Hamed-Rao (All Lags) | Hamed-Rao (`max_lags=3`) |"
    )
    print(header)
    print("| :--- | :--- | :---: | :---: | :---: |")
    for _, r in df_t1.iterrows():
        n_val = int(r["n"])
        n_str = f"**$n = {n_val}$** ({n_val // 12} years)" if r["phi"] == 0.30 else ""
        row = (
            f"| {n_str:<22} | $\\phi = {r['phi']:.2f}$ | "
            f"{r['uncorrected']:.1%} | {r['hr_all_lags']:.1%} | "
            f"**{r['hr_max_lags_3']:.1%}** |"
        )
        print(row)

    print("\n" + "=" * 88)
    print("MARKDOWN TABLE: STATISTICAL POWER BENCHMARK")
    print("=" * 88)
    print(header)
    print("| :--- | :--- | :---: | :---: | :---: |")
    for _, r in df_pw.iterrows():
        n_val = int(r["n"])
        n_str = f"**$n = {n_val}$** ({n_val // 12} years)" if r["phi"] == 0.30 else ""
        row = (
            f"| {n_str:<22} | $\\phi = {r['phi']:.2f}$ | "
            f"{r['uncorrected']:.1%} | {r['hr_all_lags']:.1%} | "
            f"**{r['hr_max_lags_3']:.1%}** |"
        )
        print(row)

    return df_t1, df_pw


if __name__ == "__main__":
    run_calibration_sweep()
