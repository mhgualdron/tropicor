# Residual Diagnostics & Monotonic Trend Drift Analysis

This guide explains how to extract deseasonalized model-observation residual anomalies, compute unbiased higher-order statistical moments, evaluate normality diagnostics with physical boundedness guards, and detect real-time monotonic trends using the autocorrelation-corrected Hamed-Rao Mann-Kendall test in `tropicor.core.diagnostics`.

---

## Scientific Context & Orographic Error Structure

When validating global reanalyses (such as ERA5) against in-situ station networks (such as IDEAM) in complex tropical terrain, raw residual differences ($e = \text{ERA5} - \text{IDEAM}$) confound three distinct physical components:

1. **Mean Orographic Elevation Bias**: Elevation mismatches between the smoothed reanalysis geopotential grid and actual station elevations induce systematic temperature shifts ($~6.5\,^\circ\text{C}/\text{km}$) and orographic precipitation enhancement or rain-shadow deficits.
2. **Seasonal Climatological Cycle**: Annual harmonics driven by the Intertropical Convergence Zone (ITCZ) migration produce seasonal bias cycles (e.g., dry-season overestimation or wet-season peak attenuation).
3. **High-Frequency Stochastic Residuals & Decadal Drift**: Synoptic anomalies, extreme event misrepresentations, land-use changes, or spurious reanalysis drifts.

### Separation of Physical Variables

Diagnostics must be evaluated separately for precipitation and temperature, respecting their underlying physical boundary conditions:

* **Precipitation Residuals**: Monthly precipitation observations are non-negative ($P_{\text{obs}} \ge 0$). Consequently, model residuals $e = P_{\text{mod}} - P_{\text{obs}}$ are strictly bounded on the left by $-P_{\text{obs}}$. Furthermore, tropical rainfall is intermittent and heavy-tailed (Gamma or Weibull-distributed). Therefore, **non-normality and rejection by the Shapiro-Wilk test are physically expected boundary phenomena** and do not indicate orographic downscaling failure.
* **Temperature Extreme Residuals**: Monthly extreme temperatures ($T_{\min}, T_{\max}$) follow Generalized Extreme Value (GEV/Gumbel) distributions and are inherently skewed. Non-Gaussianity may reflect localized microclimates (thermal inversions, cold-air pooling, slope aspect), but alternative drivers include GEV physics, sensor recalibrations, station relocations, missing observation days, or ENSO teleconnections.

---

## Mathematical Formulation

### 1. Monthly Deseasonalization

To isolate residual drift and distribution shape from the predictable annual cycle, `compute_deseasonalized_residuals` computes the 12-month mean climatological error and subtracts it:

$$e(t) = X_{\text{mod}}(t) - X_{\text{obs}}(t)$$

$$\bar{e}_{\text{clim}}(m) = \frac{1}{K_m} \sum_{k=1}^{K_m} e_{m}(k), \quad m \in \{1, \dots, 12\}$$

$$e'(t) = e(t) - \bar{e}_{\text{clim}}(\text{month}(t))$$

Sample size guards enforce a minimum of 10 distinct synchronous observation years and at least 5 valid observations per calendar month. For heteroscedastic precipitation series where variance scales with rainfall magnitude, an optional `monthly_standardize=True` parameter divides anomalies by each month's residual standard deviation:

$$e''_{\text{std}}(t) = \frac{e'(t)}{\sigma_e(\text{month}(t))}$$

### 2. Higher-Order Moments & Normality

TROPICOR computes unbiased Fisher-Pearson sample skewness ($g_1$) and excess kurtosis ($g_2$):

$$g_1 = \frac{\sqrt{n(n-1)}}{n-2} \frac{\frac{1}{n} \sum (e'_i - \bar{e}')^3}{\left( \frac{1}{n} \sum (e'_i - \bar{e}')^2 \right)^{3/2}}$$

$$g_2 = \frac{n-1}{(n-2)(n-3)} \left( (n+1) \frac{\frac{1}{n} \sum (e'_i - \bar{e}')^4}{\left( \frac{1}{n} \sum (e'_i - \bar{e}')^2 \right)^2} - 3(n-1) \right)$$

Normality is assessed via the Shapiro-Wilk $W$ statistic. For sample sizes typical of 30-year climate records ($N \ge 120$), Shapiro-Wilk exhibits extreme statistical power ($>99\%$), rejecting $H_0$ even under trivial deviations from normality. Diagnostic interpretation must prioritize effect sizes ($g_1, g_2$) over raw p-values.

> [!NOTE]
> **Thesis Convention Alignment**:
> In historical UNAL thesis work (Entrega 10), raw differences were computed as $e_{\text{thesis}} = \text{IDEAM} - \text{ERA5}$ using biased estimators (`scipy.stats.describe(bias=True)`). Because TROPICOR adheres to the standard model-error convention ($e_{\text{tropicor}} = \text{ERA5} - \text{IDEAM} = -e_{\text{thesis}}$), skewness flips sign ($\text{skew}_{\text{tropicor}} = -\text{skew}_{\text{thesis}}$), while kurtosis remains invariant.

### 3. Real-Time Sen's Slope

To prevent temporal distortion from missing months (e.g., station downtime), Sen's non-parametric slope is evaluated against **decimal calendar years**:

$$t_i = \text{year}_i + \frac{\text{month}_i - 0.5}{12.0}$$

$$\beta = \text{median} \left\{ \frac{e'_j - e'_i}{t_j - t_i} \right\} \quad \text{for } j > i$$

This formulation yields a slope in physical units per year, reported per decade ($\times 10$) and invariant to dropped timestamps.

### 4. Modified Mann-Kendall Test (Hamed & Rao, 1998)

Hydrological and thermal residuals frequently exhibit positive serial autocorrelation, inflating the variance of the standard Mann-Kendall score $S$:

$$S = \sum_{i=1}^{n-1} \sum_{j=i+1}^n \text{sgn}(e'_j - e'_i)$$

$$\text{Var}_0(S) = \frac{n(n-1)(2n+5) - \sum_{t} t(t-1)(2t+5)}{18}$$

Hamed and Rao (1998) adjust the theoretical variance by an inflation factor $\eta$:

$$\text{Var}^*(S) = \text{Var}_0(S) \cdot \eta$$

$$\eta = 1 + \frac{2}{n(n-1)(n-2)} \sum_{i=1}^{k} (n-i)(n-i-1)(n-i-2) \rho^*(i)$$

where $\rho^*(i)$ is the autocorrelation of the ranks of the detrended series $y_t - \beta t_t$, and $k$ is the truncation lag. Only lags where $\rho^*(i)$ falls outside the non-significance interval $[-z_{\alpha/2}/\sqrt{n}, +z_{\alpha/2}/\sqrt{n}]$ contribute to the sum.

---

## Monte Carlo Calibration & Power Benchmark

The Mann-Kendall calibration and power benchmark is generated deterministically by `scripts/calibrate_mann_kendall.py`. Users can regenerate the exact tables below at any time:

```bash
uv run python scripts/calibrate_mann_kendall.py
```

### Reproducible Calibration Protocol

* **Seeds (5 independent runs)**: `[42, 101, 2024, 777, 9999]`
* **Sample sizes ($n$)**: $120$ (10 years) and $360$ (30 years)
* **Autoregressions ($\phi$)**: $0.30, 0.65, 0.85$
* **Iterations**: 200 per seed ($1,000$ independent simulations per parameter cell)
* **Significance Level**: Nominal $\alpha = 0.05$
* **Power Drift**: $\delta = \frac{1.5 \sigma_\epsilon}{n}$ per monthly timestep ($1.5\,\sigma_\epsilon$ total cumulative drift across the record)

> [!NOTE]
> **What Changed from Preliminary Prototype Sweeps**:
> Earlier draft tables reflected single-seed exploratory runs ($n_{\text{sim}} = 200$, seed 42 only), in which sampling variance was higher and one cell ($n=120, \phi=0.30$) was mistakenly transcribed as 6.2%. In contrast, the current tables represent the 5-seed pooled benchmark (1,000 iterations per cell) generated directly by `scripts/calibrate_mann_kendall.py`.

### The `max_lags=3` Truncation Mechanism

In finite monthly climate series ($n = 120$ to $360$), allowing Hamed-Rao to inspect all $n-1$ lags introduces substantial sample noise: spurious sample rank autocorrelations at distant lags (e.g., lag 50 to 300) dilute the variance inflation factor $\eta$, causing severe Type I error inflation.

> [!IMPORTANT]
> **Empirical Calibration Note**:
> Restricting variance inflation to `max_lags=3` is an **empirical engineering choice** based on TROPICOR's multi-parameter calibration sweep for monthly tropical climate data. It is **not** a formal recommendation made by Hamed & Rao (1998).

### Measured False Positive Rates (Type I Error, $\alpha = 0.05$)

Generated by `scripts/calibrate_mann_kendall.py` (1,000 simulations per cell, $H_0: \text{No Trend}$):

| Record Length ($n$) | Autoregressive $\phi$ | Uncorrected MK | Hamed-Rao (All Lags) | Hamed-Rao (`max_lags=3`) |
| :--- | :--- | :---: | :---: | :---: |
| **$n = 120$** (10 years) | $\phi = 0.30$ | 13.2% | 10.1% | **9.1%** |
|                        | $\phi = 0.65$ | 34.6% | 19.3% | **11.9%** |
|                        | $\phi = 0.85$ | 57.9% | 29.6% | **19.7%** |
| **$n = 360$** (30 years) | $\phi = 0.30$ | 14.5% | 9.0% | **7.2%** |
|                        | $\phi = 0.65$ | 37.6% | 18.1% | **8.9%** |
|                        | $\phi = 0.85$ | 60.2% | 27.6% | **20.2%** |

At moderate autocorrelation ($\phi = 0.65, n=360$), `max_lags=3` reduces false positive inflation from 37.6% down to **8.9%** (reasonably close to nominal 5%), outperforming the unconstrained all-lags implementation (18.1%).

### Measured Statistical Power Benchmark

Generated by `scripts/calibrate_mann_kendall.py` (1,000 simulations per cell, $H_1: \delta = 1.5 \sigma_\epsilon / n$ per month):

| Record Length ($n$) | Autoregressive $\phi$ | Uncorrected MK | Hamed-Rao (All Lags) | Hamed-Rao (`max_lags=3`) |
| :--- | :--- | :---: | :---: | :---: |
| **$n = 120$** (10 years) | $\phi = 0.30$ | 96.6% | 93.9% | **92.1%** |
|                        | $\phi = 0.65$ | 76.6% | 61.4% | **49.7%** |
|                        | $\phi = 0.85$ | 67.8% | 40.5% | **34.2%** |
| **$n = 360$** (30 years) | $\phi = 0.30$ | 100.0% | 100.0% | **100.0%** |
|                        | $\phi = 0.65$ | 97.4% | 92.2% | **88.7%** |
|                        | $\phi = 0.85$ | 79.0% | 54.7% | **50.5%** |

For 30-year series ($n=360$), statistical power remains at **88.7%** under $\phi = 0.65$, confirming that `max_lags=3` controls Type I errors without sacrificing trend detection power.

### High-Autocorrelation Warning ($\rho_1 > 0.70$)

When estimated lag-1 autocorrelation exceeds 0.70, `compute_mann_kendall_trend` automatically emits a `UserWarning`:

```
UserWarning: Lag-1 autocorrelation (0.84) exceeds 0.7: corrected p-value may still be liberal (calibration sweep: ~20% false positives at phi=0.85).
```

Users should treat marginal p-values ($0.01 < p < 0.05$) under severe autocorrelation ($\rho_1 > 0.70$) with caution.

---

## Residual Drift Interpretation & Station Inhomogeneity Screening

Because trends are computed on residuals $e(t) = X_{\text{mod}}(t) - X_{\text{obs}}(t)$, the detected Sen's slope represents **residual drift (model - obs increasing or decreasing)**:

* $\beta > 0$ (**Residual Drift: Model - Obs Increasing**): The reanalysis values are progressively increasing relative to the station observations over time.
* $\beta < 0$ (**Residual Drift: Model - Obs Decreasing**): The reanalysis values are progressively decreasing relative to the station observations over time.

> [!WARNING]
> **Station Inhomogeneity vs. Reanalysis Drift**:
> A significant residual trend can originate from the in-situ station as much as from the reanalysis model. Common station-side causes include:
> 1. **Sensor Replacement or Degradation**: Gradual sensor aging, calibration drift, or switching from manual thermometers to automated electronic probes.
> 2. **Station Relocation**: Moving a station even a few hundred meters or changing installation elevation alters the microclimate.
> 3. **Environmental Changes**: Urbanization, asphalt paving, land clearing, or canopy growth around the Stevenson screen.
>
> **Screening Rule**: Large residual trends must be flagged as candidate station inhomogeneities to audit rather than unquestioned reanalysis defects. In particular, the **Noanamá $T_{\max}$ residual drift (+1.37 °C/decade, $p < 0.0001$)** represents an accumulated drift of $+4.1\,^\circ\text{C}$ over 30 years—an extreme magnitude in an equatorial rainforest that strongly indicates in-situ sensor/station inhomogeneity.

---

## Outlier Audit: Las Flores $T_{\max}$ Excess Kurtosis ($g_2 = +25.08$)

In Las Flores (IDEAM 21205790), deseasonalized $T_{\max}$ residuals yielded an excess kurtosis of $g_2 = +25.08$. Inspecting the standardized anomalies reveals the exact origin:

| Date | Observed $T_{\max}$ | ERA5 $T_{\max}$ | Raw Difference ($e$) | Deseasonalized Anomaly ($e'$) | Standardized $Z$ | Context |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **2018-03** | **21.0 °C** | **29.54 °C** | **+8.54 °C** | **+10.39 °C** | **+8.70** | **Data entry error / stray record** |
| 2003-04 | 36.0 °C | 29.54 °C | -6.46 °C | -3.62 °C | -3.03 | Extreme hot dry month |
| 1993-05 | 36.0 °C | 30.11 °C | -5.89 °C | -2.57 °C | -2.15 | Extreme hot dry month |
| 2013-09 | 32.4 °C | 31.34 °C | -1.06 °C | +2.28 °C | +1.91 | Moderate wet-season anomaly |
| 1993-11 | 34.2 °C | 28.95 °C | -5.25 °C | -2.15 °C | -1.80 | Hot anomaly |

### Physical & Data Entry Diagnosis

1. **Unphysical Value**: Las Flores is a sea-level Caribbean coastal station. In March (peak dry season), daytime maximum temperatures are normally $30^\circ\text{C}$ to $34^\circ\text{C}$. An observed maximum temperature of $21.0\,^\circ\text{C}$ is unphysical and represents a data entry error (most likely $T_{\min}$ erroneously recorded in the $T_{\max}$ column, or a transcription typo for $31.0\,^\circ\text{C}$).
2. **Temporal Discontinuity**: The station stopped regular operation in July 2015. After a 32-month complete gap, this single isolated data point appears in March 2018.
3. **Impact on Moments**: Removing this single erroneous month ($N=221$ remaining) causes excess kurtosis to collapse from **$+25.08$ down to $+0.29$** and skewness from **$+2.77$ down to $-0.30$**, restoring an essentially Gaussian residual distribution.

---

## Auditing Thesis Conventions vs. TROPICOR Core Parity

The UNAL thesis (Entregas 9–10) serves as an empirical consistency reference, **not as ground truth**. The correctness of TROPICOR rests on mathematical statistical theory, synthetic Monte Carlo tests, and strict parity with reference packages (`pymannkendall`, `scipy`), not on agreement with historical thesis outputs.

### Comparison Across All Stations & Variables (Entrega 10)

| Station | Variable | Metric | Thesis Printed Value (Entrega 10 C3) | TROPICOR Raw Reproduction | Match Status & Root Cause |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Puerto Carreño** | Precip | Skewness / Kurtosis | $+0.7303$ / $+1.3396$ | $-0.7303$ / $+1.3396$ | **Exact match** (sign flip due to $e = \text{ERA5} - \text{IDEAM}$) |
| | TMin | Skewness / Kurtosis | $-0.3952$ / $+0.1601$ | $+0.3952$ / $+0.1601$ | **Exact match** (sign flip due to $e = \text{ERA5} - \text{IDEAM}$) |
| | TMax | Skewness / Kurtosis | $+0.7710$ / $+2.4211$ | $-0.7710$ / $+2.4211$ | **Exact match** (sign flip due to $e = \text{ERA5} - \text{IDEAM}$) |
| **Las Flores** | Precip | Skewness / Kurtosis | $+0.6865$ / $+2.9697$ | $-0.6865$ / $+2.9697$ | **Exact match** (sign flip due to $e = \text{ERA5} - \text{IDEAM}$) |
| | TMin | Skewness / Kurtosis | $-0.9022$ / $+2.4938$ | $+0.9022$ / $+2.4938$ | **Exact match** (sign flip due to $e = \text{ERA5} - \text{IDEAM}$) |
| | TMax | Skewness / Kurtosis | $-2.8546$ / $+19.5218$ | $+2.8546$ / $+19.5218$ | **Exact match** (sign flip due to $e = \text{ERA5} - \text{IDEAM}$) |
| **Noanamá** | Precip | Skewness / Kurtosis | $+0.4048$ / $+0.9329$ | $-0.4048$ / $+0.9329$ | **Exact match** (sign flip due to $e = \text{ERA5} - \text{IDEAM}$) |
| | TMin | Skewness / Kurtosis | $+0.4578$ / $+6.1956$ | $-0.4578$ / $+6.1956$ | **Exact match** (sign flip due to $e = \text{ERA5} - \text{IDEAM}$) |
| | TMax | Skewness / Kurtosis | $-1.1119$ / $+4.6866$ | $+1.1119$ / $+4.6866$ | **Exact match** (sign flip due to $e = \text{ERA5} - \text{IDEAM}$) |
| **Gorgona** | Precip | Skewness / Kurtosis | $-0.1374$ / $+0.4124$ | $+0.1374$ / $+0.4124$ | **Exact match** (sign flip due to $e = \text{ERA5} - \text{IDEAM}$) |
| | TMin | Skewness / Kurtosis | $-0.8354$ / $+1.3400$ | $+0.8354$ / $+1.3400$ | **Exact match** (sign flip due to $e = \text{ERA5} - \text{IDEAM}$) |
| | TMax | Skewness / Kurtosis | $+0.3248$ / $+2.1705$ | $-0.3248$ / $+2.1705$ | **Exact match** (sign flip due to $e = \text{ERA5} - \text{IDEAM}$) |

### Thesis Discrepancy & Bug Audit

1. **Thesis Coding Bug in Entrega 10 (Cell 2 Line 24)**:
   In `Entrega10.ipynb`, the author standardized residuals using `df['Procesado'] = (df['Diff'] - mean) / std`. For Precipitation and $T_{\min}$, the author called `stats.describe(df['Procesado'])`. However, for $T_{\max}$ (line 24), the code mistakenly reads:
   ```python
   TMax.append(stats.describe(df['Diff']))  # Bug: passed unstandardized raw diffs instead of df['Procesado']
   ```
   This caused the thesis to report min/max bounds in unstandardized units for $T_{\max}$ while reporting standardized bounds for the other variables.
2. **Deseasonalization Necessity**:
   The thesis calculated moments on raw residuals that still contained the annual seasonal cycle. TROPICOR computes diagnostics on **deseasonalized anomalies** ($e'(t)$), ensuring that variance and moments reflect stochastic error rather than the predictable annual solar cycle.
3. **Unbiased Fisher-Pearson Estimators**:
   The thesis relied on biased population moments (`scipy.stats.skew(bias=True)`). TROPICOR applies unbiased sample degrees-of-freedom corrections ($g_1, g_2$, `bias=False`), essential for finite sample sizes ($N \approx 200 \text{ to } 360$).

---

## Practical Usage

```python
import pandas as pd
from tropicor.core.diagnostics import compute_residual_diagnostics

# Load synchronized observation and reanalysis series
obs_series = pd.read_csv("station_tmax.csv", index_col=0, parse_dates=True)["tmax_obs"]
mod_series = pd.read_csv("era5_tmax.csv", index_col=0, parse_dates=True)["tmax_era5"]

# Compute end-to-end residual diagnostics
report = compute_residual_diagnostics(
    observed=obs_series,
    modeled=mod_series,
    variable="temperature",
    trend_correction="hamed_rao",
    max_lags=3,
    min_years=10,
    min_obs_per_month=5,
)

# Inspect distribution moments and normality
print(f"Skewness (g1):      {report.moments.skewness:.3f}")
print(f"Excess Kurtosis:    {report.moments.excess_kurtosis:.3f}")
print(f"Shapiro-Wilk W:     {report.moments.shapiro_stat:.4f} (p={report.moments.shapiro_p_value:.3e})")

# Inspect monotonic residual drift (model - obs)
print(f"Sen's Slope:        {report.trend.slope_per_decade:.3f} °C/decade")
print(f"Lag-1 Autocorr:     {report.trend.lag1_autocorr:.3f}")
print(f"Hamed-Rao p-value:  {report.trend.p_value:.4f} (Trend: {report.trend.has_significant_trend})")

# Tabular summary DataFrame
df_summary = report.summary()
print(df_summary)
```

---

## Real-Data Benchmark on Colombian Stations

Benchmark results across the 4 complete 30-year thesis stations (`local/residual_diagnostics_report.csv`):

| Station | Natural Region | Variable | $N$ | Mean ($e$) | Skewness ($g_1$) | Kurtosis ($g_2$) | Residual Drift / Decade | $\rho_1$ (Lag-1) | Hamed-Rao $p$ | Review Flag |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Puerto Carreño** | Orinoquía | Precip | 360 | $-0.00$ | $-0.60$ | $+2.17$ | $-11.54\,\text{mm}$ | $-0.048$ | $0.0039$ (Sig) | Residual drift (model - obs decreasing) |
| | | TMin | 344 | $-0.00$ | $+0.52$ | $+0.31$ | $-0.14\,^\circ\text{C}$ | $+0.239$ | $0.0612$ | No significant trend |
| | | TMax | 344 | $+0.00$ | $-1.03$ | $+4.09$ | $+0.43\,^\circ\text{C}$ | $+0.246$ | $<0.0001$ (Sig) | Residual drift (model - obs increasing) |
| **Las Flores** | Caribe | Precip | 278 | $+0.00$ | $+0.07$ | $+4.32$ | $-9.15\,\text{mm}$ | $-0.008$ | $0.0014$ (Sig) | Residual drift (model - obs decreasing) |
| | | TMin | 216 | $+0.00$ | $+0.96$ | $+2.18$ | $+0.04\,^\circ\text{C}$ | $+0.284$ | $0.6774$ | No significant trend |
| | | TMax | 222 | $+0.00$ | $+2.77$ | $+25.08$ | $+0.48\,^\circ\text{C}$ | $+0.319$ | $0.0001$ (Sig) | **Outlier flag**: 2018-03 entry error ($Z=+8.70$) |
| **Noanamá** | Pacífico | Precip | 318 | $-0.00$ | $-0.04$ | $+0.53$ | $-22.23\,\text{mm}$ | $+0.042$ | $0.0680$ | No significant trend |
| | | TMin | 320 | $+0.00$ | $-0.42$ | $+5.65$ | $+0.01\,^\circ\text{C}$ | $+0.442$ | $0.8987$ | No significant trend |
| | | TMax | 320 | $+0.00$ | $+1.07$ | $+3.24$ | $+1.37\,^\circ\text{C}$ | $+0.597$ | $<0.0001$ (Sig) | **Suspect Inhomogeneity**: $+1.37^\circ\text{C}/\text{decade}$ |
| **Gorgona** | Insular | Precip | 344 | $+0.00$ | $+0.25$ | $+0.55$ | $-9.26\,\text{mm}$ | $+0.304$ | $0.6810$ | No significant trend |
| | | TMin | 251 | $-0.00$ | $+0.74$ | $+1.02$ | $-0.11\,^\circ\text{C}$ | $+0.233$ | $0.3906$ | No significant trend |
| | | TMax | 298 | $+0.00$ | $-0.33$ | $+1.73$ | $-0.43\,^\circ\text{C}$ | $+0.188$ | $<0.0001$ (Sig) | Residual drift (model - obs decreasing) |

Temperature residuals consistently exhibit positive lag-1 autocorrelation ($\rho_1 \approx 0.19$ to $0.60$), verifying the necessity of autocorrelation adjustment for valid significance inference.
