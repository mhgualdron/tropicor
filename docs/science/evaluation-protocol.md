# Pre-Specified Evaluation Protocol for Statistical Downscaling & Orographic Bias Correction

**Protocol Version**: 1.0  
**Latest Amendment**: Protocol v1.2 (Phase 2a precipitation clarifications and one substantive design decision, D.10; see Appendix D). Earlier: v1.1 (Appendix C)  
**Registration Date**: 2026-10-15  
**Framework**: `tropicor` (Tropical Correction and Orographic Resolution)  
**Baseline Commit**: `4cffe12`  
**Status**: Pre-Specified (Evaluation criteria and partitions established prior to daily method execution)  
**Target Variables**: Daily Precipitation Accumulation ($P$), Daily Maximum Temperature ($T_{\max}$), Daily Minimum Temperature ($T_{\min}$)  
**Target Reanalysis**: ECMWF ERA5 Hourly Reanalysis (single levels, 0.25° grid)  

---

## 1. Scope, Purpose & Pre-Specification Disclosures

### 1.1 Purpose
This document establishes the scientific falsification protocol, temporal partitioning, metric definitions, uncertainty quantification, operational aggregation, and parsimony progression rules for evaluating daily statistical downscaling and bias correction methods within `tropicor`. 

Statistical downscaling models are vulnerable to *over-fitting to calibration periods* and *post-hoc metric selection*, where model architectures, transfer functions, and filtering thresholds are iteratively tuned until summary statistics appear favorable. To enforce strict methodological discipline, this protocol freezes all evaluation rules, partitions, and thresholds before evaluating daily downscaling methods. If a proposed correction method fails the pre-specified out-of-sample criteria, that failure constitutes a definitive scientific finding and must be reported as such without post-hoc threshold adjustment.

### 1.2 Pre-Specification Status & Transparency Disclosures
In the interest of rigorous scientific transparency, the following baseline conditions are formally declared:

1. **Pre-Specified, Not Blind Criteria**:  
   The thresholds defined herein are **pre-specified**, not strictly blind. The author previously evaluated the raw ERA5 biases across the **complete available observation series** (including the precipitation validation years: 1975–2015 for Jardín Botánico and 1975–2000 for Aguas Claras, as well as the 1990–1994 temperature test records). Consequently, these criteria define the **minimum acceptable corrective performance** required for downscaled outputs to be credible in tropical complex terrain, informed by known raw model deficiencies.
2. **Prior Monthly Implementations vs. Daily Scope**:  
   Monthly downscaling models (`LinearScaling` and `EmpiricalQuantileMapping`) were previously implemented in Sprint 8 for monthly aggregated data. This protocol exclusively governs the implementation, training, and out-of-sample evaluation of **daily downscaling methods**.
3. **Validation Partition Integrity**:  
   As of commit `4cffe12` (the baseline of this protocol), **no daily downscaling method has been executed or evaluated on the out-of-sample validation partitions**.
4. **Amendment Requirement**:  
   Any subsequent alteration to partitions, metrics, or falsification thresholds must be recorded as a formal, versioned protocol amendment (e.g., Protocol v1.1) detailing the justification.

---

## 2. Benchmark Stations & Temporal Partitions

### 2.1 Benchmark Stations
The initial verification suite targets two contrasting physiographic and microclimatic benchmarks in Colombia. Metadata, audited coordinates, and elevations are drawn strictly from the audited station catalog (`StationCatalog` and `local/station_coords_audit.csv`):

1. **Jardín Botánico, Bogotá** (IDEAM ID `21205710`):
   - Department / Municipality: Cundinamarca / Bogotá
   - Audited Coordinates: 4.667867°N, -74.101034°W
   - Audited Surface Elevation: 2552.0 masl
   - Nearest ERA5 Grid Cell (Center): 4.75°N, -74.00°W ($z_{\text{era5}} = 2753.4\text{ m}$)
   - Elevation Difference: $\Delta z = z_{\text{station}} - z_{\text{model}} = -201.4\text{ m}$
2. **Aguas Claras, Cesar** (IDEAM ID `23215030`):
   - Department / Municipality: Cesar / Aguachica
   - Audited Coordinates: 8.228889°N, -73.602778°W
   - Audited Surface Elevation: 133.8 masl
   - Nearest ERA5 Grid Cell (Center): 8.25°N, -73.50°W ($z_{\text{era5}} = 762.4\text{ m}$)
   - Elevation Difference: $\Delta z = z_{\text{station}} - z_{\text{model}} = -628.6\text{ m}$
   - **Data Quality Limitation**: 100% of the temperature observations exported from IDEAM DHIME for Aguas Claras are flagged with approval level `Preliminar` (0% `Definitivo`). This limitation is logged as a source of observational uncertainty.

#### 2.1.1 Spatial Grid Cell Selection Rule
Downscaling extraction strictly uses the **geographically closest grid cell (center cell)**, permanently fixed. The surrounding $3 \times 3$ grid cells are evaluated purely as diagnostic sensitivity tools to inspect local orographic lapse rates. **Selecting a neighboring grid cell based on observed station bias or elevation minimization $|\Delta z|$ is strictly prohibited**, as doing so introduces post-hoc spatial selection bias. For reference, in the baseline 3×3 analysis (1990–1994, `local/reports/temperature_grid_3x3.csv`), the minimum-$|\Delta z|$ cell at Jardín Botánico (Δz = +2.9 m) showed a raw $T_{\max}$ bias of −3.18 °C versus −1.66 °C in the center cell, whereas at Aguas Claras the minimum-$|\Delta z|$ cell (Δz = −21.2 m) showed −2.67 °C versus −5.54 °C in the center cell. The effect of selecting by $|\Delta z|$ is therefore not consistent across stations.

### 2.2 Temporal Partitions
To ensure genuine out-of-sample validation, historical records are partitioned into non-overlapping calibration and validation periods. 

The temperature evaluation record begins in 1990 as a **scope decision of the download** (ERA5 itself is available since 1940; the project downloaded hourly 2m temperature starting in 1990), not because of any limit in ERA5 availability. 

For the temperature validation records (2005–2020 for Jardín Botánico and 2001–2011 for Aguas Claras), the author declares that the IDEAM observation files were only inventoried (date range, number of valid days, and Definitivo/Preliminar proportion) and that **no comparison, bias, correlation, or any other statistic against ERA5 has been computed for these years** (the ERA5 download is in progress; the files present have not been opened or compared). No downscaling method has processed them.

| Variable | Station | Total Record | Calibration Partition | Validation Partition | Evaluation Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Precipitation** ($P$) | Jardín Botánico (`21205710`) | 1975–2015 (41 yr) | **1975–1994** (20 yr) | **1995–2015** (21 yr) | Out-of-Sample Split |
| **Precipitation** ($P$) | Aguas Claras (`23215030`) | 1975–2000 (26 yr) | **1975–1987** (13 yr) | **1988–2000** (13 yr) | Out-of-Sample Split |
| **Temperature** ($T_{\max}, T_{\min}$) | Jardín Botánico (`21205710`) | 1990–2020 (31 yr) | **1990–2004** (15 yr) | **2005–2020** (16 yr) | Out-of-Sample Split (Unseen) |
| **Temperature** ($T_{\max}, T_{\min}$) | Aguas Claras (`23215030`) | 1990–2011 (22 yr) | **1990–2000** (11 yr) | **2001–2011** (11 yr) | Out-of-Sample Split (Unseen) |

### 2.3 Sensitivity to the 1979 Satellite Era Boundary
ECMWF ERA5 assimilates modern satellite radiance soundings starting in 1979. Prior to 1979, reanalyses rely primarily on surface observations and sparse radiosonde soundings. In regions of complex tropical topography, this fundamental change in data assimilation can potentially induce artificial trends or baseline shifts in humidity, temperature lapse rates, and convective precipitation.

- **Sensitivity Protocol**: Models will be recalibrated strictly on the post-satellite calibration record (1979–1994 for Jardín Botánico; 1979–1987 for Aguas Claras), excluding the pre-satellite period (1975–1978).
- **Evaluation**: The performance of the post-1979 calibrated model will be compared directly against the full-calibration model on the out-of-sample validation period. If excluding the pre-satellite era significantly improves validation metrics, the impact of assimilation changes on parameter stability will be documented.

### 2.4 Paired Missingness Mask
Ground station observations contain gaps resulting from sensor failures, maintenance, or missing reports. The `DhimeDailyPrecipitationAdapter` and `DhimeDailyTemperatureAdapter` assign a quality flag of `Faltante` or `np.nan` to these days.

- **Mandatory Masking Rule**: Any date where the station observation is missing (`np.nan` or `quality_flag == 'Faltante'`) **must be synchronously masked out** of the co-located ERA5 series prior to computing any metric.
- Unpaired evaluations are strictly prohibited to prevent uneven sample sizes from biasing frequency, percentile, or accumulation estimates.

### 2.5 Generalization Stations (Milestone v0.2.0)
To establish regional generalizability beyond the initial Andean benchmarks, candidate stations across additional natural regions of Colombia will be incorporated under milestone v0.2.0:
1. **Noanamá, Chocó** (Pacífico region).
2. **Arauca, Arauca** (Orinoquia region).
3. **Continental Caribbean Station** (Caribe region lowlands).

These stations will be integrated via formal protocol amendment (v1.1) and evaluated against the identical pre-specified criteria codified herein. **They are a separate milestone and do NOT count toward the operational designation of Section 5.4 for the two benchmark stations of v1.0.** Once added by amendment, operational designation is re-evaluated over the enlarged station set, and the v1.0 result is kept as a separate, unaltered record.

---

## 3. Metric Categorization: Decisive vs. Descriptive

Metrics are divided into **Decisive Metrics**, which govern method promotion, falsification, and progression; and **Descriptive Metrics**, which profile distribution properties without acting as falsification gates.

### 3.1 Sign Convention & Error Definitions
All comparisons adhere to a uniform definition where model bias is defined as:

$$\text{Bias} = \text{Model} - \text{Observed}$$

All error and percent bias comparisons adhere to a uniform sign convention where **positive values strictly denote improvement** (reduction in absolute error or bias):

$$\Delta \text{Error} = |\text{Error}_{\text{baseline}}| - |\text{Error}_{\text{candidate}}|$$

$$\Delta |PBIAS| = |PBIAS_{\text{baseline}}| - |PBIAS_{\text{candidate}}|$$

- $\Delta \text{Error} > 0$: Candidate method outperforms baseline (reduced error).
- $\Delta \text{Error} < 0$: Candidate method degrades performance (increased error).

### 3.2 Precipitation Decisive Metrics & Falsification Thresholds
Evaluated on out-of-sample validation series. A dry day is defined as $P < 1.0\text{ mm}$.

| Metric | Definition | Threshold / Tolerance | Rationale |
| :--- | :--- | :---: | :--- |
| **$f_{\text{wet}}$ (Total)** | Relative bias of total wet-day frequency ($P \ge 1.0\text{ mm}$): $100 \times \frac{f_{\text{mod}} - f_{\text{obs}}}{f_{\text{obs}}}$ | $\|PBIAS(f_{\text{wet}})\| \le 5\%$ | Resolves the ERA5 tropical drizzle inflation using a scale-independent relative bound. |
| **$f_{\text{wet, seasonal}}$** | Maximum seasonal relative bias for DEF, MAM, JJA, SON | $\max_{s} \|PBIAS(f_{\text{wet}, s})\| \le 10\%$ | Preserves the seasonal timing of wet/dry cycles across regimes. |
| **$P_{95}$ (Wet Days)** | Relative bias of the 95th empirical percentile of wet days ($P \ge 1.0\text{ mm}$) | $\|PBIAS(P_{95})\| \le 15\%$ | Evaluates convective heavy rainfall rates. |
| **$P_{99}$ (Wet Days)** | Relative bias of the 99th empirical percentile of wet days ($P \ge 1.0\text{ mm}$) | $\|PBIAS(P_{99})\| \le 25\%$ | Essential heavy-tail gate required to justify Level 3 (EQM) over Level 2. |
| **$CDD_{\max}$** | Annual maximum consecutive dry days ($P < 1.0\text{ mm}$) streak length | $\|PBIAS(CDD_{\max})\| \le 15\%$ | Critical for agricultural drought and hydrological low-flow modeling. |
| **$PBIAS_{\text{monthly}}$** | Percent bias of mean monthly accumulation: $100 \times \frac{\sum P_{\text{mod}} - \sum P_{\text{obs}}}{\sum P_{\text{obs}}}$ | $\|PBIAS\| \le 10\%$ | Ensures regional catchment water yield and conservation. |
| **Volume Outperformance** | Paired difference in absolute volume bias: $\Delta \|PBIAS\| = \|PBIAS_{\text{raw}}\| - \|PBIAS_{\text{corr}}\|$ | $\text{CI}_{95, \text{upper}}(\Delta \|PBIAS\|) \ge 0$ | Method must not significantly degrade total accumulation relative to raw ERA5. |

#### Completeness Rules for $CDD_{\max}$ Calculation
Missing observations can artificially truncate or generate spurious dry spells. To prevent artifacts, $CDD_{\max}$ calculation is subject to strict completeness rules:
1. **Annual Completeness**: A calendar year is evaluated only if it contains $\ge 330$ valid observation days.
2. **Monthly Completeness**: Any calendar month with $> 20\%$ missing days is excluded from streak propagation, and dry streaks crossing that month are marked invalid.

### 3.3 Precipitation Descriptive Metrics (Profiling Only)
Descriptive metrics are reported with 95% bootstrap confidence intervals for diagnostic characterization, without pass/fail thresholds:

1. **$Rx1\text{day}$** (Annual Maximum 1-day Precipitation):
   - Excluded from decisive falsification due to high single-event sampling variance over 13–21 year validation windows.
2. **$Rx5\text{day}$** (Annual Maximum Consecutive 5-day Accumulation):
   - Tracked as a diagnostic indicator of multi-day synoptic systems.
3. **$SDII$** (Simple Daily Intensity Index on wet days: $\sum P_{\text{wet}} / N_{\text{wet}}$):
   - **Mathematical Trade-Off Note**: In two-step threshold matching (Level 2), censoring sub-threshold amounts ($< P_{\text{th}}$) while scaling against total observed precipitation **tends to inflate $SDII$** on wet days ($\ge 1.0\text{ mm}$). This is a recognized mathematical property of trace truncation and is tracked descriptively rather than penalized as a model defect.

### 3.4 Temperature Decisive Metrics & Falsification Thresholds
Evaluated on out-of-sample validation series across $N_{\text{decisive}} = 3$ decisive metrics, subject to an absolute physical consistency disqualifying gate:

| Metric | Definition | Threshold / Tolerance | Rationale |
| :--- | :--- | :---: | :--- |
| **$|\text{Bias}_m|$** | Maximum absolute monthly bias across all calendar months ($m=1\dots12$): $\max\left(\max_m \|\text{Bias}_m(T_{\max})\|, \max_m \|\text{Bias}_m(T_{\min})\|\right)$ | $\le 0.5\,^\circ\text{C}$ | Prevents seasonal phase distortion in degree-day and phenological models. |
| **$DTR$ Bias** | Diurnal Temperature Range ($T_{\max} - T_{\min}$): absolute value of the mean DTR bias, $|\overline{DTR_{\text{mod}} - DTR_{\text{obs}}}|$ (not a daily MAE) | $\|\Delta DTR\| \le 0.8\,^\circ\text{C}$ | Captures boundary-layer thermal inertia and radiative damping. |
| **Extreme Percentiles** | Maximum extreme percentile bias: $\max\left(\|\Delta P_5(T_{\min})\|, \|\Delta P_{95}(T_{\max})\|\right)$ | $\le 0.8\,^\circ\text{C}$ | Evaluates thermal stress risks, frost exposure, and microclimatic extremes. |

#### Aggregation of Extreme Components
Pairwise extremes are aggregated taking the maximum absolute value. Both individual components (e.g., $P_5$ of $T_{\min}$ and $P_{95}$ of $T_{\max}$) and their respective 95% annual bootstrap confidence intervals must satisfy the $\le 0.8\,^\circ\text{C}$ threshold for the metric to achieve Pass.

#### Absolute Disqualifying Gate: Physical Consistency
- **Daily Ordering Constraint**: $T_{\min}(t) \le T_{\max}(t)$ on 100% of valid days.
- **Disqualification Rule**: Physical consistency is an absolute gate outside the $N_{\text{decisive}} = 3$ metric count. It is evaluated on the **final output** of each method, after the universal crossing rule of Section 3.5 has been applied. Any final day with $T_{\min}(t) > T_{\max}(t)$ results in **immediate fatal failure**.

### 3.5 Universal Crossing Rule (All Levels) & Crossing Diagnostics
A correction applied to $T_{\max}$ and $T_{\min}$ separately (Levels 1 and 3) may produce $T_{\min, \text{corr}}(t) > T_{\max, \text{corr}}(t)$. To avoid failing the gate on a technicality, the **same deterministic rule applies to every level (1, 2, and 3)**: if a crossing occurs, both values are set to $T_{\text{mid}} \pm 0.025\,^\circ\text{C}$, where $T_{\text{mid}}$ is the midpoint (formulas in Section 6.2).

- **Mandatory reporting**: for every method, station, and partition, report the number and the **fraction of days** on which the rule was triggered.
- **Distortion alert**: if the fraction exceeds **1%**, the scorecard must carry an explicit alert, because a method that collapses many days to $DTR = 0.05\,^\circ\text{C}$ is distorting the distribution even if it passes the gate. The alert does not change the Pass/Fail mapping but must be reported and discussed.
- **Note**: since the rule guarantees $DTR = 0.05\,^\circ\text{C} > 0$ on triggered days, the gate acts as an implementation check; the crossing fraction is the substantive diagnostic.

---

## 4. Cross-Validation Architecture & Concordance

Evaluation enforces strict architectural separation between hyperparameter selection, primary temporal validation, and decadal robustness testing.

```mermaid
flowchart TD
    subgraph Record["Complete Historical Record"]
        Cal["Calibration Partition<br/>(e.g., 1975–1994)"]
        Val["Out-of-Sample Validation<br/>(e.g., 1995–2015)"]
    end

    subgraph Internal["Hyperparameter Tuning"]
        NestedCV["Nested Block CV by Year<br/>(Strictly Inside Calibration)"]
    end

    subgraph Dual_Track["Dual Evaluation Architecture"]
        Track1["Primary Split Evaluation<br/>(Trained on Cal -> Evaluated on Val)"]
        Track2["Secondary Robustness Track<br/>(5-Year Contiguous Block CV across Full Record)"]
    end

    Cal --> NestedCV
    NestedCV -->|Fitted Parameters| Track1
    Val --> Track1
    Record --> Track2

    Track1 --> Concordance{"Weaker-of-the-Two<br/>Concordance Rule"}
    Track2 --> Concordance

    Concordance -->|Both Pass| Success["Global Success / Promotion"]
    Concordance -->|Any Fail| Fail["Falsified"]
    Concordance -->|Neither Fails, >=1 Inconclusive| Inconclusive["Not Established"]
```

### 4.1 Internal Hyperparameter Tuning (Nested Annual Block CV)
Any method with tunable parameters (such as the Level 3 EQM hyperparameter grids) must optimize those parameters **strictly via nested cross-validation inside the calibration partition**:
- **Partitioning**: Tuning must use **annual block cross-validation (splitting by whole calendar years)**, NOT random $K$-fold day-by-day resampling, to prevent synoptic autocorrelation from leaking across folds.
- **Selection Loss Function**: Hyperparameter combinations are chosen to minimize the multi-objective loss across validation folds within calibration:
  - *Temperature* (distributional loss): $\mathcal{L}_{\text{tuning}} = \sum_{V \in \{T_{\max}, T_{\min}, DTR\}} \frac{1}{19} \sum_{q \in \{5, 10, \dots, 95\}} \left| Q_q(V_{\text{mod}}) - Q_q(V_{\text{obs}}) \right|$, i.e., the mean absolute error between quantiles 5 to 95 (step 5). A daily MAE is deliberately **not** used because it rewards corrections that shrink variability and would bias the grid toward the least dispersive option, against the goal of preserving $DTR$ and the tails.
  - *Precipitation*: $\mathcal{L}_{\text{tuning}} = |PBIAS(f_{\text{wet}})| + |PBIAS(P_{95})| + |PBIAS(P_{99})|$
- Validation partition data must remain strictly unobserved during tuning.

### 4.2 Secondary Robustness Track: 5-Year Contiguous Block CV
In addition to the primary historical split, methods undergo a secondary robustness assessment using **5-year contiguous block cross-validation** across the entire record:
- The historical record is divided into contiguous 5-year blocks starting from the first year of the record (e.g., 1975–1979, 1980–1984, etc.). If fewer than 5 years remain at the end, they are merged into the final block (e.g., 41 years yield 8 blocks, with the final block spanning 6 years).
- In each fold, one 5-year block serves as the test set while the remaining blocks form the training set.
- Contiguous 5-year blocks preserve multi-year climate variability (e.g., El Niño / La Niña episodes) across folds.
- **Fixed hyperparameters**: the hyperparameters are those selected once in Section 4.1 (nested annual block CV inside the calibration partition) and remain **fixed**. In this track the cross-validation only refits the method's *parameters* (factors, thresholds, quantile maps) on each training set; it never re-tunes hyperparameters, so no information from validation-period data enters the selection.

### 4.3 Weaker-of-the-Two Concordance Rule
The overall rating is determined by the concordance between the Primary Split and the 5-Year Block CV:

$$\text{Global Outcome} = \min\left(\text{Outcome}_{\text{Primary Split}}, \text{Outcome}_{\text{5-Year Block CV}}\right)$$

Where ordinal ranks are: $\text{Fail} < \text{Inconclusive} < \text{Pass}$.
- A method achieves **Global Pass** if and only if both the Primary Split and the 5-Year Block CV achieve Pass status.
- If either track produces a Fail on any decisive metric, the global outcome is **Falsified**.
- If neither track produces a Fail but at least one is Inconclusive, the global outcome is **Not Established** (see Section 5.4).

---

## 5. Statistical Uncertainty & Three-Tier Outcome Mapping

Point estimates are evaluated alongside statistical uncertainty to differentiate true corrective skill from stochastic noise.

### 5.1 Annual Block Bootstrap Uncertainty
All metric biases and paired error differences are evaluated using a **1,000-replica block bootstrap with a fixed block length of 1 year (annual blocks)**:
- Resampling whole 1-year blocks preserves interannual climate variability (such as ENSO regimes) and prevents the artificial disruption or mixing of seasonal cycles.
- **Paired Bootstrap for Method Differences**: All paired error differences $\Delta \text{Error} = |\text{Error}_{k-1}| - |\text{Error}_k|$ and $\Delta |PBIAS| = |PBIAS_{\text{raw}}| - |PBIAS_{\text{corr}}|$ are computed via **paired annual block bootstrap**: in each of the 1,000 bootstrap iterations, the exact same resampled calendar years are evaluated for both candidate method and baseline.
- The 30-day block bootstrap is strictly reserved for daily lag analysis (temporal alignment).

### 5.2 Three-Tier Metric Outcomes
For every decisive metric $M$ with threshold $T$:
1. **Pass (Cumple)**:
   - The full 95% annual block bootstrap confidence interval lies entirely within the acceptable zone.
2. **Inconclusive (Inconcluso)**:
   - Any other case: the 95% confidence interval crosses the threshold boundary, **wherever the point estimate falls** (inside or outside the acceptable zone).
3. **Fail (Falla)**:
   - The full 95% confidence interval lies entirely outside the acceptable zone. A point estimate beyond the threshold with an interval that still includes acceptable values is **Inconclusive**, not Fail.

The rule is symmetric (it is the same logic used for $\Delta|PBIAS|$). Consequence: "Not Established" will be more frequent, and "Falsified" requires a violation with confidence.

### 5.3 Global State Mapping & 70% Rounding Rule
Across all $N_{\text{decisive}}$ decisive metrics evaluated:

- **Success / Promoted**:
  - Exactly **0** decisive metrics achieve **Fail**, AND
  - **Temperature only**: zero physical consistency violations ($T_{\min} \le T_{\max}$ on 100% of days); **precipitation only**: $P_{\text{corr}}(t) \ge 0$ and finite on every day, AND
  - **Precipitation only**: volume outperformance holds: $\text{CI}_{95, \text{upper}}(\Delta |PBIAS|) \ge 0$ (no statistically significant volume degradation relative to raw ERA5), AND
  - The number of **Pass** outcomes satisfies the 70% ceiling threshold:
    $$N_{\text{pass}} \ge \left\lceil 0.70 \times N_{\text{decisive}} \right\rceil$$
    - *Precipitation* ($N_{\text{decisive}} = 6$): $\lceil 0.70 \times 6 \rceil = \mathbf{5\text{ passes required}}$.
    - *Temperature* ($N_{\text{decisive}} = 3$): $\lceil 0.70 \times 3 \rceil = \mathbf{3\text{ passes required}}$ (all 3 must pass).
- **Not Established** (no Fail, insufficient evidence):
  - Exactly **0** decisive metrics achieve **Fail**, BUT $N_{\text{pass}} < \lceil 0.70 \times N_{\text{decisive}} \rceil$.
- **Falsified** (at least one Fail):
  - At least **1** decisive metric achieves **Fail**, OR
  - **Precipitation only**: $\text{CI}_{95, \text{upper}}(\Delta |PBIAS|) < 0$ (statistically significant degradation of volume bias relative to raw ERA5), OR
  - Temperature: any final day with $T_{\min} > T_{\max}$; precipitation: any day with $P_{\text{corr}} < 0$ or non-finite.

### 5.4 Operational Aggregation & Final Method Selection Rules
1. **Unit of Evaluation**: Every downscaling method is evaluated and scored independently **per station and per variable**.
2. **Operational Designation Rule**: A downscaling level $k$ is declared **operationally valid** for a given variable if and only if it achieves **Global Success across ALL benchmark stations**. If a level achieves success at one station but fails or is inconclusive at another, it cannot be designated as operational; station-specific performance and failure modes must be published transparently.
3. **Final Selection Rule**: For each variable, the final selected operational method is the **highest promoted level ($k$) that achieves Global Success across all benchmark stations**.
4. **Two distinct non-success outcomes** (to be reported and counted separately):
   - **Falsified**: at least one decisive metric obtained a Fail, i.e., its full 95% interval lies outside the acceptable zone (or a gate was violated). This refutes the method under this protocol, with confidence.
   - **Not Established**: no Fail, but at least one Inconclusive (e.g., confidence intervals too wide for 11–13 validation years). This reflects insufficient statistical power and **does not refute the method**.
   If no level achieves Global Success, the report states, per variable and station, which of the two outcomes occurred; "methodological falsification" is declared only if every non-successful level is Falsified. Otherwise the result is "not established", and Raw ERA5 remains uncorrected as the default meanwhile.

---

## 6. Formal Parsimony Progression & Downscaling Hierarchies

Methods are structured in strict hierarchies of increasing mathematical and parametric complexity.

### 6.1 Precipitation Downscaling Hierarchy
- **Level 0**: Raw ERA5 (Uncalibrated Baseline).
- **Level 1**: Linear Scaling (Monthly multiplicative factor: $P_{\text{corr}}(t) = s_m^{(1)} \cdot P_{\text{raw}}(t)$, with $s_m^{(1)} = \frac{\sum_{t \in \text{cal}, m(t)=m} P_{\text{obs}}(t)}{\sum_{t \in \text{cal}, m(t)=m} P_{\text{raw}}(t)}$).
- **Level 2**: Threshold Matching & Scaling (Empirical drizzle truncation + monthly scaling factor).
- **Level 3**: Non-Parametric Empirical Quantile Mapping (EQM). It **includes the drizzle truncation of Level 2** (same $P_{\text{th}, m}$ and $P_{\text{censored}}$ of Section 7), followed by EQM grouped according to the window option in Section 6.3 (calendar monthly or 31-day moving window) fitted on wet days only (observed days with $P \ge 1.0\text{ mm}$ versus censored model days with $P > 0$), using the hyperparameter grid of Section 6.3. Truncated days remain zero and no additional $s_m$ is applied (the quantile map defines the amounts).
  - **Known Trade-Off Note (Trace Volume & Negative PBIAS)**: Because Level 3 does not apply $s_m$, the volume of observed trace precipitation ($0.1\text{--}0.9\text{ mm}$) is not redistributed onto wet days. Consequently, $PBIAS_{\text{monthly}}$ can be slightly negative by design; this is reported and is evaluated solely against the pre-specified $|PBIAS_{\text{monthly}}| \le 10\%$ threshold.
- **Precipitation gate (all levels)**: $P_{\text{corr}}(t) \ge 0$ and finite on every day. Any violation is a fatal failure and sits outside the $N_{\text{decisive}}$ count.

### 6.2 Temperature Downscaling Hierarchy
The physical elevation adjustment ($-\Gamma \Delta z$) modifies both $T_{\max}$ and $T_{\min}$ equally, identically canceling in the diurnal temperature range:
$$(T_{\max} - \Gamma\Delta z) - (T_{\min} - \Gamma\Delta z) = T_{\max} - T_{\min}$$
Therefore, altitude adjustment is treated as an **optional preprocessing step**, not as an architectural tier. The temperature hierarchy is defined as:

- **Level 0**: Raw ERA5 (Uncalibrated Baseline).
- **Level 1 (Independent Monthly Bias)**:
  Applies independent monthly mean additive bias corrections for $T_{\max}$ and $T_{\min}$:
  $$T_{\max, \text{corr}}(t) = T_{\max, \text{mod}}(t) - \overline{\text{Bias}}_{m(t)}(T_{\max})$$
  $$T_{\min, \text{corr}}(t) = T_{\min, \text{mod}}(t) - \overline{\text{Bias}}_{m(t)}(T_{\min})$$
- **Level 2 (Mean State + DTR Scaling)**:
  Decomposes daily temperature into daily mean temperature ($T_{\text{mean}}$) and diurnal range ($DTR$):
  $$T_{\text{mean, mod}}(t) = \frac{T_{\max, \text{mod}}(t) + T_{\min, \text{mod}}(t)}{2}, \quad DTR_{\text{mod}}(t) = T_{\max, \text{mod}}(t) - T_{\min, \text{mod}}(t)$$
  Corrects $T_{\text{mean}}$ via monthly additive bias and scales $DTR$ via monthly multiplicative factor:
  $$T_{\text{mean, corr}}(t) = T_{\text{mean, mod}}(t) - \overline{\text{Bias}}_{m(t)}(T_{\text{mean}})$$
  $$DTR_{\text{corr}}(t) = s_{m(t)}^{DTR} \cdot DTR_{\text{mod}}(t), \quad \text{where } s_m^{DTR} = \frac{\overline{DTR}_{\text{obs}, m}}{\overline{DTR}_{\text{mod}, m}}$$
  Reconstructs daily extremes:
  $$T_{\max, \text{corr}}(t) = T_{\text{mean, corr}}(t) + \frac{DTR_{\text{corr}}(t)}{2}$$
  $$T_{\min, \text{corr}}(t) = T_{\text{mean, corr}}(t) - \frac{DTR_{\text{corr}}(t)}{2}$$
- **Level 3 (Monthly EQM with Deterministic Physical Reordering)**:
  Applies monthly non-parametric quantile mapping to $T_{\max}$ and $T_{\min}$ independently. As for every level (Section 3.5), if on any day $t$, $T_{\min, \text{corr}}(t) > T_{\max, \text{corr}}(t)$, the crossing is deterministically resolved by setting:
  $$T_{\text{mid}}(t) = \frac{T_{\max, \text{corr}}(t) + T_{\min, \text{corr}}(t)}{2}$$
  $$T_{\max, \text{final}}(t) = T_{\text{mid}}(t) + 0.025\,^\circ\text{C}$$
  $$T_{\min, \text{final}}(t) = T_{\text{mid}}(t) - 0.025\,^\circ\text{C}$$
  preserving daily mean temperature $T_{\text{mid}}$ while establishing $DTR = 0.05\,^\circ\text{C}$. This same rule is applied to Levels 1 and 2 whenever a crossing occurs. The count and fraction of triggered days ($N_{\text{crossings}}$) must be published per level, with the >1% alert of Section 3.5.

### 6.3 Level 3 Hyperparameter Grid (EQM)
Tuned strictly via nested annual block CV within the calibration partition:
1. **Window Definition**: Calendar monthly window vs. 31-day moving window centered on day of year ($t \pm 15$ days).
2. **Quantile Bins**: 50 bins vs. 100 bins (percentiles $1\dots99$).
3. **Extrapolation Scheme**: Constant extrapolation (clamping corrections outside calibration range) vs. linear extrapolation of extreme quantile ratios/deltas.

### 6.4 Progression Rule ($k-1 \to k$) & Known Multiplicity Limitation
A higher level ($k$) is promoted over a lower level ($k-1$) if and only if:
1. **Significant Improvement**: $\text{CI}_{95,\text{lower}}(\Delta \text{Error}) > 0$ for at least one metric in the **Primary Discriminating Family**:
   - *Precipitation (5 metrics)*: Total $f_{\text{wet}}$, Seasonal $f_{\text{wet}}$, Wet-day $P_{95}$, Wet-day $P_{99}$, $CDD_{\max}$.
   - *Temperature (3 metrics)*: $DTR$, $P_5$, $P_{95}$.
   - Total candidate comparisons across variables: **8 comparisons**.
2. **No Degradation**: No decisive metric has $\text{CI}_{95,\text{upper}}(\Delta \text{Error}) < 0$ (no statistically significant degradation).

> **Known Methodological Limitation (Multiplicity Risk)**:  
> Requiring improvement on "at least one" metric across a family of eight candidate decisive comparisons ($f_{\text{wet}}$ total, $f_{\text{wet}}$ seasonal, $P_{95}$, $P_{99}$, $CDD_{\max}$, $DTR$, $P_5$, $P_{95}$) retains a known non-zero risk of **false-positive promotion (Type I error)** due to multiple testing.  
> This standard is accepted in Phase 0 to avoid over-penalizing complex physical trade-offs in tropical terrain. However, it is formally documented as a limitation. In all scorecards, exact bootstrap intervals for all comparisons must be published simultaneously.

---

## 7. Mathematical Formulation of Level 2: Threshold Matching & Scaling

### 7.1 Wet-Day Frequency & Threshold Calculation
1. Observations define days with $P_{\text{obs}}(t) < P_0 = 1.0\text{ mm}$ as dry days.
2. For each calendar month $m \in \{1, \dots, 12\}$, let $f_{\text{obs}, m}^{\text{wet}}$ be the observed wet-day frequency during calibration:
   $$f_{\text{obs}, m}^{\text{wet}} = \frac{1}{N_{\text{obs}, m}} \sum_{t \in \text{cal}, m(t)=m} \mathbb{I}[P_{\text{obs}}(t) \ge P_0]$$
3. In the uncorrected ERA5 calibration series for month $m$, determine the empirical precipitation threshold $P_{\text{th}, m}$ such that the model wet-day frequency equals $f_{\text{obs}, m}^{\text{wet}}$:
   $$\Pr\left(P_{\text{ERA5}}(t) \ge P_{\text{th}, m} \mid m(t) = m, t \in \text{cal}\right) = f_{\text{obs}, m}^{\text{wet}}$$
   $P_{\text{th}, m}$ is computed directly as the $(1 - f_{\text{obs}, m}^{\text{wet}})$-th empirical quantile of the calibration ERA5 series for month $m$.

### 7.2 Trace Drizzle Censoring
All ERA5 amounts below $P_{\text{th}, m}$ are classified as drizzle and truncated to zero:

$$P_{\text{censored}}(t) = \begin{cases} P_{\text{ERA5}}(t), & \text{if } P_{\text{ERA5}}(t) \ge P_{\text{th}, m(t)} \\ 0.0, & \text{if } P_{\text{ERA5}}(t) < P_{\text{th}, m(t)} \end{cases}$$

### 7.3 Monthly Scaling Factor & Corrected Precipitation
To match the total observed water input, a monthly multiplicative scaling factor $s_m$ is calibrated:

$$s_m = \frac{\sum_{t \in \text{cal}, m(t)=m} P_{\text{obs}}(t)}{\sum_{t \in \text{cal}, m(t)=m} P_{\text{censored}}(t)}$$

The final corrected daily precipitation series for any day $t$ (calibration or validation) is:

$$P_{\text{corr}}(t) = s_{m(t)} \cdot P_{\text{censored}}(t)$$

### 7.4 Mass Conservation & SDII Trade-Off Notes
- **Mass Balance by Construction**: During calibration, $\sum_{t \in \text{cal}} P_{\text{corr}}(t) \equiv \sum_{t \in \text{cal}} P_{\text{obs}}(t)$ holds across each month $m$ **by construction**. It is an algebraic identity of the parameter fitting, **not** an emergent physical property or an out-of-sample validation achievement. Out-of-sample mass balance must be verified empirically on the validation partition.
- **SDII Inflation Tendency**: Because trace rainfall ($< 1.0\text{ mm}$) is removed from model dry days while $s_m$ preserves total observed accumulation, $s_m$ concentrates trace water volume onto the remaining wet days ($P \ge P_{\text{th}, m}$). Consequently, the mean wet-day intensity ($SDII$) **tends to be inflated** compared to observations:
  $$SDII(P_{\text{corr}}) > SDII(P_{\text{obs}})$$
  This inflation is a recognized, unavoidable structural trade-off of two-step threshold matching.
- **Level 3 Trace Volume Trade-Off**: Unlike Level 2, Level 3 does not apply $s_m$, so the volume of observed trace precipitation ($0.1\text{--}0.9\text{ mm}$) is not redistributed onto wet days. Consequently, $PBIAS_{\text{monthly}}$ may be slightly negative by design; this known trade-off is reported and only evaluated against the pre-specified $|PBIAS_{\text{monthly}}| \le 10\%$ threshold.

---

## 8. Observational Convention & Lag Audit for Temperature

When extended temperature records (2005–2020) are processed:

1. **Pre-Evaluation Sequencing**:
   The observational audit of extended temperature (2005–2020) **must be executed BEFORE any downscaling method is evaluated**.
2. **Strict Metric Boundary (No Validation Leakage)**:
   The audit must report **strictly lag correlations, cross-correlation curves, and extrema timing (NO bias, MAE, or error metrics)**, ensuring zero informational leakage regarding the validation set's mean state or variance.
3. **Protocol Amendment Gate**:
   A change is defined **only by the lag** (lag 0 versus ±1): if, in 2005–2020, the paired block-bootstrap difference between lag 0 and a lag of ±1 day has a 95% interval that excludes zero **in favor of a lag other than 0**, this protocol must be amended (**v1.1**) *before* any method evaluation. The difference between the 00–24 UTC and 05–05 UTC windows is **non-actionable** (in the 1990–1994 benchmark, the paired difference was small across all series: maximum $|\Delta r| \le 0.0057$ at lag 0 full year, $|\Delta r| \le 0.0112$ across seasonal subsets at lag 0, and $|\Delta r| \le 0.0181$ across all lags and seasons; furthermore, the direction of the difference is not consistent across stations and seasons, as documented in [`docs/guides/era5-extraction.md`](../guides/era5-extraction.md)) unless the paired difference exceeds a practical margin fixed now at $|\Delta r| > 0.05$ with a 95% interval that excludes zero; an apparent flip of the 'optimal' window within that margin does not trigger an amendment.
4. **Prohibition of Causal Speculation**:
   Reports must strictly record empirical correlation values and confidence intervals. Speculative explanations (such as undocumented changes in sensor shelter exposure or observer shifts) are prohibited without official IDEAM documentation.

---

## 9. Implementation Roadmap & Phase Linkage

- **Phase 0: Protocol Pre-Specification & Freezing** (Current):
  - Pre-specify criteria in `docs/science/evaluation-protocol.md`.
  - Validate documentation build (`mkdocs build --strict`).
  - Present for user review before committing or tagging (`protocol-v1`).
- **Phase 1: Verification Engine (`EvaluationSuite`)**:
  - Implement automated metric calculators with paired annual block bootstrap uncertainty.
  - Implement temporal splitters and 5-year contiguous block cross-validators.
  - Enforce paired missingness masking and $CDD_{\max}$ completeness rules.
- **Phase 2: Downscaling Hierarchy Implementation**:
  - Implement Level 1: Linear Scaling ($P$) & Independent Monthly Mean Bias ($T$).
  - Implement Level 2: Threshold Matching ($P$) & Mean State + DTR Scaling ($T$).
  - Implement Level 3: Monthly EQM ($P$) & Deterministic Physical Reordering EQM ($T$) with nested annual block CV.
- **Phase 3: Out-of-Sample Falsification & Reporting**:
  - Execute evaluation across benchmark partitions.
  - Generate Three-Tier Scorecards (Pass / Inconclusive / Fail).
  - Test parsimony progression ($k-1 \to k$) against the Primary Discriminating Family.
  - Publish final benchmark scorecard and falsification analyses.

---

## Appendix C: Implementation Clarifications & Conservative Rule Addition (Protocol v1.1)

This appendix codifies implementation clarifications along with one substantive conservative addition (C.8) adopted at the closure of Phase 1 (Protocol v1.1). All metric definitions, temporal partitions, and parsimony rules established in Protocol v1.0 remain strictly frozen; the addition in C.8 establishes a conservative sample size guard that can only convert potential Pass/Fail outcomes into Inconclusive when annual sample sizes are combinatorially insufficient.

### C.1 Scope and Module Architecture of `EvaluationSuite`
Section 9 refers to an `EvaluationSuite`. In Phase 1, the core verification engine is implemented at module level within `tropicor.core.protocol` (functions `compute_*`, `evaluate_*`, `apply_*`). An orchestrating class structure (e.g., encapsulating the execution pipeline `method -> station -> partition -> scorecard`) will be decided in Phase 2 based on the architectural requirements of the daily correction hierarchy.

### C.2 Literal Streak Completeness for $CDD_{\max}$
To guarantee that consecutive dry days ($P < 1.0\text{ mm}$) reflect continuous, fully observed dry regimes without interpolation or boundary artifacts:
1. **Calendar Year Boundary**: Dry streaks are computed strictly within each individual calendar year (resetting on December 31; no streak crosses into January 1 of the following year).
2. **Strict Discard Rule**: Any candidate streak that touches or contains a missing observation day ($\text{NaN}$), or that touches any day within an excluded calendar month ($> 20\%$ missing days in that month), is **completely discarded**. Streaks are not truncated or fragmented (single rule: any incompletely observed streak is discarded).
3. **Annual Aggregation and Discarded Year Rule**: The annual $CDD_{\max}$ is the maximum length among all valid remaining streaks in the calendar year. If an evaluated calendar year ($\ge 330$ valid observation days) has candidate dry streaks but **all** of them are discarded due to touching missing observations or excluded months, the year's $CDD_{\max}$ is undefined: the year is **excluded** from the annual mean calculation (incrementing `n_discarded_years`), preventing an artificial deflation of the annual mean by spurious zero values. Conversely, if an evaluated calendar year genuinely experiences zero dry days (all valid days have $P \ge 1.0\text{ mm}$), its annual maximum dry streak is strictly $0.0\text{ days}$ and the year is retained.
4. **Annual Completeness Gate**: A calendar year is evaluated only if it contains $\ge 330$ valid observation days; years with $< 330$ days are excluded from the annual mean.
5. **Model-Observation Symmetry**: Identical filtering rules are enforced synchronously on modeled series via the paired missingness mask (Section 2.4).
6. **Empirical Feasibility Check**: A feasibility check using observations only on calibration partitions (`scripts/check_cdd_feasibility.py`) confirmed that the strict streak completeness rules do not cause undue data loss. In Jardín Botánico (1975–1994, 20 years), 4 years had $< 330$ valid days, leaving 16 evaluated years; across 1,008 candidate dry streaks, only 19 (1.88%) were discarded, leaving 989 valid streaks, with 16 of 16 evaluated years retaining valid streaks. In Aguas Claras (1975–1987, 13 years), 5 years had $< 330$ valid days, leaving 8 evaluated years; across 519 candidate dry streaks, only 6 (1.16%) were discarded, leaving 513 valid streaks, with 8 of 8 evaluated years retaining valid streaks.

### C.3 Percentile Evaluation Sample ($P_{95}$ and $P_{99}$)
Precipitation extreme percentiles $P_{95}$ and $P_{99}$ are evaluated strictly on the sub-sample of **wet days** ($P \ge 1.0\text{ mm}$), as defined in separate literal rows of Table 3.2:
- For $P_{95}$: *"Relative bias of the 95th empirical percentile of wet days ($P \ge 1.0\text{ mm}$)"* ($|PBIAS| \le 15\%$).
- For $P_{99}$: *"Relative bias of the 99th empirical percentile of wet days ($P \ge 1.0\text{ mm}$)"* ($|PBIAS| \le 25\%$).

Days with $P < 1.0\text{ mm}$ (dry days) do not enter the empirical percentile sample.

### C.4 Integer Ceiling Arithmetic for 70% Global Pass Threshold
The global qualification threshold requiring $\lceil 0.70 \times N_{\text{decisive}} \rceil$ passes (Section 5.3) is evaluated strictly via exact integer ceiling arithmetic without floating-point representation error:
$$\text{Required Passes} = \lceil 7 \times N_{\text{decisive}} / 10 \rceil = -(-7 \times N_{\text{decisive}} // 10)$$
For precipitation ($N_{\text{decisive}} = 6$), the required number of passes is exactly 5. For temperature ($N_{\text{decisive}} = 3$), the required number of passes is exactly 3.

### C.5 Percentile Interpolation & Seasonal Nomenclature
1. **Empirical Percentiles**: Empirical percentiles ($P_5$, $P_{95}$, $P_{99}$) are calculated using standard linear interpolation (Hyndman and Fan Type 7; default in `numpy.percentile(..., method='linear')`).
2. **Seasonal Notation**: The abbreviation "DEF" in the Section 3.2 table denotes the standard meteorological season December–January–February (DJF).

### C.6 Indefinite Bootstrap Replicas Tracking
Bootstrap replicas resulting in non-finite metric values (e.g., due to absence of wet days or lack of valid dry streaks in a resampled block) are excluded from the empirical confidence interval percentiles (2.5 and 97.5). The verification engine explicitly records and exposes the number of excluded replicas (`n_discarded_replicas`) in diagnostic outputs without introducing arbitrary post-hoc acceptance thresholds.

### C.7 Declaration of Validation Data Integrity
As of the registration of Amendment v1.1:
1. **No Method Execution on Validation Data**: No downscaling or bias correction method (Levels 1–3) has been calibrated, tuned, fitted, or evaluated on the out-of-sample validation partitions (Precipitation: Jardín Botánico 1995–2015, Aguas Claras 1988–2000; Temperature: Jardín Botánico 2005–2020, Aguas Claras 2001–2011).
2. **Prior Exploratory Analyses Declared**: Prior exploratory analyses performed before protocol registration (evaluated exclusively on uncorrected raw ERA5 and observational series, including daily precipitation lag correlations in `scripts/lag_analysis.py`, raw ERA5 spatial neighborhood and daily comparisons in `scripts/compare_era5_ideam_phase4.py` and `scripts/compare_era5_ideam_daily.py`, and 1990–1994 pilot temperature tests in `scripts/lag_analysis_temperature.py`) are formally declared as transparent baseline disclosures in Section 1.2 and Section 8. Furthermore, monthly downscaling models implemented in Sprint 8 (`LinearScaling` and `EmpiricalQuantileMapping` in `tropicor/core/bias.py`) were evaluated exclusively on four separate stations (Puerto Carreño, Las Flores, Noanamá, and Gorgona) across 1990–2019 in `examples/validate_bias_correction.py` and on synthetic series in unit tests; neither benchmark station (Jardín Botánico nor Aguas Claras) was evaluated, resulting in zero overlap with benchmark validation partitions.
3. **Audit Ledger Status**: The append-only validation opening audit ledger (`local/reports/validation_openings.jsonl`) does not exist on this machine as of the date of this amendment (the path is excluded by .gitignore).

### C.8 Minimum Annual Blocks (Substantive Addition, Pre-Validation)
To prevent degenerate bootstrap resampling on short records or heavily masked partitions, the annual block bootstrap uncertainty engine enforces a minimum threshold of annual blocks:
$$\text{MIN\_BOOTSTRAP\_BLOCKS} = 7$$
In the verification engine, a "valid annual block" is defined strictly as follows: for non-CDD metrics, any calendar year containing at least one valid (non-NaN) observation in the synchronous paired series (and for temperature, at least one valid observation for both $T_{\max}$ and $T_{\min}$); for $CDD_{\max}$, any calendar year that satisfies the completeness gate ($\ge 330$ valid observation days) and does not have all candidate dry streaks discarded (i.e., years with an evaluated entry in `annual_obs` and `annual_mod`).

If, following synchronous paired masking (and for $CDD_{\max}$, following completeness exclusions), a series contains fewer than 7 valid annual blocks ($B < 7$):
1. **Confidence Interval Suppression**: The bootstrap confidence interval is not calculated; `ci_lower` and `ci_upper` are assigned `NaN` and flagged with `insufficient_blocks = True`.
2. **Scorecard Outcome**: Any decisive metric with insufficient blocks maps strictly to `MetricOutcome.INCONCLUSIVE` under the symmetric three-tier mapping (Section 5.2). It can never produce a `PASS` or a `FAIL`.
3. **Combinatorial Justification**: The annual block bootstrap resamples $B$ blocks with replacement. Because decisive metrics evaluate exchangeable annual or aggregated properties independent of the chronological sequence order of the annual blocks, the number of distinct resampled multisets of size $B$ drawn from $B$ available blocks is given by combinations with replacement:
$$\binom{2B - 1}{B}$$
Evaluating this combinatorial capacity across small values of $B$:
- For $B = 4$: $\binom{7}{4} = 35$ distinct combinations.
- For $B = 5$: $\binom{9}{5} = 126$ distinct combinations.
- For $B = 6$: $\binom{11}{6} = 462$ distinct combinations.
- For $B = 7$: $\binom{13}{7} = 1,716$ distinct combinations.

$B = 7$ is the lowest integer number of blocks where the combinatorial space of distinct resamples strictly exceeds the pre-specified number of bootstrap replicas ($1,000$ replicas; Section 5.1). Below 7 blocks, 1,000 bootstrap draws necessarily suffer from extreme duplicate allocations and severe pseudo-replication.
4. **Directionality & Conservatism**: This rule is strictly conservative; it can only convert potential Pass or Fail outcomes into Inconclusive due to sample size insufficiency, and cannot manufacture an artificial Pass.

---

## Appendix D: Phase 2a Daily Precipitation Downscaling Clarifications & Amendments (Protocol v1.2)

This appendix codifies the implementation clarifications and substantive design decisions adopted during Phase 2a (Daily Precipitation Downscaling Hierarchy: Levels 1–3) prior to any evaluation on out-of-sample validation data.

### D.1 Units of Daily Precipitation Series
Precipitation observations and model inputs to downscaling estimators must be expressed strictly in millimeters ($\text{mm}$). To prevent silent numerical corruption from unscaled reanalysis inputs stored natively in meters ($\text{m}$), estimators raise an explicit `ValueError` in `fit()` if the mean across valid days of either input series is below $0.01\text{ mm}$.

### D.2 Synchronous Paired Masking and NaN Preservation
1. **Calibration Masking**: Model fitting across Levels 1–3 operates strictly over the subset of days where both station observations and co-located reanalysis values are simultaneously valid (synchronous pairwise deletion of missing records).
2. **Transform NaN Propagation**: In `transform()`, missing values ($\text{NaN}$) in the input model series are strictly preserved in the output series. An input $\text{NaN}$ is never converted to a dry day ($0.0\text{ mm}$).
3. **Zero Paired Observations per Month**: In Level 1 and Level 2 fitting, if any calendar month $m \in \{1, \dots, 12\}$ contains zero valid paired observation days in the calibration record, `fit()` raises an explicit `ValueError`. (For Level 3, the requirement of at least 2 wet days per group is governed by D.13).

### D.3 Drizzle Threshold Estimation ($P_{\text{th}, m}$) via Type 7 Linear Interpolation
For each calendar month $m$, the drizzle threshold $P_{\text{th}, m}$ (Section 7.1) is estimated as the $(1 - f_{\text{obs}, m}^{\text{wet}})$-th empirical quantile of calibration reanalysis daily precipitation using Hyndman and Fan Type 7 linear interpolation (`numpy.percentile(..., method='linear')`), evaluated strictly on paired calibration days where $f_{\text{obs}, m}^{\text{wet}} = \frac{1}{N_m} \sum \mathbb{I}[P_{\text{obs}} \ge 1.0\text{ mm}]$. When a calendar month contains zero observed wet days ($N_{\text{obs, wet}, m} = 0$, so $f_{\text{obs}, m}^{\text{wet}} = 0.0$), the evaluation probability is $q = 1.0 - 0.0 = 1.0$, and the estimator sets $P_{\text{th}, m}$ to the maximum calibration model precipitation in that month ($\max(P_{\text{mod}, m})$, the 100th empirical percentile), not infinity ($\infty$).

### D.4 Literal Linear Scaling (Level 1)
In Level 1 Linear Scaling (Section 6.1), the monthly multiplicative factor $s_m^{(1)} = \sum P_{\text{obs}, m} / \sum P_{\text{raw}, m}$ is applied uniformly to all model daily values ($P_{\text{corr}} = s_m^{(1)} \cdot P_{\text{raw}}$). No drizzle truncation or thresholding is applied in Level 1; the persistence of excess wet-day frequency is a documented structural characteristic of Level 1. If monthly model accumulation $\sum P_{\text{raw}, m}$ is zero while observed accumulation $\sum P_{\text{obs}, m}$ is positive, `fit()` raises an explicit `ValueError`. If observed accumulation is zero, $s_m^{(1)} = 0.0$.

### D.5 Level 2 Scaling Without Second Truncation and Zero-Sum Rules
1. **No Second Truncation**: Following drizzle truncation at $P_{\text{th}, m}$, surviving precipitation is multiplied by the mass-conserving monthly factor $s_m = \sum P_{\text{obs}, m} / \sum P_{\text{censored}, m}$. No secondary truncation threshold is applied to $s_m \cdot P_{\text{censored}}$; values where $s_m \cdot P_{\text{censored}} < 1.0\text{ mm}$ remain valid positive precipitation amounts to preserve the exact algebraic mass balance $\sum P_{\text{corr}} \equiv \sum P_{\text{obs}}$ (Section 7.4).
2. **Zero-Sum Rules**: If the monthly model censored accumulation $\sum P_{\text{censored}, m}$ is zero while observed accumulation $\sum P_{\text{obs}, m}$ is positive, `fit()` raises an explicit `ValueError`. If observed accumulation is zero, $s_m = 0.0$.

### D.6 Level 3 Empirical Quantile Mapping Nodes
Quantile evaluation probabilities for Level 3 transfer nodes (Section 6.3) are defined symmetrically:
- **100 Bins**: 99 interior quantile nodes corresponding to probabilities $0.01, 0.02, \dots, 0.99$ (`numpy.linspace(0.01, 0.99, 99)`).
- **50 Bins**: 49 interior quantile nodes corresponding to probabilities $0.02, 0.04, \dots, 0.98$ (`numpy.linspace(0.02, 0.98, 49)`).

### D.7 Marginal Empirical Samples for Wet-Day Distributions
In Level 3 fitting, the observed wet-day sample ($P_{\text{obs}} \ge 1.0\text{ mm}$) and the model wet-day sample ($P_{\text{censored}} > 0.0\text{ mm}$) within each grouping temporal slice are drawn as marginal empirical distributions from the synchronously valid calibration paired record. They are not conditioned day-by-day on each other, reflecting the independent marginal cumulative distribution functions of precipitation amounts.

### D.8 Piecewise Linear Interpolation of Transfer Function
Within the span of calibration quantile nodes ($Q_{\text{mod}, 1} \le P \le Q_{\text{mod}, \text{top}}$), Level 3 directly interpolates observed quantiles as a piecewise linear function of model quantiles:
$$P_{\text{corr}} = \text{interp}(P, Q_{\text{mod}}, Q_{\text{obs}})$$
The estimator does not interpolate the ratio $r$ to subsequently multiply by $P$. This guarantees strict linearity between adjacent quantile nodes and eliminates non-linear multiplicative artifacts within the calibration domain.

### D.9 Constant Extrapolation Outside Calibration Quantile Nodes
When `extrapolation="constant"`, model precipitation exceeding the top calibration quantile node ($P > Q_{\text{mod}, \text{top}}$) is scaled by the constant ratio observed at the top node:
$$P_{\text{corr}} = r_{\text{top}} \cdot P, \quad \text{where } r_{\text{top}} = \frac{Q_{\text{obs}, \text{top}}}{Q_{\text{mod}, \text{top}}}$$

### D.10 Linear Extrapolation as a Substantive Design Decision (Transfer Function Extension)
A literal reading of Section 6.3 ("linear extrapolation of extreme quantile ratios") entails extrapolating the ratio $r(P) = r_{\text{top}} + \text{slope}_r \cdot (P - Q_{\text{mod}, \text{top}})$. Calibration diagnostic evidence revealed that when empirical ratios decrease across upper quantiles ($\text{slope}_r < 0$), extending the ratio produced negative ratios for extreme reanalysis events, erroneously truncating intense storm events to $0.0\text{ mm}$ upon hitting the physical zero floor. Under the literal reading, calibration cross-validation would have selected a linear configuration in one of the two benchmarks; the change was adopted for physical admissibility (avoiding negative ratios), not for fit, and with the adopted definition cross-validation selects constant extrapolation at both stations. Furthermore, transfer function extension does not bound the corrected magnitude: in calibration diagnostics the corrected maxima exceed the observed maximum; the choice between configurations is governed by the criterion of D.15.

To prevent this pathological destruction of extreme rainfall, Protocol v1.2 adopts a substantive design decision defining linear extrapolation as the linear extension of the transfer function itself:
$$P_{\text{corr}} = Q_{\text{obs}, \text{top}} + s \cdot (P - Q_{\text{mod}, \text{top}})$$
$$\text{with slope } s = \frac{Q_{\text{obs}, \text{top}} - Q_{\text{obs}, \text{top}-1}}{Q_{\text{mod}, \text{top}} - Q_{\text{mod}, \text{top}-1}}$$
If the denominator $(Q_{\text{mod}, \text{top}} - Q_{\text{mod}, \text{top}-1}) \le 10^{-9}$, the estimator falls back to the constant ratio $r_{\text{top}} \cdot P$. Because empirical quantiles are monotonically non-decreasing, the transfer slope satisfies $s \ge 0$, mathematically ensuring physically non-negative corrections and eliminating artificial truncation of extreme events.

### D.11 Transformation of Values Below the First Quantile Node
Model wet values below the first calibration quantile node ($P_{\text{th}, m} \le P < Q_{\text{mod}, 1}$) are scaled using the ratio of the first quantile node:
$$P_{\text{corr}} = \frac{Q_{\text{obs}, 1}}{Q_{\text{mod}, 1}} \cdot P = r_1 \cdot P$$
This formulation is equivalent to linear interpolation between $(0, 0)$ and $(Q_{\text{mod}, 1}, Q_{\text{obs}, 1})$. Because values below $P_{\text{th}, m}$ are set to $0.0\text{ mm}$, the corrected series is discontinuous at $P_{\text{th}, m}$. Corrected values in the interval $[P_{\text{th}, m}, Q_{\text{mod}, 1})$ can fall below $1.0\text{ mm}$.

### D.12 Physical Non-Negativity Safeguard (Zero Floor)
All estimators enforce a strict physical floor $P_{\text{corr}} = \max(0.0, P_{\text{corr}})$. Level 3 maintains a diagnostic counter `n_clipped_below_zero` tracking any numerical instance clamped by this safeguard. Under the monotonic transfer extension defined in D.10, this counter evaluates strictly to 0 during extrapolation.

### D.13 Minimum Sample Size per Group for Empirical Quantile Mapping
Level 3 estimators require at least 2 wet days in both the observed series ($P \ge 1.0\text{ mm}$) and the censored model series ($P > 0.0\text{ mm}$) within every calibration group (calendar month or 31-day window). If any group contains fewer than 2 wet days, `fit()` raises an explicit `ValueError`.

### D.14 Reference Leap Calendar (366 Days) for Circular Moving Windows
When `window="moving_window_31"`, dates map to an integer day of year $D \in [1, 366]$ within a reference leap calendar (cumulative month offsets: Jan=0, Feb=31, Mar=60, $\dots$, Dec=335). Non-leap years omit day 60 (Feb 29). Circular distance along the 366-day cycle is $\Delta(D_1, D_2) = \min(|D_1 - D_2|, 366 - |D_1 - D_2|)$. The 31-day window centered on January 1 ($D=1$) comprises $\Delta \le 15$, spanning December 17 ($D=352$) through January 16 ($D=16$) across 31 continuous calendar days.

### D.15 Evaluation of Cross-Validation Tuning Loss on Concatenated Out-of-Sample Series
Nested annual block cross-validation evaluates the multi-objective tuning loss $\mathcal{L}_{\text{tuning}}$ (Section 4.1) once on the concatenated out-of-sample predictions spanning all validation folds:
$$\mathcal{L}_{\text{tuning}}^{\text{overall}} = \text{precipitation\_tuning\_loss}(obs_{\text{cal}}, \bigcup_{k=1}^K \hat{y}_k)$$
This provides a statistically stable multi-year empirical sample for extreme percentiles ($P_{95}$ and $P_{99}$), while individual fold losses are recorded as diagnostics of interannual variability.

### D.16 Cross-Validation Folds Restricted Strictly to Valid Calendar Years
Cross-validation folds are constructed strictly for calendar years containing at least one valid synchronously paired observation. Calendar years lacking valid observed records (e.g., station out of service) are excluded from fold partitioning.

### D.17 Deterministic Parsimony Tie-Breaker with Numerical Tolerance
Configurations whose overall tuning loss is within $10^{-6}$ (absolute) of the minimum are tied; non-finite losses are ineligible. Ties are resolved strictly in favor of parsimony:
$$\text{monthly} \succ \text{moving\_window\_31} \quad \longrightarrow \quad 50\text{ bins} \succ 100\text{ bins} \quad \longrightarrow \quad \text{constant} \succ \text{linear}$$

### D.18 Calibration Data Loading Semantics and Immediate Validation Discarding
To enforce zero data leakage, calibration data ingestion (`load_calibration_precipitation_pair`) trims the observed and model series to the pre-specified calibration partition immediately after reading, before any statistic is computed or any estimator is fitted. Aggregation of ERA5 hourly files to daily values may span all available years, but only calibration-year values are retained.

### D.19 Declaration of Validation Partition Integrity
As of the registration of Protocol v1.2, no validation partition has been accessed. The validation ledger `local/reports/validation_openings.jsonl` does not exist on this machine (it is strictly excluded by `.gitignore`), and all fitting, hyperparameter tuning, and diagnostic evaluations of Phase 2a used exclusively calibration partitions via `load_calibration_precipitation_pair`.



