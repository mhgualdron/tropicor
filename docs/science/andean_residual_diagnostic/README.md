# Diagnostic of the Andean Residual (5 Opened Stations)

## 1. Purpose & Declared Steps
This directory archives the residual diagnostic conducted across the five already-opened Andean stations (Berlin, Mogotes, Matecaña, Olaya Herrera, Obonuco). The objective is to describe, against a sampling-noise floor, the calibration-to-validation level change and the seasonal structure of the Level 1 residual.

The four analysis steps declared in the task:
1. **Annual Series & Trend**: Annual raw bias d_y = mean over months of the monthly mean of (ERA5 - observation); r_y = the same after subtracting the calibration monthly bias of each month (Level 1) for all calendar years in the union of calibration and validation years with $\ge 10$ valid days/month and $\ge 9$ valid months/year. Linear trend of $r_y$ in K/decade with 95% bootstrap CI (1000 resamples of years, seed 42).
2. **Level Change**: Calibration-to-validation shift $\bar{d}_{\text{val}} - \bar{d}_{\text{cal}}$ with 95% bootstrap CI (1000 resamples of calendar years independently inside each partition, seed 42).
3. **Sampling-Noise Floor of Level 1**: 999 random draws (seed 42) partitioning the available years into pseudo-calibration and pseudo-validation of the same sizes as real partitions. Fit Level 1 on pseudo-cal, evaluate on pseudo-val, and compute empirical distribution (median, P5, P95) of $\max_m |\text{Bias}_m|$ and percentile rank of the real value.
4. **Offset versus Seasonal Amplitude**: Decomposition of Level 1 monthly validation biases into $\text{offset} = \text{mean}(b_m)$ and $\text{seasonal amplitude} = \max_m |b_m - \text{offset}|$ for real outcomes and across the 999 noise-floor draws.

## 2. Allowed & Forbidden Data
- **Allowed Data**: Observations and ERA5 2m temperature series of the five opened Andean stations (37015020, 24025040, 26135040, 27015070, 52045010) with the exact registry year lists.
- **Forbidden Data**: Observations of benchmark stations 21205710 (JB) and 23215030 (AC); validation ledger `local/reports/validation_openings.jsonl`.

## 3. Input Data & Hash Verification
All input files were verified prior to analysis against `tropicor/data/phase3_expected_inputs.json` and archived scorecards in `docs/science/appendix_g/phase3_andes/`:
- **Registry & Calibration Reports**: `andean_station_registry.json`, `lag_audit_andean_calibration.json`, and 5 calibration report JSONs matched reference hashes.
- **Observations**: 10 observation CSV files (5 Tmax, 5 Tmin) matched reference hashes.
- **ERA5 NetCDFs**: All 113 ERA5 2m temperature yearly NetCDF files (1990–2021) matched reference hashes.
- **Result**: 100% hash match (Self-Check 2: PASS).

## 4. Self-Check Results

### Level 1 Monthly Validation Bias Reproduction (Self-Check 1)
Maximum absolute difference across 12 monthly validation biases between recomputed Level 1 and archived Phase 3 scorecards:

| Station ID | Station Name | Max Absolute Diff Tmax | Max Absolute Diff Tmin | Status ($\le 1\times 10^{-6}$) |
| :--- | :--- | :--- | :--- | :--- |
| `37015020` | BERLIN [37015020] | 0.00e+00 K | 0.00e+00 K | PASS |
| `24025040` | ESCUELA AGRICOLA MOGOTES [24025040] | 0.00e+00 K | 0.00e+00 K | PASS |
| `26135040` | AEROPUERTO MATECAÑA [26135040] | 0.00e+00 K | 0.00e+00 K | PASS |
| `27015070` | AEROPUERTO OLAYA HERRERA [27015070] | 0.00e+00 K | 0.00e+00 K | PASS |
| `52045010` | OBONUCO [52045010] | 0.00e+00 K | 0.00e+00 K | PASS |

### Valid Paired Days Count
Comparison of valid paired days in the validation partition against archived scorecards:

| Station ID | Station Name | Val Days Tmax (Calc) | Val Days Tmax (Arch) | Val Days Tmin (Calc) | Val Days Tmin (Arch) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `37015020` | Berlin | 3624 | 3624 | 3540 | 3540 | PASS |
| `24025040` | Escuela Agricola Mogotes | 3633 | 3633 | 3602 | 3602 | PASS |
| `26135040` | Aeropuerto Matecaña | 4379 | 4379 | 4380 | 4380 | PASS |
| `27015070` | Aeropuerto Olaya Herrera | 5104 | 5104 | 5098 | 5098 | PASS |
| `52045010` | Obonuco | 3516 | 3516 | 3547 | 3547 | PASS |

## 5. Diagnostic Results Summary Table
Summary of Steps 2–5 across all 5 stations and variables:

| Station ID | Station Name | Var | $r_y$ Trend (K/dec) | Trend 95% CI | Level Shift (K) | Shift 95% CI | Real Max \|$B_m$\| (K) | NF Med (K) | NF P95 (K) | Real Rank | Real Offset (K) | Real Seas (K) | NF Med \|Off\| | NF Med Seas |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `37015020` | BERLIN [37015020] | Tmax | +0.0080 | [-0.1544, +0.1965] | +0.1626 | [-0.1624, +0.4710] | 0.7644 | 0.5085 | 0.8774 | 89.69% | +0.1644 | 0.5999 | 0.1049 | 0.4325 |
| `37015020` | BERLIN [37015020] | Tmin | -0.0340 | [-0.1757, +0.1337] | -0.0308 | [-0.2887, +0.2268] | 0.5388 | 0.7013 | 1.2110 | 25.43% | -0.0287 | 0.5101 | 0.0970 | 0.6232 |
| `24025040` | ESCUELA AGRICOLA MOGOTES [24025040] | Tmax | -0.1612 | [-0.3961, +0.2039] | -0.1451 | [-0.4453, +0.1428] | 0.4349 | 0.4541 | 0.6784 | 44.24% | -0.1467 | 0.3282 | 0.1262 | 0.3539 |
| `24025040` | ESCUELA AGRICOLA MOGOTES [24025040] | Tmin | -0.2101 | [-0.4465, -0.0102] | -0.2354 | [-0.5791, +0.1458] | 0.7831 | 0.6056 | 0.9418 | 81.58% | -0.2341 | 0.6077 | 0.1355 | 0.5115 |
| `26135040` | AEROPUERTO MATECAÑA [26135040] | Tmax | +0.1760 | [-0.0028, +0.3711] | +0.2532 | [-0.0060, +0.5046] | 0.4251 | 0.3965 | 0.7120 | 57.06% | +0.2535 | 0.3298 | 0.1024 | 0.3113 |
| `26135040` | AEROPUERTO MATECAÑA [26135040] | Tmin | -0.0032 | [-0.1257, +0.1734] | +0.1331 | [-0.0633, +0.3366] | 0.3411 | 0.2967 | 0.4681 | 67.17% | +0.1289 | 0.2121 | 0.0701 | 0.2420 |
| `27015070` | AEROPUERTO OLAYA HERRERA [27015070] | Tmax | +0.4333 | [+0.3312, +0.5264] | +0.5121 | [+0.2337, +0.7963] | 0.7774 | 0.3616 | 0.6382 | 99.30% | +0.5108 | 0.3265 | 0.1190 | 0.2771 |
| `27015070` | AEROPUERTO OLAYA HERRERA [27015070] | Tmin | +0.2653 | [+0.1497, +0.4018] | +0.5450 | [+0.3657, +0.7210] | 0.7824 | 0.3048 | 0.4932 | 100.00% | +0.5460 | 0.3141 | 0.0934 | 0.2320 |
| `52045010` | OBONUCO [52045010] | Tmax | +0.2307 | [-0.0877, +0.4495] | +0.1706 | [-0.2247, +0.5165] | 0.4350 | 0.4762 | 0.8188 | 39.84% | +0.1535 | 0.2992 | 0.1246 | 0.3737 |
| `52045010` | OBONUCO [52045010] | Tmin | +0.5659 | [+0.3987, +0.6825] | +0.6907 | [+0.3981, +0.9902] | 0.9117 | 0.3971 | 0.7004 | 100.00% | +0.6955 | 0.2162 | 0.1605 | 0.2544 |

NF = sampling-noise floor from the 999 draws; Real Rank = percentile rank of the real value within the noise-floor distribution; Off = offset; Seas = seasonal amplitude.

## 6. Archived File Manifest

| File Path | Size (bytes) | SHA-256 Hash |
| :--- | :--- | :--- |
| `scripts/andean_residual_diagnostic.py` | 33452 | `4f70fb38d5285c48b0387f4fe506888ff701089923257203b4ec4d3dbabed5c7` |
| `results/diagnostic_37015020.csv` | 4322 | `9be665b8481bfb8f038b03d4a7d58f4c8bbfdde91b070dc8a65904fd5bd84334` |
| `results/diagnostic_24025040.csv` | 4885 | `87195299f4b572cbeb95750b18e34452994505907084991d4055e0ba5ac96491` |
| `results/diagnostic_26135040.csv` | 5655 | `05c9d865728e1389a19b06d6af07d3f59fc6e8684293d7f6f038b03d223c15b7` |
| `results/diagnostic_27015070.csv` | 6755 | `426b0d64669e34410c3b10d644deea257028dbd3f58910190d6c9ae12f918d3b` |
| `results/diagnostic_52045010.csv` | 4185 | `b0e4c6d4dfc12e1a3ba59ded102783f1333a530f8bf9fe567c2cf968d160942f` |
| `results/summary.json` | 79017 | `a166d7505f981f0967335987bc05ec4c91032db6faabd294e731bbcfa241977d` |

## 7. Exposure Declaration
- Observations and ERA5 2m temperature series of the five opened Andean stations (37015020, 24025040, 26135040, 27015070, 52045010) were used.
- No observation of benchmark stations 21205710 (JB) or 23215030 (AC) was read.
- The validation ledger was not accessed or modified.

## 8. Accepted Methodological Deviations
Three deviations from the initial written plan were explicitly declared and accepted by the author:
1. **Independent Resampling Inside Partitions**: Resampling of calendar years was performed independently within calibration and validation partitions instead of "paired" resampling, because calibration and validation years are disjoint sets of years.
2. **Restricted Sizes Noise-Floor Variant**: Reported once because pseudo-calibration and pseudo-validation partition sizes are identical to the actual partition sizes by construction.
3. **Phase 3 Data Preparation**: Data preparation and daily aggregation logic was reproduced directly from source without importing `tropicor.core.phase3_evaluator`, as verified by exact equality in Self-Check 1.
