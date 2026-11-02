# Archive: Residual Diagnostics Round 2 (Calibration-Only JB and AC)

## 1. Purpose and Groups of Outputs

This archive records the scripts, configurations, and results of the second round of residual diagnostics conducted prior to finalizing the Phase 3 protocol amendment. All contents are descriptive only.

The outputs are organized into four groups:
1. **Andean Opened Stations Without Year 2020**: Re-evaluation of the five opened Andean stations (37015020, 24025040, 26135040, 27015070, 52045010) excluding calendar year 2020 from the validation partition, computing real residual metrics, empirical sampling noise floor, percentile rank, and Level 1 level-change intervals at 1,000 bootstrap replicas.
2. **Calibration-Only Annual Series for Benchmark Stations**: Annual bias and Level 1 residual series for Jardin Botanico (21205710, 1990–2004) and Aguas Claras (23215030, 1990–2000) using calibration years only, together with OLS trends and 1,000-replica bootstrap confidence intervals.
3. **Gap Closure (Bootstrap Replica Sensitivity)**: Comparison of Level 1 level-change 95% confidence intervals at 10,000 replicas against 1,000 replicas, evaluated under both the round 1 draw order and the round 2 draw order.
4. **Calibration Diagnostics for JB and AC**: Valid-year inventory under the Appendix J.7 criterion versus the diagnostic inclusion criterion (minimum 10 paired days per month, minimum 9 valid months per year); JB sampling noise floor recomputed strictly over included calibration years; step-change versus linear trend model comparison on calibration Level 1 residuals; and diurnal temperature range (DTR) annual series.

## 2. Allowed and Forbidden Data

- **Allowed Data**:
  - Five opened Andean stations (37015020, 24025040, 26135040, 27015070, 52045010): Daily observation CSVs and ERA5 hourly NetCDF files across the full registry year ranges (calibration and validation partitions).
  - Benchmark stations JB (21205710) and AC (23215030): Daily observation CSVs loaded via the official calibration loader (`load_calibration_temperature_pair`) and strictly sliced to calibration years only at load time (JB: 1990–2004; AC: 1990–2000).
- **Forbidden Data and Operations**:
  - No value from a validation year of JB or AC was used in any computation, printed or inspected. The official calibration loader reads the whole station observation file, including rows of validation years, and validates dates, numeric values and duplicate dates over all rows before trimming to the calibration years (tropicor/downscale/temperature.py, lines 1143, 1167–1208, 1220–1225). The loader raised no error.
  - The validation openings ledger (`local/reports/validation_openings.jsonl`) was not accessed.
  - The Phase 3 evaluator (`tropicor.core.phase3_evaluator`) and benchmark lag audit scripts were not executed.

## 3. Inputs

The input observation CSV files and ERA5 hourly NetCDF files read during the analyses and their SHA-256 hashes, as recorded in the summary JSON files:

### Observation Files (JB and AC)
Sourced from `calibration_jb_ac_summary.json` (`input_hashes`):

| Station | Variable | Path | Key in summary | SHA-256 |
| :--- | :--- | :--- | :--- | :--- |
| JB (21205710) | Tmax | `local/era5_ts/21205710/TempMaxima_Jardin_Botanico.csv` | `obs_tmax_21205710` | `d5c5a1e78205299f1022071974653e47d5846b12f5e4de5bec13de0f1439936c` |
| JB (21205710) | Tmin | `local/era5_ts/21205710/TempMinima_Jardin_Botanico.csv` | `obs_tmin_21205710` | `41e6a8df65e5282c33d60c4a3a0907d0b0a034d98982fe4faef99ed3f9b550f1` |
| AC (23215030) | Tmax | `local/era5_ts/23215030/TempMaxima_Aguas_Claras.csv` | `obs_tmax_23215030` | `46a44e5b2f45068a5fafcd37688f21c2279aa70392ee79f682258953686c7056` |
| AC (23215030) | Tmin | `local/era5_ts/23215030/TempMinima_Aguas_Claras.csv` | `obs_tmin_23215030` | `5e0c3d22d63868e63e6e581557dfc03376c7da26ece5e8baa13631e1544d9ea3` |

### ERA5 Hourly NetCDF Files Read

1. **Opened Andean Stations**: In `summary.json` (`self_check_2_inputs.manifest`), all ERA5 NetCDF files for the 5 Andean stations across their registry years were verified with SHA-256 matching disk.
2. **Benchmark Stations (JB and AC)**:
   - In `summary.json`, ERA5 NetCDF files for the calibration years were verified (15 files for JB, 1990–2004; 11 files for AC, 1990–2000).
   - In `calibration_jb_ac_summary.json` (`input_hashes`), all files matching `era5_2m_temperature_*.nc` present in the station directories were hashed (33 files for JB, 1990–2022; 23 files for AC, 1990–2012).
   - Reason why an ERA5 file of a year after the calibration partition appears in `calibration_jb_ac_summary.json`: **not stated in the scripts** (in `scripts/diagnostic_calibration_jb_ac.py` lines 266–269, the hashing loop iterated over `glob.glob(str(cfg["era5_dir"] / "era5_2m_temperature_*.nc"))` before passing `cfg["era5_dir"]` to `load_calibration_temperature_pair`). The loader selects ERA5 files whose name contains a calibration year or the year after the end of the calibration partition (tropicor/downscale/temperature.py, lines 1244–1249) and slices the model series to the calibration years immediately after aggregation (lines 1327–1330). ERA5 files of later years were hashed by the diagnostic script but were not loaded by the loader.

<details><summary>Click to expand complete input hashes table for JB and AC from calibration_jb_ac_summary.json</summary>

| Key | SHA-256 |
| :--- | :--- |
| `era5_21205710_era5_2m_temperature_1990.nc` | `e8cc8b407fd4958b1437f3510ffe10e1046b4c9e29bb0c70654ffad55aa8bb5d` |
| `era5_21205710_era5_2m_temperature_1991.nc` | `dd754918702f14d424a48b69afcf365b368010244a13eb77449de5948e35c06b` |
| `era5_21205710_era5_2m_temperature_1992.nc` | `f16c128dc20a1f24a3bc94b66ddd0fbd99f217dea638f269de117074850f78a6` |
| `era5_21205710_era5_2m_temperature_1993.nc` | `7cc9221a5edc35504d123cb08a8b3d309c0e620d4ca9ed181e4ec34033c780e6` |
| `era5_21205710_era5_2m_temperature_1994.nc` | `862e2c774cdd8e05e4628675b1e573bf676bbade2fe744fe823752578c51631c` |
| `era5_21205710_era5_2m_temperature_1995.nc` | `6ca08e0f27826c598b7ccfc59ee6678615ab53f18a41ff1096c38325d2d11b1e` |
| `era5_21205710_era5_2m_temperature_1996.nc` | `ee00d4ff788149d5b44ca30288ec5c145b870159b8e796b6d3ba0c59734ffe37` |
| `era5_21205710_era5_2m_temperature_1997.nc` | `d172d27c97e5a1de336082bd7e17bcb97d150aa07e9b93aa7a6f5924f67c76e9` |
| `era5_21205710_era5_2m_temperature_1998.nc` | `f7e2fceda7b28c8116e81fc1becabc5f6ee59c5a2ea381c7da1adcf75f087d02` |
| `era5_21205710_era5_2m_temperature_1999.nc` | `9907e138d4fbd91242fbfbb29b4b8ebacba18e9115eebd4407697318c9409b15` |
| `era5_21205710_era5_2m_temperature_2000.nc` | `8e359142381ce6c082b57572903f1a5f2413809e31fd78fd65a648f4f8705f68` |
| `era5_21205710_era5_2m_temperature_2001.nc` | `8604aa5369eefc124856eaef963c3f05468b31634abb2125aa73d488998200a5` |
| `era5_21205710_era5_2m_temperature_2002.nc` | `7d0c2548cdf4a57ef7b43d8c9eb61086198ff0407092aed26a41bec57606699e` |
| `era5_21205710_era5_2m_temperature_2003.nc` | `b3b9f08623ed56dac9a86e888ea7fc3e486766ad53902d521c64a475000e2fb2` |
| `era5_21205710_era5_2m_temperature_2004.nc` | `b0a62069d2a1b36917c9c5ce3ab0ddbb1a4f1b966dac07173e2e2af290f362bb` |
| `era5_21205710_era5_2m_temperature_2005.nc` | `1945f77a803651fea87f66b461bf05060cb1ad8fb2a69582950203e03c0145a2` |
| `era5_21205710_era5_2m_temperature_2006.nc` | `1d16cc3ec4094085e8c44abe9d87f0b19f4f83db153c722f7b860c1a36765d1c` |
| `era5_21205710_era5_2m_temperature_2007.nc` | `421257f17326e87e4b32466df345543cf0a7579515b1430ea013ce0ddb4d3ea0` |
| `era5_21205710_era5_2m_temperature_2008.nc` | `6bb8ea6273f3cd6040eb312f1609340319cf5ecc7d9ea604505a54179effbc46` |
| `era5_21205710_era5_2m_temperature_2009.nc` | `da5cd1474db26e065e0f66141f4c28eda6eac71bae4d9bcdb216f2f0111b7388` |
| `era5_21205710_era5_2m_temperature_2010.nc` | `e9bbb23362b1ef6fad3d323cfddf6a4b5ee47ef7cd8e88a66a1d9ebc7f4d23e3` |
| `era5_21205710_era5_2m_temperature_2011.nc` | `d7e33f28c91e73cb90890212570fa1c6f45ec93f6faf4d9c692cc87e4ec47e2b` |
| `era5_21205710_era5_2m_temperature_2012.nc` | `03019926bb58a04817d15d7915087508ce7a9fc33453a1cf697bb9403d7cd40d` |
| `era5_21205710_era5_2m_temperature_2013.nc` | `3ea066d2e951752c48e5501be904417be82eea3038a65aa6752ffcbd30538cd2` |
| `era5_21205710_era5_2m_temperature_2014.nc` | `77428424088a3f7cf377c3bfae0669a4e0bb6233564df98b36dfd6ad1f16199a` |
| `era5_21205710_era5_2m_temperature_2015.nc` | `46adeed91e424b9be3bf84554d2345ae14f439784eaec64a3e6f8294bcfad411` |
| `era5_21205710_era5_2m_temperature_2016.nc` | `d5d743e9249d1accfc9ece10c0182978d5646fedc6767445828b25653977543b` |
| `era5_21205710_era5_2m_temperature_2017.nc` | `d57ed4783c3efb78b59c1f911d9c22d026c0e1e96d7e266e9053d2d149d0b70d` |
| `era5_21205710_era5_2m_temperature_2018.nc` | `87c42bb35bc232b688c79863b821d8debe8ac32c7f4b71c63b7dae13cfec7c5d` |
| `era5_21205710_era5_2m_temperature_2019.nc` | `8d1804d66cb26664ff2c3c066b80a2db2de5318b86e1111c39f5b06a04262168` |
| `era5_21205710_era5_2m_temperature_2020.nc` | `273c75ff390ef6e0867b8bbcacefff42da3c32fc1616c10ca9face90a395c91a` |
| `era5_21205710_era5_2m_temperature_2021.nc` | `4a93679d5dd9f136676dbfdd29f7c0f1c10e5b79c80fbd8655deecbf52c1a0b4` |
| `era5_21205710_era5_2m_temperature_2022.nc` | `efdace9e99c85430bc68a40214c4f979080020170f82e2f1893e08d581a9fd4e` |
| `era5_23215030_era5_2m_temperature_1990.nc` | `99d828aefc1ffd3beeb142b7001985f80868ad633b169dced1ffcf53428a2f69` |
| `era5_23215030_era5_2m_temperature_1991.nc` | `6726124789ea7ecd250a0b4168cfcbc941e543ad488678b05dcc47cd90907cfa` |
| `era5_23215030_era5_2m_temperature_1992.nc` | `704548f7345041c74935f42c3f2e5684a6f55e3bf9c758e6ea6779e079ebb1f5` |
| `era5_23215030_era5_2m_temperature_1993.nc` | `54355d1ba30ba741838c6635d05c64caff20f0f4c8119600f4cafc8d0bff0286` |
| `era5_23215030_era5_2m_temperature_1994.nc` | `45de2d13c38797ddc84d6f2eefcdf5e22c4d4872ea62e24998fa16ad7144d599` |
| `era5_23215030_era5_2m_temperature_1995.nc` | `9cad476157236aab7c3f713188e765cf130fbec9833da51f8b81256b88208486` |
| `era5_23215030_era5_2m_temperature_1996.nc` | `5db59a8aa067a0503c981a716d2fc8be9958e881f4b90f524a4392c2809882d5` |
| `era5_23215030_era5_2m_temperature_1997.nc` | `16c15f75565b94ace895aff4b35dfe2ed4739fe59af3a2f45efc5c6f2be2c417` |
| `era5_23215030_era5_2m_temperature_1998.nc` | `cdaa120593c457dc69b1bd8e452e995bd46d07d58322c016004ea350157b9e1f` |
| `era5_23215030_era5_2m_temperature_1999.nc` | `9b7604a8d996da73b76f03f8a14b307bc068ef80ca6659b476f74e7787dbebcb` |
| `era5_23215030_era5_2m_temperature_2000.nc` | `8b78a4051f71e25469365a8975f1b6b3bf1e8c7d9af198b187243158dc989ce8` |
| `era5_23215030_era5_2m_temperature_2001.nc` | `299385204ecdf9ab9dd1cb8f3c7140a0b632c240a80f29d2bcd0c6b645f0fa06` |
| `era5_23215030_era5_2m_temperature_2002.nc` | `9e9af23f7547810fafd3fb815647373ab5f96e6c2102177388c5ba2fa38b4ed8` |
| `era5_23215030_era5_2m_temperature_2003.nc` | `751b68807129be318ae28adc9fbf03d3a14f15ef952cea6b383f5c59103768f2` |
| `era5_23215030_era5_2m_temperature_2004.nc` | `a1e7a08184011a19d84a5364ae0f867dea4cd5f1b9d894dd6c149a323155ea46` |
| `era5_23215030_era5_2m_temperature_2005.nc` | `8e328a832bb03e27791ff7cd0d34d45be0974303e25aadc4f521c3793e6914e0` |
| `era5_23215030_era5_2m_temperature_2006.nc` | `66ec224a3d05cb00c92a90fe0698ef9c3d0c617d5ee27e91c80463f666df2d49` |
| `era5_23215030_era5_2m_temperature_2007.nc` | `b99d5b21df7b4c0820941496c34e137763c6c3b42bd6014668866310162560b2` |
| `era5_23215030_era5_2m_temperature_2008.nc` | `d1bc7f77dc88937c56e2819a380fec1a5fd16b15d1c34018f55de29bd7d0cdfc` |
| `era5_23215030_era5_2m_temperature_2009.nc` | `c32b0bd439378d6f01f6b40bd639722bb195e48130168865710929bf02f3c24d` |
| `era5_23215030_era5_2m_temperature_2010.nc` | `fb709aa076279290b0953c96482f0c1c0382d5130309345d54d0b162d15f7e11` |
| `era5_23215030_era5_2m_temperature_2011.nc` | `1caab4327e31baf5fc6bfa37ddaec6129ff712ad5e822bb9033d9a835edc3573` |
| `era5_23215030_era5_2m_temperature_2012.nc` | `abc623c40f733226f71b66f15fa995ae45e20f492a0764ae719fcc8f1dc493ec` |
| `obs_tmax_21205710` | `d5c5a1e78205299f1022071974653e47d5846b12f5e4de5bec13de0f1439936c` |
| `obs_tmax_23215030` | `46a44e5b2f45068a5fafcd37688f21c2279aa70392ee79f682258953686c7056` |
| `obs_tmin_21205710` | `41e6a8df65e5282c33d60c4a3a0907d0b0a034d98982fe4faef99ed3f9b550f1` |
| `obs_tmin_23215030` | `5e0c3d22d63868e63e6e581557dfc03376c7da26ece5e8baa13631e1544d9ea3` |

</details>

## 4. Self-Check Results

1. **Input Hash Verification (`self_check_2_inputs`)**: All inputs verified against expected SHA-256 hashes (`status: PASS`).
2. **Level 1 Monthly Bias Reproduction (`self_check_1_level1_reproduction`)**: Re-derived Level 1 calibration monthly additive biases for all five opened Andean stations matched archived Phase 3 calibration reports with maximum absolute difference of `0.0 K` for both Tmax and Tmin across all stations:
   - Station `37015020` (BERLIN [37015020]): Tmax max diff = `0.0 K`, Tmin max diff = `0.0 K`, Pass: `True`
   - Station `24025040` (ESCUELA AGRICOLA MOGOTES [24025040]): Tmax max diff = `0.0 K`, Tmin max diff = `0.0 K`, Pass: `True`
   - Station `26135040` (AEROPUERTO MATECAÑA [26135040]): Tmax max diff = `0.0 K`, Tmin max diff = `0.0 K`, Pass: `True`
   - Station `27015070` (AEROPUERTO OLAYA HERRERA [27015070]): Tmax max diff = `0.0 K`, Tmin max diff = `0.0 K`, Pass: `True`
   - Station `52045010` (OBONUCO [52045010]): Tmax max diff = `0.0 K`, Tmin max diff = `0.0 K`, Pass: `True`

3. **Appendix J.4 Scalar Reproduction for Benchmark Stations**: Reproduced scalar raw bias against `raw_bias` in `docs/science/appendix_j/calibration/benchmark_calibration_<id>.json`:
   - Key used: `raw_bias` (`["raw_bias"]["tmax"]` and `["raw_bias"]["tmin"]`).
   - Station `21205710` (Jardin Botanico): Tmax scalar raw bias = `-1.7125 K` (expected `-1.7125 K`, MATCH); Tmin scalar raw bias = `0.6281 K` (expected `0.6281 K`, MATCH); Level 1 calibration loss = `1.1128 K`.
   - Station `23215030` (Aguas Claras): Tmax scalar raw bias = `-5.2872 K` (expected `-5.2872 K`, MATCH); Tmin scalar raw bias = `-3.1274 K` (expected `-3.1274 K`, MATCH); Level 1 calibration loss = `0.5941 K`.

## Definitions and Conventions

The definitions, formulas, and conventions implemented across the archived diagnostic scripts:

1. **Sign Convention of Raw Bias**: Evaluated as **ERA5 − observation** (model minus observation).
   - `scripts/andean_residual_diagnostic_round2.py`: line 351 (`delta_m = float(mod_s[m_mask][v].mean() - obs_s[m_mask][v].mean())`) and line 910 (`scalar_tx_bias = float(np.round((mod_tx_cal - obs_tx_cal).mean(), 4))`).
   - `scripts/diagnostic_gap_closure.py`: line 105 (`delta_m = float(mod_s[m_mask][v].mean() - obs_s[m_mask][v].mean())`).
2. **Annual Metrics $d_y$ and $r_y$**:
   - `scripts/andean_residual_diagnostic_round2.py` lines 351–358 and `scripts/diagnostic_gap_closure.py` lines 105–112:
     - Monthly raw bias $\Delta_m$: mean daily model value minus mean daily observation value for calendar month $m$ (`delta_m = float(mod_s[m_mask][v].mean() - obs_s[m_mask][v].mean())`).
     - Monthly Level 1 residual: $\Delta_m - b_{\text{ref}}[m]$ (`m_resids.append(delta_m - b_ref_dict[m])`), where $b_{\text{ref}}[m]$ is the calibration monthly additive bias.
     - Annual raw bias $d_y$: unweighted mean of $\Delta_m$ across all valid calendar months in year $y$ (`d_y = float(np.mean(m_deltas)) if is_included else np.nan`).
     - Annual residual $r_y$: unweighted mean of $(\Delta_m - b_{\text{ref}}[m])$ across all valid calendar months in year $y$ (`r_y = float(np.mean(m_resids)) if is_included else np.nan`).
3. **Month and Year Inclusion Thresholds**:
   - Month inclusion threshold:
     - `scripts/andean_residual_diagnostic_round2.py` line 55 (`MIN_DAYS_PER_MONTH = 10`) and line 350 (`if v.sum() >= MIN_DAYS_PER_MONTH:`).
     - `scripts/diagnostic_calibration_jb_ac.py` line 52 (`MIN_DAYS_PER_MONTH = 10`) and lines 316, 318, 320, 584.
     - `scripts/diagnostic_gap_closure.py` line 37 (`MIN_DAYS_PER_MONTH = 10`) and line 104.
     A calendar month is valid if it contains $\ge 10$ paired days where both observation and ERA5 are valid.
   - Year inclusion threshold:
     - `scripts/andean_residual_diagnostic_round2.py` line 56 (`MIN_MONTHS_PER_YEAR = 9`) and line 356 (`is_included = (n_valid_m >= MIN_MONTHS_PER_YEAR)`).
     - `scripts/diagnostic_calibration_jb_ac.py` line 53 (`MIN_MONTHS_PER_YEAR = 9`) and lines 323–325, 587 (`inc = bool(n_months >= MIN_MONTHS_PER_YEAR)`).
     - `scripts/diagnostic_gap_closure.py` line 38 (`MIN_MONTHS_PER_YEAR = 6`) and line 110 (`is_included = (n_valid_m >= MIN_MONTHS_PER_YEAR)`).
     A calendar year is included in annual statistics if it contains at least the required number of valid months ($\ge 9$ in diagnostic 2 and calibration scripts; $\ge 6$ in the gap closure comparison script).
   - Appendix J.7 valid-year rule: `scripts/diagnostic_calibration_jb_ac.py` lines 308–311 (`j7_valid = bool(n_4way >= 1)`): a calendar year is valid under J.7 if it contains at least 1 paired day with all four components (obs Tmax, obs Tmin, ERA5 Tmax, ERA5 Tmin) valid.

**Note on the year-inclusion threshold of diagnostic_gap_closure.py**:

`scripts/diagnostic_gap_closure.py` uses `MIN_MONTHS_PER_YEAR = 6` (line 38) while `scripts/andean_residual_diagnostic_round2.py` (line 56), `scripts/diagnostic_calibration_jb_ac.py` (line 53) and the round 1 script in `docs/science/andean_residual_diagnostic/scripts/andean_residual_diagnostic.py` (line 58) use 9 (`MIN_MONTHS_PER_YEAR = 9`). The gap-closure script produced `bootstrap_ci_10k_comparison.csv` and `gap_closure_summary.json`. The table below compares the number of calibration and validation years used per station and variable across `gap_closure_summary.json` and `summary.json`:

| Station ID | Station Name | Variable | Calibration Years (`gap_closure_summary.json`) | Validation Years (`gap_closure_summary.json`) | Calibration Years (`summary.json`) | Validation Years (`summary.json`, with 2020) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `37015020` | BERLIN [37015020] | Tmax | 11 | 10 | 11 | 10 |
| `37015020` | BERLIN [37015020] | Tmin | 11 | 10 | 11 | 10 |
| `24025040` | ESCUELA AGRICOLA MOGOTES [24025040] | Tmax | 10 | 10 | 10 | 10 |
| `24025040` | ESCUELA AGRICOLA MOGOTES [24025040] | Tmin | 10 | 10 | 10 | 10 |
| `26135040` | AEROPUERTO MATECAÑA [26135040] | Tmax | 12 | 12 | 12 | 12 |
| `26135040` | AEROPUERTO MATECAÑA [26135040] | Tmin | 12 | 12 | 12 | 12 |
| `27015070` | AEROPUERTO OLAYA HERRERA [27015070] | Tmax | 14 | 14 | 14 | 14 |
| `27015070` | AEROPUERTO OLAYA HERRERA [27015070] | Tmin | 14 | 14 | 14 | 14 |
| `52045010` | OBONUCO [52045010] | Tmax | 10 | 10 | 10 | 10 |
| `52045010` | OBONUCO [52045010] | Tmin | 10 | 10 | 10 | 10 |

The counts coincide in all 10 combinations. Since the set of years with at least 9 valid months is contained in the set with at least 6, equal counts imply identical sets of years.
4. **Level 1**: Calibration monthly additive bias adjustment, fitting $b_m = \text{mean}(\text{mod}) - \text{mean}(\text{obs})$ for calendar months $m=1,\dots,12$ on calibration data and subtracting $b_m$ from the model series (`scripts/andean_residual_diagnostic_round2.py` lines 245–257, `compute_calendar_monthly_biases`).
5. **Level Change**: Compares the **validation partition** against the **calibration partition**, computed as the mean annual raw bias in validation minus the mean annual raw bias in calibration:
   - `scripts/andean_residual_diagnostic_round2.py` lines 675–677 (`cal_dy_w = df_inc_w[df_inc_w["partition"] == "calibration"]["raw_bias_d_y"].values.astype(float)`, `val_dy_w = df_inc_w[df_inc_w["partition"] == "validation"]["raw_bias_d_y"].values.astype(float)`, `delta_lvl_w = float(np.mean(val_dy_w) - np.mean(cal_dy_w))`).
   - `scripts/diagnostic_gap_closure.py` lines 267–269 (`point_delta = float(np.mean(val_dy) - np.mean(cal_dy))`).
6. **The Bootstrap**: What is resampled, replica count, and seed:
   - Level-change bootstrap: `scripts/andean_residual_diagnostic_round2.py` lines 680–683 resamples annual $d_y$ values independently with replacement within the validation partition (`rng.choice(val_dy, len(val_dy), replace=True)`) and within the calibration partition (`rng.choice(cal_dy, len(cal_dy), replace=True)`).
   - Trend bootstrap: `scripts/andean_residual_diagnostic_round2.py` lines 375–384 resamples annual block pairs $(y, r_y)$ or $(y, d_y)$ with replacement across included years (`rng.choice(len(years), len(years), replace=True)`).
   - Number of replicas: 1,000 replicas in `scripts/andean_residual_diagnostic_round2.py` (line 52, `BOOTSTRAP_REPLICAS = 1000`); 10,000 replicas in `scripts/diagnostic_gap_closure.py` (line 35, `BOOTSTRAP_REPLICAS = 10000`).
   - Random seed: 42 across all scripts (line 54 in `andean_residual_diagnostic_round2.py`, line 36 in `diagnostic_gap_closure.py`, line 50 in `diagnostic_calibration_jb_ac.py`).
7. **Noise Floor ('Floor', p5/p95, 'Rank (%)')**:
   - `scripts/andean_residual_diagnostic_round2.py` lines 262–335 (`run_sampling_noise_simulation`): in each pseudo-realization, included years are partitioned into pseudo-calibration ($k_{\text{cal}}$) and pseudo-validation ($k_{\text{val}}$) sets. Level 1 biases fit on pseudo-calibration are subtracted from pseudo-validation data.
   - `Floor`: distribution of the resulting pseudo-validation maximum absolute monthly residual bias `draw_max_b = float(np.max(np.abs(b_pval_arr)))` (line 321). Reported as median, p5 (5th percentile), and p95 (95th percentile).
   - `Rank (%)`: `float(np.mean(np.array(nf_max_b) <= real_max_b) * 100.0)` (line 707), the percentile rank of the real validation maximum monthly bias against the empirical noise floor distribution.
8. **'Abs Off' and 'Seas Amp'**:
   - `scripts/andean_residual_diagnostic_round2.py` lines 322–327:
     - Mean offset of pseudo-validation monthly residuals: `draw_offset = float(np.mean(b_pval_arr))` (line 322).
     - Seasonal amplitude: `draw_seas_amp = float(np.max(np.abs(b_pval_arr - draw_offset)))` (line 323), the maximum absolute deviation of pseudo-validation monthly residuals from their mean offset.
     - `Abs Off`: absolute value of the mean offset (`abs(draw_offset)`, line 326).
     - `Seas Amp`: seasonal amplitude `draw_seas_amp` (line 327).
9. **Column 'Prev 15yr Max Bias' in Table 5.4**: Sourced from `results/andean_and_calibration_annual/summary.json` under key `part_b_benchmark_calibration_sampling_noise["21205710"]["variables"][var][cfg]["max_monthly_bias_k"]` (nominal 15-year run), rounded to 4 decimals (`scripts/diagnostic_calibration_jb_ac.py` lines 384–394).
10. **'Break Year' in Table 5.6**: The **last year of the left segment**, where the left segment comprises years $\le \text{break\_year}$ (`values[:k]`) and the right segment comprises years $> \text{break\_year}$ (`values[k:]`):
    - `scripts/diagnostic_calibration_jb_ac.py` lines 188–195 (`for k in range(3, n - 2): y_b = int(years[k - 1]); g1, g2 = values[:k], values[k:]; ... best_break = y_b`).
    - Lines 230–231 (`"break_year": best_break`, `"left_partition_end_year": best_break`) and line 516 (`Break after {s['break_year']} (<= {s['break_year']} [n={s['n_left']}] vs > {s['break_year']} [n={s['n_right']}])`).
11. **SSE Reductions and Permutation Rank in Table 5.6**:
    - `Linear SSE Red.` = `sse_0 - sse_lin` (`scripts/diagnostic_calibration_jb_ac.py` line 227): reduction in sum of squared errors from linear OLS relative to the constant model.
    - `Step SSE Red.` = `sse_0 - best_sse` (line 238): reduction in SSE from the best step model relative to the constant model.
    - `Step Red. (%)` = `(red_step / sse_0) * 100.0` (line 239): percentage reduction in SSE achieved by the best step model relative to constant SSE.
    - `Perm. Rank (%)` = `float(np.mean(np.array(p_reds) <= red_step) * 100.0)` (line 215): percentile rank of the step SSE reduction relative to 999 random permutations of the annual series.
    - `SSE Diff (Lin - Step)` = `sse_lin - best_sse` (line 243): difference between linear SSE and step SSE (positive indicates step model achieves lower SSE).
12. **DTR Definition and Sign**: Daily diurnal temperature range is daily Tmax minus daily Tmin (`obs_tx - obs_tn` and `mod_tx - mod_tn`). Daily DTR bias sign convention is **ERA5 DTR minus observation DTR** (`dtr_mod - dtr_obs`):
    - `scripts/diagnostic_calibration_jb_ac.py` lines 566–569 (`v4 = obs_tx.notna() & mod_tx.notna() & obs_tn.notna() & mod_tn.notna()`, `dtr_obs = obs_tx - obs_tn`, `dtr_mod = mod_tx - mod_tn`, `dtr_bias_daily = dtr_mod - dtr_obs`).
    - Evaluated only on days where all four components are valid (`v4`). Annual DTR bias $d_y$ is the unweighted mean across valid months ($\ge 10$ paired 4-way days) of monthly mean daily DTR biases (lines 581–588).

## 5. Results Tables Generated from Archived Files

### Source Script by Table

| Results Table | Table Title | Generating Script | Primary Output File(s) |
| :--- | :--- | :--- | :--- |
| **5.1** | Andean Opened Stations: With and Without Year 2020 | `scripts/andean_residual_diagnostic_round2.py` | `results/andean_and_calibration_annual/summary.json` (`part_a_andean_opened_stations`) |
| **5.2** | Four Level-Change Interval Sets Side by Side | `scripts/diagnostic_gap_closure.py` | `results/gap_closure/bootstrap_ci_10k_comparison.csv`, `results/gap_closure/gap_closure_summary.json` (`item4_bootstrap_comparison`) |
| **5.3** | JB and AC Calibration Valid-Year Inventory | `scripts/diagnostic_calibration_jb_ac.py` | `results/jb_ac_calibration/jb_ac_valid_years_inventory.csv` |
| **5.4** | JB Noise Floor Recomputed on Included Years | `scripts/diagnostic_calibration_jb_ac.py` | `results/jb_noise_floor_included_years.csv` (Prev 15yr Max Bias column from `summary.json`, produced by `scripts/andean_residual_diagnostic_round2.py`) |
| **5.5** | AC Noise Floor as Stored | `scripts/andean_residual_diagnostic_round2.py` / `scripts/diagnostic_calibration_jb_ac.py` | `results/jb_ac_calibration/calibration_jb_ac_summary.json` (`item2_ac_noise_floor_unchanged`), `results/andean_and_calibration_annual/summary.json` (`part_b_benchmark_calibration_sampling_noise`) |
| **5.6** | Step versus Linear Residual Models (Calibration Years) | `scripts/diagnostic_calibration_jb_ac.py` | `results/jb_ac_calibration/jb_ac_step_vs_linear.csv` |
| **5.7** | DTR Annual Series (Calibration Years) | `scripts/diagnostic_calibration_jb_ac.py` | `results/jb_ac_calibration/jb_ac_dtr_annual_series.csv` |
| **5.8** | DTR Calibration Summary: Level, Slope, and Best-Step Results | `scripts/diagnostic_calibration_jb_ac.py` | `results/jb_ac_calibration/calibration_jb_ac_summary.json` (`item4_dtr`) |

**Sources for the Four Interval Columns of Table 5.2**:
- **Round 1 (1k rep) 95% CI**: Computed by `scripts/diagnostic_gap_closure.py` (lines 271–278, `ci_archived_1k`) with 1,000 bootstrap replicas (seed 42), sampling calibration before validation (Round 1 draw order), evaluated on annual raw biases $d_y$ loaded from `diagnostic_{st}.csv` (produced by round 1 script `docs/science/andean_residual_diagnostic/scripts/andean_residual_diagnostic.py`).
- **Round 2 (1k rep) 95% CI**: Computed by `scripts/diagnostic_gap_closure.py` (lines 280–287, `ci_diagnostic2_1k`) with 1,000 bootstrap replicas (seed 42), sampling validation before calibration (Round 2 draw order), evaluated on annual raw biases $d_y$ loaded from `diagnostic_{st}.csv` (matches Table 5.1 values produced by `scripts/andean_residual_diagnostic_round2.py`).
- **Round 1 (10k rep) 95% CI**: Computed by `scripts/diagnostic_gap_closure.py` (lines 289–296, `ci_archived_10k`) with 10,000 bootstrap replicas (seed 42), sampling calibration before validation.
- **Round 2 (10k rep) 95% CI**: Computed by `scripts/diagnostic_gap_closure.py` (lines 298–305, `ci_diagnostic2_10k`) with 10,000 bootstrap replicas (seed 42), sampling validation before calibration.

### 5.1 Andean Opened Stations: With and Without Year 2020
Data sourced from `summary.json` (`part_a_andean_opened_stations`):

| Station ID | Station Name | Var | Config | Real Max Bias (K) | Floor Median (K) | Floor [p5, p95] (K) | Rank (%) | Level Change (K) | Level Change 95% CI (K) | Trend (K/dec) | Trend 95% CI (K/dec) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `37015020` | BERLIN [37015020] | Tmax | With 2020 | 0.7644 | 0.5085 | [0.2973, 0.8774] | 89.69% | 0.1626 | [-0.1569, 0.4613] | 0.0080 | [-0.1544, 0.1965] |
| `37015020` | BERLIN [37015020] | Tmax | Without 2020 | 0.8489 | 0.5018 | [0.3088, 0.8200] | 95.90% | 0.2412 | [-0.0454, 0.5087] | 0.0600 | [-0.1037, 0.2294] |
| `37015020` | BERLIN [37015020] | Tmin | With 2020 | 0.5388 | 0.7013 | [0.3530, 1.2110] | 25.43% | -0.0308 | [-0.2864, 0.2057] | -0.0340 | [-0.1757, 0.1337] |
| `37015020` | BERLIN [37015020] | Tmin | Without 2020 | 0.5478 | 0.7352 | [0.3844, 1.2708] | 21.72% | -0.0277 | [-0.2873, 0.2505] | -0.0342 | [-0.1755, 0.1568] |
| `24025040` | ESCUELA AGRICOLA MOGOTES [24025040] | Tmax | With 2020 | 0.4349 | 0.4541 | [0.2746, 0.6784] | 44.24% | -0.1451 | [-0.4941, 0.1483] | -0.1612 | [-0.3961, 0.2039] |
| `24025040` | ESCUELA AGRICOLA MOGOTES [24025040] | Tmax | Without 2020 | 0.3269 | 0.4221 | [0.2640, 0.6186] | 16.32% | -0.0276 | [-0.2377, 0.1892] | 0.0156 | [-0.1756, 0.2645] |
| `24025040` | ESCUELA AGRICOLA MOGOTES [24025040] | Tmin | With 2020 | 0.7831 | 0.6056 | [0.3544, 0.9418] | 81.58% | -0.2354 | [-0.5922, 0.1330] | -0.2101 | [-0.4465, -0.0102] |
| `24025040` | ESCUELA AGRICOLA MOGOTES [24025040] | Tmin | Without 2020 | 0.7245 | 0.6386 | [0.3739, 0.9617] | 67.57% | -0.1936 | [-0.5679, 0.2213] | -0.1878 | [-0.5231, 0.0898] |
| `26135040` | AEROPUERTO MATECAÑA [26135040] | Tmax | With 2020 | 0.4251 | 0.3965 | [0.2228, 0.7120] | 57.06% | 0.2532 | [0.0130, 0.4995] | 0.1760 | [-0.0028, 0.3711] |
| `26135040` | AEROPUERTO MATECAÑA [26135040] | Tmax | Without 2020 | 0.4251 | 0.3965 | [0.2228, 0.7120] | 57.06% | 0.2532 | [0.0130, 0.4995] | 0.1760 | [-0.0028, 0.3711] |
| `26135040` | AEROPUERTO MATECAÑA [26135040] | Tmin | With 2020 | 0.3411 | 0.2967 | [0.1808, 0.4681] | 67.17% | 0.1331 | [-0.0539, 0.3163] | -0.0032 | [-0.1257, 0.1734] |
| `26135040` | AEROPUERTO MATECAÑA [26135040] | Tmin | Without 2020 | 0.3411 | 0.2967 | [0.1808, 0.4681] | 67.17% | 0.1331 | [-0.0539, 0.3163] | -0.0032 | [-0.1257, 0.1734] |
| `27015070` | AEROPUERTO OLAYA HERRERA [27015070] | Tmax | With 2020 | 0.7774 | 0.3616 | [0.2181, 0.6382] | 99.30% | 0.5121 | [0.2130, 0.7806] | 0.4333 | [0.3312, 0.5264] |
| `27015070` | AEROPUERTO OLAYA HERRERA [27015070] | Tmax | Without 2020 | 0.7774 | 0.3616 | [0.2181, 0.6382] | 99.30% | 0.5121 | [0.2130, 0.7806] | 0.4333 | [0.3312, 0.5264] |
| `27015070` | AEROPUERTO OLAYA HERRERA [27015070] | Tmin | With 2020 | 0.7824 | 0.3048 | [0.1833, 0.4932] | 100.00% | 0.5450 | [0.3824, 0.7057] | 0.2653 | [0.1497, 0.4018] |
| `27015070` | AEROPUERTO OLAYA HERRERA [27015070] | Tmin | Without 2020 | 0.7824 | 0.3048 | [0.1833, 0.4932] | 100.00% | 0.5450 | [0.3824, 0.7057] | 0.2653 | [0.1497, 0.4018] |
| `52045010` | OBONUCO [52045010] | Tmax | With 2020 | 0.4350 | 0.4762 | [0.2859, 0.8188] | 39.84% | 0.1706 | [-0.2135, 0.5320] | 0.2307 | [-0.0877, 0.4495] |
| `52045010` | OBONUCO [52045010] | Tmax | Without 2020 | 0.3053 | 0.4465 | [0.2558, 0.7217] | 13.41% | 0.0637 | [-0.2898, 0.3896] | 0.0924 | [-0.2748, 0.3551] |
| `52045010` | OBONUCO [52045010] | Tmin | With 2020 | 0.9117 | 0.3971 | [0.2134, 0.7004] | 100.00% | 0.6907 | [0.4079, 1.0416] | 0.5659 | [0.3987, 0.6825] |
| `52045010` | OBONUCO [52045010] | Tmin | Without 2020 | 0.8063 | 0.3716 | [0.2113, 0.6318] | 99.70% | 0.5751 | [0.3547, 0.8152] | 0.4962 | [0.3426, 0.6301] |

**Table 5.1 Notes**:
- The level-change and trend intervals are computed at 1,000 bootstrap replicas. For station `26135040` Tmax, the level-change 95% interval at 1,000 replicas is `[+0.0130, +0.4995]` (excludes 0), whereas at 10,000 replicas it is `[-0.0116, +0.5063]` (Round 1 draw order) or `[-0.0043, +0.5073]` (Round 2 draw order), both including 0 (see Table 5.2).
- For stations `26135040` and `27015070`, the 'With 2020' and 'Without 2020' rows are identical because calendar year 2020 is not present in their validation partition (`val_has_2020: false` in `summary.json` under `part_a_andean_opened_stations[<station_id>]["val_has_2020"]`; validation years end in 2017 and 2019, respectively).

### 5.2 Four Level-Change Interval Sets Side by Side
Data sourced from `bootstrap_ci_10k_comparison.csv`:

| Station ID | Station Name | Var | Point Change (K) | Round 1 (1k rep) 95% CI | Round 2 (1k rep) 95% CI | Round 1 (10k rep) 95% CI | Round 2 (10k rep) 95% CI |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `37015020` | BERLIN [37015020] | Tmax | 0.1626 | [-0.1624, 0.4710] | [-0.1569, 0.4613] | [-0.1546, 0.4699] | [-0.1533, 0.4644] |
| `37015020` | BERLIN [37015020] | Tmin | -0.0308 | [-0.2887, 0.2268] | [-0.2864, 0.2057] | [-0.2865, 0.2360] | [-0.2874, 0.2300] |
| `24025040` | ESCUELA AGRICOLA MOGOTES [24025040] | Tmax | -0.1451 | [-0.4453, 0.1428] | [-0.4941, 0.1483] | [-0.4595, 0.1367] | [-0.4723, 0.1354] |
| `24025040` | ESCUELA AGRICOLA MOGOTES [24025040] | Tmin | -0.2354 | [-0.5791, 0.1458] | [-0.5922, 0.1330] | [-0.5946, 0.1488] | [-0.5978, 0.1333] |
| `26135040` | AEROPUERTO MATECAÑA [26135040] | Tmax | 0.2532 | [-0.0060, 0.5046] | [0.0130, 0.4995] | [-0.0116, 0.5063] | [-0.0043, 0.5073] |
| `26135040` | AEROPUERTO MATECAÑA [26135040] | Tmin | 0.1331 | [-0.0633, 0.3366] | [-0.0539, 0.3163] | [-0.0654, 0.3366] | [-0.0710, 0.3293] |
| `27015070` | AEROPUERTO OLAYA HERRERA [27015070] | Tmax | 0.5121 | [0.2337, 0.7963] | [0.2130, 0.7806] | [0.2215, 0.7905] | [0.2175, 0.7966] |
| `27015070` | AEROPUERTO OLAYA HERRERA [27015070] | Tmin | 0.5450 | [0.3657, 0.7210] | [0.3824, 0.7057] | [0.3817, 0.7099] | [0.3790, 0.7064] |
| `52045010` | OBONUCO [52045010] | Tmax | 0.1706 | [-0.2247, 0.5165] | [-0.2135, 0.5320] | [-0.1980, 0.5435] | [-0.2069, 0.5539] |
| `52045010` | OBONUCO [52045010] | Tmin | 0.6907 | [0.3981, 0.9902] | [0.4079, 1.0416] | [0.4116, 1.0001] | [0.4097, 1.0017] |

**Table 5.2 Note**: The point changes correspond to the evaluation partition with year 2020 included (`level_change_k` under `with_2020` in `summary.json`).

### 5.3 JB and AC Calibration Valid-Year Inventory
Data sourced from `jb_ac_valid_years_inventory.csv`:

| Station ID | Station Name | Year | Part. | 4-Way Valid Days | Valid J.7 | Tmax Valid Mo | Tmax Inc | Tmin Valid Mo | Tmin Inc | DTR Valid Mo | DTR Inc |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `21205710` | JARDIN BOTANICO [21205710] | 1990 | calibration | 364 | True | 12 | True | 12 | True | 12 | True |
| `21205710` | JARDIN BOTANICO [21205710] | 1991 | calibration | 362 | True | 12 | True | 12 | True | 12 | True |
| `21205710` | JARDIN BOTANICO [21205710] | 1992 | calibration | 322 | True | 11 | True | 11 | True | 11 | True |
| `21205710` | JARDIN BOTANICO [21205710] | 1993 | calibration | 359 | True | 12 | True | 12 | True | 12 | True |
| `21205710` | JARDIN BOTANICO [21205710] | 1994 | calibration | 9 | True | 0 | False | 1 | False | 0 | False |
| `21205710` | JARDIN BOTANICO [21205710] | 1995 | calibration | 0 | False | 0 | False | 0 | False | 0 | False |
| `21205710` | JARDIN BOTANICO [21205710] | 1996 | calibration | 218 | True | 9 | True | 10 | True | 9 | True |
| `21205710` | JARDIN BOTANICO [21205710] | 1997 | calibration | 241 | True | 12 | True | 12 | True | 12 | True |
| `21205710` | JARDIN BOTANICO [21205710] | 1998 | calibration | 253 | True | 12 | True | 12 | True | 12 | True |
| `21205710` | JARDIN BOTANICO [21205710] | 1999 | calibration | 202 | True | 12 | True | 9 | True | 9 | True |
| `21205710` | JARDIN BOTANICO [21205710] | 2000 | calibration | 1 | True | 12 | True | 0 | False | 0 | False |
| `21205710` | JARDIN BOTANICO [21205710] | 2001 | calibration | 227 | True | 12 | True | 9 | True | 9 | True |
| `21205710` | JARDIN BOTANICO [21205710] | 2002 | calibration | 331 | True | 12 | True | 12 | True | 12 | True |
| `21205710` | JARDIN BOTANICO [21205710] | 2003 | calibration | 276 | True | 11 | True | 11 | True | 11 | True |
| `21205710` | JARDIN BOTANICO [21205710] | 2004 | calibration | 160 | True | 9 | True | 9 | True | 9 | True |
| `23215030` | AGUAS CLARAS [23215030] | 1990 | calibration | 358 | True | 12 | True | 12 | True | 12 | True |
| `23215030` | AGUAS CLARAS [23215030] | 1991 | calibration | 360 | True | 12 | True | 12 | True | 12 | True |
| `23215030` | AGUAS CLARAS [23215030] | 1992 | calibration | 359 | True | 12 | True | 12 | True | 12 | True |
| `23215030` | AGUAS CLARAS [23215030] | 1993 | calibration | 350 | True | 12 | True | 12 | True | 12 | True |
| `23215030` | AGUAS CLARAS [23215030] | 1994 | calibration | 362 | True | 12 | True | 12 | True | 12 | True |
| `23215030` | AGUAS CLARAS [23215030] | 1995 | calibration | 363 | True | 12 | True | 12 | True | 12 | True |
| `23215030` | AGUAS CLARAS [23215030] | 1996 | calibration | 358 | True | 12 | True | 12 | True | 12 | True |
| `23215030` | AGUAS CLARAS [23215030] | 1997 | calibration | 350 | True | 12 | True | 12 | True | 12 | True |
| `23215030` | AGUAS CLARAS [23215030] | 1998 | calibration | 359 | True | 12 | True | 12 | True | 12 | True |
| `23215030` | AGUAS CLARAS [23215030] | 1999 | calibration | 345 | True | 12 | True | 12 | True | 12 | True |
| `23215030` | AGUAS CLARAS [23215030] | 2000 | calibration | 354 | True | 12 | True | 12 | True | 12 | True |

### 5.4 JB Noise Floor on Included Years
Data sourced from `jb_noise_floor_included_years.csv`:

| Station | Var | Config | N Yrs | Mode | Max Bias Med (K) | Max Bias p5 (K) | Max Bias p95 (K) | Abs Off Med (K) | Abs Off p95 (K) | Seas Amp Med (K) | Seas Amp p95 (K) | Prev 15yr Max Bias Med (K) | Prev 15yr Max Bias p95 (K) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `21205710` | Tmax | balanced | 13 | Random 999 draws (seed 42) | 0.6450 | 0.4331 | 0.9080 | 0.0869 | 0.2631 | 0.6070 | 0.8805 | 0.6636 | 0.9899 |
| `21205710` | Tmax | pseudo_val_5yr | 13 | Random 999 draws (seed 42) | 0.6636 | 0.4321 | 0.9899 | 0.0910 | 0.2683 | 0.6280 | 0.8950 | 0.7630 | 1.2242 |
| `21205710` | Tmin | balanced | 12 | Enumerated ALL 924 distinct assignments | 0.7673 | 0.5103 | 1.1658 | 0.1092 | 0.3185 | 0.7071 | 1.0318 | 0.8119 | 1.2586 |
| `21205710` | Tmin | pseudo_val_5yr | 12 | Enumerated ALL 792 distinct assignments | 0.7806 | 0.5206 | 1.1644 | 0.1173 | 0.3200 | 0.7197 | 1.0715 | 1.0599 | 1.6378 |

**Table 5.4 Note**: Calibration and validation partition split sizes from `calibration_jb_ac_summary.json` (`item2_jb_noise_floor_included_years`): for Tmax (13 included years), `balanced` uses $k_{\text{cal}} = 7, k_{\text{val}} = 6$ and `pseudo_val_5yr` uses $k_{\text{cal}} = 8, k_{\text{val}} = 5$; for Tmin (12 included years), `balanced` uses $k_{\text{cal}} = 6, k_{\text{val}} = 6$ and `pseudo_val_5yr` uses $k_{\text{cal}} = 7, k_{\text{val}} = 5$.

### 5.5 AC Noise Floor as Stored
Data sourced from `calibration_jb_ac_summary.json` (`item2_ac_noise_floor_unchanged`):

| Station | Var | Config | N Yrs | Mode | Max Bias Med (K) | Max Bias p5 (K) | Max Bias p95 (K) | Abs Off Med (K) | Abs Off p95 (K) | Seas Amp Med (K) | Seas Amp p95 (K) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `23215030` | Tmax | balanced | 11 | Enumerated ALL 462 distinct assignments | 0.7529 | 0.4310 | 1.1328 | 0.1333 | 0.3637 | 0.6387 | 0.9647 |
| `23215030` | Tmax | pseudo_val_5yr | 11 | Enumerated ALL 462 distinct assignments | 0.7529 | 0.4310 | 1.1328 | 0.1333 | 0.3637 | 0.6387 | 0.9647 |
| `23215030` | Tmin | balanced | 11 | Enumerated ALL 462 distinct assignments | 0.4290 | 0.2724 | 0.6491 | 0.0960 | 0.2794 | 0.3437 | 0.5306 |
| `23215030` | Tmin | pseudo_val_5yr | 11 | Enumerated ALL 462 distinct assignments | 0.4290 | 0.2724 | 0.6491 | 0.0960 | 0.2794 | 0.3437 | 0.5306 |

**Table 5.5 Note**: For Aguas Claras (23215030, 11 calibration years), the 'balanced' and 'pseudo_val_5yr' rows are identical because both configurations use identical partition sizes ($k_{\text{cal}} = 6, k_{\text{val}} = 5$) and enumerate the exact same $\binom{11}{5} = 462$ assignments (`summary.json`, `part_b_benchmark_calibration_sampling_noise['23215030']['configurations']`).

### 5.6 Step versus Linear Residual Models (Calibration Years)
Data sourced from `jb_ac_step_vs_linear.csv`:

| Station ID | Station Name | Var | N Yrs | Span | Constant SSE | Linear Slope (K/dec) | Linear SSE | Linear SSE Red. | Break Year | Left Mean (K) | Right Mean (K) | Step Size (K) | Step SSE | Step SSE Red. | Step Red. (%) | Perm. Rank (%) | SSE Diff (Lin - Step) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `21205710` | JARDIN BOTANICO [21205710] | Tmax | 13 | 1990-2004 | 0.7434 | -0.1709 | 0.6660 | 0.0774 | 2001 | 0.0592 | -0.2793 | -0.3386 | 0.4788 | 0.2646 | 35.59% | 83.58% | 0.1872 |
| `21205710` | JARDIN BOTANICO [21205710] | Tmin | 12 | 1990-2004 | 1.1376 | 0.0138 | 1.1371 | 0.0005 | 1996 | -0.1577 | 0.0890 | 0.2467 | 0.9601 | 0.1775 | 15.60% | 43.24% | 0.1770 |
| `23215030` | AGUAS CLARAS [23215030] | Tmax | 11 | 1990-2000 | 1.0218 | 0.5763 | 0.6564 | 0.3653 | 1994 | -0.2631 | 0.2180 | 0.4811 | 0.3905 | 0.6313 | 61.79% | 97.60% | 0.2660 |
| `23215030` | AGUAS CLARAS [23215030] | Tmin | 11 | 1990-2000 | 0.5479 | -0.5012 | 0.2716 | 0.2763 | 1994 | 0.1884 | -0.1565 | -0.3449 | 0.2234 | 0.3244 | 59.22% | 95.40% | 0.0481 |

### 5.7 DTR Annual Series (Calibration Years)
Data sourced from `jb_ac_dtr_annual_series.csv`:

| Station ID | Station Name | Year | Partition | 4-Way Valid Days | Valid Months | DTR Raw Bias $d_y$ (K) | Included in Stats |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `21205710` | JARDIN BOTANICO [21205710] | 1990 | calibration | 364 | 12 | -2.1874 | True |
| `21205710` | JARDIN BOTANICO [21205710] | 1991 | calibration | 362 | 12 | -2.3255 | True |
| `21205710` | JARDIN BOTANICO [21205710] | 1992 | calibration | 322 | 11 | -2.0920 | True |
| `21205710` | JARDIN BOTANICO [21205710] | 1993 | calibration | 359 | 12 | -2.3711 | True |
| `21205710` | JARDIN BOTANICO [21205710] | 1994 | calibration | 9 | 0 | N/A | False |
| `21205710` | JARDIN BOTANICO [21205710] | 1995 | calibration | 0 | 0 | N/A | False |
| `21205710` | JARDIN BOTANICO [21205710] | 1996 | calibration | 218 | 9 | -1.7458 | True |
| `21205710` | JARDIN BOTANICO [21205710] | 1997 | calibration | 241 | 12 | -3.2529 | True |
| `21205710` | JARDIN BOTANICO [21205710] | 1998 | calibration | 253 | 12 | -2.8287 | True |
| `21205710` | JARDIN BOTANICO [21205710] | 1999 | calibration | 202 | 9 | -2.2689 | True |
| `21205710` | JARDIN BOTANICO [21205710] | 2000 | calibration | 1 | 0 | N/A | False |
| `21205710` | JARDIN BOTANICO [21205710] | 2001 | calibration | 227 | 9 | -1.7545 | True |
| `21205710` | JARDIN BOTANICO [21205710] | 2002 | calibration | 331 | 12 | -2.7289 | True |
| `21205710` | JARDIN BOTANICO [21205710] | 2003 | calibration | 276 | 11 | -2.9534 | True |
| `21205710` | JARDIN BOTANICO [21205710] | 2004 | calibration | 160 | 9 | -2.0341 | True |
| `23215030` | AGUAS CLARAS [23215030] | 1990 | calibration | 358 | 12 | -2.5953 | True |
| `23215030` | AGUAS CLARAS [23215030] | 1991 | calibration | 360 | 12 | -2.5994 | True |
| `23215030` | AGUAS CLARAS [23215030] | 1992 | calibration | 359 | 12 | -2.5294 | True |
| `23215030` | AGUAS CLARAS [23215030] | 1993 | calibration | 350 | 12 | -2.4991 | True |
| `23215030` | AGUAS CLARAS [23215030] | 1994 | calibration | 362 | 12 | -2.8282 | True |
| `23215030` | AGUAS CLARAS [23215030] | 1995 | calibration | 363 | 12 | -1.6350 | True |
| `23215030` | AGUAS CLARAS [23215030] | 1996 | calibration | 358 | 12 | -1.9050 | True |
| `23215030` | AGUAS CLARAS [23215030] | 1997 | calibration | 350 | 12 | -1.7119 | True |
| `23215030` | AGUAS CLARAS [23215030] | 1998 | calibration | 359 | 12 | -1.8551 | True |
| `23215030` | AGUAS CLARAS [23215030] | 1999 | calibration | 345 | 12 | -2.1068 | True |
| `23215030` | AGUAS CLARAS [23215030] | 2000 | calibration | 354 | 12 | -1.5381 | True |

### 5.8 DTR Calibration Summary: Level, Slope, and Best-Step Results
Data copied directly from `calibration_jb_ac_summary.json` under key `item4_dtr`:

| Metric / Parameter | JSON Key | Jardin Botanico (`21205710`) | Aguas Claras (`23215030`) |
| :--- | :--- | :--- | :--- |
| Number of Included Years | `n_included_years` | 12 | 11 |
| OLS Linear Slope (K/decade) | `ols_slope_k_per_decade` | -0.165929 | 1.070646 |
| OLS Slope 95% CI (K/decade) | `ols_slope_95ci_k_per_decade` | [-0.627760, 0.307940] | [0.721551, 1.615021] |
| Constant Model Mean (K) | `step_vs_linear.constant.mean` | -2.378589 | -2.163936 |
| Constant Model SSE | `step_vs_linear.constant.sse` | 2.462203 | 2.104300 |
| Linear OLS Intercept (K) | `step_vs_linear.linear_ols.intercept` | 30.760123 | -215.757853 |
| Linear OLS SSE | `step_vs_linear.linear_ols.sse` | 2.391261 | 0.843389 |
| Linear OLS SSE Reduction vs Constant | `step_vs_linear.linear_ols.sse_reduction_vs_constant` | 0.070942 | 1.260912 |
| Best Step Break Year | `step_vs_linear.best_step.break_year` | 1996 | 1994 |
| Best Step $n$ Left / $n$ Right | `step_vs_linear.best_step.n_left` / `n_right` | 5 / 7 | 5 / 6 |
| Best Step Mean Left (K) | `step_vs_linear.best_step.mean_left` | -2.144360 | -2.610286 |
| Best Step Mean Right (K) | `step_vs_linear.best_step.mean_right` | -2.545895 | -1.791978 |
| Best Step Size (K) | `step_vs_linear.best_step.step_size` | -0.401535 | 0.818308 |
| Best Step SSE | `step_vs_linear.best_step.sse` | 1.991948 | 0.278043 |
| Best Step SSE Reduction vs Constant | `step_vs_linear.best_step.sse_reduction_vs_constant` | 0.470255 | 1.826257 |
| Best Step % SSE Reduction | `step_vs_linear.best_step.pct_sse_reduction` | 19.10% | 86.79% |
| Permutation Percentile Rank (%) | `step_vs_linear.best_step.permutation_percentile_rank` | 48.45% | 100.00% |
| Median Permutation SSE Reduction | `step_vs_linear.best_step.median_perm_sse_reduction` | 0.474574 | 0.387894 |
| SSE Difference (Linear − Step) | `step_vs_linear.sse_diff_linear_minus_step` | 0.399313 | 0.565346 |

## 6. Comparison of First JB Noise Floor Run with Included-Years Run

The first calibration-only noise-floor run of Jardin Botanico (21205710) stored in `summary.json` (`part_b_benchmark_calibration_sampling_noise`) used the nominal 15 calibration years (1990–2004).

The table below compares the first-run values (`summary.json`, `part_b_benchmark_calibration_sampling_noise`) with the included-years values (`jb_noise_floor_included_years.csv`) at full precision:

| Variable | Run | Configuration | Split ($k_{\text{cal}}, k_{\text{val}}$) | Median Max Monthly Bias (K) | Median Absolute Offset (K) | Median Seasonal Amplitude (K) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Tmax | First Run (nominal 15 yr) | balanced | 8, 7 | 0.6636336534543155 | 0.09103278624034061 | 0.6280291350217031 |
| Tmax | First Run (nominal 15 yr) | pseudo_val_5yr | 10, 5 | 0.7629585266113246 | 0.1124702768472113 | 0.725637345989683 |
| Tmax | Included Years (13 yr) | balanced | 7, 6 | 0.6449618129793606 | 0.08693345513878636 | 0.606960948648626 |
| Tmax | Included Years (13 yr) | pseudo_val_5yr | 8, 5 | 0.6636321249803743 | 0.09103293677023909 | 0.6280262810949289 |
| Tmin | First Run (nominal 15 yr) | balanced | 8, 7 | 0.8118519881675983 | 0.12213907645378479 | 0.7586228262436732 |
| Tmin | First Run (nominal 15 yr) | pseudo_val_5yr | 10, 5 | 1.0599026718139655 | 0.14373085849843034 | 0.966203054000093 |
| Tmin | Included Years (12 yr) | balanced | 6, 6 | 0.7672650674673225 | 0.10924049870388117 | 0.707131899568521 |
| Tmin | Included Years (12 yr) | pseudo_val_5yr | 7, 5 | 0.7806486244842681 | 0.11734543488588874 | 0.71966595655151 |

From that table only:
- For **Tmax**, the first-run `balanced` row and the included-years `pseudo_val_5yr` row coincide to four decimal places (`0.6636 K` median max monthly bias, `0.0910 K` median absolute offset, `0.6280 K` median seasonal amplitude).
- At full floating-point precision they differ at the sixth decimal place. The cause of that difference was not established.
- For **Tmin**, none of the first-run rows coincide to four decimal places with any included-years row.
- The included-years results in `jb_noise_floor_included_years.csv` supersede the first run.

## 7. File Manifest

All archived files and their size and SHA-256 hashes:

| Archived Relative Path | Original Path on Disk | Size (bytes) | SHA-256 |
| :--- | :--- | :--- | :--- |
| `scripts/andean_residual_diagnostic_round2.py` | `local/analysis_k/diagnostic2/andean_residual_diagnostic.py` | 64744 | `9e74e46698ccffbc587fb93dbe55d23ba971ddffb62444486d982aa992a0b3c2` |
| `scripts/diagnostic_gap_closure.py` | `local/analysis_k/diagnostic2/diagnostic_gap_closure.py` | 15456 | `29d6ec96e6c30d16b475c21bc89db1d9ce4de4eac6d2d51debbb3abf4e220d91` |
| `scripts/diagnostic_calibration_jb_ac.py` | `local/analysis_k/diagnostic2/diagnostic_calibration_jb_ac.py` | 31993 | `7ee56835ce43e9adcb3322c3a020c87060d4ba336504d128c28998236af7b7f9` |
| `results/andean_and_calibration_annual/diagnostic_37015020_no2020.csv` | `local/analysis_k/diagnostic2/diagnostic_37015020_no2020.csv` | 4123 | `cae8a71572c90317a8b9591628e013d69757b504767356f4031e3ae43112fed0` |
| `results/andean_and_calibration_annual/diagnostic_24025040_no2020.csv` | `local/analysis_k/diagnostic2/diagnostic_24025040_no2020.csv` | 4648 | `9411673db2c4fa146fd97098fc79e814f456ee7efb829087aba3164f5a61d1eb` |
| `results/andean_and_calibration_annual/diagnostic_26135040_no2020.csv` | `local/analysis_k/diagnostic2/diagnostic_26135040_no2020.csv` | 5655 | `05c9d865728e1389a19b06d6af07d3f59fc6e8684293d7f6f038b03d223c15b7` |
| `results/andean_and_calibration_annual/diagnostic_27015070_no2020.csv` | `local/analysis_k/diagnostic2/diagnostic_27015070_no2020.csv` | 6755 | `426b0d64669e34410c3b10d644deea257028dbd3f58910190d6c9ae12f918d3b` |
| `results/andean_and_calibration_annual/diagnostic_52045010_no2020.csv` | `local/analysis_k/diagnostic2/diagnostic_52045010_no2020.csv` | 3988 | `ff90187937573d58e42f3994b9bd21296d704a43259bbd469a96d49f214c7802` |
| `results/andean_and_calibration_annual/calibration_annual_21205710.csv` | `local/analysis_k/diagnostic2/calibration_annual_21205710.csv` | 3248 | `03842bc83e7768003c682f0e9b15d777369eb2b240af1ce7252b242d6ec17ac0` |
| `results/andean_and_calibration_annual/calibration_annual_23215030.csv` | `local/analysis_k/diagnostic2/calibration_annual_23215030.csv` | 2497 | `173cf1b635717fde49aa37f66bdd2d0c269e708144df2bea5b19ea527e7ff927` |
| `results/andean_and_calibration_annual/summary.json` | `local/analysis_k/diagnostic2/summary.json` | 109487 | `cdd6f219137aef29ac9ab0f936e0b9e9d1d587dcdb5ca2da3a66478a649106fe` |
| `results/gap_closure/gap_closure_summary.json` | `local/analysis_k/diagnostic2/gap_closure_summary.json` | 26781 | `b2471edc8b357f0227d75408363229a6592a89aadf23882e31061a262ae4ec47` |
| `results/gap_closure/bootstrap_ci_10k_comparison.csv` | `local/analysis_k/diagnostic2/bootstrap_ci_10k_comparison.csv` | 2412 | `8e66ed11f85453622acf7ac87641bc76e4261c185f1e5dd1893a4b467e8a461f` |
| `results/jb_ac_calibration/jb_ac_valid_years_inventory.csv` | `local/analysis_k/diagnostic2/jb_ac_valid_years_inventory.csv` | 2386 | `9ea4da88ee4c535bdee146588cb2902ebbbea2203067fe21a52a07088c592b6f` |
| `results/jb_ac_calibration/jb_noise_floor_included_years.csv` | `local/analysis_k/diagnostic2/jb_noise_floor_included_years.csv` | 1395 | `c39a0d5b40d3948b0813bde337aabc368de4bd4ad31017e877a144d4fa42ac81` |
| `results/jb_ac_calibration/jb_ac_step_vs_linear.csv` | `local/analysis_k/diagnostic2/jb_ac_step_vs_linear.csv` | 1488 | `a4a775d134204f9214db4ab7ac77a364ebc3bed454bf148a4bcf0fd2d8b2fc90` |
| `results/jb_ac_calibration/jb_ac_dtr_annual_series.csv` | `local/analysis_k/diagnostic2/jb_ac_dtr_annual_series.csv` | 2237 | `926dff27d8e6f268579fbb2c799bc210f579340351819ff2c93980380412fc9a` |
| `results/jb_ac_calibration/calibration_jb_ac_summary.json` | `local/analysis_k/diagnostic2/calibration_jb_ac_summary.json` | 42983 | `e6c5df626bf1e006cd20ad4e24f81db04d861387a23d610bd1991e11e263a446` |

> INTEGRITY NOTE (scripts/diagnostic_gap_closure.py): The SHA-256 printed at the end of the run that produced gap_closure_summary.json and bootstrap_ci_10k_comparison.csv was 97b8e796cef7b96f3978573be4acb97d4f8ca9bf66248ab9650dc9709294f81d. The file on disk at archive time has SHA-256 29d6ec96e6c30d16b475c21bc89db1d9ce4de4eac6d2d51debbb3abf4e220d91 (15,456 bytes), file time 2026-11-02 12:10:43 (UTC-05:00). Removing its last byte, a line feed after the final line, reproduces the first hash. The archived file is the on-disk file, unmodified. The cause of the extra byte was not established.

## 8. Exposure Declaration

Observations of the five opened Andean stations (37015020, 24025040, 26135040, 27015070, 52045010) across their calibration and validation partitions were used. For benchmark stations JB (21205710) and AC (23215030), only calibration-year observations and ERA5 were used. The validation openings ledger was not accessed. ERA5 files of years after the calibration partition were hashed by the diagnostic script but not loaded by the loader (see section 3). No value from a validation year of the JB and AC observation files was used in any computation, printed or inspected; the loader reads the whole observation file and validates it before trimming (see section 2).

## 9. Statement on Procedures

The procedure of each diagnostic was fixed in the task prompts sent to the coding agent on 2026-11-01 and 2026-11-02 before that diagnostic's results were seen. The choice of which diagnostics to run was sequential and informed by earlier results of the same round: the 10,000-replica gap closure, the valid-year inventory, the step-versus-linear comparison and the DTR annual series were requested after earlier results of this round had been seen, and the recomputation of the JB noise floor on included years was requested after the first calibration-only run showed that JB has calibration years without data. The repository records the procedures after the fact and does not prove that order.

## 10. Note on Console Outputs

The verbatim quotations and the valid-year comparison produced in the same round (bootstrap function, J.7 text, delta-z grep) were printed to the console and are not archived; the cited files at commit 69b0d707dc1052aa55f9e2f3c327ba776abb8371 allow them to be reproduced.

