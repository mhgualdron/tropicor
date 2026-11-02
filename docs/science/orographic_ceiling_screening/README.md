# Orographic Ceiling Screening (Gate 1)

## 1. Purpose
Screening of the maximum effect that a daily free-atmosphere lapse-rate term could have on the monthly bias of Level 1, on seven stations.

## 2. Gate Rule
Continue only if the effect ceiling at AC (23215030) is $\ge 2\delta$ with $\delta = 0.1$ K (i.e. ceiling $C \ge 0.2$ K).

## 3. Declared Parameters
- **Partitions per station**: Calibration = `era5_source_equivalence.calibration_years`, Validation = `partition_years` (JB: cal 1990–2004, val 2005–2020; AC: cal 1990–2000, val 2001–2011; Andean stations: per `tropicor/data/andean_station_registry.json`).
- **Local Day Definition**: Date of (UTC time minus 5 hours). Only local days falling strictly inside the partition years are evaluated.
- **Level Eligibility**: Geopotential height $Z = z / 9.80665 \ge z_{\text{era5}} + 100$ m in $\ge 99.9\%$ of calibration partition time steps.
- **Level Set $S$**: The three lowest pressure levels eligible under the criterion above.
- **Diurnal Steps**: $T_{\max}$ uses the 18:00 UTC step; $T_{\min}$ uses the 12:00 UTC step.
- **Sensitivities**: Each of the four hours (00, 06, 12, 18 UTC), 4-step daily mean, and two-level $S$ ($S_2$, two lowest eligible levels).

## 4. Pre-Specification Statement
The gate rule and parameters above were fixed in the task prompt sent to the coding agent on 2026-10-31 before any result was seen. The repository records them after the fact and does not prove that order.

## 5. Input Data & Request Structure
- **Dataset**: Copernicus Climate Data Store (CDS) `reanalysis-era5-pressure-levels-timeseries`.
- **Request Dictionary Structure**:
  ```json
  {
    "variable": ["temperature", "geopotential"],
    "location": {
      "latitude": <lat>,
      "longitude": <lon>
    },
    "date": "<start_year>-01-01/<end_year+1>-01-01",
    "data_format": "netcdf",
    "pressure_level": ["1000", "925", "850", "700", "600", "500", "400"]
  }
  ```
- **Returned Format**: CDS returns point timeseries NetCDF files containing 13 pressure levels (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50 hPa) on a 6-hourly time axis (`valid_time`), with variables `t` (temperature in K) and `z` (geopotential in $\text{m}^2\,\text{s}^{-2}$).
- **NetCDF File Hashes**:

| Station ID | Filename | Size (bytes) | SHA-256 Hash | Status |
| :--- | :--- | :--- | :--- | :--- |
| `21205710` | `era5_pl_ts_1990_2020.nc` | 5095950 | `5920e9a370b8b9505bc2fdabfed1500f6e6b4d5903a6a3c670acd4f19c2bb2d9` | Used |
| `23215030` | `era5_pl_ts_1990_2011.nc` | 3622926 | `92709b90f9f3238a42bc14fdeb4997fbb79b157af4d80fc4abc7e281c85b99a3` | Used |
| `23215030` | `era5_pl_ts_2001_2001.nc` | 186766 | `e2b51573944d41547cef9a9fc92daa53ff5f36057af5aa4baecf02a53f16d5dd` | NOT used (1-year connectivity test) |
| `37015020` | `era5_pl_ts_1991_2020.nc` | 4932430 | `5b040c288ee231fcbf73f376ac982d84b20f947f6fe62bdb7387b172fa84b0f2` | Used |
| `24025040` | `era5_pl_ts_1990_2020.nc` | 5095950 | `7997e113de331d5e7a307990faf7b937df5ae0207d57ef833b66be1b78b28bde` | Used |
| `26135040` | `era5_pl_ts_1990_2017.nc` | 4604942 | `1f551e8ddd41d0f684538836a19c393aec7882633b23fdd5a0d37fe513f19668` | Used |
| `27015070` | `era5_pl_ts_1990_2019.nc` | 4931982 | `ebc5ced817200da991e3c17d429f5d75e3778f705c92f9a2e9dcae52b89229fb` | Used |
| `52045010` | `era5_pl_ts_1990_2020.nc` | 5095950 | `9928dceff0d54cda0c8560e92a0bf28dd59fb2b9fcee122641f79f1f006c17a1` | Used |

## 6. Station Information
Elevation constants and partition definitions used by the screening script:

| Station ID | Station Name | $z_{\text{era5}}$ (m) | $\Delta z$ (m) | Calibration Years | Validation Years |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `21205710` | JB | 2753.40 | -201.40 | 1990-2004 (15 yrs) | 2005-2020 (16 yrs) |
| `23215030` | AC | 762.43 | -628.63 | 1990-2000 (11 yrs) | 2001-2011 (11 yrs) |
| `37015020` | BERLIN [37015020] | 2877.89 | -336.11 | 1991-2009 (11 yrs) | 2010-2020 (10 yrs) |
| `24025040` | ESCUELA AGRICOLA MOGOTES [24025040] | 1791.48 | 118.48 | 1990-2000 (10 yrs) | 2001-2020 (10 yrs) |
| `26135040` | AEROPUERTO MATECAÑA [26135040] | 1431.79 | 89.79 | 1990-2005 (12 yrs) | 2006-2017 (12 yrs) |
| `27015070` | AEROPUERTO OLAYA HERRERA [27015070] | 2032.56 | 540.56 | 1990-2004 (14 yrs) | 2005-2019 (14 yrs) |
| `52045010` | OBONUCO [52045010] | 2664.36 | -45.64 | 1990-1999 (10 yrs) | 2000-2020 (10 yrs) |

Calibration and validation year lists are those of the registry (Andean stations) or Table J.2 (JB, AC) and are not always contiguous; each range shows the first and last year and the number in parentheses is the number of years used.

Sign convention of the Delta z column. The rows of JB (21205710) and AC (23215030) follow Table J.2 of the evaluation protocol: Delta z = z_station - z_ERA5. The rows of the five Andean stations are printed as stored in docs/science/appendix_g/statics_g7.json: Delta z = z_ERA5 - z_station. The two conventions have opposite sign. The screening scripts use only the absolute value |Delta z| in all ceiling and daily-scale calculations (ceiling_orographic.py lines 299-301; ceiling_orographic_v2.py lines 374-376, 450), so the difference in sign convention does not affect any reported ceiling value.

## 7. Archived File Manifest

| File Path | Size (bytes) | SHA-256 Hash |
| :--- | :--- | :--- |
| `scripts/ceiling_orographic.py` | 23000 | `39a98aec4c98d6df4926c3f6ec9365b53ca03c7e27e2138493deb0d5fcb7cff1` |
| `scripts/ceiling_orographic_v2.py` | 35908 | `588050040c23232f47c0d89a73a58f8cdca2a5191984ff47fa672bd279e7d648` |
| `results/ceiling_21205710.csv` | 31008 | `ca1d8344b02c473b755f2b61aec483dd3df78bd5e4c77731f6ec14765475b8c6` |
| `results/ceiling_23215030.csv` | 30865 | `369d2fc0488adc038235633a65464b2b99d2963d7d91540294cba88fe1ffbdfe` |
| `results/ceiling_37015020.csv` | 32876 | `c29fc016cabdcc8b841ae555bc8f1f3c80d78f499b82b0febf81b64f59d26406` |
| `results/ceiling_24025040.csv` | 35079 | `16e1c6ceae2e9f1cc941f5a236f7958396e01fb506f5250f9eed9de19f8c44e1` |
| `results/ceiling_26135040.csv` | 34549 | `32e00a4c988c60401a3aab5942923f9038234dcd61298d6942fbdcfee71e6e4f` |
| `results/ceiling_27015070.csv` | 34754 | `d7359518155fdcc2079d436de5746ff7ffbc7d2f7ea4162a792c778c483eb8ec` |
| `results/ceiling_52045010.csv` | 32982 | `a9451c0ee329db26174019f89f11f2349b1146612f51ac2cdb80b94c0336440c` |
| `results/ceiling_summary.json` | 838986 | `de442b41b86c4921a71b697ef7efacfd588828081849f4a37e0a7feb0a630c9b` |

## 8. Primary Ceiling Summary Table
Generated programmatically from `results/ceiling_summary.json`:

| Station ID | Station Name | Variable | Levels (hPa) | Max Ceiling $C$ (K) | Months $\ge 0.2$ K | Daily SD (K) | Anom SD (K) | P5 Anom (K) | P95 Anom (K) | Max \|Anom\| (K) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `21205710` | JB | Tmax | `[700,600,500]` | 0.0563 | 0 | 0.1015 | 0.0938 | -0.1762 | 0.1300 | 0.5054 |
| `21205710` | JB | Tmin | `[700,600,500]` | 0.0403 | 0 | 0.0894 | 0.0831 | -0.1640 | 0.1059 | 0.5615 |
| `23215030` | AC | Tmax | `[850,700,600]` | 0.0730 | 0 | 0.2194 | 0.1938 | -0.3258 | 0.3081 | 0.7224 |
| `23215030` | AC | Tmin | `[850,700,600]` | 0.0648 | 0 | 0.1768 | 0.1567 | -0.2692 | 0.2371 | 0.8376 |
| `37015020` | BERLIN [37015020] | Tmax | `[700,600,500]` | 0.0422 | 0 | 0.1352 | 0.1306 | -0.1918 | 0.2289 | 0.5665 |
| `37015020` | BERLIN [37015020] | Tmin | `[700,600,500]` | 0.0595 | 0 | 0.1278 | 0.1056 | -0.2110 | 0.1306 | 0.7497 |
| `24025040` | ESCUELA AGRICOLA MOGOTES [24025040] | Tmax | `[700,600,500]` | 0.0111 | 0 | 0.0293 | 0.0287 | -0.0410 | 0.0501 | 0.1647 |
| `24025040` | ESCUELA AGRICOLA MOGOTES [24025040] | Tmin | `[700,600,500]` | 0.0162 | 0 | 0.0380 | 0.0327 | -0.0576 | 0.0443 | 0.2216 |
| `26135040` | AEROPUERTO MATECAÑA [26135040] | Tmax | `[700,600,500]` | 0.0056 | 0 | 0.0207 | 0.0196 | -0.0286 | 0.0341 | 0.1224 |
| `26135040` | AEROPUERTO MATECAÑA [26135040] | Tmin | `[700,600,500]` | 0.0075 | 0 | 0.0220 | 0.0200 | -0.0344 | 0.0288 | 0.1321 |
| `27015070` | AEROPUERTO OLAYA HERRERA [27015070] | Tmax | `[700,600,500]` | 0.0766 | 0 | 0.1855 | 0.1556 | -0.2227 | 0.2760 | 0.9560 |
| `27015070` | AEROPUERTO OLAYA HERRERA [27015070] | Tmin | `[700,600,500]` | 0.0450 | 0 | 0.1716 | 0.1504 | -0.2536 | 0.2224 | 0.8780 |
| `52045010` | OBONUCO [52045010] | Tmax | `[700,600,500]` | 0.0058 | 0 | 0.0159 | 0.0154 | -0.0251 | 0.0263 | 0.0731 |
| `52045010` | OBONUCO [52045010] | Tmin | `[700,600,500]` | 0.0047 | 0 | 0.0159 | 0.0154 | -0.0280 | 0.0218 | 0.0722 |

## 9. Exposure Declaration
- ERA5 pressure-level data of the validation years of all seven stations were used.
- No observation of JB (21205710) or AC (23215030) was read.
- The ledger was not accessed.
- The Andean scorecards were not used in this analysis.

## 10. Integrity Note
- Script ceiling_orographic.py: the first run reported SHA-256 dd533dfbd3f2798e908de953cea2dfe8b5ac2a7a1a59608aa1e7b4d423051df0. The file on disk now has SHA-256 39a98aec4c98d6df4926c3f6ec9365b53ca03c7e27e2138493deb0d5fcb7cff1 (23,000 bytes), LastWriteTime 2026-10-31 11:37:47 (UTC-05:00), earlier than ceiling_orographic_v2.py (11:55:44). The difference between the two hashes originated in the first task and cannot be reconstructed (local/ is not under version control). The extended task did not modify the file.
- The JB and AC CSV outputs of the first run and of the extended run are byte-identical: ceiling_21205710.csv ca1d8344b02c473b755f2b61aec483dd3df78bd5e4c77731f6ec14765475b8c6; ceiling_23215030.csv 369d2fc0488adc038235633a65464b2b99d2963d7d91540294cba88fe1ffbdfe.
- The first-run summary JSON (SHA-256 d8323979b3e5a190664cd67b35d5c8368194006bbc353cf132d71311c3fdbe35) was overwritten by the extended run (ceiling_summary.json, SHA-256 de442b41b86c4921a71b697ef7efacfd588828081849f4a37e0a7feb0a630c9b). Its hash is kept here.
- The AC computation used era5_pl_ts_1990_2011.nc (SHA-256 92709b90f9f3238a42bc14fdeb4997fbb79b157af4d80fc4abc7e281c85b99a3), confirmed by the input hash recorded in ceiling_summary.json and by the paths at lines 42 and 48 of the script; era5_pl_ts_2001_2001.nc (SHA-256 e2b51573944d41547cef9a9fc92daa53ff5f36057af5aa4baecf02a53f16d5dd) was a one-year connectivity test and was not used.
