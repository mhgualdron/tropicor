# Appendix J Primary Benchmark Calibration Archive

These reports were produced from calibration data only (Appendix J.4).

| Path | SHA-256 | Generator Git Commit | Winner Config ID |
|---|---|---|---|
| docs/science/appendix_j/calibration/benchmark_calibration_21205710.json | f5cc719602aeec9b48570662841c689b85f4b9a8b3317320c0f9cc97bd961239 | 971917d02de700fbc667a552ea7d53da28712cb0 | 7 |
| docs/science/appendix_j/calibration/benchmark_calibration_23215030.json | 7ceca885cab31d97005aaedeba059b83006805d4a49104eef96d1c295fc75bac | 971917d02de700fbc667a552ea7d53da28712cb0 | 6 |

## Appendix J.5 Validation Lag Audit Archive

These reports were produced from the observational lag audit on validation partitions (Protocol v1.8, Appendix J.5 / Section 8). Execution of this audit wrote one ledger record per station (`purpose="lag_audit"`, increasing ledger line count from 5 to 7 in `local/reports/validation_openings.jsonl`).

| Path | SHA-256 | Generator Git Commit | Mode | Optimal Lag (Tmax) | Optimal Lag (Tmin) | Sensitivity Activated |
|---|---|---|---|---|---|---|
| docs/science/appendix_j/lag_audit/benchmark_lag_audit_validation_21205710.json | 61aca73fd3daf67d98d6beb20b53f032398e71ee21729012a4e4d4e5c537e97f | 7b1493c084bf33bba221bdc5c258d12d51388de8 | validation | 0 | 0 | False |
| docs/science/appendix_j/lag_audit/benchmark_lag_audit_validation_23215030.json | 6ff7cf350a7b287769e867dbec403a57ca37fe023602ef86b49fb13865c2e2db | 7b1493c084bf33bba221bdc5c258d12d51388de8 | validation | 0 | 0 | False |

### Disclosures

1. ERA5 source equivalence (`era5_hourly` 5x5 vs `era5_ts` central cell) was verified only on the calibration years; its extension to the validation years is an assumption.
2. The loader deduplicates dates with `sort_values` + `drop_duplicates(keep="first")` (unstable sort); the audit adds a loud guard for duplicate dates mixing missing and non-missing values.
