# Data

This folder holds the two datasets used by the experiments.

| File | Rows | Used by | Target column |
|---|---|---|---|
| [`keystroke_features.csv`](keystroke_features.csv) | 500 | `keystroke_four_models.py` | `user_id` |
| [`ad_urls.csv`](ad_urls.csv) | 30 | `ad_blocker_four_models.py` | `label` |

---

## `keystroke_features.csv`

Aggregate keystroke-timing statistics, one row per typing sample.

- **Shape:** 500 rows × 9 columns, no missing values
- **Classes:** 5 users (multi-class classification)

| user_id | samples |
|---|---|
| user_001 | 94 |
| user_002 | 91 |
| user_003 | 108 |
| user_004 | 118 |
| user_005 | 89 |

### Columns

| Column | Type | Description | Range (min – max) |
|---|---|---|---|
| `dwell_mean` | float | Mean key hold time | 56.4 – 139.5 |
| `dwell_std` | float | Std. dev. of key hold time | 5.3 – 36.0 |
| `flight_mean` | float | Mean time between key release and next key press | 83.7 – 229.7 |
| `flight_std` | float | Std. dev. of flight time | 10.3 – 54.8 |
| `positive_flight_mean` | float | Mean of non-negative flight times | 96.2 – 242.8 |
| `typing_speed` | float | Typing speed | 2.2 – 8.9 |
| `dwell_flight_ratio` | float | Dwell mean divided by flight mean | 0.38 – 1.39 |
| `timing_cv` | float | Coefficient of variation of timings | 0.05 – 0.53 |
| `user_id` | string | Person who typed the sample — **the target, never a feature** | `user_001` … `user_005` |

> Column meanings are inferred from their names. The file does not state units; timing values look like milliseconds, but confirm this against how the data was collected.

### Privacy note

Keystroke timing is behavioural biometric data. Only publish it if you are authorised to share it and the `user_id` values are anonymous labels. If in doubt, keep the repository private.

---

## `ad_urls.csv`

Small, synthetic URL dataset for ALLOW / BLOCK classification.

- **Shape:** 30 rows × 2 columns, no missing values, no duplicate URLs
- **Balance:** 15 BLOCK, 15 ALLOW

| Column | Type | Description |
|---|---|---|
| `url` | string | URL to classify |
| `label` | int | `0` = ALLOW, `1` = BLOCK |

Example rows:

```
https://ads.example.net/banner/300x250.js,1
https://news.example.com/world/elections,0
```

> With only 30 URLs, the hold-out test set has about 9 rows. Scores on it say little about real-world performance.

---

## Using the data

```bash
python keystroke_four_models.py --csv data/keystroke_features.csv
python ad_blocker_four_models.py --csv data/ad_urls.csv
```
