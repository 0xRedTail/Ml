# ML Classifier Comparison: Keystroke Dynamics & Ad-URL Blocking

Two small, leakage-safe machine-learning experiments that compare the same four classifiers —
**Random Forest, Logistic Regression, SVM and XGBoost** — plus an equal-weight **Soft Voting** ensemble.

| Experiment | Task | Input | Target |
|---|---|---|---|
| **Keystroke dynamics** | Identify which user typed a sample (5-way classification) | 8 numeric timing features | `user_id` |
| **Ad-URL blocking** | Classify a URL string as `ALLOW` (0) or `BLOCK` (1) | URL text (TF-IDF features) | `label` |

Each experiment is provided as a Google Colab notebook **and** as an equivalent command-line script.

---

## Repository structure

```
.
├── Keystroke_RF_LR_SVM_XGBoost_Colab.ipynb   # Colab notebook – keystroke experiment
├── Ad_Blocker_RF_LR_SVM_XGBoost_Colab.ipynb  # Colab notebook – ad-URL experiment
├── keystroke_four_models.py                  # CLI version of the keystroke experiment
├── ad_blocker_four_models.py                 # CLI version of the ad-URL experiment
├── data/
│   ├── README.md                             # Dataset schemas and privacy notes
│   ├── ad_urls.csv                           # URL dataset (30 rows)
│   └── keystroke_features.csv                # keystroke timing dataset (500 rows)
├── requirements.txt
├── .gitignore                                # keeps trained models and outputs out of Git
└── README.md
```

> Both datasets live in [`data/`](data/README.md). The keystroke file is behavioural biometric data — see [Privacy](#privacy-and-data-handling).

## Methodology

Both experiments follow the same protocol so results are comparable and free of data leakage:

1. **Stratified hold-out split** – 25 % test (keystroke) / 30 % test (ad URLs). The test set is never used for tuning or model selection.
2. **Scikit-learn `Pipeline`s** – imputation, scaling and text vectorisation are fitted *inside* each CV fold, never on the full data.
3. **Hyperparameter tuning** – `RandomizedSearchCV` (4 configurations per model by default), 5-fold stratified CV on the training set only.
4. **Soft Voting ensemble** – the four tuned pipelines are combined with equal weights and scored with the same CV.
5. **Selection by training CV** – the winner is the model with the highest mean CV accuracy.
6. **Transparent reporting** – every candidate is scored on the hold-out set for reporting, but this never changes the winner.
7. **Outputs** – CV scores, hold-out scores, confusion matrix, metrics JSON and the selected pipeline are saved.

### Models and preprocessing

| Model | Keystroke preprocessing | Ad-URL preprocessing |
|---|---|---|
| Random Forest | Median imputation | Char TF-IDF (3–5-grams) + token TF-IDF (1–2-grams) + weighted ad-term flags |
| Logistic Regression | Median imputation + StandardScaler | same |
| SVM | Median imputation + StandardScaler (RBF) | same (linear/RBF tuned) |
| XGBoost | Median imputation | same |

## Quick start

### Option A – Google Colab
1. Open a notebook from this repo in Colab (**File → Open notebook → GitHub**).
2. Run the cells in order and upload the requested CSV when prompted.

### Option B – Local command line

```bash
git clone https://github.com/Ruchika-choudhary8/new.git
cd new
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Keystroke experiment
python keystroke_four_models.py --csv data/keystroke_features.csv

# Ad-URL experiment
python ad_blocker_four_models.py --csv data/ad_urls.csv
```

Useful options: `--n-iter 10` (longer tuning search), `--output-dir <dir>`, `--seed 42`.

## Dataset formats

**Keystroke CSV** – one row per typing sample.
`user_id` (string label) plus numeric timing-statistic columns. Non-numeric columns other than `user_id` are ignored. `user_id` is never used as a feature.

**Ad-URL CSV** – two columns: `url` (string) and `label` (`0` = ALLOW, `1` = BLOCK). Conflicting and duplicate URLs are removed so the same URL cannot land in both train and test.

See [`data/README.md`](data/README.md) for details.

## Results

Running the scripts writes the following to `outputs/` (keystroke) or `outputs/ad_blocker/` (ads):

- `*_cv_scores.csv` – training-CV comparison (used for selection)
- `*_holdout_scores.csv` – hold-out accuracy, balanced accuracy, macro-F1 (reporting only)
- `*_confusion_matrix.png`
- `*_metrics.json`
- `*_best_pipeline.joblib` – the CV-selected pipeline

Add your own numbers here after running:

| Experiment | CV-selected model | Hold-out accuracy | Majority baseline |
|---|---|---|---|
| Keystroke | _fill in_ | _fill in_ | _fill in_ |
| Ad URLs | _fill in_ | _fill in_ | _fill in_ |

## Limitations — please read

- **Keystroke:** with ~500 samples and 8 aggregate timing features, 5-way identification performs close to chance. More complex models do not fix weakly discriminative features. This is **not** suitable for authentication. A credible study needs multiple independent sessions, key-pair-level features, session-separated splits, and FAR / FRR / EER evaluation.
- **Ad URLs:** the dataset is ~30 synthetic URLs, so the test set has only a handful of rows. High accuracy here is **not** evidence of real-world performance. The code classifies URL strings only; it does **not** block ads or intercept traffic. A real blocker needs a browser extension or network filter, modern blocklist/allowlist data, and false-positive analysis.

## Privacy and data handling

Keystroke timing is behavioural biometric data.

- Only use data you are authorised to process.
- The keystroke CSV in `data/` should only be published if you are authorised to share it and it contains no real personal identifiers. Consider keeping the repository **private**.
- Do not commit trained models (`.joblib`); the `.gitignore` excludes them.
- Only load `.joblib` files you created yourself — joblib/pickle files can execute arbitrary code when loaded.

## Requirements

Python 3.10+ and the packages in `requirements.txt` (scikit-learn, XGBoost, pandas, NumPy, matplotlib, joblib).

## Author

By [Ruchika-choudhary8](https://github.com/Ruchika-choudhary8).
