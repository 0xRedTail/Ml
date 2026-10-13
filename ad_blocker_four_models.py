#!/usr/bin/env python3
"""ML-based ad-URL classification: RF, Logistic Regression, SVM, XGBoost + Soft Voting.

Classifies URL strings as ALLOW (0) or BLOCK (1). It does NOT intercept traffic or
block ads in a browser.

Usage:
    python ad_blocker_four_models.py --csv data/ad_urls.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

LABELS = {0: "ALLOW", 1: "BLOCK"}

AD_TERMS = [
    "ad", "ads", "adserver", "advertisement", "advertising", "banner",
    "doubleclick", "tracker", "tracking", "analytics", "pixel", "collect",
    "retargeting", "impression", "sponsor", "promoted", "affiliate",
    "marketing", "campaign", "beacon",
]

SEARCH_SPACES = {
    "Random Forest": {
        "model__n_estimators": [80, 120, 200],
        "model__max_depth": [None, 3, 6],
        "model__min_samples_leaf": [1, 2],
    },
    "Logistic Regression": {
        "model__C": [0.1, 1.0, 10.0, 100.0],
        "model__class_weight": [None, "balanced"],
    },
    "SVM": {
        "model__kernel": ["linear", "rbf"],
        "model__C": [0.1, 1.0, 10.0],
        "model__class_weight": [None, "balanced"],
    },
    "XGBoost": {
        "model__n_estimators": [50, 100, 180],
        "model__max_depth": [2, 3, 4],
        "model__learning_rate": [0.03, 0.1, 0.2],
        "model__reg_lambda": [1, 5],
    },
}


def load_urls(path):
    df = pd.read_csv(path)
    if not {"url", "label"}.issubset(df.columns):
        raise ValueError(f"CSV requires 'url' and 'label' columns, got {list(df.columns)}")
    df = df[["url", "label"]].dropna().copy()
    df["url"] = df["url"].astype(str).str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    df = df[df["url"] != ""]
    if not set(df["label"]).issubset({0, 1}) or df["label"].nunique() != 2:
        raise ValueError("Expected both binary labels: 0=ALLOW and 1=BLOCK")
    if (df.groupby("url")["label"].nunique() > 1).any():
        raise ValueError("The same URL occurs with conflicting labels")
    df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)
    if df["label"].value_counts().min() < 6:
        raise ValueError("Each class needs at least 6 examples for 5-fold CV")
    print(f"Dataset: {len(df)} URLs; label counts {df['label'].value_counts().to_dict()}")
    return df


def make_pipeline(model):
    extractor = FeatureUnion(
        [
            ("characters", TfidfVectorizer(analyzer="char", ngram_range=(3, 5),
                                           sublinear_tf=True)),
            ("tokens", TfidfVectorizer(token_pattern=r"(?u)[a-zA-Z]{2,}",
                                       ngram_range=(1, 2), sublinear_tf=True)),
            ("ad_terms", CountVectorizer(token_pattern=r"(?u)[a-zA-Z]{2,}",
                                         vocabulary=AD_TERMS, binary=True)),
        ],
        transformer_weights={"characters": 1.0, "tokens": 1.0, "ad_terms": 3.0},
    )
    return Pipeline([("features", extractor), ("model", model)])


def make_models(seed):
    return {
        "Random Forest": make_pipeline(
            RandomForestClassifier(n_estimators=120, random_state=seed, n_jobs=1)),
        "Logistic Regression": make_pipeline(
            LogisticRegression(max_iter=2000, random_state=seed)),
        "SVM": make_pipeline(SVC(kernel="linear", probability=True, random_state=seed)),
        "XGBoost": make_pipeline(XGBClassifier(
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", random_state=seed, n_jobs=1)),
    }


def train_and_compare(X_train, y_train, n_iter, seed):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    tuned, rows = {}, []
    for name, pipe in make_models(seed).items():
        print(f"Tuning {name} ...", flush=True)
        search = RandomizedSearchCV(
            pipe, SEARCH_SPACES[name], n_iter=n_iter, cv=folds, scoring="accuracy",
            n_jobs=1, random_state=seed, refit=True, error_score="raise")
        search.fit(X_train, y_train)
        tuned[name] = search.best_estimator_
        rows.append({"model": name, "cv_accuracy": float(search.best_score_),
                     "parameters": json.dumps(search.best_params_)})
        print(f"  CV accuracy: {search.best_score_:.1%}")

    voting = VotingClassifier(
        estimators=[("rf", clone(tuned["Random Forest"])),
                    ("lr", clone(tuned["Logistic Regression"])),
                    ("svm", clone(tuned["SVM"])),
                    ("xgb", clone(tuned["XGBoost"]))],
        voting="soft", n_jobs=1)
    scores = cross_val_score(voting, X_train, y_train, scoring="accuracy",
                             cv=folds, n_jobs=1)
    rows.append({"model": "Soft Voting (all four)", "cv_accuracy": float(scores.mean()),
                 "parameters": "equal-weight soft vote of four tuned Pipelines"})
    cv_table = (pd.DataFrame(rows)
                .sort_values(["cv_accuracy", "model"], ascending=[False, True])
                .reset_index(drop=True))
    winner = cv_table.iloc[0]["model"]
    print("\nTRAINING CV COMPARISON:")
    print(cv_table[["model", "cv_accuracy"]].to_string(index=False))
    print(f"Selected by training CV: {winner}")
    return {**tuned, "Soft Voting (all four)": voting}, cv_table, winner


def evaluate_and_save(X_train, X_test, y_train, y_test, candidates, cv_table,
                      winner, out):
    out.mkdir(parents=True, exist_ok=True)
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    baseline_acc = float(accuracy_score(y_test, baseline.predict(X_test)))
    rows, fitted_winner, preds = [], None, None
    for name, cand in candidates.items():
        fitted = clone(cand).fit(X_train, y_train)
        pred = fitted.predict(X_test)
        rows.append({
            "model": name,
            "test_accuracy": float(accuracy_score(y_test, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
            "block_precision": float(precision_score(y_test, pred, zero_division=0)),
            "block_recall": float(recall_score(y_test, pred, zero_division=0)),
            "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        })
        if name == winner:
            fitted_winner, preds = fitted, pred
    test_table = pd.DataFrame(rows).sort_values("test_accuracy", ascending=False)
    acc = float(accuracy_score(y_test, preds))
    print("\nHOLDOUT COMPARISON (not used for selection):")
    print(test_table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nMajority-class baseline: {baseline_acc:.1%}")
    print(f"CV-selected model: {winner}; test accuracy: {acc:.1%} "
          f"({int((preds == y_test).sum())}/{len(y_test)})")
    print(classification_report(y_test, preds, labels=[0, 1],
                                target_names=["ALLOW", "BLOCK"], digits=3,
                                zero_division=0))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, preds, labels=[0, 1], display_labels=["ALLOW", "BLOCK"],
        cmap="Blues", colorbar=False)
    disp.ax_.set_title(f"CV-selected model: {winner}")
    plt.tight_layout()
    plt.savefig(out / "ad_confusion_matrix.png", dpi=160)
    plt.close()

    cv_table.to_csv(out / "ad_cv_scores.csv", index=False)
    test_table.to_csv(out / "ad_holdout_scores.csv", index=False)
    pd.DataFrame({"url": X_test.to_numpy(), "actual_label": y_test.to_numpy(),
                  "predicted_label": preds}).to_csv(
        out / "ad_holdout_predictions.csv", index=False)
    joblib.dump(fitted_winner, out / "ad_best_pipeline.joblib")
    result = {"chosen_by_training_cv": winner, "heldout_accuracy": acc,
              "heldout_baseline": baseline_acc, "training_rows": len(X_train),
              "test_rows": len(X_test),
              "correct_test_predictions": int((preds == y_test).sum()),
              "note": "Toy example: small dataset; results are not generalizable"}
    (out / "ad_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\nExample classifications (demo only, no live blocking):")
    for url in ["https://ads.example.net/banner/latest.js",
                "https://news.example.org/science",
                "https://tracker.example.com/pixel?id=9"]:
        score = float(fitted_winner.predict_proba([url])[0, 1])
        label = LABELS[int(fitted_winner.predict([url])[0])]
        print(f"  {label:5s} | score={score:.3f} | {url}")
    print(f"Saved results to {out}/")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, help="CSV with url,label columns")
    p.add_argument("--output-dir", default="outputs/ad_blocker")
    p.add_argument("--n-iter", type=int, default=4, help="Random-search configs per model")
    p.add_argument("--test-size", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    df = load_urls(a.csv)
    X_train, X_test, y_train, y_test = train_test_split(
        df["url"], df["label"], test_size=a.test_size, random_state=a.seed,
        stratify=df["label"])
    cands, cv_table, winner = train_and_compare(X_train, y_train, a.n_iter, a.seed)
    evaluate_and_save(X_train, X_test, y_train, y_test, cands, cv_table, winner,
                      Path(a.output_dir))


if __name__ == "__main__":
    main()
