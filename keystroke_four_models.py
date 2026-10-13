#!/usr/bin/env python3
"""Keystroke dynamics: Random Forest, Logistic Regression, SVM, XGBoost + Soft Voting.

Leakage-safe workflow: stratified hold-out split, Pipelines, 5-fold CV tuning on the
training set only, model selection by training CV, transparent hold-out reporting.

Usage:
    python keystroke_four_models.py --csv keystroke_features.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

TARGET = "user_id"

SEARCH_SPACES = {
    "Random Forest": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 4, 8],
        "model__min_samples_leaf": [1, 3, 5],
    },
    "Logistic Regression": {
        "model__C": [0.01, 0.1, 1.0, 10.0],
        "model__class_weight": [None, "balanced"],
    },
    "SVM": {
        "model__C": [0.1, 1.0, 10.0],
        "model__gamma": ["scale", 0.01, 0.1],
        "model__class_weight": [None, "balanced"],
    },
    "XGBoost": {
        "model__n_estimators": [100, 180, 260],
        "model__max_depth": [2, 3, 5],
        "model__learning_rate": [0.03, 0.1],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
    },
}


def read_keystroke_csv(path):
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"Expected '{TARGET}' column; found {df.columns.tolist()}")
    df = df.dropna(subset=[TARGET]).copy()
    features = df.drop(columns=[TARGET]).select_dtypes(include="number")
    features = features.replace([np.inf, -np.inf], np.nan)
    if features.empty or features.isna().all().any():
        raise ValueError("Need numeric features; no feature may be entirely missing.")
    encoder = LabelEncoder()
    labels = encoder.fit_transform(df[TARGET].astype(str))
    if len(encoder.classes_) < 2:
        raise ValueError("Need two or more users to train a classifier.")
    print(f"Dataset: {len(df)} samples | {features.shape[1]} features | "
          f"{len(encoder.classes_)} users")
    return features, labels, encoder


def make_models(n_classes, seed):
    multi = n_classes > 2
    return {
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(random_state=seed, n_jobs=1)),
        ]),
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=seed)),
        ]),
        "SVM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", probability=True, random_state=seed)),
        ]),
        "XGBoost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(
                objective="multi:softprob" if multi else "binary:logistic",
                eval_metric="mlogloss" if multi else "logloss",
                tree_method="hist", random_state=seed, n_jobs=1)),
        ]),
    }


def train_and_compare(X_train, y_train, n_iter, seed):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    tuned, rows = {}, []
    for name, pipe in make_models(len(np.unique(y_train)), seed).items():
        print(f"Tuning {name} ...", flush=True)
        search = RandomizedSearchCV(
            pipe, SEARCH_SPACES[name], n_iter=n_iter, scoring="accuracy",
            cv=folds, n_jobs=1, random_state=seed, refit=True, error_score="raise")
        search.fit(X_train, y_train)
        tuned[name] = search.best_estimator_
        rows.append({"model": name, "cv_accuracy": float(search.best_score_),
                     "parameters": str(search.best_params_)})
        print(f"  CV accuracy: {search.best_score_:.1%}")

    ensemble = VotingClassifier(
        estimators=[("rf", clone(tuned["Random Forest"])),
                    ("lr", clone(tuned["Logistic Regression"])),
                    ("svm", clone(tuned["SVM"])),
                    ("xgb", clone(tuned["XGBoost"]))],
        voting="soft", n_jobs=1)
    scores = cross_val_score(ensemble, X_train, y_train, cv=folds,
                             scoring="accuracy", n_jobs=1)
    rows.append({"model": "Soft Voting (all four)", "cv_accuracy": float(scores.mean()),
                 "parameters": "Four tuned Pipelines; equal soft-vote weights"})
    cv_table = (pd.DataFrame(rows).sort_values("cv_accuracy", ascending=False)
                .reset_index(drop=True))
    winner = cv_table.iloc[0]["model"]
    print("\nTRAINING CV COMPARISON:")
    print(cv_table[["model", "cv_accuracy"]].to_string(index=False))
    print(f"Selected by training CV: {winner}")
    return {**tuned, "Soft Voting (all four)": ensemble}, cv_table, winner


def evaluate(X_train, X_test, y_train, y_test, encoder, candidates, cv_table,
             winner, out):
    out.mkdir(parents=True, exist_ok=True)
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    baseline_acc = accuracy_score(y_test, baseline.predict(X_test))
    rows, selected, selected_pred = [], None, None
    for name, cand in candidates.items():
        fitted = clone(cand).fit(X_train, y_train)
        pred = fitted.predict(X_test)
        rows.append({
            "model": name,
            "test_accuracy": accuracy_score(y_test, pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, pred),
            "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0)})
        if name == winner:
            selected, selected_pred = fitted, pred
    test_table = pd.DataFrame(rows).sort_values("test_accuracy", ascending=False)
    acc = float(accuracy_score(y_test, selected_pred))
    print(f"\nMajority-class baseline: {baseline_acc:.1%}")
    print("HOLDOUT PERFORMANCE (reporting only):")
    print(test_table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nCV-selected model: {winner} | hold-out accuracy: {acc:.1%}")
    print(classification_report(
        y_test, selected_pred, labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_, zero_division=0))

    ConfusionMatrixDisplay.from_predictions(
        encoder.inverse_transform(y_test), encoder.inverse_transform(selected_pred),
        labels=encoder.classes_, xticks_rotation=45, cmap="Blues", colorbar=False)
    plt.title(f"CV-selected model: {winner}")
    plt.tight_layout()
    plt.savefig(out / "keystroke_confusion_matrix.png", dpi=160)
    plt.close()

    cv_table.to_csv(out / "keystroke_cv_scores.csv", index=False)
    test_table.to_csv(out / "keystroke_holdout_scores.csv", index=False)
    joblib.dump({"pipeline": selected, "label_encoder": encoder,
                 "features": list(X_train.columns)},
                out / "keystroke_best_pipeline.joblib")
    result = {"selected_by_training_cv": winner, "heldout_test_accuracy": acc,
              "majority_holdout_baseline": float(baseline_acc),
              "training_rows": len(X_train), "test_rows": len(X_test),
              "user_classes": encoder.classes_.tolist()}
    (out / "keystroke_metrics.json").write_text(json.dumps(result, indent=2),
                                                encoding="utf-8")
    print(f"Saved results to {out}/")
    print("Note: this dataset may NOT support reliable user identification.")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, help="Keystroke CSV with a user_id column")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--n-iter", type=int, default=4, help="Random-search configs per model")
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    X, y, enc = read_keystroke_csv(a.csv)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=a.test_size, random_state=a.seed, stratify=y)
    cands, cv_table, winner = train_and_compare(X_train, y_train, a.n_iter, a.seed)
    evaluate(X_train, X_test, y_train, y_test, enc, cands, cv_table, winner,
             Path(a.output_dir))


if __name__ == "__main__":
    main()
