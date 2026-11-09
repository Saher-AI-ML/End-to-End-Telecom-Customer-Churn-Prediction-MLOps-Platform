"""Train the WE churn model end to end and save the artifacts the API will load.

Run from the repo root:
    python src/train.py
    python src/train.py --data data/WE_Churn_Modeling_Dataset.csv --n-iter 20

Outputs (in --out-dir, default models/):
    churn_model.joblib  full fitted Pipeline (preprocessing + model), accepts raw columns
    metadata.json       features, categories, threshold, library versions, ...
    metrics.json        CV comparison, search results, threshold, final test metrics
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from xgboost import XGBClassifier

TARGET = "churned"
ID_COL = "customer_id"  # key only, never a feature
NUM_COLS = ["monthly_bill_egp", "tenure_months", "complaint_count", "downtime_hours_monthly"]
CAT_COLS = ["governorate", "customer_type", "subscription_plan", "payment_method"]
FEATURES = NUM_COLS + CAT_COLS


def build_pipeline(model, scale=False):
    """Preprocessing + model in ONE Pipeline, so imputers/encoders are fitted on training folds only."""
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:  # only scale-sensitive models (logistic regression) need this
        num_steps.append(("scale", RobustScaler()))
    pre = ColumnTransformer([
        ("num", Pipeline(num_steps), NUM_COLS),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT_COLS),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def load_data(path):
    df = pd.read_csv(path)
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)  
    print(f"Loaded {before} rows, removed {before - len(df)} exact duplicates -> {len(df)} rows")
    repeated = df[ID_COL].duplicated(keep=False)
    if repeated.any():
        print(f"{df.loc[repeated, ID_COL].nunique()} customer_ids appear more than once with different values "
              f"-> dropping all {int(repeated.sum())} of those rows")
        df = df[~repeated].reset_index(drop=True)
    assert df[ID_COL].is_unique, "customer_id is still not unique"
    print(f"Rows used for modeling: {len(df)}")
    return df


def candidate_models(seed):
    return {
        "Dummy (most frequent)": build_pipeline(DummyClassifier(strategy="prior")),
        "Logistic Regression": build_pipeline(
            LogisticRegression(max_iter=2000, class_weight="balanced"), scale=True),
        "Random Forest": build_pipeline(
            RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=1)),
        "XGBoost": build_pipeline(
            XGBClassifier(eval_metric="logloss", random_state=seed, n_jobs=1)),
    }


SEARCH_SPACES = {
    "Random Forest": {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [None, 6, 10, 16],
        "model__min_samples_leaf": [1, 3, 5, 10],
        "model__max_features": ["sqrt", 0.5],
        "model__class_weight": [None, "balanced"],
    },
    "XGBoost": {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [3, 4, 5, 6],
        "model__learning_rate": [0.03, 0.05, 0.1, 0.2],
        "model__subsample": [0.7, 0.85, 1.0],
        "model__colsample_bytree": [0.6, 0.8, 1.0],
        "model__min_child_weight": [1, 3, 5],
        "model__reg_lambda": [1, 5, 10],
    },
}
SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "recall": "recall", "f1": "f1",
           "accuracy": "accuracy"}


def metrics_at(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "roc_auc": roc_auc_score(y_true, proba), "pr_auc": average_precision_score(y_true, proba),
        "accuracy": accuracy_score(y_true, pred), "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred), "f1": f1_score(y_true, pred),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def best_f1_threshold(y_true, proba):
    p, r, t = precision_recall_curve(y_true, proba)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(t[int(np.nanargmax(f1))])


def to_native(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


HANDOFF_CODE = r"""
import json, sys, joblib, numpy as np, pandas as pd
meta = json.load(open(sys.argv[1])); model = joblib.load(sys.argv[2]); rows = json.loads(sys.argv[3])
num, cat = meta['features']['numeric'], meta['features']['categorical']
def frame(rec): 
    df = pd.DataFrame([{c: rec.get(c) for c in num + cat}])
    for c in num: df[c] = pd.to_numeric(df[c], errors='coerce')
    for c in cat: df[c] = df[c].astype(object).where(df[c].notna(), np.nan)
    return df
for label, rec in rows.items():
    p = model.predict_proba(frame(rec))
    assert p.shape == (1, 2), p.shape
    prob = float(p[0, 1]); assert 0.0 <= prob <= 1.0
    print(f'  {label:<18} churn_probability={prob:.4f}  will_churn={prob >= meta["threshold"]}')
print('HANDOFF OK')
"""


def handoff_test(out_dir, sample_row):
    """Load the saved model in a FRESH Python process and predict on one-row DataFrames."""
    typical = {k: (None if pd.isna(v) else v) for k, v in sample_row.items()}
    missing = dict(typical, tenure_months=None, payment_method=None, downtime_hours_monthly=None)
    unseen = dict(typical, governorate="Atlantis")
    rows = {"typical row": typical, "missing values": missing, "unseen category": unseen}
    res = subprocess.run([sys.executable, "-I", "-c", HANDOFF_CODE, str(out_dir / "metadata.json"),
                          str(out_dir / "churn_model.joblib"), json.dumps(rows, default=to_native)],
                         capture_output=True, text=True)
    print(res.stdout, end="")
    if res.returncode != 0:
        print(res.stderr)
        raise SystemExit("Handoff test FAILED: the saved model does not work in a fresh session")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/WE_Churn_Modeling_Dataset.csv")
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--n-iter", type=int, default=20, help="RandomizedSearchCV iterations per model")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(a.data)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=a.seed, stratify=y)
    print(f"Train {X_train.shape}, test {X_test.shape}, churn rate train {y_train.mean():.3f}")
    cv = StratifiedKFold(a.folds, shuffle=True, random_state=a.seed)

    print(f"\n== {a.folds}-fold stratified CV ==")
    cv_results = {}
    for name, pipe in candidate_models(a.seed).items():
        r = cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1)
        cv_results[name] = {m: {"mean": float(r[f"test_{m}"].mean()), "std": float(r[f"test_{m}"].std())}
                            for m in SCORING}
        print(f"{name:<24}" + "  ".join(f"{m}={cv_results[name][m]['mean']:.3f}" for m in SCORING))

    print("\n== RandomizedSearchCV (scoring = PR-AUC) ==")
    base = candidate_models(a.seed)
    searches = {}
    for name, space in SEARCH_SPACES.items():
        s = RandomizedSearchCV(base[name], space, n_iter=a.n_iter, scoring="average_precision", cv=cv,
                               n_jobs=-1, random_state=a.seed, refit=True)
        s.fit(X_train, y_train)
        searches[name] = s
        print(f"{name:<14} best CV PR-AUC={s.best_score_:.4f}  params={s.best_params_}")
    best_name = max(searches, key=lambda n: searches[n].best_score_)
    best = searches[best_name]
    print(f"Chosen model: {best_name}")

    oof = cross_val_predict(clone(best.best_estimator_), X_train, y_train, cv=cv,
                            method="predict_proba", n_jobs=-1)[:, 1]
    threshold = best_f1_threshold(y_train, oof)
    cv_at_thr = metrics_at(y_train, oof, threshold)
    print(f"\nThreshold (max F1 on CV predictions) = {threshold:.3f}  "
          f"-> CV recall={cv_at_thr['recall']:.3f} precision={cv_at_thr['precision']:.3f} f1={cv_at_thr['f1']:.3f}")
    model = best.best_estimator_  
    test = metrics_at(y_test, model.predict_proba(X_test)[:, 1], threshold)
    print("\n== Test set (used once) ==")
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in test.items()})

    # 5) Save artifacts
    oh = model.named_steps["pre"].named_transformers_["cat"].named_steps["oh"]
    metadata = {
        "model_name": best_name, "target": TARGET, "id_column_not_a_feature": ID_COL,
        "features": {"numeric": NUM_COLS, "categorical": CAT_COLS}, "feature_order": FEATURES,
        "categories": {c: [str(v) for v in cats] for c, cats in zip(CAT_COLS, oh.categories_)},
        "numeric_ranges": {c: {"min": float(X_train[c].min()), "max": float(X_train[c].max())} for c in NUM_COLS},
        "threshold": threshold, "best_params": best.best_params_, "random_state": a.seed,
        "n_train": int(len(X_train)), "n_test": int(len(X_test)), "churn_rate_train": float(y_train.mean()),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "library_versions": {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__,
                             "xgboost": xgboost.__version__, "pandas": pd.__version__,
                             "numpy": np.__version__, "joblib": joblib.__version__},
    }
    metrics = {"cv_comparison": cv_results,
               "search": {n: {"best_cv_pr_auc": float(s.best_score_), "best_params": s.best_params_}
                          for n, s in searches.items()},
               "chosen_model": best_name, "threshold": threshold, "cv_at_threshold": cv_at_thr, "test": test}
    joblib.dump(model, out_dir / "churn_model.joblib")
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=to_native))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=to_native))
    print(f"\nSaved churn_model.joblib, metadata.json, metrics.json to {out_dir}/")

    print("\n== Handoff test (fresh process) ==")
    handoff_test(out_dir, X_test.iloc[0].to_dict())


if __name__ == "__main__":
    main()
