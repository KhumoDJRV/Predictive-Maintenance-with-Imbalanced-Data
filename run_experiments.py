"""
Predictive Maintenance with Imbalanced Data (AI4I 2020, SYNTHETIC dataset)
Usage:  python run_experiments.py --data data/ai4i2020.csv --out results

Dataset: Matzka, S. (2020). AI4I 2020 Predictive Maintenance Dataset.
         UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C
         (synthetic data -- results do NOT describe real machines)

Pipeline
  1. load + audit (class balance, missing values, duplicates)
  2. drop identifiers and the five failure-mode columns (TWF/HDF/PWF/OSF/RNF):
     they are components of the target, so using them is label leakage
  3. stratified 60/20/20 train/validation/test split (fixed seed)
  4. models: dummy, logistic regression, random forest, gradient boosting,
     each with and without class-imbalance handling
  5. decision threshold chosen on VALIDATION only (max F1), then applied once to TEST
  6. report PR AUC, recall, precision, F1, confusion matrix; save tables + figures
"""
import argparse
import json
import os
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

SEED = 42
TARGET = "Machine failure"
LEAK_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]      # failure modes = target components
ID_COLS = ["UDI", "Product ID"]
NUM = ["Air temperature [K]", "Process temperature [K]",
       "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
CAT = ["Type"]


def load(path):
    df = pd.read_csv(path)
    missing = [c for c in NUM + CAT + [TARGET] if c not in df.columns]
    if missing:
        raise SystemExit(f"Columns not found: {missing}. Found: {list(df.columns)}")
    return df


def audit(df, out):
    info = {
        "rows": int(len(df)),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows_excl_ids": int(df.drop(columns=[c for c in ID_COLS if c in df]).duplicated().sum()),
        "failures": int(df[TARGET].sum()),
        "failure_rate_pct": round(100 * df[TARGET].mean(), 3),
    }
    for c in LEAK_COLS:
        if c in df:
            info[f"mode_{c}"] = int(df[c].sum())
    if all(c in df for c in LEAK_COLS):
        # rows labelled failure but with no mode flag (known quirk of RNF / data)
        info["failure_without_any_mode_flag"] = int(((df[TARGET] == 1) & (df[LEAK_COLS].sum(axis=1) == 0)).sum())
    json.dump(info, open(f"{out}/data_audit.json", "w"), indent=2)
    print("AUDIT", info)

    ax = df[TARGET].value_counts().sort_index().plot.bar(color=["#4c72b0", "#c44e52"])
    ax.set_xticklabels(["No failure", "Failure"], rotation=0)
    ax.set_ylabel("Count"); ax.set_title("Target distribution")
    for i, v in enumerate(df[TARGET].value_counts().sort_index()):
        ax.text(i, v, str(v), ha="center", va="bottom")
    plt.tight_layout(); plt.savefig(f"{out}/fig_class_balance.png", dpi=200); plt.close()


def make_pre(scale):
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([("num", num, NUM), ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])


def build_models():
    """name -> (pipeline, uses_sample_weight)"""
    m = {}
    m["Dummy (prior)"] = (Pipeline([("pre", make_pre(False)), ("clf", DummyClassifier(strategy="prior"))]), False)
    m["LogReg"] = (Pipeline([("pre", make_pre(True)),
                             ("clf", LogisticRegression(max_iter=2000, random_state=SEED))]), False)
    m["LogReg + class weight"] = (Pipeline([("pre", make_pre(True)),
                             ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED))]), False)
    m["RandomForest"] = (Pipeline([("pre", make_pre(False)),
                             ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=SEED))]), False)
    m["RandomForest + class weight"] = (Pipeline([("pre", make_pre(False)),
                             ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED))]), False)
    m["HistGB"] = (Pipeline([("pre", make_pre(False)),
                             ("clf", HistGradientBoostingClassifier(random_state=SEED))]), False)
    m["HistGB + sample weight"] = (Pipeline([("pre", make_pre(False)),
                             ("clf", HistGradientBoostingClassifier(random_state=SEED))]), True)
    return m


def fit(pipe, X, y, use_sw):
    if use_sw:
        pipe.fit(X, y, clf__sample_weight=compute_sample_weight("balanced", y))
    else:
        pipe.fit(X, y)
    return pipe


def best_threshold(y_val, p_val):
    """Threshold maximizing F1 on the validation set (never looks at test)."""
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[np.argmax(f1)]) if len(thr) else 0.5


def metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"PR_AUC": average_precision_score(y, p),
            "Recall": recall_score(y, pred, zero_division=0),
            "Precision": precision_score(y, pred, zero_division=0),
            "F1": f1_score(y, pred, zero_division=0),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ai4i2020.csv")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    df = load(a.data)
    audit(df, a.out)
    X = df[NUM + CAT]
    y = df[TARGET].astype(int)

    # 60/20/20 stratified split
    X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=SEED)
    split = {n: {"n": int(len(v)), "failures": int(v.sum())} for n, v in
             [("train", y_tr), ("val", y_va), ("test", y_te)]}
    json.dump(split, open(f"{a.out}/split_sizes.json", "w"), indent=2)
    print("SPLIT", split)

    rows, fitted, test_probs = [], {}, {}
    for name, (pipe, sw) in build_models().items():
        fit(pipe, X_tr, y_tr, sw)
        p_va = pipe.predict_proba(X_va)[:, 1]
        p_te = pipe.predict_proba(X_te)[:, 1]
        thr = best_threshold(y_va, p_va) if name != "Dummy (prior)" else 0.5
        mv, mt = metrics(y_va, p_va, thr), metrics(y_te, p_te, thr)
        rows.append({"Model": name, "Threshold(val)": round(thr, 3),
                     "Val PR_AUC": mv["PR_AUC"], "Val F1": mv["F1"],
                     **{f"Test {k}": v for k, v in mt.items()}})
        fitted[name], test_probs[name] = pipe, p_te
        print(f"{name:32s} valPR={mv['PR_AUC']:.3f} testPR={mt['PR_AUC']:.3f} "
              f"testRecall={mt['Recall']:.3f} testPrec={mt['Precision']:.3f}")

    res = pd.DataFrame(rows).round(4)
    res.to_csv(f"{a.out}/model_comparison.csv", index=False)
    print(res.to_string(index=False))

    # --- figures ---
    plt.figure(figsize=(6, 4.5))
    for n, p in test_probs.items():
        pr, rc, _ = precision_recall_curve(y_te, p)
        plt.plot(rc, pr, label=f"{n} (AP={average_precision_score(y_te, p):.2f})")
    plt.axhline(y_te.mean(), ls="--", c="gray", label="No-skill")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-recall curves (test set)")
    plt.legend(fontsize=6); plt.tight_layout(); plt.savefig(f"{a.out}/fig_pr_curves.png", dpi=200); plt.close()

    # best model chosen by VALIDATION PR AUC (not test)
    best = res[res["Model"] != "Dummy (prior)"].sort_values("Val PR_AUC", ascending=False).iloc[0]
    bname, bthr = best["Model"], best["Threshold(val)"]
    print("BEST (by validation PR AUC):", bname)
    pred = (test_probs[bname] >= bthr).astype(int)
    ConfusionMatrixDisplay(confusion_matrix(y_te, pred, labels=[0, 1]),
                           display_labels=["No failure", "Failure"]).plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion matrix - {bname}\n(test, threshold {bthr:.2f} from validation)")
    plt.tight_layout(); plt.savefig(f"{a.out}/fig_confusion_matrix.png", dpi=200); plt.close()

    pi = permutation_importance(fitted[bname], X_te, y_te, scoring="average_precision",
                                n_repeats=20, random_state=SEED, n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=X_te.columns).sort_values()
    imp.plot.barh(xerr=pd.Series(pi.importances_std, index=X_te.columns)[imp.index])
    plt.xlabel("Drop in PR AUC when shuffled"); plt.title(f"Permutation importance - {bname}")
    plt.tight_layout(); plt.savefig(f"{a.out}/fig_permutation_importance.png", dpi=200); plt.close()

    # error analysis table (false negatives) for the report
    err = X_te.copy(); err["y_true"] = y_te.values; err["proba"] = test_probs[bname]; err["pred"] = pred
    err[(err.y_true == 1) & (err.pred == 0)].to_csv(f"{a.out}/false_negatives.csv", index=False)

    env = {"python": platform.python_version(), "sklearn": sklearn.__version__,
           "pandas": pd.__version__, "numpy": np.__version__, "seed": SEED,
           "cpu": platform.processor() or platform.machine(), "runtime_sec": round(time.time() - t0, 1)}
    json.dump(env, open(f"{a.out}/environment.json", "w"), indent=2)
    print("ENV", env)


if __name__ == "__main__":
    main()
