#!/usr/bin/env python3
"""
build_result_analysis.py
=========================
For the deployed master model (classification = 0.770*ML + 0.230*FT-ChemBERTa(pure),
regression = ML alone):

  1. Confusion matrices at threshold=0.5 on holdout (n=3,331) and external
     clinical validation (n=935 labeled).
  2. A trust-filter ablation: what happens to classification performance if
     we exclude compounds that are out-of-applicability-domain (OOD),
     boundary-unstable, low-confidence (LOW tier), all of the above, and
     every combination -- on BOTH datasets.

predict_ml_ensemble.py and predict_ft_chemberta.py (this same directory) score
the external clinical set through the two deployed final-model components;
their outputs, blended 0.770/0.230 exactly like master_predict.py, are what
outputs/submission_extended.csv already contains. The AD/boundary/
confidence-tier trust columns are read straight from that file (external) and
from the repo-root submission_extended.csv (holdout) -- both already carry
them (computed at classifier train time for holdout; recomputed against the
same persisted kNN-Tanimoto cutoff, 0.49408293, for external). This script
does no AD/boundary derivation and no blending of its own -- it only joins
those predictions against ground-truth labels and scores them.
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             matthews_corrcoef, accuracy_score, precision_score, recall_score,
                             confusion_matrix)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent


def clf_metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    return {
        "n": int(len(y)), "ROC_AUC": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
        "PR_AUC": float(average_precision_score(y, p)) if len(set(y)) > 1 else float("nan"),
        "Accuracy": float(accuracy_score(y, yhat)), "Precision": float(precision_score(y, yhat, zero_division=0)),
        "Recall": float(recall_score(y, yhat, zero_division=0)), "F1": float(f1_score(y, yhat)),
        "MCC": float(matthews_corrcoef(y, yhat)) if len(set(y)) > 1 and len(set(yhat)) > 1 else float("nan"),
    }


def main():
    # =========================================================================
    # HOLDOUT -- root submission_extended.csv already carries the deployed
    # blend's probability + AD/boundary/confidence-tier columns; join it
    # against the scaffold-split holdout manifest for the true label.
    # =========================================================================
    hold_pred = pd.read_csv(REPO_ROOT / "submission_extended.csv")
    hold_true = pd.read_csv(REPO_ROOT / "data/holdout_out/holdout_manifest.csv")[
        ["compound_id", "hERG_blocker"]]
    hold = hold_pred.merge(hold_true, on="compound_id").rename(columns={"hERG_blocker": "y_true"})
    hold["p_blend"] = hold["hERG_blocker_probability"]
    print(f"HOLDOUT matched: {len(hold):,}")

    # =========================================================================
    # EXTERNAL -- same idea: outputs/submission_extended.csv already carries
    # the deployed blend's probability + trust columns for the 957-compound
    # clinical set; join against the curated true labels.
    # =========================================================================
    ext_pred = pd.read_csv(HERE / "outputs/submission_extended.csv")
    ext_true = pd.read_csv(REPO_ROOT / "data/clinical_external_validation_set.csv")[
        ["inchikey", "cardiotox_binary"]]
    ext = ext_pred.merge(ext_true, left_on="compound_id", right_on="inchikey")
    ext = ext.dropna(subset=["cardiotox_binary"]).copy()
    ext["cardiotox_binary"] = ext["cardiotox_binary"].astype(int)
    ext["p_blend"] = ext["hERG_blocker_probability"]
    print(f"EXTERNAL matched + labeled: {len(ext):,}")

    datasets = {"holdout": (hold, "y_true"), "external": (ext, "cardiotox_binary")}

    # =========================================================================
    # 1) CONFUSION MATRICES
    # =========================================================================
    for name, (df, ycol) in datasets.items():
        y = df[ycol].to_numpy()
        yhat = (df["p_blend"].to_numpy() >= 0.5).astype(int)
        cm = confusion_matrix(y, yhat, labels=[0, 1])
        cm_df = pd.DataFrame(cm, index=["true_0 (non-blocker)", "true_1 (blocker)"],
                             columns=["pred_0 (non-blocker)", "pred_1 (blocker)"])
        cm_df.to_csv(HERE / f"confusion_matrix_{name}.csv")
        tn, fp, fn, tp = cm.ravel()
        print(f"\n{name.upper()} confusion matrix (n={len(df):,}, threshold=0.5):")
        print(cm_df)
        print(f"  TN={tn} FP={fp} FN={fn} TP={tp}  "
              f"Sensitivity(Recall)={tp/(tp+fn):.4f}  Specificity={tn/(tn+fp):.4f}")

    # =========================================================================
    # 2) TRUST-FILTER ABLATION
    # =========================================================================
    rows = []
    for name, (df, ycol) in datasets.items():
        y_all = df[ycol].to_numpy()
        p_all = df["p_blend"].to_numpy()
        baseline = clf_metrics(y_all, p_all)
        baseline["dataset"] = name; baseline["filter"] = "ALL (no filter, baseline)"
        baseline["n_removed"] = 0
        rows.append(baseline)

        scenarios = {
            "Remove out-of-AD (OOD)": ~(df["in_applicability_domain"] == 1),
            "Remove boundary-unstable": ~(df["boundary_uncertain"] == 0),
            "Remove LOW-confidence-tier": ~(df["confidence_tier"] != "LOW"),
            "Remove OOD + boundary-unstable (= remove all flagged = keep HIGH only)":
                ~((df["in_applicability_domain"] == 1) & (df["boundary_uncertain"] == 0)),
        }
        for label, remove_mask in scenarios.items():
            keep = ~remove_mask
            n_removed = int(remove_mask.sum())
            if keep.sum() < 10 or len(set(y_all[keep.to_numpy()])) < 2:
                m = {"n": int(keep.sum()), "ROC_AUC": float("nan"), "PR_AUC": float("nan"),
                    "Accuracy": float("nan"), "Precision": float("nan"), "Recall": float("nan"),
                    "F1": float("nan"), "MCC": float("nan")}
            else:
                m = clf_metrics(y_all[keep.to_numpy()], p_all[keep.to_numpy()])
            m["dataset"] = name; m["filter"] = label; m["n_removed"] = n_removed
            rows.append(m)

    ablation = pd.DataFrame(rows)[["dataset", "filter", "n", "n_removed", "ROC_AUC", "PR_AUC",
                                   "Accuracy", "Precision", "Recall", "F1", "MCC"]]
    ablation.to_csv(HERE / "trust_filter_ablation.csv", index=False)
    print("\n" + ablation.to_string(index=False))
    print(f"\nwrote -> {HERE}/confusion_matrix_holdout.csv")
    print(f"wrote -> {HERE}/confusion_matrix_external.csv")
    print(f"wrote -> {HERE}/trust_filter_ablation.csv")


if __name__ == "__main__":
    main()
