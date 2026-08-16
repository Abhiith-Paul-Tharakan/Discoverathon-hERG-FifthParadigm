#!/usr/bin/env python3
"""
assemble_submission.py

Build the Discoverathon 2026 Challenge 2 (hERG) submission by joining the
BINARY classifier's held-out predictions with the pIC50 regressor's held-out
predictions on the *shared* 20% test set (the --holdout_ids contract that both
models honour), then attaching the applicability-domain and boundary-
instability trust flags the classifier already produced per compound.

Two output files are written:

  submission.csv           EXACT rubric schema -- three columns, in order:
                             compound_id, hERG_blocker_probability, predicted hERG pIC50
                           This is the file you submit for grading.

  submission_extended.csv  The same three columns PLUS trust annotations:
                             in_applicability_domain   1 = inside the classifier's kNN-Tanimoto
                                                       applicability domain (structurally close to
                                                       training data); 0 = extrapolation.
                             ad_knn_mean_sim           mean top-k Tanimoto similarity to train set.
                             boundary_uncertain        1 = calibrated blocker probability sits in the
                                                       ambiguity band around the decision threshold
                                                       (low-margin, unstable class call); else 0.
                             pred_confidence_margin    |2*p - 1|, distance of the blocker prob from 0.5.
                             confidence_tier           HIGH   = in AD and not boundary-uncertain
                                                       MEDIUM = exactly one of those holds
                                                       LOW    = outside the applicability domain
                           Cite this file in the report/slides -- it is exactly the
                           "bound your conclusions to what the data supports" evidence
                           the Responsible-AI rubric rewards.

Why two files: the rubric's example is a minimal 3-column CSV. A name-based reader
ignores extra columns, but a strict positional grader would not -- so the graded
file is kept exactly to spec and the trust columns live alongside it.

Usage
-----
python assemble_submission.py \
    --clf_pred result_analysis/ml_ensemble_holdout_predictions/herg_combined_dataset_holdout_predictions.csv \
    --reg_pred result_analysis/ml_ensemble_holdout_predictions/heldout_predictions.csv \
    --out_dir  submission_out

Column names are auto-detected with sensible defaults; override with
--id_col / --prob_col / --pic50_col if your files differ. The script prints
exactly which columns it used and how many compounds matched across the two
models, so you can eyeball the join before submitting.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Column resolution helpers
# ---------------------------------------------------------------------------

def pick_id_column(df: pd.DataFrame, preferred: Optional[str]) -> str:
    """Choose the compound-id column: the preferred name if present, else a
    known fallback (inchikey / compound_chembl_id / compound_id)."""
    candidates = [preferred, "compound_chembl_id", "compound_id", "inchikey", "molecule_chembl_id"]
    for c in candidates:
        if c and c in df.columns:
            return c
    raise SystemExit(
        f"ERROR: could not find a compound-id column. Looked for {candidates}; "
        f"file has columns: {list(df.columns)}. Pass --id_col explicitly."
    )


def pick_prob_column(df: pd.DataFrame, preferred: Optional[str]) -> str:
    """Choose the blocker-probability column. The classifier writes the
    calibrated positive-class probability as 'proba_cal_1' (positive class =
    blocker = 1); fall back to the uncalibrated one if calibration was off."""
    for c in [preferred, "proba_cal_1", "hERG_blocker_probability",
              "blocker_probability", "proba_uncal_1", "y_proba_1"]:
        if c and c in df.columns:
            return c
    # last resort: any single 'proba_cal_*'/'proba_*' column for the max class id
    cal = sorted([c for c in df.columns if c.startswith("proba_cal_")])
    if cal:
        return cal[-1]
    unc = sorted([c for c in df.columns if c.startswith("proba_uncal_")])
    if unc:
        return unc[-1]
    raise SystemExit(
        f"ERROR: could not find a blocker-probability column in the classifier "
        f"file. Columns: {list(df.columns)}. Pass --prob_col explicitly."
    )


def pick_pic50_column(df: pd.DataFrame, preferred: Optional[str]) -> str:
    """Choose the PREDICTED pIC50 column from the regressor file, being careful
    NOT to grab the ground-truth column (y_true / pIC50). Prioritised search:
    explicit '*pred*pic50*' names, then generic prediction names."""
    if preferred and preferred in df.columns:
        return preferred
    lower = {c.lower(): c for c in df.columns}
    # 1) columns mentioning BOTH prediction and pic50
    for lc, orig in lower.items():
        if "pic50" in lc and ("pred" in lc or "hat" in lc):
            return orig
    # 2) explicit predicted-pIC50 aliases
    for name in ["predicted_pic50", "pred_pic50", "pic50_pred", "yhat", "y_hat",
                 "y_pred", "prediction", "pred", "predicted", "yhat_pic50"]:
        if name in lower:
            return lower[name]
    # 3) a single 'pic50'-like column that is clearly not the truth column
    truthy = {"pic50", "y_true", "true_pic50", "target", "label", "actual"}
    pic_like = [orig for lc, orig in lower.items()
                if "pic50" in lc and lc not in truthy]
    if len(pic_like) == 1:
        return pic_like[0]
    raise SystemExit(
        "ERROR: could not confidently identify the PREDICTED pIC50 column in the "
        f"regressor file. Columns: {list(df.columns)}. Pass --pic50_col explicitly "
        "so we don't accidentally use the ground-truth column."
    )


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def confidence_tier(in_ad: pd.Series, boundary_uncertain: pd.Series) -> pd.Series:
    """HIGH  : inside AD and not boundary-uncertain
       MEDIUM: exactly one of {inside AD, low-margin} holds
       LOW   : outside the applicability domain (extrapolation)."""
    in_ad = in_ad.fillna(0).astype(int)
    bu = boundary_uncertain.fillna(0).astype(int)
    tier = np.where(
        in_ad == 0, "LOW",
        np.where(bu == 0, "HIGH", "MEDIUM"),
    )
    return pd.Series(tier, index=in_ad.index)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clf_pred", required=True,
                    help="Classifier holdout predictions CSV "
                         "(predictions/<target>_holdout_predictions.csv).")
    ap.add_argument("--reg_pred", required=True,
                    help="Regressor holdout predictions CSV (same shared holdout).")
    ap.add_argument("--out_dir", default="submission_out")
    ap.add_argument("--id_col", default="compound_chembl_id",
                    help="Compound-id column to join on (auto-fallback to inchikey).")
    ap.add_argument("--prob_col", default=None,
                    help="Blocker-probability column in the classifier file "
                         "(default: auto -> proba_cal_1).")
    ap.add_argument("--pic50_col", default=None,
                    help="PREDICTED pIC50 column in the regressor file (default: auto-detect).")
    ap.add_argument("--id_out_name", default="compound_id",
                    help="Name of the id column in the written submission.")
    ap.add_argument("--prob_round", type=int, default=4)
    ap.add_argument("--pic50_round", type=int, default=3)
    args = ap.parse_args()

    clf = pd.read_csv(args.clf_pred)
    reg = pd.read_csv(args.reg_pred)

    clf_id = pick_id_column(clf, args.id_col)
    reg_id = pick_id_column(reg, args.id_col)
    prob_col = pick_prob_column(clf, args.prob_col)
    pic50_col = pick_pic50_column(reg, args.pic50_col)

    print(f"[assemble] classifier id column : {clf_id}")
    print(f"[assemble] blocker-prob column  : {prob_col}")
    print(f"[assemble] regressor id column  : {reg_id}")
    print(f"[assemble] predicted-pIC50 col  : {pic50_col}")

    # Build the two sides, normalising the id to a common name.
    clf_cols = {clf_id: "__id__", prob_col: "hERG_blocker_probability"}
    for opt in ["in_applicability_domain", "ad_knn_mean_sim",
                "boundary_uncertain", "pred_confidence_margin"]:
        if opt in clf.columns:
            clf_cols[opt] = opt
        else:
            print(f"[assemble] NOTE: '{opt}' not in classifier file; it will be blank.")
    clf_side = clf[list(clf_cols)].rename(columns=clf_cols)

    reg_side = reg[[reg_id, pic50_col]].rename(
        columns={reg_id: "__id__", pic50_col: "predicted hERG pIC50"}
    )

    # De-duplicate ids (holdout should already be unique per compound).
    for name, side in (("classifier", clf_side), ("regressor", reg_side)):
        dup = int(side["__id__"].duplicated().sum())
        if dup:
            print(f"[assemble] WARNING: {dup} duplicate ids in {name} file; keeping first.")
    clf_side = clf_side.drop_duplicates("__id__", keep="first")
    reg_side = reg_side.drop_duplicates("__id__", keep="first")

    merged = clf_side.merge(reg_side, on="__id__", how="inner")

    n_clf, n_reg, n_join = len(clf_side), len(reg_side), len(merged)
    print(f"[assemble] classifier holdout compounds: {n_clf}")
    print(f"[assemble] regressor  holdout compounds: {n_reg}")
    print(f"[assemble] matched (inner join)        : {n_join}")
    only_clf = set(clf_side["__id__"]) - set(reg_side["__id__"])
    only_reg = set(reg_side["__id__"]) - set(clf_side["__id__"])
    if only_clf:
        print(f"[assemble] WARNING: {len(only_clf)} compounds only in classifier "
              f"(no pIC50) -- excluded from submission. e.g. {list(only_clf)[:3]}")
    if only_reg:
        print(f"[assemble] WARNING: {len(only_reg)} compounds only in regressor "
              f"(no blocker prob) -- excluded. e.g. {list(only_reg)[:3]}")
    if n_join == 0:
        raise SystemExit(
            "ERROR: zero compounds matched across the two files. Are both models "
            "using the SAME --holdout_ids? Check the id columns "
            f"({clf_id} vs {reg_id})."
        )

    # Round for a clean submission.
    merged["hERG_blocker_probability"] = merged["hERG_blocker_probability"].astype(float).round(args.prob_round)
    merged["predicted hERG pIC50"] = merged["predicted hERG pIC50"].astype(float).round(args.pic50_round)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- canonical 3-column submission (exact rubric schema) ---
    canonical = merged[["__id__", "hERG_blocker_probability", "predicted hERG pIC50"]].copy()
    canonical = canonical.rename(columns={"__id__": args.id_out_name})
    canonical_path = out_dir / "submission.csv"
    canonical.to_csv(canonical_path, index=False)

    # --- extended submission with AD + boundary trust flags ---
    ext = canonical.copy()
    if "in_applicability_domain" in merged.columns:
        ext["in_applicability_domain"] = merged["in_applicability_domain"].astype("Int64")
    if "ad_knn_mean_sim" in merged.columns:
        ext["ad_knn_mean_sim"] = merged["ad_knn_mean_sim"].astype(float).round(4)
    if "boundary_uncertain" in merged.columns:
        ext["boundary_uncertain"] = merged["boundary_uncertain"].astype("Int64")
    if "pred_confidence_margin" in merged.columns:
        ext["pred_confidence_margin"] = merged["pred_confidence_margin"].astype(float).round(4)
    if {"in_applicability_domain", "boundary_uncertain"}.issubset(merged.columns):
        ext["confidence_tier"] = confidence_tier(
            merged["in_applicability_domain"], merged["boundary_uncertain"],
        ).values
    ext_path = out_dir / "submission_extended.csv"
    ext.to_csv(ext_path, index=False)

    # --- quick console summary the report can quote ---
    print(f"\n[assemble] wrote {canonical_path}  ({len(canonical)} rows, 3 columns)")
    print(f"[assemble] wrote {ext_path}  ({len(ext.columns)} columns)")
    if "confidence_tier" in ext.columns:
        tier_counts = ext["confidence_tier"].value_counts().to_dict()
        print(f"[assemble] confidence tiers: {tier_counts}")
    if "in_applicability_domain" in ext.columns:
        frac = float(pd.to_numeric(ext["in_applicability_domain"], errors="coerce").mean())
        print(f"[assemble] fraction inside applicability domain: {frac:.3f}")


if __name__ == "__main__":
    main()