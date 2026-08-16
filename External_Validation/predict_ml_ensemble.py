#!/usr/bin/env python3
"""
predict_ml_ensemble.py
=======================
Score an external SMILES file with the ALREADY-TRAINED ML_Ensemble classifier
(RF/ET/XGB/LGBM/CatBoost stack, calibrated) and pIC50 regressor. Both models
were persisted to .joblib during the original training run, so this is a pure
load-and-predict -- no retraining.

Usage
-----
python predict_ml_ensemble.py \
    --external ../data/clinical_external_validation_set.csv --smiles_col smiles \
    --out ml_ensemble_external_predictions.csv
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CKPT_DIR = HERE.parent / "checkpoints"
CODE_DIR = HERE.parent / "model_code"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--external", required=True)
    ap.add_argument("--smiles_col", default="smiles")
    ap.add_argument("--clf_model", default=str(CKPT_DIR / "herg_combined_dataset_holdout.joblib"))
    ap.add_argument("--reg_model", default=str(CKPT_DIR / "pic50_regressor.joblib"))
    ap.add_argument("--out", default="ml_ensemble_external_predictions.csv")
    a = ap.parse_args()

    sys.path.insert(0, str(CODE_DIR))
    import train_ml_ensemble_classifier as clf_mod
    import ml_pic50_regressor as reg_mod

    df = pd.read_csv(a.external)
    std = df[a.smiles_col].map(clf_mod.standardize_smiles)
    ok = std.notna()
    n_bad = int((~ok).sum())
    if n_bad:
        print(f"WARNING: {n_bad} rows failed SMILES standardization -- dropped")
    df = df[ok].copy()
    df["canonical_smiles"] = [s[0] for s in std[ok]]
    df["inchikey_std"] = [s[1] for s in std[ok]]
    print(f"scoring {len(df):,} compounds")

    # ---- classifier ---------------------------------------------------
    sys.modules["__main__"] = clf_mod   # joblib pickled custom classes as __main__.X
    clf_bundle = joblib.load(a.clf_model)
    Xc, names_c = clf_mod.featurize_smiles(df["canonical_smiles"].tolist(), fp_bits=2048)
    assert names_c == clf_bundle["feature_names"], "classifier feature layout mismatch"
    df["ml_ensemble_proba"] = clf_bundle["model"].predict_proba(Xc)[:, 1]
    print(f"  classifier: {clf_bundle['chosen_strategy']} | calibration={clf_bundle['calibration_enabled']}")

    # ---- regressor ------------------------------------------------------
    sys.modules["__main__"] = reg_mod
    reg_bundle = joblib.load(a.reg_model)
    Xr, names_r = reg_mod.featurize(df["canonical_smiles"].tolist(), fp_bits=reg_bundle.get("fp_bits", 2048))
    assert names_r == reg_bundle["feature_names"], "regressor feature layout mismatch"
    df["ml_ensemble_pred_pic50"] = np.round(reg_bundle["model"].predict(Xr), 4)
    print(f"  regressor: best_family={reg_bundle['best_family']}")

    df.to_csv(a.out, index=False)
    print(f"\nwrote -> {a.out}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
