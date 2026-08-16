#!/usr/bin/env python3
"""
predict_ft_chemberta.py
========================
Score an external SMILES file with the deployed ft_chemberta_pure classifier
(finetune_chemberta_pure.py, checkpoints/cb_ft_pure_results). The full
unfrozen transformer + head were persisted to disk during training
(cb_ft_final.pt + cb_ft_scaler.joblib), so this is a straight load-and-predict
-- no refitting/retraining needed.

Classifier only: the final model's pIC50 regression comes from the
ML-Ensemble regressor alone (see predict_ml_ensemble.py) -- FT-ChemBERTa has
no role in regression, so there is no regressor half here.

Usage
-----
python predict_ft_chemberta.py \
    --external ../data/clinical_external_validation_set.csv --smiles_col smiles \
    --out ft_chemberta_external_predictions.csv
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
CKPT_DIR = HERE.parent / "checkpoints"
CODE_DIR = HERE.parent / "model_code"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--external", required=True)
    ap.add_argument("--smiles_col", default="smiles")
    ap.add_argument("--clf_dir", default=str(CKPT_DIR / "cb_ft_pure_results"))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--col_prefix", default="ft_chemberta_pure",
                    help="Output column prefix.")
    ap.add_argument("--out", default="ft_chemberta_external_predictions.csv")
    a = ap.parse_args()

    sys.path.insert(0, str(CODE_DIR))
    import finetune_chemberta_pure as clf_mod

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ext = pd.read_csv(a.external)
    std = ext[a.smiles_col].map(clf_mod.standardize_smiles)
    ok = std.notna()
    n_bad = int((~ok).sum())
    if n_bad:
        print(f"WARNING: {n_bad} rows failed SMILES standardization -- dropped")
    ext = ext[ok].copy()
    ext["canonical_smiles"] = [s[0] for s in std[ok]]
    smiles = ext["canonical_smiles"].tolist()
    print(f"scoring {len(ext):,} compounds")

    # =========================================================================
    # CLASSIFIER: load the persisted final model + scaler, predict
    # =========================================================================
    print("\n-- classifier: loading persisted cb_ft head --")
    bundle = joblib.load(Path(a.clf_dir) / "cb_ft_scaler.joblib")
    vt, scaler, model_name, extra_dim = (bundle["vt"], bundle["scaler"],
                                          bundle["model_name"], bundle["extra_dim"])
    if vt is None:
        print("  PURE checkpoint (extra_dim=0) -- skipping fingerprint/descriptor computation")
        extra = np.zeros((len(smiles), 0), dtype=np.float32)
    else:
        fp_full = clf_mod.morgan_fp(smiles)
        desc = clf_mod.physchem_descriptors(smiles)
        extra = scaler.transform(np.hstack([vt.transform(fp_full), desc])).astype(np.float32)

    tokenizer = clf_mod.AutoTokenizer.from_pretrained(model_name)
    clf_model = clf_mod.HERG_ChemBERTa_FT(model_name, extra_dim).to(device)
    clf_model.load_state_dict(torch.load(Path(a.clf_dir) / "cb_ft_final.pt", map_location=device))
    clf_loader = DataLoader(
        clf_mod.HERGDataset(smiles, extra, np.zeros(len(smiles))),
        batch_size=a.batch, shuffle=False, collate_fn=clf_mod.make_collate_fn(tokenizer))
    ext[f"{a.col_prefix}_proba"] = clf_mod.predict(clf_model, clf_loader, device)
    print("  classifier scored")

    ext.to_csv(a.out, index=False)
    print(f"\nwrote -> {a.out}  ({len(ext)} rows)")


if __name__ == "__main__":
    main()
