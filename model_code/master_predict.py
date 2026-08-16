#!/usr/bin/env python3
"""
master_predict.py
==================
The final wired hERG model: one SMILES in, two predictions out, each routed
through whichever component actually won its own evaluation.

  CLASSIFICATION (hERG blocker probability)
      P = 0.770 * P_ML_Ensemble + 0.230 * P_FT-ChemBERTa(pure)
      -- the "ML+FT" weighted blend, weights optimized on out-of-fold ROC-AUC
         by simplex grid search over every pairwise/3-way combination of the
         ML/GNN/FT-ChemBERTa models during model selection, then frozen. This
         combination was the best/tied-best classifier on internal holdout AND
         the external clinical set. See README.md for the held-out numbers.

  REGRESSION (predicted pIC50)
      pIC50 = ML_Ensemble regressor, ALONE
      -- every blend that touched the regressor made holdout RMSE/R2 worse
         (ML_Ensemble's regressor has no OOF, so it can't be leakage-safely
         weighted down or up -- averaging it with weaker regressors only
         dilutes it). Confirmed the best regressor on internal holdout (real
         RMSE/R2 against measured pIC50, the metric that matters most).

FT-ChemBERTa here is the PURE checkpoint (checkpoints/cb_ft_pure_results,
extra_dim=0, no ECFP4/physchem fusion) -- never conflate it with any
frozen-embedding ChemBERTa variant. ML_Ensemble is a self-contained joblib
load-and-predict (no retraining/replay needed); the FT-ChemBERTa checkpoint is
a full fine-tuned transformer + head, also a straight load-and-predict.

Usage
-----
# single ad-hoc SMILES, printed to stdout
python master_predict.py --smiles "CCN(CC)CCNC(=O)c1ccc(N)cc1"

# batch mode over a CSV
python master_predict.py --input compounds.csv --smiles_col smiles \
    --id_col compound_id --out predictions.csv
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

W_ML, W_FT = 0.770, 0.230   # frozen classification blend weights (OOF-optimized, see docstring)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smiles", default=None, help="Single SMILES string for an ad-hoc query.")
    ap.add_argument("--input", default=None, help="CSV of compounds to score in batch (alternative to --smiles).")
    ap.add_argument("--smiles_col", default="smiles")
    ap.add_argument("--id_col", default="compound_id")
    ap.add_argument("--clf_model", default=str(CKPT_DIR / "herg_combined_dataset_holdout.joblib"))
    ap.add_argument("--reg_model", default=str(CKPT_DIR / "pic50_regressor.joblib"))
    ap.add_argument("--ft_clf_dir", default=str(CKPT_DIR / "cb_ft_pure_results"))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="master_predictions.csv")
    a = ap.parse_args()

    if not a.smiles and not a.input:
        raise SystemExit("pass either --smiles \"<SMILES>\" or --input compounds.csv")

    # ---- build the input frame ---------------------------------------------
    if a.smiles:
        df = pd.DataFrame({a.id_col: ["query_1"], a.smiles_col: [a.smiles]})
    else:
        df = pd.read_csv(a.input)
        if a.id_col not in df.columns:
            df[a.id_col] = [f"row_{i}" for i in range(len(df))]

    sys.path.insert(0, str(HERE))
    import train_ml_ensemble_classifier as ml_clf_mod
    import ml_pic50_regressor as ml_reg_mod
    import finetune_chemberta_pure as ft_mod

    std = df[a.smiles_col].map(ml_clf_mod.standardize_smiles)
    ok = std.notna()
    n_bad = int((~ok).sum())
    if n_bad:
        print(f"WARNING: {n_bad} row(s) failed SMILES standardization -- dropped", file=sys.stderr)
    df = df[ok].copy()
    df["canonical_smiles"] = [s[0] for s in std[ok]]
    smiles = df["canonical_smiles"].tolist()
    print(f"scoring {len(df):,} compound(s)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # ML_Ensemble: classifier proba + regressor pIC50 (self-contained joblib)
    # =========================================================================
    sys.modules["__main__"] = ml_clf_mod
    clf_bundle = joblib.load(a.clf_model)
    Xc, names_c = ml_clf_mod.featurize_smiles(smiles, fp_bits=2048)
    assert names_c == clf_bundle["feature_names"], "ML classifier feature layout mismatch"
    p_ml = clf_bundle["model"].predict_proba(Xc)[:, 1]

    sys.modules["__main__"] = ml_reg_mod
    reg_bundle = joblib.load(a.reg_model)
    Xr, names_r = ml_reg_mod.featurize(smiles, fp_bits=reg_bundle.get("fp_bits", 2048))
    assert names_r == reg_bundle["feature_names"], "ML regressor feature layout mismatch"
    pic50 = reg_bundle["model"].predict(Xr)
    print("  ML_Ensemble scored (classifier + regressor)")

    # =========================================================================
    # FT-ChemBERTa (pure): classifier proba only -- regression comes from ML alone
    # =========================================================================
    bundle = joblib.load(Path(a.ft_clf_dir) / "cb_ft_scaler.joblib")
    model_name, extra_dim = bundle["model_name"], bundle["extra_dim"]
    assert bundle["vt"] is None and extra_dim == 0, "expected the PURE (extra_dim=0) FT-ChemBERTa checkpoint"
    extra = np.zeros((len(smiles), 0), dtype=np.float32)

    tokenizer = ft_mod.AutoTokenizer.from_pretrained(model_name)
    ft_model = ft_mod.HERG_ChemBERTa_FT(model_name, extra_dim).to(device)
    ft_model.load_state_dict(torch.load(Path(a.ft_clf_dir) / "cb_ft_final.pt", map_location=device))
    loader = DataLoader(ft_mod.HERGDataset(smiles, extra, np.zeros(len(smiles))),
                        batch_size=a.batch, shuffle=False, collate_fn=ft_mod.make_collate_fn(tokenizer))
    p_ft = ft_mod.predict(ft_model, loader, device)
    print("  FT-ChemBERTa (pure) scored (classifier)")

    # =========================================================================
    # wire it together
    # =========================================================================
    p_blend = W_ML * p_ml + W_FT * p_ft

    out = pd.DataFrame({
        "compound_id": df[a.id_col].values,
        "hERG_blocker_probability": np.round(p_blend, 4),
        "predicted hERG pIC50": np.round(pic50, 3),
    })
    out_ext = out.copy()
    out_ext["p_ML_Ensemble"] = np.round(p_ml, 4)
    out_ext["p_FT_ChemBERTa_pure"] = np.round(p_ft, 4)

    if a.smiles:
        print("\n" + out_ext.to_string(index=False))
    else:
        out.to_csv(a.out, index=False)
        ext_path = Path(a.out).with_name(Path(a.out).stem + "_extended.csv")
        out_ext.to_csv(ext_path, index=False)
        print(f"\nwrote -> {a.out} ({len(out)} rows, 3-column rubric schema)")
        print(f"wrote -> {ext_path} (+ per-model contributions)")


if __name__ == "__main__":
    main()
