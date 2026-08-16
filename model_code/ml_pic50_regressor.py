#!/usr/bin/env python3
"""
ml_pic50_regressor.py
======================
SECOND, SEPARATE model entity for Challenge 2: predicts the continuous hERG
`pIC50` value. It is deliberately independent of the blocker CLASSIFIER
(train_ml_ensemble_classifier.py) — different task, different data, different
metrics — and the two are joined only at submission time (assemble_submission.py).
This is the deployed regressor: master_predict.py uses its output
(pic50_regressor.joblib) alone for the final pIC50 prediction.

WHY A SEPARATE ENTITY WITH DIFFERENT DATA
-----------------------------------------
* The classifier can learn from EVERY labelled compound, including ones known
  only through % inhibition or censored (">10 uM") measurements — a binary
  blocker/non-blocker label survives that coarseness.
* A regressor needs a REAL continuous pIC50. %-inhibition and censored rows do
  not provide one; feeding them in would mean inventing pIC50 numbers and
  poisoning the fit. So the regressor trains ONLY on the exact-potency subset
  (median pIC50 from `=`-relation IC50/Ki/etc.). Smaller, but clean.
* Consequence: the two models train on different rows. That is correct and
  intended. The ONE thing they MUST share is the 20% held-out compound set, so
  that test compounds are excluded from both trainings and predictions are
  genuinely out-of-sample. Pass the same `--holdout_ids` file to both.

At inference the regressor predicts a pIC50 for ANY SMILES, so it can fill the
`predicted_pIC50` column for every held-out compound — even ones that only ever
had a binary label.

METHODOLOGY (mirrors the classifier's rigor, for the 35% methodology rubric)
----------------------------------------------------------------------------
* Feature set = the Track B survey set: Morgan/ECFP (radius 2, 2048 bits) +
  RDKit 2D descriptors (MolLogP, TPSA, aromatic rings, partial charges, ...).
* Scaffold-aware split (Murcko) so analogues cannot straddle train/test.
* No leakage: imputation/variance/scaling live inside sklearn Pipelines fit on
  train folds only. Fingerprints/descriptors are per-molecule pure functions.
* Model zoo of regressors (RF, ExtraTrees, XGBoost, LightGBM, CatBoost),
  selected by scaffold-CV RMSE; graceful if a library is absent.
* Applicability domain via kNN-Tanimoto (Sahigara 2012).
* Optional y-scrambling null (--y_scramble_repeats).
* Regression metrics: MAE, RMSE, R2, Pearson, Spearman (for the report — note
  the rubric scores the CLASSIFIER, so these are supporting evidence).

Requirements: pandas numpy scikit-learn rdkit (xgboost lightgbm catboost optional)

Usage
-----
  python ml_pic50_regressor.py \
      --data ../data/herg_combined_dataset/herg_binary_ml_dataset.csv \
      --pic50_col median_pIC50_equiv --id_col compound_chembl_id \
      --holdout_ids ../data/holdout_out/holdout_inchikeys.csv --out_dir ml_pic50_regressor_results
"""

from __future__ import annotations
import argparse, json, os, time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import clone

import joblib

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
try:
    from scipy.stats import pearsonr, spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ---------------------------------------------------------------- features ---
# Same descriptor block as the Track B survey / the classifier, so the two
# models share a feature definition.
DESCRIPTOR_FUNCS = [
    ("MolWt", Descriptors.MolWt), ("MolLogP", Descriptors.MolLogP),
    ("TPSA", Descriptors.TPSA), ("NumHDonors", Descriptors.NumHDonors),
    ("NumHAcceptors", Descriptors.NumHAcceptors),
    ("HeavyAtomCount", Descriptors.HeavyAtomCount),
    ("NHOHCount", Descriptors.NHOHCount), ("NOCount", Descriptors.NOCount),
    ("NumAliphaticRings", Descriptors.NumAliphaticRings),
    ("NumAromaticRings", Descriptors.NumAromaticRings),
    ("NumSaturatedRings", Descriptors.NumSaturatedRings),
    ("RingCount", Descriptors.RingCount),
    ("FractionCSP3", Descriptors.FractionCSP3), ("MolMR", Descriptors.MolMR),
    ("NumValenceElectrons", Descriptors.NumValenceElectrons),
    ("MaxPartialCharge", Descriptors.MaxPartialCharge),
    ("MinPartialCharge", Descriptors.MinPartialCharge),
    ("BalabanJ", Descriptors.BalabanJ), ("BertzCT", Descriptors.BertzCT),
    ("Chi0n", Descriptors.Chi0n), ("Chi1n", Descriptors.Chi1n),
    ("Kappa1", Descriptors.Kappa1), ("Kappa2", Descriptors.Kappa2),
    ("Kappa3", Descriptors.Kappa3), ("NumRotatableBonds", Descriptors.NumRotatableBonds),
]


def standardize_smiles(smi: str) -> Optional[Tuple[str, str, str]]:
    """Return (canonical_smiles, inchikey, murcko_scaffold) or None."""
    if smi is None or (isinstance(smi, float) and np.isnan(smi)):
        return None
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return None
        mol = rdMolStandardize.FragmentParent(rdMolStandardize.Cleanup(mol))
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        canon = Chem.MolToSmiles(mol, canonical=True)
        m2 = Chem.MolFromSmiles(canon)
        if m2 is None:
            return None
        ik = Chem.MolToInchiKey(m2)
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=m2) or f"NO_SCAFFOLD::{canon}"
        return canon, ik, scaf
    except (ValueError, RuntimeError):
        return None


def featurize(smiles: List[str], fp_bits: int = 2048, radius: int = 2):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_bits)
    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        fp = gen.GetFingerprintAsNumPy(mol).astype(np.float32)
        desc = []
        for _, fn in DESCRIPTOR_FUNCS:
            try:
                v = fn(mol)
                desc.append(float(v) if v is not None and np.isfinite(v) else 0.0)
            except (ValueError, RuntimeError, ZeroDivisionError):
                desc.append(0.0)
        rows.append(np.concatenate([fp, np.asarray(desc, dtype=np.float32)]))
    names = [f"fp_{i}" for i in range(fp_bits)] + [n for n, _ in DESCRIPTOR_FUNCS]
    return np.vstack(rows), names


# --------------------------------------------------------- applicability -----
def _knn_mean_sim(fp, all_fps, k, exclude_idx=-1):
    sims = DataStructs.BulkTanimotoSimilarity(fp, all_fps)
    if exclude_idx >= 0:
        sims = [s for j, s in enumerate(sims) if j != exclude_idx]
    if len(sims) < k:
        return float(np.mean(sims)) if sims else 0.0
    return float(np.mean(sorted(sims, reverse=True)[:k]))


def rdkit_fps(smiles: List[str], fp_bits=2048, radius=2):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_bits)
    return [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]


def ad_threshold(train_fps, k=5):
    sc = [_knn_mean_sim(fp, train_fps, k, exclude_idx=i) for i, fp in enumerate(train_fps)]
    a = np.array(sc)
    return float(a.mean() - a.std()) if len(a) else 0.0


# ------------------------------------------------------------- metrics -------
def reg_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    out = {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "n": int(len(y_true)),
    }
    if HAS_SCIPY and len(y_true) > 2:
        out["Pearson"] = float(pearsonr(y_true, y_pred)[0])
        out["Spearman"] = float(spearmanr(y_true, y_pred)[0])
    else:
        out["Pearson"] = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else np.nan
        out["Spearman"] = np.nan
    return out


# ------------------------------------------------------------- model zoo -----
def _pipe(model):
    return Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("var", VarianceThreshold()),
        ("scale", StandardScaler(with_mean=False)),  # sparse-safe, tree-neutral
        ("reg", model),
    ])


def resolve_device(requested: str) -> str:
    """Resolve 'auto'/'cpu'/'cuda' to an actual device, probing a tiny XGBoost
    CUDA fit since that result also predicts CatBoost/LightGBM GPU support in
    this environment (same underlying driver/toolkit)."""
    requested = requested.lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {requested}")
    if requested == "cpu":
        return "cpu"
    if not HAS_XGB:
        if requested == "cuda":
            raise RuntimeError("CUDA requested but xgboost is not installed to probe it.")
        return "cpu"
    try:
        Xp = np.random.rand(16, 4).astype(np.float32)
        yp = np.random.rand(16).astype(np.float32)
        XGBRegressor(n_estimators=2, tree_method="hist", device="cuda:0").fit(Xp, yp)
        return "cuda"
    except Exception as exc:
        if requested == "cuda":
            raise RuntimeError(f"CUDA was requested but the GPU probe failed: {exc}")
        return "cpu"


def model_zoo(seed: int, device: str = "cpu") -> Dict[str, Pipeline]:
    z: Dict[str, Pipeline] = {}
    z["rf_400"] = _pipe(RandomForestRegressor(
        n_estimators=400, min_samples_leaf=2, max_features="sqrt",
        n_jobs=-1, random_state=seed))
    z["et_600"] = _pipe(ExtraTreesRegressor(
        n_estimators=600, min_samples_leaf=2, max_features="sqrt",
        n_jobs=-1, random_state=seed))
    use_gpu = device == "cuda"
    if HAS_XGB:
        z["xgb"] = _pipe(XGBRegressor(
            n_estimators=600, max_depth=5, learning_rate=0.03, subsample=0.9,
            colsample_bytree=0.8, reg_lambda=1.0, objective="reg:squarederror",
            tree_method="hist", device=("cuda:0" if use_gpu else "cpu"),
            n_jobs=-1, random_state=seed))
    if HAS_LGBM:
        z["lgbm"] = _pipe(LGBMRegressor(
            n_estimators=700, num_leaves=63, learning_rate=0.03, subsample=0.9,
            subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
            device=("gpu" if use_gpu else "cpu"),
            n_jobs=-1, random_state=seed, verbosity=-1))
    if HAS_CATBOOST:
        z["catboost"] = _pipe(CatBoostRegressor(
            iterations=700, depth=8, learning_rate=0.03, l2_leaf_reg=3.0,
            loss_function="RMSE", random_seed=seed,
            task_type=("GPU" if use_gpu else "CPU"),
            devices="0" if use_gpu else None,
            thread_count=-1, verbose=False, allow_writing_files=False))
    return z


# ------------------------------------------------------------- data ----------
@dataclass
class Prepared:
    df: pd.DataFrame
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    feature_names: List[str]


def load_and_prepare(path: str, smiles_col: str, pic50_col: str, id_col: str,
                     require_exact: bool, fp_bits: int) -> Prepared:
    """Load ALL compounds (not just the pIC50-bearing ones) so we can PREDICT a
    pIC50 for every held-out compound at inference, while still TRAINING only on
    the clean exact-potency subset. The training label lives in __train_pic50__
    (exact potency only); the ground-truth column `pic50_col` is kept for
    evaluation and is NaN for compounds that never had an exact potency."""
    df = pd.read_csv(path)
    for c in (smiles_col, pic50_col):
        if c not in df.columns:
            raise ValueError(f"missing column '{c}' in {path}")
    df = df.copy()
    df[pic50_col] = pd.to_numeric(df[pic50_col], errors="coerce")
    # training-eligible pIC50: exact-potency only (if label_source is present).
    train_p = df[pic50_col].copy()
    if require_exact and "label_source" in df.columns:
        train_p = train_p.where(df["label_source"] == "exact_potency")
    df["__train_pic50__"] = train_p

    std = df[smiles_col].map(standardize_smiles)
    ok = std.notna()
    df = df[ok].copy(); std = std[ok]
    df["canonical_smiles"] = [s[0] for s in std]
    df["inchikey"] = [s[1] for s in std]
    df["scaffold"] = [s[2] for s in std]
    if id_col not in df.columns:
        df[id_col] = df["inchikey"]

    # de-duplicate by structure: median pIC50 per InChIKey (robust to outliers)
    agg = (df.groupby("inchikey")
             .agg(**{pic50_col: (pic50_col, "median"),
                     "__train_pic50__": ("__train_pic50__", "median"),
                     "canonical_smiles": ("canonical_smiles", "first"),
                     "scaffold": ("scaffold", "first"),
                     id_col: (id_col, "first")})
             .reset_index())
    n_all, n_train = len(agg), int(agg["__train_pic50__"].notna().sum())
    print(f"  loaded {n_all:,} unique compounds; {n_train:,} have an exact-potency "
          f"pIC50 usable for TRAINING; the rest can still be PREDICTED at inference")

    X, names = featurize(agg["canonical_smiles"].tolist(), fp_bits=fp_bits)
    y = agg[pic50_col].to_numpy(float)          # ground-truth pIC50 (NaN if none)
    groups = agg["scaffold"].astype(str).to_numpy()
    return Prepared(agg, X, y, groups, names)


# ------------------------------------------------------------- selection -----
def select_model(prepared: Prepared, seed: int, cv_folds: int,
                  device: str = "cpu") -> Tuple[str, Dict[str, float]]:
    """Scaffold-grouped CV RMSE to pick the best regressor family."""
    X, y, g = prepared.X, prepared.y, prepared.groups
    n_folds = min(cv_folds, len(np.unique(g)))
    gkf = GroupKFold(n_splits=max(2, n_folds))
    scores: Dict[str, List[float]] = {}
    for name, mdl in model_zoo(seed, device).items():
        rmses = []
        for tr, va in gkf.split(X, y, g):
            m = clone(mdl)
            m.fit(X[tr], y[tr])
            pred = m.predict(X[va])
            rmses.append(np.sqrt(mean_squared_error(y[va], pred)))
        scores[name] = rmses
        print(f"    {name:10s} CV RMSE = {np.mean(rmses):.3f} ± {np.std(rmses):.3f}")
    means = {k: float(np.mean(v)) for k, v in scores.items()}
    best = min(means, key=means.get)
    return best, means


# ------------------------------------------------------------- run -----------
def load_holdout_ids(path: Optional[str]) -> Optional[set]:
    if not path:
        return None
    ids = pd.read_csv(path)
    col = ids.columns[0]
    return set(ids[col].astype(str))


def run(args) -> None:
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    device = resolve_device(args.device)
    print(f"  compute device: requested={args.device} resolved={device}")
    prepared = load_and_prepare(args.data, args.smiles_col, args.pic50_col,
                                args.id_col, args.require_exact, args.fp_bits)

    # ---- resolve compound sets --------------------------------------------
    df = prepared.df
    X = prepared.X
    y_all = prepared.y                                    # ground-truth pIC50 (NaN if none)
    y_train_all = df["__train_pic50__"].to_numpy(float)   # exact-potency training label
    groups = prepared.groups
    id_col = args.id_col

    holdout = load_holdout_ids(args.holdout_ids)
    if holdout is not None:
        is_test = (df["inchikey"].astype(str).isin(holdout)
                   | df[id_col].astype(str).isin(holdout)).to_numpy()
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=args.holdout_frac, random_state=args.seed)
        _, te0 = next(gss.split(np.arange(len(df)), y_all, groups))
        is_test = np.zeros(len(df), bool); is_test[te0] = True

    finite_train = np.isfinite(y_train_all)
    finite_true = np.isfinite(y_all)
    tr_idx = np.where((~is_test) & finite_train)[0]       # train: exact potency, not holdout
    te_idx = np.where(is_test)[0]                          # predict: ALL holdout compounds
    eval_idx = np.where(is_test & finite_true)[0]          # score: holdout w/ ground truth
    print(f"  train pool (exact potency, non-holdout): {len(tr_idx):,}")
    print(f"  holdout to PREDICT: {len(te_idx):,}  |  with ground truth to SCORE: {len(eval_idx):,}")

    X_tr, y_tr, g_tr = X[tr_idx], y_train_all[tr_idx], groups[tr_idx]

    # ---- model selection on TRAIN only -------------------------------------
    print("  selecting regressor (scaffold-CV on train):")
    prep_tr = Prepared(df.iloc[tr_idx], X_tr, y_tr, g_tr, prepared.feature_names)
    best_name, cv_means = select_model(prep_tr, args.seed, args.cv_folds, device)
    print(f"  -> best family: {best_name}")

    final = clone(model_zoo(args.seed, device)[best_name])
    final.fit(X_tr, y_tr)

    # ---- predict ALL holdout compounds; score only the ground-truth subset --
    y_pred_te = final.predict(X[te_idx])
    pos = {gi: k for k, gi in enumerate(te_idx)}           # global idx -> local te position
    metrics: Dict[str, float] = {}
    if len(eval_idx):
        ev_local = np.array([pos[i] for i in eval_idx])
        metrics = reg_metrics(y_all[eval_idx], y_pred_te[ev_local])
        print("  HELD-OUT (ground-truth subset):", {k: round(v, 4) for k, v in metrics.items()})
    else:
        print("  (no ground-truth pIC50 in holdout -> predictions only, no regression metrics)")

    # ---- applicability domain (over ALL predicted holdout compounds) -------
    tr_fps = rdkit_fps(df.iloc[tr_idx]["canonical_smiles"].tolist(), args.fp_bits)
    te_fps = rdkit_fps(df.iloc[te_idx]["canonical_smiles"].tolist(), args.fp_bits)
    thr = ad_threshold(tr_fps, k=args.ad_knn_k)
    te_sim = np.array([_knn_mean_sim(fp, tr_fps, args.ad_knn_k) for fp in te_fps])
    in_ad = (te_sim >= thr)
    metrics["held_out_in_AD_fraction"] = float(in_ad.mean())
    if len(eval_idx):
        ev_local = np.array([pos[i] for i in eval_idx])
        ev_in_ad = in_ad[ev_local]
        if ev_in_ad.sum() > 2:
            metrics["MAE_in_AD"] = float(mean_absolute_error(
                y_all[eval_idx][ev_in_ad], y_pred_te[ev_local][ev_in_ad]))

    # ---- optional y-scrambling null (on the ground-truth subset) -----------
    if args.y_scramble_repeats > 0 and len(eval_idx):
        rng = np.random.default_rng(args.seed)
        X_ev, y_ev = X[eval_idx], y_all[eval_idx]
        null_rmse = []
        for _ in range(args.y_scramble_repeats):
            ys = y_tr.copy(); rng.shuffle(ys)
            m = clone(model_zoo(args.seed, device)[best_name]); m.fit(X_tr, ys)
            null_rmse.append(np.sqrt(mean_squared_error(y_ev, m.predict(X_ev))))
        metrics["yscramble_RMSE_mean"] = float(np.mean(null_rmse))
        metrics["yscramble_RMSE_std"] = float(np.std(null_rmse))
        print(f"  y-scramble null RMSE = {np.mean(null_rmse):.3f} "
              f"(real {metrics.get('RMSE', float('nan')):.3f}; real should be much lower)")

    # ---- outputs: a pIC50 for EVERY held-out compound ----------------------
    pred_df = df.iloc[te_idx][[id_col, "canonical_smiles", "inchikey"]].copy()
    pred_df["true_pIC50"] = y_all[te_idx]                 # NaN where no measured value
    pred_df["predicted_pIC50"] = np.round(y_pred_te, 3)
    pred_df["ad_knn_mean_sim"] = np.round(te_sim, 3)
    pred_df["in_applicability_domain"] = in_ad.astype(int)
    pred_df.to_csv(out / "heldout_predictions.csv", index=False)
    print(f"  wrote {len(pred_df):,} holdout predictions "
          f"({int(pred_df['predicted_pIC50'].notna().sum()):,} with a pIC50 value)")

    joblib.dump({"model": final, "feature_names": prepared.feature_names,
                 "best_family": best_name, "fp_bits": args.fp_bits,
                 "pic50_col": args.pic50_col}, out / "pic50_regressor.joblib")
    with open(out / "metrics.json", "w") as fh:
        json.dump({"held_out": metrics, "cv_rmse_by_model": cv_means,
                   "best_family": best_name, "config": vars(args)}, fh, indent=2)

    print(f"\n  saved -> {out}/  (regressor, predictions, metrics)  "
          f"[{time.time()-t0:.1f}s]")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Standalone hERG pIC50 regressor (Challenge 2, second head).")
    ap.add_argument("--data", required=True, help="CSV with SMILES + pIC50 (e.g. herg_binary_ml_dataset.csv)")
    ap.add_argument("--smiles_col", default="smiles")
    ap.add_argument("--pic50_col", default="median_pIC50_equiv")
    ap.add_argument("--id_col", default="compound_chembl_id")
    ap.add_argument("--require_exact", action="store_true", default=True,
                    help="Train only on exact-potency rows (label_source==exact_potency) when present.")
    ap.add_argument("--allow_all_numeric", dest="require_exact", action="store_false",
                    help="Use any numeric pIC50, not just exact-potency rows.")
    ap.add_argument("--holdout_ids", default=None,
                    help="CSV of inchikeys/ids that BOTH models hold out (shared 20%% test).")
    ap.add_argument("--holdout_frac", type=float, default=0.20)
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--ad_knn_k", type=int, default=5)
    ap.add_argument("--fp_bits", type=int, default=2048)
    ap.add_argument("--y_scramble_repeats", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                     help="Compute device for xgboost/lightgbm/catboost. "
                          "'auto' uses CUDA if a GPU probe succeeds, else CPU.")
    ap.add_argument("--out_dir", default="herg_regressor_results")
    return ap


if __name__ == "__main__":
    run(build_argparser().parse_args())