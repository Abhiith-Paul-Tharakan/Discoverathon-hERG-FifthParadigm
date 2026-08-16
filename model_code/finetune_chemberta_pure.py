#!/usr/bin/env python3
"""
finetune_chemberta_pure.py
===========================
Full end-to-end ChemBERTa fine-tuning for the hERG blocker classifier. Run
with --pure (transformer-embedding only, no ECFP4/physchem fusion, extra_dim=0),
this produces the "pure" checkpoint (checkpoints/cb_ft_pure_results) that
master_predict.py loads as one of the two final-model components.

This keeps the two genuine improvements found in an external Kaggle
fine-tuning notebook -- (1) unfreezing the transformer instead of using it
as a frozen embedder, and (2) a deeper pooled-embedding MLP head -- but
otherwise follows this project's own established, better practices instead
of that notebook's:

  * trains on OUR curated, evidence-tiered dataset (herg_combined_dataset.csv,
    16,650 compounds, ~balanced classes), not the notebook's smaller
    (10,351), undocumented, 2:1-imbalanced one
  * splits by scaffold (GroupKFold / GroupShuffleSplit) and evaluates on the
    SAME shared holdout every other model in this project honors
    (--holdout_ids). The notebook used a random stratified split, and 53% of
    THIS holdout's compounds were present in its training file -- its
    reported 0.80 ROC-AUC is not a fair number to compare against.
  * optionally (--extra_dim > 0) fuses an ECFP4 fingerprint + physchem
    descriptors onto the pooled transformer embedding before the head --
    concretely improved a frozen-embedding ChemBERTa variant evaluated
    during model selection (not included in this minimal repo)
  * free fold-ensembling (average the already-trained OOF fold models on the
    holdout instead of throwing them away)
  * persists the final model AND the fitted extra-feature scaler, so a later
    external-validation script can load-and-predict instead of having to
    replay the whole training run to reconstruct them (a real pain point
    hit with the frozen-embedding scripts, which never saved either)

Unfreezing the encoder is meaningfully more expensive per epoch than the
frozen sibling -- expect minutes per fold on a GPU, not seconds.

Outputs (in --out_dir)
  cb_ft_oof_predictions.csv      out-of-fold blocker probs on the train pool
  cb_ft_holdout_predictions.csv  probs on the held-out compounds
  cb_ft_final.pt                 final model state_dict
  cb_ft_scaler.joblib            fitted extra-feature StandardScaler + VarianceThreshold

Usage
-----
python finetune_chemberta_pure.py --pure \
    --data ../data/herg_combined_dataset/herg_combined_dataset.csv \
    --smiles_col smiles --label_col hERG_blocker --id_col compound_chembl_id \
    --holdout_ids ../data/holdout_out/holdout_inchikeys.csv \
    --out_dir cb_ft_pure_results --model DeepChem/ChemBERTa-77M-MTR
    # (--pure reproduces the deployed checkpoint; omit it and this trains the
    # ECFP4/physchem-fusion variant instead, which master_predict.py does NOT use)
"""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors, QED
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel


# ----------------------------------------------------------------- chem ------
def standardize_smiles(smi) -> Optional[Tuple[str, str, str]]:
    if smi is None or (isinstance(smi, float) and np.isnan(smi)):
        return None
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return None
        mol = rdMolStandardize.Uncharger().uncharge(
            rdMolStandardize.FragmentParent(rdMolStandardize.Cleanup(mol)))
        canon = Chem.MolToSmiles(mol, canonical=True)
        m2 = Chem.MolFromSmiles(canon)
        if m2 is None:
            return None
        ik = Chem.MolToInchiKey(m2)
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=m2) or f"NO_SCAFFOLD::{canon}"
        return canon, ik, scaf
    except (ValueError, RuntimeError):
        return None


_FP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_DESC_NAMES = ["MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
               "NumRotatableBonds", "NumAromaticRings", "RingCount", "FractionCSP3",
               "HeavyAtomCount", "NumHeteroatoms", "NOCount", "NHOHCount", "MolMR",
               "NumValenceElectrons", "BertzCT", "LabuteASA"]


def morgan_fp(smiles: List[str]) -> np.ndarray:
    out = np.zeros((len(smiles), _FP_GEN.GetOptions().fpSize), dtype=np.float32)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            out[i] = _FP_GEN.GetFingerprintAsNumPy(mol)
    return out


def physchem_descriptors(smiles: List[str]) -> np.ndarray:
    out = np.full((len(smiles), len(_DESC_NAMES) + 1), np.nan, dtype=np.float64)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            out[i, :-1] = [getattr(Descriptors, n)(mol) for n in _DESC_NAMES]
            out[i, -1] = QED.qed(mol)
        except (ValueError, RuntimeError, ZeroDivisionError):
            continue
    out[~np.isfinite(out)] = np.nan
    col_med = np.nanmedian(out, axis=0)
    inds = np.where(np.isnan(out))
    out[inds] = np.take(col_med, inds[1])
    return out


# ----------------------------------------------------------------- data ------
class HERGDataset(Dataset):
    def __init__(self, smiles: List[str], extra: np.ndarray, labels: np.ndarray):
        self.smiles, self.extra, self.labels = smiles, extra, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.smiles[idx], self.extra[idx], self.labels[idx]


def make_collate_fn(tokenizer, max_len=256):
    def collate(batch):
        smiles = [b[0] for b in batch]
        extra = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.float)
        labels = torch.tensor([b[2] for b in batch], dtype=torch.float)
        enc = tokenizer(smiles, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"], extra, labels
    return collate


# ---------------------------------------------------------------- model ------
class HERG_ChemBERTa_FT(nn.Module):
    """Fine-tuned ChemBERTa (unfrozen) + fingerprint/descriptor fusion -> deep MLP head."""

    def __init__(self, model_name: str, extra_dim: int, dropout: float = 0.1):
        super().__init__()
        self.chemberta = AutoModel.from_pretrained(model_name)
        hidden = self.chemberta.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden + extra_dim, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, input_ids, attention_mask, extra):
        hidden_states = self.chemberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden_states * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.classifier(torch.cat([pooled, extra], dim=1)).squeeze(-1)


def run_epoch(model, loader, device, criterion, opt=None):
    train = opt is not None
    model.train(train)
    total_loss, all_p, all_y = 0.0, [], []
    for input_ids, attention_mask, extra, y in loader:
        input_ids, attention_mask, extra, y = (t.to(device) for t in (input_ids, attention_mask, extra, y))
        with torch.set_grad_enabled(train):
            logits = model(input_ids, attention_mask, extra)
            loss = criterion(logits, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
        total_loss += float(loss) * len(y)
        all_p.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_y.append(y.cpu().numpy())
    p, y = np.concatenate(all_p), np.concatenate(all_y)
    return total_loss / len(loader.dataset), p, y


@torch.no_grad()
def predict(model, loader, device) -> np.ndarray:
    model.eval()
    out = []
    for input_ids, attention_mask, extra, _ in loader:
        input_ids, attention_mask, extra = (t.to(device) for t in (input_ids, attention_mask, extra))
        out.append(torch.sigmoid(model(input_ids, attention_mask, extra)).cpu().numpy())
    return np.concatenate(out)


def fit_one(smiles, extra, y, groups, tr_idx, cfg, tokenizer, device, desc="train"):
    """Train on tr_idx with an internal scaffold validation carve-out for
    early stopping on val ROC-AUC. Returns predict_proba on any given idx."""
    gss = GroupShuffleSplit(n_splits=1, test_size=cfg.val_frac, random_state=cfg.seed)
    sub_tr, sub_va = next(gss.split(tr_idx, y[tr_idx], groups[tr_idx]))
    tr, va = tr_idx[sub_tr], tr_idx[sub_va]

    pos = float(y[tr].sum()); neg = float(len(tr) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = HERG_ChemBERTa_FT(cfg.model, extra.shape[1], dropout=cfg.dropout).to(device)
    opt = torch.optim.AdamW([
        {"params": model.chemberta.parameters(), "lr": cfg.lr_encoder},
        {"params": model.classifier.parameters(), "lr": cfg.lr_head},
    ], weight_decay=cfg.weight_decay)

    collate = make_collate_fn(tokenizer)
    tr_loader = DataLoader(HERGDataset([smiles[i] for i in tr], extra[tr], y[tr]),
                           batch_size=cfg.batch_size, shuffle=True, collate_fn=collate)
    va_loader = DataLoader(HERGDataset([smiles[i] for i in va], extra[va], y[va]),
                           batch_size=cfg.batch_size, shuffle=False, collate_fn=collate)

    best_auc, best_state, bad = -1.0, None, 0
    pbar = tqdm(range(1, cfg.epochs + 1), desc=desc, unit="epoch", leave=False)
    for ep in pbar:
        train_loss, _, _ = run_epoch(model, tr_loader, device, criterion, opt)
        val_loss, val_p, val_y = run_epoch(model, va_loader, device, criterion)
        try:
            auc = roc_auc_score(val_y, val_p)
        except ValueError:
            auc = float("nan")
        if np.isfinite(auc) and auc > best_auc + 1e-4:
            best_auc, best_state, bad = auc, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        pbar.set_postfix(train_loss=f"{train_loss:.3f}", val_loss=f"{val_loss:.3f}",
                         val_auc=f"{auc:.3f}", best=f"{best_auc:.3f}", bad=f"{bad}/{cfg.patience}")
        if bad >= cfg.patience:
            pbar.close()
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--smiles_col", default="smiles")
    ap.add_argument("--label_col", default="hERG_blocker")
    ap.add_argument("--id_col", default="compound_chembl_id")
    ap.add_argument("--holdout_ids", default=None)
    ap.add_argument("--model", default="DeepChem/ChemBERTa-77M-MTR",
                    help="HuggingFace checkpoint to FINE-TUNE (unfrozen).")
    ap.add_argument("--out_dir", default="cb_ft_results")
    ap.add_argument("--pure", action="store_true",
                    help="Transformer-only: skip the ECFP4 fingerprint + physchem descriptor "
                         "fusion (those overlap with the ML_Ensemble fingerprint model) so the "
                         "model's signal comes solely from the fine-tuned ChemBERTa embedding.")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--lr_encoder", type=float, default=2e-5)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--fp_var_threshold", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="Debug: cap #compounds (0 = all).")
    cfg = ap.parse_args()

    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(cfg.out_dir); out.mkdir(parents=True, exist_ok=True)
    print(f"device: {device} | fine-tuning: {cfg.model}")

    df = pd.read_csv(cfg.data)
    df["__y__"] = pd.to_numeric(df[cfg.label_col], errors="coerce")
    df = df[df["__y__"].isin([0, 1])].copy()
    std = df[cfg.smiles_col].map(standardize_smiles)
    ok = std.notna()
    df = df[ok].copy(); std = std[ok]
    df["canonical_smiles"] = [s[0] for s in std]
    df["inchikey"] = [s[1] for s in std]
    df["scaffold"] = [s[2] for s in std]
    if cfg.id_col not in df.columns:
        df[cfg.id_col] = df["inchikey"]
    df = df.drop_duplicates("inchikey").reset_index(drop=True)
    if cfg.limit:
        df = df.sample(min(cfg.limit, len(df)), random_state=cfg.seed).reset_index(drop=True)
    y = df["__y__"].to_numpy(int)
    groups = df["scaffold"].astype(str).to_numpy()
    smiles = df["canonical_smiles"].tolist()
    print(f"compounds: {len(df):,} | positive rate: {y.mean():.3f}")

    if cfg.holdout_ids:
        hold = set(pd.read_csv(cfg.holdout_ids).iloc[:, 0].astype(str))
        is_hold = (df["inchikey"].astype(str).isin(hold) | df[cfg.id_col].astype(str).isin(hold)).to_numpy()
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=cfg.seed)
        _, te0 = next(gss.split(np.arange(len(df)), y, groups))
        is_hold = np.zeros(len(df), bool); is_hold[te0] = True
    tr_idx = np.where(~is_hold)[0]
    hold_idx = np.where(is_hold)[0]
    print(f"train: {len(tr_idx):,} | holdout: {len(hold_idx):,}")

    if cfg.pure:
        vt, scaler = None, None
        extra = np.zeros((len(df), 0), dtype=np.float32)
        print("PURE mode: no ECFP4/physchem fusion -- transformer-embedding-only")
    else:
        print("computing ECFP4 + physchem extra features ...")
        fp_full = morgan_fp(smiles)
        desc = physchem_descriptors(smiles)
        vt = VarianceThreshold(threshold=cfg.fp_var_threshold).fit(fp_full[tr_idx])
        fp = vt.transform(fp_full)
        scaler = StandardScaler().fit(np.hstack([fp, desc])[tr_idx])
        extra = scaler.transform(np.hstack([fp, desc])).astype(np.float32)
        print(f"  extra feature dim: {extra.shape[1]} ({fp.shape[1]} ECFP4 bits kept + {desc.shape[1]} descriptors)")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)

    oof = np.full(len(df), np.nan)
    fold_preds_on_holdout = []
    n_folds = min(cfg.folds, len(np.unique(groups[tr_idx])))
    gkf = GroupKFold(n_splits=max(2, n_folds))
    hold_loader = DataLoader(HERGDataset([smiles[i] for i in hold_idx], extra[hold_idx], y[hold_idx]),
                             batch_size=cfg.batch_size, shuffle=False, collate_fn=make_collate_fn(tokenizer))
    for k, (a, b) in enumerate(gkf.split(tr_idx, y[tr_idx], groups[tr_idx]), 1):
        fold_tr, fold_va = tr_idx[a], tr_idx[b]
        model, auc = fit_one(smiles, extra, y, groups, fold_tr, cfg, tokenizer, device, desc=f"OOF fold {k}/{max(2, n_folds)}")
        va_loader = DataLoader(HERGDataset([smiles[i] for i in fold_va], extra[fold_va], y[fold_va]),
                               batch_size=cfg.batch_size, shuffle=False, collate_fn=make_collate_fn(tokenizer))
        oof[fold_va] = predict(model, va_loader, device)
        fold_preds_on_holdout.append(predict(model, hold_loader, device))
        print(f"  OOF fold {k}/{max(2, n_folds)}: val-AUC(early-stop)={auc:.4f} "
              f"fold-AUC={roc_auc_score(y[fold_va], oof[fold_va]):.4f}")
        del model; torch.cuda.empty_cache()

    oof_auc = roc_auc_score(y[tr_idx], oof[tr_idx])
    oof_ap = average_precision_score(y[tr_idx], oof[tr_idx])
    print(f"  ChemBERTa-FT OOF ROC-AUC={oof_auc:.4f} PR-AUC={oof_ap:.4f}")

    final, _ = fit_one(smiles, extra, y, groups, tr_idx, cfg, tokenizer, device, desc="final model")
    fold_preds_on_holdout.append(predict(final, hold_loader, device))
    hold_p_final_only = fold_preds_on_holdout[-1]
    hold_p = np.mean(fold_preds_on_holdout, axis=0)
    if len(hold_idx):
        print(f"  HOLDOUT (final model only)      ROC-AUC={roc_auc_score(y[hold_idx], hold_p_final_only):.4f} "
              f"PR-AUC={average_precision_score(y[hold_idx], hold_p_final_only):.4f}")
        print(f"  HOLDOUT ({len(fold_preds_on_holdout)}-model ensemble) ROC-AUC={roc_auc_score(y[hold_idx], hold_p):.4f} "
              f"PR-AUC={average_precision_score(y[hold_idx], hold_p):.4f}")

    def frame(idx, p):
        return pd.DataFrame({cfg.id_col: df.iloc[idx][cfg.id_col].values,
                             "inchikey": df.iloc[idx]["inchikey"].values,
                             "y_true": y[idx], "cb_ft_proba": np.round(p, 6)})
    frame(tr_idx, oof[tr_idx]).to_csv(out / "cb_ft_oof_predictions.csv", index=False)
    frame(hold_idx, hold_p).to_csv(out / "cb_ft_holdout_predictions.csv", index=False)
    torch.save(final.state_dict(), out / "cb_ft_final.pt")
    joblib.dump({"vt": vt, "scaler": scaler, "model_name": cfg.model, "extra_dim": extra.shape[1]},
               out / "cb_ft_scaler.joblib")
    print(f"\nwrote -> {out}/cb_ft_oof_predictions.csv, cb_ft_holdout_predictions.csv, "
          f"cb_ft_final.pt, cb_ft_scaler.joblib")


if __name__ == "__main__":
    main()
