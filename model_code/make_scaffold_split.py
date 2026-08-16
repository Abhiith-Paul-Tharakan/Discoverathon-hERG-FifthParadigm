#!/usr/bin/env python3
"""
make_scaffold_split.py
=======================
Combine the two hERG data files into one compound table and generate the
SHARED 20% held-out test set (seed 42) that BOTH the classifier
(train_ml_ensemble_classifier.py) and the pIC50 regressor (ml_pic50_regressor.py)
must exclude from training.

Inputs
------
* binary file  (e.g. herg_binary_ml_dataset.csv): SMILES + hERG_blocker label
  (+ compound id, median_pIC50_equiv, label_source). The SCORED universe.
* unified file (e.g. herg_pic50_unified.csv): SMILES + pIC50 (+ source).
  Extra continuous labels that enrich the regressor's training set.

Design of a "good" holdout
---------------------------
1. IDENTICAL standardization to the models (Cleanup -> FragmentParent ->
   Uncharge -> canonical SMILES -> InChIKey -> Murcko scaffold), so the
   InChIKeys in the holdout list actually match what the models compute.
2. SCAFFOLD-DISJOINT: whole Murcko scaffolds go entirely to train or to test,
   so no analogue straddles the split (the honest split hERG needs).
3. CLASS-STRATIFIED on the blocker label, so the 20% test set keeps a realistic
   blocker/non-blocker balance (selection uses StratifiedGroupKFold).
4. ENFORCED ACROSS BOTH FILES: holdout scaffolds are chosen on the scored
   (binary) universe, then EVERY compound on those scaffolds -- in either file --
   is held out. That stops the regressor learning from an analogue of a test
   compound via the unified file.

Outputs (in --out_dir)
----------------------
* holdout_inchikeys.csv   single column of InChIKeys -> pass to BOTH models via
                          --holdout_ids. (Models match on inchikey OR id.)
* combined_compounds.csv  the merged per-compound table with a `split` column
                          (train/holdout), blocker label, and pIC50 columns.
* holdout_manifest.csv    per-holdout-compound detail (id, scaffold, flags).
* holdout_stats.json      verification: sizes, class balance, scaffold overlap
                          (must be 0), pIC50 coverage of the holdout.

Usage
-----
python make_scaffold_split.py \
    --binary ../data/herg_combined_dataset/herg_binary_ml_dataset.csv \
    --unified ../data/herg_combined_dataset/herg_pic50_unified.csv \
    --out_dir ../data/holdout_out
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import StratifiedGroupKFold


# --- EXACT standardization used by both models -----------------------------
def standardize_smiles(smiles) -> Optional[Tuple[str, str, str]]:
    """Return (canonical_smiles, inchikey, murcko_scaffold) or None.
    Byte-for-byte the same pipeline the classifier and regressor use, so the
    InChIKeys produced here match those the models compute at load time."""
    if smiles is None or (isinstance(smiles, float) and np.isnan(smiles)):
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        clean = rdMolStandardize.Cleanup(mol)
        parent = rdMolStandardize.FragmentParent(clean)
        parent = rdMolStandardize.Uncharger().uncharge(parent)
        canonical = Chem.MolToSmiles(parent, canonical=True)
        mol2 = Chem.MolFromSmiles(canonical)
        if mol2 is None:
            return None
        inchikey = Chem.MolToInchiKey(mol2)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol2)
        scaffold = scaffold if scaffold else f"NO_SCAFFOLD::{canonical}"
        return canonical, inchikey, scaffold
    except (ValueError, RuntimeError):
        return None


def standardize_frame(df: pd.DataFrame, smiles_col: str, tag: str) -> pd.DataFrame:
    """Attach canonical_smiles / inchikey / scaffold; drop unparseable rows."""
    print(f"  [{tag}] standardizing {len(df):,} rows ...", flush=True)
    std = df[smiles_col].map(standardize_smiles)
    ok = std.notna()
    n_bad = int((~ok).sum())
    if n_bad:
        print(f"  [{tag}] dropped {n_bad:,} unparseable SMILES")
    df = df[ok].copy()
    std = std[ok]
    df["canonical_smiles"] = [s[0] for s in std]
    df["inchikey"] = [s[1] for s in std]
    df["scaffold"] = [s[2] for s in std]
    return df


def select_scaffolds_prioritized(sel: pd.DataFrame, frac: float, seed: int):
    """Both-label-prioritized, class-balanced, WHOLE-SCAFFOLD selection.

    Objective: maximise the number of held-out compounds that carry BOTH a
    blocker label and a pIC50 (so both models can be scored on the same
    compounds), while (a) keeping the overall ~p0 blocker balance and (b) never
    splitting a Murcko scaffold across train/test.

    Method: aggregate to scaffolds; greedily take scaffolds that CONTAIN
    both-label compounds first (richest first), then fill the remainder with
    non-pIC50 scaffolds. Every addition is checked against per-class caps
    (target_blk / target_non derived from the overall blocker fraction), so the
    balance is preserved and, because we only ever move whole scaffolds, the
    split stays fully scaffold-disjoint. `seed` drives a deterministic shuffle
    for tie-breaking.
    """
    p0 = float(sel["hERG_blocker"].mean())
    N = len(sel)
    target_total = round(frac * N)
    target_blk = round(p0 * target_total)
    target_non = target_total - target_blk

    agg = (sel.assign(_blk=sel["hERG_blocker"].astype(int),
                      _both=sel["pic50_final"].notna().astype(int))
              .groupby("scaffold")
              .agg(tot=("_blk", "size"), blk=("_blk", "sum"), both=("_both", "sum"))
              .reset_index())
    agg["non"] = agg["tot"] - agg["blk"]
    agg = agg.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # tie-break

    selected: set = set()
    tb = tn = 0
    # Phase 1 -- scaffolds carrying both-label compounds. Prefer high both-label
    # DENSITY and SMALL scaffolds first: this spends the per-class budget on the
    # most pIC50-efficient chemotypes and keeps the holdout diverse (many small
    # scaffolds) instead of a few giant SAR series.
    a1 = agg[agg["both"] > 0].copy()
    a1["_density"] = a1["both"] / a1["tot"]
    p1 = a1.sort_values(["_density", "both", "tot"], ascending=[False, False, True],
                        kind="mergesort")
    for r in p1.itertuples(index=False):
        if tb + r.blk <= target_blk and tn + r.non <= target_non:
            selected.add(r.scaffold); tb += int(r.blk); tn += int(r.non)
    # Phase 2 -- fill the remaining capacity with non-pIC50 scaffolds to hit the
    # size target and pull each class up to its cap (restores balance).
    p2 = agg[(agg["both"] == 0) & (~agg["scaffold"].isin(selected))].sort_values(
        "tot", kind="mergesort")
    for r in p2.itertuples(index=False):
        if tb >= target_blk and tn >= target_non:
            break
        if tb + r.blk <= target_blk and tn + r.non <= target_non:
            selected.add(r.scaffold); tb += int(r.blk); tn += int(r.non)
    diag = dict(p0=round(p0, 4), target_total=int(target_total),
                target_blk=int(target_blk), target_non=int(target_non),
                got_blk=int(tb), got_non=int(tn))
    return selected, diag


def build_comb_two_files(args) -> pd.DataFrame:
    """Original path: merge a separate binary file and unified pIC50 file."""
    bdf = pd.read_csv(args.binary)
    udf = pd.read_csv(args.unified)
    bdf = standardize_frame(bdf, args.binary_smiles_col, "binary")
    udf = standardize_frame(udf, args.unified_smiles_col, "unified")

    bdf["__label__"] = pd.to_numeric(bdf[args.binary_label_col], errors="coerce")
    conflict = bdf.groupby("inchikey")["__label__"].nunique()
    bad = set(conflict[conflict > 1].index)
    if bad:
        print(f"  [binary] dropping {int(bdf['inchikey'].isin(bad).sum()):,} rows "
              f"with conflicting blocker labels ({len(bad):,} InChIKeys)")
        bdf = bdf[~bdf["inchikey"].isin(bad)].copy()

    exact_p = pd.to_numeric(bdf.get(args.binary_pic50_col), errors="coerce")
    if args.binary_label_source_col in bdf.columns:
        exact_p = exact_p.where(bdf[args.binary_label_source_col] == "exact_potency")
    bdf = bdf.assign(__pic50_exact__=exact_p)

    bagg = (bdf.groupby("inchikey")
            .agg(canonical_smiles=("canonical_smiles", "first"),
                 scaffold=("scaffold", "first"),
                 compound_id=(args.binary_id_col, "first"),
                 hERG_blocker=("__label__", "first"),
                 pic50_exact=("__pic50_exact__", "median"))
            .reset_index())
    print(f"  [binary] unique compounds: {len(bagg):,}")

    udf["__pic50__"] = pd.to_numeric(udf[args.unified_pic50_col], errors="coerce")
    uagg = (udf.groupby("inchikey")
            .agg(canonical_smiles_u=("canonical_smiles", "first"),
                 scaffold_u=("scaffold", "first"),
                 pic50_unified=("__pic50__", "median"))
            .reset_index())
    print(f"  [unified] unique compounds: {len(uagg):,}")

    comb = bagg.merge(uagg, on="inchikey", how="outer")
    comb["in_binary"] = comb["canonical_smiles"].notna()
    comb["in_unified"] = comb["canonical_smiles_u"].notna()
    comb["canonical_smiles"] = comb["canonical_smiles"].fillna(comb["canonical_smiles_u"])
    comb["scaffold"] = comb["scaffold"].fillna(comb["scaffold_u"])
    comb["compound_id"] = comb["compound_id"].fillna(comb["inchikey"])
    comb["pic50_final"] = comb["pic50_exact"].fillna(comb["pic50_unified"])
    return comb.drop(columns=["canonical_smiles_u", "scaffold_u"])


def build_comb_combined(args) -> pd.DataFrame:
    """Preferred path: a single GENUINE combined dataset from
    herg_robust_extraction.py, already carrying both hERG_blocker and a pIC50
    column (one lineage, one gate stack -> no provenance mismatch)."""
    df = pd.read_csv(args.combined)
    df = standardize_frame(df, args.binary_smiles_col, "combined")
    df["__label__"] = pd.to_numeric(df[args.binary_label_col], errors="coerce")
    df["__pic50__"] = pd.to_numeric(df[args.combined_pic50_col], errors="coerce")

    conflict = df.groupby("inchikey")["__label__"].nunique()
    bad = set(conflict[conflict > 1].index)
    if bad:
        print(f"  [combined] dropping {int(df['inchikey'].isin(bad).sum()):,} rows "
              f"with conflicting labels ({len(bad):,} InChIKeys)")
        df = df[~df["inchikey"].isin(bad)].copy()

    idc = args.binary_id_col if args.binary_id_col in df.columns else None
    agg_spec = dict(canonical_smiles=("canonical_smiles", "first"),
                    scaffold=("scaffold", "first"),
                    hERG_blocker=("__label__", "first"),
                    pic50_final=("__pic50__", "median"))
    if idc:
        agg_spec["compound_id"] = (idc, "first")
    comb = df.groupby("inchikey").agg(**agg_spec).reset_index()
    if not idc:
        comb["compound_id"] = comb["inchikey"]
    comb["pic50_exact"] = comb["pic50_final"]
    comb["pic50_unified"] = comb["pic50_final"]
    comb["in_binary"] = comb["hERG_blocker"].notna()
    comb["in_unified"] = comb["pic50_final"].notna()
    print(f"  [combined] unique compounds: {len(comb):,}")
    return comb


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--combined", default=None,
                    help="Single GENUINE combined dataset (herg_robust_extraction.py output) "
                         "with both hERG_blocker and a pIC50 column. Use INSTEAD of "
                         "--binary/--unified; eliminates the provenance mismatch.")
    ap.add_argument("--combined_pic50_col", default="pIC50")
    ap.add_argument("--binary", default=None)
    ap.add_argument("--unified", default=None)
    ap.add_argument("--binary_smiles_col", default="smiles")
    ap.add_argument("--binary_id_col", default="compound_chembl_id")
    ap.add_argument("--binary_label_col", default="hERG_blocker")
    ap.add_argument("--binary_pic50_col", default="median_pIC50_equiv")
    ap.add_argument("--binary_label_source_col", default="label_source")
    ap.add_argument("--unified_smiles_col", default="smiles")
    ap.add_argument("--unified_pic50_col", default="pIC50")
    ap.add_argument("--derive_labels", action="store_true",
                    help="Augment the classifier's TRAINING set with blocker labels derived "
                         "from pIC50 (>= --pic50_threshold) for compounds that have a pIC50 "
                         "but no measured label. Train-side only; the holdout stays gold.")
    ap.add_argument("--pic50_threshold", type=float, default=5.0,
                    help="pIC50 cutoff for a derived blocker label (5.0 = IC50 <= 10 uM; "
                         "reproduces the existing binary labels with ~100%% agreement).")
    ap.add_argument("--derive_margin", type=float, default=0.0,
                    help="Drop derived labels within +/- this pIC50 of the threshold "
                         "(ambiguity band; 0 = keep all).")
    ap.add_argument("--holdout_frac", type=float, default=0.20)
    ap.add_argument("--strategy", choices=["stratified", "prioritized"], default="stratified",
                    help="stratified = plain scaffold-grouped class-stratified split; "
                         "prioritized = also bias scaffold selection toward compounds that "
                         "have BOTH a blocker label and a pIC50 (maximises shared ground "
                         "truth), while preserving class balance and scaffold-disjointness.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="holdout_out")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- build the combined compound table ---------------------------------
    if args.combined:
        comb = build_comb_combined(args)
    elif args.binary and args.unified:
        comb = build_comb_two_files(args)
    else:
        raise SystemExit("Provide --combined (one genuine combined dataset), "
                         "or BOTH --binary and --unified.")
    print(f"  combined unique compounds (union): {len(comb):,}")
    print(f"    with blocker label : {int(comb['hERG_blocker'].notna().sum()):,}")
    print(f"    with any pIC50     : {int(comb['pic50_final'].notna().sum()):,}")

    # ---- pick holdout scaffolds on the SCORED (labelled) universe ----------
    sel = comb[comb["hERG_blocker"].notna()].copy()
    if args.strategy == "prioritized":
        holdout_scaffolds, seldiag = select_scaffolds_prioritized(
            sel, args.holdout_frac, args.seed)
        print(f"  selected {len(holdout_scaffolds):,} holdout scaffolds "
              f"(both-label-prioritized; target {seldiag['target_total']:,} = "
              f"{seldiag['target_blk']:,} blocker + {seldiag['target_non']:,} non; "
              f"got {seldiag['got_blk']:,} + {seldiag['got_non']:,})")
    else:
        y = sel["hERG_blocker"].astype(int).to_numpy()
        groups = sel["scaffold"].astype(str).to_numpy()
        n_splits = max(2, round(1.0 / args.holdout_frac))
        n_splits = min(n_splits, len(np.unique(groups)))
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
        _, test_idx = next(sgkf.split(sel[["scaffold"]], y, groups))
        holdout_scaffolds = set(sel.iloc[test_idx]["scaffold"].astype(str))
        print(f"  selected {len(holdout_scaffolds):,} holdout scaffolds "
              f"(StratifiedGroupKFold, {n_splits} splits)")

    # ---- expand to ALL compounds on those scaffolds (both files) ----------
    comb["split"] = np.where(
        comb["scaffold"].astype(str).isin(holdout_scaffolds), "holdout", "train")

    # ---- optional: derive blocker labels from pIC50 to augment TRAINING ----
    # A blocker label is a pIC50 threshold (>= 5.0 == IC50 <= 10 uM), which
    # reproduces the measured labels ~exactly. So compounds that have a pIC50
    # but no measured label can be labelled and ADDED TO TRAINING. We keep them
    # train-side only (drop any on a holdout scaffold) so the test set stays
    # gold-labelled and scaffold-disjoint.
    comb["label_kind"] = np.where(comb["hERG_blocker"].notna(), "measured", "unlabelled")
    if args.derive_labels:
        cand_mask = (comb["hERG_blocker"].isna() & comb["pic50_final"].notna()
                     & ~comb["scaffold"].astype(str).isin(holdout_scaffolds))
        if args.derive_margin > 0:
            cand_mask &= (comb["pic50_final"] - args.pic50_threshold).abs() >= args.derive_margin
        der_lab = (comb.loc[cand_mask, "pic50_final"] >= args.pic50_threshold).astype(float)
        comb.loc[cand_mask, "hERG_blocker"] = der_lab
        comb.loc[cand_mask, "label_kind"] = "derived_from_pIC50"
        n_der = int(cand_mask.sum())
        n_blk = int((der_lab == 1).sum()); n_non = int((der_lab == 0).sum())
        print(f"  derived {n_der:,} TRAINING labels from pIC50>={args.pic50_threshold} "
              f"({n_blk:,} blocker / {n_non:,} non); holdout stays gold-labelled")

    hold = comb[comb["split"] == "holdout"].copy()
    train = comb[comb["split"] == "train"].copy()

    # ---- verification ------------------------------------------------------
    overlap = set(train["scaffold"].astype(str)) & set(hold["scaffold"].astype(str))
    lab_hold = hold[hold["hERG_blocker"].notna()]
    lab_train = train[train["hERG_blocker"].notna()]
    stats = {
        "combined_unique_compounds": int(len(comb)),
        "holdout_compounds": int(len(hold)),
        "train_compounds": int(len(train)),
        "holdout_fraction_overall": round(len(hold) / len(comb), 4),
        "holdout_fraction_of_labelled": round(len(lab_hold) / max(1, len(lab_hold) + len(lab_train)), 4),
        "scaffold_overlap_train_vs_holdout": int(len(overlap)),  # MUST be 0
        "blocker_frac_overall": round(float(comb["hERG_blocker"].mean(skipna=True)), 4),
        "blocker_frac_holdout": round(float(lab_hold["hERG_blocker"].mean()), 4) if len(lab_hold) else None,
        "blocker_frac_train": round(float(lab_train["hERG_blocker"].mean()), 4) if len(lab_train) else None,
        "holdout_with_blocker_label": int(lab_hold.shape[0]),
        "holdout_with_pic50": int(hold["pic50_final"].notna().sum()),
        "train_with_pic50": int(train["pic50_final"].notna().sum()),
        "n_holdout_scaffolds": len(holdout_scaffolds),
        "seed": args.seed,
    }

    # ---- write outputs -----------------------------------------------------
    pd.DataFrame({"inchikey": hold["inchikey"]}).to_csv(
        out / "holdout_inchikeys.csv", index=False)
    comb_cols = ["inchikey", "compound_id", "canonical_smiles", "scaffold",
                 "split", "hERG_blocker", "label_kind", "pic50_exact",
                 "pic50_unified", "pic50_final", "in_binary", "in_unified"]
    comb[comb_cols].to_csv(out / "combined_compounds.csv", index=False)

    # augmented classifier training file: every labelled compound (measured
    # holdout + measured train + derived train). Feed this to the classifier
    # with the SAME --holdout_ids; it excludes the holdout and trains on the rest.
    if args.derive_labels:
        aug = comb[comb["hERG_blocker"].notna()][
            ["compound_id", "canonical_smiles", "inchikey", "hERG_blocker",
             "label_kind", "split"]
        ].rename(columns={"compound_id": "compound_chembl_id",
                          "canonical_smiles": "smiles"})
        aug["hERG_blocker"] = aug["hERG_blocker"].astype(int)
        aug.to_csv(out / "augmented_binary_training.csv", index=False)
        n_train_lab = int((aug["split"] == "train").shape[0])
        print(f"  wrote -> {out}/augmented_binary_training.csv ({len(aug):,} labelled "
              f"compounds; {int((aug['label_kind']=='derived_from_pIC50').sum()):,} derived, "
              f"all train-side). Feed to the classifier with the SAME --holdout_ids.")
    hold[["inchikey", "compound_id", "scaffold", "hERG_blocker",
          "pic50_final", "in_binary", "in_unified"]].to_csv(
        out / "holdout_manifest.csv", index=False)
    with open(out / "holdout_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)

    print("\n=== HOLDOUT SUMMARY ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    if stats["scaffold_overlap_train_vs_holdout"] != 0:
        print("  !!! WARNING: scaffold overlap is not zero -- investigate.")
    else:
        print("  OK: train and holdout are scaffold-disjoint.")
    print(f"\n  wrote -> {out}/holdout_inchikeys.csv  (pass to BOTH models via --holdout_ids)")
    print(f"  wrote -> {out}/combined_compounds.csv, holdout_manifest.csv, holdout_stats.json")


if __name__ == "__main__":
    main()