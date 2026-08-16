#!/usr/bin/env python3
"""
herg_robust_extraction.py
=========================
ONE gated, from-scratch extraction that produces a GENUINE COMBINED hERG
dataset -- carrying BOTH the binary blocker label AND the continuous pIC50 for
every compound -- from ChEMBL and BindingDB under a single, consistent set of
quality gates and a single standardization.

WHY THIS REPLACES THE TWO OLD PIPELINES
---------------------------------------
Previously the binary set (herg_binary_extraction.py, strict ChEMBL query) and
the pIC50 set (unify_dataset.py, from pre-built Excel files of a different,
looser lineage) were built separately. That mismatch meant some compounds had a
pIC50 but no label (and vice-versa) purely because of *provenance*, not chemistry.

This script fixes it at the root:
  * ChEMBL and BindingDB are BOTH first-class sources, extracted from scratch.
  * The SAME quality gates apply to everything (confidence >= 7, human-only,
    data_validity clean, potential_duplicate cleared, mutant-assay filter for
    ChEMBL; parseable curated values for BindingDB). A compound that fails a
    gate is EXCLUDED -- so "flagged" rows are never silently re-admitted. Only
    genuine, quality-passing data survives ("true miss, not flagged").
  * The SAME row-level labeller (pooled endpoints, mechanism-gated) runs on both
    sources, so labels never diverge by source.
  * Collapse emits ONE table with hERG_blocker, pIC50 (median exact potency),
    label_source, provenance (`source`), and the conflict/discordance flags.

Because a compound with a quality-passing exact potency now gets BOTH a binary
label (pIC50 >= 5.0) and a pIC50 value in the same pass, the only compounds
without a pIC50 are the coarse-endpoint ones (% inhibition / censored) -- which
is correct -- and there are no "pIC50 but no label" compounds at all.

OUTPUT  (columns compatible with train_ml_ensemble_classifier.py, ml_pic50_regressor.py
and make_scaffold_split.py --combined)
  compound_chembl_id, standard_inchi_key, smiles,
  hERG_blocker,            # binary label (classifier target)
  pIC50,                   # continuous target (regressor); NaN if no exact potency
  median_pIC50_equiv,      # alias of pIC50 (back-compat with make_shared_holdout)
  std_pIC50_equiv, high_discordance, label_conflict,
  label_source,            # exact_potency | pct_inhibition | censored_rescue | other
  best_evidence_tier, n_measurements, n_exact_potency, n_pct_inhibition,
  n_censored, n_activator_rows, endpoints_used,
  source                   # chembl | bindingdb | bindingdb,chembl

Requirements:  pandas numpy rdkit  (+ chembl_XX.db, BindingDB hERG csv)
Self-test (no DB/BindingDB needed):  python herg_robust_extraction.py --selftest
"""

import os
import re
import time
import sqlite3
import argparse
import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

# ============================================================================
#  CONFIG
# ============================================================================
DB_FILE        = "chembl_37.db"
BINDINGDB_FILE = "BindingDB_hERG_raw.csv"
BINDINGDB_SEP  = ","
OUTPUT_DIR     = "herg_combined_dataset"
CHEMBL_ID      = "CHEMBL240"

CFG = {
    "threshold_uM": 10.0,          # blocker cutoff, pIC50 = 5.0
    "inhibition_pct": 50.0,
    "inhibition_conc_uM": 10.0,
    "inhib_conc_tol_uM": 2.0,
    "accept_unknown_conc": False,
    "include_ambiguous_potency": True,
    "filter_mutants": True,
    "strict_assay_type": False,
    "min_confidence_score": 7,
    "human_only": True,
    "standardize": True,           # RDKit Cleanup->FragmentParent->Uncharge + re-key
}
THRESHOLD_UM = float(CFG["threshold_uM"])
THRESHOLD_NM = THRESHOLD_UM * 1_000.0
THRESHOLD_PACT = 9.0 - np.log10(THRESHOLD_NM)     # 10 uM -> 5.0
INHIB_PCT = float(CFG["inhibition_pct"])
INHIB_CONC_UM = float(CFG["inhibition_conc_uM"])
INHIB_CONC_TOL_UM = float(CFG["inhib_conc_tol_uM"])

# ---- endpoint vocabulary (verbatim from herg_binary_extraction.py) ----------
MOLAR_POTENCY = {"IC50", "Ki", "Kd"}
LOG_POTENCY   = {"pIC50", "pKi", "pKd", "pKD"}
AMBIGUOUS_MOLAR = {"EC50", "AC50", "XC50", "Potency"}
AMBIGUOUS_LOG   = {"pEC50", "pAC50", "pA50", "pA2"}
AMBIGUOUS_POTENCY = AMBIGUOUS_MOLAR | AMBIGUOUS_LOG
PCT_INHIB = {"Inhibition", "% Inhibition", "%Inhib (Mean)", "INH",
             "Percent Inhibition", "% inhibition", "Inhibition (%)"}
BINDING_TYPES = {"Ki", "pKi", "Kd", "pKd", "pKD"}
FUNCTIONAL_TYPES = {"IC50", "pIC50"} | PCT_INHIB | AMBIGUOUS_POTENCY
MUTANT_STD_TYPES = {"delta pIC50 wt-mutant"}
UNIT_TO_NM = {"nM": 1.0, "nmol/L": 1.0, "nmol.L-1": 1.0,
              "uM": 1_000.0, "umol/L": 1_000.0, "µM": 1_000.0,
              "mM": 1_000_000.0, "pM": 0.001, "M": 1_000_000_000.0}
_CONC_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(nm|nmol|um|µm|umol|micromol|microm|mm|m)\b", re.I)
_INHIB_RE = re.compile(r"inhibit|block|blockade|antagon", re.I)
_ACTIVATE_RE = re.compile(r"activat|agonist|potentiat|opener|enhanc", re.I)
_MUTANT_WORD_RE = re.compile(r"\bmutant\b|\bmutation\b|\bmutated\b|wt-mutant", re.I)
_POINT_MUTATION_RE = re.compile(r"\b[ACDEFGHIKLMNPQRSTVWY]\d{3}[ACDEFGHIKLMNPQRSTVWY]\b")
_PORE_RESIDUE_RE = re.compile(r"\b(Y652|F656|S624|T623|G648|S631)\b")
_MUTATION_CUE_RE = re.compile(r"mutant|mutation|mutated|substitut|variant", re.I)


def _to_nM(value, units):
    if pd.isna(value):
        return np.nan
    f = UNIT_TO_NM.get(str(units))
    return float(value) * f if f is not None else np.nan


def _parse_conc_uM(desc):
    if not desc or pd.isna(desc):
        return np.nan
    m = _CONC_RE.search(str(desc))
    if not m:
        return np.nan
    val, unit = float(m.group(1)), m.group(2).lower()
    if unit.startswith("n"):
        return val / 1_000.0
    if unit.startswith(("u", "µ", "micro")):
        return val
    if unit == "mm":
        return val * 1_000.0
    if unit == "m":
        return val * 1_000_000.0
    return np.nan


def _classify_direction(desc):
    if desc is None or pd.isna(desc):
        return "ambiguous"
    text = str(desc)
    if not text or text.lower() == "nan":
        return "ambiguous"
    is_inhib = bool(_INHIB_RE.search(text))
    is_activ = bool(_ACTIVATE_RE.search(text))
    if is_inhib and not is_activ:
        return "inhibition"
    if is_activ and not is_inhib:
        return "activation"
    return "ambiguous"


def _is_mutant_assay(desc, atype=None):
    if atype in MUTANT_STD_TYPES:
        return True
    text = "" if (desc is None or pd.isna(desc)) else str(desc)
    if not text:
        return False
    if _MUTANT_WORD_RE.search(text):
        return True
    if _POINT_MUTATION_RE.search(text):
        return True
    if _PORE_RESIDUE_RE.search(text) and _MUTATION_CUE_RE.search(text):
        return True
    return False


def _assay_type_consistent(atype, assay_type):
    if assay_type is None or pd.isna(assay_type) or str(assay_type).strip() == "":
        return None
    at = str(assay_type).strip().upper()
    if atype in BINDING_TYPES:
        return at == "B"
    if atype in FUNCTIONAL_TYPES:
        return at == "F"
    return None


# ============================================================================
#  ROW-LEVEL BINARY LABEL  (verbatim from herg_binary_extraction.py)
# ============================================================================
def label_row(row):
    atype = row.get("activity_type")
    value = row.get("activity_value")
    units = row.get("activity_units")
    rel_raw = row.get("activity_relation")
    rel = "=" if pd.isna(rel_raw) else (str(rel_raw).strip() or "=")
    if atype is None or pd.isna(value):
        return None, None, None, None

    if CFG["filter_mutants"] and _is_mutant_assay(row.get("assay_description"), atype):
        return None, None, None, None

    if atype in LOG_POTENCY:
        if rel in ("=", "~", "", None):
            return (1 if value >= THRESHOLD_PACT else 0), 1, f"{atype}(log)", None
        if rel in (">", ">=") and value >= THRESHOLD_PACT:
            return 1, 3, f"{atype}{rel}", None
        if rel in ("<", "<=") and value <= THRESHOLD_PACT:
            return 0, 3, f"{atype}{rel}", None
        return None, None, None, None

    if atype in MOLAR_POTENCY:
        nM = _to_nM(value, units)
        if pd.isna(nM):
            return None, None, None, None
        if rel in ("=", "~", "", None):
            return (1 if nM <= THRESHOLD_NM else 0), 1, f"{atype}={units}", None
        if rel in (">", ">=") and nM >= THRESHOLD_NM:
            return 0, 3, f"{atype}{rel}{units}", None
        if rel in ("<", "<=") and nM <= THRESHOLD_NM:
            return 1, 3, f"{atype}{rel}{units}", None
        return None, None, None, None

    if atype in AMBIGUOUS_POTENCY:
        if not CFG["include_ambiguous_potency"]:
            return None, None, None, None
        direction = _classify_direction(row.get("assay_description"))
        if direction == "ambiguous":
            return None, None, None, None
        if direction == "activation":
            return 0, 1, f"{atype}-activator", "activator"
        if atype in AMBIGUOUS_LOG:
            if rel in ("=", "~", "", None):
                return (1 if value >= THRESHOLD_PACT else 0), 1, f"{atype}-inhib(log)", None
            if rel in (">", ">=") and value >= THRESHOLD_PACT:
                return 1, 3, f"{atype}-inhib{rel}", None
            if rel in ("<", "<=") and value <= THRESHOLD_PACT:
                return 0, 3, f"{atype}-inhib{rel}", None
            return None, None, None, None
        else:
            nM = _to_nM(value, units)
            if pd.isna(nM):
                return None, None, None, None
            if rel in ("=", "~", "", None):
                return (1 if nM <= THRESHOLD_NM else 0), 1, f"{atype}-inhib={units}", None
            if rel in (">", ">=") and nM >= THRESHOLD_NM:
                return 0, 3, f"{atype}-inhib{rel}{units}", None
            if rel in ("<", "<=") and nM <= THRESHOLD_NM:
                return 1, 3, f"{atype}-inhib{rel}{units}", None
            return None, None, None, None

    if atype in PCT_INHIB:
        if str(units) not in ("%", "percent", "", "None") and not pd.isna(units):
            if str(units) != "nan":
                return None, None, None, None
        conc = _parse_conc_uM(row.get("assay_description"))
        if pd.isna(conc):
            if not CFG["accept_unknown_conc"]:
                return None, None, None, None
            tier, ev = "2b", f"%inhib@?~{INHIB_CONC_UM:g}uM"
        else:
            if abs(conc - INHIB_CONC_UM) > INHIB_CONC_TOL_UM:
                return None, None, None, None
            tier, ev = "2a", f"%inhib@{conc:g}uM"
        return (1 if value >= INHIB_PCT else 0), tier, ev, None

    return None, None, None, None


def _apply_labels(df, gate_label="STEP"):
    df = df.copy()
    n0 = len(df)
    if CFG["filter_mutants"]:
        mask = df.apply(lambda r: _is_mutant_assay(r.get("assay_description"),
                                                   r.get("activity_type")), axis=1)
        n_mut = int(mask.sum())
        df = df.loc[~mask].copy()
        print(f"  [{gate_label}] mutant assays removed: {n_mut:,}")
    labels = df.apply(label_row, axis=1)
    df["blocker"] = [x[0] for x in labels]
    df["evidence_tier"] = [x[1] for x in labels]
    df["evidence"] = [x[2] for x in labels]
    df["mechanism"] = [x[3] for x in labels]
    df["assay_type_consistent"] = df.apply(
        lambda r: _assay_type_consistent(r.get("activity_type"), r.get("assay_type")), axis=1)
    if CFG["strict_assay_type"]:
        bad = (df["assay_type_consistent"] == False) & df["blocker"].notna()  # noqa: E712
        df.loc[bad, ["blocker", "evidence_tier", "evidence", "mechanism"]] = None
    decided = df[df["blocker"].notna()]
    print(f"  [{gate_label}] rows in: {n0:,} -> survived: {len(df):,} -> labelled: {len(decided):,}")
    return df


# ============================================================================
#  STANDARDIZATION (shared with the models + make_shared_holdout)
# ============================================================================
def standardize(smi):
    """(canonical_smiles, inchikey) or (None, None). Same pipeline as the models."""
    if not HAS_RDKIT or smi is None or (isinstance(smi, float) and pd.isna(smi)):
        return None, None
    try:
        base = str(smi).split(" ")[0]        # drop CXSMILES "|...|" suffix
        m = Chem.MolFromSmiles(base)
        if m is None:
            return None, None
        m = rdMolStandardize.Uncharger().uncharge(
            rdMolStandardize.FragmentParent(rdMolStandardize.Cleanup(m)))
        canon = Chem.MolToSmiles(m, canonical=True)
        m2 = Chem.MolFromSmiles(canon)
        if m2 is None:
            return None, None
        return canon, Chem.MolToInchiKey(m2)
    except (ValueError, RuntimeError):
        return None, None


# ============================================================================
#  STEP 1 - EXTRACT ChEMBL (strict gates)
# ============================================================================
def step1_chembl(db_file=DB_FILE):
    print("\n" + "=" * 65 + f"\nSTEP 1 - ChEMBL {CHEMBL_ID} extraction\n" + "=" * 65)
    con = sqlite3.connect(db_file)
    org = "AND ay.assay_organism = 'Homo sapiens'" if CFG["human_only"] else ""
    q = f"""
    SELECT td.chembl_id AS target_chembl_id, md.chembl_id AS molecule_chembl_id,
           cs.canonical_smiles AS smiles, cs.standard_inchi_key,
           act.standard_type AS activity_type, act.standard_relation AS activity_relation,
           act.standard_value AS activity_value, act.standard_units AS activity_units,
           ay.assay_type, ay.description AS assay_description, ay.confidence_score
    FROM target_dictionary td
    JOIN assays ay ON ay.tid = td.tid
    JOIN activities act ON act.assay_id = ay.assay_id
    JOIN molecule_dictionary md ON md.molregno = act.molregno
    JOIN compound_structures cs ON cs.molregno = md.molregno
    WHERE td.chembl_id = ?
      AND md.molecule_type IN ('Small molecule', 'Unknown')
      AND act.standard_value IS NOT NULL
      AND act.data_validity_comment IS NULL
      AND (act.potential_duplicate IS NULL OR act.potential_duplicate = 0)
      AND ay.confidence_score >= {int(CFG['min_confidence_score'])}
      {org};
    """
    df = pd.read_sql_query(q, con, params=[CHEMBL_ID]).drop_duplicates()
    con.close()
    df["source"] = "chembl"
    print(f"  extracted {len(df):,} rows / {df['standard_inchi_key'].nunique():,} compounds")
    return df


# ============================================================================
#  STEP 2 - EXTRACT BindingDB as a FULL source (not just rescue)
# ============================================================================
def step2_bindingdb(path=BINDINGDB_FILE, sep=BINDINGDB_SEP):
    print("\n" + "=" * 65 + "\nSTEP 2 - BindingDB extraction (full source)\n" + "=" * 65)
    if not path or not os.path.exists(path):
        print(f"  BindingDB file not found ({path}); skipping.")
        return pd.DataFrame()

    def parse_val(x):
        if pd.isna(x):
            return np.nan, "="
        m = re.match(r"^\s*(>=|<=|>|<|~)?\s*([0-9.eE+\-]+)", str(x).strip())
        if not m:
            return np.nan, "="
        return (float(m.group(2)) if m.group(2) else np.nan), (m.group(1) or "=")

    frames = []
    for chunk in pd.read_csv(path, sep=sep, chunksize=500_000, low_memory=False,
                             on_bad_lines="warn"):
        ik_col = "Ligand InChI Key" if "Ligand InChI Key" in chunk.columns else None
        smi_col = "Ligand SMILES" if "Ligand SMILES" in chunk.columns else None
        for col, atype in [("IC50 (nM)", "IC50"), ("Ki (nM)", "Ki"),
                           ("EC50 (nM)", "EC50"), ("Kd (nM)", "Kd")]:
            if col not in chunk.columns:
                continue
            sub = chunk[[c for c in (ik_col, smi_col, col) if c]].dropna(subset=[col]).copy()
            if sub.empty:
                continue
            vals = sub[col].apply(parse_val)
            sub["activity_value"] = [v[0] for v in vals]
            sub["activity_relation"] = [v[1] for v in vals]
            sub["activity_type"] = atype
            sub["activity_units"] = "nM"
            sub["assay_description"] = ""     # BindingDB carries no assay text
            sub["assay_type"] = "B"
            sub = sub.rename(columns={ik_col: "standard_inchi_key", smi_col: "smiles"})
            frames.append(sub.drop(columns=[col]))
    if not frames:
        print("  no usable IC50/Ki/EC50/Kd columns in BindingDB file.")
        return pd.DataFrame()
    bdb = pd.concat(frames, ignore_index=True)
    bdb["molecule_chembl_id"] = np.nan
    bdb["confidence_score"] = np.nan
    bdb["source"] = "bindingdb"
    # NOTE: BindingDB rows have no assay description -> cannot be mutant-filtered
    # or confidence-gated; they are curated binding values, tagged source=bindingdb
    # for full auditability. hERG BindingDB assays are wild-type in practice.
    print(f"  parsed {len(bdb):,} BindingDB activity rows / "
          f"{bdb['standard_inchi_key'].nunique():,} compounds")
    return bdb


# ============================================================================
#  STEP 3 - COLLAPSE TO ONE ROW PER COMPOUND (binary label + pIC50 + source)
# ============================================================================
def step3_collapse(df, output_dir=OUTPUT_DIR, output_file="herg_combined_dataset.csv"):
    print("\n" + "=" * 65 + "\nSTEP 3 - Collapse to compound level (combined)\n" + "=" * 65)
    os.makedirs(output_dir, exist_ok=True)
    d = df[df["blocker"].notna()].copy()

    def row_pact(r):
        a = r["activity_type"]
        rel_raw = r.get("activity_relation")
        rel = "=" if pd.isna(rel_raw) else (str(rel_raw).strip() or "=")
        if rel not in ("=", "~", "", None):
            return np.nan
        if r.get("mechanism") == "activator":
            return np.nan
        if a in LOG_POTENCY or a in AMBIGUOUS_LOG:
            return float(r["activity_value"])
        if a in MOLAR_POTENCY or a in AMBIGUOUS_MOLAR:
            nM = _to_nM(r["activity_value"], r["activity_units"])
            return 9.0 - np.log10(nM) if pd.notna(nM) and nM > 0 else np.nan
        return np.nan
    d["pact"] = d.apply(row_pact, axis=1)

    # optional re-standardization to a consistent key across ChEMBL + BindingDB
    if CFG["standardize"] and HAS_RDKIT:
        print("  standardizing SMILES and re-keying on a consistent InChIKey ...")
        canon_ik = d["smiles"].map(standardize)
        d["canonical_smiles"] = [c for c, _ in canon_ik]
        d["std_key"] = [k for _, k in canon_ik]
        d = d[d["std_key"].notna()].copy()
        key_col = "std_key"
    else:
        d["canonical_smiles"] = d["smiles"]
        key_col = "standard_inchi_key"

    recs = []
    for key, sub in d.groupby(key_col):
        smiles = sub["canonical_smiles"].dropna().iloc[0] if sub["canonical_smiles"].notna().any() else None
        cid = (sub["molecule_chembl_id"].dropna().iloc[0]
               if sub["molecule_chembl_id"].notna().any() else key)
        exact = sub[sub["pact"].notna()]
        pct = sub[sub["evidence_tier"].isin(["2a", "2b"])]
        cens = sub[sub["evidence_tier"] == 3]
        median_pact = float(exact["pact"].median()) if len(exact) else np.nan
        std_pact = float(exact["pact"].std()) if len(exact) > 1 else np.nan
        if len(exact):
            label, lsource = int(median_pact >= THRESHOLD_PACT), "exact_potency"
        elif len(pct):
            label, lsource = int(pct["blocker"].mean() >= 0.5), "pct_inhibition"
        elif len(cens):
            label, lsource = int(cens["blocker"].mean() >= 0.5), "censored_rescue"
        else:
            label, lsource = int(sub["blocker"].mean() >= 0.5), "other"
        recs.append({
            "compound_chembl_id": cid,
            "standard_inchi_key": key,
            "smiles": smiles,
            "hERG_blocker": label,
            "pIC50": round(median_pact, 3) if pd.notna(median_pact) else np.nan,
            "median_pIC50_equiv": round(median_pact, 3) if pd.notna(median_pact) else np.nan,
            "std_pIC50_equiv": round(std_pact, 3) if pd.notna(std_pact) else np.nan,
            "high_discordance": bool(pd.notna(std_pact) and std_pact > 1.0),
            "label_conflict": sub["blocker"].nunique() > 1,
            "label_source": lsource,
            "best_evidence_tier": sorted(sub["evidence_tier"].dropna().astype(str))[0],
            "n_measurements": len(sub),
            "n_exact_potency": len(exact),
            "n_pct_inhibition": len(pct),
            "n_censored": len(cens),
            "n_activator_rows": int((sub["mechanism"] == "activator").sum()),
            "endpoints_used": "|".join(sorted(set(sub["activity_type"].dropna()))),
            "source": ",".join(sorted(set(sub["source"].dropna()))),
        })

    out = pd.DataFrame(recs)
    path = os.path.join(output_dir, output_file)
    out.to_csv(path, index=False)

    n1 = int((out["hERG_blocker"] == 1).sum()); n0 = int((out["hERG_blocker"] == 0).sum())
    n_pic50 = int(out["pIC50"].notna().sum())
    print(f"  unique compounds        : {len(out):,}")
    print(f"  blocker=1 / blocker=0   : {n1:,} / {n0:,}  ({n1/len(out):.1%} blocker)")
    print(f"  with continuous pIC50   : {n_pic50:,} ({n_pic50/len(out):.1%})")
    print(f"  label_source breakdown  :\n{out['label_source'].value_counts().to_string()}")
    print(f"  source provenance       :\n{out['source'].value_counts().to_string()}")
    print(f"  pIC50-but-no-label      : "
          f"{int((out['pIC50'].notna() & out['hERG_blocker'].isna()).sum()):,} "
          f"(should be 0 -- single pipeline)")
    print(f"  saved -> {path}")

    # convenience task views
    out.to_csv(os.path.join(output_dir, "herg_binary_ml_dataset.csv"), index=False)
    reg = out[out["pIC50"].notna()][["smiles", "pIC50", "source"]].rename(
        columns={"pIC50": "pIC50"})
    reg.to_csv(os.path.join(output_dir, "herg_pic50_unified.csv"), index=False)
    print(f"  also wrote task views -> herg_binary_ml_dataset.csv, herg_pic50_unified.csv")
    return out


# ============================================================================
#  SELF-TEST (no DB / BindingDB needed)
# ============================================================================
def _selftest():
    print("\nSELF-TEST - label_row acceptance checks")
    passed = 0

    def check(name, cond):
        nonlocal passed
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        assert cond, name
        passed += 1

    check("IC50=5uM -> blocker",
          label_row({"activity_type": "IC50", "activity_value": 5.0, "activity_units": "uM",
                     "activity_relation": "=", "assay_description": ""})[0] == 1)
    check("IC50=20uM -> non-blocker",
          label_row({"activity_type": "IC50", "activity_value": 20.0, "activity_units": "uM",
                     "activity_relation": "=", "assay_description": ""})[0] == 0)
    r = label_row({"activity_type": "EC50", "activity_value": 2.0, "activity_units": "uM",
                   "activity_relation": "=", "assay_description": "hERG activation / agonist"})
    check("EC50 activator wording -> non-blocker, mechanism=activator",
          r[0] == 0 and r[3] == "activator")
    check("EC50 inhibition wording -> blocker",
          label_row({"activity_type": "EC50", "activity_value": 2.0, "activity_units": "uM",
                     "activity_relation": "=", "assay_description": "hERG inhibition"})[0] == 1)
    check("EC50 silent -> dropped",
          label_row({"activity_type": "EC50", "activity_value": 2.0, "activity_units": "uM",
                     "activity_relation": "=", "assay_description": ""})[0] is None)
    check("%inhib 80 @10uM -> blocker",
          label_row({"activity_type": "Inhibition", "activity_value": 80.0, "activity_units": "%",
                     "activity_relation": "=", "assay_description": "inhibition at 10 uM"})[0] == 1)
    check("%inhib 80 @1uM -> dropped",
          label_row({"activity_type": "Inhibition", "activity_value": 80.0, "activity_units": "%",
                     "activity_relation": "=", "assay_description": "inhibition at 1 uM"})[0] is None)
    check("Ka -> dropped; ED50 -> dropped",
          label_row({"activity_type": "Ka", "activity_value": 1e7, "activity_units": "M-1",
                     "activity_relation": "=", "assay_description": ""})[0] is None and
          label_row({"activity_type": "ED50", "activity_value": 3.0, "activity_units": "uM",
                     "activity_relation": "=", "assay_description": ""})[0] is None)
    check("pIC50 6.5 on Y652A mutant -> dropped",
          label_row({"activity_type": "pIC50", "activity_value": 6.5, "activity_units": None,
                     "activity_relation": "=", "assay_description": "hERG Y652A mutant"})[0] is None)
    check("IC50 <5uM censored -> blocker",
          label_row({"activity_type": "IC50", "activity_value": 5.0, "activity_units": "uM",
                     "activity_relation": "<", "assay_description": ""})[0] == 1)
    print(f"\n  {passed}/10 acceptance tests passed.")


# ============================================================================
#  MAIN
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description="Robust combined hERG extraction (binary + pIC50).")
    ap.add_argument("--db", default=DB_FILE, help="ChEMBL SQLite file.")
    ap.add_argument("--bindingdb", default=BINDINGDB_FILE, help="BindingDB hERG CSV (or '' to skip).")
    ap.add_argument("--out_dir", default=OUTPUT_DIR)
    ap.add_argument("--keep_mutants", action="store_true")
    ap.add_argument("--strict_assay_type", action="store_true")
    ap.add_argument("--no_standardize", action="store_true",
                    help="Skip RDKit re-standardization/re-keying (use raw InChIKeys).")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.keep_mutants:
        CFG["filter_mutants"] = False
    if args.strict_assay_type:
        CFG["strict_assay_type"] = True
    if args.no_standardize:
        CFG["standardize"] = False

    if args.selftest:
        _selftest()
        raise SystemExit(0)

    t0 = time.time()
    chembl = step1_chembl(args.db)
    bdb = step2_bindingdb(args.bindingdb) if args.bindingdb else pd.DataFrame()

    rows = pd.concat([chembl, bdb], ignore_index=True, sort=False) if len(bdb) else chembl
    rows = _apply_labels(rows, gate_label="LABEL")
    step3_collapse(rows, output_dir=args.out_dir)
    print(f"\nPIPELINE COMPLETE in {time.time()-t0:.1f}s -> {args.out_dir}/herg_combined_dataset.csv")


if __name__ == "__main__":
    main()