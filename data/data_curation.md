# Training & Held-out Data Curation — hERG Channel Blockade

*Discoverathon 2026, Challenge 2 (Safety). This document describes how the training and held-out
datasets were assembled, curated, labelled, standardized, and split. It covers the **internal**
hERG bioactivity data only; the independent external clinical set is documented separately in
`external_dataset_curation_report.md` (and `external_dataset_honesty_report.md`).*

Reproduced by: `model_code/herg_robust_extraction.py` (extraction + labelling + standardization) and
`model_code/make_scaffold_split.py` (held-out split). All figures below are the actual output of those
scripts.

---

## 1. Summary

A single, gated, from-scratch extraction produces one combined table that carries **both** the
binary blocker label **and** the continuous pIC50 for every compound, from ChEMBL and BindingDB,
under one consistent set of quality gates and one standardization.

| | |
|---|---|
| Target | hERG / KCNH2 (ChEMBL target **CHEMBL240**) |
| Sources | ChEMBL 37 + BindingDB (hERG) |
| Final compounds (unique, standardized) | **16,650** |
| hERG blockers / non-blockers | **8,129 / 8,521** (48.8% positive) |
| Compounds with an exact pIC50 | **10,316** |
| Held-out test set | **3,331** (20%, scaffold split, seed 42, zero scaffold overlap) |
| Training pool | **13,319** |

---

## 2. Sources and quality gates

Both sources are treated as first-class and extracted from scratch, so labels never diverge by
provenance. A row that fails **any** gate is **excluded** — flagged/low-quality rows are never
silently re-admitted (a "true miss, not flagged" policy).

**ChEMBL 37** (`CHEMBL240`), gates applied:
- `confidence_score >= 7` (assay-to-target confidence)
- Human target only
- `data_validity_comment` clean (no flagged validity issues)
- `potential_duplicate` cleared
- Mutant-assay filter — assays on hERG mutants are removed (wild-type liability only)

**BindingDB (hERG)**, gates applied:
- Only parseable, curated activity values with usable units are kept

Both sources then pass through the **same** row-level labeller and the **same** standardization, so
a compound's label depends on its chemistry and evidence, not on which database it came from.

Provenance is recorded per compound in the `source` column (`chembl` = 14,110, `bindingdb` = 2,121,
`bindingdb,chembl` = 419).

---

## 3. Activity definition and row-level labelling

**Blocker threshold:** a hERG blocker is defined at **10 µM (pIC50 = 5.0)**. A compound is a blocker
if its aggregated exact potency is pIC50 ≥ 5.0 (IC50 ≤ 10 µM).

Each measured row is labelled by endpoint type, with relation-aware ("censored") handling:

- **Exact log potency** (pIC50/pKi-type, relation `=`/`~`): blocker if value ≥ 5.0, else non-blocker
  (evidence tier 1).
- **Exact molar potency** (IC50/Ki in nM/µM, relation `=`/`~`): blocker if IC50 ≤ 10 µM (tier 1).
- **Censored potency** (inequality relations): an IC50 reported as `> 10 µM` (or pIC50 `< 5.0`) is a
  confident **non-blocker**; an IC50 `< (potent value)` is a confident **blocker** (evidence tier 3).
  These carry a label but no point pIC50.
- **% inhibition** (functional screens): a compound inhibiting **≥ 50% at 10 µM (±2 µM)** is a
  blocker, else non-blocker (evidence tiers 2a/2b). Rows with unknown concentration are not accepted.
- **Mechanism gating:** assays describing channel **activation** are labelled non-blocker and tagged
  `mechanism = activator` (they are potentiators, not blockers). Genuinely ambiguous-direction rows
  are excluded rather than guessed.

---

## 4. Standardization

Every structure is standardized with RDKit before de-duplication:

`Cleanup → FragmentParent (largest organic fragment) → Uncharge`, then re-keyed on the resulting
**InChIKey**. Salts, counter-ions and charge states are normalized away, and matching is done on the
InChIKey connectivity so the same molecule from different sources collapses to one record.

---

## 5. Per-compound aggregation

Rows are grouped by standardized **InChIKey** and collapsed into one record per compound:

- **pIC50** = median of that compound's exact-potency values (`NaN` if it has only coarse endpoints).
- **Label precedence:** exact potency → % inhibition → censored → other. The strongest available
  evidence sets the label:
  - exact potency present → `hERG_blocker = (median pIC50 ≥ 5.0)`, `label_source = exact_potency`
  - else % inhibition → majority vote of its % inhibition rows, `label_source = pct_inhibition`
  - else censored → majority vote of its censored rows, `label_source = censored_rescue`
  - else `label_source = other`
- **Quality flags** (kept as columns, not used to drop compounds):
  - `high_discordance` — SD of the compound's exact pIC50 values > 1.0 log unit (noisy measurements)
  - `label_conflict` — the compound's rows disagree on the binary label

**Resulting `label_source` composition:** exact_potency 10,316 · censored_rescue 3,787 ·
pct_inhibition 2,544 · other 3. Only coarse-endpoint compounds (% inhibition / censored) lack a
point pIC50 — by design — so there are no "pIC50 but no label" or "label but no evidence" compounds.

---

## 6. Final dataset schema

`data/herg_combined_dataset/herg_combined_dataset.csv` (16,650 rows):

| Column | Meaning |
|---|---|
| `compound_chembl_id` | Compound identifier |
| `standard_inchi_key` | Standardized InChIKey (de-dup key) |
| `smiles` | Standardized SMILES |
| `hERG_blocker` | Binary label — classifier target (1 = blocker) |
| `pIC50` | Median exact potency — regressor target (`NaN` if none) |
| `median_pIC50_equiv` | Alias of `pIC50` (back-compat) |
| `std_pIC50_equiv` | SD of exact pIC50 values |
| `high_discordance` | SD > 1.0 log unit |
| `label_conflict` | Rows disagree on the label |
| `label_source` | exact_potency / pct_inhibition / censored_rescue / other |
| `best_evidence_tier` | Strongest evidence tier for the compound |
| `n_measurements`, `n_exact_potency`, `n_pct_inhibition`, `n_censored`, `n_activator_rows` | Evidence counts |
| `endpoints_used` | Distinct activity types pooled |
| `source` | chembl / bindingdb / bindingdb,chembl |

Two derived views are emitted alongside it: `herg_binary_ml_dataset.csv` (classifier) and
`herg_pic50_unified.csv` (regressor).

---

## 7. Held-out split (leakage control)

The 20% held-out test set is carved once, up front, and shared by every model, via
`make_scaffold_split.py` (seed 42):

- **Scaffold split** on **Bemis–Murcko** scaffolds using `StratifiedGroupKFold` — compounds are
  grouped by scaffold and stratified by the blocker label, so the split is class-balanced **and**
  scaffold-disjoint.
- **Zero scaffold overlap** between train and held-out (`scaffold_overlap_train_vs_holdout = 0`) —
  a stricter, leakage-controlled protocol than a random split: the model is never tested on a
  scaffold it trained on.

| | Overall | Train | Held-out |
|---|---|---|---|
| Compounds | 16,650 | 13,319 | 3,331 (20.0%) |
| Blocker fraction | 0.4882 | 0.4882 | 0.4884 |
| With exact pIC50 | 10,316 | 8,258 | 2,058 |
| Distinct held-out scaffolds | — | — | 1,586 |

Class balance is preserved to within 0.0002, and all classification metrics reported for the models
are computed on these 3,331 held-out compounds. The `data/holdout_out/` manifest (`holdout_inchikeys.csv`,
`holdout_manifest.csv`, `holdout_stats.json`) records exactly which compounds are held out.

---

## 8. Reproducibility

```bash
# 1. Rebuild the combined dataset (needs chembl_37.db + BindingDB hERG csv)
python model_code/herg_robust_extraction.py

# 2. Recreate the exact 20% scaffold-split held-out set (seed 42)
python model_code/make_scaffold_split.py --combined data/herg_combined_dataset/herg_combined_dataset.csv
```

The committed `data/herg_combined_dataset/` and `data/holdout_out/` are the exact artifacts these
commands produce; the held-out split is deterministic under seed 42.

---

## 9. Known limitations (honest notes)

- **Threshold choice.** The 10 µM (pIC50 5.0) blocker cutoff is a standard but arbitrary boundary;
  compounds near it are inherently label-noisy. `high_discordance` and `label_conflict` flag the
  compounds where measurements disagree so they can be down-weighted or inspected.
- **Cross-assay aggregation.** Exact potencies are pooled across heterogeneous assays and both
  sources before taking the median; this trades some assay-specific precision for coverage.
- **Coarse-endpoint labels.** ~38% of compounds are labelled from % inhibition or censored values
  and carry no point pIC50; their labels are reliable for classification but do not contribute to
  the regression target.
- **hERG ≠ clinical cardiotoxicity.** This dataset measures *in-vitro hERG blockade*. The separate
  external clinical set and its honesty report bound what can be claimed about clinical outcomes.
