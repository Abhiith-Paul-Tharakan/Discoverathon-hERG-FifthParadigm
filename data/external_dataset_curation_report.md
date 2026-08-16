# Curation of the External Clinical-Cardiotoxicity Validation Set

*Project: hERG Channel Blockade (Novartis Discoverathon, Challenge 2 · Safety)*
*Purpose: document, reproducibly, how we assembled an independent external set (DICTrank + CredibleMeds QTdrugs) to validate our ChEMBL-trained hERG model.*

---

## 1. Objective and design principle

The external set exists to test whether our hERG model generalizes **beyond the ChEMBL bioactivity data it was trained on**, against an *independent, downstream, clinically-derived* endpoint. Curation was therefore governed by three rules: (1) the external labels are used only for validation, never for training; (2) every compound is reduced to a canonical structure so it can be matched to our training data and held out cleanly; and (3) every processing decision is recorded so the pipeline is reproducible and its biases are auditable.

A companion document, `external_dataset_honesty_report.md`, discusses *what kind of data this is* (expert/regulatory-curated proxy labels, not raw clinical measurements). This document covers *how it was built*.

## 2. Data sources

| Source | What it provides | Version / access | Rows ingested |
|---|---|---|---|
| **DICTrank** (FDA Drug-Induced Cardiotoxicity Rank) | Drug-level cardiotoxicity concern categories (most / less / no), paired with standardized SMILES + InChI | Structure-paired file from the Seal et al. (2024) DICTrank Predictor repository, `data/binarised/DICTrank/DICTrank_binarised_DataWarrior.csv` | 1,020 |
| **CredibleMeds QTdrugs** | Drug-level TdP risk categories: Known (KR) / Possible (PR) / Conditional (CR) / Special (SR) risk; **names only, no structures** | "Drugs to Avoid in Congenital Long QT" PDF (English), QTdrugs list rev. 16 April 2026 | 323 |

**Acquisition note (reproducibility caveat).** In our compute environment the FDA host and PubChem were network-blocked. We therefore obtained DICTrank with structures already attached from the open Seal et al. GitHub repository (cloned directly), and resolved CredibleMeds structures against *offline* lookup tables (Section 5) rather than a live name-resolution API. The DICTrank labels are identical to the FDA release; only the delivery route differed.

We deliberately took the structure-paired DICTrank file (which already carries `Standardized_SMILES`, `Standardized_InChI`, the `DICT _ Concern` category, and a binary `DICTrank` flag) rather than the raw FDA name list, eliminating a name→structure mapping step for the larger of the two sources.

## 3. Parsing the CredibleMeds PDF

CredibleMeds distributes the list as a wrapped, three-pair (Generic/Brand × 3) multi-column PDF, which does not extract cleanly as linear text. We parsed it by **word coordinates** with `pdfplumber`:

- Each drug entry ends in a risk code token — `(KR)`, `(PR)`, `(CR)`, or `(SR)`. We located these, and for each page **derived the three generic-name column x-bands from that page's own code-token positions** (rather than fixed bands), which corrected a column drift that otherwise let brand names leak into the generic column on later pages.
- Within each column band, words were accumulated in reading order and flushed into a `(name, category)` record at each code token — a scheme that also correctly reassembles names that wrap across two lines.
- Cleanup rules removed the intro/header block, stripped leaked brand fragments and stray "and others" text, and normalized whitespace and the extended-release "- ER" suffix.

The result was **323 unique drugs** — Known 24 · Possible … (full pre-dedup counts: KR 69, PR 159, CR 54, SR 41). We retained the Special-Risk (SR) drugs present in this particular list but flag them specially downstream (Section 6), because SR drugs do not prolong QT by a hERG mechanism.

## 4. Schema unification

The two sources were stacked into a single table with an explicit **`source`** column and a superset schema:

`source, generic_name, brand_name, active_ingredient, smiles, inchi, inchikey, risk_label, risk_scheme, cardiotox_binary, smiles_source`

Because the two sources grade risk on *different* scales, a **`risk_scheme`** column (`DICTrank_concern` vs `CredibleMeds_TdP`) preserves that distinction so the categories are never accidentally pooled into one ordinal.

## 5. Structure standardization and offline SMILES resolution

All structures were standardized with **RDKit 2026.03.5**: parse SMILES, keep the **largest organic fragment** (stripping salts/counter-ions), emit a canonical SMILES and an **InChIKey**. Matching keys on the InChIKey are used throughout, which makes joins robust to salt form and (via the connectivity block) to protonation and stereochemistry.

CredibleMeds ships names only, so its structures were resolved against a combined **offline name→SMILES index** built from tables shipped in the DICTrank repository:

| Lookup table | Records |
|---|---|
| DrugBank | 11,579 |
| ChEMBL / MoA (`chemicalinfo`) | 13,553 |
| EPA CompTox (DTXSID, DICTrank "Generic Names") | 1,382 |
| LINCS L1000 (`meta_SMILES`) | 41,774 |
| **Combined normalized name keys** | **~104,087** |

Name normalization lowercased, split parenthetical/slash synonyms into alternate keys, stripped ~60 salt/hydrate tokens, and generated spaceless variants, so "Dolasetron", "dolasetron mesylate", and "Dolasetron (mesylate)" all collide.

## 6. Deduplication (two passes, DICTrank preferred)

Where a drug appears in both sources we keep the DICTrank row (it already carries a curated structure) and drop the CredibleMeds duplicate:

1. **Name-based pass** — normalized name / synonym / salt-stripped match of each CredibleMeds drug against DICTrank generic *and* active-ingredient names → **210 duplicates removed**.
2. **Structure-based pass** — after SMILES resolution, any remaining CredibleMeds drug whose InChIKey connectivity block equaled a DICTrank compound was dropped as a synonym the name pass missed → **1 further removed** (Ofloxacin).

**`cardiotox_binary`** was set from DICTrank's own binary label for DICTrank rows; for CredibleMeds, KR/PR/CR → 1 and **SR → blank (NA)**, since Special-Risk drugs are not hERG-mediated QT prolongers and should not be counted as positive cardiotox labels.

## 7. Leakage-safe mapping onto the training data

Using the training package (`Data.zip`: 16,650 ChEMBL compounds, scaffold-split **12,803 train / 3,236 holdout** by connectivity block, seed 42, scaffold overlap 0), each clinical compound was assigned by InChIKey connectivity block to *train*, *holdout*, or *external(new)*:

- **957 external** (unseen by the model) — the validation set
- **118 already in training** — excluded from any external claim
- **23 in the existing holdout** — already scored as test compounds

The external validation set is the 957 unseen compounds that carry a resolved structure.

## 8. Outputs

| File | Rows | Contents |
|---|---|---|
| `clinical_cardiotox_combined.csv` | **1,132** | Full deduped set (DICTrank 1,020 + CredibleMeds-only 112); 1,122 with structures, 10 without |
| `clinical_external_validation_set.csv` | **957** | Model-unseen clinical drugs with SMILES + labels, ready to score (DICTrank 877 + CredibleMeds 80) |
| `external_dataset_honesty_report.md` | — | What kind of data this is, and how to word claims |

Combined-set label distributions: DICTrank — less 443 · most 299 · no 278; CredibleMeds-only (post-dedup) — Possible 53 · Special 24 · Known 24 · Conditional 11.

## 9. Compounds intentionally left without a structure (10)

We did **not** fabricate SMILES. Ten CredibleMeds-only drugs remain unresolved and are excluded from the scored external set: **2 combination products** (Fluticasone+Salmeterol, Sulfamethoxazole+Trimethoprim — no single molecule), **4 biologics** (Necitumumab, Tebentafusp, Inotuzumab ozogamicin, Motixafortide — no small-molecule structure, and out of scope for a SMILES model), and **4 post-2023 small molecules** absent from the offline tables (Revumenib, Mavorixafor, Milsaperidone, Ibogaine).

## 10. Curation validity check

As an internal sanity check, on the drugs that overlap our ChEMBL data the hERG blocker rate rises monotonically with DICTrank concern (no 42% → less 53% → most 71%; Known-Risk TdP drugs 88%), separation AUC ≈ 0.65. A modest but monotonic signal is the expected result for a correctly-curated *downstream* proxy — evidence the mapping and labels are aligned, without the suspiciously-perfect agreement that would signal leakage.

## 11. Environment and reproducibility notes

Python 3.11; pandas; pdfplumber; RDKit 2026.03.5. Structure matching uses the InChIKey connectivity block (stereo/protonation-insensitive by design). Because live PubChem/FDA access was unavailable, structures came from offline tables and the structure-paired DICTrank file; a rerun with online resolution could recover a few of the 10 unresolved small molecules but would not change the DICTrank-derived majority of the set.

---

### Sources

- Seal, S., et al. *Insights into Drug Cardiotoxicity from Biological and Chemical Data: The First Public Classifiers for FDA Drug-Induced Cardiotoxicity Rank.* J. Chem. Inf. Model. 2024, 64, 1172–1186. (DICTrank provenance; structure-paired data repository.)
- U.S. FDA — Drug-Induced Cardiotoxicity Rank (DICTrank), 2023; 1,318 drugs, four DICT-Concern categories.
- CredibleMeds® / AZCERT — QTdrugs List, "Drugs to Avoid in Congenital Long QT" (English), rev. 16 April 2026; ADECA™ evidence-assessment process. https://crediblemeds.org
- Reference lookup tables (DrugBank, ChEMBL/MoA, EPA CompTox, LINCS L1000) as bundled in the Seal et al. DICTrank repository.
