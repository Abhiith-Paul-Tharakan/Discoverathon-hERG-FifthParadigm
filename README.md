# Predicting hERG Channel Blockade

**Novartis Discoverathon 2026 · Challenge 2 (Safety) · Team FifthParadigm**

Machine-learning models that predict **hERG (KCNH2) channel blockade** — a blocker probability and a
pIC50 potency — directly from a molecule's SMILES, with scaffold-split held-out evaluation, an
applicability-domain confidence layer, and an honest external check against clinically-annotated drugs.

---

## Overview

Drug-induced hERG blockade causes QT prolongation and life-threatening arrhythmias, and is a leading
cause of late-stage drug attrition. This project assembles a 16,650-compound hERG bioactivity dataset
from public sources, trains and compares four model architectures, and ships a single wired
production model — the **MasterModel** — that takes a SMILES in and returns a blocker probability, a
predicted pIC50, and a confidence tier.

We deliberately report two things: how well the model does on its trained task (in-vitro hERG,
held-out set), and how well it transfers to an independent clinical cardiotoxicity set — and we are
explicit about the gap between them.

## Key results

| | Held-out (in-vitro hERG) | External (clinical cardiotox) |
|---|---|---|
| **ROC-AUC (MasterModel)** | **0.832** | 0.610 |
| Accuracy / F1 | 0.756 / 0.749 | — (read ROC-AUC) |
| pIC50 regression | R² ≈ 0.41, RMSE ≈ 0.68 | (no external pIC50 labels) |

- Held-out set: 3,331 compounds, 20% scaffold split (seed 42, **zero scaffold overlap**).
- External set: 957 marketed drugs (DICTrank + CredibleMeds), **validation only**, never trained on.
- `master_predict.py` **reproduces `submission.csv` on the held-out set to within 1e-4** (pIC50 exact),
  verified from a clean clone.

## The final model — MasterModel

`model_code/master_predict.py` — one SMILES in → blocker probability + predicted pIC50 + confidence tier.

- **Classification:** `P(blocker) = 0.770 · P(ML Ensemble) + 0.230 · P(fine-tuned ChemBERTa)`
  (weights optimized on out-of-fold ROC-AUC).
- **Regression (pIC50):** the **ML-Ensemble regressor alone** (blending diluted it on held-out data).
- **No GNN** in the final model — the GNN and other variants were evaluated during model selection
  (see [Model evaluation](#model-evaluation)) and are kept only to reproduce those comparisons.
- Every prediction also carries an **applicability-domain flag**, **boundary-instability flag**, and a
  **HIGH / MEDIUM / LOW confidence tier**.

## Repository structure

```
model_code/                     # the deployed pipeline + training scripts
  master_predict.py             # ← final wired model (entry point)
  train_ml_ensemble_classifier.py   ml_pic50_regressor.py   finetune_chemberta_pure.py
  make_scaffold_split.py        herg_robust_extraction.py   assemble_submission.py
  ml_ensemble_holdout_predictions/  # committed prediction files behind the reported metrics
external_validation/            # final-model external scoring (ML + FT-ChemBERTa + blend)
  predict_ml_ensemble.py  predict_ft_chemberta.py  build_result_analysis.py
  confusion_matrix_*.csv  trust_filter_ablation.csv  outputs/
checkpoints/                    # model weights (large one via Release — see below)
data/                           # training data, held-out split, external set, curation docs
reports/                        # technical report (PDF/DOCX)
presentation/                   # 5-minute deck
submission.csv  submission_extended.csv     # held-out predictions (rubric format + trust columns)
download_checkpoints.sh  SHA256SUMS.txt  requirements.txt
```

## Installation

```bash
git clone https://github.com/Abhiith-Paul-Tharakan/Discoverathon-hERG-FifthParadigm.git
cd Discoverathon-hERG-FifthParadigm
pip install -r requirements.txt      # Python 3.11; scikit-learn 1.8.0, torch 2.5.1 pinned
bash download_checkpoints.sh         # fetches the one >100 MB checkpoint from the GitHub Release
```

`download_checkpoints.sh` pulls `herg_combined_dataset_holdout.joblib` (~318 MB, too large for git)
from the `checkpoints-v1` Release and verifies **all four** checkpoints against `SHA256SUMS.txt`. The
other three (`pic50_regressor.joblib`, `cb_ft_final.pt`, `cb_ft_scaler.joblib`) ship in the repo.

## Usage

```bash
# single molecule
python model_code/master_predict.py --smiles "CC(=O)Nc1ccc(O)cc1"

# batch over a CSV
python model_code/master_predict.py --input compounds.csv --smiles_col smiles \
    --id_col compound_id --out predictions.csv
```

Output columns: `compound_id, hERG_blocker_probability, predicted hERG pIC50` (plus the ML/FT
component probabilities in the extended output).

## Reproducing our results

**Held-out metrics** are backed by the committed prediction files in
`model_code/ml_ensemble_holdout_predictions/` and `submission.csv`; recompute them with
`external_validation/build_result_analysis.py`.

**End-to-end reproducibility check** — regenerate the held-out predictions with the deployed model
and compare to the committed `submission.csv`:

```bash
python model_code/master_predict.py --input data/holdout_out/holdout_input.csv \
    --smiles_col smiles --id_col compound_id --out holdout_repredict.csv
# compare holdout_repredict.csv to submission.csv on compound_id
```

This reproduces `submission.csv` to within **1e-4** on probability and **exactly** on pIC50 across all
3,331 held-out compounds.

**External validation:**

```bash
python external_validation/predict_ml_ensemble.py     # ML probs/pIC50 on the clinical set
python external_validation/predict_ft_chemberta.py     # FT-ChemBERTa probs on the clinical set
python external_validation/build_result_analysis.py    # blend 0.77/0.23, score, trust-filter table
```

## Model evaluation

Four architectures were trained and compared on the same held-out scaffold split; the two strongest
and most complementary were blended into the MasterModel.

| Model | Held-out ROC-AUC |
|---|---|
| ML Ensemble (Morgan + descriptors, GBM stack) | 0.823 |
| GNN (AttentiveFP) | 0.791 |
| ChemBERTa (frozen featurizer) | 0.772 |
| Fine-tuned ChemBERTa (pure) | 0.802 |
| Blend: GNN + ML | 0.826 |
| **MasterModel (ML + FT-ChemBERTa)** | **0.832** |

Fine-tuning ChemBERTa end-to-end lifted it from 0.772 → 0.802, and its errors were complementary to
the ML ensemble's — so the ML + FT blend beat every single model and every other blend.

## Applicability domain & confidence

The ML Ensemble emits, per compound, a k-NN Tanimoto **applicability-domain (AD)** flag (threshold
≈ 0.494), a **boundary-instability** flag, and a combined **confidence tier** (HIGH 1,916 / MEDIUM
382 / LOW 1,033 on the held-out set). These are validated on held-out data — in-domain predictions
(ROC-AUC 0.858) clearly beat out-of-domain (0.708) — and support a selective-prediction workflow:
keeping only HIGH-confidence predictions raises ROC-AUC to **0.889 at 58% coverage**
(`external_validation/trust_filter_ablation.csv`).

## Data & curation

- **Training / held-out:** 16,650 compounds from ChEMBL 37 (CHEMBL240) + BindingDB, one gated
  from-scratch extraction, RDKit-standardized, labelled at 10 µM (pIC50 5.0). Full methodology,
  thresholds, quality gates, and split protocol are in **`data/data_curation.md`**.
- **External:** 957 marketed drugs from FDA DICTrank + CredibleMeds QTdrugs — a curated,
  clinically-*derived* proxy, not raw clinical data. See **`data/external_dataset_curation_report.md`**
  and **`data/external_dataset_honesty_report.md`**.

## Limitations

- **hERG ≠ clinical cardiotoxicity.** The model predicts in-vitro hERG blockade. On the external
  clinical set ROC-AUC drops to ~0.61 — an honest gap driven by (1) chemical-space shift (86% of
  external drugs lie beyond 0.40 Tanimoto to any training compound) and (2) a label-definition shift
  (hERG is only one of several cardiotoxicity mechanisms). Filtering to in-domain compounds does not
  close it, isolating the label mismatch as the binding limit.
- **Not a hard safety filter.** Passing the model ≠ safe to synthesize. In a safety-screening role,
  prioritize sensitivity (lower the threshold) and pair predictions with the confidence tiers.
- **Bounded sensitivity / precision.** Sensitivity is capped by non-hERG cardiotoxicity mechanisms;
  precision is capped by the model's blindness to dose, exposure, and multi-channel effects.

## Technical notes

- `requirements.txt` pins **scikit-learn 1.8.0** and **torch 2.5.1** (the versions the checkpoints were
  produced with). `transformers` is ranged `>=4.40` — the exact fine-tuning version wasn't captured in
  the environment snapshot, but the checkpoint is a raw state_dict and reproduces `submission.csv`
  identically across transformers 4.4x–5.x (verified).
- `master_predict.py` does not recompute the AD flag for ad-hoc single-SMILES queries; the AD/trust
  columns are derived for the held-out and external sets by `assemble_submission.py` /
  `build_result_analysis.py`.

## Integrity & acknowledgments

- **Data:** ChEMBL 37 + BindingDB (hERG bioactivity); FDA DICTrank + CredibleMeds QTdrugs (external,
  validation-only). All sources and versions are documented in `data/`.
- **Tools:** RDKit, scikit-learn, XGBoost / LightGBM / CatBoost, PyTorch + HuggingFace Transformers
  (ChemBERTa).
- **AI assistance:** AI coding assistants were used for implementation and refactoring; the scientific
  reasoning, model choices, and analysis are the team's own.

---

*Team FifthParadigm · Discoverathon 2026 · Challenge 2 — Predicting hERG Channel Blockade.*
