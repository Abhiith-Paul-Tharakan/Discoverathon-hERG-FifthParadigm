# hERG Channel Blockade Prediction — Discoverathon 2026, Challenge 2

Predicts, from a single SMILES string, whether a compound blocks the hERG cardiac ion channel
(cardiotoxicity liability) and its potency against hERG (pIC50). Built for Novartis Discoverathon
2026, Challenge 2 (Safety track).

This is a **minimal, final-model-only repo**: it contains exactly what's needed to run and
reproduce the deployed model, its training data, and its validation evidence. The broader
model-selection exploration (GNN, a frozen-embedding ChemBERTa variant, every blend combination
that was evaluated and rejected) informed the choices below but is not included here.

## The final model

> **Final model = ML-Ensemble + fine-tuned "pure" ChemBERTa for classification; ML-Ensemble
> regressor alone for pIC50. No GNN, no frozen-embedding ChemBERTa.**

Wired end-to-end in [`model_code/master_predict.py`](model_code/master_predict.py):

```
P(blocker)      = 0.770 * P(ML_Ensemble) + 0.230 * P(ft_chemberta_pure)
predicted pIC50 = ML_Ensemble regressor, alone
```

- **ML_Ensemble** ([`train_ml_ensemble_classifier.py`](model_code/train_ml_ensemble_classifier.py) /
  [`ml_pic50_regressor.py`](model_code/ml_pic50_regressor.py)) — Morgan/ECFP4 fingerprints
  (2048 bits) + 24 RDKit 2D descriptors, fed to a top-3-by-ROC-AUC soft-voting ensemble over
  {logistic regression, random forest, extra-trees, XGBoost, LightGBM, CatBoost}, calibrated.
  Classifier and regressor are separate, independently-trained models sharing the same scaffold
  split. Self-contained joblib load-and-predict, no retraining needed to run inference.
- **ft_chemberta_pure** ([`finetune_chemberta_pure.py`](model_code/finetune_chemberta_pure.py)) —
  ChemBERTa-2 (`DeepChem/ChemBERTa-77M-MTR`), fully fine-tuned (transformer unfrozen + deeper
  pooled-embedding head), trained with `--pure` — no ECFP4/physchem fusion, signal comes solely
  from the fine-tuned embedding. Classification only; it has no role in the deployed regression.
- The 0.770/0.230 classification blend weights were optimized on out-of-fold ROC-AUC during model
  selection (simplex grid search over every pairwise/3-way combination of the models evaluated)
  and then frozen — this combination was the best/tied-best classifier on internal holdout AND the
  external clinical set. Regression uses the ML-Ensemble regressor alone: every blend that touched
  it made holdout RMSE/R² worse (it has no OOF predictions, so it can't be leakage-safely weighted
  against other regressors without dilution).

## Repo map

```
README.md                     you are here
requirements.txt              pinned deps for the full repo
download_checkpoints.sh       fetches the one >100MB checkpoint from a GitHub Release
SHA256SUMS.txt                checksums for all 4 required checkpoints
submission.csv                FINAL 3-column rubric submission (compound_id, hERG_blocker_probability, predicted hERG pIC50)
submission_extended.csv       same + AD/boundary/confidence-tier trust columns (holdout, n=3,331)

checkpoints/                  REQUIRED final-model weights (see "Model checkpoints" below)
  herg_combined_dataset_holdout.joblib     ML_Ensemble classifier (~318 MB, via Release)
  pic50_regressor.joblib                   ML_Ensemble regressor (~4.2 MB, committed)
  cb_ft_pure_results/
    cb_ft_final.pt                         ft_chemberta_pure weights (~14.4 MB, committed)
    cb_ft_scaler.joblib                    ft_chemberta_pure metadata (88 B, committed)

data/
  herg_combined_dataset/       16,650-compound training pool (ChEMBL + BindingDB)
  holdout_out/                 20% scaffold-split holdout manifest (seed 42)
  clinical_external_validation_set.csv       957-compound external clinical set (DICTrank + CredibleMeds)
  external_dataset_curation_report.md        how the external set was built
  external_dataset_honesty_report.md         what kind of evidence the external set is (and isn't)

model_code/
  master_predict.py                    the deployed pipeline -- SMILES/CSV in, blend out
  train_ml_ensemble_classifier.py      trains the ML_Ensemble classifier
  ml_pic50_regressor.py                trains the ML_Ensemble regressor
  finetune_chemberta_pure.py           trains ft_chemberta_pure (--pure) and its fusion variant
  make_scaffold_split.py               builds the shared scaffold-split holdout (seed 42)
  herg_robust_extraction.py            builds herg_combined_dataset.csv from ChEMBL + BindingDB
  assemble_submission.py               joins classifier+regressor holdout predictions -> submission.csv
  ml_ensemble_holdout_predictions/     ML_Ensemble's own holdout predictions (assemble_submission.py inputs)

external_validation/
  predict_ml_ensemble.py               scores the external set through the deployed ML classifier+regressor
  predict_ft_chemberta.py              scores the external set through the deployed ft_chemberta_pure classifier
  build_result_analysis.py             confusion matrices + AD/boundary trust-filter ablation (holdout + external)
  outputs/submission.csv, submission_extended.csv    deployed blend's predictions on the 957-compound external set
  confusion_matrix_*.csv, trust_filter_ablation.csv  build_result_analysis.py's output

reports/                       placeholder -- technical report added later
presentation/                  placeholder -- deck added later
```

## Data

| Source | What it provides | Role |
|---|---|---|
| ChEMBL + BindingDB | 16,650 compounds with hERG binary blocker labels; a subset with measured pIC50 | **Training** (both classifier and regressor) |
| DICTrank (FDA Drug-Induced Cardiotoxicity Rank, Seal et al. 2024 structure-paired release) | 1,020 drug-level cardiotoxicity concern records | **External validation only** |
| CredibleMeds QTdrugs (rev. 16 Apr 2026) | 323 drug-level TdP risk records, resolved to structures offline | **External validation only** |

- **Held-out test set**: 20% scaffold-split (Murcko scaffolds, `StratifiedGroupKFold`, seed 42),
  carved off before any model selection touches the data — zero scaffold overlap with the 80%
  training pool. See `data/holdout_out/`.
- **External clinical set** (`data/clinical_external_validation_set.csv`, 957 compounds from
  DICTrank + CredibleMeds): used **only** to validate, **never** to train. Honest framing: this is
  an *expert/regulatory-curated proxy* for clinical cardiotoxicity risk, not a raw hERG-patch-clamp
  measurement, and its chemical space barely overlaps the ChEMBL/BindingDB training distribution —
  see "External validation" below and `data/external_dataset_curation_report.md` /
  `data/external_dataset_honesty_report.md` for the full curation methodology and its limits.

## Model checkpoints

`master_predict.py` loads 4 files from `checkpoints/`. Three are small enough to commit directly
to git; the ML-Ensemble classifier (~318 MB) exceeds GitHub's 100 MB push limit and ships as a
GitHub Release asset instead.

| File | Size | Hosting |
|---|---|---|
| `checkpoints/herg_combined_dataset_holdout.joblib` | ~318 MB | GitHub Release (`download_checkpoints.sh`) |
| `checkpoints/pic50_regressor.joblib` | ~4.2 MB | Committed |
| `checkpoints/cb_ft_pure_results/cb_ft_final.pt` | ~14.4 MB | Committed |
| `checkpoints/cb_ft_pure_results/cb_ft_scaler.joblib` | 88 B | Committed |

Fetch + verify:

```bash
bash download_checkpoints.sh          # downloads the one Release-hosted file, verifies all 4
```

Before this works, fill in `REPO=` / `TAG=` at the top of `download_checkpoints.sh` and upload
`checkpoints/herg_combined_dataset_holdout.joblib` as a Release asset under that tag (see the
"Gaps to fix manually" checklist below).

## Setup

```bash
python3.13 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121   # or CPU-only: pip install torch==2.5.1
bash download_checkpoints.sh
```

## Run instructions

### (a) Recreate the scaffold split

```bash
cd model_code
python herg_robust_extraction.py --out_dir ../data/herg_combined_dataset
python make_scaffold_split.py \
    --binary ../data/herg_combined_dataset/herg_binary_ml_dataset.csv \
    --unified ../data/herg_combined_dataset/herg_pic50_unified.csv \
    --out_dir ../data/holdout_out
```

### (b) Fetch / load the two final components

Already covered by `bash download_checkpoints.sh` above — `master_predict.py` loads both
components internally, no separate load step is needed.

### (c) Run `master_predict.py`

Single SMILES (propranolol, a documented hERG blocker):

```bash
cd model_code
python master_predict.py --smiles "CC(C)NCC(O)COc1cccc2ccccc12"
```

Verified output (this exact command, this repo, 2026-08-17):

```
compound_id  hERG_blocker_probability  predicted hERG pIC50  p_ML_Ensemble  p_FT_ChemBERTa_pure
    query_1                    0.3974                 5.281         0.4862               0.0999
```

Batch CSV:

```bash
cd model_code
python master_predict.py --input compounds.csv --smiles_col smiles \
    --id_col compound_id --out predictions.csv
# writes predictions.csv (3-column rubric schema) + predictions_extended.csv (+ per-model contributions)
```

### (d) Reproduce the held-out metrics and external validation

```bash
# Score the external clinical set through each deployed component:
cd external_validation
python predict_ml_ensemble.py --external ../data/clinical_external_validation_set.csv --smiles_col smiles \
    --out ml_ensemble_external_predictions.csv
python predict_ft_chemberta.py --external ../data/clinical_external_validation_set.csv --smiles_col smiles \
    --out ft_chemberta_external_predictions.csv

# Confusion matrices + AD/boundary trust-filter ablation on BOTH holdout and external
# (reads submission_extended.csv at repo root + external_validation/outputs/submission_extended.csv,
# both already-computed deployed-blend predictions -- see "Known limitations"):
python build_result_analysis.py
```

## Held-out results (n = 3,331, scaffold-disjoint from training)

Deployed classification blend (`P = 0.770*ML + 0.230*ft_chemberta_pure`), threshold = 0.5:

| Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | MCC |
|---|---|---|---|---|---|---|
| 0.7556 | 0.7508 | 0.7480 | 0.7494 | 0.8316 | 0.8271 | 0.5110 |

Source: `external_validation/trust_filter_ablation.csv` ("ALL, no filter" row). Restricting to
compounds inside the classifier's kNN-Tanimoto applicability domain (69% of holdout) raises
ROC-AUC to 0.865 / Accuracy to 0.792; restricting further to the HIGH-confidence tier (58% of
holdout) raises it to 0.889 / 0.823 — see the full ablation table for the trust-tier breakdown.

Deployed regression (ML-Ensemble regressor alone), exact-potency subset (n = 2,058):

| MAE | RMSE | R² | Pearson r |
|---|---|---|---|
| 0.4713 | 0.6857 | 0.4089 | 0.6409 |

Source: `model_code/ml_ensemble_holdout_predictions/reg_metrics.json`.

## External validation (957 clinical drugs — validation-only proxy, read this before citing it)

The external set is a genuine domain-shift stress test, not a second held-out test set from the
same distribution — treat the numbers below accordingly:

| | n | ROC-AUC | PR-AUC | Accuracy | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|---|---|
| Deployed blend (ML+FT, 0.77/0.23) | 935 labeled / 957 | 0.6096 | 0.8032 | 0.3979 | 0.8457 | 0.2030 | 0.3274 | 0.1264 |

Source: `external_validation/trust_filter_ablation.csv`. ROC-AUC drops from 0.83 (holdout) to
0.61 (external) — expected, since this set is regulatory/expert-curated clinical risk labels for
approved/withdrawn drugs, not ChEMBL bioassay labels, and only **24 of 935** labeled compounds
fall inside the classifier's training-data applicability domain (the rest is extrapolation by
construction — clinical drugs occupy a very different region of chemical space than the
ChEMBL/BindingDB screening compounds this model trained on). Precision stays high (0.85) but
recall collapses (0.20) at the default threshold — the model under-calls blockers on this set
rather than over-calling them.

Regression cannot be validated externally the normal way — the clinical set has no measured
pIC50, only a binary risk label. As a weak sanity check, predicted pIC50 from the ML-Ensemble
regressor alone correlates with the binary cardiotoxicity label at point-biserial r = 0.073
(p = 0.026, n = 935) — a real but small signal, consistent with the endpoint mismatch (potency
vs. a coarse clinical risk category) rather than a validation of regression accuracy.

## Known limitations

- `master_predict.py` does not compute the applicability-domain flag, boundary-instability flag,
  or confidence tier for ad-hoc single-SMILES/CSV queries — those trust columns are currently only
  produced for the *training holdout* (by `train_ml_ensemble_classifier.py` at train time) and for
  the *external set* (recomputed against the same persisted kNN-Tanimoto cutoff, 0.49408293, in the
  process that produced `external_validation/outputs/submission_extended.csv`). If you need
  per-query trust flags for a new compound, use `train_ml_ensemble_classifier.py`'s AD logic as a
  reference rather than assuming `master_predict.py`'s output CSV carries them — it currently does
  not.
- `external_validation/build_result_analysis.py` reads its predictions from the already-computed
  `submission_extended.csv` files (root + `external_validation/outputs/`) rather than recomputing
  AD/boundary flags itself — `predict_ml_ensemble.py` and `predict_ft_chemberta.py` reproduce the
  two components' raw probabilities, but the AD-flagging + 0.770/0.230 blending step that produced
  the committed `submission_extended.csv` files is not itself a script in this repo (it predates
  this minimal-repo cleanup). The numbers are real or a completed, not fabricated, run; if you need
  to regenerate them from scratch for a *new* compound set, blend `predict_ml_ensemble.py` +
  `predict_ft_chemberta.py`'s output columns with the frozen 0.770/0.230 weights yourself, or use
  `master_predict.py --input` directly, which does the same blend (minus the AD/tier columns, see
  above).
- The external clinical set's applicability-domain coverage is extremely low (24/935 compounds) —
  external metrics above describe near-total extrapolation, not in-distribution generalization.

## Integrity & acknowledgments

- **Data sources**: ChEMBL (bioactivity), BindingDB (bioactivity), DICTrank / Seal et al. 2024
  (FDA drug-induced cardiotoxicity ranks, structure-paired release), CredibleMeds QTdrugs list
  (rev. 16 April 2026). Full curation methodology, versions, and reproducibility caveats in
  `data/external_dataset_curation_report.md` and `data/external_dataset_honesty_report.md`.
- **Tools**: RDKit (featurization/standardization), scikit-learn, XGBoost, LightGBM, CatBoost
  (ML-Ensemble), PyTorch + Hugging Face Transformers (`DeepChem/ChemBERTa-77M-MTR` checkpoint).
- **AI assistance**: AI coding assistants (Claude) were used for implementation and for this
  repository cleanup/reorganization, under the team's own scientific reasoning, methodology
  choices, and result interpretation — the modeling decisions, metrics, and conclusions above are
  the team's.
