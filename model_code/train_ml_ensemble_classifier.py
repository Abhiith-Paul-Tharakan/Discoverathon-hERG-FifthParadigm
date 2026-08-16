#!/usr/bin/env python3
"""
train_ml_ensemble_classifier.py

Scaffold-aware, nested cross-validation pipeline for BINARY hERG-blockade
classification (blocker = 1 / non-blocker = 0), for Discoverathon 2026
Challenge 2 (Predicting hERG Channel Blockade). This is the classifier half of
the deployed ML_Ensemble component -- its output (herg_combined_dataset_holdout.joblib)
is one of the two models master_predict.py loads for the final blend.

TASK & LABELS
-------------
*   Binary label space: LABEL_MAP -> {non-blocker: 0, blocker: 1},
    ALL_CLASSES = (0, 1). normalize_label() accepts the integer 0/1
    `hERG_blocker` column directly (label_col default 'hERG_blocker') as well
    as the textual class names.
*   XGBoost uses objective='binary:logistic' (eval_metric='logloss').
*   Metrics reported match the Challenge 2 rubric: accuracy / balanced
    accuracy, precision, recall (sensitivity), specificity, F1, ROC-AUC and
    PR-AUC (average precision), plus MCC, evaluated on the 20% held-out test
    set. The rubric names all of these and singles out none as "the" primary,
    so for internal model selection this pipeline uses ROC-AUC -- threshold-
    free and comparable across models -- falling back to macro-F1 only when a
    probability-based AUC is undefined for a degenerate fold.
*   Feature set: Morgan/ECFP fingerprints (radius 2, 2048 bits) + 24 RDKit 2D
    descriptors (MolLogP, TPSA, aromatic-ring counts, partial charges, MolMR,
    FractionCSP3, ...); see featurize_smiles / DESCRIPTOR_FUNCS.

Methodology
-----------
*   TRUE held-out test set (20%% by default, ``--holdout_frac`` /
    ``--holdout_ids``): carved off BEFORE anything else touches the data
    (class-stratified + scaffold-grouped via StratifiedGroupKFold, so it's
    representative of the training pool's class balance, not just scaffold-
    disjoint from it). Model selection, calibration, threshold tuning, the
    nested-CV robustness diagnostics, and y-scrambling all run ONLY on the
    remaining 80%% train pool. Once, at the end, ONE final model -- selected
    and fit on 100%% of that 80%% pool -- predicts on the untouched 20%%.
    That is the only number in ``summary_all_targets.csv`` (the
    ``holdout_*`` columns) describing compounds the model never saw in any
    capacity. This is distinct from nested CV's rotating outer-fold test
    partitions (see below), where every compound trains in n-1 folds and
    is held out in exactly one -- useful for robustness, but not a single
    fixed test set. Pass the same ``--holdout_ids`` file (from
    ``make_scaffold_split.py``) to ``ml_pic50_regressor.py`` to score both
    models on the identical shared holdout.

*   Nested CV with scaffold-aware (Murcko) group splitting via
    StratifiedGroupKFold in both outer and inner loops. Run over the TRAIN
    POOL only (post-holdout) as a model-selection / robustness estimate --
    not the headline test number, see above.

*   Model zoo: regularised logistic regression, random forests,
    extra-trees, and (optionally, if installed) XGBoost, LightGBM and
    CatBoost, each with a small hyperparameter grid evaluated in the inner
    loop.

*   Ensemble construction uses a **pre-defined rule** (top-3 models by
    mean inner-fold ROC-AUC, soft-voting with AUC-proportional weights)
    so that both the ensemble and the best-single strategies are scored
    on the **same** held-out inner-fold validation data — eliminating
    selection bias.

*   Model selection across strategies uses **ROC-AUC** (selection_score) as
    the criterion — threshold-free and directly comparable across models —
    with a macro-F1 fallback only for degenerate folds where AUC is undefined.
    The ensemble-composition ranking uses this same metric, so members are the
    top models on exactly the metric everything else is judged on.

*   Class imbalance handled by inverse-frequency sample weighting
    (normalised to mean 1).

*   Optional out-of-fold probability calibration via a stacked
    logistic-regression calibrator.

*   Feature set: Morgan fingerprints (default 2048 bits) + 24 RDKit
    molecular descriptors.  MACCS keys excluded because they are
    largely redundant with Morgan bits and inflate the feature-to-sample
    ratio unnecessarily.  An optional mutual-information feature selector
    is available (``--mi_feature_fraction``, disabled by default) for
    targets with very small sample sizes.

    *Limitation*: ``mutual_info_classif`` does not accept sample weights,
    so MI-based selection does not account for class-rebalancing weights.
    For strongly imbalanced targets, consider using the default (no MI
    selection) and relying on the tree-based models' native feature
    sub-sampling instead.

*   Applicability domain assessed via a k-nearest-neighbour Tanimoto
    similarity approach (Sahigara et al., *J Cheminform* 2012).
    Threshold: mean(kNN_sim) − 1·std(kNN_sim) computed over the
    training set.  At prediction time each query molecule's mean top-k
    similarity to the training set is compared against this threshold,
    ensuring the train-time and test-time statistics are commensurate.

*   Per-class SHAP importance is reported when enabled, preserving the
    information about which features drive each class distinction rather
    than collapsing across classes.

*   Reproducibility achieved through explicit ``numpy.random.SeedSequence``
    / ``Generator`` instances.  No global ``np.random.seed()`` call is
    used.  Each logical unit (outer fold, inner fold, calibration,
    y-scramble repeat) receives an independently spawned Generator so
    that adding or removing candidate models does not cascade seed
    changes through unrelated code paths.

*   Full execution log saved to ``<out_dir>/run.log`` with a timestamped
    archival copy.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import random
import shutil
import traceback
from dataclasses import dataclass, asdict, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Callable

import joblib
import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize

try:
    from scipy.stats import wilcoxon
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from xgboost import XGBClassifier
    from xgboost import DMatrix
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


LOGGER = logging.getLogger("molclf_v7")

# BINARY hERG label space. Accepts several spellings for each class so the
# loader is robust to whatever the dataset uses: the integer 0/1 emitted by
# herg_binary_extraction.py (`hERG_blocker`), plain "blocker"/"non-blocker",
# or the generic active/inactive wording.
LABEL_MAP = {
    "non-blocker": 0, "non blocker": 0, "nonblocker": 0,
    "inactive": 0, "non-block": 0, "0": 0,
    "blocker": 1, "block": 1, "active": 1, "1": 1,
}

# The two real classes. Everything downstream is parameterised by
# `active_classes`, which defaults to this constant.
ALL_CLASSES: Tuple[int, ...] = (0, 1)

# Explicit reverse map (do NOT derive by dict-inversion: LABEL_MAP has several
# aliases per int, so inversion would pick an arbitrary alias).
_INT_TO_LABEL_NAME: Dict[int, str] = {0: "non-blocker", 1: "blocker"}

# Minimum number of compounds required in EACH class. A binary classifier is
# meaningless if either class is (near-)empty, so the pipeline fails closed
# below this floor rather than training on a degenerate label distribution.
MIN_CLASS_SIZE = 50


def resolve_active_classes(
    target_id: str,
    y_full: np.ndarray,
    min_class_size: int = MIN_CLASS_SIZE,
) -> Tuple[List[int], Optional[Dict[str, object]]]:
    """Validate the binary label distribution and return the active classes.

    hERG blockade is a two-class problem (non-blocker = 0, blocker = 1); there
    is no legitimate reason to train on a single class, so if either class has
    fewer than ``min_class_size`` compounds this fails closed with a clear
    error rather than proceeding. On success returns ``([0, 1], None)`` -- the
    second element (drop-info) is kept only so existing call sites remain
    unchanged and is always ``None`` for this binary task.
    """
    y_full = np.asarray(y_full).astype(int)
    counts = {c: int(np.sum(y_full == c)) for c in ALL_CLASSES}
    below_floor = sorted(c for c in ALL_CLASSES if counts[c] < min_class_size)
    if below_floor:
        c = below_floor[0]
        raise ValueError(
            f"{target_id}: class {_INT_TO_LABEL_NAME[c]} (int {c}) has "
            f"{counts[c]} < {min_class_size} compounds. Both hERG classes must "
            f"clear this floor; investigate the extraction/threshold before "
            f"training on a degenerate label distribution."
        )
    return list(ALL_CLASSES), None

DESCRIPTOR_FUNCS = [
    ("MolWt", Descriptors.MolWt),
    ("MolLogP", Descriptors.MolLogP),
    ("TPSA", Descriptors.TPSA),
    ("NumHDonors", Descriptors.NumHDonors),
    ("NumHAcceptors", Descriptors.NumHAcceptors),
    ("HeavyAtomCount", Descriptors.HeavyAtomCount),
    ("NHOHCount", Descriptors.NHOHCount),
    ("NOCount", Descriptors.NOCount),
    ("NumAliphaticRings", Descriptors.NumAliphaticRings),
    ("NumAromaticRings", Descriptors.NumAromaticRings),
    ("NumSaturatedRings", Descriptors.NumSaturatedRings),
    ("RingCount", Descriptors.RingCount),
    ("FractionCSP3", Descriptors.FractionCSP3),
    ("MolMR", Descriptors.MolMR),
    ("NumValenceElectrons", Descriptors.NumValenceElectrons),
    ("MaxPartialCharge", Descriptors.MaxPartialCharge),
    ("MinPartialCharge", Descriptors.MinPartialCharge),
    ("BalabanJ", Descriptors.BalabanJ),
    ("BertzCT", Descriptors.BertzCT),
    ("Chi0n", Descriptors.Chi0n),
    ("Chi1n", Descriptors.Chi1n),
    ("Kappa1", Descriptors.Kappa1),
    ("Kappa2", Descriptors.Kappa2),
    ("Kappa3", Descriptors.Kappa3),
]


# =====================================================================
#  Configuration
# =====================================================================

@dataclass
class RunConfig:
    data_dir: str
    out_dir: str
    pattern: str = "*_final.csv"
    smiles_col: str = "smiles"
    label_col: str = "hERG_blocker"   # column from herg_binary_extraction.py
    target_id_col: str = "target_chembl_id"
    id_col: str = "compound_chembl_id"
    holdout_ids: Optional[str] = None
    holdout_frac: float = 0.20
    seed: int = 42
    outer_splits: int = 5
    inner_splits: int = 4
    min_per_class: int = 30
    fp_bits: int = 2048
    permutation_importance: bool = False
    enable_threshold_tuning: bool = False
    threshold_grid: str = "0.85,1.0,1.15,1.3"
    enable_calibration: bool = False
    strict_mode: bool = False
    y_scramble_repeats: int = 0
    save_shap: bool = False
    # When save_shap is on: run SHAP on EVERY outer CV fold too, not just the
    # final held-out model. Expensive; left off in normal mode (final model
    # only) and turned on in audit mode.
    save_shap_all_folds: bool = False
    log_level: str = "INFO"
    run_mode: str = "normal"
    mi_feature_fraction: float = 1.0   # 1.0 = disabled; < 1.0 enables MI selection
    ad_knn_k: int = 5
    ensemble_top_n: int = 3
    xgb_device: str = "auto"
    xgb_gpu_id: int = 0
    xgb_resolved_device: str = "cpu"
    # Boundary-instability diagnostic. Static per-fold metrics are always
    # computed (free); boundary_bootstrap > 0 additionally runs B train-only
    # bootstrap refits per outer fold to measure a prediction flip rate.
    boundary_bootstrap: int = 0
    boundary_band: str = "0.4,0.6"


MODE_PRESETS: Dict[str, Dict[str, object]] = {
    "fast": {
        "outer_splits": 3,
        "inner_splits": 2,
        "enable_calibration": False,
        "enable_threshold_tuning": False,
        "strict_mode": True,
        "permutation_importance": False,
        "save_shap": False,
        "y_scramble_repeats": 0,
    },
    "normal": {
        "outer_splits": 5,
        "inner_splits": 4,
        "enable_calibration": True,
        "enable_threshold_tuning": False,
        "strict_mode": True,
        "permutation_importance": True,
        "save_shap": True,            # SHAP on the final held-out model
        "save_shap_all_folds": False,  # ...but not on every CV fold (too slow)
        "y_scramble_repeats": 0,
    },
    "audit": {
        "outer_splits": 5,
        "inner_splits": 4,
        "enable_calibration": True,
        "enable_threshold_tuning": False,
        "strict_mode": True,
        "permutation_importance": True,
        "save_shap": True,
        "save_shap_all_folds": True,   # SHAP on every fold as well
        "y_scramble_repeats": 20,
        "boundary_bootstrap": 10,
    },
}


# =====================================================================
#  Reproducibility — explicit RNG via SeedSequence / Generator
# =====================================================================

def make_rng(seed: int) -> np.random.Generator:
    """Create an explicit numpy Generator from an integer seed."""
    return np.random.default_rng(np.random.SeedSequence(seed))


def spawn_rngs(parent: np.random.Generator, n: int) -> List[np.random.Generator]:
    """Spawn *n* independent child Generators from *parent*.

    Uses SeedSequence.spawn so that each child is statistically
    independent and adding/removing children does not affect siblings.
    """
    children = parent.bit_generator.seed_seq.spawn(n)
    return [np.random.Generator(np.random.PCG64(s)) for s in children]


def derive_int_seed(rng: np.random.Generator) -> int:
    """Produce a deterministic int seed for sklearn ``random_state`` params.

    Draws from a spawned child to avoid advancing the parent's state
    in a way that would couple unrelated downstream operations.
    """
    child = spawn_rngs(rng, 1)[0]
    return int(child.integers(0, 2**31 - 1))


def seed_everything(seed: int) -> np.random.Generator:
    """One-time global setup; returns the master Generator."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    if TORCH_AVAILABLE:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    return make_rng(seed)


@lru_cache(maxsize=None)
def probe_xgb_cuda(gpu_id: int) -> Tuple[bool, str]:
    """Return whether XGBoost can execute a tiny fit on the requested CUDA device.

    A one-time, task-independent GPU capability check run once in run(). It
    uses its own fixed synthetic probe data (unrelated to the hERG labels);
    CUDA availability does not depend on the objective or class count, so the
    synthetic probe is sufficient to decide device fallback.
    """
    if not HAS_XGB:
        return False, "xgboost_unavailable"
    try:
        probe = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=1,
            max_depth=1,
            learning_rate=0.3,
            eval_metric="mlogloss",
            tree_method="hist",
            device=f"cuda:{gpu_id}",
            verbosity=0,
        )
        X_probe = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        y_probe = np.array([0, 1, 2, 0, 1, 2], dtype=np.int32)
        probe.fit(X_probe, y_probe)
        return True, f"cuda:{gpu_id}"
    except Exception as exc:
        return False, str(exc)


def resolve_xgb_device(requested_device: str, gpu_id: int) -> str:
    """Resolve the XGBoost execution device from user intent and runtime support."""
    requested = requested_device.lower()
    if requested == "cpu":
        return "cpu"
    if requested not in {"auto", "cuda"}:
        raise ValueError(f"Unsupported xgb_device: {requested_device}")

    ok, detail = probe_xgb_cuda(gpu_id)
    if ok:
        return detail
    if requested == "cuda":
        raise RuntimeError(
            "CUDA was requested for XGBoost, but the GPU probe failed: "
            f"{detail}"
        )
    LOGGER.warning(
        "XGBoost CUDA probe failed; falling back to CPU. Probe detail: %s",
        detail,
    )
    return "cpu"


# =====================================================================
#  Chemistry helpers
# =====================================================================

def normalize_label(x: object) -> Optional[int]:
    """Map a raw label cell to {0, 1} or None.

    Handles the integer/float `hERG_blocker` column (0/1, and 0.0/1.0 when the
    column carries NaNs) as well as textual class names via LABEL_MAP.
    """
    if pd.isna(x):
        return None
    # numeric 0/1 (the herg_binary_extraction.py output) -- treat first so a
    # float like 1.0 does not fall through to the string branch as "1.0".
    if isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool):
        xi = int(x)
        return xi if xi in (0, 1) and float(x) == xi else None
    s = " ".join(str(x).strip().lower().split())
    if s in ("0.0", "1.0"):          # numeric-as-string with trailing .0
        s = s[0]
    return LABEL_MAP.get(s)


def standardize_smiles(smiles: str) -> Optional[Tuple[str, str, str]]:
    if smiles is None or pd.isna(smiles):
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


def featurize_smiles(
    smiles_list: Sequence[str],
    radius: int = 2,
    fp_bits: int = 2048,
) -> Tuple[np.ndarray, List[str]]:
    """Morgan fingerprint + RDKit molecular descriptors.

    MACCS keys are intentionally excluded: they are largely redundant with
    Morgan bits and inflate the feature-to-sample ratio without meaningful
    gain in predictive performance for this task.
    """
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=fp_bits,
    )
    rows: List[np.ndarray] = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid canonical SMILES reached featurizer: {smi}")
        fp = generator.GetFingerprintAsNumPy(mol).astype(np.float32)

        desc_vals = []
        for _, func in DESCRIPTOR_FUNCS:
            try:
                val = func(mol)
                desc_vals.append(
                    float(val) if val is not None and np.isfinite(val) else 0.0
                )
            except (ValueError, RuntimeError, ZeroDivisionError):
                desc_vals.append(0.0)

        row = np.concatenate([fp, np.array(desc_vals, dtype=np.float32)])
        rows.append(row.astype(np.float32))

    feature_names = (
        [f"morgan_{i}" for i in range(fp_bits)]
        + [name for name, _ in DESCRIPTOR_FUNCS]
    )
    return np.vstack(rows), feature_names


# =====================================================================
#  Mutual-information feature selector (opt-in)
# =====================================================================

class MutualInfoSelector(BaseEstimator, TransformerMixin):
    """Select top fraction of features ranked by mutual information.

    Intended as an optional dimensionality-reduction step for targets
    with small sample sizes relative to feature count.

    Limitation
    ----------
    ``mutual_info_classif`` does not accept sample weights, so this
    selector does not account for class-rebalancing weights.  For
    strongly imbalanced targets, consider leaving MI selection disabled
    (``fraction=1.0``) and relying on tree-based models' native feature
    sub-sampling.
    """

    def __init__(self, fraction: float = 1.0, random_state: int = 42):
        self.fraction = fraction
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        n_keep = max(1, int(self.fraction * X.shape[1]))
        if n_keep >= X.shape[1]:
            self.mask_ = np.ones(X.shape[1], dtype=bool)
            return self
        mi = mutual_info_classif(
            X, y,
            discrete_features="auto",
            random_state=self.random_state,
            n_neighbors=5,
        )
        threshold = np.sort(mi)[::-1][n_keep - 1]
        self.mask_ = mi >= threshold
        # Break ties deterministically
        if self.mask_.sum() > n_keep:
            indices = np.where(self.mask_)[0]
            mi_vals = mi[self.mask_]
            order = np.argsort(mi_vals)[::-1]
            keep_idx = indices[order[:n_keep]]
            self.mask_ = np.zeros(X.shape[1], dtype=bool)
            self.mask_[keep_idx] = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.mask_]

    def get_support(self) -> np.ndarray:
        return self.mask_


# =====================================================================
#  Data preparation
# =====================================================================

@dataclass
class PreparedTarget:
    target_id: str
    df: pd.DataFrame
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    feature_names: List[str]
    audit: Dict[str, int]
    # The LABEL_MAP ints being trained on -- always [0, 1] for this binary task.
    active_classes: List[int] = field(default_factory=lambda: list(ALL_CLASSES))


def load_target_csv(
    csv_path: Path, smiles_col: str, label_col: str, target_id_col: Optional[str],
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if smiles_col not in df.columns:
        raise ValueError(f"{csv_path.name}: missing column '{smiles_col}'")
    if label_col not in df.columns:
        raise ValueError(f"{csv_path.name}: missing column '{label_col}'")
    df = df.copy()
    df["__raw_smiles__"] = df[smiles_col].astype(str)
    df["__label__"] = df[label_col].map(normalize_label)
    df["__source_file__"] = csv_path.name
    if target_id_col and target_id_col in df.columns:
        df["__target_id__"] = df[target_id_col].astype(str)
    else:
        df["__target_id__"] = csv_path.stem.replace("_final", "")
    return df


def prepare_target_dataframe(
    df: pd.DataFrame, target_id: str, min_per_class: int, fp_bits: int,
    min_class_size: int = MIN_CLASS_SIZE,
) -> Optional[PreparedTarget]:
    audit: Dict[str, int] = {
        "raw_rows": 0,
        "dropped_invalid_label": 0,
        "dropped_invalid_smiles": 0,
        "dropped_conflicting_labels": 0,
        "dropped_duplicate_inchikey": 0,
        "final_rows": 0,
    }
    work = df[df["__target_id__"] == target_id].copy()
    audit["raw_rows"] = len(work)
    if work.empty:
        return None

    valid = work["__label__"].notna()
    audit["dropped_invalid_label"] = int((~valid).sum())
    work = work[valid].copy()
    if work.empty:
        return None

    std = work["__raw_smiles__"].map(standardize_smiles)
    ok = std.notna()
    audit["dropped_invalid_smiles"] = int((~ok).sum())
    work = work[ok].copy()
    std = std[ok]
    if work.empty:
        return None

    work["canonical_smiles"] = [x[0] for x in std]
    work["inchikey"] = [x[1] for x in std]
    work["scaffold"] = [x[2] for x in std]

    conflicts = work.groupby("inchikey")["__label__"].nunique()
    bad = set(conflicts[conflicts > 1].index)
    if bad:
        audit["dropped_conflicting_labels"] = int(work["inchikey"].isin(bad).sum())
        work = work[~work["inchikey"].isin(bad)].copy()

    audit["dropped_duplicate_inchikey"] = int(work.duplicated(subset=["inchikey"]).sum())
    work = work.drop_duplicates(subset=["inchikey"]).reset_index(drop=True)
    audit["final_rows"] = len(work)

    # Fail closed if either hERG class is below the viability floor.
    active_classes, _ = resolve_active_classes(
        target_id, work["__label__"].to_numpy(), min_class_size,
    )

    cc = work["__label__"].value_counts().sort_index()
    if int(cc.min()) < min_per_class:
        LOGGER.warning(
            "%s: skipped — min class count %d < %d",
            target_id, int(cc.min()), min_per_class,
        )
        return None

    X, feature_names = featurize_smiles(work["canonical_smiles"].tolist(), fp_bits=fp_bits)
    y = work["__label__"].astype(int).to_numpy()
    groups = work["scaffold"].astype(str).to_numpy()
    return PreparedTarget(
        target_id, work, X, y, groups, feature_names, audit,
        active_classes=active_classes,
    )


# =====================================================================
#  True held-out test set (20% by default) -- excluded from ALL model
#  selection / training, never just rotated through as an outer CV fold.
# =====================================================================
# ml_pic50_regressor.py implements the same --holdout_ids / --holdout_frac
# contract; load_holdout_ids() below is deliberately identical to that
# script's loader so a single holdout_inchikeys.csv (from
# make_scaffold_split.py) excludes the SAME compounds from both models --
# the "two entities, one test set" design.

def load_holdout_ids(path: Optional[str]) -> Optional[set]:
    if not path:
        return None
    ids = pd.read_csv(path)
    col = ids.columns[0]
    return set(ids[col].astype(str))


def subset_prepared(prepared: PreparedTarget, idx: np.ndarray) -> PreparedTarget:
    """Slices a PreparedTarget down to `idx` rows, keeping target-level
    metadata (feature names, active classes) intact."""
    return PreparedTarget(
        target_id=prepared.target_id,
        df=prepared.df.iloc[idx].reset_index(drop=True),
        X=prepared.X[idx],
        y=prepared.y[idx],
        groups=prepared.groups[idx],
        feature_names=prepared.feature_names,
        audit=dict(prepared.audit),
        active_classes=prepared.active_classes,
    )


def split_holdout_indices(
    prepared: PreparedTarget, cfg: "RunConfig", rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (train_idx, holdout_idx). The holdout compounds are excluded
    from every subsequent step -- model-selection CV, calibration, final fit
    -- for this target; they are only ever used once, for the final holdout
    prediction.

    Two modes:
      * cfg.holdout_ids given: honour a shared, pre-computed compound list
        (matched on inchikey and/or cfg.id_col) so this run's test set is
        identical to another model's -- e.g. ml_pic50_regressor.py run
        with the same --holdout_ids file.
      * otherwise: self-generate a StratifiedGroupKFold split (class-
        stratified AND scaffold-grouped) and take one fold as the holdout.
        Class-stratification is what makes the held-out set representative
        of the training pool's class balance, not just scaffold-disjoint;
        plain GroupShuffleSplit (used by make_scaffold_split.py, which
        doesn't have a class label to stratify on) only guarantees the
        group-disjointness half of that.
    """
    holdout = load_holdout_ids(cfg.holdout_ids)
    if holdout is not None:
        key_ik = prepared.df["inchikey"].astype(str)
        if cfg.id_col in prepared.df.columns:
            key_id = prepared.df[cfg.id_col].astype(str)
            is_holdout = key_ik.isin(holdout) | key_id.isin(holdout)
        else:
            is_holdout = key_ik.isin(holdout)
        holdout_idx = np.where(is_holdout.to_numpy())[0]
        train_idx = np.where(~is_holdout.to_numpy())[0]
        LOGGER.info(
            "%s | shared holdout from %s: %d test / %d train",
            prepared.target_id, cfg.holdout_ids, len(holdout_idx), len(train_idx),
        )
        return train_idx, holdout_idx

    frac = float(cfg.holdout_frac)
    if frac <= 0:
        return np.arange(len(prepared.y)), np.array([], dtype=int)

    requested_splits = max(2, round(1.0 / frac))
    n_splits = pick_n_splits(prepared.y, prepared.groups, requested_splits, prepared.active_classes)
    rs = derive_int_seed(rng)
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=rs)
    train_idx, holdout_idx = next(skf.split(prepared.X, prepared.y, prepared.groups))

    n_scaf_overlap = len(set(prepared.groups[holdout_idx]) & set(prepared.groups[train_idx]))
    frac_actual = len(holdout_idx) / len(prepared.y)
    LOGGER.info(
        "%s | self-generated holdout (StratifiedGroupKFold, n_splits=%d): "
        "%d test / %d train (%.1f%%), scaffold overlap=%d (must be 0)",
        prepared.target_id, n_splits, len(holdout_idx), len(train_idx),
        100 * frac_actual, n_scaf_overlap,
    )
    return train_idx, holdout_idx


# =====================================================================
#  Feature-name tracking
# =====================================================================

def get_post_pipeline_feature_names(
    pipeline: Pipeline, original_feature_names: List[str],
) -> List[str]:
    """Track feature names through VarianceThreshold and MutualInfoSelector."""
    names = np.array(original_feature_names, dtype=object)
    for step_name in ("var", "mi_select"):
        step = pipeline.named_steps.get(step_name)
        if step is not None and hasattr(step, "get_support"):
            names = names[step.get_support()]
    return names.tolist()


# =====================================================================
#  Model builders
# =====================================================================

def _mi_steps(mi_frac: float, rs: int) -> List[Tuple[str, BaseEstimator]]:
    """Return MI-selection pipeline step if enabled, else empty list."""
    if 0.0 < mi_frac < 1.0:
        return [("mi_select", MutualInfoSelector(fraction=mi_frac, random_state=rs))]
    return []


class DeviceAwareXGBClassifier(XGBClassifier):
    """Use booster/DMatrix prediction for CUDA models to avoid device fallback."""

    def _uses_cuda(self) -> bool:
        device = str(self.get_params(deep=False).get("device", "cpu")).lower()
        return device.startswith("cuda")

    @staticmethod
    def _normalize_iteration_range(iteration_range):
        return (0, 0) if iteration_range is None else iteration_range

    def predict_proba(self, X, validate_features: bool = True, base_margin=None, iteration_range=None):
        if not self._uses_cuda():
            return super().predict_proba(
                X,
                validate_features=validate_features,
                base_margin=base_margin,
                iteration_range=iteration_range,
            )
        booster = self.get_booster()
        data = DMatrix(X)
        proba = booster.predict(
            data,
            validate_features=validate_features,
            iteration_range=self._normalize_iteration_range(iteration_range),
        )
        if proba.ndim == 1:
            proba = np.column_stack([1.0 - proba, proba])
        return np.asarray(proba, dtype=np.float32)

    def predict(self, X, output_margin: bool = False, validate_features: bool = True, base_margin=None, iteration_range=None):
        if output_margin:
            return super().predict(
                X,
                output_margin=output_margin,
                validate_features=validate_features,
                base_margin=base_margin,
                iteration_range=iteration_range,
            )
        proba = self.predict_proba(
            X,
            validate_features=validate_features,
            base_margin=base_margin,
            iteration_range=iteration_range,
        )
        if proba.ndim == 1:
            return (proba > 0.5).astype(int)
        return np.argmax(proba, axis=1)


def build_linear_pipeline(rs: int, C: float, mi_frac: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("var", VarianceThreshold()),
        ]
        + _mi_steps(mi_frac, rs)
        + [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=4000,
                random_state=rs,
                solver="lbfgs",
                C=C,
            )),
        ]
    )


def build_rf_pipeline(
    rs: int, n_estimators: int, min_samples_leaf: int,
    max_features: str, mi_frac: float,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("var", VarianceThreshold()),
        ]
        + _mi_steps(mi_frac, rs)
        + [
            ("clf", RandomForestClassifier(
                n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                max_features=max_features, n_jobs=-1, random_state=rs,
            )),
        ]
    )


def build_extratrees_pipeline(
    rs: int, n_estimators: int, min_samples_leaf: int,
    max_features: str, mi_frac: float,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("var", VarianceThreshold()),
        ]
        + _mi_steps(mi_frac, rs)
        + [
            ("clf", ExtraTreesClassifier(
                n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                max_features=max_features, n_jobs=-1, random_state=rs,
            )),
        ]
    )


def build_xgb_pipeline(
    rs: int, n_estimators: int, max_depth: int,
    learning_rate: float, mi_frac: float, xgb_device: str,
    n_classes: int = 3,
) -> Optional[Pipeline]:
    if not HAS_XGB:
        return None
    # Binary vs multiclass objective. XGBoost rejects num_class for
    # binary:logistic, so it is only passed on the multiclass path.
    if n_classes <= 2:
        clf_kwargs = dict(objective="binary:logistic", eval_metric="logloss")
    else:
        clf_kwargs = dict(objective="multi:softprob", num_class=n_classes,
                          eval_metric="mlogloss")
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("var", VarianceThreshold()),
        ]
        + _mi_steps(mi_frac, rs)
        + [
            ("clf", DeviceAwareXGBClassifier(
                n_estimators=n_estimators, max_depth=max_depth,
                learning_rate=learning_rate, subsample=0.9,
                colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=1.0,
                random_state=rs, n_jobs=-1,
                tree_method="hist", device=xgb_device,
                **clf_kwargs,
            )),
        ]
    )


def build_lgbm_pipeline(
    rs: int, n_estimators: int, num_leaves: int, learning_rate: float,
    mi_frac: float, n_classes: int = 2,
) -> Optional[Pipeline]:
    """LightGBM gradient-boosting candidate. LightGBM infers the objective
    (binary vs multiclass) from y, so only tree/regularisation params are set.
    Returns None if lightgbm is not installed, so the zoo degrades gracefully.
    """
    if not HAS_LGBM:
        return None
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("var", VarianceThreshold()),
        ]
        + _mi_steps(mi_frac, rs)
        + [
            ("clf", LGBMClassifier(
                n_estimators=n_estimators, num_leaves=num_leaves,
                learning_rate=learning_rate, subsample=0.9, subsample_freq=1,
                colsample_bytree=0.8, reg_lambda=1.0,
                random_state=rs, n_jobs=-1, verbosity=-1,
            )),
        ]
    )


def build_catboost_pipeline(
    rs: int, iterations: int, depth: int, learning_rate: float,
    mi_frac: float, n_classes: int = 2,
) -> Optional[Pipeline]:
    """CatBoost gradient-boosting candidate. loss_function is chosen by class
    count; file writing is disabled for a clean, reproducible run. Returns
    None if catboost is not installed.
    """
    if not HAS_CATBOOST:
        return None
    loss = "Logloss" if n_classes <= 2 else "MultiClass"
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("var", VarianceThreshold()),
        ]
        + _mi_steps(mi_frac, rs)
        + [
            ("clf", CatBoostClassifier(
                iterations=iterations, depth=depth, learning_rate=learning_rate,
                l2_leaf_reg=3.0, loss_function=loss, random_seed=rs,
                thread_count=-1, verbose=False, allow_writing_files=False,
            )),
        ]
    )


def get_candidate_models(
    rng: np.random.Generator, mi_frac: float, xgb_device: str = "cpu",
    n_classes: int = 3,
) -> Dict[str, BaseEstimator]:
    """Build the candidate model zoo.

    Each model receives a deterministic seed derived from a spawned
    child generator, ensuring that adding/removing a candidate does
    not change seeds for its siblings.

    n_classes defaults to 3 (unchanged). Only XGBoost needs it explicitly
    (num_class); every other model in the zoo infers class count from y at
    fit() time, same as always.
    """
    # Spawn enough children for all possible models. LightGBM (2) and CatBoost
    # (2) are appended AFTER the original slots so every pre-existing model
    # keeps its exact seed (SeedSequence.spawn is incremental).
    n_children = 3 + 8 + 2 + 2 + 2 + 2  # logreg + rf/et + xgb + baselines + lgbm + catboost
    children = spawn_rngs(rng, n_children)
    idx = 0
    models: Dict[str, BaseEstimator] = {}

    for C in [0.3, 1.0, 3.0]:
        rs = int(children[idx].integers(0, 2**31 - 1))
        idx += 1
        models[f"logreg_C{C}"] = build_linear_pipeline(rs, C, mi_frac)

    for n_est in [400, 700]:
        for leaf in [1, 2]:
            rs_rf = int(children[idx].integers(0, 2**31 - 1))
            idx += 1
            models[f"rf_n{n_est}_leaf{leaf}_mfsqrt"] = build_rf_pipeline(
                rs_rf, n_est, leaf, "sqrt", mi_frac,
            )
            rs_et = int(children[idx].integers(0, 2**31 - 1))
            idx += 1
            models[f"et_n{n_est}_leaf{leaf}_mfsqrt"] = build_extratrees_pipeline(
                rs_et, n_est, leaf, "sqrt", mi_frac,
            )

    if HAS_XGB:
        for n_est, depth, lr in [(350, 4, 0.05), (550, 6, 0.03)]:
            rs_xgb = int(children[idx].integers(0, 2**31 - 1))
            idx += 1
            xgb = build_xgb_pipeline(
                rs_xgb, n_est, depth, lr, mi_frac, xgb_device, n_classes,
            )
            if xgb is not None:
                models[f"xgb_n{n_est}_d{depth}_lr{lr}"] = xgb
    else:
        idx += 2  # consume slots to keep indexing stable

    rs_dummy = int(children[idx].integers(0, 2**31 - 1))
    idx += 1
    models["baseline_prior"] = DummyClassifier(strategy="prior")
    models["baseline_stratified"] = DummyClassifier(
        strategy="stratified", random_state=rs_dummy,
    )

    # --- LightGBM candidates (appended; graceful no-op if not installed) ---
    if HAS_LGBM:
        for n_est, leaves, lr in [(400, 31, 0.05), (700, 63, 0.03)]:
            rs_l = int(children[idx].integers(0, 2**31 - 1)); idx += 1
            lgbm = build_lgbm_pipeline(rs_l, n_est, leaves, lr, mi_frac, n_classes)
            if lgbm is not None:
                models[f"lgbm_n{n_est}_lv{leaves}_lr{lr}"] = lgbm
    else:
        idx += 2

    # --- CatBoost candidates (appended; graceful no-op if not installed) ---
    if HAS_CATBOOST:
        for n_it, depth, lr in [(400, 6, 0.05), (700, 8, 0.03)]:
            rs_c = int(children[idx].integers(0, 2**31 - 1)); idx += 1
            cat = build_catboost_pipeline(rs_c, n_it, depth, lr, mi_frac, n_classes)
            if cat is not None:
                models[f"catboost_it{n_it}_d{depth}_lr{lr}"] = cat
    else:
        idx += 2

    return models


# =====================================================================
#  Class weighting — inverse frequency only
# =====================================================================

def compute_class_weights(
    y: np.ndarray, active_classes: Sequence[int] = ALL_CLASSES,
) -> Dict[int, float]:
    """Inverse-frequency class weights, normalised to mean 1, over
    active_classes only. Default active_classes=(0,1,2) is byte-identical
    to the original hardcoded-3 arithmetic: same counts, same division
    order, same normalisation.
    """
    y = np.asarray(y).astype(int)
    active_classes = list(active_classes)
    counts = {c: int(np.sum(y == c)) for c in active_classes}
    zero = [c for c in active_classes if counts[c] == 0]
    if zero:
        raise ValueError(f"All {len(active_classes)} active classes must be present.")
    n = float(len(y))
    k = float(len(active_classes))
    raw = {c: n / (k * counts[c]) for c in active_classes}
    mean_w = float(np.mean(list(raw.values())))
    return {c: float(raw[c] / mean_w) for c in raw}


def make_sample_weight(y: np.ndarray, cw: Dict[int, float]) -> np.ndarray:
    return np.array([cw[int(v)] for v in y], dtype=float)


# =====================================================================
#  Split helpers
# =====================================================================

def class_group_spread(
    y: np.ndarray, groups: np.ndarray, active_classes: Sequence[int] = ALL_CLASSES,
) -> Dict[int, int]:
    return {c: int(np.unique(groups[y == c]).shape[0]) for c in active_classes}


def pick_n_splits(
    y: np.ndarray, groups: np.ndarray, requested: int,
    active_classes: Sequence[int] = ALL_CLASSES,
) -> int:
    cc = {c: int(np.sum(y == c)) for c in active_classes}
    spread_min = min(class_group_spread(y, groups, active_classes).values())
    n_groups = len(np.unique(groups))
    feasible = int(min(min(cc.values()), spread_min, n_groups))
    return max(2, min(requested, feasible))


def fold_diagnostics(
    y: np.ndarray, groups: np.ndarray, active_classes: Sequence[int] = ALL_CLASSES,
) -> Dict[str, int]:
    cc = {c: int(np.sum(y == c)) for c in active_classes}
    sp = class_group_spread(y, groups, active_classes)
    # Key order matches the original hardcoded-3 dict exactly (class_*_n
    # for all classes, then n_groups_total, then class_*_group_spread) so
    # the three-class case produces byte-identical DataFrame column order.
    out: Dict[str, int] = {}
    for c in active_classes:
        out[f"class_{c}_n"] = cc[c]
    out["n_groups_total"] = int(len(np.unique(groups)))
    for c in active_classes:
        out[f"class_{c}_group_spread"] = sp[c]
    return out


# =====================================================================
#  Fitting wrappers
# =====================================================================

def fit_estimator(
    model: BaseEstimator, X: np.ndarray, y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> BaseEstimator:
    fitted = clone(model)
    if sample_weight is None or isinstance(fitted, DummyClassifier):
        fitted.fit(X, y)
        return fitted
    if isinstance(fitted, Pipeline):
        last = list(fitted.named_steps.keys())[-1]
        fitted.fit(X, y, **{f"{last}__sample_weight": sample_weight})
    else:
        fitted.fit(X, y, sample_weight=sample_weight)
    return fitted


# =====================================================================
#  Ensemble / wrapper classes
# =====================================================================

class WeightedSoftVotingEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(
        self, estimators: List[Tuple[str, BaseEstimator]], weights: List[float],
    ):
        self.estimators = estimators
        self.weights = weights

    def fit(self, X, y, sample_weight=None):
        self.fitted_estimators_ = [
            (n, fit_estimator(e, X, y, sample_weight)) for n, e in self.estimators
        ]
        # Derived from the actual labels fit on, not hardcoded -- for the
        # unchanged three-class path y always contains {0,1,2}, so this is
        # byte-identical to np.array([0, 1, 2]). Generalises correctly for
        # any active_classes subset (all sub-estimators are fit on this
        # same y, so their predict_proba column order agrees).
        self.classes_ = np.array(sorted(np.unique(np.asarray(y).astype(int))))
        return self

    def predict_proba(self, X):
        probas = [est.predict_proba(X) for _, est in self.fitted_estimators_]
        w = np.array(self.weights, dtype=float)
        w /= w.sum()
        return np.tensordot(w, np.stack(probas, axis=0), axes=(0, 0))

    def predict(self, X):
        idx = np.argmax(self.predict_proba(X), axis=1)
        return self.classes_[idx]


class MulticlassProbabilityCalibrator:
    def __init__(self, random_state: int = 42, eps: float = 1e-8):
        self.random_state = random_state
        self.eps = eps
        self.model = LogisticRegression(
            max_iter=3000, 
            solver="lbfgs", random_state=random_state,
        )

    def _transform(self, proba):
        p = np.clip(proba, self.eps, 1.0)
        p /= p.sum(axis=1, keepdims=True)
        return np.log(p)

    def fit(self, proba, y, sample_weight=None):
        self.model.fit(self._transform(proba), y, sample_weight=sample_weight)
        return self

    def predict_proba(self, proba):
        return self.model.predict_proba(self._transform(proba))


class CalibratedThresholdedWrapper(BaseEstimator, ClassifierMixin):
    def __init__(
        self, base_estimator, calibrator=None, thresholds=None,
    ):
        self.base_estimator = base_estimator
        self.calibrator = calibrator
        # Resolved to per-class 1.0s at fit() time (self.n_classes_ many),
        # not hardcoded to 3 -- callers that already pass an explicit
        # thresholds list (the three-class path always does) are unaffected.
        self.thresholds = list(thresholds) if thresholds is not None else None

    def fit(self, X, y, sample_weight=None):
        self.fitted_base_ = fit_estimator(self.base_estimator, X, y, sample_weight)
        # Derived from the actual labels fit on -- see WeightedSoftVotingEnsemble.
        self.classes_ = np.array(sorted(np.unique(np.asarray(y).astype(int))))
        # Fitted attribute (not a mutated constructor param, so clone()
        # after fit() still reflects the original thresholds=None intent).
        self.thresholds_ = (
            list(self.thresholds) if self.thresholds is not None
            else [1.0] * len(self.classes_)
        )
        return self

    def predict_proba_uncalibrated(self, X):
        return self.fitted_base_.predict_proba(X)

    def predict_proba(self, X):
        p = self.predict_proba_uncalibrated(X)
        if self.calibrator is not None:
            p = self.calibrator.predict_proba(p)
        return p

    def predict(self, X):
        p = self.predict_proba(X)
        t = np.array(self.thresholds_, dtype=float)
        idx = np.argmax(p / t.reshape(1, -1), axis=1)
        return self.classes_[idx]


# =====================================================================
#  Metrics
# =====================================================================

def compute_metrics(
    y_true, y_pred, y_proba=None, active_classes: Sequence[int] = ALL_CLASSES,
) -> Dict[str, float]:
    active_classes = list(active_classes)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    pr, rc, f1v, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=active_classes, zero_division=0,
    )
    for i, c in enumerate(active_classes):
        out[f"class_{c}_precision"] = float(pr[i])
        out[f"class_{c}_recall"] = float(rc[i])
        out[f"class_{c}_f1"] = float(f1v[i])
        out[f"class_{c}_support"] = int(sup[i])
    # Binary sensitivity / specificity from the confusion matrix (positive
    # class = blocker = the larger label id). Rubric asks for these explicitly.
    if len(active_classes) == 2:
        pos = max(active_classes); neg = min(active_classes)
        tp = int(np.sum((y_true == pos) & (y_pred == pos)))
        fn = int(np.sum((y_true == pos) & (y_pred == neg)))
        tn = int(np.sum((y_true == neg) & (y_pred == neg)))
        fp = int(np.sum((y_true == neg) & (y_pred == pos)))
        out["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) else np.nan
        out["specificity"] = float(tn / (tn + fp)) if (tn + fp) else np.nan
    if y_proba is not None:
        if len(active_classes) == 2:
            # sklearn's roc_auc_score does not accept multi_class='ovr' for
            # binary problems -- it expects a 1D positive-class score.
            pos_idx = active_classes.index(max(active_classes))
            try:
                out["roc_auc_ovr_macro"] = float(
                    roc_auc_score(y_true, y_proba[:, pos_idx]),
                )
            except ValueError:
                out["roc_auc_ovr_macro"] = np.nan
            # PR-AUC (average precision) for the positive (blocker) class.
            try:
                y_pos = (np.asarray(y_true) == max(active_classes)).astype(int)
                out["pr_auc"] = float(
                    average_precision_score(y_pos, y_proba[:, pos_idx]),
                )
            except ValueError:
                out["pr_auc"] = np.nan
        else:
            y_bin = label_binarize(y_true, classes=active_classes)
            try:
                out["roc_auc_ovr_macro"] = float(
                    roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro"),
                )
            except ValueError:
                out["roc_auc_ovr_macro"] = np.nan
            try:
                out["pr_auc"] = float(
                    average_precision_score(y_bin, y_proba, average="macro"),
                )
            except ValueError:
                out["pr_auc"] = np.nan
    else:
        out["roc_auc_ovr_macro"] = np.nan
        out["pr_auc"] = np.nan
    return out


def selection_score(metrics: Dict[str, float]) -> float:
    """Model-selection criterion.

    Prefer ROC-AUC (threshold-free, the Discoverathon rubric's primary metric
    for binary hERG); fall back to macro-F1 when a probability-based AUC is
    unavailable (e.g. a degenerate validation fold). Both are on [0, 1], so the
    fallback never inflates the score relative to AUC-selected folds.
    """
    auc = metrics.get("roc_auc_ovr_macro", np.nan)
    if auc is not None and np.isfinite(auc):
        return float(auc)
    return float(metrics["macro_f1"])


# =====================================================================
#  Boundary-instability diagnostics
# =====================================================================
# Two complementary, leakage-free views of decision-boundary stability:
#   (1) STATIC, per fold, free:
#         boundary_uncertain_fraction  - fraction of test compounds in an
#             ambiguity band (binary: positive proba in [lo, hi];
#             multiclass: top1-top2 margin < thresh)
#         boundary_mean_margin         - mean confidence margin
#         boundary_calib_flip_fraction - fraction of test predictions that
#             change between the uncalibrated and calibrated/thresholded model
#   (2) DYNAMIC, optional (cfg.boundary_bootstrap > 0):
#         boundary_bootstrap_flip_fraction - refit the chosen model on B
#             bootstrap resamples of the TRAIN fold only and measure how often
#             each test compound's predicted label changes. Train-side only,
#             so it never leaks test information.

def boundary_stability_metrics(
    y_proba, y_pred_a, y_pred_b, active_classes,
    band=(0.4, 0.6), margin_thresh=0.10,
) -> Dict[str, float]:
    ac = list(active_classes)
    P = np.asarray(y_proba, dtype=float)
    out: Dict[str, float] = {}
    if len(ac) == 2:
        pos = P[:, ac.index(max(ac))]
        out["boundary_uncertain_fraction"] = (
            float(np.mean((pos >= band[0]) & (pos <= band[1]))) if len(pos) else np.nan
        )
        out["boundary_mean_margin"] = (
            float(np.mean(np.abs(2.0 * pos - 1.0))) if len(pos) else np.nan
        )
    else:
        srt = np.sort(P, axis=1)
        margin = srt[:, -1] - srt[:, -2]
        out["boundary_uncertain_fraction"] = (
            float(np.mean(margin < margin_thresh)) if len(margin) else np.nan
        )
        out["boundary_mean_margin"] = float(np.mean(margin)) if len(margin) else np.nan
    a, b = np.asarray(y_pred_a), np.asarray(y_pred_b)
    out["boundary_calib_flip_fraction"] = float(np.mean(a != b)) if len(a) else np.nan
    return out


def bootstrap_boundary_flip(
    builder, X_tr, y_tr, sample_weight, X_te, active_classes,
    n_boot: int, rng: np.random.Generator,
) -> Tuple[float, int]:
    """Per-compound prediction flip rate under TRAIN-set bootstrap resampling.

    Returns (mean_flip_fraction, n_boot_effective). flip = 1 - modal-vote
    fraction, averaged over test compounds. 0.0 = a perfectly stable boundary.
    Degenerate (single-class) resamples are skipped rather than crashing.
    """
    n = len(y_tr)
    if n_boot <= 0 or len(X_te) == 0 or n == 0:
        return float("nan"), 0
    ac = np.array(sorted(active_classes))
    preds, used = [], 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if np.unique(y_tr[idx]).size < 2:
            continue
        sw = sample_weight[idx] if sample_weight is not None else None
        try:
            est = fit_estimator(builder(), X_tr[idx], y_tr[idx], sw)
            proba = est.predict_proba(X_te)
            preds.append(ac[np.argmax(proba, axis=1)])
            used += 1
        except Exception:
            continue
    if used < 2:
        return float("nan"), used
    M = np.vstack(preds)
    flips = [1.0 - np.unique(M[:, j], return_counts=True)[1].max() / used
             for j in range(M.shape[1])]
    return float(np.mean(flips)), used


# =====================================================================
#  Applicability domain — kNN Tanimoto similarity
# =====================================================================

def smiles_to_rdkit_fp(smiles_list, radius=2, fp_bits=2048):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_bits)
    return [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles_list]


def _knn_mean_similarity(fp, all_fps, k: int, exclude_self_idx: int = -1) -> float:
    """Mean Tanimoto similarity to the k nearest neighbours."""
    sims = DataStructs.BulkTanimotoSimilarity(fp, all_fps)
    if exclude_self_idx >= 0:
        sims = [s for j, s in enumerate(sims) if j != exclude_self_idx]
    if len(sims) < k:
        return float(np.mean(sims)) if sims else 0.0
    return float(np.mean(sorted(sims, reverse=True)[:k]))


def ad_threshold_from_train(train_fps, k: int = 5) -> float:
    """kNN-distance AD threshold: mean(kNN_sim) − 1·std(kNN_sim).

    For each training molecule, the mean Tanimoto similarity to its *k*
    nearest neighbours is computed.  The threshold is set at one standard
    deviation below the mean of these per-molecule scores.

    Reference: Sahigara et al., *J Cheminform* 4:25, 2012.
    """
    scores = []
    for i, fp in enumerate(train_fps):
        s = _knn_mean_similarity(fp, train_fps, k, exclude_self_idx=i)
        scores.append(s)
    if not scores:
        return 0.0
    arr = np.array(scores)
    return float(arr.mean() - arr.std())


def ad_query_scores(query_fps, train_fps, k: int = 5) -> np.ndarray:
    """Per-query mean top-k Tanimoto similarity to the training set.

    This matches the statistic used in ``ad_threshold_from_train`` so
    that train-time and test-time metrics are commensurate.
    """
    return np.array(
        [_knn_mean_similarity(fp, train_fps, k) for fp in query_fps],
        dtype=float,
    )


# =====================================================================
#  SHAP — per-class importance
# =====================================================================

def _post_pipeline_support_indices(pipeline: Pipeline, n_features: int) -> np.ndarray:
    """Original-feature indices that survive the pipeline's VarianceThreshold
    and (optional) MI-selection steps -- the inverse mapping needed to scatter
    a member's SHAP values back into the original feature space."""
    idx = np.arange(n_features)
    for step_name in ("var", "mi_select"):
        step = pipeline.named_steps.get(step_name)
        if step is not None and hasattr(step, "get_support"):
            idx = idx[step.get_support()]
    return idx


def _pipeline_abs_shap(
    pipe: Pipeline, X_train: np.ndarray, X_test: np.ndarray,
    n_features: int, pos_class_col: Optional[int], n_test: int,
) -> Optional[Dict[str, np.ndarray]]:
    """Mean |SHAP| for a single fitted Pipeline, scattered back into the
    ORIGINAL feature space (length ``n_features``). Returns
    ``{'overall': ..., 'blocker': ...}`` or None when the classifier type is
    unsupported or SHAP fails. Working in original-feature coordinates lets
    heterogeneous ensemble members (each with its own VarianceThreshold) be
    combined on a common axis.
    """
    if not isinstance(pipe, Pipeline):
        return None
    steps = pipe.named_steps
    Xt_train, Xt_test = X_train.copy(), X_test.copy()
    for name, step in steps.items():
        if name == "clf":
            break
        Xt_train = step.transform(Xt_train)
        Xt_test = step.transform(Xt_test)

    clf = steps["clf"]
    lname = clf.__class__.__name__.lower()
    if hasattr(clf, "estimators_") or lname.startswith(
        ("randomforest", "extratrees", "xgb", "lgbm", "catboost", "gradientboost"),
    ):
        explainer = shap.Explainer(clf, Xt_train)
    elif lname.startswith("logisticregression"):
        explainer = shap.LinearExplainer(clf, Xt_train)
    else:
        return None

    sv = explainer(Xt_test[:n_test])
    vals = np.asarray(sv.values)
    if vals.ndim == 3:            # (rows, features, classes)
        overall = np.abs(vals).mean(axis=(0, 2))
        col = pos_class_col if (pos_class_col is not None and pos_class_col < vals.shape[2]) else -1
        blocker = np.abs(vals[:, :, col]).mean(axis=0)
    elif vals.ndim == 2:          # (rows, features)  -- binary log-odds
        overall = np.abs(vals).mean(axis=0)
        blocker = overall
    else:
        return None

    support = _post_pipeline_support_indices(pipe, n_features)
    if len(support) != overall.shape[0]:
        return None  # feature-count mismatch; skip rather than misalign
    full_overall = np.zeros(n_features, dtype=float)
    full_blocker = np.zeros(n_features, dtype=float)
    full_overall[support] = overall
    full_blocker[support] = blocker
    return {"overall": full_overall, "blocker": full_blocker}


def maybe_run_shap(
    final_wrapper: CalibratedThresholdedWrapper,
    X_train: np.ndarray, X_test: np.ndarray,
    original_feature_names: List[str],
    out_path: Path,
    active_classes: Sequence[int] = ALL_CLASSES,
) -> None:
    """Export SHAP feature importance for the FINAL model.

    Handles both a single-pipeline final model and a WeightedSoftVotingEnsemble:
    for an ensemble, each member's mean |SHAP| is computed, mapped to the
    original feature space, and combined with the ensemble's own soft-voting
    weights (renormalised over the members SHAP can explain). Emits a single
    ranked ``mean_abs_shap_overall`` column plus, for the binary task, a
    ``mean_abs_shap_blocker`` column (importance toward the blocker class).
    """
    if not HAS_SHAP:
        LOGGER.info("SHAP not installed; skipping.")
        return
    try:
        base = getattr(final_wrapper, "fitted_base_", None)
        if base is None:
            return
        n_features = len(original_feature_names)
        n_test = min(200, len(X_test))
        ac = list(active_classes)
        pos_class_col = ac.index(max(ac)) if len(ac) == 2 else None

        # Collect (pipeline, weight) members of the final model.
        members: List[Tuple[BaseEstimator, float]] = []
        if isinstance(base, WeightedSoftVotingEnsemble):
            w = np.array(base.weights, dtype=float)
            w = w / w.sum() if w.sum() > 0 else np.ones(len(w)) / max(1, len(w))
            for (_, est), wt in zip(base.fitted_estimators_, w):
                members.append((est, float(wt)))
        elif isinstance(base, Pipeline):
            members.append((base, 1.0))
        else:
            LOGGER.info("SHAP: unsupported final-model type %s; skipping.",
                        base.__class__.__name__)
            return

        acc_overall = np.zeros(n_features, dtype=float)
        acc_blocker = np.zeros(n_features, dtype=float)
        used_weight = 0.0
        for pipe, wt in members:
            res = _pipeline_abs_shap(
                pipe, X_train, X_test, n_features, pos_class_col, n_test,
            )
            if res is None:
                continue
            acc_overall += wt * res["overall"]
            acc_blocker += wt * res["blocker"]
            used_weight += wt

        if used_weight <= 0:
            LOGGER.info("SHAP: no explainable members in the final model; skipping.")
            return
        acc_overall /= used_weight
        acc_blocker /= used_weight

        result: Dict[str, object] = {
            "feature": original_feature_names,
            "mean_abs_shap_overall": acc_overall,
        }
        if pos_class_col is not None:
            result["mean_abs_shap_blocker"] = acc_blocker
        pd.DataFrame(result).sort_values(
            "mean_abs_shap_overall", ascending=False,
        ).to_csv(out_path, index=False)
        LOGGER.info("SHAP summary written: %s", out_path.name)
    except Exception as exc:
        LOGGER.warning("SHAP export failed: %s", exc)


# =====================================================================
#  Calibration / thresholding
# =====================================================================

def tune_thresholds_on_validation(
    y_true, y_proba, grid, active_classes: Sequence[int] = ALL_CLASSES,
) -> List[float]:
    active_classes = list(active_classes)
    ac = np.array(active_classes)
    best_thr, best_score = [1.0] * len(active_classes), -np.inf
    # itertools.product(grid, repeat=k) enumerates in the exact same order
    # as k nested for-loops (leftmost slowest) -- for k=3 this is byte-for-
    # byte the same sequence the original triple loop produced.
    for thr_tuple in itertools.product(grid, repeat=len(active_classes)):
        thr = np.array(thr_tuple)
        idx = np.argmax(y_proba / thr.reshape(1, -1), axis=1)
        pred = ac[idx]
        s = selection_score(compute_metrics(y_true, pred, y_proba, active_classes))
        if s > best_score:
            best_score = s
            best_thr = [float(x) for x in thr_tuple]
    return best_thr


def build_oof_calibrator(
    model_builder: Callable[[], BaseEstimator],
    X, y, groups, sample_weight, n_splits: int,
    rng: np.random.Generator,
    active_classes: Sequence[int] = ALL_CLASSES,
) -> Tuple[MulticlassProbabilityCalibrator, np.ndarray]:
    rs = derive_int_seed(rng)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=rs)
    oof = np.zeros((len(y), len(active_classes)), dtype=float)
    for tr_idx, va_idx in cv.split(X, y, groups):
        est = model_builder()
        fitted = fit_estimator(est, X[tr_idx], y[tr_idx], sample_weight[tr_idx])
        oof[va_idx] = fitted.predict_proba(X[va_idx])
    calibrator = MulticlassProbabilityCalibrator(random_state=rs)
    calibrator.fit(oof, y, sample_weight=sample_weight)
    return calibrator, oof


# =====================================================================
#  Inner-loop strategy selection
# =====================================================================

def evaluate_candidate(model, X_tr, y_tr, X_va, y_va, sample_weight,
                       calibrate, rng, groups_tr=None,
                       active_classes: Sequence[int] = ALL_CLASSES) -> Dict[str, float]:
    active_classes = list(active_classes)
    ac = np.array(active_classes)
    fitted = fit_estimator(model, X_tr, y_tr, sample_weight)
    y_proba = fitted.predict_proba(X_va)

    if calibrate and sample_weight is not None and groups_tr is not None:
        try:
            ns = pick_n_splits(
                y_tr, groups_tr, min(3, max(2, len(np.unique(groups_tr)))),
                active_classes,
            )
            cal_rng = spawn_rngs(rng, 1)[0]
            calibrator, _ = build_oof_calibrator(
                lambda m=clone(model): m, X_tr, y_tr, groups_tr,
                sample_weight, ns, cal_rng, active_classes,
            )
            y_proba = calibrator.predict_proba(y_proba)
        except Exception as exc:
            LOGGER.warning("Inner calibration fallback: %s", exc)

    pred = ac[np.argmax(y_proba, axis=1)]
    return compute_metrics(y_va, pred, y_proba, active_classes)


def make_predefined_ensemble(
    models: Dict[str, BaseEstimator],
    score_table: Dict[str, float],
    top_n: int = 3,
) -> Tuple[str, BaseEstimator, List[Tuple[str, float]]]:
    """Build an ensemble from the pre-defined top-N models ranked by ROC-AUC.

    Ensemble composition is determined entirely by mean inner-fold ROC-AUC
    (selection_score) rankings -- the same threshold-free criterion used to
    choose between strategies.  Because the composition rule is fixed (always
    top-N), the ensemble and the best-single strategy can be evaluated on the
    same held-out validation data without selection bias. `score_table` maps
    each model name to its mean ROC-AUC; weights are proportional to that score.
    """
    ranked = sorted(
        [(k, v) for k, v in score_table.items() if not k.startswith("baseline_")],
        key=lambda kv: kv[1],
        reverse=True,
    )
    top = ranked[:min(top_n, len(ranked))]
    if len(top) == 1:
        name = top[0][0]
        return name, clone(models[name]), [(name, 1.0)]

    names, raw_w = [], []
    for name, score in top:
        names.append(name)
        raw_w.append(max(1e-6, score))
    w = np.array(raw_w)
    w /= w.sum()
    return (
        "ensemble_top" + str(len(names)) + "_" + "_".join(names),
        WeightedSoftVotingEnsemble(
            [(n, clone(models[n])) for n in names], w.tolist(),
        ),
        list(zip(names, w.tolist())),
    )


def choose_best_strategy(
    models: Dict[str, BaseEstimator],
    X_train, y_train, groups_train,
    rng: np.random.Generator,
    requested_inner_splits: int,
    enable_calibration: bool,
    ensemble_top_n: int = 3,
    active_classes: Sequence[int] = ALL_CLASSES,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    """Inner-loop model selection with pre-defined ensemble composition.

    All individual models are scored on the full inner-fold validation set.
    The ensemble is composed from the top-N models by mean ROC-AUC across
    inner folds (a pre-defined, data-independent rule), then also scored on
    the same full validation set.  Because the composition rule does not
    peek at the evaluation data, there is no selection bias.
    """
    active_classes = list(active_classes)
    n_splits = pick_n_splits(y_train, groups_train, requested_inner_splits, active_classes)
    # Spawn independent RNGs: one for CV, one per fold
    cv_rng, fold_parent = spawn_rngs(rng, 2)
    cv_rs = int(cv_rng.integers(0, 2**31 - 1))
    inner_cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=cv_rs)

    fold_rngs = spawn_rngs(fold_parent, n_splits)
    rows: List[Dict[str, object]] = []

    # Phase 1: score all individual models across inner folds.
    # We rank models by selection_score (ROC-AUC, the same threshold-free
    # criterion used to choose between strategies), NOT macro-F1 -- so the
    # ensemble is composed of the models that are actually best on the metric
    # everything else is judged on.
    per_model_sel_scores: Dict[str, List[float]] = {
        n: [] for n in models if not n.startswith("baseline_")
    }

    for fold_i, (tr_idx, va_idx) in enumerate(
        inner_cv.split(X_train, y_train, groups_train), start=1,
    ):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        g_tr = groups_train[tr_idx]

        cw = compute_class_weights(y_tr, active_classes)
        sw = make_sample_weight(y_tr, cw)

        # Spawn per-model RNGs so model order doesn't matter
        model_rngs = spawn_rngs(fold_rngs[fold_i - 1], len(models))
        for m_idx, (name, model) in enumerate(models.items()):
            is_bl = name.startswith("baseline_")
            metrics = evaluate_candidate(
                model, X_tr, y_tr, X_va, y_va,
                None if is_bl else sw,
                enable_calibration and not is_bl,
                model_rngs[m_idx],
                None if is_bl else g_tr,
                active_classes,
            )
            row = {
                "inner_fold": fold_i, "strategy_type": "single",
                "strategy_name": name, "weight_policy": "inverse_freq",
                "calibration_in_eval": bool(enable_calibration and not is_bl),
            }
            row.update(metrics)
            rows.append(row)

            if not is_bl:
                per_model_sel_scores[name].append(selection_score(metrics))

    # Phase 2: pre-defined ensemble composition from mean ROC-AUC rankings
    mean_sel_table = {
        name: float(np.mean(scores)) for name, scores in per_model_sel_scores.items()
        if scores
    }
    ens_name, ens_model, ens_weights = make_predefined_ensemble(
        models, mean_sel_table, ensemble_top_n,
    )

    # Phase 3: score the pre-composed ensemble on the same inner folds
    inner_cv2 = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=cv_rs)
    ens_fold_rngs = spawn_rngs(rng, n_splits)

    for fold_i, (tr_idx, va_idx) in enumerate(
        inner_cv2.split(X_train, y_train, groups_train), start=1,
    ):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        g_tr = groups_train[tr_idx]
        cw = compute_class_weights(y_tr, active_classes)
        sw = make_sample_weight(y_tr, cw)

        ens_metrics = evaluate_candidate(
            ens_model, X_tr, y_tr, X_va, y_va, sw,
            enable_calibration, ens_fold_rngs[fold_i - 1], g_tr,
            active_classes,
        )
        row = {
            "inner_fold": fold_i, "strategy_type": "ensemble",
            "strategy_name": ens_name, "weight_policy": "inverse_freq",
            "ensemble_weights": json.dumps({nm: round(wt, 6) for nm, wt in ens_weights}),
            "calibration_in_eval": bool(enable_calibration),
        }
        row.update(ens_metrics)
        rows.append(row)

    # Also record best-single-finalizer per fold (best model by mean ROC-AUC)
    best_single_name = max(mean_sel_table, key=mean_sel_table.get)
    inner_cv3 = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=cv_rs)
    for fold_i, (tr_idx, va_idx) in enumerate(
        inner_cv3.split(X_train, y_train, groups_train), start=1,
    ):
        single_row = [
            r for r in rows
            if r.get("inner_fold") == fold_i
            and r.get("strategy_name") == best_single_name
            and r.get("strategy_type") == "single"
        ]
        if single_row:
            bsf = {
                "inner_fold": fold_i, "strategy_type": "best_single_finalizer",
                "strategy_name": best_single_name, "weight_policy": "inverse_freq",
                "calibration_in_eval": single_row[0].get("calibration_in_eval", False),
            }
            for k in ("macro_f1", "balanced_accuracy", "mcc", "roc_auc_ovr_macro"):
                bsf[k] = single_row[0].get(k, np.nan)
            for c in active_classes:
                for m in ("precision", "recall", "f1", "support"):
                    key = f"class_{c}_{m}"
                    bsf[key] = single_row[0].get(key, np.nan)
            rows.append(bsf)

    inner_df = pd.DataFrame(rows)

    # Compare ensemble vs best-single by mean ROC-AUC (selection_score),
    # falling back to macro-F1 only where AUC is undefined for a fold.
    summary = (
        inner_df[inner_df["strategy_type"].isin(["ensemble", "best_single_finalizer"])]
        .groupby(["strategy_type", "strategy_name", "weight_policy"], as_index=False)[
            ["roc_auc_ovr_macro", "macro_f1", "balanced_accuracy", "mcc"]
        ]
        .mean()
    )
    summary["selection_score"] = summary["roc_auc_ovr_macro"].where(
        summary["roc_auc_ovr_macro"].notna(), summary["macro_f1"],
    )
    best = summary.sort_values("selection_score", ascending=False).iloc[0]

    return {
        "strategy_type": str(best["strategy_type"]),
        "strategy_name": str(best["strategy_name"]),
        "weight_policy": str(best["weight_policy"]),
        "n_inner_splits_used": int(n_splits),
    }, inner_df


# =====================================================================
#  Stats / null model
# =====================================================================

def paired_significance(final_vals, base_vals) -> Tuple[Optional[float], str]:
    if len(final_vals) != len(base_vals) or len(final_vals) < 2:
        return None, "insufficient_pairs"
    if HAS_SCIPY:
        try:
            stat = wilcoxon(final_vals, base_vals, alternative="greater", zero_method="wilcox")
            return float(stat.pvalue), "wilcoxon_greater"
        except Exception:
            return None, "wilcoxon_failed"
    return None, "scipy_unavailable"


def build_strategy_model_builder(
    models, chosen_strategy_type, chosen_strategy_name,
    chosen_weight_policy, inner_df, ensemble_top_n: int = 3,
) -> Tuple[Callable[[], BaseEstimator], Optional[List[Tuple[str, float]]]]:
    if chosen_strategy_type == "best_single_finalizer":
        return (lambda n=chosen_strategy_name: clone(models[n])), None

    single_only = inner_df[
        (inner_df["strategy_type"] == "single")
        & (~inner_df["strategy_name"].str.startswith("baseline_"))
        & (inner_df["weight_policy"] == chosen_weight_policy)
    ]
    # Rank members by mean ROC-AUC (selection_score), matching
    # choose_best_strategy; fall back to macro-F1 only where AUC is undefined.
    means = (
        single_only.groupby("strategy_name", as_index=False)[
            ["roc_auc_ovr_macro", "macro_f1"]
        ].mean()
    )
    score_table = {
        row["strategy_name"]: float(
            row["roc_auc_ovr_macro"]
            if pd.notna(row["roc_auc_ovr_macro"])
            else row["macro_f1"]
        )
        for _, row in means.iterrows()
    }
    _, ens_model, ens_weights = make_predefined_ensemble(
        models, score_table, ensemble_top_n,
    )
    return (lambda em=ens_model: clone(em)), ens_weights


def run_y_scrambling(
    prepared: PreparedTarget, repeats: int,
    master_rng: np.random.Generator,
    requested_outer_splits: int, requested_inner_splits: int,
    enable_calibration: bool, mi_feature_fraction: float,
    ensemble_top_n: int = 3, xgb_device: str = "cpu",
) -> pd.DataFrame:
    if repeats <= 0:
        return pd.DataFrame()

    rep_rngs = spawn_rngs(master_rng, repeats)
    rows = []

    active_classes = prepared.active_classes

    for rep in range(1, repeats + 1):
        rng = rep_rngs[rep - 1]
        shuffle_rng, cv_rng, fold_parent = spawn_rngs(rng, 3)

        y_scr = prepared.y.copy()
        shuffle_rng.shuffle(y_scr)

        n_outer = pick_n_splits(y_scr, prepared.groups, requested_outer_splits, active_classes)
        cv_rs = int(cv_rng.integers(0, 2**31 - 1))
        outer_cv = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=cv_rs)
        fold_rngs = spawn_rngs(fold_parent, n_outer)
        rep_scores = []

        for fold_idx, (tr_idx, te_idx) in enumerate(
            outer_cv.split(prepared.X, y_scr, prepared.groups), start=1,
        ):
            X_tr, X_te = prepared.X[tr_idx], prepared.X[te_idx]
            y_tr, y_te = y_scr[tr_idx], y_scr[te_idx]
            g_tr = prepared.groups[tr_idx]
            frng = fold_rngs[fold_idx - 1]
            model_rng, inner_rng, cal_rng = spawn_rngs(frng, 3)

            mdls = get_candidate_models(
                model_rng, mi_feature_fraction, xgb_device, len(active_classes),
            )
            chosen, idf = choose_best_strategy(
                mdls, X_tr, y_tr, g_tr, inner_rng,
                requested_inner_splits, enable_calibration, ensemble_top_n,
                active_classes,
            )
            cw = compute_class_weights(y_tr, active_classes)
            sw = make_sample_weight(y_tr, cw)

            builder, _ = build_strategy_model_builder(
                mdls, chosen["strategy_type"], chosen["strategy_name"],
                chosen["weight_policy"], idf, ensemble_top_n,
            )

            calibrator = None
            if enable_calibration:
                calibrator, _ = build_oof_calibrator(
                    builder, X_tr, y_tr, g_tr, sw,
                    chosen["n_inner_splits_used"], cal_rng, active_classes,
                )

            fm = CalibratedThresholdedWrapper(
                base_estimator=builder(), calibrator=calibrator,
            )
            fm.fit(X_tr, y_tr, sample_weight=sw)
            rep_scores.append(f1_score(y_te, fm.predict(X_te), average="macro"))

        rows.append({
            "repeat": rep,
            "scrambled_macro_f1_mean": float(np.mean(rep_scores)),
            "scrambled_macro_f1_std": float(np.std(rep_scores)),
        })

    return pd.DataFrame(rows)


# =====================================================================
#  Outer CV
# =====================================================================

def _fit_evaluate_split(
    tid: str,
    active_classes: List[int],
    feature_names: List[str],
    X_tr: np.ndarray, y_tr: np.ndarray, g_tr: np.ndarray, train_df: pd.DataFrame,
    X_te: np.ndarray, y_te: np.ndarray, te_df: pd.DataFrame,
    frng: np.random.Generator,
    cfg: RunConfig,
    threshold_grid: List[float],
    out_dir: Path,
    split_id: str,
    split_col_value: object,
    n_outer_splits_used: int,
) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame, List[Dict[str, object]], Optional[pd.DataFrame]]:
    """Model-selects (inner CV), calibrates, fits ONE final model on
    (X_tr, y_tr, g_tr) and evaluates it on the untouched (X_te, y_te).

    This is the single implementation shared by every nested-CV outer fold
    AND by the true 20%% holdout evaluation (fit_evaluate_holdout) -- the
    holdout is scored by exactly the same methodology as a CV fold, not a
    hand-rolled variant of it; only what it's called (X_tr/X_te) differs.

    Returns (fold_row, pred_df, inner_df, diagnostics_rows, importance_df).
    """
    ac = np.array(active_classes)
    model_rng, inner_rng, cal_rng, perm_rng, boundary_rng = spawn_rngs(frng, 5)
    g_te = te_df["scaffold"].astype(str).to_numpy()

    diagnostics_rows = [
        {"target": tid, "outer_fold": split_col_value, "split": "train",
         **fold_diagnostics(y_tr, g_tr, active_classes)},
        {"target": tid, "outer_fold": split_col_value, "split": "test",
         **fold_diagnostics(y_te, g_te, active_classes)},
    ]

    models = get_candidate_models(
        model_rng, cfg.mi_feature_fraction, cfg.xgb_resolved_device,
        len(active_classes),
    )
    chosen, inner_df = choose_best_strategy(
        models, X_tr, y_tr, g_tr, inner_rng,
        cfg.inner_splits, cfg.enable_calibration, cfg.ensemble_top_n,
        active_classes,
    )
    inner_df.insert(0, "target", tid)
    inner_df.insert(1, "outer_fold", split_col_value)

    class_weights = compute_class_weights(y_tr, active_classes)
    sample_weight = make_sample_weight(y_tr, class_weights)

    # Baselines
    bl_prior = fit_estimator(models["baseline_prior"], X_tr, y_tr, None)
    bl_strat = fit_estimator(models["baseline_stratified"], X_tr, y_tr, None)
    bl_prior_m = compute_metrics(
        y_te, bl_prior.predict(X_te), bl_prior.predict_proba(X_te), active_classes,
    )
    bl_strat_m = compute_metrics(
        y_te, bl_strat.predict(X_te), bl_strat.predict_proba(X_te), active_classes,
    )

    builder, ens_weights = build_strategy_model_builder(
        models, chosen["strategy_type"], chosen["strategy_name"],
        chosen["weight_policy"], inner_df, cfg.ensemble_top_n,
    )

    # Calibration
    calibrator, oof_uncal = None, None
    if cfg.enable_calibration:
        calibrator, oof_uncal = build_oof_calibrator(
            builder, X_tr, y_tr, g_tr, sample_weight,
            chosen["n_inner_splits_used"], cal_rng, active_classes,
        )

    # Threshold tuning
    thresholds = [1.0] * len(active_classes)
    if cfg.enable_threshold_tuning and not cfg.strict_mode:
        if oof_uncal is None:
            thr_rng = spawn_rngs(frng, 1)[0]
            thr_rs = int(thr_rng.integers(0, 2**31 - 1))
            cv = StratifiedGroupKFold(
                n_splits=chosen["n_inner_splits_used"],
                shuffle=True, random_state=thr_rs,
            )
            oof_uncal = np.zeros((len(y_tr), len(active_classes)), dtype=float)
            for tr2, va2 in cv.split(X_tr, y_tr, g_tr):
                est = builder()
                f2 = fit_estimator(est, X_tr[tr2], y_tr[tr2], sample_weight[tr2])
                oof_uncal[va2] = f2.predict_proba(X_tr[va2])
        oof_tuning = (
            calibrator.predict_proba(oof_uncal) if calibrator else oof_uncal
        )
        thresholds = tune_thresholds_on_validation(
            y_tr, oof_tuning, threshold_grid, active_classes,
        )

    # Final model
    final = CalibratedThresholdedWrapper(
        base_estimator=builder(), calibrator=calibrator, thresholds=thresholds,
    )
    final.fit(X_tr, y_tr, sample_weight=sample_weight)

    y_proba_uncal = final.predict_proba_uncalibrated(X_te)
    y_pred_uncal = ac[np.argmax(y_proba_uncal, axis=1)]
    m_uncal = compute_metrics(y_te, y_pred_uncal, y_proba_uncal, active_classes)

    y_proba_cal = final.predict_proba(X_te)
    y_pred_cal = ac[np.argmax(y_proba_cal / np.array(thresholds).reshape(1, -1), axis=1)]
    m_cal = compute_metrics(y_te, y_pred_cal, y_proba_cal, active_classes)

    # Boundary-instability diagnostics (static always; bootstrap optional)
    try:
        _bl, _bh = (float(x) for x in str(cfg.boundary_band).split(","))
    except Exception:
        _bl, _bh = 0.4, 0.6
    boundary = boundary_stability_metrics(
        y_proba_cal, y_pred_cal, y_pred_uncal, active_classes, band=(_bl, _bh),
    )
    if cfg.boundary_bootstrap > 0:
        _bflip, _bused = bootstrap_boundary_flip(
            builder, X_tr, y_tr, sample_weight, X_te, active_classes,
            cfg.boundary_bootstrap, boundary_rng,
        )
        boundary["boundary_bootstrap_flip_fraction"] = _bflip
        boundary["boundary_bootstrap_n"] = _bused

    # Applicability domain
    train_fps = smiles_to_rdkit_fp(train_df["canonical_smiles"].tolist())
    test_fps = smiles_to_rdkit_fp(te_df["canonical_smiles"].tolist())
    ad_k = cfg.ad_knn_k
    ad_thr = ad_threshold_from_train(train_fps, k=ad_k)
    query_sim = ad_query_scores(test_fps, train_fps, k=ad_k)
    in_ad = query_sim >= ad_thr

    # Prediction output
    pred_df = te_df[["canonical_smiles", "inchikey", "scaffold"]].copy()
    if cfg.id_col in te_df.columns:
        pred_df.insert(0, cfg.id_col, te_df[cfg.id_col].values)
    pred_df.insert(0, "outer_fold", split_col_value)
    pred_df.insert(0, "target", tid)
    pred_df["y_true"] = y_te
    pred_df["y_pred_uncalibrated"] = y_pred_uncal
    pred_df["y_pred_calibrated"] = y_pred_cal
    for i, c in enumerate(active_classes):
        pred_df[f"proba_uncal_{c}"] = y_proba_uncal[:, i]
        pred_df[f"proba_cal_{c}"] = y_proba_cal[:, i]
    pred_df["baseline_prior_pred"] = bl_prior.predict(X_te)
    pred_df["baseline_stratified_pred"] = bl_strat.predict(X_te)
    pred_df["ad_knn_mean_sim"] = query_sim
    pred_df["ad_threshold"] = ad_thr
    pred_df["ad_method"] = f"kNN_k{ad_k}_mean_minus_1std"
    pred_df["in_applicability_domain"] = in_ad.astype(int)
    if len(active_classes) == 2:
        _pp = y_proba_cal[:, active_classes.index(max(active_classes))]
        pred_df["boundary_uncertain"] = ((_pp >= _bl) & (_pp <= _bh)).astype(int)
        pred_df["pred_confidence_margin"] = np.abs(2.0 * _pp - 1.0)

    # Confusion matrix
    cm = confusion_matrix(y_te, y_pred_cal, labels=active_classes)
    pd.DataFrame(
        cm, index=[f"true_{i}" for i in active_classes],
        columns=[f"pred_{i}" for i in active_classes],
    ).to_csv(out_dir / "confusion_matrices" / f"{tid}_{split_id}_confusion.csv")

    # Permutation importance
    importance_df = None
    if cfg.permutation_importance and len(y_te) >= 20:
        try:
            prs = derive_int_seed(perm_rng)
            perm = permutation_importance(
                final, X_te, y_te, scoring="balanced_accuracy",
                n_repeats=10, random_state=prs, n_jobs=-1,
            )
            importance_df = pd.DataFrame({
                "target": tid, "outer_fold": split_col_value,
                "feature": feature_names,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            }).sort_values("importance_mean", ascending=False)
        except Exception as exc:
            LOGGER.warning("%s %s perm-imp failed: %s", tid, split_id, exc)

    # SHAP explainability. By default this runs on the FINAL held-out model
    # (split_id == "holdout"); per-fold SHAP is only produced when
    # save_shap_all_folds is set (audit runs), since it is expensive.
    run_shap_here = (
        cfg.save_shap and len(y_te) >= 10
        and (split_id == "holdout" or cfg.save_shap_all_folds)
    )
    if run_shap_here:
        maybe_run_shap(
            final, X_tr, X_te, feature_names,
            out_dir / "feature_importance" / f"{tid}_{split_id}_shap.csv",
            active_classes=active_classes,
        )

    # Save model
    joblib.dump({
        "model": final, "feature_names": feature_names,
        "target_id": tid, "class_weights": class_weights,
        "chosen_strategy": chosen, "thresholds": thresholds,
        "n_outer_splits_used": n_outer_splits_used,
        "calibration_enabled": bool(cfg.enable_calibration),
        "ad_method": f"kNN_k{ad_k}_mean_minus_1std",
    }, out_dir / "models" / f"{tid}_{split_id}.joblib")

    fold_row = {
        "target": tid, "outer_fold": split_col_value,
        "final_strategy_type": chosen["strategy_type"],
        "final_strategy_name": chosen["strategy_name"],
        "weight_policy": chosen["weight_policy"],
        "final_class_weights": json.dumps(class_weights, sort_keys=True),
        "ensemble_weights": (
            json.dumps({nm: round(wt, 6) for nm, wt in ens_weights})
            if ens_weights else ""
        ),
        "thresholds": json.dumps(thresholds),
        "calibration_enabled": bool(cfg.enable_calibration),
        "strict_mode": bool(cfg.strict_mode),
        "n_outer_splits_used": n_outer_splits_used,
        "n_inner_splits_used": chosen["n_inner_splits_used"],
        "baseline_prior_balanced_accuracy": bl_prior_m["balanced_accuracy"],
        "baseline_prior_macro_f1": bl_prior_m["macro_f1"],
        "baseline_prior_mcc": bl_prior_m["mcc"],
        "baseline_stratified_balanced_accuracy": bl_strat_m["balanced_accuracy"],
        "baseline_stratified_macro_f1": bl_strat_m["macro_f1"],
        "baseline_stratified_mcc": bl_strat_m["mcc"],
        "test_in_ad_fraction": float(in_ad.mean()),
        "ad_threshold": float(ad_thr),
        "ad_method": f"kNN_k{ad_k}_mean_minus_1std",
        "uncalibrated_balanced_accuracy": m_uncal["balanced_accuracy"],
        "uncalibrated_macro_f1": m_uncal["macro_f1"],
        "uncalibrated_mcc": m_uncal["mcc"],
        "uncalibrated_roc_auc_ovr_macro": m_uncal["roc_auc_ovr_macro"],
    }
    fold_row.update(m_cal)
    fold_row.update(boundary)

    return fold_row, pred_df, inner_df, diagnostics_rows, importance_df


def nested_scaffold_cv(
    prepared: PreparedTarget, out_dir: Path,
    master_rng: np.random.Generator,
    cfg: RunConfig,
    threshold_grid: List[float],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Model-selection / robustness evaluation via nested CV. Call this with
    the TRAINING POOL only (i.e. after split_holdout_indices has carved off
    the true holdout) -- every compound passed in here is used as training
    data in some fold and none of it should be the held-out test set."""
    X, y, groups = prepared.X, prepared.y, prepared.groups
    tid = prepared.target_id
    active_classes = prepared.active_classes
    n_outer = pick_n_splits(y, groups, cfg.outer_splits, active_classes)

    # Spawn independent RNGs for each outer fold
    cv_rng_gen, fold_parent = spawn_rngs(master_rng, 2)
    cv_rs = int(cv_rng_gen.integers(0, 2**31 - 1))
    outer_cv = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=cv_rs)
    fold_rngs = spawn_rngs(fold_parent, n_outer)

    fold_metrics, pred_frames, inner_frames = [], [], []
    diagnostics_rows, importance_frames = [], []

    for fold_idx, (tr_idx, te_idx) in enumerate(
        outer_cv.split(X, y, groups), start=1,
    ):
        LOGGER.info("%s | outer fold %d/%d", tid, fold_idx, n_outer)
        frng = fold_rngs[fold_idx - 1]

        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        g_tr = groups[tr_idx]
        train_df = prepared.df.iloc[tr_idx].reset_index(drop=True)
        test_df = prepared.df.iloc[te_idx].reset_index(drop=True)

        fold_row, pred_df, inner_df, diag_rows, importance_df = _fit_evaluate_split(
            tid, active_classes, prepared.feature_names,
            X_tr, y_tr, g_tr, train_df,
            X_te, y_te, test_df,
            frng, cfg, threshold_grid, out_dir,
            split_id=f"fold{fold_idx}", split_col_value=fold_idx,
            n_outer_splits_used=n_outer,
        )
        fold_metrics.append(fold_row)
        pred_frames.append(pred_df)
        inner_frames.append(inner_df)
        diagnostics_rows.extend(diag_rows)
        if importance_df is not None:
            importance_frames.append(importance_df)

    metrics_df = pd.DataFrame(fold_metrics)
    preds_df = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    inner_all = pd.concat(inner_frames, ignore_index=True) if inner_frames else pd.DataFrame()
    diag_df = pd.DataFrame(diagnostics_rows)

    if importance_frames:
        pd.concat(importance_frames, ignore_index=True).to_csv(
            out_dir / "feature_importance" / f"{tid}_permutation_importance.csv",
            index=False,
        )

    return metrics_df, preds_df, inner_all, diag_df


def fit_evaluate_holdout(
    prepared_train: PreparedTarget, prepared_holdout: PreparedTarget,
    out_dir: Path, master_rng: np.random.Generator,
    cfg: RunConfig, threshold_grid: List[float], n_outer_splits_used: int,
) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame]:
    """The TRUE held-out evaluation. Model selection, calibration and the
    final fit all happen on 100%% of prepared_train and NEVER see
    prepared_holdout; that one final model then predicts on prepared_holdout
    -- compounds it has not touched in any capacity. Uses the exact same
    _fit_evaluate_split logic as every nested-CV outer fold, so the holdout
    number is held to the same methodology as the fold-level CV estimates,
    not a separately hand-rolled evaluation path.
    """
    tid = prepared_train.target_id
    active_classes = prepared_train.active_classes
    frng = spawn_rngs(master_rng, 1)[0]

    fold_row, pred_df, inner_df, diag_rows, importance_df = _fit_evaluate_split(
        tid, active_classes, prepared_train.feature_names,
        prepared_train.X, prepared_train.y, prepared_train.groups, prepared_train.df,
        prepared_holdout.X, prepared_holdout.y, prepared_holdout.df,
        frng, cfg, threshold_grid, out_dir,
        split_id="holdout", split_col_value="holdout",
        n_outer_splits_used=n_outer_splits_used,
    )
    inner_df.to_csv(out_dir / "model_selection" / f"{tid}_holdout_strategy_scores.csv", index=False)
    if importance_df is not None:
        importance_df.to_csv(
            out_dir / "feature_importance" / f"{tid}_holdout_permutation_importance.csv",
            index=False,
        )
    pd.DataFrame(diag_rows).to_csv(
        out_dir / "diagnostics" / f"{tid}_holdout_diagnostics.csv", index=False,
    )

    return fold_row, pred_df, pd.DataFrame(diag_rows)


# =====================================================================
#  Summaries
# =====================================================================

def summarize_target_metrics(metrics_df, prepared) -> Dict[str, object]:
    active_classes = prepared.active_classes
    summary: Dict[str, object] = {"n_outer_folds": int(len(metrics_df))}
    cols = [
        "accuracy", "balanced_accuracy", "macro_f1", "mcc",
        "roc_auc_ovr_macro", "pr_auc",
        "boundary_uncertain_fraction", "boundary_mean_margin",
        "boundary_calib_flip_fraction",
        "uncalibrated_balanced_accuracy", "uncalibrated_macro_f1",
        "uncalibrated_mcc", "uncalibrated_roc_auc_ovr_macro",
        "baseline_prior_balanced_accuracy", "baseline_prior_macro_f1",
        "baseline_prior_mcc",
        "baseline_stratified_balanced_accuracy", "baseline_stratified_macro_f1",
        "baseline_stratified_mcc",
    ]
    for c in active_classes:
        cols += [f"class_{c}_precision", f"class_{c}_recall", f"class_{c}_f1"]
    cols.append("test_in_ad_fraction")
    for col in cols:
        v = metrics_df[col].astype(float).to_numpy()
        summary[f"{col}_mean"] = float(np.nanmean(v))
        summary[f"{col}_std"] = float(np.nanstd(v))

    for metric, bl_metric in [
        ("balanced_accuracy", "baseline_prior_balanced_accuracy"),
        ("macro_f1", "baseline_prior_macro_f1"),
        ("balanced_accuracy", "baseline_stratified_balanced_accuracy"),
        ("macro_f1", "baseline_stratified_macro_f1"),
    ]:
        pval, method = paired_significance(
            metrics_df[metric].to_numpy(), metrics_df[bl_metric].to_numpy(),
        )
        summary[f"pvalue_{metric}_vs_{bl_metric}"] = pval
        summary[f"pvalue_method_{metric}_vs_{bl_metric}"] = method

    class_n = {c: int((prepared.y == c).sum()) for c in active_classes}
    total = len(prepared.y)
    nz = [v for v in class_n.values() if v > 0]
    summary.update({
        "target": prepared.target_id,
        "n_molecules": total,
        "n_scaffolds": int(prepared.df["scaffold"].nunique()),
        **{f"class_{c}_n": n for c, n in class_n.items()},
        **{f"class_{c}_frac": n / total for c, n in class_n.items()},
        "imbalance_ratio_max_over_min": float(max(nz) / min(nz)) if nz else np.nan,
        **prepared.audit,
    })
    return summary


def export_chemical_space(prepared, out_path, rng):
    rs = derive_int_seed(rng)
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("var", VarianceThreshold()),
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=2, random_state=rs)),
    ])
    coords = pipe.fit_transform(prepared.X)
    out = prepared.df[["canonical_smiles", "inchikey", "scaffold"]].copy()
    out["class_label"] = prepared.y
    out["pc1"] = coords[:, 0]
    out["pc2"] = coords[:, 1]
    out.to_csv(out_path, index=False)


# =====================================================================
#  Self checks
# =====================================================================

def run_self_checks(cfg: RunConfig) -> None:
    assert normalize_label("blocker") == 1
    assert normalize_label("Non-blocker") == 0
    assert normalize_label(1) == 1
    assert normalize_label(0) == 0
    assert normalize_label(1.0) == 1
    assert normalize_label("unknown") is None

    _bs = boundary_stability_metrics(
        np.array([[0.95, 0.05], [0.5, 0.5], [0.1, 0.9]]),
        np.array([0, 1, 1]), np.array([0, 0, 1]), [0, 1], band=(0.4, 0.6),
    )
    assert abs(_bs["boundary_uncertain_fraction"] - (1.0 / 3.0)) < 1e-9
    assert abs(_bs["boundary_calib_flip_fraction"] - (1.0 / 3.0)) < 1e-9

    tiny_y = np.array([0, 0, 0, 1, 1, 1])
    tiny_g = np.array(["a", "b", "c", "d", "e", "f"])
    if pick_n_splits(tiny_y, tiny_g, 5) < 2:
        raise RuntimeError("Self-check failed: pick_n_splits.")

    if any(float(x) <= 0 for x in cfg.threshold_grid.split(",")):
        raise ValueError("Threshold grid values must be > 0.")
    if cfg.min_per_class < 2:
        raise ValueError("min_per_class must be >= 2.")
    if not Path(cfg.data_dir).exists():
        raise FileNotFoundError(f"Data directory not found: {cfg.data_dir}")
    if cfg.xgb_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("xgb_device must be one of: auto, cpu, cuda.")
    if cfg.xgb_gpu_id < 0:
        raise ValueError("xgb_gpu_id must be >= 0.")


def write_manifest(out_dir: Path) -> None:
    manifest = {
        "summary_all_targets.csv": "Per-target averaged outer-fold metrics (train-pool CV) PLUS holdout_* columns "
                                    "-- the true, never-trained-on 20% test performance -- baselines, significance, class distribution.",
        "skipped_targets.csv": "Targets skipped (insufficient data or errors).",
        "run_metadata.json": "Resolved configuration and environment flags.",
        "run.log": "Execution log.",
        "run_<timestamp>.log": "Timestamped archival copy of the execution log.",
        "predictions/<target>_holdout_ids.csv": "The compounds carved out as the TRUE 20% holdout (inchikey + id_col) "
                                                 "-- reusable as --holdout_ids for another model (e.g. the regressor) "
                                                 "to score the same shared test set.",
        "predictions/<target>_holdout_predictions.csv": "Predictions on the true holdout from the ONE final model "
                                                          "fit on 100% of the train pool -- these compounds were never "
                                                          "used in model selection, calibration, or training.",
        "cv_metrics/<target>_holdout_metrics.csv": "Single-row metrics for the true holdout evaluation above.",
        "cv_metrics/<target>_outer_fold_metrics.csv": "Per-fold metrics from nested CV over the TRAIN POOL ONLY "
                                                       "(model-selection robustness estimate, not the holdout).",
        "predictions/<target>_outer_fold_predictions.csv": "Per-molecule out-of-fold predictions from the train-pool "
                                                             "nested CV, probabilities, AD flags.",
        "model_selection/<target>_inner_strategy_scores.csv": "Inner-fold strategy scores (individual + ensemble), train pool.",
        "model_selection/<target>_holdout_strategy_scores.csv": "Inner-fold strategy scores for the holdout's final-model selection.",
        "diagnostics/<target>_fold_diagnostics.csv": "Fold-wise class counts and scaffold spread, train pool.",
        "diagnostics/<target>_holdout_diagnostics.csv": "Class counts / scaffold spread for the train-pool-vs-holdout split.",
        "chemical_space/<target>_pca.csv": "2D PCA projection (full dataset, train+holdout).",
        "confusion_matrices/<target>_fold*_confusion.csv": "Outer-fold confusion matrices, train pool.",
        "confusion_matrices/<target>_holdout_confusion.csv": "Confusion matrix on the true holdout.",
        "feature_importance/<target>_permutation_importance.csv": "Permutation importance (when enabled), train pool folds.",
        "feature_importance/<target>_holdout_permutation_importance.csv": "Permutation importance on the holdout's final model.",
        "feature_importance/<target>_fold*_shap.csv": "Per-fold SHAP summaries (audit runs / save_shap_all_folds only).",
        "feature_importance/<target>_holdout_shap.csv": "SHAP feature importance for the final held-out model "
                                                          "(ranked mean_abs_shap_overall, plus mean_abs_shap_blocker); "
                                                          "ensemble members combined by their soft-voting weights.",
        "null_distributions/<target>_y_scrambling.csv": "Y-scrambling null (when enabled), train pool.",
        "models/<target>_fold*.joblib": "Saved per-fold models from train-pool nested CV.",
        "models/<target>_holdout.joblib": "THE deployable final model: fit on 100% of the train pool, "
                                          "scored once on the true holdout.",
    }
    with open(out_dir / "output_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


# =====================================================================
#  Run
# =====================================================================

def configure_logging(log_file: Path, level: str) -> logging.FileHandler:
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level))
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(getattr(logging, level))
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setLevel(getattr(logging, level))
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return fh


def apply_mode(cfg: RunConfig) -> RunConfig:
    if cfg.run_mode not in MODE_PRESETS:
        raise ValueError(f"Unknown run_mode: {cfg.run_mode}")
    for k, v in MODE_PRESETS[cfg.run_mode].items():
        setattr(cfg, k, v)
    return cfg


def run(cfg: RunConfig) -> None:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "run.log"
    fh = configure_logging(log_path, cfg.log_level)
    cfg = apply_mode(cfg)
    run_self_checks(cfg)
    cfg.xgb_resolved_device = resolve_xgb_device(cfg.xgb_device, cfg.xgb_gpu_id)
    LOGGER.info(
        "XGBoost device requested=%s resolved=%s",
        cfg.xgb_device,
        cfg.xgb_resolved_device,
    )

    master_rng = seed_everything(cfg.seed)

    for sub in [
        "models", "predictions", "cv_metrics", "model_selection",
        "confusion_matrices", "feature_importance", "diagnostics",
        "chemical_space", "null_distributions",
    ]:
        (out_dir / sub).mkdir(exist_ok=True)
    write_manifest(out_dir)

    files = sorted(Path(cfg.data_dir).glob(cfg.pattern))
    if not files:
        raise FileNotFoundError(
            f"No files in {cfg.data_dir!r} matching {cfg.pattern!r}",
        )

    full_df = pd.concat(
        [load_target_csv(p, cfg.smiles_col, cfg.label_col, cfg.target_id_col) for p in files],
        ignore_index=True,
    )
    target_ids = sorted(full_df["__target_id__"].dropna().astype(str).unique())
    LOGGER.info("Run mode: %s | Targets: %d", cfg.run_mode, len(target_ids))

    threshold_grid = [float(x) for x in cfg.threshold_grid.split(",")]
    overall, skipped = [], []

    # Spawn one independent RNG per target
    target_rngs = spawn_rngs(master_rng, len(target_ids))

    for t_idx, target_id in enumerate(target_ids):
        try:
            LOGGER.info("=" * 80)
            LOGGER.info("Target: %s", target_id)
            prepared = prepare_target_dataframe(
                full_df, target_id, cfg.min_per_class, cfg.fp_bits,
            )
            if prepared is None:
                skipped.append({"target": target_id, "reason": "insufficient_data"})
                continue

            trng = target_rngs[t_idx]
            pca_rng, cv_rng, scr_rng, split_rng, holdout_rng = spawn_rngs(trng, 5)

            # ---- true held-out test set: carved off BEFORE anything else
            # touches the data, so it is never part of model selection,
            # calibration, or the final fit -- see split_holdout_indices.
            train_idx, holdout_idx = split_holdout_indices(prepared, cfg, split_rng)
            prepared_train = subset_prepared(prepared, train_idx)
            has_holdout = len(holdout_idx) > 0
            if has_holdout:
                prepared_holdout = subset_prepared(prepared, holdout_idx)
                ids_out = prepared.df.iloc[holdout_idx][["inchikey"]].copy()
                if cfg.id_col in prepared.df.columns:
                    ids_out[cfg.id_col] = prepared.df.iloc[holdout_idx][cfg.id_col].values
                ids_out.to_csv(out_dir / "predictions" / f"{target_id}_holdout_ids.csv", index=False)
            else:
                LOGGER.warning(
                    "%s: holdout disabled (holdout_frac=%.3f, holdout_ids=%s) -- "
                    "reporting nested-CV estimates only, no true held-out number.",
                    target_id, cfg.holdout_frac, cfg.holdout_ids,
                )

            export_chemical_space(
                prepared, out_dir / "chemical_space" / f"{target_id}_pca.csv", pca_rng,
            )

            # ---- model selection / robustness CV -- TRAIN POOL ONLY --------
            mdf, pdf, idf, ddf = nested_scaffold_cv(
                prepared_train, out_dir, cv_rng, cfg, threshold_grid,
            )
            mdf.to_csv(out_dir / "cv_metrics" / f"{target_id}_outer_fold_metrics.csv", index=False)
            pdf.to_csv(out_dir / "predictions" / f"{target_id}_outer_fold_predictions.csv", index=False)
            idf.to_csv(out_dir / "model_selection" / f"{target_id}_inner_strategy_scores.csv", index=False)
            ddf.to_csv(out_dir / "diagnostics" / f"{target_id}_fold_diagnostics.csv", index=False)

            if cfg.y_scramble_repeats > 0:
                ndf = run_y_scrambling(
                    prepared_train, cfg.y_scramble_repeats, scr_rng,
                    cfg.outer_splits, cfg.inner_splits,
                    cfg.enable_calibration, cfg.mi_feature_fraction,
                    cfg.ensemble_top_n, cfg.xgb_resolved_device,
                )
                if not ndf.empty:
                    ndf.to_csv(
                        out_dir / "null_distributions" / f"{target_id}_y_scrambling.csv",
                        index=False,
                    )

            summary = summarize_target_metrics(mdf, prepared_train)
            summary["n_molecules_total"] = int(len(prepared.y))
            summary["n_holdout"] = int(len(holdout_idx))

            # ---- TRUE held-out evaluation: select + fit ONE final model on
            # 100% of the train pool, score it on the never-touched holdout.
            if has_holdout:
                n_outer_meta = pick_n_splits(
                    prepared_train.y, prepared_train.groups, cfg.outer_splits,
                    prepared_train.active_classes,
                )
                holdout_row, holdout_pdf, _ = fit_evaluate_holdout(
                    prepared_train, prepared_holdout, out_dir, holdout_rng,
                    cfg, threshold_grid, n_outer_meta,
                )
                pd.DataFrame([holdout_row]).to_csv(
                    out_dir / "cv_metrics" / f"{target_id}_holdout_metrics.csv", index=False,
                )
                holdout_pdf.to_csv(
                    out_dir / "predictions" / f"{target_id}_holdout_predictions.csv", index=False,
                )
                summary.update({
                    f"holdout_{k}": v for k, v in holdout_row.items()
                    if k not in ("target", "outer_fold")
                })
                LOGGER.info(
                    "%s | HOLDOUT (true, never-trained-on %d compounds): "
                    "balanced_accuracy=%.3f macro_f1=%.3f roc_auc=%.3f",
                    target_id, len(holdout_idx),
                    holdout_row.get("balanced_accuracy", float("nan")),
                    holdout_row.get("macro_f1", float("nan")),
                    holdout_row.get("roc_auc_ovr_macro", float("nan")),
                )

            overall.append(summary)

        except Exception as exc:
            LOGGER.error("Target %s failed: %s", target_id, exc)
            LOGGER.error(traceback.format_exc())
            skipped.append({"target": target_id, "reason": f"error: {exc}"})

    if overall:
        pd.DataFrame(overall).sort_values("macro_f1_mean", ascending=False).to_csv(
            out_dir / "summary_all_targets.csv", index=False,
        )
    else:
        pd.DataFrame(columns=["target"]).to_csv(out_dir / "summary_all_targets.csv", index=False)

    pd.DataFrame(skipped).to_csv(out_dir / "skipped_targets.csv", index=False)

    meta = asdict(cfg)
    meta.update({
        "xgboost_available": HAS_XGB, "shap_available": HAS_SHAP,
        "scipy_available": HAS_SCIPY, "torch_available": TORCH_AVAILABLE,
        "torch_cuda_available": bool(
            TORCH_AVAILABLE and torch.cuda.is_available()
        ) if TORCH_AVAILABLE else False,
        "xgb_device_resolved": cfg.xgb_resolved_device,
        "version": "v7_normal",
    })
    with open(out_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    LOGGER.info("Finished. Output: %s", out_dir)
    fh.flush()
    fh.close()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy2(log_path, out_dir / f"run_{ts}.log")


# =====================================================================
#  CLI
# =====================================================================

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Scaffold-aware nested-CV binary hERG-blockade classifier",
    )
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default="ml_ensemble_clf_results")
    ap.add_argument("--pattern", default="*_final.csv")
    ap.add_argument("--smiles_col", default="smiles")
    ap.add_argument("--label_col", default="hERG_blocker")
    ap.add_argument("--target_id_col", default="target_chembl_id")
    ap.add_argument("--id_col", default="compound_chembl_id",
        help="Compound id column carried into holdout predictions and used to "
             "match --holdout_ids alongside inchikey.")
    ap.add_argument("--holdout_ids", default=None,
        help="CSV of inchikeys/ids that BOTH models hold out (shared 20%% test), "
             "from make_scaffold_split.py. First column is "
             "matched against inchikey and --id_col. Overrides --holdout_frac.")
    ap.add_argument("--holdout_frac", type=float, default=0.20,
        help="Fraction excluded from ALL training/model-selection as a true "
             "held-out test set when --holdout_ids is not given (class-stratified, "
             "scaffold-grouped). 0 disables the holdout (old, pre-holdout behaviour).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_per_class", type=int, default=30)
    ap.add_argument("--fp_bits", type=int, default=2048)
    ap.add_argument("--threshold_grid", default="0.85,1.0,1.15,1.3")
    ap.add_argument("--run_mode", default="normal", choices=["fast", "normal", "audit"])
    ap.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    ap.add_argument(
        "--mi_feature_fraction", type=float, default=1.0,
        help="Fraction of features to retain via MI selection (1.0 = disabled).",
    )
    ap.add_argument("--ad_knn_k", type=int, default=5, help="k for kNN applicability domain.")
    ap.add_argument("--ensemble_top_n", type=int, default=3, help="Number of top models in ensemble.")
    ap.add_argument(
        "--xgb_device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device for XGBoost candidates. 'auto' probes CUDA once and falls back to CPU.",
    )
    ap.add_argument(
        "--xgb_gpu_id",
        type=int,
        default=0,
        help="CUDA device index used when --xgb_device resolves to CUDA.",
    )
    ap.add_argument(
        "--boundary_bootstrap", type=int, default=0,
        help="Train-only bootstrap refits per fold for the boundary-instability "
             "flip rate (0 = static metrics only; audit mode sets 10).",
    )
    ap.add_argument(
        "--boundary_band", default="0.4,0.6",
        help="Ambiguity band 'lo,hi' on positive-class probability (binary).",
    )
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    cfg = RunConfig(**{k: getattr(args, k) for k in vars(args)})
    run(cfg)


if __name__ == "__main__":
    main()