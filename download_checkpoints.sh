#!/usr/bin/env bash
# download_checkpoints.sh
# ========================
# Fetches the ONE checkpoint too large for git (GitHub hard-rejects >100 MB
# pushes) from a GitHub Release asset, then verifies all 4 checkpoints
# master_predict.py loads at inference against SHA256SUMS.txt.
#
# The other 3 required files (pic50_regressor.joblib, cb_ft_final.pt,
# cb_ft_scaler.joblib) are small enough to be committed directly to git and
# are already present after `git clone` -- this script still checksums them
# so a single command verifies the whole checkpoints/ directory is intact.
#
# ---- release location -------------------------------------------------------
REPO="Abhiith-Paul-Tharakan/Discoverathon-hERG-FifthParadigm"
TAG="checkpoints-v1"         # the GitHub Release tag the checkpoint was uploaded to
# -------------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT_DIR="$HERE/checkpoints"
mkdir -p "$CKPT_DIR/cb_ft_pure_results"
LARGE_FILE="herg_combined_dataset_holdout.joblib"
URL="https://github.com/${REPO}/releases/download/${TAG}/${LARGE_FILE}"
if [ -f "$CKPT_DIR/$LARGE_FILE" ]; then
    echo "[download_checkpoints] $LARGE_FILE already present, skipping download."
else
    echo "[download_checkpoints] fetching $LARGE_FILE from $URL ..."
    curl -fL --retry 3 -o "$CKPT_DIR/$LARGE_FILE" "$URL"
fi
for f in pic50_regressor.joblib cb_ft_pure_results/cb_ft_final.pt cb_ft_pure_results/cb_ft_scaler.joblib; do
    if [ ! -f "$CKPT_DIR/$f" ]; then
        echo "[download_checkpoints] ERROR: $f is missing." >&2
        echo "  This file ships committed in git -- did you clone with --depth 1" >&2
        echo "  against a shallow ref, or delete it locally? Re-clone the repo." >&2
        exit 1
    fi
done
echo "[download_checkpoints] verifying SHA-256 checksums ..."
cd "$CKPT_DIR"
sha256sum -c "$HERE/SHA256SUMS.txt"
echo "[download_checkpoints] all 4 checkpoints present and verified."