#!/usr/bin/env bash
# Fetch and preprocess the datasets the benchmark/training tests use.
#
#   ./scripts/setup_data.sh              # 32x32 images (default)
#   ./scripts/setup_data.sh --size 64    # higher resolution
#
# Needs: kaggle CLI (authenticated), python3 with numpy/pandas/pillow, unzip.
# Everything lands in data/kaggle/, which is gitignored.
set -euo pipefail

SIZE=32
while [ $# -gt 0 ]; do
    case "$1" in
        --size) SIZE="$2"; shift 2 ;;
        -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
        *) echo "unknown option: $1"; exit 1 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$REPO_ROOT/data/kaggle"
mkdir -p "$DATA_DIR"

echo ">> checking prerequisites"
command -v kaggle >/dev/null || {
    echo "ERROR: kaggle CLI not found. Install it with:  pip install kaggle"
    echo "Then create an API token at https://www.kaggle.com/settings and save it"
    echo "to ~/.kaggle/kaggle.json (chmod 600)."
    exit 1
}
command -v unzip >/dev/null || { echo "ERROR: unzip not found"; exit 1; }
python3 -c "import numpy, pandas, PIL" 2>/dev/null || {
    echo "ERROR: python deps missing. Install with:  pip install numpy pandas pillow"
    exit 1
}
kaggle competitions list >/dev/null 2>&1 || {
    echo "ERROR: kaggle CLI is installed but not authenticated."
    echo "Put your API token in ~/.kaggle/kaggle.json and chmod 600 it."
    exit 1
}

ITEM_COMP="cse-281-spring-26-item-price-prediction"
SCENE_COMP="cse-281-spring-26-scene-style-classification"

fetch() {
    local comp="$1" dest="$2"
    if [ -d "$DATA_DIR/$dest" ]; then
        echo ">> $dest already present, skipping download"
        return
    fi
    echo ">> downloading $comp"
    echo "   (you must accept the competition rules on kaggle.com first, or this 403s)"
    kaggle competitions download -c "$comp" -p "$DATA_DIR" --force
    mkdir -p "$DATA_DIR/$dest"
    unzip -oq "$DATA_DIR/$comp.zip" -d "$DATA_DIR/$dest"
    rm -f "$DATA_DIR/$comp.zip"
}

fetch "$ITEM_COMP" item_price
fetch "$SCENE_COMP" scene_style

echo ">> preprocessing (images at ${SIZE}x${SIZE})"
python3 "$REPO_ROOT/scripts/preprocess.py" --data-dir "$DATA_DIR" --size "$SIZE"

cat <<EOF

Setup complete.

  export VEARO_DATA_DIR="$DATA_DIR"
  cargo test --release -p vearo --test kaggle_bakeoff -- --ignored --nocapture

Data lives in $DATA_DIR (gitignored). Re-run with --size to change resolution.
EOF
