#!/usr/bin/env bash
# Fetch DENTEX 2023, the external dental detection set, onto the removable SSD.
#
# DENTEX provides an independent panoramic radiograph corpus with box
# annotations, which is what makes a cross-dataset transfer result possible: a
# method that only ever improves numbers on the corpus it was tuned on has not
# been shown to generalise.
#
# It is NOT a long-tailed detection benchmark. It carries four disease classes,
# not a heavy tail, and it is smaller than the primary corpus. Any transfer
# result from it should be described as cross-dataset evidence and nothing more.
#
# Lives on the SSD because the archive is ~10 GB and nothing in the training
# chain depends on it; the chain must stay runnable when the drive is detached.
# Resumable: curl -C - continues a partial file, so an interrupted download
# picks up where it stopped instead of starting over.
set -o pipefail

DEST="${DENTEX_DIR:-/media/mostafa/EGYPT_SSD/dental31/dentex}"
BASE="https://huggingface.co/datasets/ibrahimhamamci/DENTEX/resolve/main/DENTEX"

if [ ! -d "$(dirname "$DEST")" ]; then
  echo "SSD not mounted at $(dirname "$DEST") -- connect it or set DENTEX_DIR" >&2
  exit 1
fi
# Do not run while a training job owns the machine. This download competes for
# page cache, and with cache=ram training already holding most of RAM the extra
# pressure is enough to push the box into swap exhaustion -- which is exactly
# how a 100-epoch run died at epoch 35 with a truncated checkpoint.
for p in $(pgrep -f "train_seg\.py|main\.py --output_dir" 2>/dev/null); do
  case "$(ps -o comm= -p "$p" 2>/dev/null)" in
    python*)
      echo "a training job is running -- refusing to add I/O load. Rerun when idle." >&2
      exit 0 ;;
  esac
done

mkdir -p "$DEST"
cd "$DEST"

stamp() { date "+%Y-%m-%d %H:%M:%S"; }

for f in training_data validation_data; do
  echo "=== [$(stamp)] $f.zip ==="
  # -C - resumes; --retry survives a dropped connection mid-transfer
  curl -L -C - --retry 5 --retry-delay 10 --retry-connrefused \
       -o "${f}.zip" "${BASE}/${f}.zip" || {
    echo "[$(stamp)] $f failed, will resume on the next run"; continue; }
  echo "[$(stamp)] $f.zip -> $(du -h "${f}.zip" | cut -f1)"
done

echo
echo "=== [$(stamp)] verifying archives ==="
for f in training_data validation_data; do
  if [ -f "${f}.zip" ]; then
    if unzip -t "${f}.zip" >/dev/null 2>&1; then
      echo "  ${f}.zip OK"
    else
      echo "  ${f}.zip INCOMPLETE -- rerun this script to resume"
    fi
  fi
done
echo "[$(stamp)] done"
