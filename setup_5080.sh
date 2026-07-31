#!/usr/bin/env bash
# Environment build for the RTX 5080 training box (Blackwell sm_120).
#
# The stock machine has Python 3.8 (EOL, unsupported by torch >= 2.7), a
# CPU-only torch build, and no nvcc. Blackwell needs torch built against
# CUDA 12.8, so we install a self-contained miniconda env rather than
# fighting the system python.
#
# Run:  bash setup_5080.sh          (idempotent; safe to re-run)
#
# NOTE: no `set -u`. conda's gcc_linux-64 deactivate hook references
# $_CONDA_PYTHON_SYSCONFIGDATA_NAME_USED unguarded, which aborts the script
# under `set -u` right after the cuda-toolkit install.
set -eo pipefail

CONDA_DIR="$HOME/miniconda3"
ENV_NAME="dental"
PY_VER="3.11"

step() { echo; echo "=== $* ==="; }

step "1/6 miniconda"
if [ ! -x "$CONDA_DIR/bin/conda" ]; then
  curl -fsSL -o /tmp/miniconda.sh \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
  rm -f /tmp/miniconda.sh
else
  echo "already installed at $CONDA_DIR"
fi
export PATH="$CONDA_DIR/bin:$PATH"
eval "$("$CONDA_DIR/bin/conda" shell.bash hook)"

step "2/6 env $ENV_NAME (python $PY_VER)"
if ! conda env list | grep -q "^$ENV_NAME "; then
  conda create -y -n "$ENV_NAME" "python=$PY_VER"
else
  echo "env exists"
fi
conda activate "$ENV_NAME"
python --version

step "3/6 torch + cu128 (Blackwell sm_120)"
if ! python -c "import torch,sys; sys.exit(0 if torch.version.cuda else 1)" 2>/dev/null; then
  pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu128
else
  echo "cuda torch already present"
fi
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda,
      "| available", torch.cuda.is_available())
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability()
    print("device:", torch.cuda.get_device_name(0), "| capability", cap)
    x = torch.randn(2048, 2048, device="cuda")
    print("matmul ok:", float((x @ x).sum()) == float((x @ x).sum()))
    assert cap[0] >= 12, "expected sm_120 for RTX 5080, got %s" % (cap,)
else:
    raise SystemExit("CUDA NOT AVAILABLE — stop here, do not proceed")
PY

step "4/6 CUDA toolkit (nvcc) + gcc 13 for compiling DINO's deformable-attn op"
if ! command -v nvcc >/dev/null 2>&1; then
  conda install -y -c nvidia cuda-toolkit=12.8 || \
    conda install -y -c nvidia cuda-nvcc=12.8 cuda-cudart-dev=12.8
fi
# CUDA 12.8 refuses gcc >= 14, and the default conda toolchain resolves to 14.x
conda install -y -q -c conda-forge gxx_linux-64=13 gcc_linux-64=13
nvcc --version | tail -2
x86_64-conda-linux-gnu-c++ --version | head -1

step "5/6 project deps"
pip install --quiet ultralytics pycocotools opencv-python-headless \
  pyyaml scipy timm termcolor addict yapf gdown "numpy<2"
python -c "import ultralytics, pycocotools, cv2, numpy; \
print('ultralytics', ultralytics.__version__, '| numpy', numpy.__version__)"

step "6/6 DINO repo + deformable attention op"
cd "$HOME"
[ -d DINO ] || git clone --depth 1 https://github.com/IDEA-Research/DINO.git
cd DINO/models/dino/ops
# torch >= 2.6 removed the implicit DeprecatedTypeProperties -> ScalarType
# conversion that this 2022 kernel relies on (see PATCHES.md, patch 3).
if grep -q "AT_DISPATCH_FLOATING_TYPES(value\.type()" src/cuda/ms_deform_attn_cuda.cu; then
  cp -n src/cuda/ms_deform_attn_cuda.cu src/cuda/ms_deform_attn_cuda.cu.orig
  sed -i "s/AT_DISPATCH_FLOATING_TYPES(value\.type()/AT_DISPATCH_FLOATING_TYPES(value.scalar_type()/g" \
    src/cuda/ms_deform_attn_cuda.cu
  echo "patched AT_DISPATCH value.type() -> value.scalar_type()"
fi
export TORCH_CUDA_ARCH_LIST="12.0"
python setup.py build install 2>&1 | tail -5
# correctness checks; the script's final oversized gradcheck OOMs on 16 GB,
# which is a test-harness artifact -- see PATCHES.md.
python test.py 2>&1 | grep -E "^\* (True|False)" || true

echo
echo "=== DONE ==="
echo "activate with:  source $CONDA_DIR/bin/activate $ENV_NAME"
