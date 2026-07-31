# Upstream patches required to run DINO on the RTX 5080 (Blackwell, torch 2.11)

Every deviation from the stock [IDEA-Research/DINO](https://github.com/IDEA-Research/DINO)
repo is recorded here so an independent re-run reproduces byte-for-byte. Nothing
below changes model behaviour — these are API-compatibility and toolchain fixes.

## Environment as built (verified 2026-07-28)

| component | version | note |
|---|---|---|
| GPU | RTX 5080, 16 GB, driver 570.133.07 | Blackwell, compute capability **12.0** |
| python | 3.11.15 (miniconda env `dental`) | system python 3.8 is EOL and unusable with torch ≥2.7 |
| torch | **2.11.0+cu128** | cu128 required; earlier wheels have no sm_120 kernels |
| nvcc | 12.8.93 | conda `cuda-toolkit=12.8` |
| gcc/g++ | **13.4.0** (conda-forge) | see patch 2 |
| ultralytics | 8.4.108 | |
| numpy | 1.26.4 | pinned <2 for ultralytics/pycocotools compatibility |

`torch.cuda.get_device_capability()` returns `(12, 0)` and a CUDA matmul was
executed as a live check before proceeding.

## Patch 1 — `set -u` vs conda's gcc deactivate hook

`setup_5080.sh` runs with `set -eo pipefail`, **not** `set -euo pipefail`.
conda's `deactivate-gcc_linux-64.sh` dereferences
`$_CONDA_PYTHON_SYSCONFIGDATA_NAME_USED` unguarded, which aborts the script
immediately after the cuda-toolkit install under `set -u`.

## Patch 2 — CUDA 12.8 rejects gcc 14

Default conda toolchain resolved to gcc 14.3.0; CUDA 12.8 requires
`>=6.0, <14.0`:

```
RuntimeError: The current installed version of .../x86_64-conda-linux-gnu-c++
(14.3.0) is greater than the maximum required version by CUDA 12.8.
```

Fix: `conda install -c conda-forge gxx_linux-64=13 gcc_linux-64=13`.

## Patch 3 — deprecated ATen dispatch API in the deformable-attention kernel

`models/dino/ops/src/cuda/ms_deform_attn_cuda.cu` (2022 code) calls
`AT_DISPATCH_FLOATING_TYPES(value.type(), ...)`. torch 2.11 removed the
implicit `at::DeprecatedTypeProperties → c10::ScalarType` conversion:

```
error: no suitable conversion function from "const at::DeprecatedTypeProperties"
       to "c10::ScalarType" exists
```

Fix (lines 64 and 134; original preserved as `ms_deform_attn_cuda.cu.orig`):

```diff
-        AT_DISPATCH_FLOATING_TYPES(value.type(), "ms_deform_attn_forward_cuda", ([&] {
+        AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_forward_cuda", ([&] {
-        AT_DISPATCH_FLOATING_TYPES(value.type(), "ms_deform_attn_backward_cuda", ([&] {
+        AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_backward_cuda", ([&] {
```

`value.type().is_cuda()` elsewhere in the file still compiles and was left
untouched to keep the diff minimal.

Build: `TORCH_CUDA_ARCH_LIST="12.0" python setup.py build install`.

## Verification of the compiled op (`models/dino/ops/test.py`)

```
* True check_forward_equal_with_pytorch_double: max_abs_err 8.67e-19  max_rel_err 1.98e-16
* True check_forward_equal_with_pytorch_float:  max_abs_err 4.66e-10  max_rel_err 1.13e-07
* True check_gradient_numerical(D=30)
* True check_gradient_numerical(D=32)
* True check_gradient_numerical(D=64)
* True check_gradient_numerical(D=71)
* True check_gradient_numerical(D=1025)
```

The script's final oversized `gradcheck` call OOMs at ~15 GB on a 16 GB card.
That is a property of the test harness (double-precision numerical gradient
over a large tensor), not of training: DINO runs `hidden_dim=256` / 8 heads →
D=32 per head, which is covered by the passing cases above.

## Patch 4 — DINO checkpoint re-heading (not a code patch, but a required flag)

The COCO class head appears in four places (`class_embed.{0..5}`,
`transformer.decoder.class_embed.{0..5}`, `transformer.enc_out_class_embed`,
`label_enc`). `--finetune_ignore` matches by substring, so the verified
minimal cover is:

```
--finetune_ignore class_embed label_enc
```

This drops exactly 27 head tensors and loads the other 599. Passing
`transformer` would also match the entire encoder/decoder stack and silently
discard the pretrained weights. Derived with
`python tools/inspect_checkpoint.py weights/checkpoint0033_4scale.pth`.
