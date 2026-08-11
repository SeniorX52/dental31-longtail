# Experiment drivers

Every arm reported in `RESULTS.md` and in the papers was produced by one of the
scripts here. They are kept because reproducible code is a requirement of this work, and
because a result whose exact command is lost is not reproducible. They are not
meant to be re-run casually: several take a day of GPU time, and each one guards
against re-running work whose checkpoint says it already finished.

Each script carries its reasoning in the header — why the arm exists, what it is
controlled against, and what would falsify it. Read that before running one.

## Reading them

| script | what it produced |
|---|---|
| `run_seg_ablation*.sh`, `run_seg_completion.sh` | the seven-arm segmentation grid (S0–S4, SB, SCP, SNW) |
| `run_dino_baseline.sh`, `run_dino_ablation*.sh` | the DINO-DETR baseline and the frozen detection matrix (D1–D7, C1, L1–L3) |
| `run_dino_tau.sh` | the τ sweep that diagnosed the unified method's failure |
| `run_hr1280.sh`, `run_hr1600.sh`, `run_compound.sh` | the resolution probes and the from-scratch confirmation seeds |
| `run_coeff_arms.sh`, `run_k1b.sh`, `run_k2seeds.sh` | the coefficient-head arms (K1, K1b, K2) |
| `run_hires_proto.sh`, `run_architecture_arms.sh` | the prototype-resolution and architecture arms (XP2, XP3) |
| `run_maskdino.sh` | the Mask DINO family test |
| `run_phase2.sh` | learning curve (LC25/50/75) and the second-generation backbone |
| `run_noise_floor.sh` | the three-seed noise floor every claim is measured against |
| `run_toothstage.sh` | the tooth-conditioned two-stage experiment |
| `fetch_dentex.sh` | acquires DENTEX 2023 for the external check |
| `run_overnight.sh`, `run_pipeline.sh`, `final_*_run.sh` | chain drivers that sequenced the above |

## A caveat about paths

These scripts `cd` to the repository root and were written when they lived
there, so they refer to sibling drivers by bare name. That is still correct for
the `case` patterns they use to detect a running driver, which match both `./x.sh`
and `*/x.sh`. Run them as `experiments/<name>.sh` from the repository root.

Drivers for work that is still queued remain at the repository root, because a
reboot-resume cron launches them by path.
