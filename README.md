# DL1 — Sparse→Full Sinogram Restoration for LOT/SLOT

This repository contains the **DL1** code used in our Localization Optoacoustic Tomography (LOT/SLOT) pipeline:
**sparse-channel sinogram → full-channel sinogram** restoration using an **unrolled reconstruction network** with
hard data-consistency.

> ⚠️ **Data are not included** (too large). The code expects local `.mat` files with LOT sinograms.

## What this code does (DL1)

- Input: a sinogram frame `S ∈ ℝ^{496×512}` where only a subset of columns (transducer channels) is measured.
- Prefill: fills missing columns (`linear`, `zero`, or `angular` interpolation along transducer angle φ).
- Network: **UnrolledReconstructor**  
  \(p_{k+1} = \text{DC}(p_k + (1-m)\cdot \text{Prox}_\theta([p_k, x_0, m, \text{coords}, \text{Fourier}, \text{geom}]))\)  
  with **hard DC** at each step (measured columns are enforced).
- Optional: Backprojection-based loss term (BP) for debug/training.

## Repository layout

```
slot_dl1_github_ready/
  train_dl1.py          # training script
  infer_dl1.py          # inference script (can process ALL frames)
  requirements.txt
  .gitignore
  assets/
    pos_sensor_xyz/     # (NOT INCLUDED) put pos_sensor_xyz_*.mat here
    indices/            # (OPTIONAL) put keep-indices .mat here
```

## Dependencies

Python ≥ 3.9 is recommended.

```bash
pip install -r requirements.txt
```

> `torch`/`torchvision` installation is environment-dependent (CUDA/ROCm/CPU).  
> Install them following the official PyTorch instructions for your system.

## Data format expected

Each `.mat` file under `--train/--val/--test` must contain a **3D numeric array** that can be interpreted as frames of
shape `(496, 512)`:
- `(496, 512, F)` or `(512, 496, F)` (will be transposed), or
- `(F, 496, 512)` / `(F, 512, 496)` in some exports.

The loader supports both MATLAB v7 and v7.3 (HDF5) `.mat`.

## Sensor geometry (`pos_sensor_xyz_*.mat`)

`--pos_xyz_dir` must contain the geometry files used by `pick_bp_meta_for_path()`:

- `pos_sensor_xyz_transducer_type1.mat`
- `pos_sensor_xyz_transducer_type2.mat`

Each file must contain a numeric array of shape `(512, 3)` (x,y,z coordinates per channel).
If your variable name is not auto-detected, pass `--pos_xyz_var`.

## Training

Example (ratio-based sparse sampling):

```bash
python train_dl1.py   --train ./data/train   --val   ./data/val   --test  ./data/test   --out   ./outputs/dl1_run   --pos_xyz_dir ./assets/pos_sensor_xyz   --sparse_source ratio --sparse_ratio 2 --offset 2   --train_start 5001 --val_start 15001 --test_start 15001 --num_frames 1000   --prefill_mode angular
```

If you use a measured-index `.mat`:

```bash
python train_dl1.py   --train ./data/train --val ./data/val --test ./data/test   --out ./outputs/dl1_run   --pos_xyz_dir ./assets/pos_sensor_xyz   --sparse_source index_mat --index_mat ./assets/indices/keep_indices.mat
```

Outputs:
- `best.pt`, `last.pt`
- `ckpt/epoch_XXX.pt`
- `samples/{train,val,test}/epoch_XXX.png` (sinogram triplets)
- `samples/*/*_mip.png` if BP visualization is enabled

## Inference (test-time)

The inference script can process **all frames** in each `.mat` file.

```bash
python infer_dl1.py   --test ./data/test   --out  ./outputs/dl1_infer   --ckpt ./outputs/dl1_run/best.pt   --pos_xyz_dir ./assets/pos_sensor_xyz   --sparse_source ratio --sparse_ratio 2 --offset 2   --save_pred_mat
```

- `--save_pred_mat` writes a v7.3(HDF5) `.mat` per input file:
  - dataset name: `sino_pred` (dtype: `uint16`)
  - values are mapped back to **RAW units per frame** using the same `(lo, hi)` normalization range.

## Notes for a professional GitHub release

- Remove hard-coded local paths (already done in this version).
- Add a short `CITATION.cff` (template suggested).
- Consider Git LFS for large model checkpoints (`*.pt`) if you plan to publish weights.
- If the dataset is not public, add a clear **Data availability** section in your paper + repo.

---

If you want, I can also:
- split the shared utilities (I/O, prefill, model, BP) into a small Python package (`slot_dl1/`),
- add a minimal **synthetic smoke-test** dataset generator for CI,
- add `environment.yml` (conda) and a `CITATION.cff` that matches your author list.
