# SLOT Deep Learning Code (DL1–DL3)

This repository contains the **deep learning code** for the three-stage pipeline used in our **localization optoacoustic tomography (LOT)** work:

- **DL1**: sparse transducer-channel sinogram restoration (e.g., 256 → 512 channels)
- **DL2**: temporal sinogram restoration / frame interpolation (low repetition rate → full-rate sequence)
- **DL3**: 3D reconstruction enhancement for low acquisition time (input volume → GT-quality volume)

⚠️ **Data are not included** in this repository (large size / sharing constraints). The scripts assume you have local `.mat` files following the expected folder layout below.

---

## Repository layout

```
.
├── DL1_train.py
├── DL1_infer.py
├── DL2_train.py
├── DL2_infer.py
├── DL3_train.py
├── DL3_infer.py
├── requirements.txt
└── .gitignore
```

---

## Environment

### Option A) pip + venv

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

Notes:
- Install **PyTorch** according to your CUDA driver/toolkit if you use GPU.

---

## Data layout (expected)

You can adapt paths via CLI flags.

### DL1

```
data/
  dl1/
    train/   *.mat
    val/     *.mat
    test/    *.mat
  pos_sensor_xyz/
    pos_sensor_xyz_ultracup.mat
    pos_sensor_xyz_holycup.mat
```

If you use an explicit index list instead of a regular sparse pattern:

```
data/
  dl1/
    measured_indices.mat
```

### DL2

```
data/
  dl2/
    train/   *.mat
    val/     *.mat
    test/    *.mat
  pos_sensor_xyz/
    pos_sensor_xyz_ultracup.mat
    pos_sensor_xyz_holycup.mat
```

### DL3

```
data/
  dl3/
    input/
      train/ *.mat
      val/   *.mat
      test/  *.mat
    gt/
      train/ *.mat
      val/   *.mat
      test/  *.mat
```

---

## Quick start

### DL1 training

```bash
python DL1_train.py \
  --train data/dl1/train --val data/dl1/val --test data/dl1/test \
  --out outputs/dl1 \
  --pos_xyz_dir data/pos_sensor_xyz \
  --sparse_source ratio --sparse_ratio 2
```

### DL1 inference

```bash
python DL1_infer.py \
  --test data/dl1/test \
  --out outputs/dl1_infer \
  --ckpt outputs/dl1/best.pt \
  --pos_xyz_dir data/pos_sensor_xyz \
  --sparse_source ratio --sparse_ratio 2 \
  --save_pred_mat
```

### DL2 training

```bash
python DL2_train.py \
  --train data/dl2/train --val data/dl2/val --test data/dl2/test \
  --out outputs/dl2 \
  --pos_xyz_dir data/pos_sensor_xyz \
  --sparse_ratio 5 --offset 0 \
  --t_chunk 32 --stride_t 32
```

### DL2 inference

```bash
python DL2_infer.py \
  --test data/dl2/test \
  --out outputs/dl2_infer \
  --ckpt outputs/dl2/best.pt \
  --sparse_ratio 5 --offset 0
```

### DL3 training

```bash
python DL3_train.py \
  --input_root data/dl3/input \
  --gt_root data/dl3/gt \
  --out outputs/dl3 \
  --target_size 256 256 256
```

### DL3 inference

```bash
python DL3_infer.py \
  --input_root data/dl3/input/test \
  --gt_root data/dl3/gt/test \
  --workdir outputs/dl3
```

---

## Notes

- The scripts support both MATLAB v7 `.mat` and MATLAB v7.3 (HDF5-backed) `.mat`.
- Some datasets may store the sinogram/volume under a non-standard variable name; several scripts provide a `--var_hint` argument for that case.
- This repo is intended to be a **clean, shareable code release**. Paths are CLI-configurable and large outputs/data are ignored by `.gitignore`.

---

## License

MIT License (see `LICENSE`).
