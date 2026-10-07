# SLOT: Sparse Localization Optoacoustic Tomography

Code for the three networks of **"Learning-Based Sparse Localization Optoacoustic Tomography for Resource-Efficient
Super-Resolution Angiography of the Mouse Brain"** (Laser & Photonics Reviews).

| Network | Task | Input | Output |
|---|---|---|---|
| **DL 1** | sparse channel restoration | sinograms with every second channel (256 of 512) | full 512-channel sinograms |
| **DL 2** | laser repetition rate restoration | clutter-filtered sinograms at 20 Hz | clutter-filtered sinograms at 100 Hz |
| **DL 3** | short acquisition compensation | localization density and trajectory features of a short acquisition | localization density of the 270 s acquisition |

Each folder contains a training script (`train.py`) and an evaluation script (`eval.py`).
The localization and tracking of the microcapsules are not part of this repository. DL 3 starts from the localizations
and trajectories in the CSV format described below.

## Repository layout

```
DL1/  train.py, eval.py
DL2/  train.py, eval.py, lot_svd.py   (lot_svd.py: SVD clutter filter and data preparation)
DL3/  train.py, eval.py
requirements.txt
```

## Installation

```
pip install -r requirements.txt
```

Install PyTorch matching your CUDA version (https://pytorch.org). DL 2 inference requires about 13 GB of GPU memory
(peak values measured on an NVIDIA RTX 4090: DL 1 0.7 GB, DL 2 13.3 GB, DL 3 8.0 GB).

## Example data and trained weights

Download the files from the [Releases](../../releases) page and place them as follows:

```
data/
  sample_sinogram.mat        1000 frames (10 s at 100 Hz) of a test animal, dataset 'sigMat' (frames x 512 x 496, uint16)
  channel_geometry.npz       azimuth (phi) and normalized radius (r_norm) of the 512 channels, used by DL 1
  dl3_sample/
    localizations.csv.gz     frame, x, y, z, ncc
    trajectories.csv.gz      traj_id, frame, x, y, z
weights/
  DL1_best.pt  DL2_best.pt  DL3_best.pt
```

Coordinates in the CSV files are in voxels of the 128^3 reconstruction grid (78.125 um), and frames are 0-based at 100 Hz.

## Evaluation on the example data

```
python DL1/eval.py --test data/sample_sinogram.mat --ckpt weights/DL1_best.pt --geometry data/channel_geometry.npz --out results/dl1
python DL2/eval.py --test data/sample_sinogram.mat --ckpt weights/DL2_best.pt --out results/dl2 --first_start 1 --last_start 1
python DL3/eval.py --pos data/dl3_sample --ckpt weights/DL3_best.pt --out results/dl3 --save_mips
```

| Script | Reported metrics |
|---|---|
| `DL1/eval.py` | NRMSE on the missing channels for angular interpolation (network input) and DL 1 |
| `DL2/eval.py` | NRMSE on the restored frames for linear interpolation and DL 2, against the clutter-filtered 100 Hz data |
| `DL3/eval.py` | background fraction, Dice, centerline coverage, faint-vessel coverage, branch-point ratio and density error of the time-rescaled input and DL 3 for 30, 60, 90 and 135 s inputs |

Options: `--save_pred_mat` (DL 1) saves the restored sinograms, `--sparse_source random` (DL 1) and `--meas_jitter 0`
(DL 2) evaluate random channel layouts and irregular frame timing.

## Training

DL 1 and DL 2 are trained on fully sampled sinograms (`.mat`, dataset `sigMat`), from which the sparse inputs are
generated. DL 3 is trained on position folders containing `localizations.csv(.gz)` and `trajectories.csv(.gz)` of
acquisitions of at least 270 s.

```
python DL1/train.py --train <train_dir> --val <val_dir> --geometry data/channel_geometry.npz --out runs/dl1
python DL2/train.py --train <train_dir> --val <val_dir> --out runs/dl2 --residual_linear
python DL3/train.py --train_dirs <pos_1> <pos_2> ... --val_dirs <pos_val> --out runs/dl3
```

The default arguments correspond to the settings used in the paper (DL 1: 2-fold channel reduction, K = 4 iterations;
DL 2: 5-fold frame reduction, 10 s windows; DL 3: 30 to 135 s inputs). DL 2 for other repetition rates is trained with
`--ds 2`, `--ds 4` or `--ds 10`.

## Citation

If you use this code, please cite:

```
D. Seong, D. Nozdriukhin, Y. Chen, J. Kim, M. Jeon, X. L. Deán-Ben, D. Razansky,
"Learning-Based Sparse Localization Optoacoustic Tomography for Resource-Efficient Super-Resolution Angiography
of the Mouse Brain," Laser & Photonics Reviews (2026).
```

## License

MIT License (see `LICENSE`).
