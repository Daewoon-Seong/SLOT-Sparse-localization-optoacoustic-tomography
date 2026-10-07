"""
DL 3: localization density of a long acquisition from a short one (training).

Inputs per voxel (256^3 grid, 39 um): log(1 + localization count) of the short acquisition, the mean flow direction
(3 channels), mean speed and number of distinct particles from the trajectories, and the acquisition-time fraction.
The network predicts a correction to the log of the time-rescaled input density, trained with a Poisson likelihood
against the localization count of the 270 s acquisition of the same position.

usage:
  python train.py --train_dirs <pos1> <pos2> ... --val_dirs <posN> --out <dir>
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

LV3_G, LV3_SR = 256, 2                              # 256^3 grid = 2x the 128^3 reconstruction grid
LV3_F0, LV3_F1 = 2000, 28999                        # 0-based frames, 270 s at 100 Hz after the arrival of the particles
LV3_VOX_MM = 10.0 / 128                             # voxel size of the 128^3 grid (mm)
LV3_TFULL = (LV3_F1 - LV3_F0 + 1) / 100.0
LV3_EPS = 0.01                                      # floor of the rescaled baseline rate (counts per voxel)


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(inplace=True),
        )
    def forward(self,x): return self.net(x)


class Down3D(nn.Module):
    def __init__(self): super().__init__(); self.pool = nn.MaxPool3d(2)
    def forward(self,x): return self.pool(x)


class Up3D(nn.Module):
    def __init__(self,in_ch,out_ch): super().__init__(); self.up = nn.ConvTranspose3d(in_ch,out_ch,2,stride=2)
    def forward(self,x): return self.up(x)


class UNet3D_Loc(nn.Module):
    """3D U-Net (three levels) predicting a correction to the log of the time-rescaled input density.
    The last layer is zero-initialized, so that training starts exactly at time rescaling."""
    def __init__(self, in_ch, base=16):
        super().__init__()
        c = base
        self.e1 = ConvBlock3D(in_ch, c); self.d1 = Down3D()
        self.e2 = ConvBlock3D(c, 2 * c); self.d2 = Down3D()
        self.e3 = ConvBlock3D(2 * c, 4 * c); self.d3 = Down3D()
        self.b = ConvBlock3D(4 * c, 8 * c)
        self.u3 = Up3D(8 * c, 4 * c); self.dec3 = ConvBlock3D(8 * c, 4 * c)
        self.u2 = Up3D(4 * c, 2 * c); self.dec2 = ConvBlock3D(4 * c, 2 * c)
        self.u1 = Up3D(2 * c, c); self.dec1 = ConvBlock3D(2 * c, c)
        self.out = nn.Conv3d(c, 1, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x):
        e1 = self.e1(x); x = self.d1(e1); e2 = self.e2(x)
        x = self.d2(e2); e3 = self.e3(x)
        x = self.d3(e3); x = self.b(x)
        x = self.u3(x); x = torch.cat([x, e3], 1); x = self.dec3(x)
        x = self.u2(x); x = torch.cat([x, e2], 1); x = self.dec2(x)
        x = self.u1(x); x = torch.cat([x, e1], 1); x = self.dec1(x)
        return self.out(x)                          # log-rate correction


def _find(folder, stem):
    for ext in (".csv", ".csv.gz"):
        f = Path(folder) / (stem + ext)
        if f.exists():
            return f
    raise FileNotFoundError(f"{folder}: {stem}.csv(.gz) not found")


class LocPosition:
    """Localizations (NCC >= ncc) and trajectories of one scan position.
    folder/localizations.csv(.gz): frame, x, y, z, ncc     folder/trajectories.csv(.gz): traj_id, frame, x, y, z
    Coordinates in voxels of the 128^3 reconstruction grid (78.125 um), frames 0-based at 100 Hz."""
    def __init__(self, folder, ncc=0.6):
        self.name = Path(folder).name
        d = pd.read_csv(_find(folder, "localizations"), usecols=["frame", "x", "y", "z", "ncc"])
        d = d[(d.ncc >= ncc) & (d.frame >= LV3_F0) & (d.frame <= LV3_F1)]
        self.lf = d.frame.to_numpy(np.int64); self.lx = d[["x", "y", "z"]].to_numpy(np.float32)
        t = pd.read_csv(_find(folder, "trajectories"))
        t = t.rename(columns={c: c.strip() for c in t.columns})
        t = t[(t.frame >= LV3_F0) & (t.frame <= LV3_F1)].sort_values(["traj_id", "frame"], kind="stable")
        self.tid = t.traj_id.to_numpy(np.int64); self.tf = t.frame.to_numpy(np.int64)
        self.tx = t[["x", "y", "z"]].to_numpy(np.float32)
        self.target = lv3_density(self.lx)          # full 270 s counts

    def inputs(self, f_lo, f_hi, variant):
        m = (self.lf >= f_lo) & (self.lf < f_hi)
        dens = lv3_density(self.lx[m])
        T = (f_hi - f_lo) / 100.0
        chans = [np.log1p(dens)]
        if variant == "full":
            chans += lv3_track_maps(self.tid, self.tf, self.tx, f_lo, f_hi)
        chans.append(np.full_like(dens, T / LV3_TFULL))
        base = np.log(dens * (LV3_TFULL / T) + LV3_EPS)
        return np.stack(chans, 0).astype(np.float32), base.astype(np.float32)


def lv3_density(xyz):
    idx = np.clip(np.floor(xyz * LV3_SR).astype(np.int64), 0, LV3_G - 1)
    lin = np.ravel_multi_index(idx.T, (LV3_G,) * 3)
    return np.bincount(lin, minlength=LV3_G ** 3).astype(np.float32).reshape((LV3_G,) * 3)


def lv3_track_maps(tid, tf, tx, f_lo, f_hi):
    """distinct particles, mean unit flow direction (3), mean speed (mm/s / 10) of the window's track pieces"""
    G3 = LV3_G ** 3
    m = (tf >= f_lo) & (tf < f_hi)
    tid, tf, p = tid[m], tf[m], tx[m] * LV3_SR
    out = [np.zeros((LV3_G,) * 3, np.float32) for _ in range(5)]
    if len(tid) < 2:
        return out
    u, first = np.unique(tid, return_index=True)
    last = np.r_[first[1:], len(tid)] - 1
    dur = (tf[last] - tf[first]) * 0.01
    disp = (p[last] - p[first]) / LV3_SR
    ok = dur > 0
    nrm = np.linalg.norm(disp, axis=1)
    dirv = np.where(nrm[:, None] > 1e-6, disp / np.maximum(nrm, 1e-6)[:, None], 0)
    spd = np.where(ok, nrm * LV3_VOX_MM / np.maximum(dur, 1e-6), 0)
    same = tid[1:] == tid[:-1]
    a, b, st = p[:-1][same], p[1:][same], np.searchsorted(u, tid[1:][same])
    if len(a) == 0:
        return out
    L = np.linalg.norm(b - a, axis=1)
    n = np.maximum(1, np.ceil(L / 0.5).astype(np.int64)) + 1
    rep = np.repeat(np.arange(len(a)), n)
    frac = (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)) / np.repeat(n - 1, n).clip(min=1)
    pts = a[rep] + (b - a)[rep] * frac[:, None]
    lin = np.ravel_multi_index(np.clip(np.floor(pts).astype(np.int64), 0, LV3_G - 1).T, (LV3_G,) * 3)
    pair = np.unique(np.c_[st[rep], lin], axis=0)
    k, v = pair[:, 0], pair[:, 1]
    keep = ok[k]; k, v = k[keep], v[keep]
    cnt = np.bincount(v, minlength=G3).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        for j in range(3):
            out[1 + j] = np.nan_to_num(np.bincount(v, weights=dirv[k, j], minlength=G3) / cnt).astype(np.float32).reshape((LV3_G,) * 3)
        out[4] = (np.nan_to_num(np.bincount(v, weights=spd[k], minlength=G3) / cnt) / 10.0).astype(np.float32).reshape((LV3_G,) * 3)
    out[0] = np.log1p(cnt).reshape((LV3_G,) * 3)
    return out


def lv3_augment(x, base, y, variant):
    """random flips in x/y and 90 deg rotation in xy; direction channels transformed consistently"""
    full = variant == "full"
    x = x.copy()                                                # crops are views of the shared window arrays
    if random.random() < 0.5:                                   # flip x
        x, base, y = x[:, ::-1], base[::-1], y[::-1]
        if full: x[2] = -x[2]
    if random.random() < 0.5:                                   # flip y
        x, base, y = x[:, :, ::-1], base[:, ::-1], y[:, ::-1]
        if full: x[3] = -x[3]
    if random.random() < 0.5:                                   # rotate 90 deg in xy: (x, y) -> (-y, x)
        x = np.rot90(x, 1, axes=(1, 2)); base = np.rot90(base, 1, axes=(0, 1)); y = np.rot90(y, 1, axes=(0, 1))
        if full:
            ux, uy = x[2].copy(), x[3].copy(); x[2], x[3] = -uy, ux
    return np.ascontiguousarray(x), np.ascontiguousarray(base), np.ascontiguousarray(y)


def lv3_poisson(logr, y):
    return (torch.exp(logr) - y * logr).mean()


def lv3_eval(model, pos, T, variant, dev, q=0.005):
    x, base = pos.inputs(LV3_F0, LV3_F0 + int(T * 100), variant)
    xt = torch.from_numpy(x)[None].to(dev); bt = torch.from_numpy(base)[None, None].to(dev)
    logr = bt + model(xt)
    y = torch.from_numpy(pos.target)[None, None].to(dev)
    nll_m = float(lv3_poisson(logr, y)); nll_b = float(lv3_poisson(bt, y))
    from scipy.ndimage import gaussian_filter
    pred = torch.exp(logr)[0, 0].cpu().numpy(); resc = np.exp(base)
    # masks from Gaussian-smoothed maps (sigma 1 voxel), as in density_compare.py: raw count maps are dominated by
    # ties (most voxels empty), which makes a top-q threshold meaningless for the rescaled input
    gt_s = gaussian_filter(pos.target, 1.0)
    b_mask = gt_s > np.quantile(gt_s, 1 - q)
    def dice(v):
        v = gaussian_filter(v, 1.0)
        a = v > np.quantile(v, 1 - q)
        return 2 * (a & b_mask).sum() / max(a.sum() + b_mask.sum(), 1)
    return dict(nll_model=nll_m, nll_base=nll_b, ratio=nll_m / nll_b, dice_model=dice(pred), dice_rescale=dice(resc))


def train(a):
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out); (out / "ckpt").mkdir(parents=True, exist_ok=True)
    json.dump(vars(a), open(out / "run_args.json", "w"), indent=2)
    tr_dirs, va_dirs = [Path(d) for d in a.train_dirs], [Path(d) for d in a.val_dirs]
    print(f"train positions {len(tr_dirs)} | val positions {len(va_dirs)}")
    t0 = time.time()
    tr = [LocPosition(d, a.ncc) for d in tr_dirs]; va = [LocPosition(d, a.ncc) for d in va_dirs]
    print(f"loaded in {time.time()-t0:.0f}s")
    in_ch = 7 if a.variant == "full" else 2
    model = UNet3D_Loc(in_ch, base=a.base).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=a.lr * 0.05)
    log = open(out / "log.csv", "w", newline=""); lw = csv.writer(log)
    lw.writerow(["epoch", "train_loss"] + [f"{k}_{T:g}" for T in a.durations for k in ("ratio", "dice_model", "dice_rescale")] + ["sec"])
    best = float("inf"); C = a.crop
    for ep in range(1, a.epochs + 1):
        te = time.time(); model.train(); losses = []
        for _ in tqdm(range(a.windows_per_epoch), desc=f"epoch {ep}/{a.epochs}"):
            pos = random.choice(tr); T = random.choice(a.durations)
            nT = int(T * 100); s = random.randint(LV3_F0, LV3_F1 + 1 - nT)
            x, base = pos.inputs(s, s + nT, a.variant); y = pos.target
            for _ in range(a.crops):
                o = [random.randint(0, LV3_G - C) for _ in range(3)]
                sl = tuple(slice(k, k + C) for k in o)
                xc, bc, yc = lv3_augment(x[(slice(None),) + sl], base[sl], y[sl], a.variant)
                xt = torch.from_numpy(xc)[None].to(dev); bt = torch.from_numpy(bc)[None, None].to(dev)
                yt = torch.from_numpy(yc)[None, None].to(dev)
                loss = lv3_poisson(bt + model(xt), yt)
                if not torch.isfinite(loss):
                    continue
                opt.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                losses.append(float(loss))
        sched.step()
        model.eval(); row = [ep, float(np.mean(losses))]; ratios = []
        for T in a.durations:
            ev = [lv3_eval(model, p, T, a.variant, dev) for p in va]
            r = float(np.mean([e["ratio"] for e in ev])); ratios.append(r)
            dm = float(np.mean([e["dice_model"] for e in ev])); dr = float(np.mean([e["dice_rescale"] for e in ev]))
            row += [r, dm, dr]
            print(f"ep {ep:03d} T={T:>4g}s | NLL ratio model/rescale {r:.3f} | Dice model {dm:.3f} vs rescale {dr:.3f}", flush=True)
        row.append(time.time() - te); lw.writerow(row); log.flush()
        ck = {"epoch": ep, "model": model.state_dict(), "cfg": vars(a), "in_ch": in_ch}
        torch.save(ck, out / "ckpt" / f"epoch_{ep:03d}.pt")
        m = float(np.mean(ratios))
        if m < best:
            best = m; torch.save(ck, out / "best.pt")
        print(f"ep {ep:03d} train {np.mean(losses):.4f} | mean ratio {m:.3f} | {(time.time()-te)/60:.1f} min", flush=True)
    log.close()


def main():
    p = argparse.ArgumentParser(description="DL 3 training: short acquisition -> 270 s localization density")
    p.add_argument("--train_dirs", nargs="+", required=True, help="position folders with localizations and trajectories")
    p.add_argument("--val_dirs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--variant", choices=["full", "density"], default="full",
                   help="full: density + trajectory features; density: density and acquisition-time fraction only")
    p.add_argument("--durations", nargs="+", type=float, default=[30, 60, 90, 135], help="input acquisition times (s)")
    p.add_argument("--ncc", type=float, default=0.6)
    p.add_argument("--crop", type=int, default=128)
    p.add_argument("--crops", type=int, default=6, help="random crops per built window")
    p.add_argument("--windows_per_epoch", type=int, default=40)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--base", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    train(p.parse_args())


if __name__ == "__main__":
    main()
