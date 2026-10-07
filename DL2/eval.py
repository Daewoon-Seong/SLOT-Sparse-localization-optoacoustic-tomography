"""
DL 2: restoration of the laser repetition rate (evaluation).

Restores the missing frames of a reduced-rate acquisition with a trained DL 2 checkpoint. The downsampling factor is
read from the checkpoint. The output pred_sino_fil.mat (HDF5) holds the clutter-filtered 100 Hz sinograms, with the
measured frames passed through unchanged. The restored frames are compared with the clutter-filtered fully sampled
frames, and linear interpolation of the measured frames is reported as the baseline (NRMSE on the missing frames).

usage:
  python eval.py --test ../sample_data/sample_sinogram.mat --ckpt ../weights/DL2_best.pt --out results_dl2 --first_start 1 --last_start 1
  python eval.py ... --meas_jitter 0      # irregular measured frames (one random frame per block of ds frames)
"""
import json
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import lot_svd
from lot_svd import prepare_window, sliding_infer, load_block_hwt, num_frames, linear_interp_time

class DoubleConv3D(nn.Module):
    def __init__(self,in_ch,out_ch,gn=8):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv3d(in_ch,out_ch,3,padding=1), nn.GroupNorm(min(gn,out_ch), out_ch), nn.SiLU(),
            nn.Conv3d(out_ch,out_ch,3,padding=1), nn.GroupNorm(min(gn,out_ch), out_ch), nn.SiLU(),
        )
    def forward(self,x): return self.net(x)


class UNet3D(nn.Module):
    def __init__(self,in_ch=2, base=16):
        super().__init__()
        b=base
        self.d1=DoubleConv3D(in_ch,b); self.p1=nn.MaxPool3d(2)
        self.d2=DoubleConv3D(b,b*2);   self.p2=nn.MaxPool3d(2)
        self.d3=DoubleConv3D(b*2,b*4); self.p3=nn.MaxPool3d((1,2,2))
        self.mid=DoubleConv3D(b*4,b*8)
        self.u3=nn.ConvTranspose3d(b*8,b*4,(1,2,2),stride=(1,2,2)); self.c3=DoubleConv3D(b*8,b*4)
        self.u2=nn.ConvTranspose3d(b*4,b*2,2,stride=2);             self.c2=DoubleConv3D(b*4,b*2)
        self.u1=nn.ConvTranspose3d(b*2,b,2,stride=2);               self.c1=DoubleConv3D(b*2,b)
        self.out=nn.Conv3d(b,1,1)
    def forward(self, x):
        B,C,H,W,T=x.shape
        pH=(2-H%2)%2; pW=(2-W%2)%2; pT=(2-T%2)%2
        x=F.pad(x,(0,pT,0,pW,0,pH),mode='replicate')
        d1=self.d1(x); p1=self.p1(d1)
        d2=self.d2(p1); p2=self.p2(d2)
        d3=self.d3(p2); p3=self.p3(d3)
        m=self.mid(p3)
        u3=self.u3(m);  c3=self.c3(torch.cat([u3,d3],dim=1))
        u2=self.u2(c3); c2=self.c2(torch.cat([u2,d2],dim=1))
        u1=self.u1(c2); c1=self.c1(torch.cat([u1,d1],dim=1))
        y=self.out(c1)
        return y[..., :H, :W, :T]


def list_mat_files(root: str) -> List[str]:
    root=Path(root)
    return sorted([str(p) for p in root.rglob("*.mat") if p.is_file() and not p.name.startswith("._")])


def run_infer(test, out_root, ckpt_path, first_start=2001, last_start=28001, stride=None, q_div=32.0, extra_starts=()):
    import h5py
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location=dev)
    cfg = ck.get("cfg", {})
    win, t_chunk, th1 = int(cfg.get("win", 1000)), int(cfg.get("t_chunk", 64)), int(cfg.get("th1", 11))
    ds = int(cfg.get("ds", 5))                                       # downsampling factor the model was trained for
    print(f"model ds = {ds}  (measured frames {'irregular, seed %s' % lot_svd.MEAS_JITTER_SEED if lot_svd.MEAS_JITTER_SEED is not None else 'regular'})")
    stride = stride or t_chunk // 2
    model = UNet3D(in_ch=2, base=int(cfg.get("base_ch", 16))).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    files = [test] if str(test).lower().endswith(".mat") else list_mat_files(test)
    for fp in files:
        stem = Path(fp).parent.name if Path(fp).name.startswith("pred_") else Path(fp).stem
        od = Path(out_root) / stem; od.mkdir(parents=True, exist_ok=True)
        T_all = num_frames(fp)
        starts = [s for s in list(extra_starts) + list(range(first_start, last_start + 1, win)) if s - 1 + win <= T_all]
        h5p = od / "pred_sino_fil.mat"
        if h5p.exists(): h5p.unlink()
        t0 = time.time()
        with h5py.File(h5p, "w") as hf:
            dset = hf.create_dataset("sino_fil", shape=(496, 512, T_all), dtype="int16",
                                     chunks=(496, 512, 1), compression="gzip", compression_opts=1, shuffle=True)
            step = np.zeros(T_all, np.float32)
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t_prep = t_net = 0.0
            err = {"y": 0.0, "dl2": 0.0, "lin": 0.0}
            for s in tqdm(starts, desc=stem):
                _t = time.time()
                d = prepare_window(load_block_hwt(fp, s, win), s, dev, th1=th1, ds=ds)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t_prep += time.time() - _t; _t = time.time()
                rl = bool(cfg.get("residual_linear", False))
                xin = linear_interp_time(d["x"], d["m"]) if rl else d["x"]
                pred = sliding_infer(model, xin, d["m"], t_chunk, stride, amp=False, residual=rl) * d["scale"]   # fp32, raw units
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t_net += time.time() - _t
                miss = d["m"][0, 0, 0, 0] < 0.5                       # restored frames: error against the filtered GT
                y = d["y"][..., miss]; lin = linear_interp_time(d["x"], d["m"])[..., miss]
                err["y"] += float((y ** 2).sum()); err["dl2"] += float(((pred / d["scale"])[..., miss] - y).pow(2).sum())
                err["lin"] += float((lin - y).pow(2).sum())
                q = d["scale"] / q_div
                arr = torch.clamp(torch.round(pred[0, 0] / q), -32767, 32767).to(torch.int16).cpu().numpy()
                dset[:, :, s - 1:s - 1 + win] = arr
                step[s - 1:s - 1 + win] = q
                del d, pred
            hf.create_dataset("step", data=step)
        meta = dict(file=str(fp), ckpt=str(ckpt_path), starts=starts, win=win, t_chunk=t_chunk, stride=stride,
                    th1=th1, q_div=q_div, minutes=(time.time() - t0) / 60,
                    load_svd_seconds=t_prep, network_seconds=t_net, windows=len(starts),
                    peak_gpu_mb=(torch.cuda.max_memory_allocated() / 2 ** 20 if dev.type == "cuda" else None))
        meta["nrmse_linear_interpolation"] = (err["lin"] / max(err["y"], 1e-12)) ** 0.5
        meta["nrmse_dl2"] = (err["dl2"] / max(err["y"], 1e-12)) ** 0.5
        meta["file"] = Path(fp).name; meta["ckpt"] = Path(ckpt_path).name
        json.dump(meta, open(od / "metrics.json", "w"), indent=2)
        print(f"{stem}: {len(starts)} windows, {meta['minutes']:.1f} min -> {h5p}")
        print(f"{stem}: NRMSE on the restored frames, linear interpolation {meta['nrmse_linear_interpolation']:.4f}, "
              f"DL 2 {meta['nrmse_dl2']:.4f}")
        print(f"[timing] DL2 {stem}: load + SVD {t_prep:.1f} s, network {t_net:.1f} s for {len(starts)} x 10 s windows, "
              f"peak GPU {meta['peak_gpu_mb']:.0f} MB")


def main():
    import argparse
    p = argparse.ArgumentParser(description="DL 2 evaluation: restoration of the missing frames")
    p.add_argument("--test", required=True, help=".mat file or folder of fully sampled sinograms (dataset 'sigMat')")
    p.add_argument("--out", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--first_start", type=int, default=1, help="first frame (1-based) of the first 10 s window")
    p.add_argument("--last_start", type=int, default=1, help="first frame of the last window")
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--extra_starts", default="", help="additional window starts")
    p.add_argument("--meas_jitter", type=int, default=None,
                   help="irregular measured frames: one random frame in every block of ds frames (seed)")
    a = p.parse_args()
    lot_svd.MEAS_JITTER_SEED = a.meas_jitter
    extra = [int(v) for v in a.extra_starts.split(",") if v.strip()]
    run_infer(a.test, a.out, a.ckpt, a.first_start, a.last_start, a.stride, extra_starts=extra)


if __name__ == "__main__":
    main()
