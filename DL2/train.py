"""
DL 2: restoration of the laser repetition rate (training).

The sinograms of the measured frames are clutter-filtered by SVD within 10 s windows (lot_svd.py), the missing frames
are initialized by linear interpolation, and a 3D U-Net learns a residual correction in short temporal blocks. The
measured frames are passed through unchanged (hard data consistency).

usage:
  python train.py --train <dir> --val <dir> --out <dir> --ds 5 --residual_linear
"""
import csv
import json
import random
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from lot_svd import prepare_window, sliding_infer, linear_interp_time, load_block_hwt, num_frames

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
        self.d1=DoubleConv3D(in_ch,b)
        self.p1=nn.MaxPool3d(2)
        self.d2=DoubleConv3D(b,b*2)
        self.p2=nn.MaxPool3d(2)
        self.d3=DoubleConv3D(b*2,b*4)
        self.p3=nn.MaxPool3d((1,2,2))
        self.mid=DoubleConv3D(b*4,b*8)
        self.u3=nn.ConvTranspose3d(b*8,b*4,(1,2,2),stride=(1,2,2))
        self.c3=DoubleConv3D(b*8,b*4)
        self.u2=nn.ConvTranspose3d(b*4,b*2,2,stride=2)
        self.c2=DoubleConv3D(b*4,b*2)
        self.u1=nn.ConvTranspose3d(b*2,b,2,stride=2)
        self.c1=DoubleConv3D(b*2,b)
        self.out=nn.Conv3d(b,1,1)
    def forward(self, x):
        # x: (B,2,H,W,T)
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


def _v2_weighted_l1(pred, y, miss, alpha, thr):
    w = 1.0 + alpha * (y.abs() / thr).clamp(max=1.0)
    return (w * (pred - y).abs() * miss).sum() / (w * miss).sum().clamp(min=1.0)


def _v2_blur_env(x, sigma):
    """envelope |x| blurred along the sample axis (H) only. x: (B,1,H,W,T).
    The bipolar pulses cancel when blurred directly, so the magnitude is blurred instead: this spreads each
    pulse position so that large arrival-time shifts still produce a gradient."""
    e = torch.sqrt(x * x + 1e-6)
    r = int(3 * sigma + 0.5)
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) / sigma) ** 2)
    k = (k / k.sum()).view(1, 1, -1, 1, 1)
    return F.conv3d(e, k, padding=(r, 0, 0))


def _v2_ms_loss(p, y, miss, scales):
    tot = 0.0
    for sg in scales:
        d = (_v2_blur_env(p, sg) - _v2_blur_env(y, sg)).abs() * miss
        tot = tot + d.sum() / miss.sum().clamp(min=1.0)
    return tot / len(scales)


def _v2_eval_window(pred, lin, y, m, alpha, thr):
    miss = (1 - m).expand_as(y)
    return dict(l1_model=float(((pred - y).abs() * miss).sum() / miss.sum()),
                l1_lin=float(((lin - y).abs() * miss).sum() / miss.sum()),
                wl1_model=float(_v2_weighted_l1(pred, y, miss, alpha, thr)),
                wl1_lin=float(_v2_weighted_l1(lin, y, miss, alpha, thr)))


def train(a):
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out); (out / "ckpt").mkdir(parents=True, exist_ok=True)
    json.dump(vars(a), open(out / "run_args.json", "w"), indent=2)

    starts = list(range(a.first_start, a.last_start + 1, a.win))
    tr_files = list_mat_files(a.train)
    nfr = {f: num_frames(f) for f in tr_files}
    pool = [(f, s) for f in tr_files for s in starts if s - 1 + a.win <= nfr[f]]
    va_files = list_mat_files(a.val)
    val_starts = [int(v) for v in a.val_starts.split(",")]
    print(f"[v2] train files={len(tr_files)} windows={len(pool)} | val files={len(va_files)} starts={val_starts}")

    # cache validation windows on CPU (prepared once)
    val_cache = []
    for f in va_files:
        for s in val_starts:
            print(f"[v2] preparing validation window {Path(f).name} @ {s}", flush=True)
            d = prepare_window(load_block_hwt(f, s, a.win), s, dev, th1=a.th1, ds=a.ds)
            val_cache.append({k: (v.cpu() if torch.is_tensor(v) else v) for k, v in d.items()})
    print(f"[v2] cached {len(val_cache)} validation windows")

    model = UNet3D(in_ch=2, base=a.base_ch).to(dev)
    if a.residual_linear:                                     # start exactly at linear interpolation
        nn.init.zeros_(model.out.weight); nn.init.zeros_(model.out.bias)
        print("[v2] residual_linear: input = linear interpolation, output = linear + correction (zero init)")
    best = float("inf")
    if a.resume:
        rk = torch.load(a.resume, map_location=dev)
        model.load_state_dict(rk["model"])
        same_dir = Path(a.resume).resolve().parent.parent == out.resolve() or Path(a.resume).resolve().parent == out.resolve()
        if same_dir:                                          # continuing the same run: keep its best score
            best = float(rk.get("ratio_wl1", best))
        print(f"[v2] resumed from {a.resume} (epoch {rk.get('epoch')}, ratio {float(rk.get('ratio_wl1', float('nan'))):.3f}), "
              f"continuing at epoch {a.start_epoch}, best reference {'kept' if same_dir else 'reset (new output folder)'}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=a.lr * 0.05)
    for _ in range(a.start_epoch - 1):
        sched.step()                                          # fast-forward the schedule
    amp = a.amp and dev.type == "cuda"
    ch, cw = [int(v) for v in a.crop_hw.split(",")]
    new_log = not (out / "log.csv").exists() or a.start_epoch == 1
    log = open(out / "log.csv", "w" if new_log else "a", newline=""); lw = csv.writer(log)
    if new_log:
        lw.writerow(["epoch", "train_loss", "val_l1_model", "val_l1_lin", "val_wl1_model", "val_wl1_lin", "ratio_wl1", "sec"])

    for ep in range(a.start_epoch, a.epochs + 1):
        t_ep = time.time(); model.train(); losses = []
        tl = dict(read=0.0, prep=0.0, train=0.0); n_bad = 0
        sel = random.sample(pool, min(a.windows_per_epoch, len(pool)))
        for (fp, s) in tqdm(sel, desc=f"[v2] epoch {ep}/{a.epochs}"):
            _t = time.time(); raw = load_block_hwt(fp, s, a.win); tl["read"] += time.time() - _t
            _t = time.time(); d = prepare_window(raw, s, dev, th1=a.th1, ds=a.ds); del raw
            if dev.type == "cuda": torch.cuda.synchronize()
            tl["prep"] += time.time() - _t; _t = time.time()
            x, m, y = d["x"], d["m"], d["y"]
            if a.residual_linear:
                x = linear_interp_time(x, m)                      # equals the measured data at measured frames
            H, W, T = x.shape[2], x.shape[3], x.shape[4]
            for _ in range(a.crops):
                t0 = random.randint(0, T - a.t_chunk)
                h0 = random.randint(0, H - ch) if ch < H else 0
                w0 = random.randint(0, W - cw) if cw < W else 0
                sl = (..., slice(h0, h0 + min(ch, H)), slice(w0, w0 + min(cw, W)), slice(t0, t0 + a.t_chunk))
                xi, yi, mi = x[sl], y[sl], m[..., t0:t0 + a.t_chunk]
                inp = torch.cat([xi, mi.expand(-1, 1, xi.shape[2], xi.shape[3], -1)], dim=1)
                with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp):
                    p = model(inp).float()
                if a.residual_linear:
                    p = p + xi
                p = mi * xi + (1 - mi) * p                                   # hard data consistency
                miss = (1 - mi).expand_as(yi)
                loss = _v2_weighted_l1(p, yi, miss, a.alpha, a.thr)
                if a.w_grad > 0:
                    loss = loss + a.w_grad * F.l1_loss(p[..., 1:] - p[..., :-1], yi[..., 1:] - yi[..., :-1])
                if a.ms_scales and a.w_ms > 0:
                    loss = loss + a.w_ms * _v2_ms_loss(p, yi, miss, a.ms_scales)
                if not torch.isfinite(loss):
                    n_bad += 1; opt.zero_grad(set_to_none=True); continue
                opt.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                losses.append(float(loss))
            if dev.type == "cuda": torch.cuda.synchronize()
            tl["train"] += time.time() - _t
            del d, x, m, y
        sched.step()
        print(f"[v2] time per window: read {tl['read']/len(sel):.1f}s | prep(SVD) {tl['prep']/len(sel):.1f}s | "
              f"train {tl['train']/len(sel):.1f}s ({a.crops} crops) | skipped non-finite steps: {n_bad}", flush=True)

        model.eval(); ev = []
        with torch.no_grad():
            for d in val_cache:
                x, m, y = d["x"].to(dev), d["m"].to(dev), d["y"].to(dev)
                lin = linear_interp_time(x, m)
                xin = lin if a.residual_linear else x
                pred = sliding_infer(model, xin, m, a.t_chunk, a.t_chunk // 2, amp=False,
                                     residual=a.residual_linear)   # fp32: bf16 3D conv at full size crashes on torch 1.13
                ev.append(_v2_eval_window(pred, lin, y, m, a.alpha, a.thr))
                del x, m, y, pred
        agg = {k: float(np.mean([e[k] for e in ev])) for k in ev[0]}
        ratio = agg["wl1_model"] / agg["wl1_lin"]
        sec = time.time() - t_ep
        print(f"[v2] ep {ep:03d} train {np.mean(losses):.4f} | val L1 model {agg['l1_model']:.4f} vs linear {agg['l1_lin']:.4f} "
              f"| wL1 ratio model/linear {ratio:.3f} | {sec/60:.1f} min")
        lw.writerow([ep, np.mean(losses), agg["l1_model"], agg["l1_lin"], agg["wl1_model"], agg["wl1_lin"], ratio, sec]); log.flush()
        ck = {"epoch": ep, "model": model.state_dict(), "cfg": vars(a), "val": agg, "ratio_wl1": ratio}
        torch.save(ck, out / "ckpt" / f"epoch_{ep:03d}.pt")
        if ratio < best:
            best = ratio; torch.save(ck, out / "best.pt")
    log.close()


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help="folder with the training .mat files")
    p.add_argument("--val", required=True, help="folder with the validation .mat files")
    p.add_argument("--out", required=True, help="output folder for checkpoints and logs")
    p.add_argument("--win", type=int, default=1000, help="window length = GT SVD chunk (10 s)")
    p.add_argument("--first_start", type=int, default=2001)
    p.add_argument("--last_start", type=int, default=28001)
    p.add_argument("--val_starts", default="3001,13001,23001")
    p.add_argument("--th1", type=int, default=11, help="SVD cutoff, th1 = 11 removes the 10 largest components")
    p.add_argument("--t_chunk", type=int, default=64)
    p.add_argument("--crop_hw", default="256,256", help="spatial crop (samples,channels); multiples of 8")
    p.add_argument("--crops", type=int, default=48, help="random crops per loaded window (reading dominates)")
    p.add_argument("--windows_per_epoch", type=int, default=20)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--base_ch", type=int, default=16)
    p.add_argument("--alpha", type=float, default=4.0, help="extra loss weight where |target| > thr (particle signal)")
    p.add_argument("--thr", type=float, default=3.0, help="in units of the input noise scale")
    p.add_argument("--w_grad", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true", help="bf16 autocast in training (unstable on torch 1.13)")
    p.add_argument("--resume", default="", help="checkpoint to continue from (model weights)")
    p.add_argument("--ms_scales", nargs="*", type=float, default=[],
                   help="multi-scale envelope loss: Gaussian sigmas along the sample axis (samples), e.g. 2 4 8")
    p.add_argument("--w_ms", type=float, default=1.0, help="weight of the multi-scale envelope loss")
    p.add_argument("--ds", type=int, default=5, help="downsampling factor: 5 = 100 -> 20 Hz, 2 = 50 Hz, 4 = 25 Hz, 10 = 10 Hz")
    p.add_argument("--residual_linear", action="store_true",
                   help="input = linear interpolation, output = linear interpolation + learned correction")
    p.add_argument("--start_epoch", type=int, default=1, help="first epoch number when resuming")
    train(p.parse_args())


if __name__ == "__main__":
    main()
