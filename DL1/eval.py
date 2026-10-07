"""
DL 1: restoration of the inactive transducer channels (evaluation).

Every second channel of the fully sampled sinograms is removed, the missing channels are prefilled by interpolation along
the channel azimuth, and the trained unrolled network restores them. The prediction is compared with the measured
sinograms on the missing channels, and the angular interpolation (network input) is reported as the baseline.

usage:
  python eval.py --test ../sample_data/sample_sinogram.mat --ckpt ../weights/DL1_best.pt \
                 --geometry ../sample_data/channel_geometry.npz --out results_dl1 --save_pred_mat
"""
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def _h5_find_dataset(f):
    import h5py
    cands=[]
    def visit(name,obj):
        try:
            if isinstance(obj,h5py.Dataset) and obj.ndim==3 and np.issubdtype(obj.dtype,np.number):
                cands.append(name)
        except Exception: pass
    f.visititems(lambda n,o: visit(n,o))
    if not cands: raise RuntimeError("no 3D numeric dataset found in the HDF5 file")
    return cands[0]


def _reorder_axes_to_HWF(shape, H_target=496, W_target=512):
    axes=list(range(3))
    try:
        idxH=next(i for i,d in enumerate(shape) if d==H_target)
        idxW=next(i for i,d in enumerate(shape) if d==W_target)
    except StopIteration:
        dims=np.array(shape); idxH=int(np.argmin(np.abs(dims-H_target))); idxW=int(np.argmin(np.abs(dims-W_target)))
        if idxH==idxW: raise RuntimeError(f"cannot identify the H/W axes for shape {shape}")
    idxF=[i for i in axes if i not in (idxH,idxW)][0]
    return idxH,idxW,idxF


def get_num_frames(path: str) -> int:
    path=str(path)
    try:
        import h5py
        with h5py.File(path,"r") as f:
            dname=_h5_find_dataset(f); ds=f[dname]
            _,_,idxF=_reorder_axes_to_HWF(ds.shape)
            return int(ds.shape[idxF])
    except Exception:
        pass
    from scipy.io import loadmat
    md=loadmat(path)
    for k,v in md.items():
        if k.startswith("__"): continue
        if isinstance(v,np.ndarray) and v.ndim==3 and np.issubdtype(v.dtype,np.number):
            H,W,F=v.shape
            if H==496 and W==512: return int(F)
            if H==512 and W==496: return int(F)
            if F==496 and W==512: return int(H)
            if F==512 and H==496: return int(W)
            return int(v.shape[2])
    raise RuntimeError(f"{path}: no 3D array found")


def load_frame_496x512(path: str, frame_1based: int, var_hint: str = None) -> np.ndarray:
    t = int(frame_1based) - 1
    try:
        import h5py
        with h5py.File(path,"r") as f:
            dname = var_hint if (var_hint and var_hint in f) else _h5_find_dataset(f)
            ds=f[dname]; idxH,idxW,idxF=_reorder_axes_to_HWF(ds.shape)
            arr = np.array(ds[:,:,t]) if idxF==2 else (np.array(ds[:,t,:]) if idxF==1 else np.array(ds[t,:,:]))
            if arr.shape==(512,496): arr=arr.T
            if arr.shape!=(496,512): raise RuntimeError(f"{path}: frame shape {arr.shape} != (496,512)")
            return arr.astype(np.float32)
    except Exception:
        pass
    from scipy.io import loadmat
    md=loadmat(path)
    candidates=[var_hint] if (var_hint and var_hint in md) else [k for k in md.keys() if not k.startswith("__")]
    for k in candidates:
        v=md[k]
        if isinstance(v,np.ndarray) and v.ndim==3 and np.issubdtype(v.dtype,np.number):
            if v.shape==(496,512,v.shape[2]): frame=v[:,:,t]
            elif v.shape==(512,496,v.shape[2]): frame=v[:,:,t].T
            elif v.shape[0]==30000 and v.shape[1:]==(496,512): frame=v[t,:,:]
            elif v.shape[0]==30000 and v.shape[1:]==(512,496): frame=v[t,:,:].T
            else:
                frame=np.squeeze(v.take(indices=t,axis=2))
                if frame.shape==(512,496): frame=frame.T
            if frame.shape!=(496,512): continue
            return frame.astype(np.float32)
    raise RuntimeError(f"{path}: failed to load the frame")


def list_mat_files(root: str) -> List[str]:
    root=Path(root)
    return sorted([str(p) for p in root.rglob("*.mat") if p.is_file() and not p.name.startswith("._")])


def sparse_indices(ratio: int, offset: Optional[int], W: int = 512) -> np.ndarray:
    if offset is None: offset=ratio-1
    return np.arange(offset,W,ratio,dtype=np.int64)


def linear_prefill(full: np.ndarray, mask: np.ndarray) -> np.ndarray:
    H,W=full.shape
    out=np.zeros_like(full,dtype=np.float32)
    idx_all=np.arange(W); mea=mask[0]>0
    if mea.sum()>=2:
        for r in range(H): out[r,:]=np.interp(idx_all, idx_all[mea], full[r,mea])
    return out


def angular_prefill(full: np.ndarray, mask: np.ndarray, phi_col: np.ndarray) -> np.ndarray:
    """fills the missing channels by periodic interpolation along the channel azimuth"""
    H,W = full.shape
    out = np.zeros_like(full, dtype=np.float32)
    phi = np.asarray(phi_col).astype(np.float64)
    order = np.argsort(phi)
    phi_sorted = phi[order]
    full_sorted = full[:, order]
    m_sorted = mask[0, order] > 0
    mea_phi = phi_sorted[m_sorted]
    if mea_phi.size < 2:
        out[:, m_sorted] = full_sorted[:, m_sorted]
        inv = np.argsort(order)
        return out[:, inv].astype(np.float32)
    phi_ext = np.concatenate([mea_phi-2*np.pi, mea_phi, mea_phi+2*np.pi])
    x_all = phi_sorted
    for r in range(H):
        y_mea = full_sorted[r, m_sorted]
        y_ext = np.concatenate([y_mea, y_mea, y_mea])
        out_row = np.interp(x_all, phi_ext, y_ext)
        out_row[m_sorted] = full_sorted[r, m_sorted]
        out[r] = out_row
    inv = np.argsort(order)
    return out[:, inv].astype(np.float32)


def robust_normalize(frame: np.ndarray, mode="percentile", p_low=2.0, p_high=98.0, axis="global") -> np.ndarray:
    x=frame.astype(np.float32)
    if mode=="minmax":
        lo,hi=float(x.min()),float(x.max()); hi=max(hi,lo+1e-6)
        return (x-lo)/(hi-lo)
    if mode=="percentile":
        if axis=="global":
            lo=np.percentile(x,p_low); hi=np.percentile(x,p_high); hi=max(hi,lo+1e-6)
            return np.clip((x-lo)/(hi-lo),0,1)
        if axis=="row":
            out=np.empty_like(x)
            for r in range(x.shape[0]):
                lo=np.percentile(x[r],p_low); hi=np.percentile(x[r],p_high); hi=max(hi,lo+1e-6)
                out[r]=np.clip((x[r]-lo)/(hi-lo),0,1)
            return out
        if axis=="col":
            out=np.empty_like(x)
            for c in range(x.shape[1]):
                lo=np.percentile(x[:,c],p_low); hi=np.percentile(x[:,c],p_high); hi=max(hi,lo+1e-6)
                out[:,c]=np.clip((x[:,c]-lo)/(hi-lo),0,1)
            return out
    if mode=="zscore_col":
        mu=x.mean(axis=0,keepdims=True); sg=x.std(axis=0,keepdims=True)+1e-6
        z=(x-mu)/sg; return np.clip(0.5+0.1667*z,0,1)
    if mode=="zscore_row":
        mu=x.mean(axis=1,keepdims=True); sg=x.std(axis=1,keepdims=True)+1e-6
        z=(x-mu)/sg; return np.clip(0.5+0.1667*z,0,1)
    lo,hi=float(x.min()),float(x.max()); hi=max(hi,lo+1e-6)
    return (x-lo)/(hi-lo)


def fourier_feats(x: torch.Tensor, max_freq: int = 64) -> torch.Tensor:
    freqs=[]; f=1
    while f<=max_freq: freqs.append(f); f*=2
    outs=[]
    for f in freqs:
        outs += [torch.sin(np.pi*f*x), torch.cos(np.pi*f*x)]
    return torch.cat(outs,dim=1) if outs else torch.zeros_like(x)


def make_norm2d(norm: str, ch: int):
    norm=(norm or "gn").lower()
    if norm=="none": return nn.Identity()
    if norm=="in":   return nn.InstanceNorm2d(ch,affine=True,track_running_stats=False)
    if norm=="gn":   return nn.GroupNorm(min(8,ch), ch)
    return nn.BatchNorm2d(ch)


class DoubleConv(nn.Module):
    def __init__(self,in_ch,out_ch,norm="gn"):
        super().__init__()
        self.block=nn.Sequential(
            nn.Conv2d(in_ch,out_ch,3,padding=1),
            make_norm2d(norm,out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch,out_ch,3,padding=1),
            make_norm2d(norm,out_ch), nn.ReLU(True),
        )
    def forward(self,x): return self.block(x)


class UNet2D(nn.Module):
    def __init__(self,in_ch, base=64, norm="gn"):
        super().__init__()
        self.inc   = DoubleConv(in_ch, base, norm)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base, base*2, norm))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base*2, base*4, norm))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base*4, base*8, norm))
        self.bot   = DoubleConv(base*8, base*16, norm)
        self.up3   = nn.ConvTranspose2d(base*16, base*4, 2,2); self.conv3=DoubleConv(base*8, base*4, norm)
        self.up2   = nn.ConvTranspose2d(base*4,  base*2, 2,2); self.conv2=DoubleConv(base*4, base*2, norm)
        self.up1   = nn.ConvTranspose2d(base*2,  base,   2,2); self.conv1=DoubleConv(base*2, base,   norm)
        self.out   = nn.Conv2d(base, 1, 1)
    def forward(self,x):
        B,C,H,W=x.shape
        padH=(16-H%16)%16; padW=(16-W%16)%16
        x = F.pad(x,(0,padW,0,padH),mode="replicate")
        x1=self.inc(x); x2=self.down1(x1); x3=self.down2(x2); x4=self.down3(x3); xb=self.bot(x4)
        x=self.up3(xb);  x=F.interpolate(x, size=x3.shape[-2:], mode="bilinear", align_corners=False); x=self.conv3(torch.cat([x,x3],1))
        x=self.up2(x);   x=F.interpolate(x, size=x2.shape[-2:], mode="bilinear", align_corners=False); x=self.conv2(torch.cat([x,x2],1))
        x=self.up1(x);   x=F.interpolate(x, size=x1.shape[-2:], mode="bilinear", align_corners=False); x=self.conv1(torch.cat([x,x1],1))
        x=self.out(x);   x=x[..., :H, :W]
        return x


class UnrolledReconstructor(nn.Module):
    def __init__(self, base_feat_ch: int, K: int = 4, base=64, norm="gn", share_weights=True):
        super().__init__()
        self.K=int(K); self.share=bool(share_weights)
        prox_in = 1 + base_feat_ch
        if share_weights:
            self.prox = UNet2D(prox_in, base=base, norm=norm)
        else:
            self.prox_list = nn.ModuleList([UNet2D(prox_in, base=base, norm=norm) for _ in range(self.K)])
    def step(self, k, p, base_feat):
        x0 = base_feat[:,0:1]; mask = base_feat[:,1:2]
        prox_in = torch.cat([p, base_feat], dim=1)
        r = (self.prox if self.share else self.prox_list[k])(prox_in)
        p_tilde = p + (1.0 - mask) * r
        p_next  = p_tilde*(1.0 - mask) + x0*mask
        return p_next
    def forward(self, base_feat):
        p = base_feat[:,0:1].clone()
        for k in range(self.K):
            p = self.step(k, p, base_feat)
        return p

def load_channel_geometry(path):
    """channel azimuth phi and normalized radius r_norm (512 channels each)"""
    g = np.load(path)
    return g["phi"].astype(np.float32), g["r_norm"].astype(np.float32)


def nrmse(a, b, cols):
    """normalized root-mean-square error between a and the reference b on the given channels"""
    d = a[:, cols] - b[:, cols]
    return float(np.sqrt((d ** 2).mean()) / (np.sqrt((b[:, cols] ** 2).mean()) + 1e-12))


def main():
    import argparse
    p = argparse.ArgumentParser(description="DL 1 evaluation: sparse to full channel restoration")
    p.add_argument("--test", required=True, help=".mat file or folder of fully sampled sinograms (dataset 'sigMat')")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--geometry", required=True, help="channel_geometry.npz (phi, r_norm)")
    p.add_argument("--out", required=True)
    p.add_argument("--max_frames", type=int, default=0, help="0 = all frames")
    p.add_argument("--sparse_source", default="ratio", choices=["ratio", "random"])
    p.add_argument("--sparse_ratio", type=int, default=2)
    p.add_argument("--offset", type=int, default=1, help="first measured channel (0-based); 1 keeps channels 2, 4, ..., 512")
    p.add_argument("--rand_n", type=int, default=256, help="random: number of measured channels")
    p.add_argument("--rand_seed", type=int, default=0)
    p.add_argument("--prefill_mode", default="angular", choices=["angular", "linear", "zero"])
    p.add_argument("--fourier_max", type=int, default=64)
    p.add_argument("--p_low", type=float, default=2.0)
    p.add_argument("--p_high", type=float, default=98.0)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--base_ch", type=int, default=64)
    p.add_argument("--norm", default="gn", choices=["none", "in", "gn"])
    p.add_argument("--no_share", action="store_true")
    p.add_argument("--save_pred_mat", action="store_true", help="save the restored sinograms (uint16, raw units)")
    a = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path(a.out); out_root.mkdir(parents=True, exist_ok=True)
    phi, r_norm = load_channel_geometry(a.geometry)
    H, W = 496, 512
    if a.sparse_source == "random":
        sidx = np.sort(np.random.default_rng(a.rand_seed).choice(W, a.rand_n, replace=False)).astype(np.int64)
    else:
        sidx = sparse_indices(a.sparse_ratio, a.offset, W)
    miss = np.setdiff1d(np.arange(W), sidx)

    def _n_fourier(maxf):
        c = 0; f = 1
        while f <= maxf:
            c += 2; f *= 2
        return c
    base_feat_ch = 1 + 1 + 1 + 1 + _n_fourier(a.fourier_max) + _n_fourier(a.fourier_max) + 3
    model = UnrolledReconstructor(base_feat_ch=base_feat_ch, K=a.K, base=a.base_ch, norm=a.norm,
                                  share_weights=(not a.no_share)).to(dev)
    ck = torch.load(a.ckpt, map_location=dev)
    model.load_state_dict(ck["model"] if "model" in ck else ck); model.eval()

    geom_cols = np.stack([np.cos(phi), np.sin(phi), r_norm], axis=0)
    geom_t = torch.from_numpy(np.transpose(np.repeat(geom_cols[None, :, :], H, axis=0), (1, 0, 2)).astype(np.float32))
    j = torch.linspace(-1, 1, W, dtype=torch.float32)[None, :].repeat(H, 1)
    tcoord = torch.linspace(-1, 1, H, dtype=torch.float32)[:, None].repeat(1, W)
    j_t = j.unsqueeze(0); t_t = tcoord.unsqueeze(0)
    jf = fourier_feats(j_t.unsqueeze(0), max_freq=a.fourier_max).squeeze(0)
    tf = fourier_feats(t_t.unsqueeze(0), max_freq=a.fourier_max).squeeze(0)

    files = [a.test] if str(a.test).lower().endswith(".mat") else list_mat_files(a.test)
    for fp in files:
        stem = Path(fp).stem; out = out_root / stem; out.mkdir(parents=True, exist_ok=True)
        T = get_num_frames(fp); T_eff = T if a.max_frames <= 0 else min(T, a.max_frames)
        d_pred = hf = None
        if a.save_pred_mat:
            import h5py
            hf = h5py.File(str(out / "pred_sino.mat"), "w")
            d_pred = hf.create_dataset("sino_pred", shape=(H, W, T_eff), dtype="uint16", chunks=(H, W, 1),
                                       compression="gzip", compression_opts=6, shuffle=True)
        rows = []
        for t in tqdm(range(1, T_eff + 1), desc=stem):
            full_raw = load_frame_496x512(fp, t)
            meas_vals = full_raw[:, sidx].reshape(-1)
            lo = np.percentile(meas_vals, a.p_low); hi = np.percentile(meas_vals, a.p_high); hi = max(hi, lo + 1e-6)
            full = np.clip((full_raw - lo) / (hi - lo), 0, 1).astype(np.float32)
            msk = np.zeros_like(full, dtype=np.float32); msk[:, sidx] = 1.0
            if a.prefill_mode == "angular":
                xin = angular_prefill(full, msk, phi)
            elif a.prefill_mode == "linear":
                xin = linear_prefill(full, msk)
            else:
                xin = np.zeros_like(full, dtype=np.float32)
            xin[:, sidx] = full[:, sidx]
            x0 = torch.from_numpy(xin).unsqueeze(0); m0 = torch.from_numpy(msk).unsqueeze(0)
            base_feat = torch.cat([x0, m0, j_t, t_t, jf, tf, geom_t], dim=0).unsqueeze(0).to(dev)
            with torch.no_grad():
                pred = model(base_feat)[0, 0].cpu().numpy().astype(np.float32)
            rows.append((t, nrmse(xin, full, miss), nrmse(pred, full, miss)))
            if d_pred is not None:
                d_pred[:, :, t - 1] = np.clip(np.round(pred * (hi - lo) + lo), 0, 65535).astype(np.uint16)
        if hf is not None:
            hf.close()
        r = np.array(rows)
        np.savetxt(out / "metrics_per_frame.csv", r, delimiter=",", header="frame,nrmse_interpolation,nrmse_dl1",
                   comments="", fmt=["%d", "%.6f", "%.6f"])
        summ = dict(file=Path(fp).name, frames=int(T_eff), measured_channels=int(len(sidx)),
                    nrmse_interpolation=float(r[:, 1].mean()), nrmse_dl1=float(r[:, 2].mean()))
        json.dump(summ, open(out / "metrics.json", "w"), indent=2)
        print(f"{stem}: NRMSE on missing channels, interpolation {summ['nrmse_interpolation']:.4f}, "
              f"DL 1 {summ['nrmse_dl1']:.4f}")


if __name__ == "__main__":
    main()
