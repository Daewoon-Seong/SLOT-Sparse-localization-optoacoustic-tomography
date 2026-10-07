"""
DL 1: restoration of the inactive transducer channels (training).

Every second channel of the fully sampled sinograms is removed, the missing channels are prefilled by interpolation
along the channel azimuth, and an unrolled network (2D U-Net, K iterations with shared weights) restores them. After
each iteration the measured channels are restored (hard data consistency). The loss is computed on the sinograms.

usage:
  python train.py --train <dir> --val <dir> --geometry channel_geometry.npz --out <dir>
"""
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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


def load_frame_496x512(path: str, frame_1based: int, var_hint: str = None) -> np.ndarray:
    t = int(frame_1based) - 1
    # v7.3
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
    # v7
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


def get_num_frames(path: str) -> int:
    path=str(path)
    try:
        import h5py
        with h5py.File(path,"r") as f:
            dname=_h5_find_dataset(f); ds=f[dname]
            idxH,idxW,idxF=_reorder_axes_to_HWF(ds.shape)
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
    phi = np.asarray(phi_col).astype(np.float64)  # (W,)
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
        p_list=[]
        for k in range(self.K):
            p = self.step(k, p, base_feat)
            p_list.append(p)
        return p, p_list


def _gauss(ch=1,k=9,sigma=2.0,device="cpu"):
    ax=torch.arange(-k//2+1.,k//2+1.,device=device)
    xx,yy=torch.meshgrid(ax,ax,indexing='ij')
    g=torch.exp(-(xx**2+yy**2)/(2*sigma**2)); g=g/(g.sum()+1e-8)
    return g.view(1,1,k,k).repeat(ch,1,1,1)


def lcn(x,k=9,sigma=2.0):
    B,C,H,W=x.shape
    w=_gauss(ch=C,k=k,sigma=sigma,device=x.device)
    mu=F.conv2d(x,w,padding=k//2,groups=C)
    var=F.conv2d(x*x,w,padding=k//2,groups=C)-mu*mu
    std=torch.sqrt(var.clamp_min(1e-6))
    return (x-mu)/(std+1e-6)


def grad_xy(x):
    sx=torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]],dtype=torch.float32,device=x.device).view(1,1,3,3)
    sy=sx.transpose(2,3)
    gx=F.conv2d(x,sx,padding=1); gy=F.conv2d(x,sy,padding=1)
    return gx,gy


def grad_orient_loss(x,y,eps=1e-6):
    gx1,gy1=grad_xy(x); gx2,gy2=grad_xy(y)
    n1=torch.sqrt(gx1*gx1+gy1*gy1+eps); n2=torch.sqrt(gx2*gx2+gy2*gy2+eps)
    ux1,uy1=gx1/(n1+eps), gy1/(n1+eps)
    ux2,uy2=gx2/(n2+eps), gy2/(n2+eps)
    cos=(ux1*ux2 + uy1*uy2).clamp(-1,1)
    return (1.0 - cos).mean()


class SinoLoss(nn.Module):
    def __init__(self, w_mse=1.0, w_missing=6.0, w_grad=0.25, w_lcn=0.0, w_gori=0.0, use_ssim=False):
        super().__init__()
        self.w_mse=float(w_mse); self.w_missing=float(w_missing); self.w_grad=float(w_grad)
        self.w_lcn=float(w_lcn); self.w_gori=float(w_gori); self.use_ssim=bool(use_ssim)

    def _ssim(self,a,b):
        return 0.0*a.mean()                     # SSIM term not used (use_ssim=False in all experiments)

    @staticmethod
    def _grad_w(x): return x[:,:,:,1:] - x[:,:,:,:-1]

    def forward(self, pred, target, mask, x0, r):
        mse = F.mse_loss(pred, target)
        miss = (1.0 - mask)
        res_t = (target - x0).detach()
        mse_miss = F.mse_loss(miss * r, miss * res_t)
        gw = F.l1_loss(self._grad_w(pred), self._grad_w(target))
        lcn_mse = F.mse_loss(lcn(pred), lcn(target)) if self.w_lcn>0 else 0.0*mse
        gori = grad_orient_loss(pred,target) if self.w_gori>0 else 0.0*mse
        ssim = self._ssim(pred,target)
        total = ( self.w_mse*(0.5*mse + 0.5*ssim) + self.w_missing*mse_miss + self.w_grad*gw
                  + self.w_lcn*lcn_mse + self.w_gori*gori)
        comps = {"mse": float(mse.item()), "mse_miss": float(mse_miss.item()),
                 "grad_w": float(gw.item()),
                 "lcn_mse": float(lcn_mse.item()) if self.w_lcn>0 else None,
                 "gori": float(gori.item()) if self.w_gori>0 else None}
        return total, comps


def list_mat_files(root: str) -> List[str]:
    root=Path(root)
    return sorted([str(p) for p in root.rglob("*.mat") if p.is_file() and not p.name.startswith("._")])


def load_channel_geometry(path):
    """channel azimuth phi and normalized radius r_norm (512 channels each)"""
    g = np.load(path)
    return g["phi"].astype(np.float32), g["r_norm"].astype(np.float32)


class SinoOnlyDataset(Dataset):
    """frames of fully sampled sinograms -> (network input features, target)"""
    def __init__(self, mat_paths, geometry, sparse_ratio=2, offset=1, frame_start_1based=5001, num_frames=1000,
                 prefill_mode="angular", fourier_max=64, p_low=2.0, p_high=98.0):
        self.paths = [str(p) for p in mat_paths]
        self.prefill_mode = prefill_mode; self.fourier_max = int(fourier_max)
        self.p_low, self.p_high = p_low, p_high
        self.H, self.W = 496, 512
        self.sidx = sparse_indices(int(sparse_ratio), offset, self.W)
        self.phi, r_norm = load_channel_geometry(geometry)
        geom_cols = np.stack([np.cos(self.phi), np.sin(self.phi), r_norm], axis=0)
        self.geom_t = torch.from_numpy(np.transpose(np.repeat(geom_cols[None, :, :], self.H, axis=0), (1, 0, 2)).astype(np.float32))
        self.index = []
        for fi, p in enumerate(self.paths):
            T = get_num_frames(p)
            s0 = max(0, int(frame_start_1based) - 1); e0 = min(T, s0 + int(num_frames))
            self.index += [(fi, t) for t in range(s0, e0)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        fi, t0 = self.index[i]
        full_raw = load_frame_496x512(self.paths[fi], t0 + 1)
        meas_vals = full_raw[:, self.sidx].reshape(-1)
        lo = np.percentile(meas_vals, self.p_low); hi = np.percentile(meas_vals, self.p_high); hi = max(hi, lo + 1e-6)
        full = np.clip((full_raw - lo) / (hi - lo), 0, 1).astype(np.float32)
        msk = np.zeros_like(full, dtype=np.float32); msk[:, self.sidx] = 1.0
        if self.prefill_mode == "angular":
            xin = angular_prefill(full, msk, self.phi)
        elif self.prefill_mode == "linear":
            xin = linear_prefill(full, msk)
        else:
            xin = np.zeros_like(full, dtype=np.float32)
        xin[:, self.sidx] = full[:, self.sidx]
        H, W = self.H, self.W
        j = torch.linspace(-1, 1, W, dtype=torch.float32)[None, :].repeat(H, 1)
        tcoord = torch.linspace(-1, 1, H, dtype=torch.float32)[:, None].repeat(1, W)
        j_t = j.unsqueeze(0); t_t = tcoord.unsqueeze(0)
        jf = fourier_feats(j_t.unsqueeze(0), max_freq=self.fourier_max).squeeze(0)
        tf = fourier_feats(t_t.unsqueeze(0), max_freq=self.fourier_max).squeeze(0)
        x0 = torch.from_numpy(xin).unsqueeze(0); m0 = torch.from_numpy(msk).unsqueeze(0)
        y = torch.from_numpy(full).unsqueeze(0)
        return torch.cat([x0, m0, j_t, t_t, jf, tf, self.geom_t], dim=0), y


def train(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    tr_files, va_files = list_mat_files(a.train), list_mat_files(a.val)
    print(f"[files] train {len(tr_files)}, val {len(va_files)}")
    kw = dict(sparse_ratio=a.sparse_ratio, offset=a.offset, prefill_mode=a.prefill_mode, fourier_max=a.fourier_max,
              p_low=a.p_low, p_high=a.p_high)
    train_ds = SinoOnlyDataset(tr_files, a.geometry, frame_start_1based=a.train_start, num_frames=a.num_frames, **kw)
    val_ds = SinoOnlyDataset(va_files, a.geometry, frame_start_1based=a.val_start, num_frames=1, **kw)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_ds, batch_size=a.bs, shuffle=True, num_workers=a.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=a.workers, pin_memory=True)

    base_feat_ch = next(iter(train_loader))[0].shape[1]
    model = UnrolledReconstructor(base_feat_ch=base_feat_ch, K=a.K, base=a.base_ch, norm=a.norm,
                                  share_weights=(not a.no_share)).to(dev)
    sino_crit = SinoLoss(a.w_mse, a.w_missing, a.w_grad, a.w_lcn, a.w_gori, use_ssim=False)
    out = Path(a.out); (out / "ckpt").mkdir(parents=True, exist_ok=True)
    json.dump(vars(a), open(out / "run_args.json", "w"), indent=2)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)

    best_val, best_ep = float("inf"), -1
    for ep in range(1, a.epochs + 1):
        model.train(); tr_losses = []
        for base_feat, y in tqdm(train_loader, desc=f"Epoch {ep}/{a.epochs} [train]"):
            base_feat = base_feat.to(dev, non_blocking=True); y = y.to(dev, non_blocking=True)
            x0 = base_feat[:, 0:1]; mask = base_feat[:, 1:2]
            p_last, p_all = model(base_feat)
            loss_last, _ = sino_crit(p_last, y, mask, x0, p_last - x0)
            ds = 0.0 * loss_last
            if a.K > 1 and a.deep_sup_w > 0:                         # deep supervision on the intermediate steps
                for k in range(len(p_all) - 1):
                    dsk, _ = sino_crit(p_all[k], y, mask, x0, p_all[k] - x0); ds = ds + dsk
                ds = a.deep_sup_w * ds / max(1, len(p_all) - 1)
            loss = loss_last + ds
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tr_losses.append(float(loss.item()))
        mean_tr = float(np.mean(tr_losses)) if tr_losses else float("nan")

        model.eval(); va_losses = []
        with torch.no_grad():
            for base_feat, y in tqdm(val_loader, desc=f"Epoch {ep}/{a.epochs} [val]"):
                base_feat = base_feat.to(dev); y = y.to(dev)
                x0 = base_feat[:, 0:1]; mask = base_feat[:, 1:2]
                p_last, _ = model(base_feat)
                ls, _ = sino_crit(p_last, y, mask, x0, p_last - x0); va_losses.append(float(ls.item()))
        mean_va = float(np.mean(va_losses)) if va_losses else float("nan")
        print(f"[epoch {ep:03d}/{a.epochs}] loss train {mean_tr:.4f}, val {mean_va:.4f}")
        sched.step(mean_va)
        ck = {"epoch": ep, "model": model.state_dict(), "val_loss": mean_va, "train_loss": mean_tr}
        torch.save(ck, out / "ckpt" / f"epoch_{ep:03d}.pt")
        if mean_va < best_val:
            best_val, best_ep = mean_va, ep; torch.save(ck, out / "best.pt")
    print(f"best epoch {best_ep}, val loss {best_val:.4f}")


def main():
    import argparse
    p = argparse.ArgumentParser(description="DL 1 training: sparse to full channel restoration")
    p.add_argument("--train", required=True, help="folder with fully sampled training sinograms (.mat, dataset 'sigMat')")
    p.add_argument("--val", required=True)
    p.add_argument("--geometry", required=True, help="channel_geometry.npz (phi, r_norm)")
    p.add_argument("--out", required=True)
    p.add_argument("--sparse_ratio", type=int, default=2)
    p.add_argument("--offset", type=int, default=1, help="first measured channel (0-based); 1 keeps channels 2, 4, ..., 512")
    p.add_argument("--train_start", type=int, default=5001, help="first training frame (1-based)")
    p.add_argument("--val_start", type=int, default=15001)
    p.add_argument("--num_frames", type=int, default=1000, help="training frames per file")
    p.add_argument("--prefill_mode", default="angular", choices=["angular", "linear", "zero"])
    p.add_argument("--fourier_max", type=int, default=64)
    p.add_argument("--p_low", type=float, default=2.0)
    p.add_argument("--p_high", type=float, default=98.0)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--base_ch", type=int, default=64)
    p.add_argument("--norm", default="gn", choices=["none", "in", "gn"])
    p.add_argument("--no_share", action="store_true", help="do not share the weights across iterations")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--bs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--w_mse", type=float, default=1.0)
    p.add_argument("--w_missing", type=float, default=6.0)
    p.add_argument("--w_grad", type=float, default=0.25)
    p.add_argument("--w_lcn", type=float, default=0.0)
    p.add_argument("--w_gori", type=float, default=0.0)
    p.add_argument("--deep_sup_w", type=float, default=0.2)
    train(p.parse_args())


if __name__ == "__main__":
    main()
