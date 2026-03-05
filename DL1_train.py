# -*- coding: utf-8 -*-
"""
DL1 (SLOT): Sparse transducer-channel sinogram restoration.

This script trains an unrolled reconstruction network in the sinogram domain to
restore a full 512-channel optoacoustic sinogram from sparse measurements
(e.g., 256 channels).

Implementation notes (kept from the original research code):
- Optional prefill for missing channels: "linear", "angular" (circular interpolation in φ),
  "zero", or "blend".
- Hard data-consistency (DC) at every unrolled step: measured channels are enforced.
- Geometry feature channels and optional Fourier features.
- Optional back-projection supervision using per-file BP metadata.

This repository does NOT include the experimental data (.mat). See README for the
expected folder layout.
"""

import os, json, warnings
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

# ----------------- SSIM (optional) -----------------
try:
    from pytorch_msssim import ms_ssim
    HAS_MSSSIM = True
except Exception:
    ms_ssim = None
    HAS_MSSSIM = False

# ----------------- Small utils -----------------
def _as_pos_tensor(x, device: torch.device) -> torch.Tensor:
    """
    Convert pos_sensor_xyz to a (Nc, 3) float32 tensor on the given device.
    - Unwrap list/tuple of length 1.
    - Accepts numpy arrays or torch tensors.
    - Squeezes (1, Nc, 3) -> (Nc, 3) if needed.
    - Validates the final shape.
    """
    if isinstance(x, (list, tuple)) and len(x) == 1:
        x = x[0]
    if isinstance(x, torch.Tensor):
        t = x.to(device)
    else:
        arr = np.asarray(x)
        t = torch.from_numpy(arr).to(device)
    t = t.float()
    if t.ndim == 3 and t.shape[0] == 1:
        t = t.squeeze(0)
    if t.ndim != 2 or t.shape[1] != 3:
        raise RuntimeError(f"pos_xyz tensor has wrong shape {tuple(t.shape)}; expected (Nc, 3)")
    return t

def _align_sig_pos_for_bp(sino_2d: torch.Tensor,
                          base_feat: torch.Tensor,
                          pos_xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    sino_2d: (H, W), base_feat: (C0, H, W), pos_xyz: (Nc, 3)
    Returns: (sinogram_for_bp, corresponding_pos_xyz).
    Rules:
      - If pos_xyz contains the full 512 sensors (Nc == W): use all columns for BP.
      - If pos_xyz contains only the measured sensors (Nc == Nm): select the measured sinogram columns accordingly.
    """
    device = sino_2d.device
    H, W = sino_2d.shape
    Nc_pos = pos_xyz.shape[0]

    mask = (base_feat[1:2] > 0).to(device)   # (1,H,W)
    meas_cols = torch.where(mask[0, 0] > 0)[0]  # (Nm,)
    # Keep indices on the same device as pos_xyz.
    meas_cols = meas_cols.to(device=pos_xyz.device, dtype=torch.long)
    Nm = int(meas_cols.numel())

    if Nm == 0:
        # Fallback: still use the full set.
        return sino_2d, pos_xyz

    if Nc_pos == W:
        # NOTE
        # pos_xyz contains all columns -> run BP on the full sinogram.
        return sino_2d, pos_xyz

    if Nc_pos == Nm:
        # pos_xyz contains only measured sensors -> align sinogram to those columns.
        sino_used = sino_2d.index_select(dim=1, index=meas_cols.to(device))
        pos_used  = pos_xyz  # Assumes pos_xyz is already in measured-order.
        return sino_used, pos_used

    raise RuntimeError(f"[BP align] pos_xyz Nc({Nc_pos}) != W({W}) and != measured Nm({Nm}). Alignment rule missing (pos_xyz does not match either full W or measured Nm).")

# ========================= I/O (.mat) =========================
def _h5_find_dataset(f):
    import h5py
    cands=[]
    def visit(name,obj):
        try:
            if isinstance(obj,h5py.Dataset) and obj.ndim==3 and np.issubdtype(obj.dtype,np.number):
                cands.append(name)
        except Exception: pass
    f.visititems(lambda n,o: visit(n,o))
    if not cands: raise RuntimeError("Could not find a 3D numeric dataset in the HDF5 file.")
    return cands[0]

def _reorder_axes_to_HWF(shape, H_target=496, W_target=512):
    axes=list(range(3))
    try:
        idxH=next(i for i,d in enumerate(shape) if d==H_target)
        idxW=next(i for i,d in enumerate(shape) if d==W_target)
    except StopIteration:
        dims=np.array(shape); idxH=int(np.argmin(np.abs(dims-H_target))); idxW=int(np.argmin(np.abs(dims-W_target)))
        if idxH==idxW: raise RuntimeError(f"Cannot determine H/W axes from shape {shape}")
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
    raise RuntimeError(f"{path}: failed to load frame")

def load_pos_sensor_xyz(mat_path: str, var_name: Optional[str] = None) -> np.ndarray:
    from scipy.io import loadmat
    md=loadmat(mat_path)
    if var_name and var_name in md: arr=md[var_name]
    else:
        arr=None
        for k,v in md.items():
            if k.startswith("__"): continue
            if isinstance(v,np.ndarray) and v.ndim==2 and v.shape[1]==3 and np.issubdtype(v.dtype,np.number):
                arr=v; break
    if arr is None: raise RuntimeError("Could not find a (Nc, 3) numeric array in the pos_sensor_xyz .mat file.")
    return np.asarray(arr,dtype=np.float32)

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
    raise RuntimeError(f"{path}: Could not find a 3D array in the .mat file.")

# ====================== Prefill & Normalize ====================
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

def angular_prefill(full: np.ndarray, mask: np.ndarray, pos_xyz_col: np.ndarray) -> np.ndarray:
    """Fill missing columns by circular linear interpolation along azimuth φ. pos_xyz_col: (W, 3) aligned with sinogram columns."""
    H,W = full.shape
    out = np.zeros_like(full, dtype=np.float32)
    phi = np.arctan2(pos_xyz_col[:,1], pos_xyz_col[:,0]).astype(np.float64)  # (W,)
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

def linear_alpha(ep:int,start:int,length:int)->float:
    if length<=0: return 1.0
    if ep<start: return 1.0
    if ep>=start+length: return 0.0
    return float(1.0 - (ep-start)/max(1,length))

# ===================== Fourier features ======================
def fourier_feats(x: torch.Tensor, max_freq: int = 64) -> torch.Tensor:
    freqs=[]; f=1
    while f<=max_freq: freqs.append(f); f*=2
    outs=[]
    for f in freqs:
        outs += [torch.sin(np.pi*f*x), torch.cos(np.pi*f*x)]
    return torch.cat(outs,dim=1) if outs else torch.zeros_like(x)

# =============== Per-file meta (filename → pos_xyz & BP) ===============
def _normalized_name(path_like: str) -> str:
    name = Path(path_like).stem.lower()
    name = " ".join(name.split())
    return name

def pick_bp_meta_for_path(mat_path: str,
                          pos_xyz_dir: str,
                          pos_xyz_var: Optional[str]) -> Dict[str, Any]:
    """
    Select the pos_xyz file and BP parameters based on the filename.
    - ultracup → pos_sensor_xyz_ultracup.mat, fs=40e6, t0=832, c0=1505, dims=(7e-3,7e-3,5e-3), res=(175,175,125), filt=(0.2e6,8e6)
    - holycup/holocup → pos_sensor_xyz_holycup.mat,
        Default: fs=24e6, t0=501, c0=1500, dims=(10e-3,10e-3,105e-3), res=(128,128,128), filt=(0.2e6,8e6)
        If '40 msps' is present: fs=40e6, t0=832
        If 'sos' is present: c0=1480 (override)
    """
    name = _normalized_name(mat_path)
    d = Path(pos_xyz_dir)
    has = lambda s: s in name

    meta: Dict[str, Any] = {
        "c0": 1505.0,
        "bp_dims": (7e-3, 7e-3, 5e-3),
        "bp_res":  (175, 175, 125),
        "bp_filter": (0.2e6, 8e6),
        "chan_block": 64,
        "z_slab": 10,
    }

    if "ultracup" in name:
        pfile = d / "pos_sensor_xyz_ultracup.mat"
        meta.update({"bp_fs": 40e6, "bp_t0": 832, "tag": "ultracup"})
    elif ("holycup" in name) or ("holocup" in name):
        pfile = d / "pos_sensor_xyz_holycup.mat"
        meta.update({
            "c0": 1500.0,
            "bp_dims": (10e-3, 10e-3, 10e-3),
            "bp_res":  (128, 128, 128),
            "bp_filter": (0.1e6, 6e6),
        })
        if has("40 msps"):
            meta.update({"bp_fs": 40e6, "bp_t0": 832, "tag": "holycup_40msps"})
        else:
            meta.update({"bp_fs": 24e6, "bp_t0": 501, "tag": "holycup_24msps"})
        if has("sos"):
            meta["c0"] = 1480.0
            meta["tag"] += "_sos"
    else:
        warnings.warn(f"[BP meta] {Path(mat_path).name}: tag not found -> fallback to ultracup parameters")
        pfile = d / "pos_sensor_xyz_ultracup.mat"
        meta.update({"bp_fs": 40e6, "bp_t0": 832, "tag": "fallback_ultracup"})

    if not pfile.is_file():
        raise FileNotFoundError(f"pos_xyz file not found: {pfile}")

    meta["pos_xyz"] = load_pos_sensor_xyz(str(pfile), pos_xyz_var).astype(np.float32)  # (Nc,3)
    return meta

# ===================== Dataset (coords + geom per file) ==============
class SinoOnlyDataset(Dataset):
    def __init__(self, mat_paths: List[str],
                 sparse_ratio=2, offset=2,
                 frame_start_1based=2001, num_frames=50,
                 prefill_mode="linear", fourier_max=64,
                 norm_mode="percentile", p_low=2.0, p_high=98.0, norm_axis="global",
                 norm_from_measured=True,
                 pos_xyz_dir: str = "", pos_xyz_var: Optional[str] = None,
                 sidx_override: Optional[np.ndarray] = None):
        self.paths=[str(p) for p in mat_paths]
        self.ratio=int(sparse_ratio); self.offset=offset
        self.frame_start=int(frame_start_1based); self.num_frames=int(num_frames)
        self.prefill_mode=prefill_mode; self.alpha=1.0
        self.fourier_max=int(fourier_max)
        self.norm_mode=norm_mode; self.p_low=p_low; self.p_high=p_high; self.norm_axis=norm_axis
        self.norm_from_measured=bool(norm_from_measured)
        self.H,self.W=496,512

        # Determine measured columns (mask -> column indices)
        if sidx_override is not None:
            sidx = np.asarray(sidx_override, dtype=np.int64)
            assert sidx.ndim==1, "sidx_override must be 1D"
            self.sidx = sidx
        else:
            self.sidx = sparse_indices(self.ratio, self.offset, self.W)

        self.pos_xyz_dir = pos_xyz_dir
        self.pos_xyz_var = pos_xyz_var

        self.file_meta: Dict[int, Dict[str, Any]] = {}
        self.index=[]
        for fi,p in enumerate(self.paths):
            meta = pick_bp_meta_for_path(p, self.pos_xyz_dir, self.pos_xyz_var)
            self.file_meta[fi] = meta
            T=get_num_frames(p)
            s0=max(0,self.frame_start-1); e0=min(T,s0+self.num_frames)
            for t in range(s0,e0):
                self.index.append((fi,t))

    def __len__(self): return len(self.index)

    def __getitem__(self,i):
        fi,t0=self.index[i]; t=t0+1
        mat_path = self.paths[fi]
        meta     = self.file_meta[fi]
        pos_xyz  = meta["pos_xyz"]
        if pos_xyz.shape[0] != self.W:
            raise RuntimeError(f"[Dataset] pos_xyz Nc({pos_xyz.shape[0]}) != W({self.W}). Alignment logic required.")

        full_raw=load_frame_496x512(mat_path, t)

        if self.norm_from_measured:
            meas_vals = full_raw[:, self.sidx].reshape(-1)
            lo = np.percentile(meas_vals, self.p_low); hi = np.percentile(meas_vals, self.p_high); hi=max(hi,lo+1e-6)
            full = np.clip((full_raw - lo)/(hi-lo), 0, 1).astype(np.float32)
        else:
            full = robust_normalize(full_raw, mode=self.norm_mode, p_low=self.p_low, p_high=self.p_high, axis=self.norm_axis)

        msk=np.zeros_like(full,dtype=
        np.float32); msk[:,self.sidx]=1.0

        if self.prefill_mode=="angular":
            xin = angular_prefill(full, msk, pos_xyz)
        elif self.prefill_mode=="linear":
            xin = linear_prefill(full, msk)
        elif self.prefill_mode=="zero":
            xin = np.zeros_like(full, dtype=np.float32)
        elif self.prefill_mode=="blend":
            xin = self.alpha*linear_prefill(full, msk) + (1.0-self.alpha)*np.zeros_like(full, np.float32)
        else:
            raise ValueError(f"Unknown prefill_mode: {self.prefill_mode}")
        xin[:, self.sidx] = full[:, self.sidx]

        H,W=self.H,self.W
        j=torch.linspace(-1,1,W,dtype=torch.float32)[None,:].repeat(H,1)
        tcoord=torch.linspace(-1,1,H,dtype=torch.float32)[:,None].repeat(1,W)
        j_t=j.unsqueeze(0); t_t=tcoord.unsqueeze(0)
        jf=fourier_feats(j_t.unsqueeze(0),max_freq=self.fourier_max).squeeze(0)
        tf=fourier_feats(t_t.unsqueeze(0),max_freq=self.fourier_max).squeeze(0)

        px, py, pz = pos_xyz[:,0], pos_xyz[:,1], pos_xyz[:,2]
        phi = np.arctan2(py, px).astype(np.float32)
        r = np.sqrt(px**2 + py**2 + pz**2).astype(np.float32)
        r_norm = (r / (np.max(r)+1e-6) * 2.0 - 1.0).astype(np.float32)
        geom_cols = np.stack([np.cos(phi), np.sin(phi), r_norm], axis=0)  # (3,W)
        geom_np = np.repeat(geom_cols[None, :, :], H, axis=0)             # (H,3,W)
        geom_np = np.transpose(geom_np, (1,0,2)).astype(np.float32)       # (3,H,W)
        geom_t = torch.from_numpy(geom_np)

        x0=torch.from_numpy(xin).unsqueeze(0)
        m0=torch.from_numpy(msk).unsqueeze(0)
        y =torch.from_numpy(full).unsqueeze(0)

        base_feat = torch.cat([x0, m0, j_t, t_t, jf, tf, geom_t], dim=0)
        return base_feat, y, meta

# ===================== ProxNet (UNet) =======================
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

# ===================== Unrolled model ========================
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

# ================== BP (SoftMIP) =================
@torch.no_grad()
def _fft_bandpass(sig_mat: torch.Tensor, fs: float, f_lo: float, f_hi: float) -> torch.Tensor:
    device = sig_mat.device
    Nt, Nc = sig_mat.shape
    S = torch.fft.fft(sig_mat, dim=0)
    f = torch.arange(Nt, device=device, dtype=torch.float32) * (fs / Nt)
    H = ((f >= float(f_lo)) & (f <= float(f_hi))).to(sig_mat.dtype)
    H = H + torch.flip(H, dims=[0])
    return torch.fft.ifft(S * H[:, None], dim=0).real

def backproject_gpu_matchCPU(sig_mat: torch.Tensor,
                             recon_dims: Tuple[float,float,float],
                             resolution: Tuple[int,int,int],
                             pos_sensor_xyz: torch.Tensor,
                             c0: float, fs: float, t0_offset: int,
                             f_filter: Tuple[float,float],
                             chan_block: int = 32, z_slab: int = 10) -> torch.Tensor:
    device=sig_mat.device
    Nt,Nc=sig_mat.shape; Nx,Ny,Nz=map(int, resolution); Lx,Ly,Lz=map(float, recon_dims)
    t=(t0_offset+torch.arange(1,Nt+1,device=device,dtype=torch.float32))/fs
    with torch.no_grad():
        sig_f=_fft_bandpass(sig_mat.detach(),fs,f_filter[0],f_filter[1])
    sig=sig_f + (sig_mat - sig_mat.detach())

    x=torch.linspace(-Lx/2,Lx/2,steps=Nx,device=device)
    y=torch.linspace(-Ly/2,Ly/2,steps=Ny,device=device)
    z=torch.linspace(-Lz/2,Lz/2,steps=Nz,device=device)
    fsamp=fs; t0s=t[0]*fsamp

    tx=pos_sensor_xyz[:,0]; ty=pos_sensor_xyz[:,1]; tz=pos_sensor_xyz[:,2]
    Recon=torch.zeros((Ny,Nx,Nz),dtype=torch.float32,device=device)

    for z0 in range(0,Nz,z_slab):
        z1=min(Nz,z0+z_slab); zc=z[z0:z1]
        Y,X,Z=torch.meshgrid(y,x,zc,indexing='ij')
        Nvox=Y.numel()
        xx=X.reshape(-1); yy=Y.reshape(-1); zz=Z.reshape(-1)
        bp_slab=torch.zeros((Nvox,),dtype=torch.float32,device=device)

        for k1 in range(0,Nc,chan_block):
            k2=min(Nc,k1+chan_block)
            TPT=sig[:,k1:k2]
            B=TPT.shape[1]
            dPdt=(TPT[1:,:]-TPT[:-1,:])*fsamp
            pad0=torch.zeros((1,B),dtype=dPdt.dtype,device=device)
            TdPdt=torch.cat([dPdt,pad0],dim=0)*t[:,None]
            TPT_vec=TPT.transpose(0,1).contiguous().view(-1)
            TdPdt_vec=TdPdt.transpose(0,1).contiguous().view(-1)

            txb=tx[k1:k2][None,:]; tyb=ty[k1:k2][None,:]; tzb=tz[k1:k2][None,:]
            dx=xx[:,None]-txb; dy=yy[:,None]-tyb; dz=zz[:,None]-tzb
            d=torch.sqrt(dx*dx+dy*dy+dz*dz)
            t_float=d*(fsamp/c0)-t0s
            tbp=torch.round(t_float).long().clamp(0,Nt-1)
            B_eff = tbp.shape[-1]
            tbp2  = tbp.reshape(-1, B_eff)
            base  = (torch.arange(B_eff, device=device, dtype=torch.long) * Nt).view(1, B_eff)
            idx   = (tbp2 + base).reshape(-1)

            vals_T  = TPT_vec[idx]
            vals_dT = TdPdt_vec[idx]
            bp_slab += (vals_T - vals_dT).view(-1, B_eff).sum(dim=1)

        Recon[:,:,z0:z1]=bp_slab.reshape(Ny,Nx,-1)

    return -Recon

def softmip(vol: torch.Tensor, tau: float = 0.1, dim: int = 2) -> torch.Tensor:
    w=torch.softmax(vol/tau,dim=dim)
    return (w*vol).sum(dim=dim)

# ----------------- Index list loader (for custom measured cols) -----------------
def load_keep_indices(mat_path: str, var_name: Optional[str], W: int = 512,
                      index_1based: Optional[bool] = None) -> np.ndarray:
    """
    Load measured detector indices from a .mat file and return them as 0-based indices.
    - Auto-detects 1-based vs 0-based (can be overridden).
    - Accepts any shape: flattens -> unique -> sort.
    - Raises if out of range.
    """
    from scipy.io import loadmat
    md = loadmat(mat_path)
    cand = None
    if var_name and var_name in md:
        cand = md[var_name]
    else:
        # Pick one numeric array variable
        for k, v in md.items():
            if k.startswith("__"): continue
            if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
                cand = v
                break
    if cand is None:
        raise RuntimeError(f"[indices] {mat_path}: Could not find a numeric array. Check --index_var.")

    idx = np.asarray(cand).astype(np.int64).ravel()

    if idx.size == 0:
        raise RuntimeError(f"[indices] {mat_path}: Index array is empty.")

    # Detect 1-based vs 0-based indexing
    if index_1based is None:
        if np.any(idx == 0):
            index_1based = False
        else:
            # If all indices are within 1..W, assume 1-based indexing.
            index_1based = (idx.min() >= 1 and idx.max() <= W)
    if index_1based:
        idx = idx - 1

    # Range check
    if idx.min() < 0 or idx.max() >= W:
        raise RuntimeError(f"[indices] 0-based indices are out of range [0,{W-1}]: min={idx.min()}, max={idx.max()}")

    idx = np.unique(idx)
    return idx


def _scale_to_ref_minmax(vol: torch.Tensor, ref: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    vol, ref: (Ny,Nx,Nz) or batched shapes -> internally scale both using ref min/max.
    Returns (vol_s, ref_s), both in 0..1.
    """
    while vol.ndim < 3: vol = vol.squeeze(0)
    while ref.ndim < 3: ref = ref.squeeze(0)
    lo, hi = ref.min(), ref.max()
    vol_s = ((vol - lo) / (hi - lo + 1e-6)).clamp(0,1)
    ref_s = ((ref - lo) / (hi - lo + 1e-6)).clamp(0,1)
    return vol_s, ref_s

def _bp_volume_from_sino(sino_2d: torch.Tensor,
                         base_feat: torch.Tensor,
                         meta: Dict[str,Any],
                         device: torch.device,
                         bp_row_start: int = 3) -> torch.Tensor:
    """
    sino_2d: (H,W), base_feat: (C0,H,W). Uses the mask to determine measured columns (if pos_xyz has 512 sensors, use all columns).
    Run differentiable 3D back-projection using BP parameters from meta.
    Returns (Ny,Nx,Nz) float32 tensor (supports autograd).
    """
    pos_dev_full = _as_pos_tensor(meta["pos_xyz"], device)
    mask = (base_feat[1:2] > 0).to(device)
    # If pos_xyz has all 512 sensors: use full sinogram; otherwise align to measured subset.
    s_used, pos_dev = _align_sig_pos_for_bp(sino_2d, base_feat, pos_dev_full)
    if bp_row_start > 0:
        s_used = s_used[bp_row_start:, :]
        bp_t0_adj = int(meta["bp_t0"]) + int(bp_row_start)
    else:
        bp_t0_adj = int(meta["bp_t0"])
    vol = backproject_gpu_matchCPU(
        s_used, tuple(meta["bp_dims"]), tuple(meta["bp_res"]), pos_dev,
        float(meta["c0"]), float(meta["bp_fs"]), bp_t0_adj,
        tuple(meta["bp_filter"]), int(meta["chan_block"]), int(meta["z_slab"])
    )  # (Ny,Nx,Nz)
    return vol


# ============== LCN / Grad-Orient (optional) ==============
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

# =================== Sino Loss ==================
class SinoLoss(nn.Module):
    def __init__(self, w_mse=1.0, w_missing=6.0, w_grad=0.25, w_lcn=0.0, w_gori=0.0, use_ssim=False):
        super().__init__()
        self.w_mse=float(w_mse); self.w_missing=float(w_missing); self.w_grad=float(w_grad)
        self.w_lcn=float(w_lcn); self.w_gori=float(w_gori); self.use_ssim=bool(use_ssim)

    def _ssim(self,a,b):
        if (not self.use_ssim) or (not HAS_MSSSIM): return 0.0*a.mean()
        dr=(b.max()-b.min()).clamp_min(1e-6).detach()
        return 1.0 - ms_ssim(a,b,data_range=float(dr.item()),size_average=True)

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

# ====================== Visualization helpers =================
@torch.no_grad()
def _interp_input_for_viz(xin, msk, mode="linear"):
    x=xin.squeeze(0).cpu().numpy(); m=msk.squeeze(0).cpu().numpy()>0
    H,W=x.shape; out=np.copy(x); idx_all=np.arange(W); mea=m[0,:]
    if mea.sum()>=2 and mode=="linear":
        for r in range(H): out[r,:]=np.interp(idx_all, idx_all[mea], x[r,mea])
    return torch.from_numpy(out).unsqueeze(0).to(xin.device)

@torch.no_grad()
def save_triplet_png(x_in, pred, gt, out_path: str, same_scale=True, viz_input="sparse"):
    xin0=x_in[0:1]; mask=x_in[1:2]
    xpred=pred; xgt=gt
    xin_vis=_interp_input_for_viz(xin0,mask,mode=viz_input) if viz_input!="sparse" else xin0
    if same_scale:
        lo,hi=xgt.min(),xgt.max(); sc=lambda t: ((t-lo)/(hi-lo+1e-6)).clamp(0,1)
        xin_vis,xpred,xgt=sc(xin_vis),sc(xpred),sc(xgt)
    grid=torch.cat([xin_vis, xpred, xgt], dim=0).unsqueeze(1)
    save_image(grid.cpu(), out_path, nrow=3, padding=0)

@torch.no_grad()
def save_mip_triplet_png(base_feat, pred, gt, out_path: str, meta: Dict[str,Any], device: torch.device, bp_row_start: int = 3):
    s_in=base_feat[0]; s_pr=pred[0]; s_gt=gt[0]
    pos_dev_full = _as_pos_tensor(meta["pos_xyz"], device)
    s_in_used,  pos_dev = _align_sig_pos_for_bp(s_in, base_feat, pos_dev_full)
    s_pr_used,  _       = _align_sig_pos_for_bp(s_pr, base_feat, pos_dev_full)
    s_gt_used,  _       = _align_sig_pos_for_bp(s_gt, base_feat, pos_dev_full)
    if bp_row_start > 0:
        s_in_used = s_in_used[bp_row_start:, :]
        s_pr_used = s_pr_used[bp_row_start:, :]
        s_gt_used = s_gt_used[bp_row_start:, :]
    bp_t0_adj = int(meta["bp_t0"]) + int(bp_row_start)

    vol_in=backproject_gpu_matchCPU(s_in_used, tuple(meta["bp_dims"]), tuple(meta["bp_res"]), pos_dev,
                                    float(meta["c0"]), float(meta["bp_fs"]), bp_t0_adj,
                                    tuple(meta["bp_filter"]), int(meta["chan_block"]), int(meta["z_slab"]))
    vol_pr=backproject_gpu_matchCPU(s_pr_used, tuple(meta["bp_dims"]), tuple(meta["bp_res"]), pos_dev,
                                    float(meta["c0"]), float(meta["bp_fs"]), bp_t0_adj,
                                    tuple(meta["bp_filter"]), int(meta["chan_block"]), int(meta["z_slab"]))
    vol_gt=backproject_gpu_matchCPU(s_gt_used, tuple(meta["bp_dims"]), tuple(meta["bp_res"]), pos_dev,
                                    float(meta["c0"]), float(meta["bp_fs"]), bp_t0_adj,
                                    tuple(meta["bp_filter"]), int(meta["chan_block"]), int(meta["z_slab"]))
    m_in=softmip(vol_in.unsqueeze(0), tau=0.1, dim=3).unsqueeze(1)
    m_pr=softmip(vol_pr.unsqueeze(0), tau=0.1, dim=3).unsqueeze(1)
    m_gt=softmip(vol_gt.unsqueeze(0), tau=0.1, dim=3).unsqueeze(1)
    lo,hi=m_gt.min(),m_gt.max(); sc=lambda t: ((t-lo)/(hi-lo+1e-6)).clamp(0,1)
    grid=torch.cat([sc(m_in),sc(m_pr),sc(m_gt)],dim=0)
    save_image(grid.cpu(), out_path, nrow=3, padding=0)

# ============================ Train ===========================
def list_mat_files(root: str) -> List[str]:
    root=Path(root)
    return sorted([str(p) for p in root.rglob("*.mat") if p.is_file() and not p.name.startswith("._")])

# def train(train_dir: str, val_dir: str, test_dir: str, out_dir: str,
#           sparse_ratio=4, offset=1, frame_start_1based=2001, train_start=5001, val_start=15001, test_start=15001, num_frames=50,
#           prefill_mode="linear", fourier_max=64,
#           norm_mode="percentile", p_low=2.0, p_high=98.0, norm_axis="global", norm_from_measured=True,
#           blend_start=0, blend_len=0,
#           K=4, base_ch=64, norm="gn", share_weights=True,
#           epochs=100, bs=1, lr=2e-4, workers=2, seed=42, weight_decay=0.0,
#           w_mse=1.0, w_missing=6.0, w_grad=0.25, w_lcn=0.0, w_gori=0.0, deep_sup_w=0.2,
#           use_bp=True, w_bp=1.0, bp_row_start: int = 3,
#           enforce_measured_eval=True, save_train_every=0,
#           pos_xyz_dir="", pos_xyz_var=None, sparse_source="ratio", index_mat=None, index_var=None, index_1based=False):
def train(train_dir: str, val_dir: str, test_dir: str, out_dir: str,
          sparse_ratio=2, offset=2, frame_start_1based=2001, train_start=5001, val_start=15001, test_start=15001,
          num_frames=50,
          prefill_mode="linear", fourier_max=64,
          norm_mode="percentile", p_low=2.0, p_high=98.0, norm_axis="global", norm_from_measured=True,
          blend_start=0, blend_len=0,
          K=4, base_ch=64, norm="gn", share_weights=True,
          epochs=100, bs=1, lr=2e-4, workers=2, seed=42, weight_decay=0.0,
          w_mse=1.0, w_missing=6.0, w_grad=0.25, w_lcn=0.0, w_gori=0.0, deep_sup_w=0.2,
          use_bp=True, w_bp=1.0, bp_row_start: int = 3, bp_loss: str = "volume",
          enforce_measured_eval=True, save_train_every=0,
          pos_xyz_dir="", pos_xyz_var=None, sparse_source="ratio", index_mat=None, index_var=None,
          index_1based=False):

    torch.manual_seed(seed);
    np.random.seed(seed)
    assert pos_xyz_dir and Path(
        pos_xyz_dir).exists(), "--pos_xyz_dir is required (should contain pos_sensor_xyz_ultracup.mat / pos_sensor_xyz_holycup.mat)."

    # --- File list ---
    tr_files = list_mat_files(train_dir);
    va_files = list_mat_files(val_dir);
    te_files = list_mat_files(test_dir)
    print(f"[Files] train={len(tr_files)} val={len(va_files)} test={len(te_files)} (.mat files)")

    # --- Sparse sampling source ---
    sidx_override = None
    if sparse_source == "index_mat":
        if not index_mat:
            raise RuntimeError("--index_mat is required when --sparse_source is 'index_mat'.")
        sidx_override = load_keep_indices(index_mat, index_var, W=512,
                                          index_1based=(True if index_1based else None))
        print(f"[indices] Loaded {sidx_override.size} measured columns from {index_mat}")

    # --- Build datasets (inject sidx_override if provided) ---
    train_ds = SinoOnlyDataset(tr_files, sparse_ratio, offset, frame_start_1based, num_frames,
                               prefill_mode, fourier_max, norm_mode, p_low, p_high, norm_axis, norm_from_measured,
                               pos_xyz_dir=pos_xyz_dir, pos_xyz_var=pos_xyz_var,
                               sidx_override=sidx_override)

    val_ds = SinoOnlyDataset(va_files, sparse_ratio, offset,
                             frame_start_1based=val_start,
                             num_frames=min(1, num_frames),
                             prefill_mode=prefill_mode, fourier_max=fourier_max,
                             pos_xyz_dir=pos_xyz_dir, pos_xyz_var=pos_xyz_var,
                             sidx_override=sidx_override)

    test_ds = SinoOnlyDataset(te_files, sparse_ratio, offset,
                              frame_start_1based=test_start,
                              num_frames=min(1, num_frames),
                              prefill_mode=prefill_mode, fourier_max=fourier_max,
                              pos_xyz_dir=pos_xyz_dir, pos_xyz_var=pos_xyz_var,
                              sidx_override=sidx_override)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader=DataLoader(train_ds,batch_size=bs,shuffle=True,num_workers=workers,pin_memory=True,drop_last=True)
    val_loader  =DataLoader(val_ds,  batch_size=1,shuffle=False,num_workers=workers,pin_memory=True)
    test_loader =DataLoader(test_ds, batch_size=1,shuffle=False,num_workers=workers,pin_memory=True)

    base_feat_ch = next(iter(train_loader))[0].shape[1]
    model = UnrolledReconstructor(base_feat_ch=base_feat_ch, K=K, base=base_ch, norm=norm, share_weights=share_weights).to(device)
    sino_crit = SinoLoss(w_mse, w_missing, w_grad, w_lcn, w_gori, use_ssim=False)

    ckpt_dir=Path(out_dir)/"ckpt"; smp_tr=Path(out_dir)/"samples"/"train"; smp_va=Path(out_dir)/"samples"/"val"; smp_te=Path(out_dir)/"samples"/"test"
    for d in [ckpt_dir,smp_tr,smp_va,smp_te]: d.mkdir(parents=True, exist_ok=True)

    opt=torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5, verbose=True)

    # Debug measured cols
    bf_example, _, _ = next(iter(train_loader))
    mask_example = bf_example[:,1:2]
    meas_cols = int(mask_example.sum().item() / (bf_example.shape[0]*train_ds.H))
    print(f"[DEBUG] measured cols = {meas_cols}")

    best_val=float('inf'); best_ep=-1

    for ep in range(1, epochs+1):
        if train_ds.prefill_mode=="blend":
            train_ds.alpha=linear_alpha(ep, blend_start, blend_len)

        # ---------- Train ----------
        model.train(); tr_losses=[]
        for bidx,(base_feat,y,meta) in enumerate(tqdm(train_loader, desc=f"Epoch {ep}/{epochs} [Train]")):
            base_feat=base_feat.to(device, non_blocking=True); y=y.to(device, non_blocking=True)
            x0=base_feat[:,0:1]; mask=base_feat[:,1:2]

            p_last, p_all = model(base_feat)
            r_last = p_last - x0

            loss_last, _ = sino_crit(p_last, y, mask, x0, r_last)

            if K>1 and deep_sup_w>0:
                ds=0.0
                for k in range(len(p_all)-1):
                    pk=p_all[k]; rk=pk-x0
                    dsk,_=sino_crit(pk,y,mask,x0,rk); ds+=dsk
                ds = deep_sup_w*ds/max(1,len(p_all)-1)
            else:
                ds=0.0*loss_last

            # --- BP loss ---
            loss_bp = 0.0 * loss_last
            if use_bp and w_bp > 0:
                # pred/gt sinogram
                s_pr = p_last[0, 0]  # (H,W)
                s_gt = y[0]  # (H,W)

                if bp_loss == "volume":
                    # 3D volume BP -> min/max normalize by reference -> MSE
                    vol_pr = _bp_volume_from_sino(s_pr, base_feat[0], meta, device,
                                                  bp_row_start=bp_row_start)  # (Ny,Nx,Nz)
                    vol_gt = _bp_volume_from_sino(s_gt, base_feat[0], meta, device, bp_row_start=bp_row_start)
                    vol_pr_s, vol_gt_s = _scale_to_ref_minmax(vol_pr, vol_gt)
                    loss_bp = F.mse_loss(vol_pr_s, vol_gt_s)

                else:  # "mip" (legacy SoftMIP mode)
                    pos_dev_full = _as_pos_tensor(meta["pos_xyz"], device)
                    s_pr_used, pos_dev = _align_sig_pos_for_bp(s_pr, base_feat[0], pos_dev_full)
                    s_gt_used, _ = _align_sig_pos_for_bp(s_gt, base_feat[0], pos_dev_full)
                    if bp_row_start > 0:
                        s_pr_used = s_pr_used[bp_row_start:, :]
                        s_gt_used = s_gt_used[bp_row_start:, :]
                    bp_t0_adj = int(meta["bp_t0"]) + int(bp_row_start)

                    vol_pr = backproject_gpu_matchCPU(s_pr_used, tuple(meta["bp_dims"]), tuple(meta["bp_res"]), pos_dev,
                                                      float(meta["c0"]), float(meta["bp_fs"]), bp_t0_adj,
                                                      tuple(meta["bp_filter"]), int(meta["chan_block"]),
                                                      int(meta["z_slab"]))
                    vol_gt = backproject_gpu_matchCPU(s_gt_used, tuple(meta["bp_dims"]), tuple(meta["bp_res"]), pos_dev,
                                                      float(meta["c0"]), float(meta["bp_fs"]), bp_t0_adj,
                                                      tuple(meta["bp_filter"]), int(meta["chan_block"]),
                                                      int(meta["z_slab"]))
                    mip_pr = softmip(vol_pr.unsqueeze(0), tau=0.08, dim=3).unsqueeze(1)
                    mip_gt = softmip(vol_gt.unsqueeze(0), tau=0.08, dim=3).unsqueeze(1)
                    lo, hi = mip_gt.min(), mip_gt.max()
                    mip_pr = (mip_pr - lo) / (hi - lo + 1e-6);
                    mip_gt = (mip_gt - lo) / (hi - lo + 1e-6)
                    loss_bp = F.mse_loss(mip_pr, mip_gt)

            else:
                loss_bp=0.0*loss_last

            loss = loss_last + ds + w_bp*loss_bp

            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
            tr_losses.append(float(loss.item()))

            if save_train_every>0 and (bidx % save_train_every==0):
                save_triplet_png(base_feat[0], p_last[0], y[0], str(smp_tr/f"epoch_{ep:03d}_b{bidx:04d}.png"), viz_input="sparse")
                save_mip_triplet_png(base_feat[0], p_last[0], y[0], str(smp_tr/f"epoch_{ep:03d}_b{bidx:04d}_mip.png"), meta, device, bp_row_start=bp_row_start)

        mean_tr=float(np.mean(tr_losses)) if tr_losses else float('nan')

        # ---------- Val ----------
        model.eval(); va_losses=[]
        with torch.no_grad():
            for base_feat,y,meta in tqdm(val_loader, desc=f"Epoch {ep}/{epochs} [Val]"):
                base_feat=base_feat.to(device); y=y.to(device)
                x0=base_feat[:,0:1]; mask=base_feat[:,1:2]
                p_last,_=model(base_feat)
                r_last=p_last-x0
                ls,_=sino_crit(p_last,y,mask,x0,r_last)

                lb = 0.0 * ls
                if use_bp and w_bp > 0:
                    s_pr = p_last[0, 0]
                    s_gt = y[0]

                    if bp_loss == "volume":
                        vol_pr = _bp_volume_from_sino(s_pr, base_feat[0], meta, device, bp_row_start=bp_row_start)
                        vol_gt = _bp_volume_from_sino(s_gt, base_feat[0], meta, device, bp_row_start=bp_row_start)
                        vol_pr_s, vol_gt_s = _scale_to_ref_minmax(vol_pr, vol_gt)
                        lb = F.mse_loss(vol_pr_s, vol_gt_s)

                    else:  # "mip"
                        pos_dev_full = _as_pos_tensor(meta["pos_xyz"], device)
                        s_pr_used, pos_dev = _align_sig_pos_for_bp(s_pr, base_feat[0], pos_dev_full)
                        s_gt_used, _ = _align_sig_pos_for_bp(s_gt, base_feat[0], pos_dev_full)
                        if bp_row_start > 0:
                            s_pr_used = s_pr_used[bp_row_start:, :]
                            s_gt_used = s_gt_used[bp_row_start:, :]
                        bp_t0_adj = int(meta["bp_t0"]) + int(bp_row_start)

                        vol_pr = backproject_gpu_matchCPU(s_pr_used, tuple(meta["bp_dims"]), tuple(meta["bp_res"]),
                                                          pos_dev,
                                                          float(meta["c0"]), float(meta["bp_fs"]), bp_t0_adj,
                                                          tuple(meta["bp_filter"]), int(meta["chan_block"]),
                                                          int(meta["z_slab"]))
                        vol_gt = backproject_gpu_matchCPU(s_gt_used, tuple(meta["bp_dims"]), tuple(meta["bp_res"]),
                                                          pos_dev,
                                                          float(meta["c0"]), float(meta["bp_fs"]), bp_t0_adj,
                                                          tuple(meta["bp_filter"]), int(meta["chan_block"]),
                                                          int(meta["z_slab"]))
                        mip_pr = softmip(vol_pr.unsqueeze(0), tau=0.08, dim=3).unsqueeze(1)
                        mip_gt = softmip(vol_gt.unsqueeze(0), tau=0.08, dim=3).unsqueeze(1)
                        lo, hi = mip_gt.min(), mip_gt.max()
                        mip_pr = (mip_pr - lo) / (hi - lo + 1e-6);
                        mip_gt = (mip_gt - lo) / (hi - lo + 1e-6)
                        lb = F.mse_loss(mip_pr, mip_gt)

                else:
                    lb=0.0*ls
                va_losses.append(float((ls + w_bp*lb).item()))

            # Example PNG (one sample for train/val)
            bt,yt,mt = next(iter(DataLoader(train_ds,batch_size=1,shuffle=True)))
            bt=bt.to(device); yt=yt.to(device); pt,_=model(bt)
            save_triplet_png(bt[0], pt[0], yt[0], str(smp_tr/f"epoch_{ep:03d}.png"), viz_input="sparse")
            save_mip_triplet_png(bt[0], pt[0], yt[0], str(smp_tr/f"epoch_{ep:03d}_mip.png"), mt, device, bp_row_start=bp_row_start)

            bv,yv,mv = next(iter(DataLoader(val_ds,batch_size=1,shuffle=True)))
            bv=bv.to(device); yv=yv.to(device); pv,_=model(bv)
            save_triplet_png(bv[0], pv[0], yv[0], str(smp_va/f"epoch_{ep:03d}.png"), viz_input="sparse")
            save_mip_triplet_png(bv[0], pv[0], yv[0], str(smp_va/f"epoch_{ep:03d}_mip.png"), mv, device, bp_row_start=bp_row_start)

        mean_va=float(np.mean(va_losses)) if va_losses else float('nan')
        print(f"[Epoch {ep:03d}/{epochs}] total loss tr {mean_tr:.4f} / val {mean_va:.4f}")
        sched.step(mean_va)

        # ---------- Test (single sample) ----------
        with torch.no_grad():
            for base_feat,y,meta in test_loader:
                base_feat=base_feat.to(device); y=y.to(device)
                p,_=model(base_feat)
                save_triplet_png(base_feat[0], p[0], y[0], str(smp_te/f"epoch_{ep:03d}.png"), viz_input="sparse")
                save_mip_triplet_png(base_feat[0], p[0], y[0], str(smp_te/f"epoch_{ep:03d}_mip.png"), meta, device, bp_row_start=bp_row_start)
                break

        # ckpt
        torch.save({"epoch":ep,"model":model.state_dict(),"val_loss":mean_va,"train_loss":mean_tr},
                   ckpt_dir/f"epoch_{ep:03d}.pt")
        if mean_va<best_val:
            best_val, best_ep=mean_va, ep
            torch.save({"epoch":ep,"model":model.state_dict(),"val_loss":best_val,"train_loss":mean_tr},
                       Path(out_dir)/"best.pt")

    torch.save({"epoch":epochs,"model":model.state_dict(),"val_loss":best_val,"best_epoch":best_ep},
               Path(out_dir)/"last.pt")

# ============================ Main ============================
if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser(description="Unrolled sparse→full PAT (per-file pos_xyz & BP meta + angular prefill + geom channels)")

    # paths
    p.add_argument("--train", type=str, default="data/dl1/train")
    p.add_argument("--val",   type=str, default="data/dl1/val")
    p.add_argument("--test",  type=str, default="data/dl1/test")
    p.add_argument("--out",   type=str, default="outputs/dl1")

    # pos_xyz directory (auto-selected based on filename)
    p.add_argument("--pos_xyz_dir", type=str, default="data/pos_sensor_xyz")
    p.add_argument("--pos_xyz_var", type=str, default=None)

    # sampling & prefill & norm
    p.add_argument("--sparse_ratio", type=int, default=2)
    p.add_argument("--offset", type=int, default=None, help="Optional 0-based offset for the sparse pattern (default: ratio-1)")
    # p.add_argument("--frame_start", type=int, default=
    p.add_argument("--train_start", type=int, default=1, help="1-based start frame index for training")
    p.add_argument("--val_start", type=int, default=1, help="1-based start frame index for validation")
    p.add_argument("--test_start", type=int, default=1, help="1-based start frame index for testing")
    p.add_argument("--num_frames",  type=int, default=1000)
    p.add_argument("--prefill_mode", type=str, default="angular", choices=["angular","linear","zero","blend"])
    p.add_argument("--fourier_max", type=int, default=64)

    p.add_argument("--norm_mode", type=str, default="percentile", choices=["minmax","percentile","zscore_row","zscore_col"])
    p.add_argument("--p_low", type=float, default=2.0)
    p.add_argument("--p_high", type=float, default=98.0)
    p.add_argument("--norm_axis", type=str, default="global", choices=["global","row","col"])
    p.add_argument("--norm_from_measured", dest="norm_from_measured", action="store_true",
                   help="Estimate normalization range from measured columns only.")
    p.add_argument("--no_norm_from_measured", dest="norm_from_measured", action="store_false",
                   help="Estimate normalization range from the full (prefilled) sinogram.")
    p.set_defaults(norm_from_measured=True)

    p.add_argument("--blend_start", type=int, default=0)
    p.add_argument("--blend_len", type=int, default=0)

    # model
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--base_ch", type=int, default=64)
    p.add_argument("--norm", type=str, default="gn", choices=["none","in","gn"])
    p.add_argument("--no_share", action="store_true", help="do not share prox weights across steps")

    # training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--bs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--weight_decay", type=float, default=0.0)

    # sino loss
    p.add_argument("--w_mse", type=float, default=1.0)
    p.add_argument("--w_missing", type=float, default=6.0)
    p.add_argument("--w_grad", type=float, default=0.25)
    p.add_argument("--w_lcn", type=float, default=0.0)
    p.add_argument("--w_gori", type=float, default=0.0)
    p.add_argument("--deep_sup_w", type=float, default=0.2)

    # BP
    p.add_argument("--use_bp", action="store_true")
    p.add_argument("--w_bp", type=float, default=0.0) # default = 5.0
    p.add_argument("--bp_row_start", type=int, default=3, help="Row offset for BP input sinogram (0-based). MATLAB 4:end corresponds to 3.")
    p.add_argument("--enforce_measured_eval", dest="enforce_measured_eval", action="store_true",
                   help="Enforce measured channels (hard DC) during validation/test.")
    p.add_argument("--no_enforce_measured_eval", dest="enforce_measured_eval", action="store_false",
                   help="Do not enforce measured channels during validation/test (debug only).")
    p.set_defaults(enforce_measured_eval=True)
    p.add_argument("--save_train_every", type=int, default=0)

    p.add_argument("--sparse_source", type=str, default="ratio", choices=["ratio", "index_mat"], help="'ratio': use sparse_ratio/offset. 'index_mat': load measured indices from a .mat file.")
    # p.add_argument("--sparse_source", type=str, default="index_mat", help="'ratio': use sparse_ratio/offset. 'index_mat': load measured indices from a .mat file.")
    p.add_argument("--index_mat", type=str, default="data/dl1/measured_indices.mat")
    p.add_argument("--index_var", type=str, default=None, help="Variable name inside .mat (if None, the script will try to auto-detect).")
    p.add_argument("--index_1based", action="store_true", help="Force 1-based interpretation (default: auto-detect).")

    p.add_argument("--bp_loss", type=str, default="volume", choices=["mip", "volume"], help="BP supervision type: 'volume' uses 3D volume MSE; 'mip' uses SoftMIP MSE.")
    args=p.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out)/"run_args.json","w",encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    train(
        train_dir=args.train, val_dir=args.val, test_dir=args.test, out_dir=args.out,
        sparse_ratio=args.sparse_ratio, offset=args.offset,
        train_start=args.train_start, val_start=args.val_start, test_start=args.test_start,
        num_frames=args.num_frames,
        prefill_mode=args.prefill_mode, fourier_max=args.fourier_max,
        norm_mode=args.norm_mode, p_low=args.p_low, p_high=args.p_high, norm_axis=args.norm_axis,
        norm_from_measured=args.norm_from_measured,
        blend_start=args.blend_start, blend_len=args.blend_len,
        K=args.K, base_ch=args.base_ch, norm=args.norm, share_weights=(not args.no_share),
        epochs=args.epochs, bs=args.bs, lr=args.lr, workers=args.workers, seed=args.seed,
        weight_decay=args.weight_decay,
        w_mse=args.w_mse, w_missing=args.w_missing, w_grad=args.w_grad, w_lcn=args.w_lcn, w_gori=args.w_gori,
        deep_sup_w=args.deep_sup_w,
        use_bp=args.use_bp, w_bp=args.w_bp, bp_row_start=args.bp_row_start, bp_loss=args.bp_loss,
        enforce_measured_eval=args.enforce_measured_eval, save_train_every=args.save_train_every,
        pos_xyz_dir=args.pos_xyz_dir, pos_xyz_var=args.pos_xyz_var,
        sparse_source=args.sparse_source, index_mat=args.index_mat,
        index_var=args.index_var, index_1based=args.index_1based
    )
