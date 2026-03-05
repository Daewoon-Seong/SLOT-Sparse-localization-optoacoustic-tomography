# -*- coding: utf-8 -*-
"""
Test-time inference for PAT sparse→full (Unrolled Reconstruction) — SLIM
- Reads ALL frames in each .mat under --test directory (no frame limit)
- Supports sparse source: 'ratio' or 'index_mat'
- Saves per-frame PNGs (sino triplet) and optional BP-SoftMIP PNGs (debug)
- Saves ONLY ONE dataset to v7.3(HDF5) .mat via h5py (streaming):
  * sino_pred (uint16, values are in **GT RAW units** per frame)

How it works:
- Inference runs on 0..1 normalized inputs.
- Before saving, the prediction is inverted back to RAW per-frame using lo/hi
  estimated from the measured columns (or fallback if disabled), then quantized
  to uint16. No duplicated stacks.

Requires: torch, numpy, scipy, h5py, torchvision, tqdm, pillow
"""

import os, json, warnings, sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm

# ---------------- I/O helpers ----------------
def _h5_find_dataset(f):
    import h5py
    cands=[]
    def visit(name,obj):
        try:
            if isinstance(obj,h5py.Dataset) and obj.ndim==3 and np.issubdtype(obj.dtype,np.number):
                cands.append(name)
        except Exception: pass
    f.visititems(lambda n,o: visit(n,o))
    if not cands: raise RuntimeError("HDF5 내 3D numeric dataset을 찾지 못했습니다.")
    return cands[0]

def _reorder_axes_to_HWF(shape, H_target=496, W_target=512):
    axes=list(range(3))
    try:
        idxH=next(i for i,d in enumerate(shape) if d==H_target)
        idxW=next(i for i,d in enumerate(shape) if d==W_target)
    except StopIteration:
        dims=np.array(shape); idxH=int(np.argmin(np.abs(dims-H_target))); idxW=int(np.argmin(np.abs(dims-W_target)))
        if idxH==idxW: raise RuntimeError(f"형상 {shape}에서 H/W 축 판단 불가")
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
    raise RuntimeError(f"{path}: 3D 배열을 찾지 못했습니다.")

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
    raise RuntimeError(f"{path}: 프레임 로드 실패")

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
    if arr is None: raise RuntimeError("pos_sensor_xyz (.mat)에서 (Nc,3) 배열을 찾지 못했습니다.")
    return np.asarray(arr,dtype=np.float32)

def list_mat_files(root: str) -> List[str]:
    root=Path(root)
    return sorted([str(p) for p in root.rglob("*.mat") if p.is_file() and not p.name.startswith("._")])

# --------------- sparse & prefill & normalize ----------------
def sparse_indices(ratio: int, offset: Optional[int], W: int = 512) -> np.ndarray:
    if offset is None: offset=ratio-1
    return np.arange(offset,W,ratio,dtype=np.int64)

def load_keep_indices(mat_path: str, var_name: Optional[str], W: int = 512,
                      index_1based: Optional[bool] = None) -> np.ndarray:
    from scipy.io import loadmat
    md = loadmat(mat_path)
    cand = None
    if var_name and var_name in md:
        cand = md[var_name]
    else:
        for k, v in md.items():
            if k.startswith("__"): continue
            if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
                cand = v; break
    if cand is None:
        raise RuntimeError(f"[indices] {mat_path}: 숫자 배열을 찾지 못했습니다.")
    idx = np.asarray(cand).astype(np.int64).ravel()
    if idx.size == 0:
        raise RuntimeError(f"[indices] {mat_path}: 인덱스가 비어있습니다.")
    if index_1based is None:
        if np.any(idx == 0):
            index_1based = False
        else:
            index_1based = (idx.min() >= 1 and idx.max() <= W)
    if index_1based: idx = idx - 1
    if idx.min() < 0 or idx.max() >= W:
        raise RuntimeError(f"[indices] 0-based 인덱스 범위 초과: min={idx.min()}, max={idx.max()}")
    return np.unique(idx)

def linear_prefill(full: np.ndarray, mask: np.ndarray) -> np.ndarray:
    H,W=full.shape
    out=np.zeros_like(full,dtype=np.float32)
    idx_all=np.arange(W); mea=mask[0]>0
    if mea.sum()>=2:
        for r in range(H): out[r,:]=np.interp(idx_all, idx_all[mea], full[r,mea])
    return out

def angular_prefill(full: np.ndarray, mask: np.ndarray, pos_xyz_col: np.ndarray) -> np.ndarray:
    H,W = full.shape
    out = np.zeros_like(full, dtype=np.float32)
    phi = np.arctan2(pos_xyz_col[:,1], pos_xyz_col[:,0]).astype(np.float64)
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

def _as_pos_tensor(x, device: torch.device) -> torch.Tensor:
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

def _normalized_name(path_like: str) -> str:
    name = Path(path_like).stem.lower()
    name = " ".join(name.split())
    return name

def pick_bp_meta_for_path(mat_path: str, pos_xyz_dir: str, pos_xyz_var: Optional[str]) -> Dict[str, Any]:
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
            # meta.update({"bp_fs": 40e6, "bp_t0": 824, "tag": "holycup_40msps"})
        else:
            meta.update({"bp_fs": 24e6, "bp_t0": 501, "tag": "holycup_24msps"})
        if has("sos"):
            meta["c0"] = 1480.0
            meta["tag"] += "_sos"
    else:
        warnings.warn(f"[BP meta] {Path(mat_path).name}: tag 미검출 → ultracup fallback")
        pfile = d / "pos_sensor_xyz_ultracup.mat"
        meta.update({"bp_fs": 40e6, "bp_t0": 832, "tag": "fallback_ultracup"})
    if not pfile.is_file():
        raise FileNotFoundError(f"pos_xyz 파일 없음: {pfile}")
    meta["pos_xyz"] = load_pos_sensor_xyz(str(pfile), pos_xyz_var).astype(np.float32)
    return meta

# ---------- BP (SoftMIP) ----------
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

def align_sig_pos_for_bp_full(sino_2d: torch.Tensor,
                              mask: torch.Tensor,
                              pos_xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    device = sino_2d.device
    H, W = sino_2d.shape
    Nc_pos = pos_xyz.shape[0]
    meas_cols = torch.where(mask[0,0] > 0)[0].to(device=sino_2d.device, dtype=torch.long)
    Nm = int(meas_cols.numel())
    if Nm == 0:
        return sino_2d, pos_xyz
    if Nc_pos == W:
        return sino_2d, pos_xyz
    if Nc_pos == Nm:
        sino_used = sino_2d.index_select(1, meas_cols)
        return sino_used, pos_xyz
    raise RuntimeError(f"[BP align] pos_xyz Nc({Nc_pos}) not matched to W({W}) nor Nm({Nm}).")

# ------------------ UNet & Unrolled ------------------
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

# ---------------------- viz helpers ----------------------
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
    if viz_input == "sparse":
        xin_vis = xin0 * mask
    elif viz_input == "prefilled":
        xin_vis = xin0
    elif viz_input == "linear":
        xin_vis = _interp_input_for_viz(xin0*mask, mask, mode="linear")
    else:
        xin_vis = xin0 * mask
    xpred=pred; xgt=gt
    if same_scale:
        lo,hi=xgt.min(),xgt.max(); sc=lambda t: ((t-lo)/(hi-lo+1e-6)).clamp(0,1)
        xin_vis,xpred,xgt=sc(xin_vis),sc(xpred),sc(xgt)
    grid=torch.cat([xin_vis, xpred, xgt], dim=0).unsqueeze(1)
    save_image(grid.cpu(), out_path, nrow=3, padding=0)

@torch.no_grad()
def save_mip_png(base_feat, pred, gt, out_path: str, meta: Dict[str,Any], device: torch.device, bp_row_start: int = 3):
    mask = (base_feat[1:2] > 0).to(device)
    s_in = base_feat[0].to(device); s_pr = pred[0].to(device); s_gt = gt[0].to(device)
    pos_dev_full = _as_pos_tensor(meta["pos_xyz"], device)
    s_in_used,  pos_dev = align_sig_pos_for_bp_full(s_in, mask, pos_dev_full)
    s_pr_used,  _       = align_sig_pos_for_bp_full(s_pr, mask, pos_dev_full)
    s_gt_used,  _       = align_sig_pos_for_bp_full(s_gt, mask, pos_dev_full)
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
    def softmip(vol, tau=0.1, dim=2):
        w=torch.softmax(vol/tau,dim=dim)
        return (w*vol).sum(dim=dim)
    m_in=softmip(vol_in.unsqueeze(0), tau=0.1, dim=3).unsqueeze(1)
    m_pr=softmip(vol_pr.unsqueeze(0), tau=0.1, dim=3).unsqueeze(1)
    m_gt=softmip(vol_gt.unsqueeze(0), tau=0.1, dim=3).unsqueeze(1)
    lo,hi=m_gt.min(),m_gt.max(); sc=lambda t: ((t-lo)/(hi-lo+1e-6)).clamp(0,1)
    grid=torch.cat([sc(m_in),sc(m_pr),sc(m_gt)],dim=0)
    save_image(grid.cpu(), out_path, nrow=3, padding=0)

# --------------------------- main test loop ---------------------------
def main():
    import argparse
    p=argparse.ArgumentParser(description="Test-time inference for Unrolled PAT (all frames) — SLIM (uint16 RAW only)")
    # paths
    p.add_argument("--test", type=str, required=True)
    p.add_argument("--out",  type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--pos_xyz_dir", type=str, required=True)
    p.add_argument("--pos_xyz_var", type=str, default=None)
    p.add_argument("--var_hint", type=str, default=None)

    # sparse & prefill & norm
    p.add_argument("--sparse_source", type=str, default="ratio", choices=["ratio","index_mat"])
    p.add_argument("--sparse_ratio", type=int, default=4)
    p.add_argument("--offset", type=int, default=2)
    p.add_argument("--index_mat", type=str, default=None)
    p.add_argument("--index_var", type=str, default=None)
    p.add_argument("--index_1based", action="store_true")

    p.add_argument("--prefill_mode", type=str, default="angular", choices=["angular","linear","zero","blend"])
    p.add_argument("--fourier_max", type=int, default=64)
    p.add_argument("--norm_mode", type=str, default="percentile", choices=["minmax","percentile","zscore_row","zscore_col"])
    p.add_argument("--p_low", type=float, default=2.0)
    p.add_argument("--p_high", type=float, default=98.0)
    p.add_argument("--norm_axis", type=str, default="global", choices=["global","row","col"])
    p.add_argument("--norm_from_measured", dest="norm_from_measured", action="store_true",
                   help="Normalize each frame using percentiles computed from measured columns only.")
    p.add_argument("--no_norm_from_measured", dest="norm_from_measured", action="store_false",
                   help="Disable measured-column-only normalization and use robust_normalize().")
    p.set_defaults(norm_from_measured=True)

    # model
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--base_ch", type=int, default=64)
    p.add_argument("--norm", type=str, default="gn", choices=["none","in","gn"])
    p.add_argument("--no_share", action="store_true")

    # BP & save
    p.add_argument("--use_bp", action="store_true")
    p.add_argument("--bp_row_start", type=int, default=3)
    p.add_argument("--save_png_every", type=int, default=10)
    p.add_argument("--save_pred_mat", action="store_true")

    p.add_argument("--max_frames", type=int, default=0,
                   help="파일당 처리할 최대 프레임 수 (0이면 전체)")

    args=p.parse_args()
    H,W=496,512
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_root=Path(args.out); out_root.mkdir(parents=True, exist_ok=True)

    # measured indices
    if args.sparse_source=="index_mat":
        if not args.index_mat:
            raise RuntimeError("--sparse_source index_mat 인데 --index_mat 필요")
        sidx = load_keep_indices(args.index_mat, args.index_var, W=W,
                                 index_1based=(True if args.index_1based else None))
    else:
        sidx = sparse_indices(args.sparse_ratio, args.offset, W)

    # model
    def _n_fourier(maxf):
        c=0; f=1
        while f<=maxf: c+=2; f*=2
        return c
    base_feat_ch = 1 + 1 + 1 + 1 + _n_fourier(args.fourier_max) + _n_fourier(args.fourier_max) + 3
    model = UnrolledReconstructor(base_feat_ch=base_feat_ch, K=args.K, base=args.base_ch,
                                  norm=args.norm, share_weights=(not args.no_share)).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()

    mat_files = list_mat_files(args.test)
    print(f"[TEST] files={len(mat_files)}")

    for fp in mat_files:
        fp = str(fp)
        stem = Path(fp).stem
        out_dir = out_root / stem
        (out_dir/"frames").mkdir(parents=True, exist_ok=True)

        # per-file meta
        meta = pick_bp_meta_for_path(fp, args.pos_xyz_dir, args.pos_xyz_var)
        pos_xyz = meta["pos_xyz"]
        if pos_xyz.shape[0] != W:
            raise RuntimeError(f"[Dataset] pos_xyz Nc({pos_xyz.shape[0]}) != W({W}). 정합 로직 필요.")

        T = get_num_frames(fp)
        T_eff = T if args.max_frames <= 0 else min(T, int(args.max_frames))
        print(f"[{stem}] frames={T} → using first {T_eff}")

        # v7.3(HDF5) streaming outputs — ONLY ONE DATASET
        if args.save_pred_mat:
            import h5py
            h5_path = out_dir / "pred_sino_v7p3.mat"
            if h5_path.exists(): h5_path.unlink()
            hf = h5py.File(str(h5_path), "w")

            d_pred_u16 = hf.create_dataset(
                "sino_pred", shape=(H, W, T_eff), dtype="uint16",
                chunks=(H, W, 1), compression="gzip", compression_opts=6, shuffle=True
            )
            # attrs for meaning
            d_pred_u16.attrs["scale"] = "uint16_raw_units"
            d_pred_u16.attrs["note"]  = "Each frame is inverted back to GT RAW scale using per-frame lo/hi, then quantized to uint16."

        else:
            hf = None; d_pred_u16 = None

        for t in tqdm(range(1, T_eff+1), desc=f"{stem}"):
            full_raw = load_frame_496x512(fp, t, var_hint=args.var_hint)  # (H,W) float32 (RAW)

            # normalize (frame-wise) from measured columns if enabled
            if args.norm_from_measured:
                meas_vals = full_raw[:, sidx].reshape(-1)
                lo = np.percentile(meas_vals, args.p_low); hi = np.percentile(meas_vals, args.p_high); hi=max(hi,lo+1e-6)
                full = np.clip((full_raw - lo)/(hi-lo), 0, 1).astype(np.float32)
            else:
                lo = np.float32("nan"); hi = np.float32("nan")
                full = robust_normalize(full_raw, mode=args.norm_mode, p_low=args.p_low, p_high=args.p_high, axis=args.norm_axis)

            # mask & prefill
            msk = np.zeros_like(full, dtype=np.float32); msk[:, sidx] = 1.0
            if args.prefill_mode=="angular":
                xin = angular_prefill(full, msk, pos_xyz)
            elif args.prefill_mode=="linear":
                xin = linear_prefill(full, msk)
            elif args.prefill_mode=="zero":
                xin = np.zeros_like(full, dtype=np.float32)
            elif args.prefill_mode=="blend":
                xin = linear_prefill(full, msk)  # test에선 alpha=1 가정
            else:
                raise ValueError(f"Unknown prefill_mode: {args.prefill_mode}")
            xin[:, sidx] = full[:, sidx]  # hard-DC on input

            # coords/fourier/geom
            j=torch.linspace(-1,1,W,dtype=torch.float32)[None,:].repeat(H,1)
            tcoord=torch.linspace(-1,1,H,dtype=torch.float32)[:,None].repeat(1,W)
            j_t=j.unsqueeze(0); t_t=tcoord.unsqueeze(0)
            def _ff(x): return fourier_feats(x.unsqueeze(0),max_freq=args.fourier_max).squeeze(0)
            jf=_ff(j_t); tf=_ff(t_t)
            px, py, pz = pos_xyz[:,0], pos_xyz[:,1], pos_xyz[:,2]
            phi = np.arctan2(py, px).astype(np.float32)
            r = np.sqrt(px**2 + py**2 + pz**2).astype(np.float32)
            r_norm = (r / (np.max(r)+1e-6) * 2.0 - 1.0).astype(np.float32)
            geom_cols = np.stack([np.cos(phi), np.sin(phi), r_norm], axis=0)
            geom_np = np.transpose(np.repeat(geom_cols[None, :, :], H, axis=0), (1,0,2)).astype(np.float32)
            geom_t = torch.from_numpy(geom_np)

            x0=torch.from_numpy(xin).unsqueeze(0)
            m0=torch.from_numpy(msk).unsqueeze(0)
            y =torch.from_numpy(full).unsqueeze(0)
            base_feat = torch.cat([x0, m0, j_t, t_t, jf, tf, geom_t], dim=0).unsqueeze(0).to(device)

            with torch.no_grad():
                p = model(base_feat)    # (1,1,H,W)
            pred = p[0]                 # (1,H,W) in 0..1

            # save PNGs (optional debug)
            if (t % max(1, args.save_png_every)) == 0:
                png_path = out_dir/"frames"/f"{t:06d}_sino.png"
                save_triplet_png(base_feat[0].detach().cpu(), pred.detach().cpu(), y, str(png_path), viz_input="sparse")
                if args.use_bp:
                    mip_path = out_dir/"frames"/f"{t:06d}_mip.png"
                    save_mip_png(base_feat[0].detach().cpu(), pred.detach().cpu(), y, str(mip_path),
                                 meta, device, bp_row_start=args.bp_row_start)

            # --- streaming save (HDF5) --- only uint16 in RAW units
            if d_pred_u16 is not None:
                pred_np01 = pred[0].detach().cpu().numpy().astype(np.float32)  # (H,W)

                # inverse to RAW (per-frame)
                if not np.isnan(lo):
                    pred_raw = pred_np01 * (hi - lo) + lo
                else:
                    lo_fb = float(full_raw.min()); hi_fb = float(full_raw.max()); hi_fb = max(hi_fb, lo_fb+1e-6)
                    pred_raw = pred_np01 * (hi_fb - lo_fb) + lo_fb

                # RAW → uint16
                # 대부분 범위 내이므로 직접 라운딩; 범위 넘어가면 clip
                pred_u16 = np.clip(np.round(pred_raw), 0, 65535).astype(np.uint16)

                d_pred_u16[:, :, t-1] = pred_u16

        if hf is not None:
            hf.flush(); hf.close()

        with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump({
                "file": fp,
                "frames_total": int(T),
                "frames_used": int(T_eff),
                "sparse_source": args.sparse_source,
                "sidx_count": int(len(sidx)),
                "prefill_mode": args.prefill_mode,
                "bp_used": bool(args.use_bp),
                "saved": {
                    "sino_pred_u16_raw_units": bool(args.save_pred_mat)
                }
            }, f, indent=2, ensure_ascii=False)

    print("[DONE] All test files processed.")

# ---------------------- Run with defaults (no args needed) ----------------------
if __name__ == "__main__":
    main()
