"""
DL2 (SLOT): Temporal sinogram restoration (frame interpolation).

This script trains a 3D U-Net in the sinogram domain to restore missing frames
that occur when the laser repetition rate is reduced (e.g., 100 Hz -> 20 Hz).

High-level behavior (kept from the original research code):
- Loads a contiguous window of frames per file into RAM, then slices it into
  (T_chunk) temporal blocks for training.
- Input channels: [sparse_prefill, mask], Output: full sinogram block.
- Loss: sinogram-domain terms + optional BP-SoftMIP after a warm-up period.
- Logs metrics on missing frames only (PSNR/L1) to avoid bias from measured frames.

This repository does NOT include the experimental data (.mat). See README for the
expected folder layout.
"""

import os, warnings, json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

# ============================= HDF5 / MAT I/O =============================
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
        dims=np.array(shape)
        idxH=int(np.argmin(np.abs(dims-H_target)))
        idxW=int(np.argmin(np.abs(dims-W_target)))
        if idxH==idxW: raise RuntimeError(f"Cannot determine H/W axes from shape {shape}")
    idxF=[i for i in axes if i not in (idxH,idxW)][0]
    return idxH,idxW,idxF

def load_frame_496x512(path: str, frame_1based: int, var_hint: str = None) -> np.ndarray:
    """Per-frame loader for legacy v7 .mat files (prefer block loading when possible)."""
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
    raise RuntimeError(f"{path}: failed to load frame {frame_1based}")

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
            return int(v.shape[2])
    raise RuntimeError(f"{path}: Could not find a 3D array in the .mat file.")

def _load_block_496x512(path: str, start_1based: int, count: int, var_hint: str=None) -> np.ndarray:
    """Load a contiguous block with shape (H=496, W=512, T=count).

v7.3 (.mat/HDF5): uses dataset slicing.
v7 (.mat): falls back to per-frame loading.
"""
    t0 = int(start_1based) - 1
    try:
        import h5py
        with h5py.File(path,"r") as f:
            dname = var_hint if (var_hint and var_hint in f) else _h5_find_dataset(f)
            ds=f[dname]
            idxH,idxW,idxF=_reorder_axes_to_HWF(ds.shape)
            if idxF==2: arr = np.array(ds[:,:,t0:t0+count])                     # (H,W,T)
            elif idxF==1: arr = np.array(ds[:,t0:t0+count,:]).transpose(0,2,1)   # (H,W,T)
            else:         arr = np.array(ds[t0:t0+count,:,:]).transpose(1,2,0)   # (H,W,T)
            if arr.shape[:2] == (512,496): arr = arr.transpose(1,0,2)
            if arr.shape[:2] != (496,512): raise RuntimeError(f"{path}: block shape {arr.shape}")
            return arr.astype(np.float32)
    except Exception:
        frames=[load_frame_496x512(path, t) for t in range(start_1based, start_1based+count)]
        return np.stack(frames, axis=-1).astype(np.float32)

from collections import OrderedDict
class _RAMWindowCache(OrderedDict):
    """(path, start, count) → np.float32 (H,W,Twin)"""
    def __init__(self, max_items=4): super().__init__(); self.max_items=max_items
    def get_or_load(self, key, loader):
        if key in self:
            self.move_to_end(key); return self[key]
        val = loader()
        self[key]=val
        if len(self)>self.max_items: self.popitem(last=False)
        return val
_RAM = _RAMWindowCache(max_items=4)

# ============================= BP / pos_xyz =============================
@torch.no_grad()
def _fft_bandpass(sig_mat: torch.Tensor, fs: float, f_lo: float, f_hi: float) -> torch.Tensor:
    Nt, Nc = sig_mat.shape
    S = torch.fft.fft(sig_mat, dim=0)
    f = torch.arange(Nt, device=sig_mat.device, dtype=torch.float32) * (fs / Nt)
    H = ((f >= float(f_lo)) & (f <= float(f_hi))).to(sig_mat.dtype)
    H = H + torch.flip(H, dims=[0])
    return torch.fft.ifft(S * H[:, None], dim=0).real

def backproject_gpu_matchCPU(sig_mat: torch.Tensor,
                             recon_dims: Tuple[float,float,float],
                             resolution: Tuple[int,int,int],
                             pos_sensor_xyz: torch.Tensor,
                             c0: float, fs: float, t0_offset: int,
                             f_filter: Tuple[float,float],
                             chan_block: int = 128, z_slab: int = 10) -> torch.Tensor:
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

def softmip(vol: torch.Tensor, tau: float = 0.08, dim: int = 2) -> torch.Tensor:
    w=torch.softmax(vol/tau,dim=dim)
    return (w*vol).sum(dim=dim)

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

def pick_bp_meta_by_filename(mat_path: str, pos_xyz_dir: str, pos_xyz_var: Optional[str]) -> Dict[str, Any]:
    name = Path(mat_path).stem.lower()
    D = Path(pos_xyz_dir)
    meta: Dict[str, Any] = {
        "c0": 1505.0,
        "bp_dims": (7e-3, 7e-3, 5e-3),
        "bp_res":  (175, 175, 125),
        "bp_filter": (0.2e6, 8e6),
        "bp_fs": 40e6, "bp_t0": 832,
        "chan_block": 128, "z_slab": 10, "tag": "ultracup"
    }
    if "holycup" in name or "holocup" in name:
        meta.update({"c0":1500.0,"bp_dims":(10e-3,10e-3,10e-3),"bp_res":(128,128,128),"bp_filter":(0.1e6,6e6)})
        if "40 msps" in name: meta.update({"bp_fs":40e6,"bp_t0":832,"tag":"holycup_40msps"})
        else: meta.update({"bp_fs":24e6,"bp_t0":501,"tag":"holycup_24msps"})
    pfile = D / ("pos_sensor_xyz_holycup.mat" if ("holycup" in name or "holocup" in name) else "pos_sensor_xyz_ultracup.mat")
    if not pfile.is_file(): raise FileNotFoundError(f"pos_xyz file not found: {pfile}")
    meta["pos_xyz"] = load_pos_sensor_xyz(str(pfile), pos_xyz_var).astype(np.float32)
    return meta

# ============================= Dataset (Temporal Blocks) =============================
def make_time_mask(T: int, ratio: int, offset: int = 0) -> np.ndarray:
    m = np.zeros(T, dtype=np.float32); m[offset::ratio] = 1.0; return m

def robust_norm_from_measured(vol: np.ndarray, mask_t: np.ndarray, p_low=2.0, p_high=98.0) -> np.ndarray:
    vals = vol[:,:,mask_t>0].reshape(-1)
    lo = np.percentile(vals, p_low); hi = np.percentile(vals, p_high); hi = max(hi, lo+1e-6)
    return np.clip((vol - lo)/(hi-lo), 0, 1).astype(np.float32)

class TemporalBlockDataset(Dataset):
    """
    - Caches a per-file window [start .. start+use_frames-1] in RAM.
    - Slices it into samples using (T_chunk, stride_T).
    """
    def __init__(self, root_dir: str,
                 sparse_ratio=2, offset=0,
                 start_frame_1based=5001, use_frames=200,
                 T_chunk=32, stride_T=32,
                 norm_from_measured=True, p_low=2.0, p_high=98.0,
                 pos_xyz_dir: str = "", pos_xyz_var: Optional[str] = None,
                 random_offset: bool = False):
        self.paths = sorted([str(p) for p in Path(root_dir).rglob("*.mat") if p.is_file() and not p.name.startswith("._")])
        assert len(self.paths)>0, f"No .mat under {root_dir}"
        self.sparse_ratio=int(sparse_ratio); self.offset=int(offset)
        self.random_offset=bool(random_offset)
        self.start=int(start_frame_1based); self.use_frames=int(use_frames)
        self.T_chunk=int(T_chunk); self.stride_T=int(stride_T)
        self.norm_from_measured=bool(norm_from_measured); self.p_low=p_low; self.p_high=p_high
        self.H,self.W=496,512

        self.file_meta: Dict[int, Dict[str, Any]] = {}
        self.indices=[]
        for fi, p in enumerate(self.paths):
            Tfile = get_num_frames(p)
            s0=max(1,self.start); e0=min(Tfile, s0 + self.use_frames - 1)
            if e0 - s0 + 1 < self.T_chunk: continue
            self.file_meta[fi] = pick_bp_meta_by_filename(p, pos_xyz_dir, pos_xyz_var)
            t0 = s0
            while t0 + self.T_chunk - 1 <= e0:
                self.indices.append((fi, t0))
                t0 += self.stride_T

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        fi, t0_abs = self.indices[i]
        path = self.paths[fi]

        # ---- RAM window load (once per file) ----
        key=(path, self.start, self.use_frames)
        win = _RAM.get_or_load(key, lambda: _load_block_496x512(path, self.start, self.use_frames))  # (H,W,Twin)

        rel = t0_abs - self.start
        vol = win[:, :, rel:rel+self.T_chunk]  # (H,W,Tc)

        # mask & normalization
        offset = np.random.randint(0, self.sparse_ratio) if self.random_offset else self.offset
        mask_t = make_time_mask(self.T_chunk, self.sparse_ratio, offset)  # (Tc,)
        if self.norm_from_measured and mask_t.sum()>0:
            vol_n = robust_norm_from_measured(vol, mask_t, self.p_low, self.p_high)
        else:
            lo,hi=vol.min(),vol.max(); hi=max(hi,lo+1e-6); vol_n=((vol-lo)/(hi-lo)).astype(np.float32)

        sparse = np.zeros_like(vol_n, dtype=np.float32)
        sparse[:,:,mask_t>0] = vol_n[:,:,mask_t>0]
        prefilled = sparse.copy()  # zero prefill

        x  = torch.from_numpy(prefilled).unsqueeze(0)           # (1,H,W,T)
        m  = torch.from_numpy(mask_t[None,None,None,:])         # (1,1,1,T)
        y  = torch.from_numpy(vol_n).unsqueeze(0)               # (1,H,W,T)
        meta = self.file_meta[fi]
        return x, m, y, meta

# ============================= Model (3D U-Net) =============================
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

# ============================= Losses =============================
class SinoTemporalLoss(nn.Module):
    def __init__(self, w_obs=2.0, w_miss=1.0, w_grad=0.1):
        super().__init__(); self.w_obs=w_obs; self.w_miss=w_miss; self.w_grad=w_grad
    @staticmethod
    def grad_t(x):
        return x[..., 1:] - x[..., :-1]
    def forward(self, pred, gt, mask):
        L_obs = F.l1_loss(pred*mask, gt*mask)
        L_miss= F.l1_loss(pred*(1-mask), gt*(1-mask))
        L_g   = F.l1_loss(self.grad_t(pred), self.grad_t(gt))
        return self.w_obs*L_obs + self.w_miss*L_miss + self.w_grad*L_g, {"L_obs":L_obs.item(),"L_miss":L_miss.item(),"L_grad":L_g.item()}

# ============================= Visualization & Metrics =============================
def pick_vis_index_from_mask(mask_t: torch.Tensor, prefer_missing: bool = True) -> int:
    """
    mask_t: (B,1,1,1,T) or (1,1,1,T) or any broadcastable shape
    -> Flatten to a 1D boolean mask (T,) and pick the frame closest to the center (prefer missing if requested).
    """
    m = mask_t
    if not isinstance(m, torch.Tensor):
        m = torch.as_tensor(m)
    m = (m > 0)

    m = m.flatten()

    T = int(m.numel())
    if T == 0:
        raise RuntimeError("pick_vis_index_from_mask: empty mask")

    device = m.device
    idx = torch.arange(T, device=device)
    center = T // 2

    if prefer_missing and (~m).any():
        miss = idx[~m]
        tvis = miss[torch.argmin(torch.abs(miss - center))]
        return int(tvis.item())

    if m.any():
        meas = idx[m]
        tvis = meas[torch.argmin(torch.abs(meas - center))]
        return int(tvis.item())

    return 0


@torch.no_grad()
def save_triplet_png_temporal(x, m, pred, gt, out_path: str, prefer_missing: bool = True):
    tvis = pick_vis_index_from_mask(m, prefer_missing=prefer_missing)
    T = pred.shape[-1]
    tvis = max(0, min(int(tvis), T-1))

    xin = x[0,0,:,:,tvis].unsqueeze(0).unsqueeze(0)
    xpr = pred[0,0,:,:,tvis].unsqueeze(0).unsqueeze(0)
    xgt = gt[0,0,:,:,tvis].unsqueeze(0).unsqueeze(0)
    lo,hi = xgt.min(), xgt.max()
    sc = lambda t: ((t-lo)/(hi-lo+1e-6)).clamp(0,1)
    grid = torch.cat([sc(xin), sc(xpr), sc(xgt)], dim=0)
    save_image(grid.cpu(), out_path, nrow=3, padding=0)

def _as_pos_tensor_any(pos_xyz_any, device):
    if isinstance(pos_xyz_any, torch.Tensor):
        t = pos_xyz_any.to(device).float()
    else:
        arr = np.asarray(pos_xyz_any)
        t = torch.from_numpy(arr).to(device).float()
    if t.ndim == 3 and t.shape[0] == 1:
        t = t.squeeze(0)          # (1,Nc,3) -> (Nc,3)
    if t.ndim != 2 or t.shape[1] != 3:
        raise RuntimeError(f"pos_xyz shape {tuple(t.shape)} invalid; expected (Nc,3)")
    return t

@torch.no_grad()
def mip_from_frame(frame_2d: torch.Tensor, meta: Dict[str,Any]) -> torch.Tensor:
    device = frame_2d.device
    s_used = frame_2d[3:, :]                        # (Nt-3, Nc)
    pos = _as_pos_tensor_any(meta["pos_xyz"], device)  # (Nc,3)

    Nc_frame = s_used.shape[1]
    if pos.shape[0] != Nc_frame:
        raise RuntimeError(f"pos_xyz Nc({pos.shape[0]}) != frame Nc({Nc_frame}); "
                           f"check pos file & column pairing")

    vol = backproject_gpu_matchCPU(
        s_used,
        tuple(meta["bp_dims"]), tuple(meta["bp_res"]),
        pos,
        float(meta["c0"]), float(meta["bp_fs"]), int(meta["bp_t0"]) + 3,
        tuple(meta["bp_filter"]), int(meta["chan_block"]), int(meta["z_slab"])
    )
    return softmip(vol.unsqueeze(0), tau=0.08, dim=3).unsqueeze(1)

@torch.no_grad()
def save_epoch_pngs(split_name: str, out_dir: Path, x, m, pred, gt, meta: Dict[str,Any], save_measured_too: bool = True):
    if isinstance(meta, (list, tuple)):
        meta = meta[0]

    out_img_miss = out_dir / f"{split_name}_miss.png"
    out_mip_miss = out_dir / f"{split_name}_miss_mip.png"
    save_triplet_png_temporal(x, m, pred, gt, str(out_img_miss), prefer_missing=True)

    tvis_miss = pick_vis_index_from_mask(m, prefer_missing=True)
    xin = x[0,0,:,:,tvis_miss]; xpr = pred[0,0,:,:,tvis_miss]; xgt = gt[0,0,:,:,tvis_miss]
    mip_in = mip_from_frame(xin, meta)
    mip_pr = mip_from_frame(xpr, meta)
    mip_gt = mip_from_frame(xgt, meta)
    lo, hi = mip_gt.min(), mip_gt.max()
    sc = lambda t: ((t - lo) / (hi - lo + 1e-6)).clamp(0, 1)
    grid = torch.cat([sc(mip_in), sc(mip_pr), sc(mip_gt)], dim=0)
    save_image(grid.cpu(), str(out_mip_miss), nrow=3, padding=0)

    if save_measured_too:
        out_img_meas = out_dir / f"{split_name}_meas.png"
        out_mip_meas = out_dir / f"{split_name}_meas_mip.png"
        save_triplet_png_temporal(x, m, pred, gt, str(out_img_meas), prefer_missing=False)

        tvis_meas = pick_vis_index_from_mask(m, prefer_missing=False)
        xin2 = x[0,0,:,:,tvis_meas]; xpr2 = pred[0,0,:,:,tvis_meas]; xgt2 = gt[0,0,:,:,tvis_meas]
        mip_in2 = mip_from_frame(xin2, meta)
        mip_pr2 = mip_from_frame(xpr2, meta)
        mip_gt2 = mip_from_frame(xgt2, meta)
        lo2, hi2 = mip_gt2.min(), mip_gt2.max()
        sc2 = lambda t: ((t - lo2) / (hi2 - lo2 + 1e-6)).clamp(0, 1)
        grid2 = torch.cat([sc2(mip_in2), sc2(mip_pr2), sc2(mip_gt2)], dim=0)
        save_image(grid2.cpu(), str(out_mip_meas), nrow=3, padding=0)

@torch.no_grad()
def missing_only_metrics(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    """
    pred, gt: (B,1,H,W,T)
    mask:     (B,1,1,1,T) or any broadcastable shape -> expand to (B,1,H,W,T) and compute metrics on missing frames only
    """
    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask, device=pred.device)
    mask = (mask > 0)  # bool

    miss_t = ~mask

    while miss_t.ndim < pred.ndim:
        miss_t = miss_t.unsqueeze(2)
    miss_mask_full = miss_t.expand_as(pred)  # (B,1,H,W,T)

    num = int(miss_mask_full.sum().item())
    if num == 0:
        return {"psnr": float('nan'), "l1": float('nan')}

    p = pred[miss_mask_full]
    g = gt[miss_mask_full]
    if p.numel() == 0:
        return {"psnr": float('nan'), "l1": float('nan')}

    mse = ((p-g)**2).mean().item()
    psnr = 10*np.log10(1.0/(mse+1e-12))
    l1 = (p-g).abs().mean().item()
    return {"psnr": psnr, "l1": l1}

# ============================= Baseline: Temporal Linear Interp =============================
@torch.no_grad()
def baseline_linear_interp_from_sparse(y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Temporal linear-interpolation baseline (uses observed frames only).
    y    : (B,1,H,W,T) - GT scale; baseline uses y * mask (observations only).
    mask : (B,1,1,1,T) - observed(1) / missing(0)
    return: (B,1,H,W,T)
    """
    assert y.ndim == 5 and mask.ndim >= 2
    B, C, H, W, T = y.shape
    device = y.device

    m = (mask > 0).to(y.dtype)                     # (B,1,1,1,T)
    sparse = y * m

    t_idx = torch.arange(T, device=device).view(1, 1, 1, 1, T).to(y.dtype)

    prev_idx = torch.zeros((B, 1, 1, 1, T), dtype=torch.long, device=device)
    next_idx = torch.zeros((B, 1, 1, 1, T), dtype=torch.long, device=device)

    m_t = (mask > 0).flatten(-1)
    m_t = m_t.view(B, 1, 1, 1, T)

    last_obs = torch.zeros((B,1,1,1), dtype=torch.long, device=device)
    for t in range(T):
        is_obs = m_t[..., t]
        last_obs = torch.where(is_obs, torch.full_like(last_obs, t), last_obs)
        prev_idx[..., t] = last_obs

    next_obs = torch.full((B,1,1,1), T-1, dtype=torch.long, device=device)
    for t in reversed(range(T)):
        is_obs = m_t[..., t]
        next_obs = torch.where(is_obs, torch.full_like(next_obs, t), next_obs)
        next_idx[..., t] = next_obs

    prev_f = prev_idx.to(y.dtype)
    next_f = next_idx.to(y.dtype)
    denom = (next_f - prev_f).clamp(min=1)
    alpha = (t_idx - prev_f) / denom        # (B,1,1,1,T) in [0,1]

    gather_prev = prev_idx.expand(B, C, H, W, T)
    gather_next = next_idx.expand(B, C, H, W, T)

    y_prev = torch.gather(sparse, dim=-1, index=gather_prev)
    y_next = torch.gather(sparse, dim=-1, index=gather_next)

    base = (1 - alpha) * y_prev + alpha * y_next
    return base

@torch.no_grad()
def baseline_metrics_and_png(x, m, y, meta, out_dir: Path, tag: str = "baseline"):
    """
    Baseline -> missing-only metrics -> visualization (including BP-SoftMIP)
    x, m, y, meta are the same tensors used in the train/val loops.
    """
    base = baseline_linear_interp_from_sparse(y, m)

    mx = missing_only_metrics(base, y, m)
    print(f"[{tag}] miss-PSNR {mx['psnr']:.2f} dB | miss-L1 {mx['l1']:.4f}")

    out_img = out_dir / f"{tag}_miss.png"
    out_mip = out_dir / f"{tag}_miss_mip.png"

    tvis = pick_vis_index_from_mask(m, prefer_missing=True)
    T = y.shape[-1]
    tvis = max(0, min(int(tvis), T-1))

    xin = x[0,0,:,:,tvis].unsqueeze(0).unsqueeze(0)
    xbs = base[0,0,:,:,tvis].unsqueeze(0).unsqueeze(0)
    xgt = y[0,0,:,:,tvis].unsqueeze(0).unsqueeze(0)
    lo,hi = xgt.min(), xgt.max(); sc=lambda t: ((t-lo)/(hi-lo+1e-6)).clamp(0,1)
    grid = torch.cat([sc(xin), sc(xbs), sc(xgt)], dim=0)
    save_image(grid.cpu(), str(out_img), nrow=3, padding=0)

    mip_in = mip_from_frame(x[0,0,:,:,tvis],    meta)
    mip_bs = mip_from_frame(base[0,0,:,:,tvis], meta)
    mip_gt = mip_from_frame(y[0,0,:,:,tvis],    meta)
    lo,hi = mip_gt.min(), mip_gt.max()
    sc2 = lambda t: ((t - lo) / (hi - lo + 1e-6)).clamp(0, 1)
    grid2 = torch.cat([sc2(mip_in), sc2(mip_bs), sc2(mip_gt)], dim=0)
    save_image(grid2.cpu(), str(out_mip), nrow=3, padding=0)

    return mx

# ============================= Train Loop =============================
def list_mat_files(root: str) -> List[str]:
    root=Path(root)
    return sorted([str(p) for p in root.rglob("*.mat") if p.is_file() and not p.name.startswith("._")])

def train(train_root: str, val_root: str, test_root: str, out_dir: str,
          pos_xyz_dir: str, pos_xyz_var: Optional[str],
          # frame slicing
          train_start=5001, val_start=15001, test_start=15001,
          train_use_frames=200, val_use_frames=200, test_use_frames=200,
          # temporal chunks
          T_chunk=32, stride_T=32,
          # sparsity
          sparse_ratio=2, offset=0,
          # training
          epochs=50, bs=1, lr=2e-4, workers=0, seed=42, weight_decay=0.0,
          # model
          base_ch=16,
          # losses
          w_obs=2.0, w_miss=1.0, w_grad=0.1,
          # BP
          use_bp=True, bp_Kframes=1, bp_warmup_epochs=3,
          # aug
          random_offset=False):

    torch.manual_seed(seed); np.random.seed(seed)
    out = Path(out_dir); (out/"ckpt").mkdir(parents=True, exist_ok=True)
    (out/"samples"/"train").mkdir(parents=True, exist_ok=True)
    (out/"samples"/"val").mkdir(parents=True, exist_ok=True)
    (out/"samples"/"test").mkdir(parents=True, exist_ok=True)

    tr_ds = TemporalBlockDataset(train_root, sparse_ratio, offset, train_start, train_use_frames,
                                 T_chunk, stride_T, pos_xyz_dir=pos_xyz_dir, pos_xyz_var=pos_xyz_var,
                                 random_offset=random_offset)
    va_ds = TemporalBlockDataset(val_root,   sparse_ratio, offset, val_start,   val_use_frames,
                                 T_chunk, stride_T, pos_xyz_dir=pos_xyz_dir, pos_xyz_var=pos_xyz_var,
                                 random_offset=False)
    te_ds = TemporalBlockDataset(test_root,  sparse_ratio, offset, test_start,  test_use_frames,
                                 T_chunk, stride_T, pos_xyz_dir=pos_xyz_dir, pos_xyz_var=pos_xyz_var,
                                 random_offset=False)

    train_loader=DataLoader(tr_ds, batch_size=bs, shuffle=True,
                            num_workers=workers, pin_memory=False, drop_last=True)
    val_loader  =DataLoader(va_ds, batch_size=1, shuffle=False,
                            num_workers=workers, pin_memory=False, drop_last=False)
    test_loader =DataLoader(te_ds, batch_size=1, shuffle=False,
                            num_workers=workers, pin_memory=False)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(in_ch=2, base=base_ch).to(device)
    sino_loss = SinoTemporalLoss(w_obs, w_miss, w_grad)

    opt=torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5, verbose=True)

    best_val=float('inf'); best_ep=-1

    for ep in range(1, epochs+1):
        use_bp_now = (use_bp and (ep > bp_warmup_epochs))

        # ---------------- Train ----------------
        model.train(); trL=[]; trPSNR=[]; trL1=[]
        for x, m, y, meta in tqdm(train_loader, desc=f"Epoch {ep}/{epochs} [Train]"):
            x=x.to(device, non_blocking=True)        # (B,1,H,W,T)
            m=m.to(device, non_blocking=True)        # (B,1,1,1,T)
            y=y.to(device, non_blocking=True)        # (B,1,H,W,T)
            inp = torch.cat([x, m.expand(-1,1,x.shape[2],x.shape[3],x.shape[4])], dim=1)  # (B,2,H,W,T)

            pred = model(inp)                         # (B,1,H,W,T)
            Ls, _ = sino_loss(pred, y, m)

            if use_bp_now and bp_Kframes>0:
                B,_,H,W,T = pred.shape
                K = min(bp_Kframes, T)
                t_idx = torch.linspace(0, T-1, steps=K, dtype=torch.long, device=device)
                lb_list=[]
                for b in range(B):
                    meta_b = meta
                    for tsel in t_idx.tolist():
                        xpr = pred[b,0,:,:,tsel]
                        xgt = y[b,0,:,:,tsel]
                        mip_pr = mip_from_frame(xpr, meta_b)
                        mip_gt = mip_from_frame(xgt, meta_b)
                        lo,hi = mip_gt.min(), mip_gt.max()
                        mip_pr = (mip_pr-lo)/(hi-lo+1e-6); mip_gt=(mip_gt-lo)/(hi-lo+1e-6)
                        lb_list.append(F.mse_loss(mip_pr, mip_gt))
                Lbp = torch.stack(lb_list).mean() if lb_list else 0.0*Ls
            else:
                Lbp = 0.0*Ls

            loss = Ls + Lbp
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            trL.append(float(loss.item()))

            mx = missing_only_metrics(pred.detach(), y.detach(), m)
            if not np.isnan(mx["psnr"]): trPSNR.append(mx["psnr"])
            if not np.isnan(mx["l1"]):   trL1.append(mx["l1"])

        tr_mean = float(np.mean(trL)) if trL else float('nan')
        tr_psnr = float(np.mean(trPSNR)) if trPSNR else float('nan')
        tr_l1   = float(np.mean(trL1))   if trL1   else float('nan')

        # ---------------- Val ----------------
        model.eval(); vaL=[]; vaPSNR=[]; vaL1=[]
        with torch.no_grad():
            for x, m, y, meta in tqdm(val_loader, desc=f"Epoch {ep}/{epochs} [Val]"):
                x=x.to(device); m=m.to(device); y=y.to(device)
                inp=torch.cat([x, m.expand(-1,1,x.shape[2],x.shape[3],x.shape[4])], dim=1)
                pred = model(inp)
                Ls,_ = sino_loss(pred, y, m)
                if use_bp_now:
                    tsel = pick_vis_index_from_mask(m, prefer_missing=True)
                    mip_pr = mip_from_frame(pred[0,0,:,:,tsel], meta)
                    mip_gt = mip_from_frame(y[0,0,:,:,tsel],   meta)
                    lo,hi=mip_gt.min(),mip_gt.max()
                    mip_pr=(mip_pr-lo)/(hi-lo+1e-6); mip_gt=(mip_gt-lo)/(hi-lo+1e-6)
                    Lbp = F.mse_loss(mip_pr, mip_gt)
                else:
                    Lbp = 0.0*Ls
                vaL.append(float((Ls+Lbp).item()))

                mx = missing_only_metrics(pred, y, m)
                if not np.isnan(mx["psnr"]): vaPSNR.append(mx["psnr"])
                if not np.isnan(mx["l1"]):   vaL1.append(mx["l1"])

            xb, mb, yb, metab = next(iter(train_loader))
            xb = xb.to(device);
            mb = mb.to(device);
            yb = yb.to(device)
            pb = model(torch.cat([xb, mb.expand(-1, 1, xb.shape[2], xb.shape[3], xb.shape[4])], dim=1))
            save_epoch_pngs(f"epoch_{ep:03d}", out / "samples" / "train",
                            xb, mb, pb, yb, metab, save_measured_too=True)
            # -> baseline (train)
            baseline_metrics_and_png(xb, mb, yb, metab, out / "samples" / "train",
                                     tag=f"epoch_{ep:03d}_baseline_train")

            xv, mv, yv, metav = next(iter(val_loader))
            xv = xv.to(device);
            mv = mv.to(device);
            yv = yv.to(device)
            pv = model(torch.cat([xv, mv.expand(-1, 1, xv.shape[2], xv.shape[3], xv.shape[4])], dim=1))
            save_epoch_pngs(f"epoch_{ep:03d}", out / "samples" / "val",
                            xv, mv, pv, yv, metav, save_measured_too=True)
            # -> baseline (val)
            baseline_metrics_and_png(xv, mv, yv, metav, out / "samples" / "val",
                                     tag=f"epoch_{ep:03d}_baseline_val")

            xt, mt, yt, metat = next(iter(test_loader))
            xt = xt.to(device);
            mt = mt.to(device);
            yt = yt.to(device)
            pt = model(torch.cat([xt, mt.expand(-1, 1, xt.shape[2], xt.shape[3], xt.shape[4])], dim=1))
            save_epoch_pngs(f"epoch_{ep:03d}", out / "samples" / "test",
                            xt, mt, pt, yt, metat, save_measured_too=True)
            # -> baseline (test)
            baseline_metrics_and_png(xt, mt, yt, metat, out / "samples" / "test",
                                     tag=f"epoch_{ep:03d}_baseline_test")

        va_mean = float(np.mean(vaL)) if vaL else float('nan')
        va_psnr = float(np.mean(vaPSNR)) if vaPSNR else float('nan')
        va_l1   = float(np.mean(vaL1))   if vaL1   else float('nan')

        print(f"[Epoch {ep:03d}/{epochs}] TrainLoss {tr_mean:.4f} | ValLoss {va_mean:.4f} | "
              f"Train(miss) PSNR {tr_psnr:.2f}dB L1 {tr_l1:.4f} | "
              f"Val(miss) PSNR {va_psnr:.2f}dB L1 {va_l1:.4f}")
        sched.step(va_mean)

        # ckpt
        torch.save({"epoch":ep,"model":model.state_dict(),"val_loss":va_mean,"train_loss":tr_mean,
                    "train_miss_psnr":tr_psnr,"train_miss_l1":tr_l1,"val_miss_psnr":va_psnr,"val_miss_l1":va_l1},
                   out/"ckpt"/f"epoch_{ep:03d}.pt")
        if va_mean < best_val:
            best_val, best_ep = va_mean, ep
            torch.save({"epoch":ep,"model":model.state_dict(),"val_loss":best_val,"train_loss":tr_mean,
                        "train_miss_psnr":tr_psnr,"train_miss_l1":tr_l1,"val_miss_psnr":va_psnr,"val_miss_l1":va_l1},
                       out/"best.pt")
    torch.save({"best_epoch":best_ep,"best_val":best_val}, out/"last.pt")

# ============================= Main (Hard-coded run) =============================

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="DL2 (SLOT) training: temporal sinogram restoration (frame interpolation).")

    # Paths
    p.add_argument("--train", type=str, default="data/dl2/train", help="Training directory containing .mat files.")
    p.add_argument("--val",   type=str, default="data/dl2/val",   help="Validation directory containing .mat files.")
    p.add_argument("--test",  type=str, default="data/dl2/test",  help="Test directory containing .mat files.")
    p.add_argument("--out",   type=str, default="outputs/dl2",    help="Output directory (checkpoints, logs, samples).")

    # pos_sensor_xyz
    p.add_argument("--pos_xyz_dir", type=str, default="data/pos_sensor_xyz",
                   help="Directory containing pos_sensor_xyz_*.mat files.")
    p.add_argument("--pos_xyz_var", type=str, default=None, help="Variable name inside pos_sensor_xyz .mat (optional).")

    # Windowing / slicing
    p.add_argument("--train_start", type=int, default=1, help="1-based start frame for the per-file training window.")
    p.add_argument("--val_start",   type=int, default=1, help="1-based start frame for the per-file validation window.")
    p.add_argument("--test_start",  type=int, default=1, help="1-based start frame for the per-file test window.")
    p.add_argument("--train_use_frames", type=int, default=1000, help="Number of frames to load per file for training.")
    p.add_argument("--val_use_frames",   type=int, default=1000, help="Number of frames to load per file for validation.")
    p.add_argument("--test_use_frames",  type=int, default=1000, help="Number of frames to load per file for testing.")

    p.add_argument("--t_chunk",  type=int, default=32, help="Temporal block length (frames).")
    p.add_argument("--stride_t", type=int, default=32, help="Stride for sliding temporal blocks.")

    # Temporal sparsity (simulate low repetition rate)
    p.add_argument("--sparse_ratio", type=int, default=5, help="Keep 1 of every N frames.")
    p.add_argument("--offset", type=int, default=0, help="Offset for the keep pattern (0..ratio-1).")
    p.add_argument("--random_offset", action="store_true", help="Randomize offset per sample (augmentation).")

    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch",  type=int, default=1)
    p.add_argument("--lr",     type=float, default=2e-4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--weight_decay", type=float, default=0.0)

    # Model
    p.add_argument("--base_ch", type=int, default=16, help="Base channel count for 3D U-Net.")

    # Loss weights
    p.add_argument("--w_obs",  type=float, default=2.0, help="Loss weight on observed frames.")
    p.add_argument("--w_miss", type=float, default=1.0, help="Loss weight on missing frames.")
    p.add_argument("--w_grad", type=float, default=0.1, help="Spatial gradient regularization weight.")

    # Optional BP supervision
    p.add_argument("--use_bp", action="store_true", help="Enable BP-SoftMIP supervision.")
    p.add_argument("--bp_kframes", type=int, default=1, help="Number of frames used for BP loss within each block.")
    p.add_argument("--bp_warmup_epochs", type=int, default=3, help="Warm-up epochs before enabling BP loss.")

    args = p.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out) / "run_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    train(
        train_root=args.train, val_root=args.val, test_root=args.test, out_dir=args.out,
        pos_xyz_dir=args.pos_xyz_dir, pos_xyz_var=args.pos_xyz_var,
        train_start=args.train_start, val_start=args.val_start, test_start=args.test_start,
        train_use_frames=args.train_use_frames, val_use_frames=args.val_use_frames, test_use_frames=args.test_use_frames,
        T_chunk=args.t_chunk, stride_T=args.stride_t,
        sparse_ratio=args.sparse_ratio, offset=args.offset,
        epochs=args.epochs, bs=args.batch, lr=args.lr, workers=args.workers, seed=args.seed, weight_decay=args.weight_decay,
        base_ch=args.base_ch,
        w_obs=args.w_obs, w_miss=args.w_miss, w_grad=args.w_grad,
        use_bp=args.use_bp, bp_Kframes=args.bp_kframes, bp_warmup_epochs=args.bp_warmup_epochs,
        random_offset=args.random_offset
    )
