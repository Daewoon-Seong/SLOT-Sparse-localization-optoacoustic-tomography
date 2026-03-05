"""
DL2 (SLOT): Inference for temporal sinogram restoration (frame interpolation).

Runs the trained DL2 3D U-Net on sinogram stacks stored in .mat files and saves:
- a single v7.3 .mat (HDF5) file containing uint16 predictions in raw units, and
- optional per-frame debug PNGs.

This repository does NOT include the experimental data (.mat). See README for the
expected folder layout.
"""

import os, sys, json, warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.utils import save_image

# ---------------- I/O helpers (MAT/HDF5) ----------------
def _h5_find_dataset(f):
    import h5py
    cands=[]
    def visit(name,obj):
        try:
            if isinstance(obj,h5py.Dataset) and obj.ndim==3 and np.issubdtype(obj.dtype,np.number):
                cands.append(name)
        except Exception:
            pass
    f.visititems(lambda n,o: visit(n,o))
    if not cands:
        raise RuntimeError("Could not find a 3D numeric dataset in the HDF5 file.")
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
        if idxH==idxW:
            raise RuntimeError(f"Cannot determine H/W axes from shape {shape}")
    idxF=[i for i in axes if i not in (idxH,idxW)][0]
    return idxH,idxW,idxF

def get_num_frames(path: str) -> int:
    path=str(path)
    try:
        import h5py
        with h5py.File(path,"r") as f:
            for key in ("sino_pred_raw","sino_pred"):
                if key in f:
                    ds=f[key]; _,_,idxF=_reorder_axes_to_HWF(ds.shape)
                    return int(ds.shape[idxF])
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
            return int(v.shape[2])
    raise RuntimeError(f"{path}: Could not find a 3D array in the .mat file.")

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
    raise RuntimeError(f"{path}: failed to load frame {frame_1based}")

def _read_h5_block_reordered(ds, t0, count):
    idxH,idxW,idxF=_reorder_axes_to_HWF(ds.shape)
    if idxF==2: arr = np.array(ds[:,:,t0:t0+count])
    elif idxF==1: arr = np.array(ds[:,t0:t0+count,:]).transpose(0,2,1)
    else:         arr = np.array(ds[t0:t0+count,:,:]).transpose(1,2,0)
    if arr.shape[:2] == (512,496): arr = arr.transpose(1,0,2)
    if arr.shape[:2] != (496,512): raise RuntimeError(f"HDF5 block shape {arr.shape} != (496,512,*)")
    return arr.astype(np.float32)

def _load_block_496x512_RAW(path: str, start_1based: int, count: int, var_hint: str=None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Supports Stage-1 outputs + generic 3D raw loading"""
    t0 = int(start_1based) - 1
    try:
        import h5py
        with h5py.File(path, "r") as f:
            if "sino_pred_raw" in f:
                ds = f["sino_pred_raw"]
                arr = _read_h5_block_reordered(ds, t0, count)
                return arr.astype(np.float32), None
            if "sino_pred" in f and "norm_lo_hi" in f:
                ds = f["sino_pred"]
                arr_u16 = _read_h5_block_reordered(ds, t0, count).astype(np.float32)
                arr01   = np.clip(arr_u16 / 65535.0, 0.0, 1.0)
                lohi    = np.array(f["norm_lo_hi"][:, t0:t0+count], dtype=np.float32)
                lo, hi  = lohi[0], lohi[1]
                raw = np.empty_like(arr01, dtype=np.float32)
                for i in range(count):
                    if np.isfinite(lo[i]) and np.isfinite(hi[i]) and hi[i] > lo[i]:
                        raw[:,:,i] = arr01[:,:,i] * (hi[i]-lo[i]) + lo[i]
                    else:
                        raw[:,:,i] = arr01[:,:,i]
                return raw, lohi
            dname = var_hint if (var_hint and var_hint in f) else _h5_find_dataset(f)
            ds = f[dname]
            arr = _read_h5_block_reordered(ds, t0, count)
            return arr.astype(np.float32), None
    except Exception:
        frames=[load_frame_496x512(path, t) for t in range(start_1based, start_1based+count)]
        arr = np.stack(frames, axis=-1).astype(np.float32)
        return arr, None

def list_mat_files(root: str) -> List[str]:
    root=Path(root)
    return sorted([str(p) for p in root.rglob("*.mat") if p.is_file() and not p.name.startswith("._")])

# --------------- GLOBAL scaling helper ----------------
def _global_lohi_measured(fp: str, var_hint: Optional[str],
                          ratio: int, offset: int, step: int,
                          p_low: float, p_high: float) -> Tuple[float,float]:
    """Estimate global lo/hi (percentiles) by sampling observed frames at a fixed step"""
    T = get_num_frames(fp)
    vals_all = []
    sub = max(1, (496*512)//4096)
    for t in range(1, T+1):
        if ((t-1) % ratio) != (offset % ratio):  # measured only
            continue
        if ((t-1) // ratio) % max(1, step) != 0:
            continue
        blk, _ = _load_block_496x512_RAW(fp, t, 1, var_hint=var_hint)
        vals_all.append(blk[...,0].reshape(-1)[::sub].astype(np.float32))
    if not vals_all:
        blk, _ = _load_block_496x512_RAW(fp, 1, 1, var_hint=var_hint)
        vals_all = [blk[...,0].reshape(-1)[::sub].astype(np.float32)]
    vals = np.concatenate(vals_all, axis=0)
    lo = float(np.percentile(vals, p_low))
    hi = float(np.percentile(vals, p_high))
    if hi <= lo: hi = lo + 1e-6
    return lo, hi

# ---------------- Model ----------------
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

@torch.no_grad()
def _save_png_global(x_raw, pred_01, gt_raw, out_path: str, lo_g: float, hi_g: float):
    """
    Panel order: [Input (raw), Output (pred in raw), GT/Ref (raw)]
    - x_raw, gt_raw: tensors in raw units (1,1,H,W)
    - pred_01: 0..1 tensor (1,1,H,W) -> converted to raw for visualization
    """
    x  = x_raw.squeeze(0).squeeze(0)     # RAW
    pr = pred_01.squeeze(0).squeeze(0)   # 0..1
    gt = gt_raw.squeeze(0).squeeze(0)    # RAW

    device = pr.device
    lo = torch.tensor(lo_g, device=device, dtype=pr.dtype)
    hi = torch.tensor(hi_g, device=device, dtype=pr.dtype)

    pr_raw = pr * (hi - lo) + lo         # convert to raw units

    sc = lambda t: ((t - lo) / (hi - lo + 1e-6)).clamp(0, 1)
    grid = torch.stack([sc(x), sc(pr_raw), sc(gt)], dim=0).unsqueeze(1)  # (3,1,H,W)
    save_image(grid.detach().cpu(), out_path, nrow=3, padding=0)

# ---------------- main inference (streaming) ----------------
def run_infer_all(
    test_root: str,
    out_root: str,
    ckpt_path: str,
    T_chunk: int = 32, stride_T: int = 32,
    sparse_ratio: int = 5, offset: int = 0,
    p_low: float = 2.0, p_high: float = 98.0,
    base_ch: int = 16,
    var_hint: Optional[str] = None,
    save_png_every: int = 500,
    scale_step: int = 10,
):
    H,W=496,512
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_root = Path(test_root)
    out_root=Path(out_root); out_root.mkdir(parents=True, exist_ok=True)

    # model
    model = UNet3D(in_ch=2, base=base_ch).to(device)
    ckpt=torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.eval()

    # File collection: recurse if directory, otherwise treat as a single file
    if test_root.is_dir():
        mat_files = list_mat_files(str(test_root))
    elif test_root.is_file() and test_root.suffix.lower()==".mat":
        mat_files = [str(test_root)]
    else:
        raise FileNotFoundError(f"Check --test path: {test_root}")

    print(f"[TEST] files={len(mat_files)}")

    for fp in mat_files:
        fp_path = Path(fp)
        stem = fp_path.stem

        try:
            rel_parent = fp_path.parent.relative_to(test_root if test_root.is_dir() else fp_path.parent)
        except ValueError:
            # If relative-path computation fails, fall back to using only the filename
            rel_parent = Path("")

        out_dir = out_root / rel_parent / stem
        (out_dir/"frames").mkdir(parents=True, exist_ok=True)

        # Log input/output relative paths
        print(f"[IN ] {fp_path}")
        print(f"[OUT] {out_dir}")

        T = get_num_frames(fp)
        print(f"[{rel_parent / stem}] frames={T}")

        # GLOBAL lo/hi (measured-only)
        lo_g, hi_g = _global_lohi_measured(fp, var_hint=var_hint,
                                           ratio=sparse_ratio, offset=offset,
                                           step=scale_step, p_low=p_low, p_high=p_high)
        print(f"[{rel_parent / stem}] GLOBAL lo/hi (measured-only): {lo_g:.6f} / {hi_g:.6f}")

        import h5py
        h5_path = out_dir / "pred_sino_v7p3.mat"
        if h5_path.exists(): h5_path.unlink()
        hf = h5py.File(str(h5_path), "w")

        # Single dataset: uint16 (raw units)
        d_pred = hf.create_dataset(
            "sino_pred", shape=(H, W, T), dtype="uint16",
            chunks=(H, W, 1), compression="gzip", compression_opts=6, shuffle=True
        )
        d_pred.attrs["scale"] = "uint16_raw_units"
        d_pred.attrs["note"]  = "Saved with single global RAW scale (measured-only p2-p98)."

        assert stride_T == T_chunk, "For inference, stride_T == T_chunk (no overlap) is recommended"

        for t0 in tqdm(range(1, T+1, T_chunk), desc=str(rel_parent / stem)):
            count = min(T_chunk, T - (t0-1))
            if count <= 0: continue

            # LOAD RAW block
            vol_raw, _ = _load_block_496x512_RAW(fp, t0, count, var_hint=var_hint)  # (H,W,count)

            # TIME MASK (informational; used to build sparse input)
            global_idx = np.arange(t0-1, t0-1+count, dtype=np.int64)
            mask_t = np.zeros(count, dtype=np.float32)
            mask_t[(global_idx % sparse_ratio) == (offset % sparse_ratio)] = 1.0

            # GLOBAL normalize for inference
            vol_n = np.clip((vol_raw - lo_g)/(hi_g - lo_g), 0, 1).astype(np.float32)

            # MODEL INFERENCE
            sparse = np.zeros_like(vol_n, dtype=np.float32)
            sparse[:,:,mask_t>0] = vol_n[:,:,mask_t>0]
            x  = torch.from_numpy(sparse).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W,count)
            m  = torch.from_numpy(mask_t[None,None,None,:]).float().to(device)  # (1,1,1,count)
            inp= torch.cat([x, m.expand_as(x)], dim=1)
            with torch.no_grad():
                pred = model(inp)                                              # (1,1,H,W,count)
            pred_np01 = pred[0,0].detach().cpu().numpy().astype(np.float32)   # (H,W,count)

            if save_png_every > 0:
                tvis_local = int(count // 2)
                in_sparse_n = np.zeros_like(vol_n[:,:,tvis_local], dtype=np.float32)
                if mask_t[tvis_local] > 0:
                    in_sparse_n = vol_n[:,:,tvis_local]   # 0..1 (measured)
                else:
                    in_sparse_n = np.zeros_like(in_sparse_n)  # 0..1 (missing)

                x_vis_raw  = torch.from_numpy(in_sparse_n*(hi_g - lo_g) + lo_g).to(device).unsqueeze(0).unsqueeze(0)
                gt_vis_raw = torch.from_numpy(vol_raw[:,:,tvis_local]).to(device).unsqueeze(0).unsqueeze(0)
                pr_vis_01  = pred[0:1,0:1,:,:,tvis_local]  # 0..1

                _save_png_global(
                    x_vis_raw, pr_vis_01, gt_vis_raw,
                    str(out_dir/"frames"/f"{t0:06d}_{tvis_local:02d}_inSparse_pred_gt.png"),
                    lo_g, hi_g
                )

            # SAVE block → dataset (inverse using GLOBAL scale)
            t_slice = slice(t0-1, t0-1+count)
            pred_u16 = np.empty_like(pred_np01, dtype=np.uint16)
            for i in range(count):
                raw = pred_np01[:,:,i]*(hi_g - lo_g) + lo_g
                pred_u16[:,:,i] = np.clip(np.round(raw), 0, 65535).astype(np.uint16)
            d_pred[:,:,t_slice] = pred_u16

        hf.flush(); hf.close()

        # quick consistency check
        _quick_check_saved_mat(str(h5_path), sparse_ratio)

        with open(out_dir/"meta_infer.json","w",encoding="utf-8") as f:
            json.dump({
                "file": str(fp_path), "relative_parent": str(rel_parent),
                "frames": int(T),
                "sparse_ratio": int(sparse_ratio), "offset": int(offset),
                "T_chunk": int(T_chunk),
                "global_lo": float(lo_g), "global_hi": float(hi_g),
                "percentiles": [float(p_low), float(p_high)],
                "saved": {"sino_pred_u16_raw_units": True,
                          "png_dir": str((out_dir/'frames').resolve())}
            }, f, indent=2, ensure_ascii=False)

    print("[DONE] All test files processed.]")

# ---------- simple saved-mat consistency check (SAFE) ----------
def _quick_check_saved_mat(h5_path: str, sparse_ratio: int):
    import h5py
    import numpy as np
    from pathlib import Path
    try:
        with h5py.File(h5_path, "r") as f:
            if "sino_pred" not in f:
                print(f"[CHECK] {h5_path}: 'sino_pred' not found.")
                return
            ds = f["sino_pred"]  # (H,W,T) uint16
            H, W, T = ds.shape
            # Subsample (~4k points)
            target_pts = 4096
            step = max(1, int(np.sqrt((H*W)/target_pts)))
            means = np.empty((T,), dtype=np.float64)
            for t in range(T):
                frame = ds[:,:,t]
                if step > 1:
                    frame = frame[::step, ::step]
                means[t] = float(frame.mean())
            overall_std = float(means.std())
            bucket_std = []
            for r in range(sparse_ratio):
                bucket = means[r::sparse_ratio]
                bucket_std.append(float(bucket.std()) if bucket.size > 1 else float("nan"))
            print(f"[CHECK] {Path(h5_path).name}: frame_mean std={overall_std:.6f} | "
                  f"bucket stds (mod {sparse_ratio})={['%.6f' % x for x in bucket_std]}")
    except Exception as e:
        print(f"[CHECK] {h5_path}: error during check -> {e}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="DL2 (SLOT) inference: temporal sinogram restoration (frame interpolation).")
    p.add_argument("--test", type=str, default="data/dl2/test", help="Input .mat file or directory.")
    p.add_argument("--out",  type=str, default="outputs/dl2_infer", help="Output directory.")
    p.add_argument("--ckpt", type=str, default="outputs/dl2/best.pt", help="Path to model checkpoint (.pt).")

    p.add_argument("--t_chunk", type=int, default=32)
    p.add_argument("--stride_t", type=int, default=32)
    p.add_argument("--sparse_ratio", type=int, default=5, help="Keep 1 of every N frames.")
    p.add_argument("--offset", type=int, default=0, help="Offset for keep pattern (0..ratio-1).")

    p.add_argument("--p_low", type=float, default=2.0, help="Global percentile low for (raw -> 0..1) normalization.")
    p.add_argument("--p_high", type=float, default=98.0, help="Global percentile high for (raw -> 0..1) normalization.")

    p.add_argument("--base_ch", type=int, default=16)
    p.add_argument("--var_hint", type=str, default=None, help="Optional variable name inside .mat (for non-standard files).")

    p.add_argument("--save_png_every", type=int, default=500, help="Save a debug PNG every N frames (0 disables).")
    p.add_argument("--scale_step", type=int, default=10, help="Frame step used when estimating global percentiles.")
    p.add_argument("--run_sanity_check", action="store_true", help="Run a lightweight check on the saved .mat (debug).")

    args = p.parse_args()
    run_infer_all(
        test_root=args.test, out_root=args.out, ckpt_path=args.ckpt,
        T_chunk=args.t_chunk, stride_T=args.stride_t,
        sparse_ratio=args.sparse_ratio, offset=args.offset,
        p_low=args.p_low, p_high=args.p_high,
        base_ch=args.base_ch, var_hint=args.var_hint,
        save_png_every=args.save_png_every, scale_step=args.scale_step,
    )

    if args.run_sanity_check:
        # Optional: quick check on the saved HDF5 (debug)
        try:
            from pathlib import Path
            out_root = Path(args.out)
            # Find recently produced files (best-effort)
            mats = list(out_root.rglob("pred_sino_v7p3.mat"))
            for mpath in mats[:5]:
                _quick_check_saved_mat(str(mpath), sparse_ratio=args.sparse_ratio)
        except Exception as e:
            print(f"[WARN] sanity check failed: {e}")
