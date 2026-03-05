"""
DL3 (SLOT): Inference for low acquisition time 3D enhancement.

Loads a trained DL3 3D U-Net checkpoint and runs inference on input volumes.
Outputs are saved as .mat files (either v7 or v7.3/HDF5) and optional XY-MIP PNGs.

This repository does NOT include the experimental data (.mat/.h5). See README for the
expected folder layout.
"""

import os, glob, warnings
warnings.filterwarnings('ignore')

import numpy as np
import h5py
from scipy.io import loadmat, savemat

import torch
import torch.nn as nn
import torch.nn.functional as F

from skimage import io as skio
from matplotlib import cm as mpl_cm

# ======================
# Paths & options
# ======================
CONFIG = dict(    input_root = r'data/dl3/input/test',
    gt_root    = r'data/dl3/gt/test',
    workdir    = r'outputs/dl3',

    # Final output size (all predictions are saved at this size)
    target_out_size = (512, 512, 512),   # (D,H,W)

    # Save format
    save_format = 'mat7',                # 'mat7' | 'mat73'
    mat_varname = 'pred',

    # Optional outputs
    save_mip_png   = True,
    save_uint16png = False,
    save_E_GT_mat  = False,

    # MIP colormap (e.g., 'magma', 'inferno', 'viridis', 'gray')
    mip_colormap   = 'hot',

    # Debug: prefer CONFIG paths over paths stored in the checkpoint
    force_config_paths = True,
)

# Environment-variable overrides (optional)
CONFIG['input_root'] = os.environ.get('SLOT_DL3_INPUT_ROOT', CONFIG['input_root'])
CONFIG['gt_root']    = os.environ.get('SLOT_DL3_GT_ROOT',    CONFIG['gt_root'])
CONFIG['workdir']    = os.environ.get('SLOT_DL3_WORKDIR',    CONFIG['workdir'])

# ======================
# I/O & pre-processing
# ======================
def _largest_3d_from_h5(f: h5py.File):
    cand = None
    def visit(name, obj):
        nonlocal cand
        if isinstance(obj, h5py.Dataset) and len(obj.shape) == 3:
            if cand is None or np.prod(obj.shape) > np.prod(cand.shape):
                cand = obj
    f.visititems(visit)
    return cand

def load_volume_from_mat(path: str) -> np.ndarray:
    # Try v7.3(HDF5)
    try:
        with h5py.File(path, 'r') as ff:
            dset = _largest_3d_from_h5(ff)
            if dset is not None:
                return np.array(dset)
    except Exception:
        pass
    # Fallback v7
    md = loadmat(path)
    vol=None
    for k,v in md.items():
        if isinstance(v, np.ndarray) and v.ndim==3:
            if vol is None or v.size>vol.size:
                vol=v
    if vol is None:
        raise RuntimeError(f"No 3D array in {path}")
    return vol

def to_DHW(vol: np.ndarray) -> np.ndarray:
    # (H,W,Z) -> (D,H,W)
    assert vol.ndim==3
    return np.moveaxis(vol, -1, 0)

def from_DHW_to_HWZ(vol_DHW: np.ndarray) -> np.ndarray:
    # (D,H,W) -> (H,W,Z)
    return np.moveaxis(vol_DHW, 0, -1)

def resize_vol_trilinear(vol_DHW: np.ndarray, target) -> np.ndarray:
    v = torch.from_numpy(vol_DHW.astype(np.float32))[None, None, ...]
    v2 = F.interpolate(v, size=target, mode='trilinear', align_corners=False)
    return v2[0,0].cpu().numpy()

def xy_mip(vol_DHW: np.ndarray) -> np.ndarray:
    return vol_DHW.max(axis=0)  # (H,W)

def robust_norm(x, lo, hi, eps=1e-6):
    return np.clip((x - lo) / (hi - lo + eps), 0, 1).astype(np.float32)

def save_img(path, img01, uint16=True, cmap=None):
    """
    img01: 0..1 image (H,W)
    cmap:
      - None or 'gray' -> save grayscale (choose uint16 or 8-bit)
      - otherwise (e.g., 'magma') -> save RGB 8-bit color PNG (uint16 is ignored)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = np.clip(img01, 0, 1)

    if cmap is not None and str(cmap).lower() != 'gray':
        lut = mpl_cm.get_cmap(cmap)
        rgba = lut(img)                     # float32 HxWx4
        rgb  = (rgba[..., :3] * 255.0 + 0.5).astype(np.uint8)
        skio.imsave(path, rgb)
    else:
        if uint16:
            skio.imsave(path, (img*65535.0 + 0.5).astype(np.uint16))
        else:
            skio.imsave(path, (img*255.0 + 0.5).astype(np.uint8))

# ======================
# ======================
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

class UNet3D_E2GT(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        c=base
        self.e1=ConvBlock3D(1,c); self.d1=Down3D()
        self.e2=ConvBlock3D(c,2*c); self.d2=Down3D()
        self.e3=ConvBlock3D(2*c,4*c); self.d3=Down3D()
        self.b =ConvBlock3D(4*c,8*c)
        self.u3=Up3D(8*c,4*c); self.dec3=ConvBlock3D(8*c,4*c)
        self.u2=Up3D(4*c,2*c); self.dec2=ConvBlock3D(4*c,2*c)
        self.u1=Up3D(2*c,c);  self.dec1=ConvBlock3D(2*c,c)
        self.out=nn.Conv3d(c,1,1)
    def forward(self,x):
        e1=self.e1(x); x=self.d1(e1); e2=self.e2(x)
        x=self.d2(e2); e3=self.e3(x)
        x=self.d3(e3); x=self.b(x)
        x=self.u3(x); x=torch.cat([x,e3],dim=1); x=self.dec3(x)
        x=self.u2(x); x=torch.cat([x,e2],dim=1); x=self.dec2(x)
        x=self.u1(x); x=torch.cat([x,e1],dim=1); x=self.dec1(x)
        return torch.sigmoid(self.out(x))

# ======================
# Checkpoint
# ======================
def find_best_ckpt(ckpt_dir: str) -> str:
    best = os.path.join(ckpt_dir, 'best.pt')
    if os.path.exists(best):
        return best
    cands = sorted(glob.glob(os.path.join(ckpt_dir, 'ep*.pt')))
    if not cands:
        raise FileNotFoundError(f"No checkpoint in {ckpt_dir}")
    def epnum(p):
        b=os.path.basename(p)
        try: return int(b.replace('ep','').replace('.pt',''))
        except: return -1
    cands.sort(key=epnum)
    return cands[-1]

# ======================
# Inference
# ======================
@torch.no_grad()

def _parse_args_and_update_config() -> None:
    import argparse
    p = argparse.ArgumentParser(description="DL3 (SLOT) inference: low acquisition time 3D enhancement.")
    p.add_argument("--input_root", type=str, default=CONFIG["input_root"], help="Input volume root (file or directory).")
    p.add_argument("--gt_root", type=str, default=CONFIG["gt_root"], help="Optional GT root (file or directory).")
    p.add_argument("--workdir", type=str, default=CONFIG["workdir"], help="Workdir that contains checkpoints/.")

    p.add_argument("--target_out_size", type=int, nargs=3, default=list(CONFIG["target_out_size"]), metavar=("D","H","W"))
    p.add_argument("--save_format", type=str, default=CONFIG["save_format"], choices=["mat7","mat73"])
    p.add_argument("--mat_varname", type=str, default=CONFIG["mat_varname"])

    p.add_argument("--save_mip_png", action="store_true", default=CONFIG["save_mip_png"])
    p.add_argument("--save_uint16png", action="store_true", default=CONFIG["save_uint16png"])
    p.add_argument("--save_E_GT_mat", action="store_true", default=CONFIG["save_E_GT_mat"])
    p.add_argument("--mip_colormap", type=str, default=CONFIG.get("mip_colormap","hot"))

    p.add_argument("--force_config_paths", action="store_true", default=CONFIG.get("force_config_paths", True))

    args = p.parse_args()
    CONFIG["input_root"] = args.input_root
    CONFIG["gt_root"] = args.gt_root
    CONFIG["workdir"] = args.workdir
    CONFIG["target_out_size"] = tuple(int(x) for x in args.target_out_size)
    CONFIG["save_format"] = args.save_format
    CONFIG["mat_varname"] = args.mat_varname
    CONFIG["save_mip_png"] = args.save_mip_png
    CONFIG["save_uint16png"] = args.save_uint16png
    CONFIG["save_E_GT_mat"] = args.save_E_GT_mat
    CONFIG["mip_colormap"] = args.mip_colormap
    CONFIG["force_config_paths"] = args.force_config_paths


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_dir = os.path.join(CONFIG['workdir'], 'checkpoints')
    ckpt_path = find_best_ckpt(ckpt_dir)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    cfg_ckpt = ckpt.get('config', {})

    norm_mode  = ckpt.get('norm_mode', cfg_ckpt.get('norm_mode', 'pair_mip'))
    g_lo = ckpt.get('g_lo', None); g_hi = ckpt.get('g_hi', None)
    p_lo = float(cfg_ckpt.get('p_lo', 1.0)); p_hi = float(cfg_ckpt.get('p_hi', 99.7))
    target_in = tuple(cfg_ckpt.get('target_size', (256,256,256)))

    if CONFIG['force_config_paths']:
        input_root = CONFIG['input_root']; gt_root = CONFIG['gt_root']; workdir = CONFIG['workdir']
    else:
        input_root = cfg_ckpt.get('input_root', CONFIG['input_root'])
        gt_root    = cfg_ckpt.get('gt_root',    CONFIG['gt_root'])
        workdir    = cfg_ckpt.get('workdir',    CONFIG['workdir'])

    print(f"[INFO] ckpt   : {ckpt_path}")
    print(f"[INFO] device : {device}")
    print(f"[INFO] norm   : {norm_mode} (g_lo={g_lo}, g_hi={g_hi}, p_lo={p_lo}, p_hi={p_hi})")
    print(f"[INFO] in/out : model_in={target_in}, final_save={CONFIG['target_out_size']}")
    print(f"[INFO] roots  : input_root={input_root}  gt_root={gt_root}  workdir={workdir}")

    # model
    base_ch = int(cfg_ckpt.get('base_channels', 16))
    model = UNet3D_E2GT(base=base_ch).to(device)
    model.load_state_dict(ckpt['model']); model.eval()

    # list test mats (flat + recursive)
    in_dir = os.path.join(input_root, 'test')
    gt_dir = os.path.join(gt_root,   'test')

    mats = sorted(list({os.path.abspath(p)
                        for p in (glob.glob(os.path.join(in_dir, '*.[mM][aA][tT]')) +
                                  glob.glob(os.path.join(in_dir, '**', '*.[mM][aA][tT]'), recursive=True))}))
    if not mats:
        raise RuntimeError(f"No .mat under {in_dir}")

    rel_keys = [os.path.relpath(p, start=in_dir) for p in mats]
    def safe_stem(relp: str):
        s=os.path.splitext(relp)[0]
        return s.replace(os.sep,'__').replace('/','__')
    safe_stems = [safe_stem(r) for r in rel_keys]

    out_dir = os.path.join(workdir, 'infer_512')
    os.makedirs(out_dir, exist_ok=True)

    # quick GT presence
    have_gt = (len(glob.glob(os.path.join(gt_dir, '*.[mM][aA][tT]'))) > 0 or
               len(glob.glob(os.path.join(gt_dir, '**', '*.[mM][aA][tT]'), recursive=True)) > 0)

    for i,(mpath, rk, stem) in enumerate(zip(mats, rel_keys, safe_stems), 1):
        print(f"[{i}/{len(mats)}] {rk}", flush=True)
        try:
            # -------- load input / optional GT
            E_raw_hwz = load_volume_from_mat(mpath)      # (H,W,Z)
            E_raw = to_DHW(E_raw_hwz)                    # (D,H,W)

            G_raw = None
            if have_gt:
                try_gt = os.path.join(gt_dir, rk)
                if os.path.exists(try_gt):
                    G_raw = to_DHW(load_volume_from_mat(try_gt))
                else:
                    base = os.path.basename(rk)
                    cands = glob.glob(os.path.join(gt_dir, '**', base), recursive=True)
                    if not cands:
                        base2 = os.path.splitext(base)[0] + '.MAT'
                        cands = glob.glob(os.path.join(gt_dir, '**', base2), recursive=True)
                    if cands:
                        G_raw = to_DHW(load_volume_from_mat(cands[0]))

            # -------- resize to model input size
            E_small = resize_vol_trilinear(E_raw, target_in)
            if G_raw is not None:
                G_small = resize_vol_trilinear(G_raw, target_in)

            # -------- normalization (exactly like training)
            if norm_mode == 'pair_mip':
                if G_raw is not None:
                    mip = xy_mip(G_small)
                    lo = float(np.percentile(mip, p_lo)); hi = float(np.percentile(mip, p_hi))
                elif (g_lo is not None) and (g_hi is not None):
                    lo,hi = float(g_lo), float(g_hi)
                    print("  [WARN] No GT: using ckpt global lo/hi for pair_mip")
                else:
                    # last resort: input-based percentiles
                    mipE = xy_mip(E_small)
                    lo = float(np.percentile(mipE, p_lo)); hi = float(np.percentile(mipE, p_hi))
                    print("  [WARN] No GT/global lo/hi: using INPUT MIP percentiles")
            elif norm_mode in ('global_mip','global3d'):
                lo,hi = float(g_lo), float(g_hi)
            else:
                raise ValueError("Unknown norm_mode")

            E01 = robust_norm(E_small, lo, hi)
            if G_raw is not None:
                G01 = robust_norm(G_small, lo, hi)

            # -------- predict
            tin = torch.from_numpy(E01[None,None,...]).to(device)
            P_small = model(tin)[0,0].float().cpu().numpy()

            # -------- upsample to final 512^3 (or CONFIG target)
            target_out = CONFIG['target_out_size']
            P_big = resize_vol_trilinear(P_small, target_out)    # (D,H,W) in [0,1]
            P_big_hwz = from_DHW_to_HWZ(P_big).astype(np.float32)
            assert tuple(P_big.shape) == tuple(target_out), "final size mismatch"

            # -------- save .mat
            out_mat = os.path.join(out_dir, f"{stem}_pred_{target_out[0]}x{target_out[1]}x{target_out[2]}.{ 'mat' if CONFIG['save_format']=='mat7' else 'mat' }")
            if CONFIG['save_format'] == 'mat7':
                savemat(out_mat, {CONFIG['mat_varname']: P_big_hwz}, do_compression=True)
            else:  # 'mat73'
                with h5py.File(out_mat, 'w') as f:
                    f.create_dataset(CONFIG['mat_varname'], data=P_big_hwz, compression='gzip')
            print(f"  [SAVED] {out_mat}  shape={P_big_hwz.shape} dtype=float32 [0,1]", flush=True)

            # -------- optional: save normalized E/GT as 512^3 mats (debug)
            if CONFIG['save_E_GT_mat']:
                E_big = resize_vol_trilinear(E01, target_out)
                E_big_hwz = from_DHW_to_HWZ(E_big).astype(np.float32)
                emat = os.path.join(out_dir, f"{stem}_E_{target_out[0]}x{target_out[1]}x{target_out[2]}.mat")
                if CONFIG['save_format']=='mat7':
                    savemat(emat, {'E':E_big_hwz}, do_compression=True)
                else:
                    with h5py.File(emat, 'w') as f: f.create_dataset('E', data=E_big_hwz, compression='gzip')
                if G_raw is not None:
                    G_big = resize_vol_trilinear(G01, target_out)
                    G_big_hwz = from_DHW_to_HWZ(G_big).astype(np.float32)
                    gmat = os.path.join(out_dir, f"{stem}_GT_{target_out[0]}x{target_out[1]}x{target_out[2]}.mat")
                    if CONFIG['save_format']=='mat7':
                        savemat(gmat, {'GT':G_big_hwz}, do_compression=True)
                    else:
                        with h5py.File(gmat, 'w') as f: f.create_dataset('GT', data=G_big_hwz, compression='gzip')

            # -------- optional: save XY-MIPs (PNG) with colormap
            if CONFIG['save_mip_png']:
                mip_E  = xy_mip(E01)
                mip_Ps = xy_mip(P_small)
                save_img(os.path.join(out_dir, f"{stem}_E_xyMIP.png"),
                         mip_E,  uint16=CONFIG['save_uint16png'],
                         cmap=CONFIG.get('mip_colormap', 'magma'))
                save_img(os.path.join(out_dir, f"{stem}_OUT_xyMIP.png"),
                         mip_Ps, uint16=CONFIG['save_uint16png'],
                         cmap=CONFIG.get('mip_colormap', 'magma'))
                if G_raw is not None:
                    mip_G = xy_mip(G01)
                    save_img(os.path.join(out_dir, f"{stem}_GT_xyMIP.png"),
                             mip_G, uint16=CONFIG['save_uint16png'],
                             cmap=CONFIG.get('mip_colormap', 'magma'))

        except Exception as e:
            print(f"  [ERROR] {rk}: {e}", flush=True)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

if __name__ == "__main__":
    _parse_args_and_update_config()
    main()
