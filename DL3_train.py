"""
DL3 (SLOT): 3D reconstruction enhancement for low acquisition time.

Trains a 3D U-Net that maps low-acquisition-time 3D volumes (input) to
high-quality ground-truth volumes.

Implementation notes (kept from the original research code):
- Normalization modes: pair_mip / global_mip / global3d (percentile-based).
- Loss: weighted MSE + (1 - SSIM3D), with optional gradient loss.
- Optional vessel-weight mask derived from GT for emphasizing vascular structures.

This repository does NOT include the experimental data (.mat/.h5). See README for the
expected folder layout.
"""

import os, glob, warnings, json
warnings.filterwarnings('ignore')

import numpy as np
import h5py
from scipy.io import loadmat
from scipy.ndimage import sobel

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, ball
from skimage import io as skio

from tqdm import tqdm

# =======================
# CONFIG
# =======================
CONFIG = dict(
    # paths
    input_root=r'data/dl3/input',
    gt_root   =r'data/dl3/gt',
    workdir   ='outputs/dl3',

    seed       = 2024,

    # training
    epochs     = 120,
    batch      = 1,          # for 256^3 volumes, batch size is typically 1
    lr         = 3e-4,
    num_workers= 4,

    # volumes
    target_size = (256,256,256),   # (D,H,W)
    dtype       = 'float32',

    # normalization
    # 'pair_mip'   : per-sample lo/hi from that sample's GT XY-MIP
    # 'global_mip' : median lo/hi from GT/train XY-MIPs
    # 'global3d'   : median lo/hi from GT/train 3D vols
    norm_mode  = 'pair_mip',
    p_lo       = 1.0,
    p_hi       = 99.7,

    # losses
    # Note: variable name kept for backward-compatibility; w_charb weights the weighted-MSE term
    w_charb = 1.0,   # weighted MSE (PSNR ↑)
    w_ssim  = 0.5,   # (1-SSIM3D)
    w_grad  = 0.0,   # set >0 to enable gradient loss

    # vessel-weight from GT
    min_obj_vox = 64,
    close_r     = 0,   # 0 -> off

    # saving
    save_every_epoch = 1,
    save_train_first_only = True,
    save_uint16 = True,
    mip_save_hw = (512, 512),     # XY-MIP save resolution (None: no resize)

    # model
    base_channels = 16,           # set to 8 if you hit OOM
    use_amp       = False,        # set True if needed
)

CONFIG['input_root'] = os.environ.get('SLOT_DL3_INPUT_ROOT', os.environ.get('LOT_INPUT_ROOT', CONFIG['input_root']))
CONFIG['gt_root']    = os.environ.get('SLOT_DL3_GT_ROOT', os.environ.get('LOT_GT_ROOT', CONFIG['gt_root']))
CONFIG['workdir']    = os.environ.get('SLOT_DL3_WORKDIR', CONFIG['workdir'])

# =======================
# Utils
# =======================
def set_seed(seed=2024):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ensure_dirs():
    os.makedirs(CONFIG['workdir'], exist_ok=True)
    os.makedirs(os.path.join(CONFIG['workdir'], 'checkpoints'), exist_ok=True)

def _largest_3d_from_h5(f: h5py.File):
    cand = None
    def visit(name, obj):
        nonlocal cand
        if isinstance(obj, h5py.Dataset) and len(obj.shape)==3:
            if cand is None or np.prod(obj.shape) > np.prod(cand.shape):
                cand = obj
    f.visititems(visit)
    return cand

def load_volume_from_mat(path: str) -> np.ndarray:
    # Try HDF5 (v7.3)
    try:
        with h5py.File(path, 'r') as f:
            dset = _largest_3d_from_h5(f)
            if dset is not None:
                return np.array(dset)
    except Exception:
        pass
    # Fallback (v7-)
    md = loadmat(path)
    vol=None
    for k,v in md.items():
        if isinstance(v, np.ndarray) and v.ndim==3:
            if vol is None or vol.size<v.size:
                vol=v
    if vol is None:
        raise RuntimeError(f"No 3D array in {path}")
    return vol

def to_DHW(vol: np.ndarray) -> np.ndarray:
    if vol.ndim != 3:
        raise ValueError("Volume must be 3D")
    return np.moveaxis(vol, -1, 0)  # (D,H,W)

def resize_vol_trilinear(vol_DHW: np.ndarray, target=(256,256,256)) -> np.ndarray:
    v = torch.from_numpy(vol_DHW.astype(np.float32))[None,None,...]
    tgtD, tgtH, tgtW = target
    v2 = F.interpolate(v, size=(tgtD,tgtH,tgtW), mode='trilinear', align_corners=False)
    return v2[0,0].cpu().numpy()

def xy_mip(vol_DHW: np.ndarray) -> np.ndarray:
    return vol_DHW.max(axis=0)  # (H,W)

def robust_norm(x, lo, hi, eps=1e-6):
    return np.clip((x - lo) / (hi - lo + eps), 0, 1).astype(np.float32)

def global_percentiles_from_gt_mips(gt_root_train: str, target=(256,256,256), p_lo=1.0, p_hi=99.7):
    files = sorted(glob.glob(os.path.join(gt_root_train, '*.mat')))
    if len(files)==0:
        raise RuntimeError(f"No GT train files in {gt_root_train}")
    lows=[]; highs=[]
    for fp in tqdm(files, desc='scan GT-train XY-MIP percentiles', leave=False):
        vol = to_DHW(load_volume_from_mat(fp)).astype(np.float32)
        if target is not None:
            vol = resize_vol_trilinear(vol, target)
        mip = xy_mip(vol)
        lows.append(np.percentile(mip, p_lo))
        highs.append(np.percentile(mip, p_hi))
    lo = float(np.median(lows)); hi = float(np.median(highs))
    return lo, hi

def global_percentiles_from_gt_3d(gt_root_train: str, target=(256,256,256), p_lo=1.0, p_hi=99.7):
    files = sorted(glob.glob(os.path.join(gt_root_train, '*.mat')))
    if len(files)==0:
        raise RuntimeError(f"No GT train files in {gt_root_train}")
    lows=[]; highs=[]
    for fp in tqdm(files, desc='scan GT-train 3D percentiles', leave=False):
        vol = to_DHW(load_volume_from_mat(fp)).astype(np.float32)
        if target is not None:
            vol = resize_vol_trilinear(vol, target)
        lows.append(np.percentile(vol, p_lo))
        highs.append(np.percentile(vol, p_hi))
    lo = float(np.median(lows)); hi = float(np.median(highs))
    return lo, hi

def build_weights_from_gt(gt01: np.ndarray) -> np.ndarray:
    th = threshold_otsu(gt01)
    mask = gt01 > th
    if CONFIG['close_r']>0:
        from scipy.ndimage import binary_closing as bc
        mask = bc(mask, structure=ball(CONFIG['close_r']))
    if CONFIG['min_obj_vox']>0:
        mask = remove_small_objects(mask, min_size=CONFIG['min_obj_vox'])
    gx = sobel(gt01, axis=2); gy = sobel(gt01, axis=1); gz = sobel(gt01, axis=0)
    grad = np.sqrt(gx*gx + gy*gy + gz*gz)
    grad = grad / (grad.max()+1e-6)
    w = 1.0 + 2.0*mask.astype(np.float32) + 1.0*grad.astype(np.float32)
    return w.astype(np.float32)

def save_img(path, img01, uint16=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = np.clip(img01, 0, 1)
    if uint16:
        skio.imsave(path, (img*65535.0+0.5).astype(np.uint16))
    else:
        skio.imsave(path, (img*255.0+0.5).astype(np.uint8))

def _resize2d_01(img01: np.ndarray, out_hw):
    """img01: (H,W) [0,1] → resize to out_hw (H,W) with bilinear."""
    if (out_hw is None) or (tuple(out_hw) == tuple(img01.shape)):
        return img01
    H, W = out_hw
    t = torch.from_numpy(img01.astype(np.float32))[None, None, ...]
    r = F.interpolate(t, size=(H, W), mode='bilinear', align_corners=False)
    return r[0,0].cpu().numpy()

# =======================
# Dataset
# =======================
class VolDataset(Dataset):
    def __init__(self, input_root, gt_root, split, target_size, mode, g_lo=None, g_hi=None, p_lo=1.0, p_hi=99.7):
        self.in_dir = os.path.join(input_root, split)
        self.gt_dir = os.path.join(gt_root,   split)
        self.files = sorted([os.path.basename(p) for p in glob.glob(os.path.join(self.in_dir, '*.mat'))])
        self.files = [f for f in self.files if os.path.exists(os.path.join(self.gt_dir, f))]
        if len(self.files)==0:
            raise RuntimeError(f"No matching .mat files in {self.in_dir} and {self.gt_dir}")
        self.tgt = target_size
        self.mode = mode  # 'pair_mip', 'global_mip', 'global3d'
        self.g_lo = g_lo; self.g_hi = g_hi
        self.p_lo = p_lo; self.p_hi = p_hi
        self.split=split

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        Ein_raw = to_DHW(load_volume_from_mat(os.path.join(self.in_dir, fname))).astype(np.float32)
        GT_raw  = to_DHW(load_volume_from_mat(os.path.join(self.gt_dir,  fname))).astype(np.float32)
        Ein = resize_vol_trilinear(Ein_raw, self.tgt)
        GT  = resize_vol_trilinear(GT_raw,  self.tgt)

        # normalization (MIP-aligned)
        if self.mode == 'pair_mip':
            mip = xy_mip(GT)
            lo = np.percentile(mip, self.p_lo); hi = np.percentile(mip, self.p_hi)
        elif self.mode == 'global_mip':
            lo, hi = self.g_lo, self.g_hi
        elif self.mode == 'global3d':
            lo, hi = self.g_lo, self.g_hi
        else:
            raise ValueError("Unknown norm_mode")

        Ein01 = robust_norm(Ein, lo, hi)
        GT01  = robust_norm(GT,  lo, hi)
        W = build_weights_from_gt(GT01)

        E = torch.from_numpy(Ein01[None,...])  # (1,D,H,W)
        G = torch.from_numpy(GT01[None,...])
        WW= torch.from_numpy(W[None,...])
        return {'E':E, 'GT':G, 'W':WW, 'fname':fname}

def build_loader(split, target_size, batch, shuffle, num_workers, norm_mode, g_lo=None, g_hi=None, p_lo=1.0, p_hi=99.7):
    ds = VolDataset(CONFIG['input_root'], CONFIG['gt_root'], split, target_size, norm_mode, g_lo, g_hi, p_lo, p_hi)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=False)

# =======================
# 3D U-Net
# =======================
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
        y=torch.sigmoid(self.out(x))
        return y

# =======================
# Losses
# =======================
class Charbonnier(nn.Module):
    """(Reference only; currently unused.)"""
    def __init__(self, eps=1e-3): super().__init__(); self.eps=eps
    def forward(self, x, y, w=None):
        diff = torch.sqrt((x - y)**2 + self.eps**2)
        if w is not None:
            return (diff * w).sum() / (w.sum()+1e-6)
        return diff.mean()

def _gaussian3d(win, sigma, device, channel=1):
    c = torch.arange(win, dtype=torch.float32, device=device) - win//2
    g1 = torch.exp(-(c**2)/(2*sigma**2)); g1 = g1 / (g1.sum()+1e-6)
    g = g1[:,None,None] * g1[None,:,None] * g1[None,None,:]
    g = g/g.sum()
    g = g.view(1,1,win,win,win)
    return g.expand(channel,1,win,win,win).contiguous()

class SSIM3D(nn.Module):
    def __init__(self, window=7, sigma=1.0, channel=1): super().__init__(); self.win=window; self.sigma=sigma; self.channel=channel
    def forward(self,x,y):
        C1=0.01**2; C2=0.03**2; device=x.device
        w=_gaussian3d(self.win,self.sigma,device,self.channel)
        mu_x=F.conv3d(x,w,padding=self.win//2,groups=self.channel)
        mu_y=F.conv3d(y,w,padding=self.win//2,groups=self.channel)
        mu_x2=mu_x*mu_x; mu_y2=mu_y*mu_y; mu_xy=mu_x*mu_y
        sig_x2=F.conv3d(x*x,w,padding=self.win//2,groups=self.channel)-mu_x2
        sig_y2=F.conv3d(y*y,w,padding=self.win//2,groups=self.channel)-mu_y2
        sig_xy=F.conv3d(x*y,w,padding=self.win//2,groups=self.channel)-mu_xy
        ssim_map=((2*mu_xy+C1)*(2*sig_xy+C2))/((mu_x2+mu_y2+C1)*(sig_x2+sig_y2+C2))
        return 1.0-ssim_map.mean()  # loss form: (1 - SSIM)

def grad_loss3d(x, y):
    kx = torch.tensor([[[[-1,0,1]]]], dtype=torch.float32, device=x.device).view(1,1,1,1,3)
    ky = torch.tensor([[[[-1,0,1]]]], dtype=torch.float32, device=x.device).view(1,1,1,3,1)
    kz = torch.tensor([[[[-1,0,1]]]], dtype=torch.float32, device=x.device).view(1,1,3,1,1)
    gx = F.conv3d(x, kx, padding=(0,0,1)); gy = F.conv3d(x, ky, padding=(0,1,0)); gz = F.conv3d(x, kz, padding=(1,0,0))
    gtx= F.conv3d(y, kx, padding=(0,0,1)); gty= F.conv3d(y, ky, padding=(0,1,0)); gtz= F.conv3d(y, kz, padding=(1,0,0))
    return F.l1_loss(gx,gtx) + F.l1_loss(gy,gty) + F.l1_loss(gz,gtz)

class WeightedMSE(nn.Module):
    """Weighted MSE that emphasizes the foreground using a vessel mask W."""
    def __init__(self): super().__init__()
    def forward(self, x, y, w=None):
        diff2 = (x - y) ** 2
        if w is not None:
            return (diff2 * w).sum() / (w.sum() + 1e-6)
        return diff2.mean()

def psnr_metric(pred, target, max_val=1.0, eps=1e-8):
    """
    PSNR(dB) = 20*log10(max_val) - 10*log10(MSE)
    Assumes GT/Pred are normalized to 0..1, so max_val=1.0.
    """
    mse = F.mse_loss(pred, target)
    maxv = torch.tensor(max_val, dtype=pred.dtype, device=pred.device)
    return 20.0*torch.log10(maxv + eps) - 10.0*torch.log10(mse + eps)

# =======================
# Train / Eval / Save MIPs
# =======================
def train_one_epoch(model, dl, optim, device, scaler=None):
    model.train()
    mse_fn = WeightedMSE()
    ssim3d = SSIM3D()
    logs={'mse':0,'ssim1m':0,'grad':0,'psnr':0,'tot':0}
    use_amp = CONFIG['use_amp'] and scaler is not None

    for b in tqdm(dl, desc='train', leave=False):
        E  = b['E'].to(device, non_blocking=True)
        GT = b['GT'].to(device, non_blocking=True)
        W  = b['W'].to(device, non_blocking=True)
        optim.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                P = model(E)
                loss_m = mse_fn(P, GT, W)   # PSNR-oriented term
                loss_s = ssim3d(P, GT)      # 1-SSIM
                loss_g = grad_loss3d(P, GT) # optional edge term
                loss = (CONFIG['w_charb']*loss_m +
                        CONFIG['w_ssim'] *loss_s +
                        CONFIG['w_grad'] *loss_g)
            scaler.scale(loss).backward()
            scaler.step(optim); scaler.update()
        else:
            P = model(E)
            loss_m = mse_fn(P, GT, W)
            loss_s = ssim3d(P, GT)
            loss_g = grad_loss3d(P, GT)
            loss = (CONFIG['w_charb']*loss_m +
                    CONFIG['w_ssim'] *loss_s +
                    CONFIG['w_grad'] *loss_g)
            loss.backward(); optim.step()

        psnr = psnr_metric(P, GT)

        logs['mse']   += loss_m.item()
        logs['ssim1m']+= loss_s.item()
        logs['grad']  += loss_g.item()
        logs['psnr']  += psnr.item()
        logs['tot']   += loss.item()

    n=len(dl)
    return {k:v/n for k,v in logs.items()}

@torch.no_grad()
def evaluate(model, dl, device):
    model.eval(); ssim3d=SSIM3D()
    logs={'MSE':0,'SSIM3D':0,'PSNR':0}
    for b in tqdm(dl, desc='val', leave=False):
        E  = b['E'].to(device, non_blocking=True)
        GT = b['GT'].to(device, non_blocking=True)
        P  = model(E)
        mse  = F.mse_loss(P,GT)
        ssim = 1.0-ssim3d(P,GT).item()
        psnr = psnr_metric(P,GT)
        logs['MSE']    += mse.item()
        logs['SSIM3D'] += ssim
        logs['PSNR']   += psnr.item()
    n=len(dl)
    return {k:v/n for k,v in logs.items()}

@torch.no_grad()
def save_epoch_mips(model, dl, device, outdir, save_first_only=False, tag='val'):
    """
    Save per-batch XY-MIP images:
      - Individual files: *_E_xyMIP.(tif/png), *_OUT_xyMIP.(tif/png), *_GT_xyMIP.(tif/png)
      - Triptych (side-by-side): *_TRIPT_xyMIP.png  [Input | Output | GT]
    """
    model.eval(); os.makedirs(outdir, exist_ok=True)
    for i,b in enumerate(tqdm(dl, desc=f'save-mip:{tag}', leave=False)):
        E=b['E'].to(device, non_blocking=True)
        GT=b['GT'].to(device, non_blocking=True)
        fname=os.path.splitext(b['fname'][0])[0]

        P=model(E)

        # numpy (0..1)
        E_np = E[0,0].detach().float().cpu().numpy()
        G_np = GT[0,0].detach().float().cpu().numpy()
        P_np = P[0,0].detach().float().cpu().numpy()

        # XY-MIP
        mip_E = E_np.max(axis=0)
        mip_G = G_np.max(axis=0)
        mip_P = P_np.max(axis=0)

        # upsample for visualization
        out_hw = CONFIG.get('mip_save_hw', None)
        if out_hw:
            mip_E = _resize2d_01(mip_E, out_hw)
            mip_G = _resize2d_01(mip_G, out_hw)
            mip_P = _resize2d_01(mip_P, out_hw)

        # save individual (including Input)
        save_img(os.path.join(outdir, f"{fname}_E_xyMIP.tif"),   mip_E, uint16=CONFIG['save_uint16'])
        save_img(os.path.join(outdir, f"{fname}_OUT_xyMIP.tif"), mip_P, uint16=CONFIG['save_uint16'])
        save_img(os.path.join(outdir, f"{fname}_GT_xyMIP.tif"),  mip_G, uint16=CONFIG['save_uint16'])

        save_img(os.path.join(outdir, f"{fname}_E_xyMIP.png"),   mip_E, uint16=False)
        save_img(os.path.join(outdir, f"{fname}_OUT_xyMIP.png"), mip_P, uint16=False)
        save_img(os.path.join(outdir, f"{fname}_GT_xyMIP.png"),  mip_G, uint16=False)

        # Triptych (Input | Output | GT)
        triptych = np.clip(np.concatenate([mip_E, mip_P, mip_G], axis=1), 0, 1)
        save_img(os.path.join(outdir, f"{fname}_TRIPT_xyMIP.png"), triptych, uint16=False)

        if save_first_only and i == 0:
            break

# =======================
# Main
# =======================
def autorun():
    set_seed(CONFIG['seed']); ensure_dirs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Target size for training: {CONFIG['target_size']}")
    if CONFIG.get('mip_save_hw'):
        print(f"[INFO] Saving XY-MIPs at {CONFIG['mip_save_hw']} (+ Triptych)")

    # compute global lo/hi if needed
    g_lo=g_hi=None
    if CONFIG['norm_mode']=='global_mip':
        g_lo, g_hi = global_percentiles_from_gt_mips(os.path.join(CONFIG['gt_root'],'train'),
                                                     CONFIG['target_size'], CONFIG['p_lo'], CONFIG['p_hi'])
        print(f"[INFO] Global lo/hi from GT/train XY-MIPs: lo={g_lo:.6f}, hi={g_hi:.6f}")
    elif CONFIG['norm_mode']=='global3d':
        g_lo, g_hi = global_percentiles_from_gt_3d(os.path.join(CONFIG['gt_root'],'train'),
                                                   CONFIG['target_size'], CONFIG['p_lo'], CONFIG['p_hi'])
        print(f"[INFO] Global lo/hi from GT/train 3D vols: lo={g_lo:.6f}, hi={g_hi:.6f}")
    else:
        print("[INFO] Pair-wise MIP normalization (per-sample lo/hi from each GT XY-MIP)")

    # loaders
    dl_tr = build_loader('train', CONFIG['target_size'], CONFIG['batch'], True, CONFIG['num_workers'],
                         CONFIG['norm_mode'], g_lo, g_hi, CONFIG['p_lo'], CONFIG['p_hi'])
    dl_va = build_loader('val',   CONFIG['target_size'], 1, False, max(1,CONFIG['num_workers']//2),
                         CONFIG['norm_mode'], g_lo, g_hi, CONFIG['p_lo'], CONFIG['p_hi'])
    dl_tr_full = build_loader('train', CONFIG['target_size'], 1, False, max(1,CONFIG['num_workers']//2),
                              CONFIG['norm_mode'], g_lo, g_hi, CONFIG['p_lo'], CONFIG['p_hi'])
    dl_va_full = build_loader('val',   CONFIG['target_size'], 1, False, max(1,CONFIG['num_workers']//2),
                              CONFIG['norm_mode'], g_lo, g_hi, CONFIG['p_lo'], CONFIG['p_hi'])
    dl_te_full=None
    te_in = os.path.join(CONFIG['input_root'],'test'); te_gt = os.path.join(CONFIG['gt_root'],'test')
    if len(glob.glob(os.path.join(te_in,'*.mat'))) and len(glob.glob(os.path.join(te_gt,'*.mat'))):
        dl_te_full = build_loader('test', CONFIG['target_size'], 1, False, max(1,CONFIG['num_workers']//2),
                                  CONFIG['norm_mode'], g_lo, g_hi, CONFIG['p_lo'], CONFIG['p_hi'])

    # model/optim
    model = UNet3D_E2GT(base=CONFIG['base_channels']).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], betas=(0.9,0.999), weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler() if (device.type=='cuda' and CONFIG['use_amp']) else None

    ckpt_dir = os.path.join(CONFIG['workdir'],'checkpoints')
    mips_dir = os.path.join(CONFIG['workdir'],'mips'); os.makedirs(mips_dir, exist_ok=True)

    # epochs
    for ep in range(1, CONFIG['epochs']+1):
        tr = train_one_epoch(model, dl_tr, optim, device, scaler=scaler)
        va = evaluate(model, dl_va, device)
        print({'epoch':ep, **{f"tr_{k}":v for k,v in tr.items()}, **{f"va_{k}":v for k,v in va.items()}})

        torch.save({'model':model.state_dict(),'optim':optim.state_dict(),'config':CONFIG,'epoch':ep,
                    'norm_mode':CONFIG['norm_mode'],'g_lo':g_lo,'g_hi':g_hi},
                   os.path.join(ckpt_dir, f"ep{ep:03d}.pt"))

        if ep % CONFIG['save_every_epoch']==0:
            save_epoch_mips(model, dl_tr_full, device,
                            os.path.join(mips_dir,'train',f"ep{ep:03d}"),
                            save_first_only=CONFIG['save_train_first_only'], tag='train')
            save_epoch_mips(model, dl_va_full, device,
                            os.path.join(mips_dir,'val',  f"ep{ep:03d}"),
                            save_first_only=False, tag='val')
            if dl_te_full is not None:
                save_epoch_mips(model, dl_te_full, device,
                                os.path.join(mips_dir,'test', f"ep{ep:03d}"),
                                save_first_only=False, tag='test')



def _parse_args_and_update_config() -> None:
    import argparse
    p = argparse.ArgumentParser(description="DL3 (SLOT) training: low acquisition time 3D enhancement.")

    p.add_argument("--input_root", type=str, default=CONFIG["input_root"], help="Input volume root (expects train/val/test subfolders).")
    p.add_argument("--gt_root",    type=str, default=CONFIG["gt_root"],    help="Ground-truth root (expects train/val/test subfolders).")
    p.add_argument("--out",        type=str, default=CONFIG["workdir"],    help="Output directory (checkpoints, mips, logs).")

    p.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    p.add_argument("--batch",  type=int, default=CONFIG["batch"])
    p.add_argument("--lr",     type=float, default=CONFIG["lr"])
    p.add_argument("--workers", type=int, default=CONFIG["num_workers"])
    p.add_argument("--seed", type=int, default=CONFIG["seed"])

    p.add_argument("--target_size", type=int, nargs=3, default=list(CONFIG["target_size"]), metavar=("D","H","W"),
                   help="Training volume size after resizing/cropping, as three integers.")
    p.add_argument("--norm_mode", type=str, default=CONFIG["norm_mode"], choices=["pair_mip","global_mip","global3d"])
    p.add_argument("--p_lo", type=float, default=CONFIG["p_lo"])
    p.add_argument("--p_hi", type=float, default=CONFIG["p_hi"])

    p.add_argument("--w_mse",  type=float, default=CONFIG["w_charb"], help="Weight of weighted-MSE term.")
    p.add_argument("--w_ssim", type=float, default=CONFIG["w_ssim"],  help="Weight of (1-SSIM3D) term.")
    p.add_argument("--w_grad", type=float, default=CONFIG["w_grad"],  help="Weight of gradient term (0 disables).")

    p.add_argument("--min_obj_vox", type=int, default=CONFIG["min_obj_vox"])
    p.add_argument("--close_r", type=int, default=CONFIG["close_r"])

    p.add_argument("--save_every_epoch", type=int, default=CONFIG["save_every_epoch"])
    p.add_argument("--save_train_first_only", action="store_true", default=CONFIG["save_train_first_only"])
    p.add_argument("--save_uint16", action="store_true", default=CONFIG["save_uint16"])
    p.add_argument("--mip_save_hw", type=int, nargs=2, default=list(CONFIG["mip_save_hw"]) if CONFIG.get("mip_save_hw") else None,
                   metavar=("H","W"), help="Optional XY-MIP save resolution (H W).")

    p.add_argument("--base_channels", type=int, default=CONFIG["base_channels"])
    p.add_argument("--use_amp", action="store_true", default=CONFIG["use_amp"])

    args = p.parse_args()
    CONFIG["input_root"] = args.input_root
    CONFIG["gt_root"]    = args.gt_root
    CONFIG["workdir"]    = args.out
    CONFIG["epochs"]     = args.epochs
    CONFIG["batch"]      = args.batch
    CONFIG["lr"]         = args.lr
    CONFIG["num_workers"]= args.workers
    CONFIG["seed"]       = args.seed
    CONFIG["target_size"]= tuple(int(x) for x in args.target_size)
    CONFIG["norm_mode"]  = args.norm_mode
    CONFIG["p_lo"]       = args.p_lo
    CONFIG["p_hi"]       = args.p_hi
    CONFIG["w_charb"]    = args.w_mse
    CONFIG["w_ssim"]     = args.w_ssim
    CONFIG["w_grad"]     = args.w_grad
    CONFIG["min_obj_vox"]= args.min_obj_vox
    CONFIG["close_r"]    = args.close_r
    CONFIG["save_every_epoch"] = args.save_every_epoch
    CONFIG["save_train_first_only"] = args.save_train_first_only
    CONFIG["save_uint16"] = args.save_uint16
    CONFIG["mip_save_hw"] = tuple(args.mip_save_hw) if args.mip_save_hw is not None else None
    CONFIG["base_channels"] = args.base_channels
    CONFIG["use_amp"] = args.use_amp

    # Persist run config
    os.makedirs(CONFIG["workdir"], exist_ok=True)
    with open(os.path.join(CONFIG["workdir"], "run_args.json"), "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=2)

if __name__ == "__main__":
    _parse_args_and_update_config()
    autorun()
