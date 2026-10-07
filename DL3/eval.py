"""
DL 3: evaluation on one scan position.

For each input acquisition time, the time-rescaled input density and the DL 3 output are compared with the localization
density of the 270 s acquisition (GT) of the same position:
  bg_fraction     share of the density outside the dilated GT vessels (lower = higher contrast)
  dice            top 0.5 % mask vs GT top 0.5 % mask
  centerline      fraction of the GT centerline inside the dilated prediction mask
  faint           GT faint vessels (top 3 % minus top 0.5 %) covered by the prediction's top 3 % mask
  branch_ratio    number of skeleton branch points of the prediction mask relative to GT
  density_relerr  median relative error of the density inside the GT vessel mask (top 2 %)

usage:
  python eval.py --pos ../sample_data/dl3_sample --ckpt ../weights/DL3_best.pt --out results_dl3
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi

from train import LocPosition, UNet3D_Loc, LV3_F0


def top_mask(v, q):
    return v > np.quantile(v, 1 - q)


def skeleton_branches(mask):
    from skimage.morphology import skeletonize
    sk = np.zeros_like(mask, bool)
    if mask.any():                                   # skeletonize only the bounding box (much faster)
        idx = np.argwhere(mask); lo = np.maximum(idx.min(0) - 2, 0); hi = idx.max(0) + 3
        sl = tuple(slice(l, h) for l, h in zip(lo, hi))
        sk[sl] = skeletonize(mask[sl]).astype(bool)
    nb = ndi.convolve(sk.astype(np.uint8), np.ones((3, 3, 3), np.uint8), mode="constant") - 1
    lab, n = ndi.label(sk & (nb <= 2), structure=np.ones((3, 3, 3)))
    sizes = np.bincount(lab.ravel())[1:]
    return sk, int((sizes >= 4).sum())


def metrics(pred, gt_s, gt_masks, gt_skel, gt_nbr, q):
    p = ndi.gaussian_filter(pred, 1.0)
    pm = top_mask(p, q); pf = top_mask(p, 0.03)
    g, g3, g2 = gt_masks[q], gt_masks[0.03], gt_masks[0.02]
    faint = g3 & ~gt_masks[0.005]
    _, nbr = skeleton_branches(pm)
    return dict(dice=float(2 * (pm & g).sum() / (pm.sum() + g.sum())),
                centerline=float((ndi.binary_dilation(pm) & gt_skel).sum() / max(gt_skel.sum(), 1)),
                faint=float((pf & faint).sum() / max(faint.sum(), 1)),
                false_vessel=float((pm & ~ndi.binary_dilation(g2, iterations=2)).sum() / max(pm.sum(), 1)),
                branch_ratio=float(nbr / max(gt_nbr, 1)))


def density_metrics(pred, gt_s, gt_masks):
    p = ndi.gaussian_filter(pred, 1.0)
    vm = gt_masks[0.02]
    bg = ~ndi.binary_dilation(gt_masks[0.03], iterations=3)
    x, y = p[vm], gt_s[vm]
    corr = float(np.corrcoef(x, y)[0, 1])
    relerr = float(np.median(np.abs(x - y) / np.maximum(y, 1e-6)))
    thr = 0.2 * np.percentile(x, 99)
    lab, n = ndi.label((p > thr) & bg, structure=np.ones((3, 3, 3)))
    bg_frac = float(p[bg].sum() / max(p.sum(), 1e-12))          # share of the density outside the vessels
    return dict(density_corr=corr, density_relerr=relerr, bg_spots=float(n / bg.sum() * 1e6), bg_fraction=bg_frac)


@torch.no_grad()
def predict(model, pos, T, variant, dev):
    x, base = pos.inputs(LV3_F0, LV3_F0 + int(T * 100), variant)
    xt = torch.from_numpy(x)[None].to(dev); bt = torch.from_numpy(base)[None, None].to(dev)
    return np.exp(base), torch.exp(bt + model(xt))[0, 0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description="DL 3 evaluation")
    ap.add_argument("--pos", required=True, help="position folder with localizations.csv(.gz) and trajectories.csv(.gz)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--durations", nargs="+", type=float, default=[30, 60, 90, 135], help="input acquisition times (s)")
    ap.add_argument("--q", type=float, default=0.005)
    ap.add_argument("--save_mips", action="store_true", help="save XY MIPs of input, DL 3 and GT")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location=dev); cfg = ck.get("cfg", {})
    in_ch = int(cfg.get("in_ch", ck.get("in_ch", 7))); variant = cfg.get("variant", "full" if in_ch == 7 else "density")
    model = UNet3D_Loc(in_ch, base=int(cfg.get("base", 16))).to(dev); model.load_state_dict(ck["model"]); model.eval()
    pos = LocPosition(a.pos)
    gt_s = ndi.gaussian_filter(pos.target, 1.0)
    gt_masks = {q: top_mask(gt_s, q) for q in {a.q, 0.005, 0.02, 0.03}}
    gt_skel, gt_nbr = skeleton_branches(gt_masks[a.q])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for T in a.durations:
        resc, pred = predict(model, pos, T, variant, dev)
        for name, v in (("Rescale", resc), ("DL 3", pred)):
            r = dict(method=name, T=T, **metrics(v, gt_s, gt_masks, gt_skel, gt_nbr, a.q), **density_metrics(v, gt_s, gt_masks))
            rows.append(r)
        if a.save_mips:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            for ax, (nm, v) in zip(axs, (("Input (rescaled)", resc), ("DL 3", pred), ("GT 270 s", pos.target))):
                m = ndi.gaussian_filter(v, 1.0).max(axis=2)
                ax.imshow(np.clip(m / np.percentile(m[m > 0], 99.8), 0, 1).T, cmap="hot", origin="lower"); ax.set_title(nm)
                ax.axis("off")
            fig.suptitle(f"{T:g} s input"); fig.tight_layout(); fig.savefig(out / f"mip_{T:g}s.png", dpi=150); plt.close(fig)
    df = pd.DataFrame(rows)
    cols = ["method", "T", "bg_fraction", "dice", "centerline", "faint", "branch_ratio", "density_relerr"]
    df[cols].to_csv(out / "metrics.csv", index=False)
    print(df[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
