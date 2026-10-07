"""
lot_svd.py: clutter filtering and data preparation shared by DL 2 training and inference.

Processing of one window of sinograms:
  1) systematic noise removal: X <- X - mean_c(X - mean_t(X))            (per frame and sample)
  2) SVD clutter filter: eigen-decomposition of the T x T Gram matrix, components th1..th2 kept
                         (1-based, inclusive), so that th1 = 11 removes the 10 largest components
  3) channel-group noise removal: the mean of each acquisition channel group is subtracted per frame and sample
Layout: (T, C, S) = (frames, 512 channels, 493 samples after removing the first three samples).

Conventions
  - measured frames at the reduced rate: global 1-based frame g with (g - 1) % ds == 0 (1, 6, 11, ... for ds = 5)
  - windows of 1000 frames (10 s at 100 Hz)
  - reference: 100 Hz window filtered with (th1, th2 = 1000)
  - input: the measured frames of the same window filtered on their own (th1, th2 = number of measured frames)
  - both scaled by a robust noise estimate of the input (1.4826 * median |x|)
  - network layout (B, C, H = 496, W = 512, T), rows 0-2 are zero
"""
import numpy as np
import torch

GROUPS = ((0, 64), (64, 128), (128, 256), (256, 512))
ROW0 = 3            # the first three time samples are discarded
H_NET = 496
DS = 5


# ----------------------------------------------------------------------------- filter
@torch.no_grad()
def lot_svd_filter(X, th1=11, th2=None, groups=GROUPS):
    """X: torch float32 (T, C, S) on any device. Returns filtered (T, C, S) float32."""
    T = X.shape[0]
    th2 = T if th2 is None else th2
    D = X - X.mean(0, keepdim=True)
    Y = X - D.mean(1, keepdim=True)                          # remove_systematic_noise
    Y2 = Y.reshape(T, -1)
    G = Y2.double() @ Y2.double().T                          # T x T
    _, V = torch.linalg.eigh(G)                              # ascending
    V = V.flip(1)                                            # descending
    r1, r2 = max(1, th1), min(th2, T)
    Vf = V[:, r1 - 1:r2].to(Y2.dtype)
    Fm = (Vf @ (Vf.T @ Y2)).reshape(Y.shape)                 # SVD_filter_fast
    for a, b in groups:                                      # remove_noise_ch
        Fm[:, a:b, :] -= Fm[:, a:b, :].mean(1, keepdim=True)
    return Fm


def lot_svd_filter_np(X, th1=11, th2=None, groups=GROUPS):
    """numpy mirror of lot_svd_filter (CPU, used for tests and small analyses)."""
    X = np.asarray(X, np.float64)
    T = X.shape[0]
    th2 = T if th2 is None else th2
    Y = X - (X - X.mean(0, keepdims=True)).mean(1, keepdims=True)
    Y2 = Y.reshape(T, -1)
    _, V = np.linalg.eigh(Y2 @ Y2.T)
    V = V[:, ::-1]
    Vf = V[:, max(1, th1) - 1:min(th2, T)]
    Fm = (Vf @ (Vf.T @ Y2)).reshape(Y.shape)
    for a, b in groups:
        Fm[:, a:b, :] -= Fm[:, a:b, :].mean(1, keepdims=True)
    return Fm.astype(np.float32)


# ----------------------------------------------------------------------------- IO
def load_block_hwt(path, start_1based, count):
    """Raw sigMat or DL output (.mat v7.3). Returns numpy float32 (496, 512, count)."""
    import h5py
    with h5py.File(path, "r") as f:
        cands = []
        f.visititems(lambda n, o: cands.append(o) if isinstance(o, h5py.Dataset) and o.ndim == 3 else None)
        ds = max(cands, key=lambda d: np.prod(d.shape))
        shp = ds.shape
        ax_h = shp.index(496); ax_w = shp.index(512)
        ax_t = [i for i in range(3) if i not in (ax_h, ax_w)][0]
        sl = [slice(None)] * 3
        sl[ax_t] = slice(start_1based - 1, start_1based - 1 + count)
        a = np.asarray(ds[tuple(sl)], dtype=np.float32)
    return np.ascontiguousarray(np.transpose(a, (ax_h, ax_w, ax_t)))


def num_frames(path):
    import h5py
    with h5py.File(path, "r") as f:
        cands = []
        f.visititems(lambda n, o: cands.append(o) if isinstance(o, h5py.Dataset) and o.ndim == 3 else None)
        shp = max(cands, key=lambda d: np.prod(d.shape)).shape
    return [s for s in shp if s not in (496, 512)][0]


# ----------------------------------------------------------------------------- window prep
MEAS_JITTER_SEED = None     # None: regular (g - 1) % ds == 0. int: one measured frame at a random position in every
                            # block of ds frames (same mean rate, irregular intervals 1 to 2*ds - 1), reproducible


_JITTER_OFFS = {}


def measured_global_mask(g, ds=DS):
    """g: global 1-based frame numbers. True where the frame is measured."""
    g = np.asarray(g, np.int64)
    if MEAS_JITTER_SEED is None:
        return (g - 1) % ds == 0
    k = (g - 1) // ds                                            # block index
    key = (MEAS_JITTER_SEED, ds)
    if key not in _JITTER_OFFS:                                  # uniform random offset per block, reproducible
        _JITTER_OFFS[key] = np.random.default_rng(MEAS_JITTER_SEED).integers(0, ds, size=1_000_000)
    return (g - 1) % ds == _JITTER_OFFS[key][k]


def measured_local_idx(start_1based, T, ds=DS):
    g = np.arange(start_1based, start_1based + T)
    return np.where(measured_global_mask(g, ds))[0]


@torch.no_grad()
def prepare_window(raw_hwt, start_1based, device, th1=11, ds=DS, with_gt=True):
    """
    raw_hwt: numpy (496, 512, T) raw window (as returned by _load_block_496x512).
    Returns dict with network tensors (1, 1, 496, 512, T) on device:
      x (measured frames filtered at 20 Hz, zeros elsewhere), m (1 at measured frames),
      y (100 Hz filtered GT, if with_gt), scale (float), meas (local indices)
    """
    X = torch.from_numpy(np.ascontiguousarray(raw_hwt[ROW0:].transpose(2, 1, 0))).float().to(device)  # (T,C,S)
    T = X.shape[0]
    meas = torch.as_tensor(measured_local_idx(start_1based, T, ds), device=device)
    Xm = lot_svd_filter(X[meas], th1=th1, th2=len(meas))
    scale = float(1.4826 * Xm.abs().median().item()) + 1e-6

    def to_net(t_tcs):
        out = torch.zeros((1, 1, H_NET, t_tcs.shape[1], t_tcs.shape[0]), device=device)
        out[0, 0, ROW0:] = t_tcs.permute(2, 1, 0)
        return out

    x_tcs = torch.zeros_like(X)
    x_tcs[meas] = Xm
    d = dict(x=to_net(x_tcs) / scale, scale=scale, meas=meas)
    m = torch.zeros((1, 1, 1, 1, T), device=device)
    m[..., meas] = 1.0
    d["m"] = m
    if with_gt:
        d["y"] = to_net(lot_svd_filter(X, th1=th1, th2=T)) / scale
    del X
    return d


def linear_interp_time(x, m):
    """x: (B,1,H,W,T) with zeros at missing frames, m: (B,1,1,1,T). Linear in time between measured frames,
    nearest measured frame beyond the ends."""
    T = x.shape[-1]
    idx = torch.nonzero(m[0, 0, 0, 0] > 0.5).flatten()
    out = x.clone()
    t = torch.arange(T, device=x.device)
    k = torch.searchsorted(idx, t, right=True) - 1
    k0 = k.clamp(0, len(idx) - 1)
    k1 = (k + 1).clamp(0, len(idx) - 1)
    t0, t1 = idx[k0], idx[k1]
    w = torch.where(t1 > t0, (t - t0).float() / (t1 - t0).clamp(min=1).float(), torch.zeros_like(t, dtype=torch.float))
    out = x[..., t0] * (1 - w) + x[..., t1] * w
    return out


# ----------------------------------------------------------------------------- inference
@torch.no_grad()
def sliding_infer(model, x, m, t_chunk=64, stride=32, amp=True, residual=False):
    """Overlapping temporal chunks, Hann-weighted merge, hard data consistency at measured frames.
    residual=True: x is the linearly interpolated window and the network predicts a correction to it."""
    T = x.shape[-1]
    starts = list(range(0, max(T - t_chunk, 0) + 1, stride))
    if starts[-1] + t_chunk < T:
        starts.append(T - t_chunk)
    w = torch.hann_window(t_chunk + 2, periodic=False, device=x.device)[1:-1]
    acc = torch.zeros_like(x)
    wsum = torch.zeros((T,), device=x.device)
    for s in starts:
        xi, mi = x[..., s:s + t_chunk], m[..., s:s + t_chunk]
        inp = torch.cat([xi, mi.expand(-1, 1, xi.shape[2], xi.shape[3], -1)], dim=1)
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=amp and x.is_cuda):
            p = model(inp).float()
        if residual:
            p = p + xi
        acc[..., s:s + t_chunk] += p * w
        wsum[s:s + t_chunk] += w
    pred = acc / wsum.clamp(min=1e-6)
    return m * x + (1 - m) * pred
