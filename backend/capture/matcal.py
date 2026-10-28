"""Self-calibration of the printed mat.

Measured on this mat (stage 1): with the printed planar layout every frame of all three videos reprojects at 2.5-3.7 px rms, worse at low elevation
(V3: 5.1 px below 30 deg, 2.0 px above 70 deg). Freeing the 3D position of each marker corner (with a prior toward the printed layout) in a bundle adjustment over
many frames drops that to about 1.3-1.5 px, and the recovered corner heights agree across the three independent videos: the sheet is not flat at the 0.2-1 mm level.
So the mat's 3D corner positions are calibrated jointly over all videos (shared corners, per-video intrinsics, per-frame poses) and then used for every pose.
"""
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from . import mat

IDS = sorted(mat.WORLD)
BASE = np.array([mat.WORLD[i] for i in IDS])                      # (26, 4, 3) printed layout


def rodrigues(r):
    """(F, 3) rotation vectors -> (F, 3, 3)."""
    th = np.linalg.norm(r, axis=1)[:, None, None]
    th = np.where(th < 1e-12, 1e-12, th)
    k = r[:, None, :] / th
    kx = np.zeros((len(r), 3, 3))
    kx[:, 0, 1], kx[:, 0, 2], kx[:, 1, 0], kx[:, 1, 2], kx[:, 2, 0], kx[:, 2, 1] = -k[:, 0, 2], k[:, 0, 1], k[:, 0, 2], -k[:, 0, 0], -k[:, 0, 1], k[:, 0, 0]
    return np.eye(3) + np.sin(th) * kx + (1 - np.cos(th)) * (kx @ kx)


def project(R, t, X, intr):
    """R (F,3,3), t (F,3), X (F,N,3) world points -> (F,N,2) pixels, k1,k2,p1,p2 distortion."""
    fx, fy, cx, cy, k1, k2, p1, p2 = intr
    c = np.einsum("fij,fnj->fni", R, X) + t[:, None, :]
    x, y = c[..., 0] / c[..., 2], c[..., 1] / c[..., 2]
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2 * r2
    return np.stack([fx * (x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)) + cx, fy * (y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y) + cy], -1)


def intr_of(K, dist):
    return np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2], dist[0], dist[1], dist[2], dist[3]])


def calibrate_joint(videos: dict, prior=2.0, max_nfev=40, W0=None):
    """videos: name -> dict(obs=[(ids, corners)], poses=[(rvec, tvec)], K, dist). prior: px of penalty per mm a corner moves from the printed layout
    (inf = planar layout, corners fixed). Returns dict(W (26,4,3), per-video intr, per-video poses, rms)."""
    names = list(videos)
    frames = [(n, k) for n in names for k in range(len(videos[n]["obs"]))]
    F, V, nM = len(frames), len(names), len(IDS)
    slots = [np.array([IDS.index(int(i)) for i in videos[n]["obs"][k][0]]) for n, k in frames]
    vi = np.array([names.index(n) for n, _ in frames])
    free_W = np.isfinite(prior)
    W_init = (BASE if W0 is None else W0).copy()
    x0 = np.concatenate([np.concatenate([np.ravel(videos[n]["poses"][k][0]), np.ravel(videos[n]["poses"][k][1])]) for n, k in frames] +
                        [intr_of(videos[n]["K"], videos[n]["dist"]) for n in names] + ([W_init.ravel()] if free_W else []))
    iw = 6 * F + 8 * V

    def unpack(x):
        return x[:6 * F].reshape(F, 6), x[6 * F:iw].reshape(V, 8), (x[iw:].reshape(nM, 4, 3) if free_W else W_init)

    def res(x):
        P, intr, W = unpack(x)
        R = rodrigues(P[:, :3])
        out = []
        for f in range(F):
            pr = project(R[f:f + 1], P[f:f + 1, 3:], W[slots[f]].reshape(1, -1, 3), intr[vi[f]])[0].reshape(-1, 4, 2)
            out.append((pr - videos[frames[f][0]]["obs"][frames[f][1]][1]).ravel())
        if free_W:
            out.append(((W - BASE) * prior).ravel())
        return np.concatenate(out)

    n_res = sum(8 * len(s) for s in slots) + (nM * 12 if free_W else 0)
    S = lil_matrix((n_res, len(x0)), dtype=int)
    r0 = 0
    for f in range(F):
        n = 8 * len(slots[f])
        S[r0:r0 + n, 6 * f:6 * f + 6] = 1
        S[r0:r0 + n, 6 * F + 8 * vi[f]:6 * F + 8 * vi[f] + 8] = 1
        if free_W:
            for j, s in enumerate(slots[f]):
                S[r0 + 8 * j:r0 + 8 * j + 8, iw + 12 * s:iw + 12 * s + 12] = 1
        r0 += n
    if free_W:
        for k in range(nM * 12):
            S[r0 + k, iw + k] = 1
    r = least_squares(res, x0, jac_sparsity=S, loss="soft_l1", f_scale=1.0, x_scale="jac", max_nfev=max_nfev)
    e = res(r.x)[:n_res - (nM * 12 if free_W else 0)]
    P, intr, W = unpack(r.x)
    return {"W": W, "intr": {n: intr[i] for i, n in enumerate(names)}, "rms_px": float(np.sqrt((e.reshape(-1, 2) ** 2).sum(1).mean())),
            "poses": {n: [(P[f, :3], P[f, 3:]) for f in range(F) if frames[f][0] == n] for n in names}}


def world_dict(W):
    return {i: np.asarray(W[k], float) for k, i in enumerate(IDS)}


def save(path: Path, W, intr: dict, info: dict):
    path.write_text(json.dumps({"W": np.asarray(W).tolist(), "intr": {n: np.asarray(v).tolist() for n, v in intr.items()}, "info": info}))


def load(path: Path):
    d = json.loads(path.read_text())
    return np.array(d["W"]), {n: np.array(v) for n, v in d["intr"].items()}, d["info"]


def floor_surface(W, kind="plane"):
    """Callable (x, y) -> mat surface height from the calibrated corners: a least-squares plane, plus (kind='tps') a thin-plate spline."""
    P = W.reshape(-1, 3)
    if kind == "plane":
        A = np.column_stack([np.ones(len(P)), P[:, 0], P[:, 1]])
        c = np.linalg.lstsq(A, P[:, 2], rcond=None)[0]
        return lambda x, y: c[0] + c[1] * np.asarray(x) + c[2] * np.asarray(y)
    from scipy.interpolate import RBFInterpolator
    f = RBFInterpolator(P[:, :2], P[:, 2], kernel="thin_plate_spline", smoothing=1.0)
    return lambda x, y: f(np.column_stack([np.ravel(x), np.ravel(y)])).reshape(np.shape(x))
