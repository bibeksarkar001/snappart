"""Registration of the videos into one part frame (the frame of the reference video's mat).

T_v maps part-frame points to video v's mat frame (4x4). Step 1: trimmed ICP between per-video hull surfaces from many initial rotations (24 axis
rotations x yaw steps), ranked by trimmed chamfer distance. Step 2: refine each T_v by maximising the joint robust hull volume over ALL views
(silhouette agreement): a correct alignment keeps every cone consistent, so the intersection is largest. Voxels a transformed video cannot see
count as outside, so a transform cannot win by moving the part out of that video's images."""
import itertools

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from . import carve


def cube_rotations() -> list[np.ndarray]:
    out = []
    for perm in itertools.permutations(range(3)):
        for sg in itertools.product([1, -1], repeat=3):
            R = np.zeros((3, 3))
            for i, (p, s) in enumerate(zip(perm, sg)):
                R[i, p] = s
            if np.linalg.det(R) > 0:
                out.append(R)
    return out


def rz(deg):
    return Rotation.from_euler("z", deg, degrees=True).as_matrix()


def kabsch(A, B):
    """Rigid (R, t) minimising |R A + t - B|."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cb - R @ ca


def icp(src, dst, tree, R, t, iters=25, trim=0.7):
    """Trimmed point-to-point ICP; returns (R, t, cost) with cost = symmetric trimmed mean nearest distance (mm)."""
    for _ in range(iters):
        p = src @ R.T + t
        d, i = tree.query(p)
        keep = d <= np.quantile(d, trim)
        R2, t2 = kabsch(src[keep], dst[i[keep]])
        R, t = R2, t2
    p = src @ R.T + t
    d1 = tree.query(p)[0]
    d2 = cKDTree(p).query(dst)[0]
    tr = lambda d: np.sort(d)[: int(trim * len(d))].mean()
    return R, t, float(tr(d1) + tr(d2))


def candidates(src, dst, yaw_step=20, topn=6, trim=0.7, sep_deg=8.0, sep_mm=2.0):
    """All initial rotations -> ICP -> distinct best solutions. src/dst are (N,3) hull surface points; result maps src -> dst."""
    tree = cKDTree(dst)
    cs, cd = src.mean(0), dst.mean(0)
    sols = []
    for Rc in cube_rotations():
        for yaw in range(0, 360, yaw_step):
            R0 = rz(yaw) @ Rc
            R, t, c = icp(src, dst, tree, R0, cd - R0 @ cs, trim=trim)
            sols.append((c, R, t))
    sols.sort(key=lambda s: s[0])
    out = []
    for c, R, t in sols:
        if all(np.degrees(Rotation.from_matrix(R @ o[1].T).magnitude()) > sep_deg or np.linalg.norm(t - o[2]) > sep_mm for o in out):
            out.append((c, R, t))
        if len(out) >= topn:
            break
    return out


def hom(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def apply_delta(T, x, centre):
    """T composed with a small change x = (rotvec, translation[, log scale]) applied about `centre` (part frame) before T. A scale makes T a similarity:
    the video's mat scale (marker size, intrinsics) may differ from the others by a few tenths of a percent."""
    s = np.exp(x[6]) if len(x) > 6 else 1.0
    D = hom(s * Rotation.from_rotvec(x[:3]).as_matrix(), np.zeros(3))
    D[:3, 3] = x[3:6] + centre - D[:3, :3] @ centre
    return T @ D


def joint_volume(views_by_video, T_by_video, X, k=2, vs=0.5, min_seen=0.8, grow=0.0):
    """Soft volume (mm3) of the joint hull over candidate points X (part frame), views moved by T_v, floors enforced, unseen = outside."""
    n_views = sum(len(v) for v in views_by_video.values())
    best = np.full((k, len(X)), carve.INF, np.float32)
    seen = np.zeros(len(X))
    floors = []
    for name, views in views_by_video.items():
        T = T_by_video[name]
        floors.append(T)
        for v in views:
            val = carve.view_values(v.moved(T), X)
            seen += val < carve.INF
            if grow:
                val = np.where(val < carve.INF, val + grow, val)
            for j in range(k):
                sw = val < best[j]
                best[j], val = np.where(sw, val, best[j]), np.where(sw, best[j], val)
    f = np.clip(best[k - 1], -3, 3)
    f = np.where(best[k - 1] >= carve.INF, -3.0, f)
    f = np.where(seen < min_seen * n_views, -3.0, f)
    for T in floors:
        f = np.minimum(f, np.clip(X @ T[2, :3] + T[2, 3], -3, 3))
    return float(np.clip(f / vs + 0.5, 0, 1).sum() * vs ** 3)


def refine(views_by_video, T_init, free, X, centre, k=2, vs=0.5, log=print, maxiter=3, grow=0.0, with_scale=False):
    """Powell refinement of the free videos' transforms (6 parameters each) against the joint volume. Returns (T_by_video, history)."""
    T = {n: m.copy() for n, m in T_init.items()}
    hist = [joint_volume(views_by_video, T, X, k, vs, grow=grow)]
    scale = np.array([0.01, 0.01, 0.01, 0.2, 0.2, 0.2] + ([0.002] if with_scale else []))
    for rnd in range(maxiter):
        for n in free:
            base = T[n].copy()

            def cost(x):
                T[n] = apply_delta(base, x * scale, centre)
                return -joint_volume(views_by_video, T, X, k, vs, grow=grow)

            r = minimize(cost, np.zeros(len(scale)), method="Powell", options={"xtol": 0.05, "ftol": 1e-4, "maxiter": 6})
            T[n] = apply_delta(base, r.x * scale, centre)
            hist.append(-r.fun)
            log(f"   round {rnd} {n}: joint volume {hist[-2]:.0f} -> {hist[-1]:.0f} mm3, step rot {np.degrees(np.linalg.norm(r.x[:3] * scale[:3])):.2f} deg, trans {np.linalg.norm(r.x[3:6] * scale[3:6]):.2f} mm" + (f", scale {np.exp(r.x[6] * scale[6]):.4f}" if with_scale else ""))
        if abs(hist[-1] - hist[-1 - len(free)]) < 0.002 * hist[-1]:
            break
    return T, hist


def candidates_fast(src, dst, coarse_n=800, yaw_step=60, topn=3, coarse_iters=12, fine_iters=30, trim=0.7, seed=0):
    """Coarse-to-fine version of candidates(): ICP from 24 axis rotations x (360 / yaw_step) yaws on coarse_n random points of each cloud
    (coarse_iters iterations), then the topn distinct results re-run on all points (fine_iters). Sorted by the fine trimmed chamfer."""
    rng = np.random.default_rng(seed)
    s = src[rng.choice(len(src), min(coarse_n, len(src)), replace=False)]
    d = dst[rng.choice(len(dst), min(coarse_n, len(dst)), replace=False)]
    tree = cKDTree(d)
    cs, cd = s.mean(0), d.mean(0)
    sols = []
    for Rc in cube_rotations():
        for yaw in range(0, 360, yaw_step):
            R0 = rz(yaw) @ Rc
            sols.append(icp(s, d, tree, R0, cd - R0 @ cs, iters=coarse_iters, trim=trim))
    sols.sort(key=lambda q: q[2])
    keep = []
    for R, t, c in sols:
        if all(np.degrees(Rotation.from_matrix(R @ o[0].T).magnitude()) > 8 or np.linalg.norm(t - o[1]) > 2 for o in keep):
            keep.append((R, t, c))
        if len(keep) >= topn:
            break
    full = cKDTree(dst)
    out = [icp(src, dst, full, R, t, iters=fine_iters, trim=trim) for R, t, _ in keep]
    return sorted([(c, R, t) for R, t, c in out], key=lambda q: q[0])
