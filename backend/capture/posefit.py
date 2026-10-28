"""Silhouette-based pose refinement. Marker poses carry ~0.3 mm of inconsistency at the part (measured: the robust hull sits ~0.4 mm inside every mask).
Each view's pose gets a small camera-frame correction (rotation + translation) that puts the contour of the model carved from all views onto that view's mask edge.
Corrections are re-estimated over a few carve/refine rounds. A view excluded from the fit keeps its marker pose."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from . import carve, holdout


def vertex_normals(v, f):
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, f[:, k], fn)
    return vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12)


def correct(view: carve.View, d: np.ndarray) -> carve.View:
    """Apply a camera-frame correction d = (rotvec, translation mm) to a view."""
    Rd = Rotation.from_rotvec(d[:3]).as_matrix()
    return carve.View(view.K, Rd @ view.R, Rd @ view.t + d[3:], view.sdf, view.name, view.video)


def fit_view(mv: carve.View, v, vn, d0=None, gate_mm=2.0, normal_tol=0.2, sig_c=0.15, sig_r=0.0017, sig_t=0.4, max_rot=0.006, max_t=1.0):
    """Total correction (6,) of the marker-pose view `mv` against mesh vertices (part frame) and normals.

    MAP estimate: contour residuals (mm, sigma sig_c) plus a prior pulling the correction to zero (sigma sig_r rad, sig_t mm), so poses cannot drift
    along the optical axis (depth and scale are nearly interchangeable for silhouettes). Returns (d, n_used, rms_before, rms_after)."""
    d0 = np.zeros(6) if d0 is None else d0
    cur = correct(mv, d0)
    c = -cur.R.T @ cur.t
    dirs = c - v
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    cand = np.abs((vn * dirs).sum(1)) < normal_tol
    X = v[cand]
    keep = np.abs(carve.view_values(cur, X)) < gate_mm
    X = X[keep]
    if len(X) < 50:
        return d0, len(X), np.nan, np.nan
    ok = carve.view_values(cur, X) < 1e8
    prior = np.array([sig_r] * 3 + [sig_t] * 3)

    def res(d):
        r = np.where(ok, carve.view_values(correct(mv, d), X), 0.0) / sig_c
        return np.concatenate([r, d / prior])

    sol = least_squares(res, d0, loss="soft_l1", f_scale=2.0, x_scale=np.array([0.002] * 3 + [0.3] * 3),
                        bounds=([-max_rot] * 3 + [-max_t] * 3, [max_rot] * 3 + [max_t] * 3))
    rms = lambda d: float(np.sqrt((carve.view_values(correct(mv, d), X)[ok] ** 2).mean()))
    return sol.x, int(ok.sum()), rms(d0), rms(sol.x)


def fit_poses(views_by_video, T, k=8, vs=0.4, rounds=3, exclude=(), ref=None, log=print, grow=0.0, **kw):
    """{(video, index): total correction (6,)} for every view not excluded; each round re-fits from the marker poses against the model of the previous round."""
    corr = {(n, i): np.zeros(6) for n, vs_ in views_by_video.items() for i in range(len(vs_)) if (n, i) not in exclude}
    orig = {key: views_by_video[key[0]][key[1]].moved(T[key[0]]) for key in corr}
    floors = [T[n] for n in views_by_video if n != ref]
    for rnd in range(rounds):
        cur = [correct(orig[key], d) for key, d in corr.items()]
        v, f = holdout.carve_mesh(cur, vs, k, floors, grow=grow)
        vn = vertex_normals(v, f)
        before, after = [], []
        for key in list(corr):
            d, n, r0, r1 = fit_view(orig[key], v, vn, corr[key], **kw)
            if n >= 50:
                corr[key] = d
                before.append(r0)
                after.append(r1)
        log(f"   pose fit round {rnd}: contour rms {np.nanmean(before):.3f} -> {np.nanmean(after):.3f} mm over {len(before)} views; mean |dR| "
            f"{np.degrees(np.mean([np.linalg.norm(d[:3]) for d in corr.values()])):.3f} deg, mean |dt| {np.mean([np.linalg.norm(d[3:]) for d in corr.values()]):.3f} mm")
    return corr
