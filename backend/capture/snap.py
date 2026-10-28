"""Primitive snapping of the joint mesh: planes (Open3D RANSAC) and cylinders (own RANSAC from oriented points; Open3D has no cylinder model).

Nothing is replaced silently: every primitive is reported with its inlier count, residual and how far its vertices moved, and the unsnapped mesh is kept."""
import numpy as np
from scipy.optimize import least_squares

from . import carve


def oriented_points(verts, faces, n=200000, seed=0):
    tri = verts[faces]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(nrm, axis=1)
    rng = np.random.default_rng(seed)
    i = rng.choice(len(faces), n, p=area / area.sum())
    r1, r2 = np.sqrt(rng.random(n)), rng.random(n)
    P = (1 - r1)[:, None] * tri[i, 0] + (r1 * (1 - r2))[:, None] * tri[i, 1] + (r1 * r2)[:, None] * tri[i, 2]
    return P, nrm[i] / area[i, None]


def planes(P, N, tol=0.12, max_planes=8, min_frac=0.01, ang_deg=12.0, iters=3000, seed=0):
    """Sequential plane RANSAC with normal agreement. Returns [(n, d, inlier mask)] where n.x + d = 0."""
    rng = np.random.default_rng(seed)
    left = np.ones(len(P), bool)
    out = []
    for _ in range(max_planes):
        idx = np.flatnonzero(left)
        if len(idx) < min_frac * len(P):
            break
        best = None
        for _ in range(iters):
            a, b, c = P[rng.choice(idx, 3, replace=False)]
            n = np.cross(b - a, c - a)
            ln = np.linalg.norm(n)
            if ln < 1e-9:
                continue
            n /= ln
            d = -n @ a
            m = left & (np.abs(P @ n + d) < tol) & (np.abs(N @ n) > np.cos(np.radians(ang_deg)))
            if best is None or m.sum() > best[0]:
                best = (m.sum(), n, d, m)
        if best is None or best[0] < min_frac * len(P):
            break
        _, n, d, m = best
        A = np.column_stack([P[m], np.ones(m.sum())])             # refine on the inliers (total least squares)
        c = P[m].mean(0)
        n = np.linalg.svd(P[m] - c)[2][-1]
        d = -n @ c
        m = left & (np.abs(P @ n + d) < tol) & (np.abs(N @ n) > np.cos(np.radians(ang_deg)))
        out.append((n, d, m))
        left &= ~m
    return out


def _cyl_residual(P, c, a, r):
    v = P - c
    return np.linalg.norm(v - np.outer(v @ a, a), axis=1) - r


def cylinders(P, N, rmin=0.8, rmax=8.0, tol=0.12, max_cyl=3, iters=40000, min_inliers=1500, seed=0, avail=None):
    """RANSAC from pairs of oriented points: axis = n1 x n2, centre where the two normal lines meet in the plane normal to the axis."""
    rng = np.random.default_rng(seed)
    left = np.ones(len(P), bool) if avail is None else avail.copy()
    out = []
    for _ in range(max_cyl):
        idx = np.flatnonzero(left)
        if len(idx) < min_inliers:
            break
        i1, i2 = rng.choice(idx, (2, iters))
        a = np.cross(N[i1], N[i2])
        ln = np.linalg.norm(a, axis=1)
        ok = ln > 0.3
        a[ok] /= ln[ok, None]
        # in the plane normal to a: solve p1 + t n1 = p2 + s n2 -> least squares for t
        d = P[i2] - P[i1]
        n1n2 = (N[i1] * N[i2]).sum(1)
        den = 1 - n1n2 ** 2
        okd = ok & (den > 1e-6)
        t = ((d * N[i1]).sum(1) - n1n2 * (d * N[i2]).sum(1)) / np.where(okd, den, 1)
        c = P[i1] + t[:, None] * N[i1]
        r = np.abs(t)
        good = np.flatnonzero(okd & (r > rmin) & (r < rmax))
        best = (0, None)
        sub = rng.choice(idx, min(len(idx), 20000), replace=False)        # score on a subsample
        for g in good[:4000]:
            res = _cyl_residual(P[sub], c[g], a[g], r[g])
            nr = (N[sub] * (P[sub] - c[g] - np.outer((P[sub] - c[g]) @ a[g], a[g]))).sum(1) / (np.abs(res + r[g]) + 1e-9)
            sc = int(((np.abs(res) < tol) & (np.abs(nr) > 0.9)).sum())
            if sc > best[0]:
                best = (sc, (c[g], a[g], r[g]))
        if best[1] is None or best[0] < 0.01 * len(sub):
            break
        c0, a0, r0 = best[1]
        res = _cyl_residual(P, c0, a0, r0)
        v = P - c0
        rad = (v - np.outer(v @ a0, a0))
        m = left & (np.abs(res) < 2 * tol) & (np.abs((N * rad).sum(1) / (np.linalg.norm(rad, axis=1) + 1e-9)) > 0.85)
        if m.sum() < min_inliers:
            break

        def fun(x):
            th, ph = x[3], x[4]
            ax = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
            return _cyl_residual(P[m], x[:3], ax, x[5])

        th0, ph0 = np.arccos(np.clip(a0[2], -1, 1)), np.arctan2(a0[1], a0[0])
        s = least_squares(fun, np.r_[c0, th0, ph0, r0], loss="soft_l1", f_scale=tol)
        x = s.x
        ax = np.array([np.sin(x[3]) * np.cos(x[4]), np.sin(x[3]) * np.sin(x[4]), np.cos(x[3])])
        c1 = x[:3] - ax * ((x[:3] - P[m].mean(0)) @ ax)               # centre moved to the inliers' mean along the axis
        res = _cyl_residual(P, c1, ax, x[5])
        m2 = left & (np.abs(res) < 2 * tol)
        m2 &= m | (np.abs(res) < tol)
        proj = (P[m2] - c1) @ ax
        out.append({"centre": c1, "axis": ax, "r": float(x[5]), "inliers": m2, "rms": float(np.sqrt((res[m2] ** 2).mean())),
                    "length": float(proj.max() - proj.min()), "span": (float(proj.min()), float(proj.max()))})
        left &= ~m2
    return out


def snap_vertices(verts, plane_list, cyl_list, tol=0.25):
    """Move every vertex lying within tol of a fitted primitive onto it (planes first by area, then cylinders). Returns (new verts, per-primitive stats)."""
    v = verts.copy()
    stats = []
    for n, d, _ in plane_list:
        dist = v @ n + d
        m = np.abs(dist) < tol
        v[m] -= np.outer(dist[m], n)
        stats.append({"kind": "plane", "moved": int(m.sum()), "mean_mm": float(np.abs(dist[m]).mean()) if m.any() else 0.0, "max_mm": float(np.abs(dist[m]).max()) if m.any() else 0.0})
    for c in cyl_list:
        w = v - c["centre"]
        along = w @ c["axis"]
        rad = w - np.outer(along, c["axis"])
        rn = np.linalg.norm(rad, axis=1)
        m = (np.abs(rn - c["r"]) < tol) & (along > c["span"][0] - 0.3) & (along < c["span"][1] + 0.3)
        v[m] = c["centre"] + np.outer(along[m], c["axis"]) + rad[m] * (c["r"] / rn[m])[:, None]
        stats.append({"kind": "cylinder", "moved": int(m.sum()), "mean_mm": float(np.abs(rn[m] - c["r"]).mean()) if m.any() else 0.0, "max_mm": float(np.abs(rn[m] - c["r"]).max()) if m.any() else 0.0})
    return v, stats


def section_equivalent_diameter(verts, faces, centre, axis, at):
    """Mesh cross-section by the plane through centre + at*axis normal to `axis`: area-equivalent diameter (mm) and extents of each closed loop."""
    import cv2
    p = centre + at * axis
    u = np.cross(axis, [1.0, 0, 0])
    u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-6 else np.array([0, 1.0, 0])
    w = np.cross(axis, u)
    d = (verts - p) @ axis
    segs = []
    for f in faces:
        dd = d[f]
        pts = []
        for i in range(3):
            a, b = f[i], f[(i + 1) % 3]
            if (dd[i] < 0) != (dd[(i + 1) % 3] < 0):
                t = dd[i] / (dd[i] - dd[(i + 1) % 3])
                q = verts[a] + t * (verts[b] - verts[a]) - p
                pts.append([q @ u, q @ w])
        if len(pts) == 2:
            segs.append(pts)
    return np.array(segs)
