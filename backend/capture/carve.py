"""Robust voxel carving in mat coordinates.

Each view carries a signed-distance image of its mask (px, positive inside). A voxel's value in a view is that SDF at its
projection, converted to mm at the voxel's depth. The robust carve keeps the k-th smallest value over all views that see
the voxel: a voxel is removed only if at least k views put it outside, so one bad mask cannot erase part of the shape.
The zero level of that field is a sub-voxel surface for marching cubes.
"""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy import sparse
from skimage import measure

INF = np.float32(1e9)


@dataclass
class View:
    K: np.ndarray            # 3x3, pinhole (frames are undistorted)
    R: np.ndarray            # 3x3 world -> camera
    t: np.ndarray            # 3
    sdf: np.ndarray          # HxW float32, px, positive inside the mask
    name: str = ""
    video: str = ""

    @property
    def size(self):
        return self.sdf.shape[1], self.sdf.shape[0]

    def moved(self, T: np.ndarray) -> "View":
        """The same camera for points given in a part frame, where `T` (4x4) maps part -> this view's world."""
        return View(self.K, self.R @ T[:3, :3], self.R @ T[:3, 3] + self.t, self.sdf, self.name, self.video)


def mask_sdf(mask: np.ndarray) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    din = cv2.distanceTransform(m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    dout = cv2.distanceTransform(1 - m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return np.where(m > 0, din - 0.5, -(dout - 0.5)).astype(np.float32)


def project(view: View, X: np.ndarray):
    """(u, v, depth) of world points (N, 3)."""
    c = X @ view.R.T + view.t
    z = c[:, 2]
    zs = np.where(z > 1e-6, z, 1.0)
    return view.K[0, 0] * c[:, 0] / zs + view.K[0, 2], view.K[1, 1] * c[:, 1] / zs + view.K[1, 2], z


def view_values(view: View, X: np.ndarray) -> np.ndarray:
    """Signed distance in mm of each point to the view's silhouette; INF where the point is not seen (behind / outside image)."""
    u, v, z = project(view, X)
    w, h = view.size
    seen = (z > 1e-6) & (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
    n = len(u)
    pad = (-n) % 1024                                  # cv2.remap needs both map dimensions below 32767
    mu = np.pad(u.astype(np.float32), (0, pad)).reshape(-1, 1024)
    mv = np.pad(v.astype(np.float32), (0, pad)).reshape(-1, 1024)
    val = cv2.remap(view.sdf, mu, mv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).ravel()[:n]
    f = 0.5 * (view.K[0, 0] + view.K[1, 1])
    return np.where(seen, val * z / f, INF).astype(np.float32)


class Grid:
    def __init__(self, lo, hi, vs):
        self.vs = float(vs)
        self.lo = np.asarray(lo, float)
        self.shape = tuple(int(np.ceil((h - l) / vs)) + 1 for l, h in zip(lo, hi))

    def points(self) -> np.ndarray:
        ax = [self.lo[i] + self.vs * np.arange(n) for i, n in enumerate(self.shape)]
        return np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)


def carve_field(views: list[View], X: np.ndarray, k: int = 2, floor_mm: float | None = 0.0, clip_mm: float = 3.0,
                return_count: bool = False):
    """Robust field at points X (N, 3): the k-th smallest per-view distance (mm, >0 inside), clipped to +-clip_mm.

    floor_mm: voxels with z below this are outside (the mat plane), a hard constraint, not a vote. None disables it.
    Points seen by fewer than k views are +clip (never carved); return_count also gives the number of views seeing each.
    """
    best = np.full((k, len(X)), INF, np.float32)         # ascending
    n_seen = np.zeros(len(X), np.int16)
    for v in views:
        val = view_values(v, X)
        n_seen += val < INF
        for j in range(k):                                  # insert into the sorted k smallest
            swap = val < best[j]
            best[j], val = np.where(swap, val, best[j]), np.where(swap, best[j], val)
    f = np.clip(best[k - 1], -clip_mm, clip_mm)
    f = np.where(best[k - 1] >= INF, clip_mm, f)
    if floor_mm is not None:
        f = np.minimum(f, np.clip(X[:, 2] - floor_mm, -clip_mm, clip_mm))
    return (f, n_seen) if return_count else f


def carve(views, lo, hi, vs=0.25, k=2, floor_mm=0.0, chunk=2_000_000):
    """Carve a box. Returns (Grid, field (nx, ny, nz) float32, seen-count (nx, ny, nz))."""
    g = Grid(lo, hi, vs)
    X = g.points()
    f = np.empty(len(X), np.float32)
    n = np.empty(len(X), np.int16)
    for s in range(0, len(X), chunk):
        f[s:s + chunk], n[s:s + chunk] = carve_field(views, X[s:s + chunk], k, floor_mm, return_count=True)
    return g, f.reshape(g.shape), n.reshape(g.shape)


def soft_volume(field: np.ndarray, vs: float) -> float:
    """Smooth volume in mm3 of the inside (field > 0), partial credit within half a voxel of the surface."""
    return float(np.clip(field / vs + 0.5, 0, 1).sum() * vs ** 3)


# ---------------------------------------------------------------- mesh

def largest_component(field, g):
    """Zero out inside voxels not connected to the largest inside blob (floaters from stray votes)."""

    from scipy import ndimage
    lab, n = ndimage.label(field > 0)
    if n <= 1:
        return field
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
    keep = 1 + int(np.argmax(sizes))
    out = field.copy()
    out[(lab != keep) & (field > 0)] = -0.01
    return out


def to_mesh(field: np.ndarray, g: Grid, keep_largest=True):
    """Marching cubes at the zero level. The field is padded outside so the surface is closed. (verts mm, faces)."""
    f = largest_component(field, g) if keep_largest else field
    f = np.pad(f, 1, constant_values=-1.0)
    verts, faces, _, _ = measure.marching_cubes(f, 0.0, spacing=(g.vs,) * 3)
    return verts + g.lo - g.vs, faces[:, ::-1]      # skimage winds inward for an inside-positive field


def taubin(verts, faces, iters=10, lam=0.5, mu=-0.53):
    """Taubin lambda|mu smoothing with a uniform Laplacian (volume preserving, light)."""
    n = len(verts)
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    A = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(float)
    W = sparse.diags(1.0 / np.maximum(np.asarray(A.sum(1)).ravel(), 1)) @ A
    v = verts.copy()
    for _ in range(iters):
        v = v + lam * (W @ v - v)
        v = v + mu * (W @ v - v)
    return v


def edge_census(verts, faces) -> dict:
    """Watertight / manifold check on an indexed mesh."""
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    tri = verts[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    vol = float(np.einsum("ij,ij->", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6)
    return {"triangles": int(len(faces)), "boundary_edges": int((cnt == 1).sum()), "non_manifold_edges": int((cnt > 2).sum()),
            "degenerate_triangles": int((area < 1e-12).sum()), "watertight": bool((cnt == 2).all()), "volume_mm3": vol}


def write_stl(path, verts, faces):
    tri = verts[faces].astype("<f4")
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm = (nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-20)).astype("<f4")
    rec = np.zeros(len(faces), np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    rec["n"], rec["v"] = nrm, tri
    with open(path, "wb") as fh:
        fh.write(b"SnapPart carve".ljust(80, b" "))
        fh.write(np.uint32(len(faces)).tobytes())
        fh.write(rec.tobytes())


def sample_surface(verts, faces, n=60000, seed=0) -> np.ndarray:
    tri = verts[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    rng = np.random.default_rng(seed)
    i = rng.choice(len(faces), n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    return (1 - s)[:, None] * tri[i, 0] + (s * (1 - r2))[:, None] * tri[i, 1] + (s * r2)[:, None] * tri[i, 2]


def surface_distance(a, b, n=60000) -> dict:
    """Symmetric surface distances (mm) between two meshes given as (verts, faces)."""
    from scipy.spatial import cKDTree
    pa, pb = sample_surface(*a, n=n), sample_surface(*b, n=n)
    d_ab, d_ba = cKDTree(pb).query(pa)[0], cKDTree(pa).query(pb)[0]
    d = np.concatenate([d_ab, d_ba])
    return {"mean_mm": float(d.mean()), "p95_mm": float(np.percentile(d, 95)), "max_mm": float(d.max()),
            "a_to_b_mean_mm": float(d_ab.mean()), "b_to_a_mean_mm": float(d_ba.mean())}


def obb(verts) -> dict:
    """Oriented bounding box by PCA of the vertices, then tight extents along those axes (mm, sorted descending)."""
    c = verts - verts.mean(0)
    _, vec = np.linalg.eigh(np.cov(c.T))
    ext = (c @ vec).max(0) - (c @ vec).min(0)
    order = np.argsort(-ext)
    return {"extents_mm": [float(e) for e in ext[order]], "axes": vec[:, order].T.tolist()}
