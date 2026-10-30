"""Quality gates every STL passes before it is called ready, and per-dimension error bars.

Gates: watertight and manifold; a single body (the carve keeps the largest component; anything else here is flagged); no zero-thickness
skins (voxels whose two opposite neighbours along an axis are both outside); size and volume sane for the printed mat; held-out IoU above
a threshold (when it was measured); roughness of snapped faces (reported, no threshold). A failed gate makes the model a DRAFT with plain
reasons. Error bars combine measured spreads (held-out folds, registration) with the voxel size; they are estimates, never an accuracy
percentage."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from . import carve, config as C, snap

MIN_HELD_OUT_IOU = 0.90          # the browser-path part B run measured 0.936 (v3); the 4K path 0.953
MIN_DIM_MM = 2.0
MAX_DIMS_MM = (C.BOX_HI[0] - C.BOX_LO[0], C.BOX_HI[1] - C.BOX_LO[1], C.BOX_HI[2] - C.BOX_LO[2])   # the carve box on the mat


def components(faces, n_verts) -> tuple[int, np.ndarray]:
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    A = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n_verts, n_verts))
    return connected_components(A, directed=False)


def thin_skins(field, vs, min_voxels=50) -> list[dict]:
    """Zero-thickness skins: connected sheets (>= min_voxels) of inside voxels that are one voxel thick along some axis (both opposite
    neighbours outside). Isolated thin voxels (e.g. at the mat floor) are noise and not counted."""
    from scipy import ndimage
    occ = field > 0
    thin = np.zeros_like(occ)
    for ax in range(3):
        thin |= occ & ~np.roll(occ, 1, ax) & ~np.roll(occ, -1, ax)
    lab, n = ndimage.label(thin, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(lab.ravel())[1:]
    out = []
    for i in np.nonzero(sizes >= min_voxels)[0]:
        idx = np.argwhere(lab == i + 1)
        out.append({"voxels": int(sizes[i]), "area_mm2": round(float(sizes[i] * vs * vs), 1), "index_centre": idx.mean(0).round(1).tolist(),
                    "extent_mm": ((idx.max(0) - idx.min(0)) * vs).round(2).tolist()})
    return out


def roughness(verts, faces, n=40000, max_planes=6) -> list[dict]:
    """RMS distance (mm) of each large planar face's samples to its plane: reported for snapped / flat faces."""
    P, N = snap.oriented_points(verts, faces, n)
    out = []
    for nn, d, m in snap.planes(P, N, tol=0.15, max_planes=max_planes, min_frac=0.02, iters=600):
        r = P[m] @ nn + d
        out.append({"normal": nn.round(3).tolist(), "samples": int(m.sum()), "rms_mm": round(float(np.sqrt((r ** 2).mean())), 3)})
    return out


def check(verts, faces, field=None, held_out_iou=None, grid=None) -> dict:
    cen = carve.edge_census(verts, faces)
    ncomp, lab = components(faces, len(verts))
    pl = carve.obb_plan(verts)
    dims = (pl["length_mm"], pl["width_mm"], pl["z_max"] - pl["z_min"])
    skin = thin_skins(field, grid.vs if grid is not None else C.VOXEL_MM) if field is not None else None
    if skin and grid is not None:
        for k in skin:
            k["centre_mm"] = (grid.lo + np.asarray(k.pop("index_centre")) * grid.vs).round(1).tolist()
    gates = [
        {"gate": "watertight_manifold", "ok": bool(cen["watertight"] and cen["non_manifold_edges"] == 0),
         "value": {"boundary_edges": cen["boundary_edges"], "non_manifold_edges": cen["non_manifold_edges"]},
         "reason": "The model has holes or bad edges, so it may not print."},
        {"gate": "single_body", "ok": ncomp == 1, "value": {"bodies": int(ncomp)}, "reason": f"The model has {ncomp} separate pieces."},
        {"gate": "no_zero_thickness", "ok": None if skin is None else not skin, "value": {"skins": skin},
         "reason": "Parts of the model are paper-thin (one voxel thick), which usually means a wrong surface."},
        {"gate": "size_sane", "ok": all(MIN_DIM_MM <= d <= m for d, m in zip(sorted(dims, reverse=True), sorted(MAX_DIMS_MM, reverse=True)))
         and cen["volume_mm3"] > 0, "value": {"dims_mm": [round(d, 2) for d in dims], "volume_mm3": round(float(cen["volume_mm3"]), 1)},
         "reason": "The size does not fit a part filmed on the printed mat."},
        {"gate": "held_out_iou", "ok": None if held_out_iou is None else held_out_iou >= MIN_HELD_OUT_IOU,
         "value": {"held_out_iou": held_out_iou, "min": MIN_HELD_OUT_IOU},
         "reason": "The model does not match the photos it was not built from well enough."},
    ]
    failed = [g for g in gates if g["ok"] is False]
    return {"status": "draft" if failed else "ready", "gates": gates, "reasons": [g["reason"] for g in failed],
            "not_run": [g["gate"] for g in gates if g["ok"] is None], "roughness": roughness(verts, faces)}


def error_bars(fold_dims: list[dict], registration_gap_mm: float | None, voxel_mm: float) -> dict:
    """+- per dimension (mm, estimated): sqrt(spread of the held-out folds' dimensions^2 + (registration gap / 2)^2 + (voxel / 2)^2).
    fold_dims: [{"length", "width", "height"}] from carves that each left some views out."""
    out = {}
    for k in ("length", "width", "height"):
        xs = [d[k] for d in fold_dims] if fold_dims else []
        sd = float(np.std(xs)) if len(xs) > 1 else 0.0
        out[k] = round(float(np.sqrt(sd ** 2 + ((registration_gap_mm or 0.0) / 2) ** 2 + (voxel_mm / 2) ** 2)), 2)
    return {"plus_minus_mm": out, "status": "estimated",
            "how": "spread of held-out carves, half the registration gap and half a voxel, combined; not an accuracy percentage"}
