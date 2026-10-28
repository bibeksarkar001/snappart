"""Synthetic capture rig for tests: a known part, virtual cameras around it, rendered mat images with markers."""
import cadquery as cq
import cv2
import numpy as np

from . import mat

CENTRE = np.array([mat.PAGE_W / 2, mat.PAGE_H / 2])


def test_part(bar_d=4.0):
    """Plate 34 x 30 x 6 with a 22 x 12 through-slot and a round bar (default 4.0 mm) across the slot, flat on z = 0, at the page centre."""
    plate = cq.Workplane("XY").box(34, 30, 6, centered=(True, True, False)).cut(
        cq.Workplane("XY").box(22, 12, 20, centered=(True, True, True)))
    bar = cq.Workplane("YZ").workplane(offset=-14).center(0, 3).circle(bar_d / 2).extrude(28)
    solid = plate.union(bar).translate((CENTRE[0], CENTRE[1], 0))
    v, t = solid.val().tessellate(0.01, 0.1)
    return np.array([[p.x, p.y, p.z] for p in v]), np.array(t)


def look_at(pos, target):
    f = target - pos
    f = f / np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    R = np.stack([r, d, f])
    return R, -R @ pos


def camera_ring(n=60, dist=330.0, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        az = 2 * np.pi * i / n + rng.normal(0, 0.03)
        el = np.radians(15 + 74 * ((i * 0.381966) % 1))       # elevations 15-89 deg, spread by the golden ratio
        c = np.array([*CENTRE, 3.0])
        out.append(look_at(c + dist * np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)]), c))
    return out


def render_silhouette(verts, faces, K, R, t, size, ss=4):
    """Anti-aliased silhouette (0/255) of a mesh: triangles filled on a ss-times canvas then reduced."""
    w, h = size
    c = verts @ R.T + t
    uv = (c[:, :2] / c[:, 2:3]) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    big = np.zeros((h * ss, w * ss), np.uint8)
    pts = np.round((uv[faces] + 0.5) * ss - 0.5).astype(np.int32)
    for tri in pts:
        cv2.fillConvexPoly(big, tri, 255)
    return (cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA) > 127).astype(np.uint8) * 255


def render_mat_view(canvas, ppm, K, R, t, size):
    """The mat seen from a camera: canvas pixels (u, v) = (x, page_h - y) * ppm warped through the plane z = 0."""
    Hc = K @ np.column_stack([R[:, 0], R[:, 1], t]) @ np.array([[1 / ppm, 0, 0], [0, -1 / ppm, mat.PAGE_H], [0, 0, 1]])
    return cv2.warpPerspective(cv2.GaussianBlur(canvas, (0, 0), 0.8), Hc, size, flags=cv2.INTER_LINEAR, borderValue=170)


def build(k=2, n=60, bar_d=4.0, vs=0.25, size=(1920, 1080), f=1900.0, taubin_iters=5):
    """Full pipeline on the rig: render mat + silhouettes, detect markers, estimate poses, carve.

    Returns truth (verts, faces), carved (verts, faces), per-view (camera-centre error mm, reprojection rms px).
    """
    from . import carve
    K = np.array([[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1.0]])
    verts, faces = test_part(bar_d)
    canvas, det = mat.render_mat(8.0), mat.make_detector()
    views, perr = [], []
    for i, (R, t) in enumerate(camera_ring(n)):
        ids, cs = mat.detect(det, render_mat_view(canvas, 8.0, K, R, t, size))
        pose = mat.solve_pose(ids, cs, K, None)
        assert pose is not None and len(pose[3]) >= 4, f"view {i}: markers {len(ids)}"
        Rd = cv2.Rodrigues(pose[0])[0]
        perr.append((np.linalg.norm(-Rd.T @ pose[1].ravel() + R.T @ t), pose[2]))
        views.append(carve.View(K, Rd, pose[1].ravel(), carve.mask_sdf(render_silhouette(verts, faces, K, R, t, size)), f"v{i}"))
    g, fld, _ = carve.carve(views, [CENTRE[0] - 25, CENTRE[1] - 25, -1.5], [CENTRE[0] + 25, CENTRE[1] + 25, 12], vs, k)
    v, fc = carve.to_mesh(fld, g)
    return (verts, faces), (carve.taubin(v, fc, taubin_iters), fc), np.array(perr)


def report(truth, mesh, perr) -> dict:
    """Surface distances overall and by height band: the visual hull of a flat part is inflated above its top face."""
    from scipy.spatial import cKDTree
    from . import carve
    pm, pt = carve.sample_surface(*mesh, n=100000), carve.sample_surface(*truth, n=100000)
    d_m, d_t = cKDTree(pt).query(pm)[0], cKDTree(pm).query(pt)[0]
    zone = lambda lo, hi: (lambda s: {"mean_mm": float(d_m[s].mean()), "max_mm": float(d_m[s].max()), "n": int(s.sum())})((pm[:, 2] >= lo) & (pm[:, 2] < hi))
    return {"overall": carve.surface_distance(truth, mesh), "census": carve.edge_census(*mesh), "truth_volume_mm3": carve.edge_census(*truth)["volume_mm3"],
            "carved_pts_by_z_band": {"z -1..1": zone(-1, 1), "z 1..5.5 (side walls, slot)": zone(1, 5.5), "z 5.5..6.5 (top face)": zone(5.5, 6.5), "z >6.5 (above the part)": zone(6.5, 99)},
            "carved_extent_mm": (mesh[0].max(0) - mesh[0].min(0)).tolist(), "truth_extent_mm": (truth[0].max(0) - truth[0].min(0)).tolist(),
            "camera_centre_err_mm": {"mean": float(perr[:, 0].mean()), "max": float(perr[:, 0].max())}, "reproj_rms_px_max": float(perr[:, 1].max())}
