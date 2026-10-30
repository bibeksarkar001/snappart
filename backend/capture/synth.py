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


# ---------------------------------------------------------------- generalisation set (backend stage 5): CadQuery parts, noisy capture

def _at_centre(wp):
    """Tessellated mesh of a CadQuery solid, moved so its footprint is centred on the page and its bottom is on the mat (z = 0)."""
    bb = wp.val().BoundingBox()
    solid = wp.translate((CENTRE[0] - (bb.xmin + bb.xmax) / 2, CENTRE[1] - (bb.ymin + bb.ymax) / 2, -bb.zmin))
    v, t = solid.val().tessellate(0.02, 0.2)
    return np.array([[p.x, p.y, p.z] for p in v]), np.array(t)


def parts() -> dict:
    """Seven synthetic parts covering the shapes SnapPart targets (all made here with CadQuery; no real data)."""
    W = cq.Workplane("XY")
    ridge_profile = W.workplane().polyline([(-1.5, 0), (1.5, 0), (0.4, 2.5), (-0.4, 2.5)]).close()
    return {
        "box": W.box(30, 20, 10),
        "cylinder": W.circle(10).extrude(12),
        "L-bracket": W.box(30, 20, 5, centered=(False, True, False)).union(W.box(8, 20, 15, centered=(False, True, False))),
        "plate with holes": W.box(40, 30, 4).faces(">Z").workplane().pushPoints([(-10, 0), (10, 0)]).circle(2.5).cutThruAll()
                              .faces(">Z").workplane().pushPoints([(10, 0)]).circle(4.0).cutThruAll(),
        "plate with slots": W.box(40, 30, 4).faces(">Z").workplane().pushPoints([(-8, 0), (8, 0)]).slot2D(10, 3, 90).cutThruAll(),
        "hill ridge part": W.box(30, 20, 4, centered=(True, True, False)).union(
            cq.Workplane("YZ").polyline([(-1.5, 4), (1.5, 4), (0.4, 6.5), (-0.4, 6.5)]).close().extrude(20).translate((-10, 0, 0))),
        "thin-walled part": W.box(30, 20, 12, centered=(True, True, False)).faces(">Z").shell(-1.2),
    }


def noisy_silhouette(mask, rng, noise_px=0.5):
    """Sub-pixel boundary jitter (blur, threshold moved by +-noise) and sometimes a 1 px erosion or dilation: SAM-like edge error."""
    m = cv2.GaussianBlur(mask.astype(np.float32) / 255, (0, 0), 1.0)
    m = (m > 0.5 + rng.uniform(-0.25, 0.25) * noise_px).astype(np.uint8) * 255
    r = rng.random()
    if r < 0.2:
        m = cv2.erode(m, np.ones((3, 3), np.uint8))
    elif r < 0.4:
        m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    return m


def ring(n, el_lo, el_hi, dist=330.0, seed=0, phase=0.0):
    """n cameras round the page centre, elevations spread between el_lo and el_hi (deg)."""
    rng = np.random.default_rng(seed)
    c = np.array([*CENTRE, 3.0])
    out = []
    for i in range(n):
        az = 2 * np.pi * i / n + phase + rng.normal(0, 0.03)
        el = np.radians(el_lo + (el_hi - el_lo) * ((i * 0.381966) % 1))
        out.append(look_at(c + dist * np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)]), c))
    return out


def flip_transform(verts):
    """4x4 T mapping the part frame to the flipped video's mat frame: turned over about the x axis through its centre, resting on z = 0."""
    lo, hi = verts.min(0), verts.max(0)
    c = (lo + hi) / 2
    R = np.diag([1.0, -1.0, -1.0])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array([CENTRE[0], CENTRE[1], 0.0]) - R @ np.array([c[0], c[1], hi[2]])
    return T


def build_part(verts, faces, n=60, vs=0.25, size=(1920, 1080), f=1900.0, seed=0, hold=10, noise_px=0.5, image_noise=3.0, k=None):
    """A noisy three-video capture of one part on the mat, carved like the real pipeline: video 1 high orbit (el 45-85), video 2 low orbit
    (el 12-35), video 3 the part turned over (el 20-80); poses from detected markers on a noisy rendered mat; silhouettes with SAM-like
    edge error; registration of video 3 from the known flip (perfect: registration error is not simulated); per-video mat floors; k scaled
    with the view count; silhouettes grown by GROW_MM. `hold` views (spread over all videos) are left out for the held-out IoU."""
    from . import carve, config as C, learnings as L
    rng = np.random.default_rng(seed)
    K = np.array([[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1.0]])
    canvas, det = mat.render_mat(8.0), mat.make_detector()
    Tf = flip_transform(verts)
    vf = verts @ Tf[:3, :3].T + Tf[:3, 3]
    per = n // 3
    rigs = [(np.eye(4), verts, ring(per, 45, 85, seed=seed)), (np.eye(4), verts, ring(per, 12, 35, seed=seed + 1, phase=0.4)),
            (Tf, vf, ring(n - 2 * per, 20, 80, seed=seed + 2, phase=0.8))]
    views, errs = [], []
    for T, vw, cams in rigs:
        for i, (R, t) in enumerate(cams):
            img = render_mat_view(canvas, 8.0, K, R, t, size)
            img = np.clip(img + rng.normal(0, image_noise, img.shape), 0, 255).astype(np.uint8)
            ids, cs = mat.detect(det, img)
            pose = mat.solve_pose(ids, cs, K, None)
            if pose is None or len(pose[3]) < 4:
                continue
            Rd = cv2.Rodrigues(pose[0])[0]
            errs.append(float(np.linalg.norm(-Rd.T @ pose[1].ravel() + R.T @ t)))
            m = noisy_silhouette(render_silhouette(vw, faces, K, R, t, size), rng, noise_px)
            views.append(carve.View(K, Rd, pose[1].ravel(), carve.mask_sdf(m), f"v{len(views)}").moved(T))
    held = set(range(0, len(views), max(len(views) // hold, 1))[:hold])
    train = [v for i, v in enumerate(views) if i not in held]
    lo, hi = verts.min(0) - 4, verts.max(0) + 4
    lo[2] = -1.5
    g, fld, _ = carve.carve(train, lo, hi, vs, k or L.k_for_views(len(train)), grow_mm=C.GROW_MM, floors=[Tf])
    v, fc = carve.to_mesh(fld, g)
    v = carve.taubin(v, fc, C.TAUBIN_ITERS)
    iou = float(np.mean([carve.view_iou(v, fc, views[i]) for i in sorted(held)]))
    return {"mesh": (v, fc), "held_out_iou": iou, "camera_centre_err_mm": float(np.max(errs)), "views": len(views), "train_views": train, "floors": [Tf]}


def part_metrics(truth, res) -> dict:
    from . import carve
    pt, pc = carve.obb_plan(truth[0]), carve.obb_plan(res["mesh"][0])
    dims = lambda p: (p["length_mm"], p["width_mm"], p["z_max"] - p["z_min"])
    t, c = dims(pt), dims(pc)
    vt, vc = carve.edge_census(*truth)["volume_mm3"], carve.edge_census(*res["mesh"])["volume_mm3"]
    return {"truth_mm": [round(x, 2) for x in t], "measured_mm": [round(x, 2) for x in c], "error_mm": [round(b - a, 2) for a, b in zip(t, c)],
            "held_out_iou": round(res["held_out_iou"], 4), "volume_ratio": round(vc / vt, 3), "views": res["views"],
            "camera_centre_err_max_mm": round(res["camera_centre_err_mm"], 3)}
