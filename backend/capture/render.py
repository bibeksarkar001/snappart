"""Mesh renders for review crops, question cards and before/after pictures: a look-at pinhole camera (carve.View), Lambert-shaded
painter's-algorithm fill, and a triangle-id buffer (which triangle is visible at each pixel)."""
import cv2
import numpy as np

from . import carve


def look_at(eye, target, up=(0, 0, 1), f=1600.0, size=(1000, 700), centre=None):
    """carve.View for a camera at eye looking at target; image size (w, h); centre = principal point (default: image centre)."""
    eye, target, up = (np.asarray(a, float) for a in (eye, target, up))
    zc = target - eye
    zc /= np.linalg.norm(zc)
    xc = np.cross(zc, up)
    xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    R = np.stack([xc, yc, zc])
    cx, cy = centre if centre is not None else (size[0] / 2, size[1] / 2)
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    return carve.View(K, R, -R @ eye, np.zeros((size[1], size[0]), np.float32))


def orbit(target, az_deg, el_deg, dist, roll_deg=0.0, **kw):
    """Camera on a sphere around target (az from +x toward +y, el from the xy plane toward +z), rolled about its viewing axis."""
    a, e = np.radians(az_deg), np.radians(el_deg)
    d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    up = np.array([0, 0, 1.0]) if abs(d[2]) < 0.99 else np.array([0, 1.0, 0])
    v = look_at(np.asarray(target, float) + dist * d, target, up, **kw)
    r = np.radians(roll_deg)
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return carve.View(v.K, Rz @ v.R, Rz @ v.t, v.sdf)


def render(verts, faces, view, light=(0.3, -0.5, -1.0), ids=False):
    """Shaded gray BGR image (and, with ids=True, an int32 buffer of the visible triangle per pixel, -1 = background)."""
    c = verts @ view.R.T + view.t
    z = np.maximum(c[:, 2], 1e-6)
    uv = np.stack([view.K[0, 0] * c[:, 0] / z + view.K[0, 2], view.K[1, 1] * c[:, 1] / z + view.K[1, 2]], 1)
    tri = c[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    L = np.asarray(light, float) / np.linalg.norm(light)
    shade = (70 + 170 * np.clip(np.abs(n @ L), 0, 1)).astype(int)
    w, h = view.size
    img = np.zeros((h, w, 3), np.uint8)
    idb = np.full((h, w), -1, np.int32) if ids else None
    pts = np.round(uv[faces] * 16).astype(np.int32)
    order = np.argsort(-tri[:, :, 2].mean(1))
    order = order[(n[order, 2] < 0) & (tri[order, :, 2] > 1e-3).all(1)]
    for i in order:
        g = int(shade[i])
        cv2.fillConvexPoly(img, pts[i], (g, g, g), cv2.LINE_8, 4)
        if ids:
            cv2.fillConvexPoly(idb, pts[i], int(i), cv2.LINE_8, 4)
    return (img, idb) if ids else img
