"""Held-out silhouette evaluation: carve without some views, render the mesh into them, and compare with their masks (IoU)."""
import numpy as np

from . import carve

LO, HI = [148.5 - 35, 105 - 35, -1.5], [148.5 + 35, 105 + 35, 45.0]


def folds(n_views: int, n_hold=10, n_folds=3):
    """Spread held-out indices over the view list: fold f takes every (n/n_hold)-th view starting at offset f."""
    step = max(n_views // n_hold, 1)
    return [list(range(f % step, n_views, step))[:n_hold] for f in range(n_folds)]


def carve_mesh(views, vs, k, floors=(), smooth=5):
    g, f, n = carve.carve_adaptive(views, LO, HI, vs, k, floors=floors)
    v, fc = carve.to_mesh(f, g)
    return carve.taubin(v, fc, smooth), fc


def single_video(views, k, vs=0.25, n_hold=10, n_folds=3):
    """Mean IoU on held-out views of one video, carved from that video's other views alone."""
    ious = []
    for held in folds(len(views), n_hold, n_folds):
        train = [v for i, v in enumerate(views) if i not in held]
        m = carve_mesh(train, vs, k)
        ious += [carve.view_iou(*m, views[i]) for i in held]
    return float(np.mean(ious)), ious


def joint(views_by_video, T, k, vs=0.25, n_hold=10, n_folds=3, ref=None):
    """Joint model: hold out `n_hold` views per fold spread over all videos (proportionally), carve the rest in the part frame, IoU on the held-out ones.

    Returns (mean IoU, {video: mean IoU on that video's held-out views}, list of (view, iou))."""
    allv = [(n, i, v) for n, vs_ in views_by_video.items() for i, v in enumerate(vs_)]
    per = {n: [] for n in views_by_video}
    allr = []
    for f in range(n_folds):
        hold = set()
        for n, vs_ in views_by_video.items():
            share = max(round(n_hold * len(vs_) / len(allv)), 1)
            step = max(len(vs_) // share, 1)
            hold |= {(n, i) for i in list(range(f % step, len(vs_), step))[:share]}
        train = [v.moved(T[n]) for n, i, v in allv if (n, i) not in hold]
        floors = [T[n] for n in views_by_video if n != ref]
        m = carve_mesh(train, vs, k, floors)
        for n, i in sorted(hold):
            v = views_by_video[n][i].moved(T[n])
            iou = carve.view_iou(*m, v)
            per[n].append(iou)
            allr.append((views_by_video[n][i], iou))
    return float(np.mean([r for _, r in allr])), {n: float(np.mean(x)) for n, x in per.items()}, allr


def constraint_counts(views_moved, verts, tol_mm=0.3):
    """For each vertex, the number of views whose silhouette edge passes within tol_mm of it: those views determine the surface there."""
    cnt = np.zeros(len(verts), int)
    for v in views_moved:
        val = carve.view_values(v, verts)
        cnt += np.abs(val) < tol_mm
    return cnt
