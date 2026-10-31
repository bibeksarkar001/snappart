""""Show me where" question cards: for each question the app asks, the best real frame of the feature, cropped, with the feature outlined in a
highlight colour (one of three reserved colours, chosen by a contrast test against the crop, with a white outer stroke) and a labelled
dimension arrow for the exact measurement asked; card.png + card.json, plus a clean SVG diagram per question type.

Geometry comes from the model's features (part frame); frames are the capture views (cameras in the part frame). No model call.
"""
import json
from pathlib import Path

import cv2
import numpy as np

from . import carve, config as C, review as Rv

HIGHLIGHTS = {"magenta": (255, 0, 255), "cyan": (255, 220, 0), "yellow": (0, 230, 255)}     # BGR, reserved for cards
TEXT = {
    "groove_depth_mm": ("How deep are the rectangular slots? (one answer for all of them)", "mm", "Push the depth rod of your calipers into a slot until it touches the bottom."),
    "bump_height_mm": ("How far does the lock ridge stand out from the wall it is on?", "mm", "Rest the depth rod on the wall beside the ridge; measure to its top."),
    "bump_width_mm": ("How wide is the lock ridge at its base?", "mm", "Close the outside jaws across the ridge where it meets the wall."),
    "bump_length_mm": ("How long is the lock ridge (end to end)?", "mm", "Close the outside jaws over its two ends."),
    "bump_shape": ("What shape is the top of this bump?", "", "Look along it from one end: square (box), rounded, or a hill that narrows to the top."),
    "bump_top_shape": ("What does the top of the lock ridge look like?", "", "Look along the ridge from one end."),
    "plate_thickness_mm": ("How thick is the flat plate, measured in its middle?", "mm", "Use the outside jaws in the middle of the plate, not at its edge."),
}
CHOICES = {"bump_top_shape": list(C.TOP_SHAPE_RADIUS_MM), "bump_shape": list(C.BUMP_SHAPES)}


# ---------- answer sanity checks ----------

MIN_MM = 0.2                                           # answers under ~0.2 mm are not printable features


def bounds(qid, part):
    """(min, max) mm a numeric answer may take for this part (max exclusive where it equals a part dimension), or None for choices."""
    if qid in CHOICES:
        return None
    if qid == "bump_length_mm":
        return MIN_MM, round(max(part["length_mm"], part["width_mm"]), 2)
    return MIN_MM, round(part["height_mm"], 2)


def check_answer(qid, value, part):
    """None if the answer is plausible for this part, else the reason it is rejected. part: {"height_mm", "length_mm", "width_mm",
    "plate_span_mm" (optional)}."""
    if qid in CHOICES:
        return None if value in CHOICES[qid] else f"choose one of {CHOICES[qid]}"
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "enter a number in mm"
    if not np.isfinite(x) or x <= 0:
        return "must be a positive number of mm"
    if x < MIN_MM:
        return f"at least {MIN_MM} mm"
    if qid == "groove_depth_mm" and x >= part["height_mm"]:
        return f"a slot deeper than the part is tall ({part['height_mm']:.1f} mm) would cut through it"
    if qid == "plate_thickness_mm" and x >= part["height_mm"]:
        return f"the plate cannot be thicker than the whole part ({part['height_mm']:.1f} mm)"
    if qid in ("bump_height_mm", "bump_width_mm") and x >= part["height_mm"]:
        return f"larger than the part's height ({part['height_mm']:.1f} mm)"
    if qid == "bump_length_mm" and x > max(part["length_mm"], part["width_mm"]):
        return "longer than the part"
    return None


# ---------- geometry of what each question asks about ----------

def anchors(qid, feats, answers):
    """3D outline polylines, a dimension (p0, p1, label), the face normal (for choosing the frame) and the points that must be visible."""
    a = answers
    if qid == "groove_depth_mm":
        gs = feats["grooves"]
        g = gs[len(gs) // 2]
        c, ax = np.asarray(g["centre"]), np.asarray(g["axes"])
        rect = [c + s * ax[0] * g["long_mm"] / 2 + t * ax[1] * g["short_mm"] / 2 for s, t in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        polys = [[np.asarray(gg["centre"]) + s * np.asarray(gg["axes"][0]) * gg["long_mm"] / 2 + t * np.asarray(gg["axes"][1]) * gg["short_mm"] / 2
                  for s, t in ((-1, -1), (1, -1), (1, 1), (-1, 1))] for gg in gs]
        p0 = rect[1]
        inset = [np.asarray(gg["centre"]) + 0.6 * (p - np.asarray(gg["centre"])) for gg, q in zip(gs, polys) for p in q]   # inside the openings:
        return {"polys": polys, "dim": (p0, p0 - ax[2] * a["groove_depth_mm"], "depth ?"), "normal": ax[2], "pts": np.array(inset)}  # visible down the hole
    if qid.startswith("bump"):
        b = feats["bump"]
        L, W, H = a["bump_length_mm"], a["bump_width_mm"], a["bump_height_mm"]
        if b.get("kind") == "ridge":                   # v4 ridge (a pipeline ridge is anchored on its wall: base from the answered width)
            c, ax, w, n = (np.asarray(b[k], float) for k in ("base_mid" if "base_mid" in b else "wall_mid", "axis", "across", "normal"))
            if "base_mid" not in b:
                c = c + w * W / 2
        else:                                          # v3 bump (pipeline): the block model.bump_op builds, centred W/2 in from the far edge
            ax, w, n = (np.asarray(b[k], float) for k in ("axis", "toward_openings", "normal"))
            c = np.asarray(b["edge_mid"], float) + w * W / 2
        base = [c + s * ax * L / 2 + t * w * W / 2 for s, t in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        peak = [c + n * H - ax * L / 2, c + n * H + ax * L / 2]
        e = c + ax * L / 4
        dim = {"bump_height_mm": (e, e + n * H, "height ?"), "bump_width_mm": (e - w * W / 2, e + w * W / 2, "base width ?"),
               "bump_length_mm": (c - ax * L / 2 + n * H, c + ax * L / 2 + n * H, "length ?"),
               "bump_top_shape": (e + n * H, e + n * H, "top shape ?"), "bump_shape": (e + n * H, e + n * H, "shape ?")}[qid]
        seen_from = n - w                              # the ridge stands in a pocket: look at it from where the pocket opens, not only along n
        return {"polys": [base, peak], "dim": dim, "normal": seen_from / np.linalg.norm(seen_from), "pts": np.array(base + peak)}
    if qid == "plate_thickness_mm":
        p = feats["plate"]
        n, d = np.asarray(p["normal"], float), float(p["d"])
        if "polygon_xy" in p:
            poly = np.asarray(p["polygon_xy"], float)
        else:                                          # pipeline plate: the region_xy rectangle its mask covers
            x0, x1, y0, y1 = p["region_xy"]
            poly = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
        on = lambda xy: np.r_[xy, -(d + xy @ n[:2]) / n[2]]
        outline = [on(q) - n * a["plate_thickness_mm"] for q in poly]
        mid = on(poly.mean(0))
        return {"polys": [outline], "dim": (mid, mid - n * a["plate_thickness_mm"], "thickness ?"), "normal": -n, "pts": np.array(outline)}
    raise KeyError(qid)


# ---------- drawing ----------

def pick_colour(crop, mask_band):
    """Reserved highlight colour with the largest mean Lab distance from the crop's pixels along the outline."""
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3)[mask_band.ravel() > 0].astype(float)
    if not len(lab):
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)
    best = None
    for name, bgr in HIGHLIGHTS.items():
        c = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float)
        score = float(np.linalg.norm(lab - c, axis=1).mean())
        if best is None or score > best[1]:
            best = (name, score)
    return best


def stroke(img, pts, colour, w=3, closed=True):
    p = np.round(np.asarray(pts) * 8).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [p], closed, (255, 255, 255), w + 4, cv2.LINE_AA, 3)
    cv2.polylines(img, [p], closed, colour, w, cv2.LINE_AA, 3)


def arrow(img, a, b, colour, label):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.linalg.norm(b - a) > 4:
        for p, q in ((a, b), (b, a)):
            cv2.arrowedLine(img, tuple(np.round(p).astype(int)), tuple(np.round(q).astype(int)), (255, 255, 255), 6, cv2.LINE_AA, 0, 0.18)
            cv2.arrowedLine(img, tuple(np.round(p).astype(int)), tuple(np.round(q).astype(int)), colour, 3, cv2.LINE_AA, 0, 0.18)
    m = (a + b) / 2 + np.array([12, -12])
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    x, y = int(np.clip(m[0], 4, img.shape[1] - tw - 8)), int(np.clip(m[1], th + 8, img.shape[0] - 6))
    cv2.rectangle(img, (x - 4, y - th - 6), (x + tw + 4, y + 6), (255, 255, 255), -1)
    cv2.rectangle(img, (x - 4, y - th - 6), (x + tw + 4, y + 6), colour, 2)
    cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)


def frame_card(geom, views, images_bgr, occluder, out_px=480, min_arrow_px=12, top=40, full_arrow_px=40):
    """Best frame for the geometry, cropped, outlined, with the dimension arrow: among the `top` frames by review.best_view's score, the
    best by that score x min(1, arrow length / full_arrow_px), with the arrow for the asked measurement at least min_arrow_px long (a depth
    seen head-on would vanish) and >= 30 % of the feature visible. Returns (card image, record) or (None, None)."""
    pts = geom["pts"]
    gray = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in images_bgr]
    span = lambda vw: float(np.linalg.norm(np.subtract(*[np.array(carve.project(vw, np.atleast_2d(p))[:2]).ravel() for p in geom["dim"][:2]])))
    point_dim = np.allclose(geom["dim"][0], geom["dim"][1])
    cands = Rv.best_view(views, gray, pts.mean(0), np.asarray(geom["normal"], float), pts, occluder=occluder, top=top, ranked=True)
    if not point_dim:
        cands = sorted(cands, key=lambda c: -c[1] * min(span(views[c[0]][1]) / full_arrow_px, 1.0))
    ok = [c for c in cands if c[5] > 0.3 and (point_dim or span(views[c[0]][1]) >= min_arrow_px)]
    if not cands:
        return None, None
    i, score, (x0, y0, x1, y1), cos, sharp, vis = (ok or cands)[0]
    fallback = not ok                                  # no frame met the visibility and arrow-length rules: best frame anyway, flagged
    vw = views[i][1]
    pad = int(0.35 * max(x1 - x0, y1 - y0))
    H, W = images_bgr[i].shape[:2]
    x0, y0, x1, y1 = max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, W), min(y1 + pad, H)
    crop = images_bgr[i][y0:y1, x0:x1].copy()
    s = out_px / max(crop.shape[:2])
    crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    proj = lambda P: np.stack(carve.project(vw, np.atleast_2d(P))[:2], 1) * s - np.array([x0, y0]) * s
    band = np.zeros(crop.shape[:2], np.uint8)
    for poly in geom["polys"]:
        cv2.polylines(band, [np.round(proj(np.asarray(poly))).astype(np.int32).reshape(-1, 1, 2)], True, 1, 9)
    name, contrast = pick_colour(crop, band)
    col = HIGHLIGHTS[name]
    for poly in geom["polys"]:
        stroke(crop, proj(np.asarray(poly)), col, closed=len(poly) > 2)
    a, b2 = proj(geom["dim"][0])[0], proj(geom["dim"][1])[0]
    arrow(crop, a, b2, col, geom["dim"][2])
    return crop, {"frame_id": vw.name, "video": views[i][0], "score": round(float(score), 3), "cos": round(cos, 2), "sharpness": round(sharp, 1),
                  "visible_share": round(vis, 2), "crop_px": [x0, y0, x1, y1], "highlight": name, "highlight_contrast": round(contrast, 1),
                  "arrow_px": round(span(vw), 1), "fallback_frame": fallback}


def region_card(tri, v, fc, normal, views, images_bgr, occluder, label, out_px=480):
    """Card picture for a review item: best frame, the region (mesh triangles) outlined as seen from that camera, a label."""
    from . import render as Rn
    pv = np.unique(fc[tri])
    pts = v[pv][:: max(len(pv) // 400, 1)]
    gray = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in images_bgr]
    b = Rv.best_view(views, gray, pts.mean(0), np.asarray(normal, float), pts, occluder=occluder)
    if b is None:
        return None, None
    i, score, (x0, y0, x1, y1), cos, sharp, vis = b
    vw = views[i][1]
    _, ids = Rn.render(v, fc, vw, ids=True)
    reg = np.isin(ids, tri).astype(np.uint8)
    pad = int(0.35 * max(x1 - x0, y1 - y0))
    H, W = images_bgr[i].shape[:2]
    x0, y0, x1, y1 = max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, W), min(y1 + pad, H)
    s = out_px / max(y1 - y0, x1 - x0)
    crop = cv2.resize(images_bgr[i][y0:y1, x0:x1], None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    reg = cv2.resize(reg[y0:y1, x0:x1], (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
    cs, _ = cv2.findContours(reg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    band = np.zeros(crop.shape[:2], np.uint8)
    cv2.drawContours(band, cs, -1, 1, 9)
    name, contrast = pick_colour(crop, band)
    cv2.drawContours(crop, cs, -1, (255, 255, 255), 7, cv2.LINE_AA)
    cv2.drawContours(crop, cs, -1, HIGHLIGHTS[name], 3, cv2.LINE_AA)
    if cs:
        x, y, w, h = cv2.boundingRect(np.vstack(cs))
        arrow(crop, (x + w / 2, y + h / 2), (x + w / 2, y + h / 2), HIGHLIGHTS[name], label)
    return crop, {"frame_id": vw.name, "video": views[i][0], "score": round(float(score), 3), "cos": round(cos, 2), "sharpness": round(sharp, 1),
                  "visible_share": round(vis, 2), "crop_px": [x0, y0, x1, y1], "highlight": name, "highlight_contrast": round(contrast, 1),
                  "region_px_in_crop": int(reg.sum())}


# ---------- diagrams ----------

def _svg(body, w=320, h=200):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="sans-serif" font-size="13">'
            '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            '<path d="M0 0L10 5L0 10z" fill="#d0109c"/></marker></defs>'
            f'<rect width="{w}" height="{h}" fill="#fff"/>{body}</svg>')


def _dim(x1, y1, x2, y2, label, lx, ly):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#d0109c" stroke-width="2.5" marker-start="url(#a)" marker-end="url(#a)"/>'
            f'<text x="{lx}" y="{ly}" fill="#222">{label}</text>')


PART = 'fill="#e9e2da" stroke="#555" stroke-width="2"'


def diagram(qid):
    """A clean schematic of what is measured, per question type (SVG text)."""
    if qid == "groove_depth_mm":
        return _svg(f'<path d="M40 40H280V170H40Z" {PART}/><path d="M130 40V115H190V40" fill="#fff" stroke="#555" stroke-width="2"/>'
                    + _dim(215, 40, 215, 115, "depth", 222, 82) + '<text x="40" y="190" fill="#666">section through one slot</text>')
    if qid in ("bump_height_mm", "bump_width_mm", "bump_top_shape", "bump_shape", "bump_length_mm"):
        hill = f'<path d="M60 170H260V150H190L172 72Q160 58 148 72L130 150H60Z" {PART}/>'
        d = {"bump_height_mm": _dim(215, 150, 215, 64, "height", 222, 110), "bump_width_mm": _dim(130, 185, 190, 185, "base width", 125, 200),
             "bump_length_mm": '<text x="60" y="30" fill="#222">length: end to end, along the ridge</text>',
             "bump_top_shape": '<text x="196" y="66" fill="#222">top shape?</text>',
             "bump_shape": '<text x="196" y="66" fill="#222">box, rounded or hill?</text>'}[qid]
        return _svg(hill + d + '<text x="60" y="194" fill="#666">ridge cross-section</text>' * (qid != "bump_width_mm"), h=210)
    if qid == "plate_thickness_mm":
        return _svg(f'<path d="M30 90H290V120H30Z" {PART}/>' + _dim(160, 90, 160, 120, "thickness (middle)", 168, 110)
                    + '<text x="30" y="150" fill="#666">section through the plate</text>')
    return _svg(f'<path d="M40 120Q90 95 120 118T200 112T280 120V170H40Z" {PART}/><path d="M40 120H280" stroke="#d0109c" stroke-dasharray="6 5" stroke-width="2"/>'
                '<text x="40" y="60" fill="#222">flagged region: keep, remove, or make flat?</text>')


def write_card(out_dir: Path, q, img, rec, svg):
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "card.png"), img)
    (out_dir / "diagram.svg").write_text(svg)
    card = {"question_id": q["id"], "text": q["question"], "unit": q.get("unit", ""), "hint": q.get("hint", ""), "default": q.get("default"),
            "choices": q.get("choices"), "image": "card.png", "diagram": "diagram.svg", **(rec or {})}
    (out_dir / "card.json").write_text(json.dumps(card, indent=1, default=float))
    return card
