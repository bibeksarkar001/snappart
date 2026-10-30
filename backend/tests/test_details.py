"""Small-detail candidates on synthetic CadQuery parts (exact meshes) and a synthetic rectified face image."""
import cadquery as cq
import cv2
import numpy as np

from capture import details as D


def mesh(wp):
    v, t = wp.val().tessellate(0.01, 0.1)
    return np.array([[p.x, p.y, p.z] for p in v]), np.array(t)


def kinds(cands, kind):
    return [c for c in cands if c["kind"] == kind]


def test_holes_and_bosses():
    plate = cq.Workplane("XY").box(40, 30, 5).faces(">Z").workplane().pushPoints([(-10, 0), (10, 0)]).circle(2.0).cutThruAll()
    plate = plate.faces(">Z").workplane().pushPoints([(10, 0)]).circle(3.0).cutThruAll()
    holes = kinds(D.mesh_candidates(*mesh(plate)), "hole")
    dias = sorted(h["size_mm"]["diameter"] for h in holes)
    assert len(dias) == 2 and abs(dias[0] - 4.0) < 0.3 and abs(dias[1] - 6.0) < 0.3 and all(h["status"] == "measured" for h in holes)
    boss = cq.Workplane("XY").box(30, 30, 5).faces(">Z").workplane().circle(4.0).extrude(5)
    b = kinds(D.mesh_candidates(*mesh(boss)), "boss")
    assert len(b) == 1 and abs(b[0]["size_mm"]["diameter"] - 8.0) < 0.3 and b[0]["arc_deg"] > 300


def test_steps_fillets_chamfers():
    l_bracket = cq.Workplane("XY").box(30, 20, 5, centered=(False, True, False)).union(
        cq.Workplane("XY").box(8, 20, 15, centered=(False, True, False)))
    steps = kinds(D.mesh_candidates(*mesh(l_bracket)), "step")
    assert any(abs(s["size_mm"]["height"] - 10.0) < 0.3 for s in steps)
    fil = cq.Workplane("XY").box(30, 20, 10).edges("|Z").fillet(2.0)
    f = kinds(D.mesh_candidates(*mesh(fil)), "fillet")
    assert f and all(abs(x["size_mm"]["radius"] - 2.0) < 0.3 and x["arc_deg"] < 180 for x in f)
    ch = cq.Workplane("XY").box(30, 20, 10).edges(">Z").chamfer(2.0)
    c = kinds(D.mesh_candidates(*mesh(ch)), "chamfer")
    assert c and all(abs(x["size_mm"]["width"] - 2.0 * np.sqrt(2)) < 0.5 for x in c)


def test_image_openings_text_and_tiny_marks():
    ppm = 10
    face = np.full((300, 400), 200, np.uint8)
    cv2.circle(face, (100, 150), 15, 40, -1)                                   # 3 mm hole
    cv2.rectangle(face, (200, 140), (280, 160), 40, -1)                        # 8 x 2 mm slot
    cv2.circle(face, (60, 60), 1, 40, -1)                                       # 0.3 mm speck
    cv2.putText(face, "SNAP", (220, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 110, 1, cv2.LINE_AA)   # engraved text (lighter than holes)
    rng = np.random.default_rng(0)
    per_view = []
    for k in range(4):
        img = np.clip(face + rng.normal(0, 2, face.shape), 0, 255).astype(np.uint8)
        op = D.dark_openings(img, ppm)
        per_view.append(op + D.text_areas(img, ppm, op))
    cands = D.cluster_views(per_view)
    hole, slot, text = kinds(cands, "hole"), kinds(cands, "slot"), kinds(cands, "text_or_mark")
    assert len(hole) == 1 and abs(hole[0]["size_mm"]["short"] - 3.0) < 0.4 and hole[0]["status"] == "measured" and hole[0]["evidence"]["views"] == 4
    assert len(slot) == 1 and abs(slot[0]["size_mm"]["long"] - 8.0) < 0.5 and abs(slot[0]["size_mm"]["short"] - 2.0) < 0.4
    assert text and all(t["status"] == "question" for t in text)               # text is always asked about
    assert all(c["size_mm"].get("short", 9) >= 1.0 for c in cands)              # the 0.3 mm speck is never claimed
    assert all(c["not_detected"]["depth"] for c in cands)
    once = D.cluster_views(per_view[:1])
    assert kinds(once, "hole")[0]["status"] == "question"                       # one view only: asked, not claimed
