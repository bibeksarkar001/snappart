"""Deterministic extrusion of an outline.Part into a cadquery solid. CAD frame: x right, y UP (the outline's y is down, so y is mirrored)."""
import math

import cadquery as cq


def _loop(prims):
    """Closed cadquery wire (as a Workplane) through the primitives, y mirrored."""
    wp = cq.Workplane("XY")
    if prims[0]["type"] == "circle":
        c = prims[0]["c"]
        return wp.moveTo(c[0], -c[1]).circle(prims[0]["r"])
    wp = wp.moveTo(prims[0]["p0"][0], -prims[0]["p0"][1])
    for p in prims:
        if p["type"] == "line":
            wp = wp.lineTo(p["p1"][0], -p["p1"][1])
        else:
            c, r = p["c"], p["r"]
            a = math.atan2(p["p0"][1] - c[1], p["p0"][0] - c[0]) + math.radians(p["sweep_deg"]) * (1 if p["ccw"] else -1) / 2
            wp = wp.threePointArc((c[0] + r * math.cos(a), -(c[1] + r * math.sin(a))), (p["p1"][0], -p["p1"][1]))
    return wp.close()


def extrude_part(part, thickness_mm: float) -> cq.Workplane:
    """Outline extruded to thickness_mm with every detected hole and slot cut through."""
    solid = _loop(part.primitives).extrude(thickness_mm)
    for h in part.holes:
        solid = solid.cut(_loop(h.primitives).extrude(thickness_mm))
    return solid


def export_stl(solid: cq.Workplane, path) -> None:
    cq.exporters.export(solid, str(path), tolerance=0.005, angularTolerance=0.05)
