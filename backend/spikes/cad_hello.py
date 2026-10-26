"""Spike 2: CadQuery hello. Hardcoded plate (extrude) and wheel (revolve) to STL."""
import time
from pathlib import Path

import cadquery as cq

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

# 60x40 plate, corners cut 5 mm, 4 mm thick, one 8 mm hole in the centre
PLATE_POLY = [(-30, -15), (-25, -20), (25, -20), (30, -15), (30, 15), (25, 20), (-25, 20), (-30, 15)]
# wheel half-profile as (radius, axial position): 4 mm axle hole, 20 mm outer radius, 10 mm wide
WHEEL_PROFILE = [(2, 0), (18, 0), (20, 2), (20, 8), (18, 10), (2, 10)]


def report(name, solid):
    bb = solid.BoundingBox()
    print(f"{name}: valid={solid.isValid()} volume={solid.Volume():.1f} mm3 "
          f"bbox={bb.xlen:.2f} x {bb.ylen:.2f} x {bb.zlen:.2f} mm")


def main():
    t = time.perf_counter()
    plate = cq.Workplane("XY").polyline(PLATE_POLY).close().extrude(4).faces(">Z").workplane().hole(8)
    report("plate", plate.val())
    cq.exporters.export(plate, str(OUT / "plate.stl"))
    print(f"plate build+export: {time.perf_counter() - t:.2f}s")

    t = time.perf_counter()
    wheel = cq.Workplane("XY").polyline(WHEEL_PROFILE).close().revolve(360, (0, 0, 0), (0, 1, 0))
    report("wheel", wheel.val())
    cq.exporters.export(wheel, str(OUT / "wheel.stl"))
    print(f"wheel build+export: {time.perf_counter() - t:.2f}s")


if __name__ == "__main__":
    main()
