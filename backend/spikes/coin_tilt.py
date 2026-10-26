"""Phase 1 step 2, spike 4: ellipse-fit the SAM coin mask and estimate camera tilt.

A circle seen at tilt t becomes an ellipse with minor/major = cos(t). Needs out/coin_mask_sam.png.
Caveat: the coin has a rim of finite height and the mask edge is only +-1-2 px, so tilts under ~3 degrees are not resolvable.
"""
import math

import cv2
import numpy as np

from spike_common import OUT

FLAG_DEG = 5.0


def fit_coin(mask):
    """Returns ((cx, cy), major_px, minor_px, angle_deg) from the biggest contour of a 0/255 mask."""
    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    (cx, cy), axes, angle = cv2.fitEllipse(max(cnts, key=cv2.contourArea))
    return (cx, cy), max(axes), min(axes), angle


def tilt_deg(major, minor):
    return math.degrees(math.acos(min(1.0, minor / major)))


def main():
    mask = cv2.imread(str(OUT / "coin_mask_sam.png"), cv2.IMREAD_GRAYSCALE)
    (cx, cy), major, minor, angle = fit_coin(mask)
    t = tilt_deg(major, minor)
    print(f"ellipse centre=({cx:.1f},{cy:.1f}) major={major:.1f}px minor={minor:.1f}px major-axis angle={angle:.1f} deg")
    print(f"axis ratio minor/major = {minor / major:.4f}  ->  estimated tilt = {t:.1f} deg")
    lo, hi = tilt_deg(major, minor - 2), tilt_deg(major, min(major, minor + 2))
    print(f"if the minor axis is off by +-2 px: tilt between {hi:.1f} and {lo:.1f} deg")
    print(f"equivalent-circle diameter sqrt(major*minor) = {math.sqrt(major * minor):.1f}px")
    print("FLAG: tilt above 5 deg" if t > FLAG_DEG else f"no flag: estimated tilt is at or below {FLAG_DEG:.0f} deg")


if __name__ == "__main__":
    main()
