"""Phase 1 step 2, spike 5: scale from the coin and size of the part, for both candidate coin diameters.

Needs out/coin_mask_sam.png, out/part_mask_cv.png (holes), out/part_mask_sam.png. Expected ranges are the iPhone Measure
app rounded to whole cm +-5 mm: a plausibility check only, NOT ground truth and NOT accuracy.
"""
import math

import cv2
import numpy as np

from coin_tilt import fit_coin, tilt_deg
from spike_common import COIN_MM, OUT, PHOTO, save_overlay

EXPECTED = {"length": (65, 75), "body width": (25, 35), "neck width": (15, 25)}


def align(mask):
    """Rotate a 0/255 mask so its long axis is vertical. Returns the rotated mask and the angle used."""
    ys, xs = np.nonzero(mask)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    mean, vecs = cv2.PCACompute(pts, np.empty(0))[:2]
    vx, vy = vecs[0]
    ang = math.degrees(math.atan2(vx, vy))  # angle of the long axis from vertical
    h, w = mask.shape
    m = cv2.getRotationMatrix2D((float(mean[0][0]), float(mean[0][1])), -ang, 1.0)
    return cv2.warpAffine(mask, m, (w, h), flags=cv2.INTER_NEAREST), ang, m


def width_profile(rot):
    rows = np.nonzero(rot.any(axis=1))[0]
    top, bottom = rows[0], rows[-1]
    prof = np.array([(lambda xs: xs[-1] - xs[0] + 1)(np.nonzero(rot[y])[0]) for y in range(top, bottom + 1)])
    return prof, bottom - top + 1


def outline_sizes(mask):
    rot, ang, _ = align(mask)
    prof, length = width_profile(rot)
    at = lambda a, b: float(np.median(prof[int(a * len(prof)):int(b * len(prof))]))
    # ponytail: section positions are fractions of the length picked for THIS part by eye from the profile printout;
    # a general part needs real notch detection.
    neck = (at(0.14, 0.24) + at(0.76, 0.86)) / 2
    return {"length": length, "body width": at(0.36, 0.62), "neck width": neck, "end-cap width": float(prof.max())}, prof, ang


def holes(mask, rot_m):
    """Inner contours of the part mask that look like round holes near the two ends. Returns [(diam_px, circularity, (x, y))]."""
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    ys, xs = np.nonzero(mask)
    rotpts = cv2.transform(np.column_stack([xs, ys]).astype(np.float32)[None], rot_m)[0]
    y0, y1 = rotpts[:, 1].min(), rotpts[:, 1].max()
    out = []
    for c, hh in zip(cnts, hier[0]):
        if hh[3] < 0:
            continue  # outer contour
        a, p = cv2.contourArea(c), cv2.arcLength(c, True)
        if a < 100:
            continue
        (x, y), = cv2.transform(np.array([[c.reshape(-1, 2).mean(0)]], np.float32), rot_m)[0]
        out.append((math.sqrt(4 * a / math.pi), 4 * math.pi * a / p ** 2, tuple(c.reshape(-1, 2).mean(0)), float((y - y0) / (y1 - y0))))
    return out


def verdict(v, rng):
    return "n/a (no reference)" if rng is None else ("in range" if rng[0] <= v <= rng[1] else "OUT of range")


def main():
    coin = cv2.imread(str(OUT / "coin_mask_sam.png"), cv2.IMREAD_GRAYSCALE)
    cv_mask = cv2.imread(str(OUT / "part_mask_cv.png"), cv2.IMREAD_GRAYSCALE)
    sam_mask = cv2.imread(str(OUT / "part_mask_sam.png"), cv2.IMREAD_GRAYSCALE)
    _, major, minor, _ = fit_coin(coin)
    print(f"coin ellipse: major={major:.1f}px minor={minor:.1f}px, tilt~{tilt_deg(major, minor):.1f} deg")
    print("scale uses the MAJOR axis (tilt only shrinks the minor axis). Tilt also squeezes the scale along one direction by up to "
          f"{(1 - minor / major):.1%}, which this does not correct.")
    for name, mm in COIN_MM.items():
        print(f"coin {name}: {mm / major:.5f} mm/px  ({major / mm:.2f} px/mm)")

    cv_sizes, prof, ang = outline_sizes(cv_mask)
    sam_sizes, _, _ = outline_sizes(sam_mask)
    print(f"\npart long axis is {ang:.1f} deg from vertical. width profile (px) every 5% of length, CV mask:")
    print("  " + " ".join(f"{int(prof[min(len(prof) - 1, int(f / 100 * len(prof)))])}" for f in range(0, 101, 5)))

    print("\nSIZES (CV mask; SAM mask in brackets). 'in range' is ONLY a plausibility check against +-5 mm Measure-app readings.")
    for name, mm in COIN_MM.items():
        s = mm / major
        print(f"\n-- coin {name}: {s:.5f} mm/px --")
        print(f"{'feature':<14}{'px':>8}{'mm':>9}{'[sam mm]':>10}  {'expected mm':<12} check")
        for k, v in cv_sizes.items():
            rng = EXPECTED.get(k)
            print(f"{k:<14}{v:>8.0f}{v * s:>9.1f}{sam_sizes[k] * s:>10.1f}  {str(rng or '-'):<12} {verdict(v * s, rng)}")

    _, _, rot_m = align(cv_mask)
    hs = [h for h in holes(cv_mask, rot_m) if h[1] > 0.6 and (h[3] < 0.2 or h[3] > 0.8)]
    # the biggest round candidate at each end is taken as the hole; the rest are reflections or noise
    ends = [[h for h in hs if h[3] < 0.5], [h for h in hs if h[3] >= 0.5]]
    ignored = len(hs) - sum(bool(e) for e in ends)
    hs = [max(e, key=lambda h: h[0]) for e in ends if e]
    print("\nHOLES (round inner contours of the CV mask in the outer 20% at either end, circularity>0.6):")
    for i, (d, circ, (x, y), pos) in enumerate(hs, 1):
        print(f"  hole {i}: diameter {d:.0f}px = " + " / ".join(f"{d * mm / major:.2f} mm @{mm:g}" for mm in COIN_MM.values())
              + f"  circularity={circ:.2f} at {pos:.0%} along the length")
    if not hs:
        print("  none found")
    print(f"  ({ignored} smaller round candidates ignored as reflections/noise)")
    img = cv2.imread(str(PHOTO))
    for d, _, (x, y), _ in hs:
        cv2.circle(img, (int(x), int(y)), int(d / 2) + 25, (0, 0, 255), 6)
    save_overlay(OUT / "holes_overlay.png", img)
    print("saved holes_overlay.png (red circles = chosen holes)")
    print("Boss outer diameters are NOT measured: the boss is dark on a dark part, so its edge is not in the silhouette.")
    print("There is no reference value for the holes, so no in-range check.")


if __name__ == "__main__":
    main()
