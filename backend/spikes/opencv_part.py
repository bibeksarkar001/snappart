"""Phase 1 step 2, spike 3: OpenCV fallback segmentation of the dark part, scored against the SAM mask.

1. find the paper: the biggest bright region, holes filled (so the coin and part sit inside it)
2. inside the paper only, keep pixels much darker than the paper (a black mat outside the paper never counts)
3. clean up, keep the biggest component
Needs out/part_mask_sam.png from sam_ladder.py. Saves out/part_mask_cv.png.
"""
import cv2
import numpy as np

from spike_common import OUT, PHOTO, save_overlay, tint


def paper_region(gray):
    blur = cv2.GaussianBlur(gray, (0, 0), 5)
    thr, bright = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bright)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cnts, _ = cv2.findContours((lab == biggest).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    paper = np.zeros_like(gray)
    cv2.drawContours(paper, cnts, -1, 255, cv2.FILLED)  # filled: objects on the paper count as inside it
    return paper, thr


def segment_part(img, dark_frac=0.35):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    paper, otsu = paper_region(gray)
    paper_inner = cv2.erode(paper, np.ones((41, 41), np.uint8))  # stay off the paper edge
    paper_level = float(np.median(gray[(paper_inner > 0) & (gray > otsu)]))
    dark = ((cv2.GaussianBlur(gray, (0, 0), 2) < dark_frac * paper_level) & (paper_inner > 0)).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(dark)
    order = np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1
    part = (lab == order[0]).astype(np.uint8) * 255
    return part, paper, paper_level, otsu, [int(stats[i, cv2.CC_STAT_AREA]) for i in order[:5]]


def iou(a, b):
    a, b = a > 0, b > 0
    return (a & b).sum() / (a | b).sum()


def main():
    img = cv2.imread(str(PHOTO))
    part, paper, level, otsu, areas = segment_part(img)
    print(f"paper area: {(paper > 0).mean():.1%} of frame (otsu={otsu:.0f}, paper gray median={level:.0f})")
    print(f"dark components inside paper (px, biggest first): {areas}")
    sam = cv2.imread(str(OUT / "part_mask_sam.png"), cv2.IMREAD_GRAYSCALE)
    print(f"part area cv={int((part > 0).sum())} px, sam={int((sam > 0).sum())} px")
    print(f"IoU(cv, sam) = {iou(part, sam):.4f}")
    # where do they disagree?
    print(f"in sam only: {int(((sam > 0) & (part == 0)).sum())} px, in cv only: {int(((part > 0) & (sam == 0)).sum())} px")
    cv2.imwrite(str(OUT / "part_mask_cv.png"), part)
    diff = tint(tint(img, (part > 0) & (sam == 0), (0, 0, 255), 0.8), (sam > 0) & (part == 0), (255, 0, 0), 0.8)
    save_overlay(OUT / "cv_vs_sam_overlay.png", tint(diff, (part > 0) & (sam > 0), (0, 200, 0), 0.4))
    print("saved part_mask_cv.png, cv_vs_sam_overlay.png (green=both, red=cv only, blue=sam only)")


if __name__ == "__main__":
    main()
