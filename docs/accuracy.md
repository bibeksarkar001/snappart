# Capture accuracy (measured)

One part has been measured so far: "part B" of the lock set (calipers 51.6 x 41.9 x 11.8 mm), filmed in three orbit videos on the ArUco mat (two flat, one on its edge). Every number below was measured on these videos; nothing here is an accuracy claim beyond this one part. No private images are included.

## Method in one paragraph

Mat self-calibration (marker corners free in 3D, per-video intrinsics) -> marker poses (rms <= 1.0 px; 1.25 px for one video) -> SAM 3.1 silhouettes on 80 spread frames per video -> robust silhouette carving (a voxel is removed only if 12 views agree, silhouettes grown 0.1 mm, 0.15 mm voxels) -> videos registered into one part frame -> small features that silhouettes cannot see (openings) found by rectifying each view onto a fitted plane and triangulating the openings' corners. Settings are frozen in `backend/capture/config.py`.

## Whole part

| Quantity | Measured | Caliper | Error |
|---|---|---|---|
| Footprint length | 51.34 mm | 51.6 mm | -0.26 mm |
| Footprint width | 41.70 mm | 41.9 mm | -0.20 mm |
| Height | 11.69 mm | 11.8 mm | -0.11 mm |

Silhouette agreement on views left out of the carve (3 folds x 10 views): **held-out IoU 0.953** (lowest single view 0.881). The earlier baseline (k=2, no growth, 0.25 mm) scored 0.926 on the same folds and had a stray second component.

## Features

| Feature | Found? | Measured | Caliper | Error |
|---|---|---|---|---|
| End grooves, count | yes, 48-105 agreeing views each, 2 videos | 4 | 4 | 0 |
| Groove opening, long side (mean of 4) | yes | 7.43 mm (7.36-7.53) | 7.4 mm | +0.03 mm |
| Groove opening, short side (mean of 4) | yes | 2.54 mm (2.45-2.65) | 2.6 mm | -0.06 mm |
| Groove depth | not measured | 3.7 mm used, **user-supplied** | 3.7 mm | n/a |
| Lock bump, length | yes, 30 agreeing views (video 3) | 19.39 mm | 19.9 mm | -0.51 mm |
| Lock bump, width | not measured (opposite edge not separable; 2.82 mm is an upper bound) | n/a | 2.2 mm (owner) | n/a |
| Lock bump, protrusion | not measured (parallax within +-0.3 mm, sign not consistent) | n/a | not measured | n/a |
| Lock pin | part of the bump (owner); found at the bump's location, nothing separate | n/a | n/a | n/a |

Only the grooves are in the model, as pockets cut to the user-supplied depth. No bump geometry was added because its protrusion is unknown.

## Known limits (measured)

- The videos disagree by about 1.6 mm in height at the grooved end (registration). Groove positions come from the video with the most views.
- The visual hull does not see the recessed face the grooves open in: that face lies about 2.0 mm inside the hull's bottom face (video 3).
- 2.8 % of the surface is fixed by fewer than 3 views.
- Held-out IoU is limited to about 0.95 by marker-pose error and mask edge noise.

## Time per user run (this laptop, 10 cores)

| Stage | Seconds |
|---|---|
| Marker scan (3 x 4K videos, one process each) | 77 |
| Mat self-calibration | 15 |
| Poses, frame selection, extraction | 37 |
| Crops + masks (SAM cached) | 19 |
| Registration (0.5 mm refinement, default) | 138 |
| Joint carve (8 workers) | 37 |
| Grooves + pockets | 53 |
| **Total without SAM** | **376 (6.3 min)** |
| SAM, 276 calls at 4 in parallel (measured throughput 0.88 s/call) | ~243 |
| **Estimated total** | **~10.3 min** |

The 5-minute target is **not reached**. A 1.0 mm registration grid cuts registration to 29 s (4.5 min without SAM), but it moved one video by 0.304 mm against the reference transform, over the 0.3 mm limit, so it is not the default. Running SAM at 16 in parallel would take about 61 s; that figure is a linear extrapolation and has not been measured.

## Regression test

`backend/tests/test_golden_B.py` reruns the carve and groove steps from the private cache and asserts footprint within 0.5 mm, height within 0.3 mm, held-out IoU >= 0.93 and four grooves. It passes in 104 s and is skipped when the cache is missing.
