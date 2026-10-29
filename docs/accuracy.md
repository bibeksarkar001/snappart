# Capture accuracy (measured)

One part has been measured so far: "part B" of the lock set (calipers 51.6 x 41.9 x 11.8 mm), filmed in three orbit videos on the ArUco mat (two flat, one on its edge). Every number below was measured on these videos, or is labelled user-supplied. Nothing here is an accuracy claim beyond this one part. No private images are included.

## Method in one paragraph

Mat self-calibration (marker corners free in 3D, per-video intrinsics) -> marker poses -> SAM 3.1 silhouettes -> robust silhouette carving (a voxel is removed only if k views agree: k = 12 at 229 views, scaled with the view count; silhouettes grown 0.1 mm; 0.15 mm voxels) -> videos registered into one part frame. Features silhouettes cannot see are found by rectifying views onto fitted planes: slot openings by triangulating their corners, the lock bump's position by its edge steps, and the plate's faces by photo-consistency plane sweeps. Sizes the camera cannot see come from user answers. `backend/capture/model.py` applies them to the stored carve field; `rebuild(model, answers)` reapplies them with no model call. Settings and answers are in `backend/capture/config.py`.

## Whole part (plain carve, no user-supplied features)

| Quantity | Measured | Caliper | Error |
|---|---|---|---|
| Footprint length | 51.34 mm | 51.6 mm | -0.26 mm |
| Footprint width | 41.70 mm | 41.9 mm | -0.20 mm |
| Height | 11.69 mm | 11.8 mm | -0.11 mm |

Held-out silhouette IoU (3 folds x 10 views left out of the carve, 0.15 mm): **0.9529** for the plain carve and **0.9532** with grooves, bump and plate applied.

## Features

| Feature | Status | Value | Caliper | Error |
|---|---|---|---|---|
| Slots, count | measured (48-105 views each) | 4 | 4 | 0 |
| Slot opening, long side (mean) | measured | 7.43 mm | 7.4 | +0.03 |
| Slot opening, short side (mean) | measured | 2.54 mm | 2.6 | -0.06 |
| Slot depth | user-supplied | 3.7 mm | 3.7 | n/a |
| Lock bump, position | measured (30 views) | far edge at x 165.7 mm, centred over slots 2-3 | n/a | n/a |
| Lock bump, length | measured / user-supplied | 19.39 mm detected; 19.7 mm used | 19.7 | -0.31 (detected) |
| Lock bump, width | user-supplied | 2.3 mm | 2.3 (2.2 also given) | n/a |
| Lock bump, height | user-supplied | 2.3 mm | 2.3 | n/a |
| Lock pin | user-supplied | part of the bump | n/a | n/a |
| Plate, face separation | measured (125 views) | 0.2-0.5 mm (depends on the region) | 2.6 | -2.1 to -2.4 |
| Plate, thickness used | user-supplied | 2.6 mm | 2.6 | n/a |

The plate conflict is unresolved: two plane sweeps (from opposite sides, different videos) put its faces 0.2-0.5 mm apart in the middle, while the owner's caliper reads 2.6 mm. The model uses 2.6 mm and the report asks for a re-measurement in the middle of the plate.

Plate-face roughness (RMS distance to its own plane) is 0.764 mm before on the lower face and 1.898 mm on the upper face. After the snap both are 0, by construction.

## Known limits (measured)

- Videos 2 and 3 disagree by 1.65 mm (1.58-1.72) in the height of the same four slot openings (registration). Positions use the video with more views.
- Silhouettes cannot see recesses: the face the slots open in lies about 2 mm inside the carve, the region under the plate between the rails is filled, and only 14 % of the bump block lies outside the carve.
- The owner says the bump starts at one end of the slot column; the detection puts it in the middle, with about 8 mm free at both ends.
- 2.8 % of the surface is fixed by fewer than 3 views.

## Time per user run (browser path: 40 frames per video at 1920 px, 10-core laptop with 16 GB)

| Stage | Seconds |
|---|---|
| Marker detection (120 frames) | 2.0 |
| Mat self-calibration | 9.5 |
| Poses + undistorted crops | 2.9 |
| SAM: 140 calls at 16 parallel (rate measured on 40 real calls: 0.37 s/call, no 429s) | 52 |
| Registration (1.0 mm grid) | 29.4 |
| Joint carve | 38.7 |
| Features (slots, bump, plate) | 41.0 |
| Apply answers + mesh + STL | 5.9 |
| **Total** | **181 (3.0 min)** |

The SAM time is extrapolated from the 40-call test to 140 calls. Decoding frames from the video is not counted, because in the app the browser sends the frames.

Effect of the browser path (40 frames/video at 1920 px instead of 80 at 4K), measured:
- Mat corners move a mean 0.05 mm.
- Footprint 51.31 x 41.76 mm, height 11.82 mm.
- Held-out IoU 0.936 at 0.25 mm voxels (0.953 at full resolution).
- The same 4 slots are found with the same sizes, within 0.07 mm.

## v4: after the owner's review of the v3 model

The owner marked the v3 model on five screenshots. A camera was fitted to each screenshot by silhouette IoU (0.87-0.96), so every mark maps onto 3D surface. All five screenshots look at the side with the slot openings and the deep pocket over the plate.

| Feature | Status | Value | Caliper | Error |
|---|---|---|---|---|
| Lock ridge, face and direction | measured | slot column's inner wall, pointing toward the plate | n/a | n/a |
| Lock ridge, position along the edge | measured (29 views) | centred at y 101.85 (part centre 101.5) | n/a | n/a |
| Lock ridge, length | measured / user-supplied | 19.37 mm detected; 19.7 mm used | 19.7 | -0.33 (detected) |
| Lock ridge, height out of the wall | measured (24 head-on views) / user-supplied | 1.90 mm (sd 0.08) measured; 2.3 mm used | 2.3 | -0.40 (measured) |
| Lock ridge, base width | user-supplied | 2.3 mm | 2.3 | n/a |
| Lock ridge, top radius and side slope | estimated | 0.5 mm, 71 deg | n/a | n/a |
| Plate, face separation (planes parallel to the bottom plane) | measured (53 + 79 views) | 0.90 mm | 2.6 | -1.70 |
| Plate, thickness used | user-supplied | 2.6 mm | 2.6 | n/a |

- **Where the ridge is.** Neither the v3 direction nor the brief's "underside" fits the video:
  - Out of the rim face (v3): a 2.3 mm block would reach through the mat video 1 rests on.
  - On the opposite, flat face (the brief): video 1 shows that face flat, and video 3 lies flat on it.
  - The frames show a rounded rail inside the pocket, just below the rim. A per-pixel plane sweep (video 3) puts its surface at z 3.0-3.6 for x 165.2-165.6; a hill with a 2.3 mm base peaks at z 3.04. The owner should confirm the side.
- **Plate.** The model puts the plate's outer face on the bottom plane, which is fitted to the rails' flat face (rms 0.27 mm). The plane sweep places that face 1.30 mm under the plane, which conflicts with "flat like the bottom". Both plate conflicts need calipers.
- **Flatness** (RMS to their own plane, pocket side, measured short of the ridge):
  - review_4 region: 0.676 mm before, 0.013 mm after.
  - review_5 strip: 0.853 mm before, 0.008 mm after.
- **Window edge.** Where the plate starts, the carved edge was already straight (0.029 mm rms to a fitted line). The jagged look was the hull ramp under the plate, which the plate now replaces.
- **Held-out IoU** (3 folds, 0.15 mm): 0.9529 for the plain carve, 0.9516 with features and corrections (v3 with its features: 0.9532).
- **Anomaly finder** (`backend/capture/review.py`), on the unfixed v3 model:
  - Recall 3 of the owner's 3 marks with 0 false alarms in 2 items, after tuning over 2304 threshold sets. The untuned placeholders gave 15 false alarms in 24 items.
  - Tuned and tested on the same marks of one part, so this is not a held-out number. The few-views flag is off: every setting of it added false alarms here.
- **Corrections** (`backend/capture/model.py`, schema in `corrections.py`): `rebuild(model, answers, corrections)` turns the v3 model into v4 in 1.9 s with no model call, and gives the same mesh as the v4 run.
- **Question cards** (`backend/capture/question_cards.py`): one card per question and review item, each checked by eye to mark the right feature. Two cards (ridge base width, plate thickness) use a frame where the dimension arrow is under 12 px, because no frame shows it larger.
- Two cached crops (A_orbit_02_1586, A_orbit_03_910) are the owner's annotated label images. Cards skip them; neither was used for the ridge detection.

## Regression tests

- `backend/tests/test_golden_B.py` reruns the carve and feature steps from the private cache at 0.25 mm voxels in 114 s. v4 adds these checks:
  - the owner's corrections rebuild the v4 model from the v3 model (same vertices), watertight
  - the lock ridge is a hill on the slot column's inner wall (peak 2.3 mm out, narrow at the top)
  - the review_4 region is flat (pocket side), with its outer face within 0.15 mm rms of the bottom plane
  - the anomaly finder (frozen thresholds) flags the owner's three marks on the v3 model
  - a card exists for every question

  It also asserts:
  - footprint within 0.5 mm and height within 0.3 mm of the calipers
  - held-out IoU >= 0.93
  - four slots
  - bump present and watertight
  - plate thickness 2.6 ± 0.15 mm
  - plate roughness not worse than before
  - `rebuild()` reproduces the saved model

  It is skipped when the cache is missing.
- `backend/tests/test_rebuild.py` checks `rebuild()` on a synthetic part, with no private data.
- `backend/tests/test_corrections.py` checks each correction, the hill profile, the schema and the sentence hook (with a fake model call) on a synthetic slab.
- `backend/tests/test_question_cards.py` checks the answer sanity rules (for example, a slot deeper than the part is tall is rejected), the diagrams and the highlight-colour choice.
