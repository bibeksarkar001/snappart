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

## Generic path checks (backend stage 5)

- **Part B's answers are now fixture data** (`backend/tests/fixtures/part_b.json`): slot depth 3.7, ridge 19.7 / 2.3 / 2.3, plate 2.6. The pipeline's defaults are per-job estimates:
  - measured where the video gives a value of at least 1 mm (for example the ridge's 1.90 mm extent, its 19.37 mm length);
  - otherwise estimated (slot depth = half the part's height) and asked.
  - The golden test passes with unchanged numbers: footprint 51.32 x 41.69 mm, height 11.70, held-out IoU 0.9512, bump 19.37 mm.
- **Colour model:** the part's colour is learned per job from SAM's own masks; an achromatic part turns colour gating off. On 204 cached part-B crops:
  - the learned hue window starts at 38.3 deg (the hand-tuned value was 38);
  - the refined masks equal the old ones, median IoU 0.9997 (min 0.9983, 68 crops).
- **SAM concepts** are generic now ("object", "plastic part", "part"; a job's part name goes first). The generic concepts have not been tried on real frames yet.
- **Coverage on part B:**
  - 2.8 % of the surface is fixed by fewer than 3 silhouettes;
  - one area of 31.2 mm2 (the plate's face) had 69 cameras facing it, so it is reported as a recess, not_detected;
  - no area needs more footage.
- **Quality gates on part B:**
  - the plain carve passes every gate;
  - the v4 model (with the owner's corrections) fails the zero-thickness gate: two one-voxel fins (4.2 and 3.4 mm2) at x 167, z 9.3, where the re-placed plate meets the bevelled slot column, a defect of the v4 plate correction. It is a draft until that is fixed.
- **Error bars on part B** (spread of 3 held-out carves at 0.25 mm, half the 0.125 mm registration gap, half a voxel): +-0.10 mm on length, width and height.
  - They do NOT cover the caliper errors (-0.26, -0.20, -0.11).
  - The folds hardly differ (spread under 0.01 mm), so the bars capture random spread, not part B's systematic under-size.
  - Treat them as a lower bound until a systematic term is measured on more than one part.

## Synthetic parts

SYNTHETIC: CadQuery shapes, not real parts.

**Capture:** three videos rendered on the mat, 60 views in total: high orbit, low orbit, and the part turned over. Marker poses come from a noisy rendered mat (largest camera-centre error 0.2 mm). Silhouettes have SAM-like edge noise (sub-pixel jitter; 1 px erosion or dilation in 40 % of views).

**Simplifications:** registration of the flipped video is perfect (its error is not simulated), the voxels are 0.25 mm, and 10 views are held out.

**Reproduce:** `python scripts/synthetic_parts.py`.

| Part (synthetic) | Truth L x W x H (mm) | Error L / W / H (mm) | Held-out IoU | Volume ratio | Not detected (mm2) |
|---|---|---|---|---|---|
| box | 30.0 x 20.0 x 10.0 | +0.96 / +1.15 / +0.05 | 0.974 | 1.05 | 0 |
| cylinder | 20.0 x 20.0 x 12.0 | +0.42 / +0.40 / +0.05 | 0.969 | 1.05 | 0 |
| L-bracket | 30.0 x 20.0 x 15.0 | +0.96 / +1.30 / +0.05 | 0.964 | 1.33 | 95 |
| plate with holes | 40.0 x 30.0 x 4.0 | +0.58 / +0.60 / +0.05 | 0.978 | 1.02 | 0 |
| plate with slots | 40.0 x 30.0 x 4.0 | +0.58 / +0.60 / +0.05 | 0.978 | 1.02 | 0 |
| hill ridge part | 30.0 x 20.0 x 6.5 | +0.57 / +0.62 / +0.04 | 0.964 | 1.38 | 0 |
| thin-walled part | 30.0 x 20.0 x 12.0 | +1.09 / +1.29 / +0.05 | 0.974 | 3.88 | 0 |

**Findings:**
- **Footprint overshoot:** footprints come out 0.4-1.3 mm too large, while heights are within 0.06 mm (fixed by the two mat planes). The overshoot stays with all noise off (box: +0.96 / +1.14) and grows with k (k = 6: +1.50 / +1.73). It comes from the robust carve keeping material near convex edges that fewer than k views see tangentially.
  - Real part B (229 views, k = 12) was 0.2-0.26 mm under-size instead.
  - No threshold was tuned on these parts.
- **Volume:** the hull fills what silhouettes cannot see: the hill ridge part (x1.38) and the L-bracket's inside corner (x1.33).
- **Known gap:** the thin-walled cup (x3.88) is filled solid and NOT reported as not_detected. Its open side rests on the mat in the flipped video, so the filled area looks fixed by the mat plane. `test_synthetic_parts.py` keeps this as a strict expected failure.

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
- `backend/tests/test_learnings.py` covers each learnings rule (`backend/capture/learnings.py`), the per-job answers (no part-B constants) and the learned colour model.
- `backend/tests/test_details.py` checks the small-detail candidates on synthetic CadQuery parts (holes, boss, step, fillets, chamfers) and on a synthetic face image (hole, slot, text; a 0.3 mm speck is never claimed).
- `backend/tests/test_synthetic_parts.py` checks three synthetic parts against the bounds above, plus the cup gap as a strict expected failure.

