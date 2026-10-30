# What the part-B runs taught, as rules

Only one real part has been verified so far: part B of the lock set, filmed in three videos and compared with calipers. Every rule below comes from that part, or from synthetic tests that are labelled as synthetic. **A second real part is the next test.** Until then, any number here may not hold for other shapes, materials or colours.

The rules live in `backend/capture/learnings.py` (plus `model.estimate_answers`, `masks.learn_colour`, `report.coverage_requests` and `gates.py`) and are protected by tests.

| # | Rule | Evidence | Protected by |
|---|---|---|---|
| 1a | The robust carve's k (views that must agree a voxel is empty) scales with the view count: k = 12 at 229 views. | Held-out IoU on part B: k = 12 at 229 views (v1). With 114 browser-sized views, k = 12 gave 0.916 and k = 6 gave 0.936 (v3). | `test_learnings.py::test_k_and_voxel_scale_with_views` |
| 1b | Voxels are 0.15 mm in production. | Part B held-out IoU 0.9529 (3 folds); the golden test's one fold at 0.25 mm gives 0.9512. Finer voxels were not measured. | same |
| 2 | Silhouettes cannot see recesses. Surface fixed by fewer than 3 silhouettes, and not resting on a mat plane, is reported. If 3 or more cameras faced it, it is a recess (`not_detected`). If fewer did, the user is asked to film it, and a large area (40 mm2 or more) blocks "done" until filmed or skipped. | Part B: the slot face lies about 2 mm inside the carve, and the pocket under the plate was filled. 2.8 % of its surface is weak; one 31.2 mm2 area faced by 69 cameras is a recess. | `test_learnings.py::test_recesses_are_not_detected`, `test_app_stage5.py::test_more_footage_loop` |
| 3 | When videos place the same feature apart by more than 0.5 mm, the gap is reported and the video with more views is used. | Part B: videos 2 and 3 put the same four slot openings 1.58-1.72 mm apart in height. | `test_learnings.py::test_registration_disagreement_uses_more_views` |
| 4 | When a plane-sweep thickness and the user's caliper value disagree by more than 0.3 mm, the question reopens as a conflict that shows both values. | Part B's plate: video 0.2-0.9 mm against caliper 2.6 mm. | `test_learnings.py::test_plate_conflict_asks_and_shows_both` |
| 5 | A feature's location must be confirmed by two videos; otherwise the user is asked (for example "which side is it on?"). | Part B's ridge: found by video 3 only (29 views; video 2 had 2 agreeing views). | `test_learnings.py::test_location_agreement` |
| 6 | Features and measured sizes under about 1 mm are never claimed as measured; they become estimates and are asked about. | Silhouette edge precision is about 0.2-0.3 mm, and the plate sweeps varied by 0.2-0.5 mm. | `test_learnings.py::test_sub_millimetre_is_not_claimed`, `test_details.py` |
| 7 | Answers come from the job: measured where the video shows them, otherwise estimated. No part's caliper values live in the generic code. | Part B's values moved to `backend/tests/fixtures/part_b.json`; the golden numbers are unchanged. | `test_learnings.py::test_answers_come_from_the_job`, `test_golden_B.py` |
| 8 | The part's colour is learned per job from SAM's masks. An achromatic part turns colour gating off. | On part B the learned window starts at 38.3 deg (hand-tuned: 38); refined masks match the old ones (median IoU 0.9997). | `test_learnings.py::test_colour_model_is_learned_or_off` |
| 9 | Every STL passes quality gates before it is called ready; a failure makes it a labelled draft with plain reasons. The gates: watertight and manifold, one body, no one-voxel skins, a sane size, held-out IoU at least 0.90 when measured. | Part B's plain carve passes. The v4 model has two one-voxel fins where the plate correction meets the bevel, so it is a draft. | `test_app_stage5.py::test_gates`, `test_app_stage5.py::test_skip_and_draft_labelling` |

## What is still unknown

- **Error bars:** they combine held-out spread, registration gap and voxel size. On part B they are ±0.10 mm and do not cover the caliper errors (−0.26 / −0.20 / −0.11). A systematic term is missing and cannot be measured from one part.
- **Footprint size:** on synthetic parts the carve oversizes footprints by 0.4-1.3 mm; real part B was 0.2-0.26 mm undersized. Which effect dominates on other real parts is not known.
- **Hollow parts:** a cup or any hollow part whose opening rests on the mat in the flipped video is filled solid, and not flagged (see `docs/accuracy.md`, Synthetic parts). The image-based detail pass may see the opening from above, but this is untested.
- **Generic SAM concepts** ("object", "plastic part", "part") have not been tried on real frames. Part B was segmented with "brown plastic buckle".
- **Small details:** the detail pass is tested on synthetic shapes and images only. On a real carve, steps and cylinders depend on how clean the surface is.
- **Thresholds:** the review thresholds (`config.py`, REVIEW_*) were tuned on the owner's three marks of part B; their recall on another part is not known.
- **Video poses:** filming in different poses than part B's (video 2 was on its edge; the product asks for high orbit, low orbit and flipped) has not been run end to end yet (stage 8).
