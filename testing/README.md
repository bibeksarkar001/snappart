# SnapPart testing: what we measured, what we got, what's still broken

This folder shows the test results for SnapPart as they are. It includes the failures. Nothing here is tuned for these parts.

## What is in this folder
- `summary.csv`: one row per part, with the app's sizes, the caliper readings, the errors, the overlap score and the quality gate.
- `caliper_vs_app.md`: the same comparison in plain text, with notes on each part.
- `parts/partNN/app_before.json`: the app's result from the scan alone, with no answers from a person.
- `parts/partNN/app_after_with_caliper_answers.json`: the result after a person answered the app's questions with calipers. These answers are not the app's own measurements.
- `parts/partNN/caliper.txt`: the caliper readings, typed as text.
- `parts/partNN/app_preview_3_angles.jpg`: the app's own 3D preview from three angles. No hand-drawn marks.

## How the tests were run
- **Parts:** 8 parts measured with calipers (3 readings per dimension where written). The full test set has 15 parts; the others have app results but no caliper check yet.
- **Filming:** each part was filmed with a phone as 2 to 6 videos around a printed A4 mat with 26 ArUco markers. File names were kept as they came from the phone.
- **Pipeline:** the app pulls sharp frames, segments the part with SAM 3.1, carves a shape from the silhouettes, and checks it against held-out views. Each run ends with a quality gate.
- **Answers:** the app asks questions, and a person answers them with calipers. The "after" results use those answers.
- **Golden test:** the briefcase lock plate (part B) measured 51.6 x 41.9 x 11.8 mm with calipers. The app gave 51.32 x 41.69 x 11.70 mm, an overlap of 0.951. This is the best case, and it was used while building the app, so it is not a fair test on its own.
- **Caution on readings:** caliper values were read by eye from photos. Where photos disagree, the notes say so, and that part is not used for conclusions.

## Results (8 parts with calipers)
See `caliper_vs_app.md` for the full table. In summary:
- Width or diameter is within about 1 mm for parts 2, 3, 6 and 7.
- Length is far off for parts 1 and 3 (about 52 to 68 mm short).
- Height is wrong for most parts. Several thin plates (parts 4 and 5) come out about 45 mm tall. This looks like a systematic height bug, and it is the first thing to fix.
- Part 5 produced no result in the last full run.
- Part 8 passes the app's quality gate. Its caliper readings conflict, so it is not compared yet.

## What the app does not do yet
- Holes, slots and groove depths are detected, but they are not cut into the 3D model. The app lists them as "detected, not in model."
- Thin, shiny and dark parts are weak.
- Areas no camera can see are not modelled. The app marks them as not visible.
- Overall accuracy is not yet good enough for the printed parts to fit. Do not print a part from these models without measuring it first.

## Cost
- About 13,900 SAM calls across all test runs, about $35 at the rate used in the cost estimate (check against the Meta billing page).

## Reproduce
- Golden test: `pytest backend/tests` (needs private test files; skipped in the public repo without them).
- Mock build (no paid calls): run the backend with `SNAPPART_MOCK=1` and follow the README's run steps.
