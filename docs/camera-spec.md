# Guided camera capture: spec

Goal: every user takes the same photos the same way, so measurement and the models get consistent input.

## Capture sequence

Every shot includes a reference object (bank card recommended, or any coin) in frame, on the same surface as the part, not touching it. See Reference object below.

| Step | Label shown | Instruction text | Required |
| --- | --- | --- | --- |
| 1 | Top | Place the part flat. Coin beside it. Hold the phone directly above, camera parallel to the table. | Yes |
| 2 | Front | Rotate the part so its front faces you. Phone at table height, looking straight at the edge. Coin beside it. | Yes (measures thickness) |
| 3 | Back | Turn the part around. Same distance and height as the front shot. | Optional |
| 4 | Right side | Turn the part so its right side faces you. Same distance and height. | Optional |
| 5 | Left side | Turn the part so its left side faces you. Same distance and height. | Optional |

Required minimum: Top + one side view. More views improve validation. Show "Add more views for a better check" after step 2.

## Screen contents

- Step counter ("Step 2 of 5"), step title, one-line instruction, always visible and in the same place.
- Guide overlay: rectangle for the part area and a circle marked "coin" so the user frames it consistently. Overlay changes per step.
- Thumbnail strip of captured steps. Tap a thumbnail to retake that step.
- Shutter button (min 44 px), Retake, Use photo, Skip (optional steps only).
- Level indicator (pitch and roll), warn only, green within 2 degrees. Hide it if the device does not expose orientation.
- After capture: preview with "Is the whole coin visible?" and "Is the part fully in frame?" before accepting.

## Flash and light

- Default: flash OFF. Flash causes glare on shiny parts.
- Show a flash toggle only if the browser exposes torch support: check `track.getCapabilities().torch`, set with `track.applyConstraints({ advanced: [{ torch: true }] })`. Android Chrome supports this; iOS Safari does not.
- If torch is unsupported, hide the toggle and show a tip: "Use soft, even light. Avoid flash on shiny parts."
- Glare tip appears on step 1: "Shiny part? Turn flash off and shoot near a window."

## Fallback

- `getUserMedia` needs HTTPS or localhost. If the camera is unavailable or denied, fall back to the native file picker with `capture="environment"` for each step, keeping the same guided text.

## Data handling

- Each photo is stored with its view label (top, front, back, right, left).
- Resize to 2048 px on the long side, JPEG quality about 0.85, apply EXIF orientation before upload.
- Send the labels to the backend so Muse Spark knows which view is which.

## Reference object

- Step 0 on the capture screen: choose the reference: Bank card (recommended, 85.60 x 53.98 mm), Coin (any), or Printed SnapPart sheet (later).
- The guide overlay matches the choice: a card-shaped rectangle or a circle marked "reference".
- Overlay is for placement only. It never claims to measure real size.
- After processing, show what was detected ("Detected: bank card, 85.60 x 53.98 mm. Correct?") with Change and Enter mm manually. Processing waits for confirmation on coins.
- Draw a scale bar on the result photo from the computed mm per pixel.
- Photos of the same job must use the same reference object and the same distance.

## Fit-to-guide overlay

- Live camera shows a fixed cutout: a reference shape (circle for a coin, card-shaped rectangle for a bank card) and, beside it, a part box. The user moves the phone until the real reference object fills the cutout. This fixes the camera distance, so every view of a job has the same magnification.
- Size modes (small, medium, large part) change the cutout and box sizes. Show the box's real size, for example "box = 80 x 60 mm", computed as box_px / (reference_px / reference_mm).
- The guide is for framing only. The measurement always comes from the reference object's pixels in the captured photo, found by SAM 3.1, never from the overlay.
- Consistency check after capture: if the measured reference size differs from the guide by more than about 5%, warn "Reference does not match the guide. Retake."
- Coin mode: the user picks the coin from a short list (default by locale), the circle resizes to that coin, and Muse Spark checks the photo against the choice.
- The reference and the part must lie on the same flat surface. The phone is parallel to the surface.
