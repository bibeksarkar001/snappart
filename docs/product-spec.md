# SnapPart product spec

Scope: parts that are flat (extruded) or round (turned). Free-form organic shapes are out of scope. Never claim accuracy we have not measured against calipers.

## Part types (user chooses first)

| Type | Examples | Required photos | Optional photos |
| --- | --- | --- | --- |
| Flat part (extrude) | brackets, plates, tags, gaskets | Top, Front | Back, Right, Left |
| Round part (revolve) | wheels, knobs, spacers, bushings, axles | End view (looking along the axle), Side view (profile, axle horizontal) | Opposite end, close-up of axle hole |

Flat: top silhouette gives outline and holes. Front/side silhouette gives thickness in mm from the reference scale. Other views validate that the part is a straight extrusion and flag mismatches.

Round: side silhouette is the half-profile, revolved around the axle. The end view gives concentric radii: outer, hub, axle hole. If the axle hole is too few pixels wide to measure reliably, the app requests a close-up instead of guessing. Stretch: intersect the revolved body with the extruded end-view silhouette for spoked wheels.

## Design mode (2D to 3D)

Label in the UI: "Simple designs only for now: sketches, logos, names."

- Inputs: upload a high-contrast PNG or JPG (sketch, logo), or type a name and choose a font. SVG upload is a stretch.
- No reference object: user sets target width in mm.
- Options: thickness, optional base plate, raised or cut-out, optional keychain hole.
- Pipeline: Muse Spark checks the image is simple enough and says what to fix if not -> threshold and clean -> contours with holes (RETR_CCOMP style, letter counters kept) -> smooth -> CadQuery extrude -> STL. Reuses the flat-part extrude code.
- Warn on features thinner than about 1 mm (may not print). Reject shaded photos, gradients and fine detail with a clear message. Never silently produce junk.
- Stretch: Muse Image (muse-image-1.0) generates a simple black-on-white silhouette from a text prompt, then the same pipeline runs. Prompt it for flat, no shading, white background.
- Build order: after the extrude and revolve paths.

## Capture screen (see docs/camera-spec.md)

- Top banner, same place every step: "Step N of M: <view>", one line of instruction, a small pose diagram, then "what next" after each capture.
- Thumbnails with a tick, tap to retake, review screen at the end.
- Fit-to-guide overlay, reference ladder (bank card, coin, manual), flash toggle where supported (off by default).

## Holding the part

- Best: stand the part on putty or a clip so no hand is in frame.
- If a hand is needed: fingertips only, below or behind the part, never over the outline. Use a plain contrasting background.
- Run SAM 3.1 with the concept "hand" and subtract it from the part mask (to be tested). If the hand mask touches the part silhouette, warn "Hand covers the edge" and ask for a retake.
- Side and end views: stand the card or coin upright next to the part, face to camera, same distance. A coin lying flat shows as a line from the side.

## Feature set

1. Part type choice (Flat or Round)
2. Guided multi-view capture, fit-to-guide overlay, reference ladder
3. Flash toggle where supported
4. Hand and support guidance with SAM hand subtraction
5. Photo check by Muse Spark, also identifies the reference object and views
6. SAM 3.1 segmentation of reference, part and hand
7. Millimetre outline, exact holes, automatic thickness from the side view
8. Revolve builder for wheels, knobs, axles
9. Muse Spark writes CadQuery; repair, Muse Wonder escalation, deterministic template fallback
10. Pipeline panel with model badges, real timings and cost
11. 3D viewer, STL download, instant parameter edits, accuracy card
12. About page and Roadmap page
13. Mobile-first, smooth UI
14. Design mode: simple 2D image or typed name to a 3D printable file

Hero demo parts: a toy car wheel (revolve), a bracket (extrude), and a name keychain (Design mode).
