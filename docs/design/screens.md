# SnapPart screens and behavior

Spec for the real frontend. The clickable mockup is in `docs/design/prototype/` (read as reference, never copy its syntax). Rules: no fake data in production; if a value is missing, show an honest empty or error state. Placeholders such as `[time]`, `[cost]`, `[MEASURED ERROR]` in the mockup must become real values or stay hidden until real values exist.

## Routes
`/` Home, `/part` Part type, `/capture` Camera, `/review`, `/run/:jobId`, `/result/:jobId`, `/design`, `/about`, `/roadmap`. Desktop `/workspace` is a single-page version of the same flow. All state lives in one store (Zustand): mode, part, reference, shots (view to image), job, parameters.

## 1. Home
Wordmark, About link. Photo mode card (button "Snap your part" to Part type). Design mode card ("Simple designs only for now." button to Design). Example chips (Bracket, Wheel, Name keychain) run a saved example job from `/api/examples` without the camera. Footer links to Roadmap.

## 2. Part type and reference
- Flat part (2 photos: Top, Front) or Round part (2 photos: End, Side). Selecting changes the photo list and resets the typed-size defaults.
- Reference object, segmented: **Bank card** (default, 85.60 x 53.98 mm), **Coin**, **Size in mm**. Info button explains why a reference is needed.
- Coin: list (US quarter 24.26, US penny 19.05, Euro 1 23.25, UK GBP 1 23.43 mm; verify the table before shipping) plus "Another coin": Muse Spark identifies it with web search and the user confirms the diameter before measuring.
- Size in mm needs TWO values, because a circle has no "side":
  - Flat: Length (longest side) and Width (across).
  - Round: Diameter (across the circle) and Length (along the axle).
  Steppers with minus and plus; these become the scale and seed the result values.
- Line listing the optional extra photos. "Continue to camera".

## 3. Camera (guided capture)
Views: Flat = Top, Front (required); Back, Right, Left (optional). Round = End, Side (required); Other end, Axle hole (optional).
- Top banner "Step N of M" plus the view name and a pose label; optional views say "Optional extra view".
- Overlay: dashed cutout for the reference (card landscape; card upright for the Round side view; circle for a coin; none for typed size) plus a blue part box labelled "Part here". A tip line per view. "Phone is level" indicator (real device orientation).
- Thumbnail row for all views: tap to jump or retake; taken ones show a check; optional ones are dashed and labelled.
- Shutter takes the photo and advances to the next missing view. Retaking a taken view replaces it.
- Flash toggle only where `track.getCapabilities().torch` exists (Android Chrome); off by default; hide or explain on iOS Safari. Glare tip when on.
- "Use gallery" opens a sheet: Photo library, Take with Camera app (native fallback).
- Close asks "Discard these photos?" (Discard / Keep shooting).
- Hand check: if SAM finds a hand touching the part outline, show the sheet "Your hand touches the part" with Retake (default) and Use anyway (marks the photo with a warning).
- Review button stays disabled until the required photos exist and says which are missing.
- Photos are resized to 2048px, JPEG about 0.85, EXIF-corrected, sent with labels.

## 4. Review
List of views with status, Retake/Take, Remove (optional only). Reference summary with Change. "Build part" blocked until required photos exist.

## 5. Build / pipeline
Five steps in order, each with a model badge and real timing:
1. Check the photos: Muse Spark 1.3
2. Find reference and part: SAM 3.1
3. Measure in millimetres: OpenCV (not AI)
4. Write the CAD script: Muse Spark 1.3 (stream the code)
5. Build the solid: CadQuery (not AI)
Tap a step to expand what it did. Total time and model cost are real, from job events (SSE). Failure path: show each repair attempt (two Muse Spark repairs, then Muse Wonder 1.0, then the deterministic template) with honest text. Cancel stops the job. "See the result" enabled when done. Design mode steps: Read the design (Muse Spark), Trace outline (SAM 3.1) or Set the lettering (font engine), Scale, Write script, Build.

## 6. Result
- 3D viewer (three.js) loading the real STL; Solid / Wireframe; 3D / Top view; dimension badges from the real model.
- Title, feature tags (e.g. "Revolved profile", "1 centre hole").
- Parameters with minus/plus (wheel: outer diameter, axle hole, thickness; bracket: length, width, thickness, hole; keychain: width, thickness). Changing one enables "Rebuild with new values" (state "Up to date" when clean). Rebuild calls the rebuild endpoint and reports the real time.
- Accuracy card: user types their caliper reading; show the difference in mm and %. Green at or under 2%. The "typical error" line shows only real measured test results (hide until the accuracy table exists).
- Download STL (real file, mm). Copy script opens a sheet with the real CadQuery code and a Copy button ("Copied" state).
- Links: Edit photos (photo mode), Edit design (design mode), Start over.

## 7. Design mode
Tabs Upload / Type a name. Label "Simple designs only for now."
- Upload: choose image sheet (Photo library / Take a photo), preview, Remove. Tip: dark lines on white paper.
- Type a name: text field (1 to 12 characters, live preview), font chips, preview.
- Width and thickness steppers, Raised / Cut-out, keychain hole switch. "Make 3D" blocked while empty.

## 8. About and 9. Roadmap
About: problem, how it works (3 steps), which Meta model does what (Muse Spark, SAM 3.1, Muse Wonder, no-AI steps), measured accuracy (real numbers only), limits said plainly. Roadmap: Now / Next / Later (tabs on phone, columns on desktop); Next and Later are clearly marked "not built".

## 10. Desktop workspace
Centered, max width 1440. Left column: mode switch, part type, photo toggles/uploads, reference (with the two size steppers), Build. Center: pipeline as a horizontal row of five step cards with total time and cost, then a large viewer. Right column: parameters with Rebuild, accuracy card, Download STL, Copy script, Start over.

## Accessibility and quality bar
Real buttons and labelled inputs, aria-pressed on toggles, aria-label on icon buttons, visible focus, 44px targets, works at 390px and 1440px, smooth transitions, no layout jumps. Test on a real phone over HTTPS (camera needs it).
