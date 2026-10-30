# Approved design: SnapPart prototype v3 (Oct 30, 2026)

`snappart_prototype_v3.html` is the approved design for the whole SnapPart frontend: the website (Home, About, What's next, Privacy, Terms) and the app flow (Prepare, Capture, Build, Review, Result, Left somewhere?). It replaces the older files in `docs/design/` (screens.md, tokens.md, the earlier prototype), which describe the retired one-photo flow.

Build the real frontend (React, Vite, TypeScript, Tailwind, three.js) to match this file: its colour tokens (dark and light themes, with the switch in the header), typography (Geist and Geist Mono), layout, components, copy and states. Open it in a browser to see every screen; the hash routes are #home, #prepare, #capture, #build, #review, #result, #resume, #about, #next, #privacy, #terms.

## What the prototype fakes, and what the real app must connect to the backend
- Capture: upload, frame extraction, checks, coverage ring and the accuracy estimate are simulated.
- Build: the pipeline is a 10x replay of a real part B run; the app shows live progress events from the backend.
- Review: questions, the "Tell SnapPart something" box and the mic are simulated; the app uses the backend's questions, corrections and Muse Spark / Muse Voice calls.
- Result: Download STL, clearance rebuild, QR link, Delete this part and the report rows come from the backend.
- Left somewhere?: only the code "amber river hinge" works; the app looks codes up on the backend.
- Meta model badges ("Running" / "Not running yet") must reflect what actually ran. Never show "Running" for a model that did not run.
- The Prototype controls button (bottom right) is not part of the product. Remove it.
- The 3D part is embedded only in the private full copy. The app loads the model from the backend as a GLB.

## Placeholders still to fill
See the "SnapPart launch checklist: placeholders to fill" Google Doc: print-fit clip, caliper photo, mat PDF, social image, favicon, demo video link, live domain, contact email, last-updated date, backend host, governing law, and retention period (7 days).
