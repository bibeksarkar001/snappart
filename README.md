# SnapPart

> Snap a photo, print a part.

SnapPart turns a photo of a flat part, taken next to a coin of known size, into a scaled, parametric CAD model and a printable STL file.

Built for the **Meta Global AI Developer Hackathon** (Oct 26 to Nov 4, 2026).

## Built during the hackathon

Everything in this repository was written during the hackathon window. This repo's history starts on Oct 26, 2026 (IST) with the initial commit. No code from before kickoff is included.

## How it works (planned)

1. **Photo check**: Muse Spark 1.3 looks at the photo and flags a missing coin or a tilted shot.
2. **Find coin and part**: SAM 3.1 segments the coin and the part.
3. **Measure**: OpenCV turns the masks into a millimetre outline, using the coin as the scale.
4. **Write CAD**: Muse Spark 1.3 writes a CadQuery script from the measured geometry.
5. **Build**: the script runs in a sandbox and is validated. A deterministic template builder is the fallback.
6. **View and download**: 3D viewer in the browser and an STL download.

## Meta models used

| Model | Where |
| --- | --- |
| SAM 3.1 (`sam-3.1`) | Coin and part segmentation |
| Muse Spark 1.3 (`muse-spark-1.3`) | Photo check and CadQuery script generation |
| Muse Wonder (`muse-wonder-1.0`) | Escalation when the script fails (planned) |

All calls go through the Meta Model API.

## Status

Phase 0: repository and environment setup. Results and measured accuracy will be added here as they exist.

## Run it

Setup instructions will be added as the backend and frontend land.

## License

MIT
