# Recorded run: part B

`part_b/` is a **recorded real run** of part B, a briefcase lock plate, made on **2026-10-31** (UTC). The three videos went through the live API: SAM 3.1 outlined the frames (220 calls), Muse Spark checked the footage, wrote the question wording and the summary, and read one spoken answer (11 calls), and Muse Voice Transcribe turned that spoken answer into text (1 call). Every call is listed in `part_b/meta_calls.jsonl`.

**What is replayed** (shown as "recorded run, 2026-10-31", never as a live call):
- build progress and per-stage timings (`timings.json`);
- SAM outlines: the part-only frame crops in `inside/` and camera poses in `inside/poses.json`;
- Muse Spark output: the footage check, question wording and the summary (written after the owner's answers);
- Muse Voice output: the transcript of a spoken answer to the open slot-depth question, and the proposed answer (`voice.json`; the audio is not stored);
- the report before answers (`report_before.json`, `preview_before.glb`) and after the owner's caliper answers (`report.json`, `preview.glb`, `answers.json`, labelled user_supplied).

**What runs live on your computer:** rebuilds. When you answer a question, mark an area or accept a proposal, the model is rebuilt from `model.npz` in a few seconds, with no AI call. The STL files are made the same way on first download. These results are labelled "rebuilt on your computer from a recorded run; no model call".

**Live mode with your own key:** put `MODEL_API_KEY=...` in the repository's `.env` (never commit it), then restart the API and the worker. The API prints its mode when it starts. See "Run it" in the main README.

All footage behind this recording was captured during the hackathon. The videos themselves are not in this repository; only derived results are, and every image shows the part alone on a white background.
