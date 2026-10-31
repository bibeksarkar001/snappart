# SnapPart frontend

React, Vite, TypeScript, Tailwind and three.js. It implements the approved design in `docs/design/approved/` and uses the API in `docs/api.md`.

## Run

Start the API on :8000 first (see `docs/ops.md`; replay mode needs no key). Then, in `frontend/`:

```bash
npm run dev
```

The dev server runs on https://localhost:5173 with a self-signed certificate, which the microphone needs. It proxies `/v1`, `/readyz` and `/healthz` to the API. Set `SNAPPART_API` to proxy a different API URL. For a build served from another origin, set `VITE_API_BASE` and list that origin in the API's `SNAPPART_CORS_ORIGINS`.

## Example model

The Home hero loads `public/example/part_b.glb`. The file is not in the repository; put the recorded part B preview there to show it. Without it, the hero shows "3D preview unavailable".

## Licences

Runtime dependencies and their licences:

| Package | Licence | Use |
|---|---|---|
| react, react-dom | MIT | UI |
| three | MIT | 3D viewer |
| qrcode 1.5.4 | MIT | QR code of the resume link, made in the browser (no online service) |

## Test

With the API and a worker running in replay mode (`SNAPPART_REPLAY=1`):

```bash
npx playwright test
```

The smoke test walks the main path: create a job, extract frames in the browser, upload, run the capture check, build, follow SSE progress, answer questions, record voice, open the result, download the STL and resume. It also saves a screenshot of every screen to `screenshots/`.
