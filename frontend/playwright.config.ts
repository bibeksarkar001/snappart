import os from 'node:os'
import path from 'node:path'
import { defineConfig } from '@playwright/test'

// The tests start their own API, worker and dev server, each with a fresh scratch data folder under the system temp folder, on ports
// that do not clash with a dev setup (8000/5173). "recording" serves the recorded run as it is. "camera-only" serves a scratch copy of
// it set to its own camera-only state (its report_before.json and preview_before.glb, without the owner's answers), so the "questions
// open" state can be tested; nothing in it is invented.
const PY = process.env.SNAPPART_PYTHON ?? 'python'
const ROOT = path.resolve('..')
const TMP = path.join(os.tmpdir(), 'snappart-e2e')

function servers(name: string, api: number, web: number, cameraOnly: boolean) {
  const d = path.join(TMP, name), b = `${d}/bundle`
  const env = `SNAPPART_REPLAY=1 SNAPPART_DATA_DIR=${d}/data` + (cameraOnly ? ` SNAPPART_REPLAY_DIR=${b}` : '')
  const setup = `rm -rf ${d} && mkdir -p ${d}/data` + (cameraOnly ? ` && cp -R ${ROOT}/examples/recorded/part_b ${b} && rm ${b}/answers.json && cp ${b}/report_before.json ${b}/report.json && cp ${b}/preview_before.glb ${b}/preview.glb` : '')
  return [
    { command: `sh -c '${setup} && cd ${ROOT} && (${env} ${PY} -m backend.app.worker &) && ${env} exec ${PY} -m uvicorn backend.app.main:app --port ${api}'`,
      url: `http://127.0.0.1:${api}/readyz`, reuseExistingServer: false, timeout: 90_000 },
    { command: `npx vite --port ${web} --strictPort`, env: { SNAPPART_API: `http://127.0.0.1:${api}` }, url: `https://localhost:${web}`,
      ignoreHTTPSErrors: true, reuseExistingServer: false },
  ]
}

export default defineConfig({
  testDir: 'e2e',
  timeout: 240_000,
  workers: 1,
  use: {
    ignoreHTTPSErrors: true,
    viewport: { width: 1280, height: 900 },
    permissions: ['microphone'],
    launchOptions: { args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream'] },
  },
  projects: [
    { name: 'recording', testMatch: 'smoke.spec.ts', use: { baseURL: 'https://localhost:5180' } },
    { name: 'camera-only', testMatch: 'camera-only.spec.ts', use: { baseURL: 'https://localhost:5181' } },
  ],
  webServer: [...servers('recording', 8100, 5180, false), ...servers('camera-only', 8101, 5181, true)],
})
