import { defineConfig } from '@playwright/test'

// Needs the API on :8000 in replay mode (see e2e/smoke.spec.ts). The dev server is started here if it is not running.
export default defineConfig({
  testDir: 'e2e',
  timeout: 180_000,
  use: {
    baseURL: 'https://localhost:5173',
    ignoreHTTPSErrors: true,
    viewport: { width: 1280, height: 900 },
    permissions: ['microphone'],
    launchOptions: { args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream'] },
  },
  webServer: { command: 'npx vite --port 5173 --strictPort', url: 'https://localhost:5173', reuseExistingServer: true, ignoreHTTPSErrors: true },
})
