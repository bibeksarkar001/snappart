import tailwindcss from '@tailwindcss/vite'
import basicSsl from '@vitejs/plugin-basic-ssl'
import react from '@vitejs/plugin-react'
import type { ClientRequest, ServerResponse } from 'node:http'
import { defineConfig } from 'vite'

// The API runs on :8000 (docs/ops.md); the dev server proxies it so the app is same-origin (no CORS setup needed).
const api = process.env.SNAPPART_API ?? 'http://127.0.0.1:8000'
// When the browser drops a request (a closed event stream), drop the upstream one too; otherwise each closed stream stays open on the
// API and uses up its per-client stream limit (8), since every proxied request comes from the same address.
const proxy = { target: api, configure: (p: { on: (e: 'proxyReq', f: (up: ClientRequest, req: unknown, res: ServerResponse) => void) => void }) =>
  p.on('proxyReq', (up, _req, res) => res.on('close', () => up.destroy())) }

// Dev only: scripts/dev.sh also runs a second API in replay mode on :8200. The cookie sp-mode=recorded (set by the header switch) sends
// every API call there, event streams included (a cookie travels with EventSource, a header would not).
const replayApi = process.env.SNAPPART_API_REPLAY ?? 'http://127.0.0.1:8200'
const pickMode = { name: 'sp-mode', configureServer(s: { middlewares: { use: (f: (req: { url?: string; headers: { cookie?: string } }, res: unknown, next: () => void) => void) => void } }) {
  s.middlewares.use((req, _res, next) => { if (/(^|;\s*)sp-mode=recorded/.test(req.headers.cookie ?? '') && /^\/(v1|healthz|readyz)/.test(req.url ?? '')) req.url = '/replay' + req.url; next() })
} }

export default defineConfig({
  plugins: [react(), tailwindcss(), basicSsl(), pickMode],
  // Localhost only. SNAPPART_LAN=1 opens it to the network (for a phone); then anyone on that network can use the live API and spend its key.
  server: { host: process.env.SNAPPART_LAN === '1', proxy: { '/replay': { ...proxy, target: replayApi, rewrite: (p: string) => p.replace(/^\/replay/, '') }, '/v1': proxy, '/readyz': api, '/healthz': api } },
})
