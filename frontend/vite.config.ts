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

export default defineConfig({
  plugins: [react(), tailwindcss(), basicSsl()],
  server: { host: true, proxy: { '/v1': proxy, '/readyz': api, '/healthz': api } },
})
