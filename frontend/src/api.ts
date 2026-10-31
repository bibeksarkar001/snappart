// Client for the SnapPart API (docs/api.md, docs/openapi.json). Every call goes through req(), which turns the error envelope into
// ApiError and reports the server's mode headers to whoever listens (the mode banner).

export const BASE: string = import.meta.env.VITE_API_BASE ?? ''

export type Mode = 'replay' | 'mock' | 'real'
export type Recorded = { label: string; recorded_at: string; part?: string; note?: string; live_model_calls?: boolean
  recorded_meta_calls?: Record<string, { calls: number; ok: number }>; pipeline_per_stage_s?: Record<string, number> | null }
export type Limits = { videos_per_job: number; max_frames_per_video: number; max_frame_bytes: number; max_frame_long_edge_px: number }
export type JobCreated = { job_id: string; resume_code: string; state: string; mode: Mode; limits: Limits }
export type VideoCheck = { verdict: string; frames: number; usable_frames: number; tips: string[]; role?: string; azimuth_bins_covered?: number[] }
export type CaptureCheck = {
  usable: boolean; verdict: string; videos: Record<string, VideoCheck>; method?: string
  coverage_map?: { azimuth_bin_deg: number; counts: number[][] }
  second_opinion?: { status: string; source?: string; overall?: string; tip?: string; reason?: string } | null
}
export type Job = {
  job_id: string; state: string; mode: Mode; warnings: string[]; error: { code: string; message: string } | null
  queue_position: number | null; eta_s: number | null; outputs: string[]; uploads: { frames: number; bytes: number }
  capture_check: { verdict: string; videos: Record<string, VideoCheck> } | null; recorded?: Recorded; created_at: number
}
export type Card = { image: string; diagram: string; meta: string } | null
export type Question = {
  id: string; feature: string; label: string; kind: 'number' | 'choice' | 'conflict'; title: string; detail: string; hint: string
  unit: string | null; default: number | string | null; default_source?: string; min: number | null; max: number | null
  choices: string[] | null; candidates: (number | string | { label?: string; value: number })[] | null; conflict_message?: string | null
  applies_to: number | null; impact?: string | null; recorded_answer?: RecAnswer | null; answered: boolean; answer: number | string | null; status: string | null; card: Card
  friendly?: { title: string; detail: string; hint: string; source: string; recorded?: string } | null
}
export type ReviewItem = { id: string; kind: string; reason: string; question: string; choices: string[]; image: string | null; decision: string | null; note?: string | null }
export type CoverageRequest = { id: string; message: string; blocking?: boolean; decision: string | null }
// Replay: what the owner answered in the recording (docs/api.md, recorded-answers); each question carries it as recorded_answer.
type RecAnswer = { value: number | string; status?: string; recorded?: string }
export type Questions = { questions: Question[]; open_questions: number; review_items: ReviewItem[]; coverage_requests: CoverageRequest[]; recorded?: Recorded }
export function recordedAnswers(q: Questions | null): Record<string, { value: number | string; unit: string }> {
  const out: Record<string, { value: number | string; unit: string }> = {}
  for (const x of q?.questions ?? []) if (x.recorded_answer) out[x.id] = { value: x.recorded_answer.value, unit: typeof x.recorded_answer.value === 'number' ? x.unit ?? '' : '' }
  return out
}
export type Change = { id: string; from: unknown; to: unknown; status_from?: string; status_to?: string }
export type Rebuild = { seconds: number; changes: Change[]; open_questions: number; watertight: boolean; state: string; quality?: { status: string; reasons: string[] } }
export type Proposal = { correction: Record<string, unknown> | null; explanation: string; action: string; text: string; source?: string }
export type Voice = { transcript: string; audio_seconds: number; source: string; audio_kept: boolean; proposal: Proposal }
export type ModelState = { id: string; running: boolean; reason?: string | null; calls_ok: number; calls_failed: number }
export type MetaModels = { mode: Mode; models: ModelState[]; recorded?: Recorded }
// GET /v1/jobs/{id}/inside is being added by another session; this is the shape the panel reads, every field optional.
export type Inside = {
  available?: boolean
  frames?: { video?: number; index?: number; image?: string | null; outline_accepted?: boolean; width?: number; height?: number; outline?: number[][] | number[][][] }[]
  cameras?: { video?: number; index?: number; position: number[] }[]
  stages?: { name?: string; stage?: string; seconds: number }[] | Record<string, number>
  meta_calls?: { model: string; latency_ms?: number; status?: number; ok?: boolean; timestamp?: string; endpoint?: string }[]
  recorded?: Recorded
}

export class ApiError extends Error {
  code: string; status: number; retryAfter: number | null; requestId: string | null
  constructor(status: number, code: string, message: string, retryAfter: number | null = null, requestId: string | null = null) {
    super(message); this.status = status; this.code = code; this.retryAfter = retryAfter; this.requestId = requestId
  }
}

type ModeInfo = { mode: Mode; recorded: string | null }
const listeners = new Set<(m: ModeInfo) => void>()
export const onMode = (f: (m: ModeInfo) => void) => { listeners.add(f); return () => { listeners.delete(f) } }
function noteMode(r: Response) {
  const mode = r.headers.get('X-SnapPart-Mode') as Mode | null
  if (mode) listeners.forEach(f => f({ mode, recorded: r.headers.get('X-SnapPart-Recorded') }))
}

export async function raw(path: string, init: RequestInit = {}): Promise<Response> {
  let r: Response
  try { r = await fetch(BASE + path, init) } catch { throw new ApiError(0, 'offline', 'Could not reach SnapPart. Check your connection and try again.') }
  noteMode(r)
  if (!r.ok) {
    let e: { code?: string; message?: string; retry_after_s?: number | null; request_id?: string } = {}
    try { e = (await r.json()).error ?? {} } catch { /* not the envelope */ }
    throw new ApiError(r.status, e.code ?? 'internal', e.message ?? `Request failed (${r.status}).`, e.retry_after_s ?? null, e.request_id ?? null)
  }
  return r
}

async function req<T>(path: string, method = 'GET', body?: unknown): Promise<T> {
  const r = await raw(path, body === undefined ? { method } : { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
  return r.json() as Promise<T>
}

const J = (id: string) => `/v1/jobs/${encodeURIComponent(id)}`
export const fileUrl = (id: string, name: string) => BASE + `${J(id)}/files/${name}`
export const abs = (p: string) => (p.startsWith('/') ? BASE + p : p)

export const api = {
  create: (part_name?: string) => req<JobCreated>('/v1/jobs', 'POST', part_name ? { part_name: part_name.slice(0, 80) } : {}),
  job: (id: string) => req<Job>(J(id)),
  del: (id: string) => req<{ deleted: boolean }>(J(id), 'DELETE'),
  uploads: (id: string) => req<{ videos: Record<string, number[]>; frames: number }>(`${J(id)}/uploads`),
  putFrame: (id: string, video: number, index: number, jpeg: Blob) =>
    raw(`${J(id)}/videos/${video}/frames/${index}`, { method: 'PUT', headers: { 'Content-Type': 'image/jpeg' }, body: jpeg }).then(r => r.json()),
  check: (id: string) => req<CaptureCheck>(`${J(id)}/capture-check`, 'POST'),
  start: (id: string, anyway = false) => req<Job>(`${J(id)}/start`, 'POST', anyway ? { build_anyway: true } : {}),
  resume: (code: string) => req<{ job_id: string; state: string }>(`/v1/resume/${encodeURIComponent(code.trim())}`),
  questions: (id: string) => req<Questions>(`${J(id)}/questions`),
  answers: (id: string, answers: Record<string, number | string>) => req<{ saved: string[]; open_questions: number }>(`${J(id)}/answers`, 'POST', { answers }),
  correction: (id: string, correction: Record<string, unknown>) => req<{ correction_id: string }>(`${J(id)}/corrections`, 'POST', { correction }),
  rebuild: (id: string) => req<Rebuild>(`${J(id)}/rebuild`, 'POST'),
  mark: (id: string, point: number[], normal: number[], radius_mm = 3) => req<ReviewItem>(`${J(id)}/marks`, 'POST', { point, normal, radius_mm }),
  review: (id: string, item: string, choice: string) => req<{ decision: string }>(`${J(id)}/review/${item}`, 'POST', { choice }),
  moreFootage: (id: string, request_id: string, action: 'film' | 'skip') =>
    req<{ state: string; next_frame_index: Record<string, number> }>(`${J(id)}/more-footage`, 'POST', { request_id, action }),
  accept: (id: string) => req<Job>(`${J(id)}/accept`, 'POST'),
  propose: (id: string, text: string, item_id?: string) => req<Proposal>(`${J(id)}/proposals`, 'POST', item_id ? { text, item_id } : { text }),
  voice: (id: string, wav: Blob, item_id?: string) => {
    const f = new FormData(); f.append('audio', wav, 'speech.wav'); if (item_id) f.append('item_id', item_id)
    return raw(`${J(id)}/voice`, { method: 'POST', body: f }).then(r => r.json() as Promise<Voice>)
  },
  summary: (id: string) => req<{ summary: string; source: string; recorded?: Recorded }>(`${J(id)}/summary`),
  report: (id: string) => req<Record<string, any>>(`${J(id)}/report`),
  inside: (id: string) => req<Inside>(`${J(id)}/inside`),
  models: (id?: string) => req<MetaModels>('/v1/meta/models' + (id ? `?job_id=${encodeURIComponent(id)}` : '')),
}

// Job state -> the step that shows it.
export function stepFor(state: string): string {
  if (['created', 'uploading', 'checking', 'ready_to_build'].includes(state)) return 'capture'
  if (['queued', 'building', 'failed', 'cancelled'].includes(state)) return 'build'
  if (['needs_input', 'reviewing'].includes(state)) return 'review'
  if (state === 'done') return 'result'
  return 'home'
}
