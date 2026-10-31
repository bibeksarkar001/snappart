import { createContext, useContext } from 'react'
import type { Mode, ModelState, Recorded } from './api'
import type { Draft } from './idb'

export const ROUTES = ['home', 'prepare', 'capture', 'build', 'review', 'result', 'resume', 'about', 'next', 'privacy', 'terms']
export const FLOW = ['prepare', 'capture', 'build', 'review', 'result']

export type App = {
  route: string
  go: (r: string, anchor?: string) => void
  draft: Draft | null
  save: (d: Partial<Draft> & { jobId?: string }) => Promise<void>
  open: (d: Draft) => Promise<void>
  forget: () => Promise<void>
  mode: Mode | null
  recorded: string | null
  recording: Recorded | null               // replay: the recording's facts from GET /v1/meta/models (date, part, call counts, stage times)
  models: ModelState[]
  refreshModels: () => void
  toast: (t: string) => void
  online: boolean
  pending: number
  setPending: (n: number) => void
}

export const Ctx = createContext<App>(null as unknown as App)
export const useApp = () => useContext(Ctx)

/** A question to open when the Review page mounts (set by links on Result). */
export const pendingQuestion: { id: string | null } = { id: null }

/** A static file of this site, if it exists: a dev server answers a missing file with the page, so the type is checked too. */
export async function mediaExists(path: string, type: string): Promise<boolean> {
  try { const r = await fetch(path, { method: 'HEAD' }); return r.ok && (r.headers.get('content-type') ?? '').startsWith(type) } catch { return false }
}

// Plain names for the report's feature ids and quality checks (the same words the questions use).
const PART: Record<string, string> = { grooves: 'Slots', lock_bump: 'Lock ridge', lock_pin: 'Lock pin', plate: 'Flat plate' }
const PIECE: Record<string, string> = { count: 'how many', long: 'long side', short: 'short side', faces_separation: 'gap between its faces', length_detected: 'length seen' }
export const plainName = (id: string) => {
  const [a, b] = id.split('.')
  const p = PART[a] ?? a.replace(/_/g, ' '), q = b ? PIECE[b] ?? b.replace(/_/g, ' ') : ''
  return q ? `${p}, ${q}` : p
}
const CHECKS: Record<string, string> = { watertight_manifold: 'Watertight, no holes', single_body: 'One solid piece', no_zero_thickness: 'No paper-thin walls',
  size_sane: 'Size is plausible', held_out_iou: 'Matches frames it was not built from' }
export const plainCheck = (g: string) => CHECKS[g] ?? g.replace(/_/g, ' ').replace(/^./, c => c.toUpperCase())

export const fmtTime = (x: number) => `${Math.floor(x / 60)}:${String(Math.floor(x % 60)).padStart(2, '0')}`
export const errText = (e: unknown) => (e instanceof Error ? e.message : String(e))
