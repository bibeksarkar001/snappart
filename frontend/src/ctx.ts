import { createContext, useContext } from 'react'
import type { Mode, ModelState } from './api'
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

export const fmtTime = (x: number) => `${Math.floor(x / 60)}:${String(Math.floor(x % 60)).padStart(2, '0')}`
export const errText = (e: unknown) => (e instanceof Error ? e.message : String(e))
