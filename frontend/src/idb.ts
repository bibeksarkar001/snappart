// Drafts and extracted frames, kept on this device in IndexedDB so a reload or a lost connection resumes at the same step.

export type Draft = {
  jobId: string; resumeCode: string; partName: string; size: string; desc: string; step: string
  lastSeq: number; frames: Record<string, number>; frameOffset: Record<string, number>; updatedAt: number
}

let dbp: Promise<IDBDatabase> | null = null
function db(): Promise<IDBDatabase> {
  return (dbp ??= new Promise((ok, bad) => {
    const r = indexedDB.open('snappart', 1)
    r.onupgradeneeded = () => { r.result.createObjectStore('drafts'); r.result.createObjectStore('frames'); r.result.createObjectStore('meta') }
    r.onsuccess = () => ok(r.result)
    r.onerror = () => bad(r.error)
  }))
}
async function op<T>(store: string, mode: IDBTransactionMode, f: (s: IDBObjectStore) => IDBRequest): Promise<T> {
  const d = await db()
  return new Promise((ok, bad) => {
    const r = f(d.transaction(store, mode).objectStore(store))
    r.onsuccess = () => ok(r.result as T)
    r.onerror = () => bad(r.error)
  })
}

export const getDraft = (id: string) => op<Draft | undefined>('drafts', 'readonly', s => s.get(id))
export const putDraft = (d: Draft) => op('drafts', 'readwrite', s => s.put({ ...d, updatedAt: Date.now() }, d.jobId))
export const currentId = () => op<string | undefined>('meta', 'readonly', s => s.get('current'))
export const setCurrent = (id: string | null) => op('meta', 'readwrite', s => (id ? s.put(id, 'current') : s.delete('current')))
export const putFrame = (job: string, video: number, i: number, b: Blob) => op('frames', 'readwrite', s => s.put(b, `${job}/${video}/${i}`))
export const getFrame = (job: string, video: number, i: number) => op<Blob | undefined>('frames', 'readonly', s => s.get(`${job}/${video}/${i}`))
export async function dropJob(job: string) {
  await op('drafts', 'readwrite', s => s.delete(job))
  await op('frames', 'readwrite', s => s.delete(IDBKeyRange.bound(`${job}/`, `${job}/￿`)))
}
export async function dropVideo(job: string, video: number) {
  await op('frames', 'readwrite', s => s.delete(IDBKeyRange.bound(`${job}/${video}/`, `${job}/${video}/￿`)))
}

export const newDraft = (jobId: string, resumeCode: string, partName: string, size: string, desc: string): Draft =>
  ({ jobId, resumeCode, partName, size, desc, step: 'capture', lastSeq: 0, frames: {}, frameOffset: {}, updatedAt: Date.now() })
