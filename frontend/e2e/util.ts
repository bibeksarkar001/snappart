import type { Page } from '@playwright/test'

/** The job the app has open, from its IndexedDB draft store. */
export const currentJob = (page: Page) => page.evaluate(() => new Promise<string>(ok => { const r = indexedDB.open('snappart'); r.onsuccess = () => { const g = r.result.transaction('meta').objectStore('meta').get('current'); g.onsuccess = () => ok(g.result) } }))
