// Flow guard, stale draft and "server not reachable" message (project "recording": the app in replay mode on a scratch data folder).
import { expect, test } from '@playwright/test'

test.use({ colorScheme: 'dark' })

test('a flow step with no draft goes back home', async ({ page }) => {
  await page.goto('/#review')
  await expect(page).toHaveURL(/#home$/)
  await expect(page.locator('.path')).toHaveCount(0)
})

test('a saved draft whose job is gone is not trusted', async ({ page }) => {
  await page.goto('/#home')
  await page.evaluate(() => new Promise<void>((ok, bad) => {
    const r = indexedDB.open('snappart', 1)
    r.onerror = () => bad(r.error)
    r.onsuccess = () => {
      const d = { jobId: 'a'.repeat(32), resumeCode: 'gone-gone-gone-00', partName: 'Gone part', size: 'm', desc: '', step: 'result', lastSeq: 0, frames: {}, frameOffset: {}, updatedAt: Date.now() }
      const tx = r.result.transaction(['drafts', 'meta'], 'readwrite')
      tx.objectStore('drafts').put(d, d.jobId); tx.objectStore('meta').put(d.jobId, 'current')
      tx.oncomplete = () => ok()
    }
  }))
  await page.goto('/#result'); await page.reload()
  await expect(page).toHaveURL(/#prepare$/)                                    // the draft's own step is not trusted
  await expect(page.locator('.path li.locked')).toHaveCount(4)                  // nothing past Prepare is reachable, no ticks
  await expect(page.locator('.path li.done')).toHaveCount(0)
  await page.goto('/#home'); await page.reload()
  await expect(page.getByText('This part is no longer available')).toBeVisible()
})

test('an unreachable server says so, with Retry', async ({ page }) => {
  await page.route('**/v1/jobs', r => (r.request().method() === 'POST' ? r.fulfill({ status: 502, body: 'Bad Gateway' }) : r.continue()))
  await page.goto('/#prepare')
  await page.locator('#pname').fill('Offline part')
  await page.getByRole('button', { name: 'Continue to capture →' }).click()
  const err = page.locator('.err', { hasText: "Can't reach" })
  await expect(err).toContainText("Can't reach the SnapPart server. Check that the backend is running, then retry.")
  await expect(err.getByRole('button', { name: 'Retry' })).toBeVisible()
  await expect(page.getByTestId('health')).toHaveClass(/on/)
})
