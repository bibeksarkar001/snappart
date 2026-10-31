// Main-path smoke test against the API in replay mode, from the repo root:
//   SNAPPART_REPLAY=1 uvicorn backend.app.main:app --port 8000      and      SNAPPART_REPLAY=1 python -m backend.app.worker
// then, in frontend/:  npx playwright test
// It also saves a screenshot of every screen to frontend/screenshots/.
import { expect, test, type Page } from '@playwright/test'

test.use({ colorScheme: 'dark' })    // the design's default theme; the last screenshot is light

// Before shows the camera-only draft (the HUD says so and the GLB request is preview_before.glb); After returns to preview.glb.
async function toggleBeforeAfter(page: Page) {
  const ba = page.getByTestId('before-after')
  await expect(ba.getByRole('button', { name: 'Before your answers' })).toBeEnabled({ timeout: 15_000 })
  const req = page.waitForRequest(/files\/preview_before\.glb/)
  await ba.getByRole('button', { name: 'Before your answers' }).click()
  await req
  await expect(page.locator('.res .hud').first()).toContainText('BEFORE: CAMERA-ONLY DRAFT')
  await expect(page.locator('.res .stage-sm .empty')).toHaveCount(0, { timeout: 20_000 })
  await ba.getByRole('button', { name: 'After your answers' }).click()
  await expect(page.locator('.res .hud').first()).toContainText('SIZE, MM')
}

const shot = (page: Page, name: string) => page.screenshot({ path: `screenshots/${name}.png`, fullPage: true })

// A short WebM recorded in the page from a canvas: the frame extractor runs on it like on a phone video.
async function testVideo(page: Page): Promise<Buffer> {
  const b64 = await page.evaluate(async () => {
    const c = document.createElement('canvas'); c.width = 640; c.height = 360
    const x = c.getContext('2d')!, rec = new MediaRecorder(c.captureStream(30), { mimeType: 'video/webm' }), parts: Blob[] = []
    rec.ondataavailable = e => parts.push(e.data)
    rec.start()
    for (let i = 0; i < 60; i++) {
      x.fillStyle = '#eee'; x.fillRect(0, 0, 640, 360); x.fillStyle = '#222'
      for (let k = 0; k < 12; k++) x.fillRect((k * 53 + i * 4) % 640, 40 + (k % 4) * 70, 30, 30)
      await new Promise(r => setTimeout(r, 33))
    }
    rec.stop(); await new Promise(r => (rec.onstop = r))
    const buf = new Uint8Array(await new Blob(parts).arrayBuffer())
    let s = ''; for (const v of buf) s += String.fromCharCode(v)
    return btoa(s)
  })
  return Buffer.from(b64, 'base64')
}

test('main path in replay mode', async ({ page, browser }) => {
  await page.goto('/#home')
  await expect(page.getByTestId('modebar')).toContainText('recorded run, 2026-10-30')
  for (const m of ['sam-3.1', 'muse-spark-1.3', 'muse-voice-transcribe-1.0'])
    await expect(page.getByTestId('model-' + m)).toContainText('Not running yet')    // replay: never "Running"
  await shot(page, '01-home')

  await page.getByRole('button', { name: 'Film a part →' }).first().click()
  await page.getByRole('button', { name: 'Continue to capture →' }).click()
  await expect(page.getByText('Give your part a name')).toBeVisible()
  await page.locator('#pname').fill('Smoke test latch')
  await shot(page, '02-prepare')
  await page.getByRole('button', { name: 'Continue to capture →' }).click()
  await expect(page.getByRole('heading', { name: 'High ring.' })).toBeVisible()
  const code = (await page.locator('.codechip').textContent())!.trim()
  expect(code).toMatch(/^[a-z]+-[a-z]+-[a-z]+-\d\d$/)

  const video = await testVideo(page)
  for (let v = 1; v <= 3; v++) {
    await page.getByTestId('video-input').setInputFiles({ name: `v${v}.webm`, mimeType: 'video/webm', buffer: video })
    await expect(page.getByText(/Capture check: /)).toBeVisible({ timeout: 90_000 })
    if (v === 1) await shot(page, '03-capture-checked')
    if (v < 3) await page.getByRole('button', { name: `Next: video ${v + 1}` }).click()
  }
  const up = await page.request.get(`/v1/resume/${code}`).then(r => r.json())
  const uploads = await page.request.get(`/v1/jobs/${up.job_id}/uploads`).then(r => r.json())
  expect(uploads.videos['1'].length).toBeGreaterThan(10)                              // frames really reached the server

  await page.getByRole('button', { name: 'Start the build →' }).click()
  await expect(page.getByText(/not good enough to build from/)).toBeVisible()          // synthetic frames: no mat codes
  await page.getByRole('button', { name: 'Build anyway' }).click()

  await expect(page.getByRole('button', { name: /Answer \d+ questions →/ })).toBeEnabled({ timeout: 60_000 })
  // The recorded stage times arrive as SSE events; compare with the same run's timings from /inside.
  const stages: Record<string, number> = (await page.request.get(`/v1/jobs/${up.job_id}/inside`).then(r => r.json())).stages
  await expect(page.getByTestId('pipe')).toContainText(`${stages.SAM} s`)
  const total = Object.values(stages).reduce((a, x) => a + x, 0)
  await expect(page.getByTestId('timer')).toHaveText(`${Math.floor(total / 60)}:${String(Math.floor(total % 60)).padStart(2, '0')}`)
  await expect(page.getByTestId('inside')).toContainText(/Not available yet|Frames|Stage times|Meta call log/)
  await shot(page, '04-build')

  await page.getByRole('button', { name: /Answer \d+ questions →/ }).click()
  const first = page.locator('[data-testid^="q-"]').first()
  await expect(first.locator('.cardimg img')).toBeVisible()
  await first.getByRole('textbox').fill('999')
  await first.getByRole('button', { name: 'Update model' }).click()
  await expect(first.getByText(/doesn't fit this part/)).toBeVisible()                // bounds checked before sending
  await first.getByRole('textbox').fill('3')
  await first.getByRole('button', { name: 'Update model' }).click()
  // A replay bundle with the editable model rebuilds; one without it refuses (409 replay_read_only). Either is shown.
  await expect(page.getByText(/Rebuilt in [\d.]+ s|Read-only in a recorded run/).first()).toBeVisible({ timeout: 60_000 })
  await page.getByLabel('Tell SnapPart something').fill('this strip is flat like the bottom')
  await page.getByRole('button', { name: 'Send' }).click()
  await expect(page.getByTestId('proposal').or(page.getByText('Read-only in a recorded run').last())).toBeVisible()

  // Voice: the browser records WAV, mono, 16-bit, 16 kHz; check the bytes the server receives.
  let wavHeader: Buffer | null = null
  await page.route('**/voice', async r => { wavHeader = r.request().postDataBuffer(); await r.continue() })
  await page.getByRole('button', { name: 'Speak' }).click()
  await page.waitForTimeout(1200)
  await page.getByRole('button', { name: 'Stop recording' }).click()
  await expect(page.getByText(/Voice is not available here/)).toBeVisible()
  const body = wavHeader! as Buffer, i = body.indexOf('RIFF')
  expect(body.toString('latin1', i + 8, i + 12)).toBe('WAVE')
  expect(body.readUInt16LE(i + 22)).toBe(1)          // channels
  expect(body.readUInt32LE(i + 24)).toBe(16000)      // sample rate
  expect(body.readUInt16LE(i + 34)).toBe(16)         // bits
  await shot(page, '05-review')

  await page.getByRole('button', { name: /see my part →/ }).click()
  await expect(page.getByTestId('stl')).toHaveAttribute('href', /files\/model\.stl$/)
  const stl = await page.request.get((await page.getByTestId('stl').getAttribute('href'))!)
  expect(stl.status()).toBe(200)
  expect(stl.headers()['content-disposition']).toContain('attachment')
  await expect(page.locator('.res canvas')).toHaveCount(1)
  await expect(page.locator('.res .stage-sm .empty')).toHaveCount(0, { timeout: 20_000 })      // the GLB loaded
  await expect(page.getByTestId('report')).toContainText('Measured')
  await expect(page.getByTestId('report')).not.toContainText(/\d\.\d{3}/)                    // 2 decimals at most
  await expect(page.getByTestId('report')).toContainText(/used \([\d.]+ mm detected\)/)       // detected and used in one row
  await expect(page.getByTestId('qr').locator('svg')).toBeVisible()
  // 7. Before / After: replay serves the camera-only draft as preview_before.glb.
  await toggleBeforeAfter(page)
  await page.getByRole('button', { name: 'Too tight' }).click()
  await expect(page.getByText('Planned', { exact: true })).toBeVisible()
  await expect(page.getByTestId('code')).toHaveText(code)
  await shot(page, '06-result')

  // Resume on this device reopens the exact step left (Result); on a new device it opens the step for the job's state (Review).
  await page.goto('/#resume')
  await page.locator('#rcode').fill(code.replace(/-/g, ' '))
  await page.getByRole('button', { name: 'Find my part' }).click()
  await expect(page.getByText(/opens at Result/)).toBeVisible()
  await shot(page, '07-resume')
  const other = await (await browser.newContext({ ignoreHTTPSErrors: true })).newPage()
  await other.goto('/#resume')
  await other.locator('#rcode').fill(code.toUpperCase())
  await other.getByRole('button', { name: 'Find my part' }).click()
  await expect(other.getByText(/opens at Review/)).toBeVisible()
  await other.getByRole('button', { name: 'Continue →' }).click()
  await expect(other).toHaveURL(/#review$/)
  await expect(other.locator('[data-testid^="q-"]').first()).toBeVisible()
  await other.context().close()

  for (const r of ['about', 'next', 'privacy', 'terms']) {
    await page.goto('/#' + r)
    await expect(page.getByTestId('modebar')).toBeVisible()
    await shot(page, '08-' + r)
  }
  await page.setViewportSize({ width: 390, height: 844 })
  for (const r of ['home', 'review', 'result']) { await page.goto('/#' + r); await page.waitForTimeout(800); await shot(page, '09-mobile-' + r) }
  await page.emulateMedia({ colorScheme: 'light' })
  await page.goto('/#home'); await page.waitForTimeout(500); await shot(page, '10-mobile-home-light')
})

// 1, 3, 6 and the hero example: the replay shortcut straight to the recorded build, the questions and the result.
test('recorded example shortcut, turn over, mark something, replay guidance', async ({ page }) => {
  await page.goto('/#home')
  await expect(page.getByTestId('modebar')).toContainText('Replay')
  await expect(page.getByTestId('tour')).toContainText('1. See the result. 2. Toggle Before / After. 3. Try answering a question.')
  await expect(page.getByText('3D preview unavailable').first()).toBeVisible({ timeout: 15_000 })   // public/example/part_b.glb is not in the repo
  // Refuse the event stream (as the server does past 8 streams per client): progress must still arrive by long-poll.
  await page.route('**/events?**', r => (r.request().headers()['accept']?.includes('text/event-stream')
    ? r.fulfill({ status: 429, contentType: 'application/json', headers: { 'Retry-After': '5' }, body: '{"error":{"code":"too_many_streams","message":"Too many open progress streams."}}' })
    : r.continue()))
  await page.getByTestId('open-example').click()
  await expect(page).toHaveURL(/#build$/)
  await page.getByRole('button', { name: /Answer \d+ questions →/ }).click({ timeout: 60_000 })
  await expect(page).toHaveURL(/#review$/)
  await page.unroute('**/events?**')

  // 3. Turn over: the camera swings below the part (the canvas changes) and the button flips to "Turn back".
  const stage = page.locator('.revstage .stage-sm')
  await expect(stage.locator('.empty')).toHaveCount(0, { timeout: 20_000 })
  const before = await stage.locator('canvas').screenshot()
  await stage.getByRole('button', { name: 'Turn over' }).click()
  await expect(stage.getByRole('button', { name: 'Turn back' })).toBeVisible()
  await page.waitForTimeout(1500)
  expect(Buffer.compare(before, await stage.locator('canvas').screenshot())).not.toBe(0)
  await stage.getByRole('button', { name: 'Turn back' }).click()
  await page.waitForTimeout(1500)

  // 6. Mark something: click points on the model until one hits it, then choose Keep (always allowed) and expect the rebuild diff.
  await stage.getByRole('button', { name: 'Mark something' }).click()
  const box = (await stage.locator('canvas').boundingBox())!
  for (const [fx, fy] of [[0.5, 0.5], [0.42, 0.5], [0.58, 0.5], [0.5, 0.42], [0.5, 0.58], [0.38, 0.45], [0.62, 0.55], [0.45, 0.62], [0.55, 0.38]]) {
    await page.mouse.click(box.x + box.width * fx, box.y + box.height * fy)
    if (await page.getByTestId('mark').getByText('spot marked on the model').isVisible()) break
  }
  await expect(page.getByTestId('mark')).toContainText('spot marked on the model')
  await page.getByTestId('mark').getByRole('button', { name: 'Keep' }).click()
  await expect(page.getByTestId('mark').getByText(/Rebuilt in [\d.]+ s|Read-only in a recorded run/)).toBeVisible({ timeout: 60_000 })
  await shot(page, '12-review-mark')
  await expect(page.getByTestId('recorded-answer')).toHaveCount(0)                 // the API sends no recorded answers yet: nothing extra

  // Replay guidance. A recording with answers.json starts its replay job with the owner's answers applied, and each question carries
  // recorded_answer (docs/api.md). This recording has none yet, so the test sets up the same state: it saves one answer for real
  // through the API (a fixture value, not a caliper reading) and adds the matching recorded_answer to the questions response.
  const jid = await page.evaluate(() => new Promise<string>(ok => { const r = indexedDB.open('snappart'); r.onsuccess = () => { const g = r.result.transaction('meta').objectStore('meta').get('current'); g.onsuccess = () => ok(g.result) } }))
  expect((await page.request.post(`/v1/jobs/${jid}/answers`, { data: { answers: { groove_depth_mm: 3.2 } } })).status()).toBe(200)
  expect((await page.request.post(`/v1/jobs/${jid}/rebuild`)).status()).toBe(200)
  await page.route('**/v1/jobs/*/questions', async r => {
    const res = await r.fetch(), j = await res.json()
    for (const q of j.questions) if (q.id === 'groove_depth_mm') q.recorded_answer ??= { value: 3.2, status: 'user_supplied', recorded: 'recorded run, test fixture' }
    await r.fulfill({ response: res, json: j })
  })
  await page.getByRole('button', { name: /see my part →/ }).click()
  await expect(page).toHaveURL(/#result$/)
  // 1. Result opens on the "after answers" state and says so; Before is the camera-only draft.
  await expect(page.getByTestId('after-answers')).toContainText("Showing the model after the owner's recorded caliper answers")
  await expect(page.getByTestId('report')).toContainText('3.2 mm')
  await toggleBeforeAfter(page)
  await expect(page.locator('.photos img').first()).toBeVisible()                 // the API's part-only crops for this job
  // 3. The tour shows on Result too; dismissing it is remembered.
  await expect(page.getByTestId('tour')).toBeVisible()
  await shot(page, '13-result')
  await page.getByRole('button', { name: 'Dismiss tour' }).click()
  await expect(page.getByTestId('tour')).toHaveCount(0)
  await page.reload()
  await expect(page.getByTestId('report')).toBeVisible()
  await expect(page.getByTestId('tour')).toHaveCount(0)

  const ask = page.getByTestId('report').getByRole('button', { name: 'answer the question' }).first()
  await expect(ask).toBeVisible()                                                 // estimates without a recorded answer still ask
  await ask.click()
  await expect(page).toHaveURL(/#review$/)
  await expect(page.locator('.q.active').first()).toBeVisible()
  await expect(page.locator('.q.active').first().getByTestId('recorded-answer')).toHaveCount(0)

  // 2. A card with a recorded answer shows it and submits it with one tap, then rebuilds.
  await page.locator('.q.answered', { hasText: 'slots' }).getByRole('button', { name: 'Change' }).click()
  const card = page.getByTestId('q-groove_depth_mm')
  await expect(card.getByTestId('recorded-answer')).toHaveText(/Answer recorded from the owner's calipers: 3\.2 mm/)
  await shot(page, '14-review-recorded-answer')
  await card.getByRole('button', { name: 'Use this answer' }).click()
  await expect(page.getByTestId('diff').first()).toContainText(/Rebuilt in [\d.]+ s/, { timeout: 60_000 })
  await page.unroute('**/v1/jobs/*/questions')

  // Home hero with the example file present (served here from this job's own GLB; nothing is added to frontend/).
  const job = await page.evaluate(() => new Promise<string>(ok => { const r = indexedDB.open('snappart'); r.onsuccess = () => { const g = r.result.transaction('meta').objectStore('meta').get('current'); g.onsuccess = () => ok(g.result) } }))
  const glb = await (await page.request.get(`/v1/jobs/${job}/files/preview.glb`)).body()
  await page.route('**/example/part_b.glb', r => r.fulfill({ body: glb, contentType: 'model/gltf-binary' }))
  await page.goto('/#home')
  await expect(page.getByTestId('tour')).toHaveCount(0)                           // dismissal holds on Home as well
  await expect(page.getByTestId('hero-label')).toContainText('Example: part B, recorded run 2026-10-30')
  await expect(page.locator('.hero .stage-sm .empty')).toHaveCount(0, { timeout: 20_000 })
  await shot(page, '11-home-example')
})
