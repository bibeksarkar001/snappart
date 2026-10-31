// The "questions open" state, against project "camera-only": a scratch copy of the recording set to its own camera-only state (its
// report_before.json and preview_before.glb, without the owner's answers). Nothing here is invented; no screenshots.
import { expect, test } from '@playwright/test'
import { currentJob } from './util'

test.use({ colorScheme: 'dark' })

test('result is a draft while questions are open, and links to them', async ({ page }) => {
  await page.goto('/#home')
  await page.getByTestId('open-example').click()
  await page.getByRole('button', { name: /Answer \d+ questions →/ }).click({ timeout: 60_000 })
  const q = await page.request.get(`/v1/jobs/${await currentJob(page)}/questions`).then(r => r.json())
  expect(q.open_questions).toBeGreaterThan(0)
  await expect(page.getByTestId('recorded-answer')).toHaveCount(0)                // no recorded answers: nothing extra on the cards

  await page.getByRole('button', { name: /see my part →/ }).click()
  const n = q.open_questions, text = `Draft: ${n} question${n === 1 ? '' : 's'} open`
  await expect(page.getByTestId('status')).toHaveText(text + '.')
  await expect(page.getByTestId('print-status')).toHaveText(text)
  await expect(page.locator('body')).not.toContainText('Ready to print')
  await expect(page.getByTestId('after-answers')).toHaveCount(0)
  await page.getByTestId('status').locator('..').getByRole('button', { name: /Answer (it|them) →/ }).click()
  await expect(page).toHaveURL(/#review$/)
  await expect(page.locator('.q.active').first()).toBeVisible()
})
