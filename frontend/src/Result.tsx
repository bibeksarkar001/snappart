import { useEffect, useMemo, useState, type ReactNode } from 'react'
import QRCode from 'qrcode'
import { abs, api, ApiError, fileUrl, raw, type Inside as InsideT, type Question } from './api'
import { errText, pendingQuestion, useApp } from './ctx'
import { NoPart, RecChip } from './Flow'
import { Viewer } from './Viewer'

/** Before / After for a part: preview_before.glb if the API has one for this job, else the toggle is disabled. */
const TOUR = 'sp-tour-dismissed'

/** Replay only: a short, dismissible guide for a first look (dismissal kept in localStorage). */
export function Tour() {
  const { mode } = useApp()
  const [off, setOff] = useState(() => { try { return localStorage.getItem(TOUR) === '1' } catch { return false } })
  if (mode !== 'replay' || off) return null
  const close = () => { try { localStorage.setItem(TOUR, '1') } catch { /* no storage: hidden for this visit only */ } setOff(true) }
  return <div className="welcome" data-testid="tour" style={{ marginBottom: 20 }}><div className="ic" aria-hidden="true">→</div><div className="tx"><b>1. See the result. 2. Toggle Before / After. 3. Try answering a question.</b></div>
    <button className="ib" aria-label="Dismiss tour" onClick={close}>×</button></div>
}

export function useBeforeAfter(jobId: string | null, fallbackSrc: string | null = null) {
  const [has, setHas] = useState<boolean | null>(null), [before, setBefore] = useState(false)
  useEffect(() => {
    setBefore(false); setHas(jobId ? null : false)
    // ponytail: the probe downloads the file once; switch to a HEAD request if the API adds one
    if (jobId) raw(`/v1/jobs/${jobId}/files/preview_before.glb`).then(() => setHas(true), () => setHas(false))
  }, [jobId])
  const src = jobId ? `/v1/jobs/${jobId}/files/${before ? 'preview_before' : 'preview'}.glb` : fallbackSrc
  const seg: ReactNode = <div data-testid="before-after"><div className="seg" style={{ width: 'max-content' }}>
    <button aria-pressed={before} disabled={!has} title={has === false ? 'Not available for this run' : undefined} onClick={() => setBefore(true)}>Before your answers</button>
    <button aria-pressed={!before} onClick={() => setBefore(false)}>After your answers</button></div>
    {has === false && <p className="tiny muted" style={{ marginTop: 6 }}>Before: Not available for this run.</p>}</div>
  return { src, before, seg }
}

const chip = (s: string) => ({ measured: 'c-meas', user_supplied: 'c-you', estimated: 'c-est', not_detected: 'c-unseen' }[s] ?? 'c-est')
const word = (s: string) => ({ measured: 'Measured', user_supplied: 'You answered', estimated: 'Estimated', not_detected: "Can't be seen" }[s] ?? s)
const val = (v: unknown, u: string) => (typeof v === 'number' ? `${+v.toFixed(2)}${u ? ' ' + u : ''}` : String(v))

export function Result() {
  const app = useApp(), { draft } = app
  const [rep, setRep] = useState<Record<string, any> | null>(null), [err, setErr] = useState('')
  const [summary, setSummary] = useState<{ summary: string; source: string; recorded?: { label: string } } | null>(null)
  const [draftModel, setDraftModel] = useState(false), [fit, setFit] = useState(''), [del, setDel] = useState<'idle' | 'ask' | 'done'>('idle')
  const [qs, setQs] = useState<Question[]>([]), [photos, setPhotos] = useState<NonNullable<InsideT['frames']>>([]), [qr, setQr] = useState('')
  const jobId = draft?.jobId, code = draft?.resumeCode
  const link = code ? `${location.origin}${location.pathname}?code=${encodeURIComponent(code)}#resume` : ''
  const cmp = useBeforeAfter(jobId ?? null)
  // Replay starts after the owner's recorded answers (the server applies them): say so, so Before / After reads right.
  const afterOwner = app.mode === 'replay' && qs.some(x => x.recorded_answer && x.answered && x.answer === x.recorded_answer.value)
  useEffect(() => {
    if (!jobId) return
    api.report(jobId).then(setRep).catch(e => setErr(errText(e)))
    api.summary(jobId).then(setSummary).catch(() => {})
    api.questions(jobId).then(q => setQs(q.questions)).catch(() => {})
    // The part as filmed: the API's part-only crops for this job (one per video).
    api.inside(jobId).then(i => setPhotos([1, 2, 3].map(v => (i.frames ?? []).find(f => f.video === v && f.image && f.outline_accepted)).filter(f => !!f))).catch(() => {})
  }, [jobId])
  useEffect(() => { if (link) QRCode.toString(link, { type: 'svg', margin: 0, color: { dark: '#060809', light: '#ffffff' } }).then(setQr).catch(() => setQr('')) }, [link])
  if (!draft) return <NoPart />
  if (del === 'done') return <div className="wrap flow"><div className="stack" style={{ maxWidth: 560 }}><h2>Deleted.</h2><p className="lead">Its frames, model, report and answers were removed from the server and this device.</p><button className="btn btn-primary" style={{ width: 'max-content' }} onClick={() => app.go('prepare')}>Film a part →</button></div></div>

  const d = rep?.dims_mm, size = d ? [d.length, d.width, d.height].map((x: any) => (+x.measured).toFixed(2)) : null
  const q = rep?.quality, gates: { gate: string; ok: boolean; reason: string }[] = q?.gates ?? []
  const bars = gates.find(g => g.gate === 'error_bars') as any ?? rep?.gates?.error_bars
  const isDraft = draftModel || q?.status === 'draft'
  type F = { id: string; status: string; value: unknown; unit: string; views?: number }
  const feats: F[] = rep?.features ?? []
  const byId = new Map(feats.map(f => [f.id, f]))
  // Estimated sizes are defaults, not measurements: point to the question that would measure them.
  const questionFor = (f: F) => qs.find(x => f.id.startsWith(x.feature + '.') && x.id.includes(f.id.split('.')[1]))
  const ask = (id: string) => { pendingQuestion.id = id; app.go('review') }
  const rowFor = (f: F) => {
    if (f.status === 'estimated') {
      const q = questionFor(f)
      return <Row key={f.id} n={f.id} st="ask" s={<>Not measured yet: {q ? <button className="link tiny" onClick={() => ask(q.id)}>answer the question</button> : 'answer the question'}</>} />
    }
    const det = byId.get(f.id + '_detected')
    const v = det && typeof det.value === 'number' && typeof f.value === 'number' ? `${val(f.value, f.unit)} used (${val(det.value, det.unit)} detected)` : val(f.value, f.unit)
    return <Row key={f.id} n={f.id} st={f.status} s={`${v}${f.views ? ` · ${f.views} views` : det?.views ? ` · ${det.views} views` : ''}`} />
  }
  return (
    <div className="wrap flow">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'end', gap: 16, flexWrap: 'wrap', marginBottom: 24 }}>
        <div><span className="label" style={{ color: 'var(--accent)' }}>Result · {draft.partName}</span> <RecChip /><h2 style={{ marginTop: 10 }}>{isDraft ? 'Your model is a draft.' : 'Your part is ready.'}</h2></div>
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}><button className="btn btn-ghost btn-sm" onClick={() => app.go('review')}>Back to questions</button><a className="btn btn-primary btn-sm" href={fileUrl(draft.jobId, 'model.stl')} download data-testid="stl">Download STL{isDraft ? ' (draft)' : ''}</a></div>
      </div>
      {err && <div className="err">{err}</div>}
      <Tour />
      {afterOwner && <div className="note" data-testid="after-answers" style={{ marginBottom: 20 }}><RecChip />Showing the model after the owner's recorded caliper answers. Before shows the camera-only draft.</div>}
      <div className="res">
        <div className="stack">
          <Viewer src={cmp.src} spin style={{ minHeight: 540 }} onDraft={setDraftModel}>
            <div className="hud" style={{ left: 16, top: 16 }}><span className="k">{cmp.before ? 'BEFORE: CAMERA-ONLY DRAFT' : 'SIZE, MM'}</span><span className="v a">{size ? size.join(' × ') : '…'}</span></div></Viewer>
          {cmp.seg}
          {!!photos.length && <div className="panel pp"><b>Your part, as filmed</b><div className="photos">{photos.map(f => <figure key={`${f.video}_${f.index}`}><img src={abs(f.image!)} alt={`The part in video ${f.video}`} loading="lazy" /><figcaption>video {f.video} · frame {f.index}</figcaption></figure>)}</div><p className="tiny muted">Part-only crops of your frames, with SAM 3.1's outline.</p></div>}
          <div className="panel"><div className="kpis">
            <div><div className="bign">{rep?.held_out_iou != null ? (+rep.held_out_iou).toFixed(2) : '—'}</div><p className="small muted">Held-out score: how well the model matches frames it was never built from.</p></div>
            <div><div className="bign">{bars?.value?.max_mm != null ? `≥ ±${bars.value.max_mm}` : '—'}</div><p className="small muted">{bars ? 'Size uncertainty from held-out views, mm. At least this; calipers can differ more.' : 'No size error measured for this part. Check key sizes with calipers before relying on it.'}</p></div>
          </div></div>
          {summary && <div className="panel pp"><b>In plain words</b><p className="small dim">{summary.summary}</p><span className="chip c-meta" style={{ width: 'max-content' }}>{summary.source}{summary.recorded ? ` · ${summary.recorded.label}` : ''}</span></div>}
        </div>
        <div className="stack">
          <div className="panel rep" data-testid="report">
            {size && <Row n="Overall size" s={`${size.join(' × ')} mm${d.length.caliper ? ` · calipers ${d.length.caliper} × ${d.width.caliper} × ${d.height.caliper}` : ''}`} st="measured" />}
            {feats.filter(f => !f.id.endsWith('_detected') && !Array.isArray(f.value)).map(rowFor)}{/* positions are mat coordinates: hidden */}
            {(rep?.not_detected ?? []).map((n: any) => <Row key={n.id} n={`Hidden area, ${+n.area_mm2.toFixed(0)} mm²`} s={n.reason} st="not_detected" />)}
          </div>
          <div className="panel pp"><b>{isDraft ? 'Not ready to print yet' : 'Ready to print'}</b><div className="checks">
            {gates.map(g => <div className="c" key={g.gate}><span className={'dot' + (g.ok ? ' ok' : '')} style={g.ok ? undefined : { borderColor: 'var(--bad)' }} />{g.gate.replace(/_/g, ' ')}{g.ok ? '' : `: ${g.reason}`}</div>)}
            {size && <div className="c"><span className="dot ok" />Millimetres · {size.map(x => (+x).toFixed(1)).join(' × ')}</div>}
            {(q?.not_run ?? []).map((n: string) => <div className="c" key={n}><span className="dot" />Not checked: {n.replace(/_/g, ' ')}</div>)}
          </div>{size && <p className="tiny muted">Your slicer should read about {size.map(x => (+x).toFixed(1)).join(' × ')} mm. If not, check its scale is 100%.</p>}</div>
          <div className="panel pp"><b>How did the print go?</b><div className="ans">{['Fits well', 'Too tight', 'Too loose', 'Warped', 'Something is missing'].map(x => <button key={x} className="choice" aria-pressed={fit === x} onClick={() => setFit(x)}>{x}</button>)}
            <button className="ib voice" aria-label="Say how it went" style={{ width: 40, height: 40 }} onClick={() => app.toast('Speak on the Review step: "Tell SnapPart something".')}>🎙</button></div>
            {/Too/.test(fit) && <div className="note"><span className="chip c-est">Planned</span>Clearance rebuilds are not built yet. Nothing was changed.</div>}
            {fit === 'Something is missing' && <p className="small">Tell SnapPart what is missing on the Review step. <button className="link" onClick={() => app.go('review')}>Open Review</button></p>}
            {fit === 'Warped' && <p className="small dim">Warping comes from printing, not the model. Try a brim, or print it flat side down.</p>}
            {fit === 'Fits well' && <p className="small" style={{ color: 'var(--ok)', fontWeight: 600 }}>Great. Thanks for telling us.</p>}</div>
          <div className="panel pp"><b>Save your link</b><div style={{ display: 'flex', gap: 14, alignItems: 'center' }}>{qr && <div className="qr" data-testid="qr" role="img" aria-label="QR code of your link" dangerouslySetInnerHTML={{ __html: qr }} />}<div style={{ minWidth: 0 }}><div className="mono" style={{ color: 'var(--accent)', fontSize: 16 }} data-testid="code">{draft.resumeCode}</div><p className="tiny muted">Opens this part on any device for 7 days. The code was shown once; it is kept only on this device.</p>
            <button className="btn btn-ghost btn-sm" style={{ marginTop: 8 }} onClick={async () => { try { await navigator.clipboard.writeText(link); app.toast('Link copied.') } catch { app.toast(`Copy failed. Your code is ${draft.resumeCode}.`) } }}>Copy link</button></div></div></div>
          <div className="panel pp"><b>Delete this part</b><p className="small muted">Removes its frames, model, report and answers now.</p>
            {del === 'idle' ? <button className="btn btn-ghost btn-sm" style={{ width: 'max-content' }} onClick={() => setDel('ask')}>Delete this part</button>
              : <><p className="small" style={{ color: 'var(--bad)', fontWeight: 600 }}>Delete everything for this part now? This can't be undone.</p><div style={{ display: 'flex', gap: 8 }}>
                <button className="btn btn-sm" style={{ background: 'var(--bad)', color: '#2a0a07' }} onClick={async () => { try { await api.del(draft.jobId); await app.forget(); setDel('done') } catch (e) { app.toast(errText(e)) } }}>Delete</button>
                <button className="btn btn-ghost btn-sm" onClick={() => setDel('idle')}>Keep it</button></div></>}</div>
        </div>
      </div>
      <div style={{ marginTop: 20 }}><Inside jobId={draft.jobId} /></div>
    </div>
  )
}

function Row({ n, s, st }: { n: string; s: ReactNode; st: string }) {
  const name = n.replace(/[._]/g, ' ')
  return <div className="row"><div><b>{name[0].toUpperCase() + name.slice(1)}</b><small>{s}</small></div><span className={'chip ' + (st === 'ask' ? 'c-ask' : chip(st))}>{st === 'ask' ? 'Not measured yet' : word(st)}</span></div>
}

/** "How it works" for this part, from GET /v1/jobs/{id}/inside: frames with SAM outlines, camera positions, stage times, Meta call log. */
export function Inside({ jobId }: { jobId: string }) {
  const [d, setD] = useState<InsideT | null>(null), [st, setSt] = useState<'loading' | 'missing' | 'error' | 'ok'>('loading'), [err, setErr] = useState('')
  useEffect(() => {
    api.inside(jobId).then(x => { setD(x); setSt('ok') })
      .catch(e => { if (e instanceof ApiError && (e.status === 404 || e.status === 405)) setSt('missing'); else { setErr(errText(e)); setSt('error') } })
  }, [jobId])
  const stages = !d?.stages ? [] : Array.isArray(d.stages) ? d.stages.map(s => ({ name: s.name ?? s.stage ?? '?', seconds: s.seconds })) : Object.entries(d.stages).map(([name, seconds]) => ({ name, seconds }))
  const maxS = Math.max(1, ...stages.map(s => s.seconds))
  const calls = d?.meta_calls ?? []
  const per = calls.reduce<Record<string, { n: number; ok: number; ms: number }>>((a, c) => { const k = (a[c.model] ??= { n: 0, ok: 0, ms: 0 }); k.n++; k.ok += c.ok === false ? 0 : 1; k.ms += c.latency_ms ?? 0; return a }, {})
  const cams = useMemo(() => d?.cameras?.map(c => c.position), [d])
  return (
    <details className="panel pp" open data-testid="inside"><summary style={{ cursor: 'pointer', fontWeight: 600 }}>How it works, for this part {d?.recorded && <span className="rec">{d.recorded.label}</span>}</summary>
      {st === 'loading' && <p className="small muted">Loading…</p>}
      {st === 'missing' && <p className="small muted">Not available yet: the server does not offer this part's inside view (GET /v1/jobs/…/inside) yet.</p>}
      {st === 'error' && <p className="err">{err}</p>}
      {st === 'ok' && d?.available === false && <p className="small muted">Not available for this part: {(d as { reason?: string }).reason}.</p>}
      {st === 'ok' && d && d.available !== false && <div className="stack">
        {!!d.frames?.some(f => f.image) && <div><span className="label">Frames and SAM 3.1 outlines</span><div className="inside-frames" style={{ marginTop: 8 }}>{d.frames.filter(f => f.image).map((f, k) => {
          const polys = (f.outline && typeof f.outline[0]?.[0] === 'number' ? [f.outline as number[][]] : (f.outline as number[][][] | undefined)) ?? []
          return <figure key={k}>{f.image && <img src={abs(f.image)} alt={`video ${f.video} frame ${f.index}`} loading="lazy" />}
            {f.width && f.height && <svg viewBox={`0 0 ${f.width} ${f.height}`} preserveAspectRatio="none">{polys.map((p, i) => <polygon key={i} points={p.map(q => q.join(',')).join(' ')} fill="rgba(107,149,255,.18)" stroke="#6B95FF" strokeWidth={Math.max(2, f.width! / 200)} />)}</svg>}
            <figcaption>video {f.video} · frame {f.index}</figcaption></figure>
        })}</div></div>}
        {!!cams?.length && <div><span className="label">Where the camera was ({cams.length} frames)</span><Viewer src={`/v1/jobs/${jobId}/files/preview.glb`} cameras={cams} style={{ minHeight: 320, marginTop: 8 }} /></div>}
        {!!stages.length && <div><span className="label">Stage times</span><div className="checks" style={{ marginTop: 8 }}>{stages.map(s => <div className="c" key={s.name} style={{ display: 'grid', gridTemplateColumns: '1fr 80px 56px', gap: 10 }}><span>{s.name}</span><span className="tbar" style={{ width: `${(s.seconds / maxS) * 100}%` }} /><span className="mono tiny" style={{ textAlign: 'right' }}>{s.seconds} s</span></div>)}</div></div>}
        {!!calls.length && <div><span className="label">Meta call log</span>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', margin: '8px 0' }}>{Object.entries(per).map(([m, v]) => <span key={m} className="chip c-meta">{m} · {v.ok}/{v.n} ok · {(v.ms / 1000).toFixed(1)} s total</span>)}</div>
          <div style={{ maxHeight: 280, overflow: 'auto' }}><table className="calls"><thead><tr><th>Model</th><th className="n">Time</th><th>Status</th></tr></thead><tbody>{calls.map((c, k) => <tr key={k}><td className="mono">{c.model}</td><td className="n">{c.latency_ms != null ? `${(c.latency_ms / 1000).toFixed(2)} s` : '—'}</td><td style={{ color: c.ok === false ? 'var(--bad)' : 'var(--ok)' }}>{c.status ?? (c.ok ? 'ok' : 'failed')}</td></tr>)}</tbody></table></div></div>}
      </div>}
    </details>
  )
}
