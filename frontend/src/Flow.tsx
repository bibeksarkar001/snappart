import { useCallback, useEffect, useRef, useState } from 'react'
import { api, ApiError, BASE, raw, stepFor, type CaptureCheck, type Job } from './api'
import { errText, FLOW, fmtTime, useApp } from './ctx'
import * as idb from './idb'
import { guide, MAT, ring, SLOTS } from './illus'
import { extractFrames } from './media'
import { Inside } from './Result'

const CANDIDATES = 60, KEEP = 40, LONG_EDGE = 1920        // API limits: 60 frames per video, 1920 px long edge

export const RecChip = () => { const { mode, recorded } = useApp(); return mode === 'replay' ? <span className="rec">{recorded ?? 'recorded run'}</span> : null }
const noVoiceHere = 'Voice works once your part is built, on the Review step. Type here for now.'

export function Prepare() {
  const app = useApp()
  const [name, setName] = useState(''), [nameErr, setNameErr] = useState('')
  const [size, setSize] = useState('m'), [desc, setDesc] = useState(''), [scale, setScale] = useState('')
  const [kit, setKit] = useState([false, false, false]), [busy, setBusy] = useState(false), [err, setErr] = useState('')
  const v = parseFloat(scale)
  const next = async () => {
    if (!name.trim()) { setNameErr('Give your part a name, like "toolbox lock".'); document.getElementById('pname')?.focus(); return }
    setBusy(true); setErr('')
    try {
      const j = await api.create(name.trim())
      await app.open(idb.newDraft(j.job_id, j.resume_code, name.trim(), size, desc))
      app.go('capture')
    } catch (e) { setErr(errText(e)) } finally { setBusy(false) }
  }
  return (
    <div className="wrap flow"><div className="fgrid">
      <div className="fleft"><button className="back" onClick={() => app.go('home')}>← Home</button><span className="label" style={{ color: 'var(--accent)' }}>Prepare · about 2 min</span><h2>Tell us about the part.</h2><p className="lead">Name it, pick its size, then print and tape the mat.</p><details className="why"><summary>Why the mat matters</summary><p>Each code is exactly 22 mm. Seeing six or more in a frame tells SnapPart where the camera was and how big a millimetre is, with no special hardware.</p></details></div>
      <div className="stack">
        <div className="field"><label htmlFor="pname">Part name</label><input className="input" id="pname" placeholder="Toolbox lock" value={name} maxLength={80} onChange={e => { setName(e.target.value); setNameErr('') }} /><div className="err">{nameErr}</div></div>
        <div className="field"><label>Rough size</label><div className="seg" style={{ width: 'max-content' }}>{[['s', 'Under 20 mm'], ['m', '20 to 100 mm'], ['l', 'Over 100 mm']].map(([k, t]) => <button key={k} aria-pressed={size === k} onClick={() => setSize(k)}>{t}</button>)}</div>
          <p className="tiny muted">{{ s: 'Small parts come out rough on the A4 mat. A small-code mat is on the way.', m: 'Uses the A4 mat with 22 mm codes.', l: 'Over 100 mm may not fit A4. Print the same mat on A3.' }[size]}</p></div>
        <div className="field"><label htmlFor="pdesc">Tell us about your part <span className="muted" style={{ fontWeight: 400 }}>optional</span></label><div className="composer"><button className="ib voice" aria-label="Speak instead of typing" onClick={() => app.toast(noVoiceHere)}>🎙</button><textarea id="pdesc" rows={2} placeholder="What is it, what does it do, what's hard about it?" value={desc} onChange={e => setDesc(e.target.value)} /></div><p className="tiny muted">Kept on this device. Spoken answers use Muse Voice Transcribe on the Review step; audio is deleted right after it becomes text.</p></div>
        <div className="matbox"><div dangerouslySetInnerHTML={{ __html: MAT }} /><div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, marginTop: 14, flexWrap: 'wrap' }}><span className="small muted">A4 landscape · 26 codes · 22 mm each</span><a className="btn btn-ghost btn-sm" href="/snappart_mat_A4.pdf" download>Download the mat (PDF)</a></div></div>
        <div className="field"><label htmlFor="scale">Check the print: measure one code</label><div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}><input className="input mono" id="scale" inputMode="decimal" placeholder="22.0" style={{ maxWidth: 150 }} value={scale} onChange={e => setScale(e.target.value)} /><span className="muted">mm</span><span className="small">{isNaN(v) ? '' : v >= 21.8 && v <= 22.2 ? <span style={{ color: 'var(--ok)', fontWeight: 600 }}>✓ Right scale</span> : <span style={{ color: 'var(--bad)', fontWeight: 600 }}>Reprint at 100%: should be 22 mm</span>}</span></div><p className="tiny muted">Print at 100%, not "fit to page". 21.8 to 22.2 mm is fine.</p></div>
        <div className="field"><label>Kit check</label><div className="kit">{[['Mat printed and taped flat', 'All four corners, on a rigid table.'], ['Part clean, soft even light', 'No harsh shadows across the mat.'], ['Phone and about 10 minutes', 'Calipers help later, but are optional.']].map((k, i) => (
          <button key={i} aria-pressed={kit[i]} onClick={() => setKit(kit.map((x, j) => (j === i ? !x : x)))}><span className="box">✓</span><span><b>{k[0]}</b><br /><span className="small muted">{k[1]}</span></span></button>))}</div></div>
        <button className="btn btn-primary btn-block pcta" onClick={next} disabled={busy}>{busy ? 'Creating your part…' : 'Continue to capture →'}</button>
        <div className="err">{err}</div>
        <p className="tiny muted">Private by default. No login. Frames are deleted after 7 days. <button className="link tiny" onClick={() => app.go('privacy')}>Privacy policy</button></p>
      </div></div></div>
  )
}

type Slot = { st: 'waiting' | 'extracting' | 'uploading' | 'kept' | 'error'; p: number; msg: string }

export function Capture() {
  const app = useApp(), { draft, toast, setPending } = app
  const [slot, setSlot] = useState(0)
  const [slots, setSlots] = useState<Slot[]>(() => SLOTS.map(() => ({ st: 'waiting', p: 0, msg: '' })))
  const [check, setCheck] = useState<CaptureCheck | null>(null)
  const [checking, setChecking] = useState(false)
  const [startErr, setStartErr] = useState<ApiError | null>(null)
  const [starting, setStarting] = useState(false)
  const file = useRef<HTMLInputElement>(null)
  const draftRef = useRef(draft); draftRef.current = draft

  const set = (i: number, s: Partial<Slot>) => setSlots(x => x.map((y, k) => (k === i ? { ...y, ...s } : y)))

  const runCheck = useCallback(async () => {
    const d = draftRef.current; if (!d) return
    setChecking(true)
    try { setCheck(await api.check(d.jobId)) } catch (e) { toast(errText(e)) } finally { setChecking(false) }
  }, [toast])

  // Upload every frame of a video the server does not have yet (resumable: it asks /uploads first, frames come from IndexedDB).
  const upload = useCallback(async (i: number) => {
    const d = draftRef.current!; const v = i + 1, n = d.frames[v] ?? 0, off = d.frameOffset[v] ?? 0
    set(i, { st: 'uploading', p: 0, msg: '' })
    const have = new Set((await api.uploads(d.jobId)).videos[v] ?? [])
    const todo = Array.from({ length: n }, (_, k) => k + off).filter(k => !have.has(k))
    let done = n - todo.length
    setPending(todo.length)
    for (const k of todo) {
      const b = await idb.getFrame(d.jobId, v, k)
      if (!b) throw new Error('A frame is missing on this device. Choose the video again.')
      for (let tries = 0; ; tries++) {
        if (!navigator.onLine) await new Promise(ok => window.addEventListener('online', ok, { once: true }))
        try { await api.putFrame(d.jobId, v, k, b); break } catch (e) {
          const retry = e instanceof ApiError && (e.status === 0 || e.status === 408 || e.status === 429 || e.status >= 500)
          if (!retry || tries >= 4) throw e
          await new Promise(ok => setTimeout(ok, ((e as ApiError).retryAfter ?? 2 ** tries) * 1000))
        }
      }
      done++; setPending(n - done); set(i, { p: done / n })
    }
    set(i, { st: 'kept', p: 1 })
  }, [setPending])

  // On open: anything saved on this device but not yet on the server uploads now.
  useEffect(() => {
    const d = draftRef.current; if (!d) return
    let live = true
    ;(async () => {
      const j = await api.job(d.jobId).catch(() => null)
      if (!live || !j) return
      if (stepFor(j.state) !== 'capture') return
      for (let i = 0; i < 3; i++) if (d.frames[i + 1]) { try { await upload(i) } catch (e) { set(i, { st: 'error', msg: errText(e) }) } }
      if (live && Object.keys(d.frames).length) {
        if (j.state === 'ready_to_build' && j.capture_check) setCheck(j.capture_check as CaptureCheck)
        else await runCheck()
      }
      const first = [0, 1, 2].find(i => !d.frames[i + 1]); setSlot(first ?? 2)
    })()
    return () => { live = false }
  }, [upload, runCheck])

  if (!draft) return <NoPart />

  const choose = async (f: File) => {
    const i = slot, v = i + 1
    set(i, { st: 'extracting', p: 0, msg: '' }); setStartErr(null)
    try {
      const frames = await extractFrames(f, CANDIDATES, KEEP, LONG_EDGE, (a, b) => set(i, { p: a / b }))
      const off = draft.frameOffset[v] ?? 0
      await idb.dropVideo(draft.jobId, v)
      for (let k = 0; k < frames.length; k++) await idb.putFrame(draft.jobId, v, k + off, frames[k])
      await app.save({ frames: { ...draft.frames, [v]: frames.length } })
      draftRef.current = { ...draft, frames: { ...draft.frames, [v]: frames.length } }
      await upload(i)
      await runCheck()
    } catch (e) { set(i, { st: 'error', msg: errText(e) }) }
  }
  const retake = async (i: number) => {
    await idb.dropVideo(draft.jobId, i + 1)
    const frames = { ...draft.frames }; delete frames[i + 1]
    await app.save({ frames }); set(i, { st: 'waiting', p: 0, msg: '' }); setSlot(i)
    app.toast('Removed. Choose a new video when ready.')
  }
  const start = async (anyway: boolean) => {
    setStarting(true); setStartErr(null)
    try { await api.start(draft.jobId, anyway); app.go('build') } catch (e) { setStartErr(e instanceof ApiError ? e : new ApiError(0, 'internal', errText(e))) } finally { setStarting(false) }
  }

  const s = slots[slot], all = slots.every(x => x.st === 'kept'), vc = check?.videos[String(slot + 1)]
  const nextSlot = slots.findIndex(x => x.st !== 'kept')
  const main = () => {
    if (s.st === 'kept') { if (all) void start(false); else setSlot(nextSlot) }
    else file.current?.click()
  }
  const busy = s.st === 'extracting' || s.st === 'uploading'
  return (
    <div className="wrap flow"><div className="fgrid">
      <div className="fleft"><button className="back" onClick={() => app.go('prepare')}>← Back</button><span className="label" style={{ color: 'var(--accent)' }}>Capture · video {slot + 1} of 3 · required</span><h2>{SLOTS[slot].t}</h2><p className="lead">{SLOTS[slot].l}</p>
        <ul className="small dim" style={{ paddingLeft: 18, margin: 0, display: 'grid', gap: 6 }}><li>Record with your phone's camera app, then choose the video.</li><li>Keep 6 or more codes in view. Hands out of frame.</li><li>Lock focus on the part. Distance can vary.</li></ul>
        <details className="why"><summary>Why three videos</summary><p>High and low rings see the sides from two heights. The flipped video sees the face that rests on the mat.</p></details></div>
      <div className="stack">
        <div className="slots">{SLOTS.map((x, k) => (
          <button key={k} className={'slot' + (slots[k].st === 'kept' ? ' kept' : '')} aria-current={k === slot ? 'step' : undefined} onClick={() => setSlot(k)}><span className="k">{x.k}</span><b>Required</b><span className="st">{slots[k].st === 'kept' ? '✓ Kept' : slots[k].st === 'error' ? 'Needs a retry' : k === slot ? 'Now' : 'Waiting'}</span></button>))}
          <button className="slot" disabled><span className="k">4 · Demo</span><b>Optional</b><span className="st">Not available yet</span></button>
        </div>
        <div className="guide" dangerouslySetInnerHTML={{ __html: guide(slot) }} />
        <input ref={file} type="file" accept="video/*" hidden data-testid="video-input" onChange={e => { const f = e.target.files?.[0]; e.target.value = ''; if (f) void choose(f) }} />
        <div className="stack">
          {busy && <div className="panel" style={{ padding: 16, display: 'flex', flexDirection: 'column', gap: 10 }}><div style={{ display: 'flex', justifyContent: 'space-between' }}><b>{s.st === 'extracting' ? `Pulling sharp frames from video ${slot + 1}` : `Uploading video ${slot + 1}`}</b><span className="mono small">{Math.round(s.p * 100)}%</span></div><div className="prog"><i style={{ width: `${s.p * 100}%` }} /></div><div className="checks"><div className="c"><span className="dot run" />{s.st === 'extracting' ? 'Pulling sharp frames in your browser' : `${draft.frames[slot + 1] ?? 0} frames, resumable: a lost connection continues where it stopped`}</div></div></div>}
          {s.st === 'error' && <div className="note" style={{ borderColor: 'var(--bad)' }}><span style={{ color: 'var(--bad)', fontWeight: 600 }}>{s.msg}</span><button className="btn btn-ghost btn-sm" onClick={() => (draft.frames[slot + 1] ? upload(slot).then(runCheck).catch(e => set(slot, { st: 'error', msg: errText(e) })) : file.current?.click())}>Try again</button></div>}
          {s.st === 'kept' && <div className="checks">
            <div className="c"><span className="dot ok" />{draft.frames[slot + 1]} sharp frames pulled in your browser and uploaded</div>
            {checking && <div className="c"><span className="dot run" />Checking the footage…</div>}
            {vc && <div className="c"><span className={'dot' + (vc.verdict === 'ok' ? ' ok' : '')} style={vc.verdict === 'ok' ? undefined : { borderColor: vc.verdict === 'weak' ? 'var(--ask)' : 'var(--bad)' }} />Capture check: <b>{vc.verdict}</b> · {vc.usable_frames} of {vc.frames} frames usable (mat codes, sharpness, exposure)</div>}
            {check?.second_opinion?.status === 'ok' && <div className="c"><span className={'dot' + (check.second_opinion.overall === 'looks_fine' ? ' ok' : '')} />Light and part checked <span className="chip c-meta">{check.second_opinion.source}</span>{check.second_opinion.tip ? ` · ${check.second_opinion.tip}` : ''}</div>}
          </div>}
          {s.st === 'kept' && vc && <div className="panel" style={{ padding: 16 }}><div className="cov"><span dangerouslySetInnerHTML={{ __html: ring(Array.from({ length: 360 / (check?.coverage_map?.azimuth_bin_deg ?? 30) }, (_, k) => (vc.azimuth_bins_covered ?? []).includes(k))) }} />
            <div style={{ flex: 1, minWidth: 180 }}><b>{vc.verdict === 'ok' ? 'All the way round' : vc.verdict === 'weak' ? 'One side is thin' : 'Not usable yet'}</b>
              {vc.tips.length ? vc.tips.map((t, k) => <p key={k} className="small dim" style={{ marginTop: 4 }}>{t}</p>) : <p className="small dim" style={{ marginTop: 4 }}>Every angle around the part has frames.</p>}
              <div style={{ display: 'flex', gap: 8, marginTop: 10, flexWrap: 'wrap' }}><button className="btn btn-ghost btn-sm" onClick={() => retake(slot)}>Retake</button><span className="chip c-meas" style={{ height: 38, padding: '0 14px' }}>✓ Kept</span></div></div></div></div>}
          {startErr && <div className="note" style={{ borderColor: 'var(--bad)', flexDirection: 'column', alignItems: 'flex-start' }}><span style={{ color: 'var(--bad)', fontWeight: 600 }}>{startErr.message}</span>
            {startErr.code === 'capture_unusable' && <button className="btn btn-ghost btn-sm" onClick={() => start(true)} disabled={starting}>Build anyway</button>}</div>}
        </div>
        <button className="btn btn-primary btn-block pcta" onClick={main} disabled={busy || starting || checking}>{s.st === 'kept' ? (all ? (starting ? 'Starting…' : 'Start the build →') : `Next: video ${nextSlot + 1}`) : `Choose video ${slot + 1}`}</button>
        <p className="tiny muted">Only about {KEEP} sharp frames per video are uploaded, not the video. They are kept on this device until the upload finishes.</p>
      </div></div></div>
  )
}

export function NoPart() {
  const { go } = useApp()
  return <div className="wrap flow"><div className="stack" style={{ maxWidth: 560 }}><h2>No part open.</h2><p className="lead">Start a new part, or enter your code to pick up where you were.</p><div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}><button className="btn btn-primary" onClick={() => go('prepare')}>Film a part →</button><button className="btn btn-ghost" onClick={() => go('resume')}>Left somewhere?</button></div></div></div>
}

const KNOWN = ['marker detection', 'mat self-calibration', 'poses + undistorted crops', 'SAM', 'registration', 'joint carve', 'features', 'apply answers + mesh + STL']
const tool = (st: string) => (st === 'SAM' ? ['SAM 3.1', true] : /marker|poses/.test(st) ? ['OpenCV', false] : ['our code', false]) as [string, boolean]
type Row = { name: string; st: 'wait' | 'run' | 'done'; seconds?: number }
type Ev = { seq: number; type: string; data: Record<string, any>; ts: number }

export function Build() {
  const app = useApp(), { draft } = app
  const [job, setJob] = useState<Job | null>(null)
  const [rows, setRows] = useState<Row[]>(KNOWN.map(name => ({ name, st: 'wait' })))
  const [frac, setFrac] = useState(0), [queue, setQueue] = useState<number | null>(null)
  const [warnings, setWarnings] = useState<string[]>([]), [state, setState] = useState(''), [error, setError] = useState<string | null>(null)
  const [questions, setQuestions] = useState<number | null>(null), [recordedRun, setRecordedRun] = useState(false)
  const [t0, setT0] = useState<number | null>(null), [t1, setT1] = useState<number | null>(null), [now, setNow] = useState(Date.now() / 1000)
  const jobId = draft?.jobId

  useEffect(() => {
    if (!jobId) return
    api.job(jobId).then(j => { setJob(j); setState(j.state); setQueue(j.queue_position); if (j.error) setError(j.error.message); setWarnings(j.warnings) }).catch(e => setError(errText(e)))
    let es: EventSource | null = null, last = 0, closed = false
    const stage = (name: string, p: Partial<Row>) => setRows(rs => {
      const k = rs.findIndex(r => name === r.name || name.startsWith(r.name + ' '))
      return k < 0 ? [...rs, { name, st: 'wait', ...p }] : rs.map((r, i) => (i === k ? { ...r, name, ...p } : r))
    })
    const on = (e: Ev) => {
      last = Math.max(last, e.seq)
      const d = e.data
      if (e.type === 'state_changed') { setState(d.to); if (d.to === 'building') setT0(e.ts); if (d.error) setError(d.error.message ?? String(d.error)); if (['failed', 'cancelled', 'expired'].includes(d.to)) stop() }
      if (e.type === 'stage_started') stage(d.stage, { st: 'run' })
      if (e.type === 'stage_finished') { stage(d.stage, { st: 'done', seconds: d.seconds }); if (d.recorded) setRecordedRun(true) }
      if (e.type === 'progress') setFrac(d.fraction)
      if (e.type === 'queue') setQueue(d.position)
      if (e.type === 'warning') setWarnings(w => [...w, d.message])
      if (e.type === 'build_finished') { setQuestions(d.questions); setT1(e.ts); setFrac(1); stop() }
    }
    // Once the build is over there is nothing more to follow: close, so the stream slot (8 per client) is free for others.
    const stop = () => { closed = true; es?.close() }
    // The documented fallback when a stream is refused (429 too_many_streams) or lost: long-poll the same events.
    const poll = async () => {
      while (!closed) {
        try {
          const r: { events: Ev[] } = await raw(`/v1/jobs/${encodeURIComponent(jobId)}/events?after=${last}&wait=20`).then(x => x.json())
          r.events.forEach(on)
        } catch (e) {
          if (e instanceof ApiError && e.status === 404) { stop(); setError('This part was deleted.'); return }
          await new Promise(ok => setTimeout(ok, ((e as ApiError).retryAfter ?? 3) * 1000))
        }
      }
    }
    const open = () => {
      // The browser's EventSource resends Last-Event-ID on its own reconnects; after a hard close we reopen with ?after=<last seq>.
      es = new EventSource(`${BASE}/v1/jobs/${encodeURIComponent(jobId)}/events?after=${last}`)
      for (const t of ['state_changed', 'stage_started', 'stage_finished', 'progress', 'warning', 'queue', 'build_finished'])
        es.addEventListener(t, m => on(JSON.parse((m as MessageEvent).data)))
      es.addEventListener('deleted', () => { closed = true; es?.close(); setError('This part was deleted.') })
      es.onerror = () => {
        if (closed) return
        // the browser retries by itself while the state is CONNECTING
        if (es?.readyState === EventSource.CLOSED) void poll()
      }
    }
    open()
    const tick = window.setInterval(() => setNow(Date.now() / 1000), 1000)
    return () => { closed = true; es?.close(); clearInterval(tick) }
  }, [jobId])

  const { refreshModels } = app
  useEffect(() => { if (questions !== null || ['failed', 'cancelled', 'expired'].includes(state)) refreshModels() }, [questions, state, refreshModels])

  if (!draft) return <NoPart />
  const finished = questions !== null || ['needs_input', 'reviewing', 'done'].includes(state)
  const failed = ['failed', 'cancelled', 'expired'].includes(state)
  const recTotal = rows.reduce((a, r) => a + (r.seconds ?? 0), 0)
  const elapsed = recordedRun ? recTotal : t0 ? (t1 ?? (finished ? t0 : now)) - t0 : 0
  return (
    <div className="wrap flow"><div className="fgrid">
      <div className="fleft"><span className="label" style={{ color: 'var(--accent)' }}>Build</span><h2>{finished ? 'Your model is built.' : failed ? 'The build stopped.' : 'Building your model.'}</h2><p className="lead">Close this tab if you like. Your code brings you back, and the build keeps going.</p>
        <div><div className="timer" data-testid="timer">{fmtTime(elapsed)}</div><p className="small muted" style={{ marginTop: 8 }}>{recordedRun ? 'recorded build time, sum of the measured stages' : 'elapsed'}{queue ? ` · ${queue} build${queue > 1 ? 's' : ''} ahead${job?.eta_s ? `, about ${fmtTime(job.eta_s)}` : ''}` : ''}</p></div>
        <div className="note">📏 Keep calipers or a ruler nearby. You may be asked for a number or two.</div>
</div>
      <div className="stack">
        {recordedRun && <div className="note"><RecChip />Replay of a real part B build. The times are the measured ones; nothing was re-run now.</div>}
        {warnings.map((w, k) => <div className="note" key={k} style={{ borderColor: 'var(--ask)' }}><span className="chip c-ask">warning</span>{w}</div>)}
        <div className="prog"><i style={{ width: `${(finished ? 1 : frac) * 100}%` }} /></div>
        <div className="pipe" data-testid="pipe">{rows.map(r => { const [t, meta] = tool(r.name); return (
          <div key={r.name} className={`prow ${meta ? 'meta' : ''} ${r.st === 'run' ? 'run' : r.st === 'done' ? 'done' : ''}`}><span className={'dot' + (r.st === 'done' ? ' ok' : r.st === 'run' ? ' run' : '')} />
            <div><div className="nm">{r.name} <span className={'chip ' + (meta ? 'c-meta' : 'c-code')}>{t}</span></div><div className="dt">{r.st === 'done' ? 'done' : r.st === 'run' ? 'running…' : finished ? 'not reported' : 'waiting'}</div></div>
            <span className="ms">{r.seconds != null ? `${r.seconds} s` : ''}</span></div>) })}</div>
        {error && <div className="note" style={{ borderColor: 'var(--bad)' }}><span style={{ color: 'var(--bad)', fontWeight: 600 }}>{error}</span><button className="btn btn-ghost btn-sm" onClick={async () => { await app.forget(); app.go('prepare') }}>Start again</button></div>}
        <button className="btn btn-primary btn-block pcta" disabled={!finished} onClick={() => app.go(questions === 0 || state === 'done' ? 'result' : 'review')}>
          {!finished ? 'Questions appear when the build finishes' : questions ? `Answer ${questions} question${questions > 1 ? 's' : ''} →` : 'See my part →'}</button>
        {finished && <Inside jobId={draft.jobId} />}
      </div></div></div>
  )
}

export function Resume() {
  const app = useApp()
  const [code, setCode] = useState(() => new URLSearchParams(location.search).get('code') ?? ''), [err, setErr] = useState(''), [found, setFound] = useState<{ d: idb.Draft; state: string; q: number | null } | null>(null)
  const [where, setWhere] = useState(''), [whereErr, setWhereErr] = useState(''), [step, setStep] = useState<string | null | undefined>(undefined)
  const [busy, setBusy] = useState(false)
  const find = async () => {
    const v = code.trim().toLowerCase().replace(/[\s_]+/g, '-')
    if (!v) { setErr('Enter the code from your link, like "main-tray-blind-99".'); setFound(null); return }
    if (!app.online) { setErr("You're offline. Reconnect to look up your part."); return }
    setBusy(true); setErr('')
    try {
      const r = await api.resume(v)
      const local = await idb.getDraft(r.job_id).catch(() => undefined)
      const d = local ?? { ...idb.newDraft(r.job_id, v, 'Your part', 'm', ''), step: stepFor(r.state) }
      const q = r.state === 'needs_input' ? (await api.questions(r.job_id).catch(() => null))?.open_questions ?? null : null
      setFound({ d, state: r.state, q })
    } catch (e) { setErr(e instanceof ApiError && e.code === 'not_found' ? 'No part with that code. Check the words and the number.' : errText(e)); setFound(null) } finally { setBusy(false) }
  }
  const pick = () => {
    const v = where.trim().toLowerCase()
    if (!v) { setWhereErr('Say or type where you were first.'); return }
    setWhereErr('')
    setStep(/print|tight|loose|fit|warp/.test(v) ? 'result' : /question|depth|asked/.test(v) ? 'review' : /video|film|record/.test(v) ? 'capture' : /build|wait/.test(v) ? 'build' : null)
  }
  const label = (s: string) => s[0].toUpperCase() + s.slice(1)
  return (
    <div className="wrap flow"><div className="fgrid">
      <div className="fleft"><span className="label" style={{ color: 'var(--accent)' }}>Left somewhere?</span><h2>Pick up where you were.</h2><p className="lead">Enter the code from your link. It opens your part on any device, at the exact step you left.</p><p className="small muted">Your link is private. Anyone with the code can open this part. It's deleted after 7 days.</p></div>
      <div className="stack">
        <div className="field"><label htmlFor="rcode">Resume code</label><input className="input mono" id="rcode" placeholder="main-tray-blind-99" autoComplete="off" value={code} onChange={e => { setCode(e.target.value); setErr('') }} onKeyDown={e => e.key === 'Enter' && find()} /><div className="err">{err}</div><button className="btn btn-primary btn-block" onClick={find} disabled={busy}>{busy ? 'Looking…' : 'Find my part'}</button><p className="tiny muted">Three words and a number, with spaces or hyphens.{app.draft ? <> On this device: <button className="link tiny" onClick={() => setCode(app.draft!.resumeCode)}>{app.draft.resumeCode}</button></> : null}</p></div>
        {found && <div className="welcome" style={{ margin: 0 }}><div className="ic">✓</div><div className="tx"><b>{found.d.partName}</b><div className="muted">{found.state.replace(/_/g, ' ')}{found.q ? ` · ${found.q} question${found.q > 1 ? 's' : ''} waiting` : ''} · opens at {label(FLOW.includes(found.d.step) ? found.d.step : stepFor(found.state))}</div></div>
          <button className="btn btn-primary btn-sm" onClick={async () => { await app.open(found.d); app.go(FLOW.includes(found.d.step) ? found.d.step : stepFor(found.state)) }}>Continue →</button></div>}
        <div className="field" style={{ borderTop: '1px solid var(--line)', paddingTop: 20 }}><label htmlFor="where">Tell us where you were <span className="muted" style={{ fontWeight: 400 }}>optional</span></label><p className="small muted">This picks the step. It can't find a part on its own.</p>
          <div className="composer"><button className="ib voice" aria-label="Speak" onClick={() => app.toast(noVoiceHere)}>🎙</button><textarea id="where" rows={1} placeholder="e.g. I printed it and the hinge was too tight" value={where} onChange={e => { setWhere(e.target.value); setWhereErr('') }} /><button className="ib fill" aria-label="Find my step" onClick={pick}>↑</button></div><div className="err">{whereErr}</div>
          {step !== undefined && (step ? <div className="understood"><span className="tiny" style={{ color: 'var(--meta)' }}>Picked from your words</span><b>Understood: you were at {label(step)}.</b><span className="small dim">Enter your code above to open your part there.</span><div><button className="btn btn-primary btn-sm" onClick={() => app.go(step)}>Go to {label(step)}</button></div></div>
            : <div className="understood"><b>Not sure which step. Pick one:</b><div className="ans">{FLOW.map(f => <button key={f} className="choice" onClick={() => app.go(f)}>{label(f)}</button>)}</div></div>)}</div>
      </div></div></div>
  )
}
