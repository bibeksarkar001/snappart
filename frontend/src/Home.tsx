import { useEffect, useState } from 'react'
import { api, ApiError, stepFor, type ModelState, type Recorded } from './api'
import { errText, fmtTime, mediaExists, useApp, type App } from './ctx'
import { newDraft } from './idb'
import { ART } from './illus'
import { Tour, useBeforeAfter } from './Result'
import { Viewer } from './Viewer'

const META = [
  { id: 'sam-3.1', nm: 'SAM 3.1', role: 'Sees', d: 'Outlines your part in every video frame, so the shape comes from what the camera actually saw.', inp: 'your video frames', out: 'an outline of the part in each frame', stage: 'SAM' },
  { id: 'muse-spark-1.3', nm: 'Muse Spark 1.3', role: 'Reasons', d: 'Checks your videos and outlines, writes the questions, turns your words into changes, and writes the report.', inp: 'what the camera could not measure', out: 'questions in plain words, and the report', stage: '' },
  { id: 'muse-voice-transcribe-1.0', nm: 'Muse Voice Transcribe', role: 'Listens', d: 'Turns your spoken description and answers into text, so you can keep your hands on the part.', inp: 'what you say about the part', out: 'text, then a change you confirm', stage: '' },
]
const day = (r: Recorded | null) => r?.recorded_at?.slice(0, 10) ?? ''

const EXAMPLE = '/example/part_b.glb'   // the owner places the recorded part B preview here; the hero falls back when it is missing

/** Running only from a live successful call; in replay, the recording's own calls; "Not running yet" only with no calls at all. */
export function Badge({ m, rec }: { m: ModelState | undefined; rec: Recorded | null }) {
  if (m?.running) return <span className="chip c-meas" title={`${m.calls_ok} calls succeeded`}>Running</span>
  const n = rec?.recorded_meta_calls?.[m?.id ?? '']?.calls
  if (n) return <span className="chip c-ex">Recorded run, {day(rec)}: {n} call{n === 1 ? '' : 's'}</span>
  if (!m) return <span className="chip c-est">Checking…</span>
  return <span className="chip c-ask">Not running yet</span>
}

/** Replay (or mock): start a build of the recorded run. The API needs one frame before its capture check; in replay nothing uploaded
 * is processed, so a plain labelled placeholder frame is sent and the build is started "anyway". */
export async function watchRecorded(app: App) {
  const j = await api.create('part B (recorded run)')
  const c = document.createElement('canvas'); c.width = 640; c.height = 480
  const x = c.getContext('2d')!; x.fillStyle = '#ddd'; x.fillRect(0, 0, 640, 480); x.fillStyle = '#333'; x.font = '24px sans-serif'; x.fillText('SnapPart replay placeholder', 150, 240)
  const b = await new Promise<Blob>(ok => c.toBlob(b => ok(b!), 'image/jpeg', 0.8))
  await api.putFrame(j.job_id, 1, 0, b)
  await api.check(j.job_id)
  await api.start(j.job_id, true)
  await app.open({ ...newDraft(j.job_id, j.resume_code, 'part B (recorded run)', 'm', ''), step: 'build' })
  app.go('build')
}

export function Home() {
  const app = useApp(), { draft, go, models, mode, recording } = app
  const [state, setState] = useState<string | null>(null)
  const [waiting, setWaiting] = useState(0)
  const [busy, setBusy] = useState(false)
  useEffect(() => {
    if (!draft) return
    api.job(draft.jobId).then(j => {
      setState(j.state)
      if (j.state === 'needs_input') api.questions(draft.jobId).then(q => setWaiting(q.open_questions)).catch(() => {})
    }).catch(e => setState(e instanceof ApiError && e.status === 404 ? 'expired' : null))
  }, [draft])
  const replayable = mode === 'replay' || mode === 'mock'
  const watch = async () => {
    if (!replayable) { app.toast('Recorded runs play in replay mode. Film your own part instead.'); return }
    setBusy(true)
    try { await watchRecorded(app) } catch (e) { app.toast(errText(e)) } finally { setBusy(false) }
  }
  const hasModel = !!draft && ['needs_input', 'reviewing', 'done'].includes(state ?? '')
  const cmp = useBeforeAfter(hasModel ? draft!.jobId : null, EXAMPLE)
  const date = day(recording)
  // Every number here comes from the recording (GET /v1/meta/models) or, once a recorded part is open, from its report.
  const [rep, setRep] = useState<Record<string, any> | null>(null)
  const [media, setMedia] = useState({ clip: false, photo: false })
  useEffect(() => { if (hasModel && mode === 'replay') api.report(draft!.jobId).then(setRep).catch(() => {}) }, [hasModel, mode, draft])
  useEffect(() => { void Promise.all([mediaExists('/media/print_fit.mp4', 'video/'), mediaExists('/media/original_and_copy.jpg', 'image/')]).then(([clip, photo]) => setMedia({ clip, photo })) }, [])
  const stages = recording?.pipeline_per_stage_s ?? null
  const buildS = stages ? Object.values(stages).reduce((a, x) => a + x, 0) : null
  const sam = recording?.recorded_meta_calls?.['sam-3.1']
  const iou: number | null = rep?.held_out_iou ?? null
  const dims = rep?.dims_mm ? (['length', 'width', 'height'] as const).map(k => ({ k, ...rep.dims_mm[k] })).filter(x => x.caliper != null) : []
  const maxErr = dims.length ? Math.max(...dims.map(x => Math.abs(x.measured - x.caliper))) : null
  const lbl = `recorded run, ${date}`
  const tiles: [string, string][] = [
    ...(maxErr != null ? [[`±${maxErr.toFixed(2)} mm`, `largest size error against calipers · ${lbl}`] as [string, string]] : []),
    ...(buildS != null ? [[fmtTime(buildS), `measured build time, ${recording!.part ?? 'part B'} · ${lbl}`] as [string, string]] : []),
    ...(iou != null ? [[iou.toFixed(2), `held-out score · ${lbl}`] as [string, string]] : []),
    ...(sam ? [[`${sam.ok} / ${sam.calls}`, `SAM 3.1 calls that succeeded · ${lbl}`] as [string, string]] : []),
  ]
  const welcome = !draft || !state ? null
    : state === 'expired' ? [`${draft.partName} has expired`, 'Frames and model were deleted after 7 days, or the part was deleted.', 'Start again', 'prepare']
      : state === 'needs_input' && draft.step === 'build' ? ['Your model finished while you were away', `${draft.partName} · ${waiting || 'some'} question${waiting === 1 ? ' is' : 's are'} waiting.`, 'Answer', 'review']
        : [`Welcome back · ${draft.partName}`, `${stepFor(state).replace(/^./, c => c.toUpperCase())}. Your progress is saved on this device.`, 'Continue', stepFor(state)]

  return (
    <>
      <div className="hero"><div className="wrap">
        <div style={{ maxWidth: 640, margin: '0 auto' }}><Tour /></div>
        {welcome && <div className="welcome"><div className="ic"><svg width="18" height="18" viewBox="0 0 20 20" aria-hidden="true"><rect x="2" y="5" width="11" height="10" rx="2" stroke="currentColor" strokeWidth="1.6" fill="none" /><path d="M13 8.5l5-2.5v8l-5-2.5" stroke="currentColor" strokeWidth="1.6" fill="none" strokeLinejoin="round" /></svg></div><div className="tx"><b>{welcome[0]}</b><div className="muted">{welcome[1]}</div></div>
          <button className="btn btn-primary btn-sm" onClick={async () => { if (state === 'expired') await app.forget(); go(welcome[3]) }}>{welcome[2]} →</button></div>}
        <h1>Film a part.<br /><span className="grad">Print a copy.</span></h1>
        <p className="lead">Put a broken or missing piece on a printed mat and film three short videos. SnapPart measures it in millimetres, asks you what a camera can't see, and hands you a file that prints and fits.</p>
        <div className="hero-cta"><button className="btn btn-primary" onClick={() => go('prepare')}>Film a part →</button><button className="btn btn-ghost" onClick={watch} disabled={busy}>{busy ? 'Starting…' : 'Watch a real build'}</button>
          {mode === 'replay' && <button className="btn btn-ghost" onClick={watch} disabled={busy} data-testid="open-example">Open the recorded example</button>}</div>
        <Viewer src={EXAMPLE} spin fallback="3D preview unavailable" style={{ marginTop: 'clamp(36px,5vw,64px)', height: 'clamp(380px,52vw,640px)', borderRadius: 28 }}>
          <div className="hud" style={{ left: 18, top: 18 }} data-testid="hero-label"><span className="k">PART B · EXAMPLE RUN</span><span className="v">Example: part B, recorded run{date ? ` ${date}` : ''}</span></div>
          {sam && <div className="hud opt" style={{ right: 18, top: 18 }}><span className="k">OUTLINES · {lbl.toUpperCase()}</span><span className="v m">SAM 3.1 · {sam.calls} calls</span></div>}
          {iou != null && <div className="hud opt" style={{ left: 18, bottom: 18 }}><span className="k">HELD-OUT SCORE · {lbl.toUpperCase()}</span><span className="v a">{iou.toFixed(2)}</span></div>}
        </Viewer>
        {!!tiles.length && <div className="stats" data-testid="stats" style={{ gridTemplateColumns: `repeat(${tiles.length},1fr)` }}>{tiles.map(t => <div key={t[1]}><b>{t[0]}</b><span>{t[1]}</span></div>)}</div>}
        <div className="doors">
          <button className="door" onClick={() => (hasModel ? go('result') : watch())}><span className="t">1 minute</span><b>Explore a real result</b><span className="small muted">Turn the model and see every measured size.</span><span className="go">Open result →</span></button>
          <button className="door" onClick={watch}><span className="t">2 minutes</span><b>Watch a real build</b><span className="small muted">A measured run of part B, then answer its questions yourself.</span><span className="go">Start replay →</span></button>
          <button className="door" onClick={() => go('prepare')}><span className="t">About 10 minutes</span><b>Film your own part</b><span className="small muted">Print the mat, film three videos, get your file.</span><span className="go">Get started →</span></button>
        </div>
      </div></div>

      <section id="how"><div className="wrap">
        <div className="sec-head"><span className="label">How it works</span><h2>Three phone videos in.<br /><span className="muted">A measured part out.</span></h2></div>
        <div className="steps">
          {[['01 · Film', 'Every frame knows its scale', '26 printed codes give each frame its camera position and real millimetres.'], ['02 · Outline', 'SAM 3.1 traces the part', "Meta's segmentation model outlines your part in every frame it appears in."], ['03 · Carve', 'Outlines carve the solid', 'Space outside any outline is removed. Flat faces and round bars snap true.'], ['04 · Ask', 'Muse Spark asks, you answer', "What a camera can't see, it asks. Type it, tap it or say it."]].map((s, k) => (
            <div className="step" key={k}><div className="art" dangerouslySetInnerHTML={{ __html: ART[k] }} /><span className="n">{s[0]}</span><h3>{s[1]}</h3><p className="small dim">{s[2]}</p></div>))}
        </div>
      </div></section>

      <section id="meta" style={{ overflow: 'hidden' }}><div className="glow" style={{ width: 520, height: 520, background: 'rgba(107,149,255,.18)', right: -120, top: 40 }} /><div className="wrap" style={{ position: 'relative' }}>
        <div className="sec-head"><span className="label" style={{ color: 'var(--meta)' }}>Meta AI at the core</span><h2>Three Meta models.<br />One measured part.</h2><p className="lead">SnapPart's measuring engine is ours. Seeing, reasoning and listening are Meta's. Each model has a job nothing else in the pipeline does.</p></div>
        <div className="flowline" aria-label="Pipeline"><span className="fnode">Your videos</span><span className="fline" /><span className="fnode meta">SAM 3.1</span><span className="fline" /><span className="fnode">Measuring engine</span><span className="fline" /><span className="fnode meta">Muse Spark 1.3</span><span className="fline" /><span className="fnode meta">Muse Voice</span><span className="fline" /><span className="fnode out">STL + report</span></div>
        <div className="meta-grid">{META.map(m => (
          <div className="mcard" key={m.id} data-testid={'model-' + m.id}><div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}><span className="role">{m.role}</span><Badge m={models.find(x => x.id === m.id)} rec={recording} /></div><div className="nm">{m.nm}</div><p className="small dim">{m.d}</p>
            <div className="io"><span className="in">in </span> {m.id === 'sam-3.1' && recording?.part ? recording.part : m.inp}<br /><span className="out">out</span> {m.out}
              {recording?.recorded_meta_calls?.[m.id] && <><br /><span className="in">{lbl}: </span>{recording.recorded_meta_calls[m.id].calls} call{recording.recorded_meta_calls[m.id].calls === 1 ? '' : 's'}{m.stage && stages?.[m.stage] != null ? ` · ${stages[m.stage]} s` : ''}</>}</div></div>))}</div>
        <p className="tiny muted" style={{ marginTop: 16 }}>A model shows as running only when a call to it succeeds now. Call counts are from the recording.</p>
      </div></section>

      <section><div className="wrap">
        <div className="sec-head"><span className="label" style={{ color: 'var(--ask)' }}>Ask, don't guess</span><h2>It asks before it invents.</h2><p className="lead">The first carve of part B had a lumpy strip and a plate a camera couldn't see. SnapPart flagged both. Three answers later, the model matched the real part.</p></div>
        <div className="cmpwrap">
          <Viewer src={cmp.src} spin fallback="3D preview unavailable" style={{ minHeight: 420 }}><div className="hud" style={{ left: 16, top: 16 }}><span className="k">{cmp.before ? 'BEFORE YOUR ANSWERS' : 'AFTER YOUR ANSWERS'}</span><span className="v a" style={cmp.before ? { color: 'var(--ask)' } : undefined}>{hasModel ? draft!.partName : 'Example: part B'}</span></div></Viewer>
          <div className="changes">
            {cmp.seg}
            <div className="chg"><span className="d" /><div><b>Lock ridge sized from your answers</b><p className="small dim">Found on the slot wall; its height and top shape came from you.</p></div></div>
            <div className="chg"><span className="d" /><div><b>Bumpy strip made flat</b><p className="small dim">Made flat after you said "it's flat".</p></div></div>
            <div className="chg"><span className="d" /><div><b>Plate set level with the bottom</b><p className="small dim">A face the camera never saw, filled from your answer.</p></div></div>
          </div>
        </div>
      </div></section>

      <section><div className="wrap acc">
        <div className="stack"><span className="label">Honest about accuracy</span><h2>Measured to fit, and clear about what it can't see.</h2><p className="lead">Other tools make a model that looks right. SnapPart measures, then tells you where it's sure and where it needs your number.</p>
          <div className="states"><div className="row"><span><span className="chip c-meas">Measured</span></span>From many views and the mat's scale.</div><div className="row"><span><span className="chip c-you">You entered</span></span>Something a camera can't see, like slot depth.</div><div className="row"><span><span className="chip c-est">Estimated</span></span>Only when you skip a question. Always labelled.</div><div className="row"><span><span className="chip c-unseen">Can't be seen</span></span>Hollows, blind pockets, features under 1 mm.</div></div></div>
        <div className="panel" style={{ overflow: 'hidden' }}><div style={{ padding: '18px 20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 10, flexWrap: 'wrap', borderBottom: '1px solid var(--line)' }}><b>Part B against digital calipers</b>{date && <span className="chip c-ex">{lbl}</span>}</div>
          {dims.length ? <div style={{ overflowX: 'auto' }}><table className="tbl"><thead><tr><th>Size</th><th className="n">Model</th><th className="n">Calipers</th><th className="n">Error</th></tr></thead><tbody>
            {dims.map(r => <tr key={r.k}><td>{r.k[0].toUpperCase() + r.k.slice(1)}</td><td className="n">{(+r.measured).toFixed(2)}</td><td className="n">{(+r.caliper).toFixed(2)}</td><td className="n err-good">{(r.measured - r.caliper > 0 ? '+' : '−') + Math.abs(r.measured - r.caliper).toFixed(2)}</td></tr>)}
          </tbody></table></div>
            : <p className="small dim" style={{ padding: '18px 20px' }}>{hasModel ? 'This recording has no caliper readings, so no size errors are shown.' : 'Open the recorded example to see its measured sizes.'}</p>}
          <p className="tiny muted" style={{ padding: '16px 20px', borderTop: '1px solid var(--line)' }}>Millimetres. One test part; results depend on light, video and how flat the mat lies. <button className="link tiny" onClick={() => go('about')}>Method and numbers</button></p></div>
      </div></section>

      {(media.clip || media.photo) && <section data-testid="proof"><div className="wrap">
        <div className="sec-head"><span className="label">Proof on the bench</span><h2>Printed. Fitted. Working.</h2><p className="lead">Part B was filmed, built, printed and fitted to the briefcase lock it came from. Light sanding, and the mechanism works.</p></div>
        <div className="proof" style={media.clip && media.photo ? undefined : { gridTemplateColumns: '1fr' }}>
          {media.clip && <video className="ph" style={{ padding: 0, width: '100%', objectFit: 'cover' }} src="/media/print_fit.mp4" muted loop autoPlay={!matchMedia('(prefers-reduced-motion: reduce)').matches} playsInline controls aria-label="Print-fit clip: the copy on the real lock" />}
          {media.photo && <img className="ph" style={{ padding: 0, width: '100%', objectFit: 'cover' }} src="/media/original_and_copy.jpg" alt="The original part and the printed copy, with calipers" />}
        </div>
      </div></section>}

      <section style={{ paddingTop: 0 }}><div className="wrap"><div className="cta-band"><h2>The part you can't buy,<br />you can now print.</h2><p className="lead" style={{ margin: '18px auto 0' }}>Film three short videos, then let it build.{buildS != null ? ` The recorded run built in ${fmtTime(buildS)} (${lbl}).` : ''} No account.</p><div className="hero-cta"><button className="btn btn-primary" onClick={() => go('prepare')}>Film a part →</button><button className="btn btn-ghost" onClick={() => go('resume')}>Left somewhere?</button></div></div></div></section>
    </>
  )
}
