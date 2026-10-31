import { useCallback, useEffect, useRef, useState } from 'react'
import { abs, api, ApiError, recordedAnswers, type Change, type Proposal, type Question, type Questions, type Rebuild, type ReviewItem, type Voice } from './api'
import { errText, pendingQuestion, useApp } from './ctx'
import { NoPart, RecChip } from './Flow'
import { record, type Recorder } from './media'
import { Viewer, type Pick } from './Viewer'

const editErr = (e: unknown) => (e instanceof ApiError && e.code === 'replay_read_only' ? 'Read-only in a recorded run.' : errText(e))
const fmt = (v: unknown): string => (typeof v === 'number' ? `${+v.toFixed(3)}` : Array.isArray(v) ? v.map(fmt).join(', ') : v == null ? '—' : String(v))

export function Diff({ r }: { r: Rebuild }) {
  return (
    <div className="understood" data-testid="diff"><b>Rebuilt in {r.seconds} s{r.watertight ? ', watertight' : ', NOT watertight'}{r.quality?.status === 'draft' ? ' · still a draft' : ''}</b>
      {r.changes.length ? <div className="diff">{r.changes.map((c: Change) => <div key={c.id}><span>{c.id}</span><span className="from">{fmt(c.from)}</span>→<span className="to">{fmt(c.to)}</span>{c.status_to && <span className={'chip ' + (c.status_to === 'user_supplied' ? 'c-you' : c.status_to === 'measured' ? 'c-meas' : 'c-est')}>{c.status_to.replace('_', ' ')}</span>}</div>)}</div>
        : <span className="small dim">No size changed.</span>}
      {r.quality?.reasons?.map((x, k) => <span key={k} className="small" style={{ color: 'var(--ask)' }}>{x}</span>)}
    </div>
  )
}

export function Review() {
  const app = useApp(), { draft } = app
  const [q, setQ] = useState<Questions | null>(null), [loadErr, setLoadErr] = useState('')
  const [report, setReport] = useState<Record<string, any> | null>(null)
  const [active, setActive] = useState<string | null>(() => { const id = pendingQuestion.id; pendingQuestion.id = null; return id })
  const [marking, setMarking] = useState(false), [picked, setPicked] = useState<Pick | null>(null), [markErr, setMarkErr] = useState('')
  const [errs, setErrs] = useState<Record<string, string>>({}), [busy, setBusy] = useState<string | null>(null)
  const [last, setLast] = useState<{ id: string; r: Rebuild } | null>(null)
  const [ver, setVer] = useState(0), [focusItem, setFocusItem] = useState<string | null>(null)
  const jobId = draft?.jobId

  const load = useCallback(() => {
    if (!jobId) return
    api.questions(jobId).then(setQ).catch(e => setLoadErr(errText(e)))
    api.report(jobId).then(setReport).catch(() => {})
  }, [jobId])
  useEffect(load, [load])

  if (!draft) return <NoPart />
  const jid = draft.jobId
  // Every change goes the same way: save it, rebuild (no AI call), show what moved, reload the model and the questions.
  const apply = async (key: string, f: () => Promise<unknown>) => {
    setBusy(key); setErrs(e => ({ ...e, [key]: '' }))
    try {
      await f()
      const r = await api.rebuild(jid)
      setLast({ id: key, r }); setVer(v => v + 1); app.toast(`Model rebuilt in ${r.seconds} s.`); load(); setActive(null)
    } catch (e) { setErrs(x => ({ ...x, [key]: editErr(e) })) } finally { setBusy(null) }
  }
  // Mark something: the clicked spot becomes review item Uxx, the choice is recorded on it, then the model is rebuilt.
  const sendMark = async (choice: string) => {
    if (!picked) return
    setBusy('mark'); setMarkErr('')
    try {
      const it = await api.mark(jid, picked.point, picked.normal)
      await api.review(jid, it.id, choice)
      const r = await api.rebuild(jid)
      setLast({ id: 'mark', r }); setVer(v => v + 1); app.toast(`Model rebuilt in ${r.seconds} s.`); load()
    } catch (e) { setMarkErr(editErr(e)) } finally { setBusy(null) }
  }
  const qs = q?.questions ?? []
  const recAns = recordedAnswers(q)
  const open = qs.filter(x => !x.answered)
  const cur = active ?? open[0]?.id ?? null
  const feats: { status: string }[] = report?.features ?? []
  const measured = feats.filter(f => f.status === 'measured').length
  const you = qs.filter(x => x.status === 'user_supplied').length, est = qs.filter(x => x.status === 'estimated').length
  const curQ = qs.find(x => x.id === cur)
  const toResult = async () => {
    if (q && q.open_questions === 0) await api.accept(jid).catch(e => app.toast(errText(e)))
    app.go('result')
  }

  return (
    <div className="wrap flow"><div className="rev">
      <div className="revstage"><Viewer src={`/v1/jobs/${jid}/files/preview.glb?v=${ver}`} style={{ minHeight: 520 }} pick={marking} onPick={p => { setPicked(p); setMarkErr(''); setLast(l => (l?.id === 'mark' ? null : l)) }} mark={marking ? picked?.point : null}
        dragExtra={<> · <button className="link tiny" style={{ pointerEvents: 'auto' }} onClick={() => { setMarking(!marking); setPicked(null); setMarkErr('') }}>{marking ? 'Stop marking' : 'Mark something'}</button></>}>
        <div className="hud" style={{ left: 16, top: 16 }}><span className="k">AMBER</span><span className="v" style={{ color: 'var(--ask)' }}>{curQ ? curQ.label.toLowerCase() : 'all answered'}</span></div></Viewer>
        {marking && <div className="q active" data-testid="mark" style={{ marginTop: 12 }}><div className="who"><span className="chip c-ask">Your mark</span>{picked ? 'spot marked on the model' : 'click a point on the model'}</div>
          {picked && <><h3>What should happen to the spot you marked?</h3><div className="ans">{['Keep', 'Remove', 'Make flat'].map(c => <button key={c} className="choice" onClick={() => sendMark(c)} disabled={!!busy}>{c}</button>)}</div>
            <p className="tiny muted">A 3 mm circle around the point. Remove and Make flat need a spot that faces up or down.</p></>}
          <div className="err">{markErr}</div>{last?.id === 'mark' && <Diff r={last.r} />}</div>}
        <p className="tiny muted" style={{ marginTop: 8 }}>The preview model. Highlighting the asked-about area on it needs region data the API does not send yet; each question's picture shows the spot.</p></div>
      <div className="qs">
        <div><span className="label" style={{ color: 'var(--ask)' }}>Review · {q ? (open.length ? `${qs.length - open.length + 1} of ${qs.length}` : 'all answered') : '…'}</span> <RecChip />
          <h2 style={{ marginTop: 10 }}>A few questions before it prints.</h2>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 14 }}><span className="chip c-meas">{measured} measured</span><span className="chip c-you">{you} you answered</span><span className="chip c-est">{est} estimated</span><span className="chip c-ask">{q?.open_questions ?? '…'} waiting</span></div></div>
        {loadErr && <div className="err">{loadErr}</div>}
        <div className="qs" data-testid="questions">{qs.map(x => <QCard key={x.id} x={x} jobId={jid} rec={app.mode === 'replay' ? recAns[x.id] : undefined} active={x.id === cur} busy={busy === x.id} err={errs[x.id]} diff={last?.id === x.id ? last.r : null}
          activate={() => setActive(x.id)} answer={v => apply(x.id, () => api.answers(jid, { [x.id]: v }))} />)}</div>
        {(q?.review_items ?? []).map(it => <ItemCard key={it.id} it={it} focused={focusItem === it.id} busy={busy === it.id} err={errs[it.id]} diff={last?.id === it.id ? last.r : null}
          focus={() => setFocusItem(focusItem === it.id ? null : it.id)} choose={c => apply(it.id, () => api.review(jid, it.id, c))} />)}
        {(q?.coverage_requests ?? []).filter(c => !c.decision).map(c => <div className="q" key={c.id}><div className="who"><span className="chip c-ask">Barely seen</span>{c.blocking ? 'needed before finishing' : 'optional'}</div><p className="small">{c.message}</p>
          <div className="ans"><button className="choice" disabled={!!busy} onClick={async () => { try { const r = await api.moreFootage(jid, c.id, 'film'); await app.save({ frames: {}, frameOffset: r.next_frame_index }); app.go('capture') } catch (e) { setErrs(x => ({ ...x, [c.id]: errText(e) })) } }}>Film more</button>
            <button className="choice" disabled={!!busy} onClick={() => api.moreFootage(jid, c.id, 'skip').then(load).catch(e => setErrs(x => ({ ...x, [c.id]: errText(e) })))}>Skip this area</button></div><div className="err">{errs[c.id]}</div></div>)}
        <Composer jobId={jid} itemId={focusItem} onApplied={(r) => { setLast({ id: 'free', r }); setVer(v => v + 1); load() }} />
        {last?.id === 'free' && <Diff r={last.r} />}
        <button className="btn btn-primary btn-block pcta" onClick={toResult}>{q && !open.length ? 'See my part →' : 'Skip the rest and see my part →'}</button>
      </div></div></div>
  )
}

function Picture({ x, jobId }: { x: Question; jobId: string }) {
  // card is null when the photo is missing (replay withholds it); the diagram may still exist under the documented name.
  const base = `/v1/jobs/${jobId}/files/cards/${x.id}/`
  const card = x.card ?? { image: base + 'card.png', diagram: base + 'diagram.svg', meta: base + 'card.json' }
  const [src, setSrc] = useState<string | null>(abs(card.image))
  const [lbl, setLbl] = useState('')
  useEffect(() => {
    fetch(abs(card.meta)).then(r => (r.ok ? r.json() : null)).then(m => m && setLbl(`video ${m.video} · frame ${String(m.frame_id).split('_').pop()}${m.fallback_frame ? ' · nearest frame' : ''}`)).catch(() => {})
  }, [card.meta])
  if (!src) return <p className="tiny muted">No picture was made for this question.</p>
  const diagram = src === abs(card.diagram)
  return <div className="cardimg"><img src={src} alt={`Where: ${x.label}`} onError={() => setSrc(diagram ? null : abs(card.diagram))} /><span className="lbl">{diagram ? `diagram${lbl ? ` · photo (${lbl}) not available` : ''}` : lbl || 'frame with the area marked'}</span></div>
}

function QCard({ x, jobId, rec, active, busy, err, diff, activate, answer }: { x: Question; jobId: string; rec?: { value: number | string; unit: string }; active: boolean; busy: boolean; err?: string; diff: Rebuild | null; activate: () => void; answer: (v: number | string) => void }) {
  const [val, setVal] = useState(''), [localErr, setLocalErr] = useState('')
  const w = x.friendly && x.friendly.source !== 'fallback' ? x.friendly : x
  const head = <div className="who">{x.friendly && x.friendly.source !== 'fallback' && <span className="chip c-meta">{x.friendly.source}{x.friendly.recorded ? ` · ${x.friendly.recorded}` : ''}</span>}<span className="chip c-ask">{x.label}</span>{x.impact ? `impact ${x.impact}` : null}{x.applies_to && x.applies_to > 1 ? ` · one answer for ${x.applies_to}` : ''}</div>
  if (x.answered && !active) return (
    <div className="q answered">{head}<h3>{w.title}</h3><div className="ans"><span className={'chip ' + (x.status === 'estimated' ? 'c-est' : 'c-you')}>{x.status === 'estimated' ? `Estimated · ${fmt(x.default)} ${x.unit ?? ''}` : `${fmt(x.answer)} ${typeof x.answer === 'number' ? x.unit ?? '' : ''}`}</span><button className="link small" onClick={activate}>Change</button></div>{diff && <Diff r={diff} />}</div>)
  if (!active) return <div className="q wait" onClick={activate} style={{ cursor: 'pointer' }} role="button" tabIndex={0} onKeyDown={e => e.key === 'Enter' && activate()}><div className="who"><span className="chip c-ask">{x.label}</span>waiting · tap to answer</div><b>{w.title}</b></div>
  const num = () => {
    const v = parseFloat(val.replace(',', '.'))
    if (!val.trim() || isNaN(v)) { setLocalErr(`Enter a number in ${x.unit ?? 'mm'} first.`); return }
    if ((x.min != null && v < x.min) || (x.max != null && v > x.max)) { setLocalErr(`${v} ${x.unit ?? 'mm'} doesn't fit this part (${x.min} to ${x.max} ${x.unit ?? 'mm'}).`); return }
    setLocalErr(''); answer(v)
  }
  const numInput = <><input className="input mono" inputMode="decimal" placeholder={fmt(x.default)} aria-label={w.title} value={val} onChange={e => { setVal(e.target.value); setLocalErr('') }} onKeyDown={e => e.key === 'Enter' && num()} /><span className="muted">{x.unit ?? 'mm'}</span></>
  return (
    <div className="q active" data-testid={'q-' + x.id}>{head}<h3>{w.title}</h3>
      <Picture x={x} jobId={jobId} />
      <p className="small dim">{x.kind === 'conflict' && x.conflict_message ? x.conflict_message : w.detail}</p>
      <p className="small" style={{ display: 'flex', gap: 8 }}><span aria-hidden="true">📏</span>{w.hint}</p>
      {rec && <div className="understood" data-testid="recorded-answer"><span>Answer recorded from the owner's calipers: <b className="mono">{fmt(rec.value)}{rec.unit ? ` ${rec.unit}` : ''}</b></span>
        <div><button className="btn btn-ghost btn-sm" onClick={() => answer(rec.value)} disabled={busy}>Use this answer</button></div></div>}
      {x.kind === 'number' && <><div className="ans">{numInput}<button className="btn btn-primary btn-sm" onClick={num} disabled={busy}>{busy ? 'Rebuilding…' : 'Update model'}</button><button className="choice" onClick={() => answer('skip')} disabled={busy}>I don't know</button></div>
        <p className="tiny muted">Between {x.min} and {x.max} {x.unit}. "I don't know" keeps {fmt(x.default)} {x.unit}, labelled estimated.</p></>}
      {x.kind === 'choice' && <div className="ans">{(x.choices ?? []).map(c => <button key={c} className="choice" aria-pressed={x.answer === c} onClick={() => answer(c)} disabled={busy}>{c}</button>)}<button className="choice" onClick={() => answer('skip')} disabled={busy}>I don't know</button></div>}
      {x.kind === 'conflict' && <><div className="ans">{(x.candidates ?? []).map((c, k) => { const v = typeof c === 'object' ? c.value : c; const l = typeof c === 'object' && c.label ? c.label : `${fmt(v)} ${x.unit ?? ''} is right`; return <button key={k} className="choice" onClick={() => answer(v)} disabled={busy}>{l}</button> })}</div>
        <div className="ans">{numInput}<button className="btn btn-ghost btn-sm" onClick={num} disabled={busy}>Use this</button><button className="choice" onClick={() => answer('skip')} disabled={busy}>I don't know</button></div></>}
      <div className="err">{localErr || err}</div>
      {diff && <Diff r={diff} />}
    </div>
  )
}

function ItemCard({ it, focused, busy, err, diff, focus, choose }: { it: ReviewItem; focused: boolean; busy: boolean; err?: string; diff: Rebuild | null; focus: () => void; choose: (c: string) => void }) {
  return (
    <div className={'q' + (focused ? ' active' : '')} data-testid={'item-' + it.id}><div className="who"><span className="chip c-ask">{it.id.startsWith('D') ? 'Small detail' : it.id.startsWith('U') ? 'Your mark' : 'Area to review'}</span>{it.decision ? `you chose ${it.decision}` : 'waiting'}<button className="link tiny" onClick={focus}>{focused ? 'Unselect' : 'Talk about this area'}</button></div>
      <h3>{it.question}</h3>
      {it.image ? <div className="cardimg"><img src={abs(it.image)} alt={`Area ${it.id}`} onError={e => ((e.target as HTMLImageElement).parentElement!.style.display = 'none')} /><span className="lbl">area {it.id}</span></div> : null}
      <p className="small dim">{it.reason}</p>
      <div className="ans">{it.choices.map(c => <button key={c} className="choice" aria-pressed={it.decision === c} onClick={() => choose(c)} disabled={busy}>{c}</button>)}</div>
      <div className="err">{err}</div>{diff && <Diff r={diff} />}
    </div>
  )
}

function Composer({ jobId, itemId, onApplied }: { jobId: string; itemId: string | null; onApplied: (r: Rebuild) => void }) {
  const app = useApp()
  const [text, setText] = useState(''), [err, setErr] = useState(''), [busy, setBusy] = useState('')
  const [prop, setProp] = useState<Proposal | null>(null), [heard, setHeard] = useState<Voice | null>(null)
  const [rec, setRec] = useState(false), recorder = useRef<Recorder | null>(null)
  useEffect(() => () => recorder.current?.cancel(), [])
  const send = async () => {
    if (!text.trim()) { setErr('Type or say what is wrong first.'); return }
    setBusy('Thinking…'); setErr(''); setHeard(null)
    try { setProp(await api.propose(jobId, text.trim(), itemId ?? undefined)) } catch (e) { setErr(editErr(e)); setProp(null) } finally { setBusy('') }
  }
  const mic = async () => {
    if (rec) {
      setRec(false); setBusy('Transcribing…'); setErr('')
      try {
        const wav = await recorder.current!.stop(); recorder.current = null
        const v = await api.voice(jobId, wav, itemId ?? undefined)
        setHeard(v); setText(v.transcript); setProp(v.proposal)
      } catch (e) { setErr(editErr(e)); setProp(null) } finally { setBusy('') }
      return
    }
    try { recorder.current = await record(); setRec(true); setErr('') } catch { setErr('The microphone is not available. Allow it in the browser, or type instead.') }
  }
  const confirm = async () => {
    if (!prop?.correction) return
    setBusy('Applying…')
    try {
      await api.correction(jobId, prop.correction)
      const r = await api.rebuild(jobId)
      setProp(null); setHeard(null); setText(''); app.toast(`Applied. Model rebuilt in ${r.seconds} s.`); onApplied(r)
    } catch (e) { setErr(editErr(e)) } finally { setBusy('') }
  }
  return (
    <div className="q"><div className="who"><b style={{ color: 'var(--ink)' }}>Tell SnapPart something</b><span>· voice or text{itemId ? ` · about area ${itemId}` : ''}</span></div>
      <div className="composer"><button className={'ib voice' + (rec ? ' mic-on' : '')} aria-label={rec ? 'Stop recording' : 'Speak'} aria-pressed={rec} onClick={mic} disabled={!!busy}>{rec ? '■' : '🎙'}</button>
        <textarea rows={1} placeholder="e.g. this strip is flat like the bottom" value={text} onChange={e => { setText(e.target.value); setErr('') }} aria-label="Tell SnapPart something" />
        <button className="ib fill" aria-label="Send" onClick={send} disabled={!!busy || rec}>↑</button></div>
      <p className="tiny muted">{rec ? 'Recording… tap ■ to stop. ' : ''}Spoken with Muse Voice Transcribe. The audio is deleted right after it becomes text; nothing changes until you confirm.</p>
      {busy && <span className="small dim">{busy}</span>}
      <div className="err">{err}</div>
      {heard && <div className="note"><span className="chip c-meta">{heard.source}</span>Heard ({heard.audio_seconds} s, audio not kept): "{heard.transcript}"</div>}
      {prop && <div className="understood" data-testid="proposal"><span className="tiny" style={{ color: 'var(--meta)' }}>{prop.source ?? 'proposal'}</span>
        <b>{prop.correction ? 'Understood: ' : ''}{prop.explanation}</b>
        {prop.correction && <span className="small dim mono">{String(prop.correction.op).replace(/_/g, ' ')}{prop.action !== 'none' ? ` · ${prop.action.replace(/_/g, ' ')}` : ''}</span>}
        <div style={{ display: 'flex', gap: 8 }}>{prop.correction && <button className="btn btn-primary btn-sm" onClick={confirm} disabled={!!busy}>Confirm</button>}<button className="btn btn-ghost btn-sm" onClick={() => { setProp(null); setHeard(null) }}>Cancel</button></div></div>}
    </div>
  )
}

