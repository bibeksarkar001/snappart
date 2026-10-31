import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, onMode, type Mode, type ModelState, type Recorded } from './api'
import { Ctx, FLOW, mediaExists, ROUTES, type App as AppT } from './ctx'
import * as idb from './idb'
import { MOON, SUN } from './illus'
import { Home } from './Home'
import { Build, Capture, Prepare, Resume } from './Flow'
import { Review } from './Review'
import { Result } from './Result'
import { About, Next, Privacy, Terms } from './Docs'

const FN: Record<string, [string, string]> = { prepare: ['Prepare', 'about 2 min'], capture: ['Capture', '5 to 8 min'], build: ['Build', 'about 3 min'], review: ['Review', '1 to 3 min'], result: ['Result', ''] }
const PAGES: Record<string, () => React.JSX.Element> = { home: Home, prepare: Prepare, capture: Capture, build: Build, review: Review, result: Result, resume: Resume, about: About, next: Next, privacy: Privacy, terms: Terms }

const hashRoute = () => { const r = location.hash.slice(1); return ROUTES.includes(r) ? r : 'home' }
const isLight = () => { const t = document.documentElement.getAttribute('data-theme'); return t ? t === 'light' : matchMedia('(prefers-color-scheme: light)').matches }

export default function App() {
  const [route, setRoute] = useState(hashRoute)
  const [draft, setDraft] = useState<idb.Draft | null>(null)
  const [mode, setMode] = useState<Mode | null>(null)
  const [recorded, setRecorded] = useState<string | null>(null), [recording, setRecording] = useState<Recorded | null>(null)
  const [demo, setDemo] = useState(false)
  const [models, setModels] = useState<ModelState[]>([])
  const [toastT, setToastT] = useState(''); const tt = useRef(0)
  const [online, setOnline] = useState(navigator.onLine)
  const [pending, setPending] = useState(0)
  const [light, setLight] = useState(isLight)
  const [menu, setMenu] = useState(false)
  const draftRef = useRef<idb.Draft | null>(null)

  const toast = useCallback((t: string) => { setToastT(t); clearTimeout(tt.current); tt.current = window.setTimeout(() => setToastT(''), 2600) }, [])
  const go = useCallback((r: string, anchor?: string) => {
    if (!ROUTES.includes(r)) r = 'home'
    if (location.hash.slice(1) !== r) history.replaceState(null, '', '#' + r)
    setRoute(r); setMenu(false)
    requestAnimationFrame(() => { const el = anchor && document.getElementById(anchor); if (el) el.scrollIntoView({ behavior: 'smooth' }); else window.scrollTo(0, 0) })
  }, [])

  const save = useCallback(async (p: Partial<idb.Draft>) => {
    const cur = draftRef.current
    if (!cur) return
    const d = { ...cur, ...p }
    draftRef.current = d; setDraft(d)
    try { await idb.putDraft(d) } catch { /* private mode: the session still works, it just won't survive a reload */ }
  }, [])
  const open = useCallback(async (d: idb.Draft) => {
    draftRef.current = d; setDraft(d)
    try { await idb.putDraft(d); await idb.setCurrent(d.jobId) } catch { /* see above */ }
  }, [])
  const forget = useCallback(async () => {
    const id = draftRef.current?.jobId
    draftRef.current = null; setDraft(null)
    try { if (id) await idb.dropJob(id); await idb.setCurrent(null) } catch { /* see above */ }
  }, [])

  const jobId = draft?.jobId
  const refreshModels = useCallback(() => {
    api.models(jobId).then(m => { setModels(m.models); setMode(m.mode); setRecording(m.recorded ?? null); if (m.recorded) setRecorded(m.recorded.label) })
      .catch(() => api.models().then(m => { setModels(m.models); setMode(m.mode) }).catch(() => {}))
  }, [jobId])

  useEffect(() => {
    const h = () => setRoute(hashRoute())
    const on = () => setOnline(true), off = () => setOnline(false)
    window.addEventListener('hashchange', h); window.addEventListener('online', on); window.addEventListener('offline', off)
    const un = onMode(m => { setMode(m.mode); if (m.recorded) setRecorded(m.recorded) })
    idb.currentId().then(id => (id ? idb.getDraft(id) : undefined)).then(d => { if (d) { draftRef.current = d; setDraft(d) } }).catch(() => {})
    const mq = matchMedia('(prefers-color-scheme: light)'), th = () => setLight(isLight())
    mq.addEventListener('change', th)
    return () => { window.removeEventListener('hashchange', h); window.removeEventListener('online', on); window.removeEventListener('offline', off); un(); mq.removeEventListener('change', th) }
  }, [])
  useEffect(refreshModels, [refreshModels])
  useEffect(() => { void mediaExists('/media/demo.mp4', 'video/').then(setDemo) }, [])
  useEffect(() => { if (draft && FLOW.includes(route) && route !== 'prepare' && draft.step !== route) void save({ step: route }) }, [route, draft, save])

  const theme = () => {
    const t = isLight() ? 'dark' : 'light'
    try { localStorage.setItem('sp-theme', t) } catch { /* no storage */ }
    document.documentElement.setAttribute('data-theme', t); setLight(t === 'light')
  }

  const f = FLOW.includes(route), i = FLOW.indexOf(route)
  const app: AppT = useMemo(() => ({ route, go, draft, save, open, forget, mode, recorded, recording, models, refreshModels, toast, online, pending, setPending }),
    [route, go, draft, save, open, forget, mode, recorded, recording, models, refreshModels, toast, online, pending])
  const Page = PAGES[route]
  const nav = (
    <>
      <a onClick={() => go('home', 'how')}>How it works</a><a onClick={() => go('home', 'meta')}>Meta AI</a>
      <a onClick={() => go('about')}>About</a><a onClick={() => go('next')}>What's next</a><a onClick={() => go('resume')}>Left somewhere?</a>
    </>
  )

  return (
    <Ctx.Provider value={app}>
      <header className="hdr">
        <div className="wrap hrow">
          <button className="logo" onClick={() => go('home')} aria-label="SnapPart home">
            <svg width="26" height="26" viewBox="0 0 26 26" aria-hidden="true"><rect x=".5" y=".5" width="25" height="25" rx="7" fill="var(--panel)" stroke="var(--line-2)" /><rect x="6" y="6" width="5" height="5" rx="1" fill="var(--ink)" /><rect x="15" y="6" width="5" height="5" rx="1" fill="var(--ink)" /><rect x="6" y="15" width="5" height="5" rx="1" fill="var(--ink)" /><rect x="15" y="15" width="5" height="5" rx="1" fill="var(--accent)" /></svg>SnapPart
          </button>
          <nav className="nav" aria-label="Main">{nav}</nav>
          {f && draft && <button className="codechip" onClick={() => go('resume')} title="Your resume code">{draft.resumeCode}</button>}
          <button className="themebtn" onClick={theme} aria-label={light ? 'Switch to dark mode' : 'Switch to light mode'} title="Light or dark">
            <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true" dangerouslySetInnerHTML={{ __html: light ? MOON : SUN }} />
          </button>
          {!f && <button className="btn btn-primary btn-sm" onClick={() => go('prepare')}>Film a part</button>}
          <button className="menu-btn" onClick={() => setMenu(!menu)} aria-label="Open menu" aria-expanded={menu}><svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true"><path d="M2 5h14M2 9h14M2 13h14" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" /></svg></button>
        </div>
        <div className="wrap"><nav className={'mnav' + (menu ? ' on' : '')}>{nav}</nav></div>
        <ModeBar mode={mode} recorded={recorded} />
        <div className={'offline' + (online ? '' : ' on')}><div className="wrap">You're offline. Your progress is saved on this device{pending ? `, and ${pending} upload${pending > 1 ? 's' : ''} will resume when you reconnect` : ''}.</div></div>
        {f && (
          <div className="path"><div className="wrap">
            <ol>{FLOW.map((x, k) => (
              <li key={x} className={k < i ? 'done' : k === i ? 'now' : ''} onClick={() => go(x)} onKeyDown={e => e.key === 'Enter' && go(x)} tabIndex={0} role="link">
                <span className="bar" /><span className="nm">{k < i ? '✓ ' : ''}{FN[x][0]}</span><span className="tm">{FN[x][1]}</span>
              </li>))}</ol>
            <div className="path-m"><div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 14, marginBottom: 8 }}><b>{FN[route][0]}</b><span className="mono muted tiny">Step {i + 1} of 5</span></div><div className="prog"><i style={{ width: `${(i + 1) * 20}%` }} /></div></div>
          </div></div>
        )}
      </header>
      <main><div className="page on" key={route}><Page /></div></main>
      <footer className="foot"><div className="wrap"><div className="cols">
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}><b style={{ color: 'var(--ink)', fontSize: 16 }}>SnapPart</b><span>Film a part. Print a copy.</span><span>Made for the <a href="https://cerebralvalley.ai/e/meta-ai-developer-hackathon" target="_blank" rel="noopener" style={{ display: 'inline', textDecoration: 'underline' }}>Meta Global AI Developer Hackathon</a>. Not made or endorsed by Meta.</span></div>
        <div><a onClick={() => go('about')}>About</a><a onClick={() => go('next')}>What's next</a><a onClick={() => go('resume')}>Left somewhere?</a><a onClick={() => go('privacy')}>Privacy</a><a onClick={() => go('terms')}>Terms</a></div>
        <div><a href="https://github.com/bibeksarkar001/snappart" target="_blank" rel="noopener">Repository</a><a href="https://github.com/bibeksarkar001/snappart#readme" target="_blank" rel="noopener">README</a><a href="https://github.com/bibeksarkar001" target="_blank" rel="noopener">Bibek Sarkar on GitHub</a>{demo && <a href="/media/demo.mp4" target="_blank" rel="noopener">Demo video</a>}</div>
      </div></div></footer>
      <div className={'toast' + (toastT ? ' on' : '')} role="status" aria-live="polite">{toastT}</div>
    </Ctx.Provider>
  )
}

function ModeBar({ mode, recorded }: { mode: Mode | null; recorded: string | null }) {
  if (!mode) return null
  const text = mode === 'replay'
    ? <>Replay · {recorded ?? 'recorded run'}. You are seeing a recorded real run of part B. Nothing you upload is processed and no Meta model is called now.</>
    : mode === 'mock'
      ? <>Mock mode · canned results of a synthetic part. No upload is processed and no Meta model is called.</>
      : <>Live · your footage is processed and Meta models are called.</>
  return <div className={'modebar ' + mode} data-testid="modebar"><div className="wrap"><span className="mono tiny" style={{ textTransform: 'uppercase', letterSpacing: '.06em' }}>{mode === 'real' ? 'live' : mode}</span><span>{text}</span></div></div>
}
