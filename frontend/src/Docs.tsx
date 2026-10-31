import { useApp } from './ctx'

const ext = { target: '_blank', rel: 'noopener' }

export function About() {
  const { go } = useApp()
  return (
    <div className="wrap flow"><div className="doc">
      <span className="label">About</span><h1 style={{ fontSize: 'clamp(42px,6vw,72px)', marginTop: 16 }}>Film a part.<br /><span className="grad">Print a copy.</span></h1>
      <p className="lead" style={{ marginTop: 22 }}>A part you can't buy shouldn't be the end of the thing it belongs to. A cracked latch on a toolbox, a snapped bracket under a favourite toy car, a hinge nobody sells anymore. SnapPart lets you film the piece on a printed mat and get a measured 3D model you can print.</p>
      <h3>How it works</h3><p><b style={{ color: 'var(--ink)' }}>Film.</b> Print the mat, tape it flat, film a high orbit, a low orbit and the part flipped. <b style={{ color: 'var(--ink)' }}>Build.</b> The codes give each frame its camera position and scale, SAM 3.1 outlines the part in every frame, and the outlines carve the shape. <b style={{ color: 'var(--ink)' }}>Review and print.</b> Muse Spark asks for what a camera can't see; answer by typing, tapping or speaking with Muse Voice, then download the STL.</p>
      <h3>Honest about accuracy</h3><p>On our test part (51.6 × 41.9 × 11.8 mm by calipers), overall sizes came within 0.1 to 0.26 mm and the four slots measured 7.43 × 2.54 mm against 7.4 × 2.6 mm. The held-out score of 0.95 says how well the model matches 10 frames it was never built from. Your results depend on light, video and how flat the mat lies. <a className="link" href="https://github.com/bibeksarkar001/snappart/blob/main/docs/accuracy.md" {...ext}>Method and numbers</a></p>
      <h3>Built with open source</h3><p>OpenCV, NumPy, SciPy, scikit-image, Open3D, trimesh, FastAPI, React and three.js, among others. <a className="link" href="https://github.com/bibeksarkar001/snappart#readme" {...ext}>All, with licences, in the README</a></p>
      <h3>Credits</h3><div className="dev"><div className="avatar" aria-hidden="true">BS</div><div style={{ flex: 1, minWidth: 180 }}><b style={{ fontSize: 17 }}>Bibek Sarkar</b><div className="mono tiny muted">github.com/bibeksarkar001</div></div><a className="btn btn-ghost btn-sm" href="https://github.com/bibeksarkar001" {...ext}><svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M8 0a8 8 0 0 0-2.53 15.59c.4.07.55-.17.55-.38v-1.34c-2.23.48-2.7-1.07-2.7-1.07-.36-.92-.89-1.17-.89-1.17-.73-.5.06-.49.06-.49.8.06 1.23.83 1.23.83.72 1.22 1.88.87 2.33.66.07-.52.28-.87.5-1.07-1.78-.2-3.65-.89-3.65-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.6 7.6 0 0 1 4 0c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.28.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48v2.2c0 .21.15.46.55.38A8 8 0 0 0 8 0z" /></svg>View GitHub profile</a></div>
      <p style={{ marginTop: 14 }}>Made by <b style={{ color: 'var(--ink)' }}>Bibek Sarkar</b>, with help from <b style={{ color: 'var(--ink)' }}>Bhashwati Roy</b>. Built during the Meta Global AI Developer Hackathon, Oct 27 to Nov 4, 2026. The commit history is in the repository.</p>
      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginTop: 22 }}><a className="btn btn-ghost btn-sm" href="https://github.com/bibeksarkar001/snappart" {...ext}>Repository</a><a className="btn btn-ghost btn-sm" href="https://github.com/bibeksarkar001/snappart#readme" {...ext}>README</a><button className="btn btn-ghost btn-sm" onClick={() => go('next')}>What's next</button><button className="btn btn-ghost btn-sm" onClick={() => go('privacy')}>Privacy</button><button className="btn btn-ghost btn-sm" onClick={() => go('terms')}>Terms</button></div>
    </div></div>
  )
}

const RD: [string, string, [string, string, string][]][] = [
  ['Now', 'being built', [['In progress', 'Film a part, get a measured STL', 'The pipeline works on a test part; the app is being built.'], ['In progress', '"Show me where" questions', "Muse Spark asks about the exact spot it can't measure."], ['In progress', 'Review your model', 'Flags anything unusual: keep, remove or make flat.'], ['In progress', 'Answer by voice', 'Muse Voice Transcribe. You confirm every change.'], ['In progress', 'Left somewhere? resume', 'A code and link back to the exact step.']]],
  ['Next', 'after the core works', [['Planned', 'Small-code mat', 'For parts under about 25 mm.'], ['Planned', 'Snap to standard sizes', 'Recognises an M4 screw or 608 bearing and offers published sizes, with a source.'], ['Planned', 'Clearance profiles', 'Ready-made fits for hinges, latches and snap fits.'], ['Planned', 'Snap your caliper', 'Photograph a reading; SnapPart reads the number.']]],
  ['Later', 'ideas', [['Idea', 'Enclosure builder', 'Film your parts, describe the project, print a case that fits them.'], ['Idea', 'Automatic re-carving', 'Ask for one more short video when the score is low.'], ['Idea', 'Share repaired parts', 'Only parts you own the rights to, with consent and a way to remove.']]],
]

export function Next() {
  return (
    <div className="wrap flow"><div className="sec-head"><span className="label">What's next</span><h2>Where SnapPart is going.</h2><p className="lead">Nothing below is available until it's marked Built. No dates yet.</p><div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}><span className="chip c-meas">Built</span><span className="chip c-you">In progress</span><span className="chip c-est">Planned</span><span className="chip c-unseen">Idea</span></div></div>
      <div className="road">{RD.map(c => <div className="col" key={c[0]}><h3>{c[0]} <span className="small muted" style={{ fontWeight: 400 }}>{c[1]}</span></h3>{c[2].map(i => (
        <div className={'ri' + (i[0] === 'Idea' ? ' idea' : '')} key={i[1]}><span className={'chip ' + (i[0] === 'In progress' ? 'c-you' : i[0] === 'Planned' ? 'c-est' : 'c-unseen')} style={{ width: 'max-content' }}>{i[0]}</span><b>{i[1]}</b><span className="small dim">{i[2]}</span></div>))}</div>)}</div></div>
  )
}

export function Privacy() {
  return (
    <div className="wrap flow"><div className="doc"><span className="label">Privacy policy</span><h2 style={{ marginTop: 12 }}>Your part, your files.</h2><p className="small muted">Last updated <span className="phl">[date]</span></p>
      <h3>What we collect</h3><ul><li>Frames your browser pulls from your videos (not the full videos).</li><li>Voice recordings, only if you speak, only to make text.</li><li>What you type: part name, description, answers, corrections.</li><li>What we make: model, report, question cards, review images.</li><li>IP address and request times, only to prevent abuse.</li></ul><p>No name, email or account.</p>
      <h3>How it's processed</h3><p>Frames, text and voice go to Meta's Model API (SAM 3.1, Muse Spark 1.3, Muse Voice Transcribe) on the standard endpoints, not the "contributor" ones under which content may be used for training. Site on Vercel, build server on <span className="phl">[backend host]</span>. We never sell data or use it for ads.</p>
      <h3>How long we keep it</h3><p>Voice: deleted right after transcription. Frames, model, report and answers: 7 days, then deleted. Delete your part anytime from its Result page.</p>
      <h3>Who can see your part</h3><p>Anyone with your private link or code. We don't list or publish parts. Contact <span className="phl">[contact email]</span>.</p></div></div>
  )
}

export function Terms() {
  return (
    <div className="wrap flow"><div className="doc"><span className="label">Terms of use</span><h2 style={{ marginTop: 12 }}>Before you print.</h2><p className="small muted">Last updated <span className="phl">[date]</span></p>
      <h3>Measurements can be wrong</h3><p>Check the sizes that matter before printing. Don't use SnapPart parts where failure could hurt someone (safety, vehicle, medical, electrical, load-bearing, child-safety, pressure or food-contact) unless a qualified person has checked them.</p>
      <h3>What you may make</h3><p>Parts you own or may copy, for repair and personal use. You're responsible for patents, design rights, copyrights and trademarks. <b style={{ color: 'var(--ink)' }}>No weapons or weapon parts, including any firearm component, and nothing illegal to make or own.</b></p>
      <h3>No warranty</h3><p>Provided "as is". To the fullest extent the law allows, we aren't liable for loss or damage from using SnapPart or what you print. SnapPart is not made or endorsed by Meta. Governed by the laws of <span className="phl">[governing law]</span>.</p></div></div>
  )
}
