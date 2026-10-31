// Static illustrations from the approved prototype (docs/design/approved/snappart_prototype_v3.html), as SVG strings.
const reduce = typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches
const A = '#3DDBC3', MB = '#6B95FF', AM = '#F6B44C'

function rnd(s: number) { return () => ((s = (s * 9301 + 49297) % 233280) / 233280) }
function code(x: number, y: number, s: number, seed: number, fg = 'var(--ink)', bg = 'var(--panel)') {
  const r = rnd(seed), c = s / 6
  let o = `<rect x="${x}" y="${y}" width="${s}" height="${s}" fill="${fg}"/><rect x="${x + c}" y="${y + c}" width="${c * 4}" height="${c * 4}" fill="${bg}"/>`
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) if (r() > 0.5) o += `<rect x="${x + c * (1 + i)}" y="${y + c * (1 + j)}" width="${c}" height="${c}" fill="${fg}"/>`
  return o
}

export const MAT = (() => {
  const W = 297, H = 210, s = 17, p: number[][] = []
  for (let i = 0; i < 9; i++) { p.push([14 + (i * (W - 28 - s)) / 8, 12]); p.push([14 + (i * (W - 28 - s)) / 8, H - 12 - s]) }
  for (let j = 1; j <= 4; j++) { const y = 12 + (j * (H - 24 - s)) / 5; p.push([14, y]); p.push([W - 14 - s, y]) }
  let o = `<svg viewBox="-12 -10 ${W + 24} ${H + 34}" role="img" aria-label="A4 landscape mat with 26 codes"><rect width="${W}" height="${H}" rx="4" fill="#F2F5F4"/>`
  p.forEach((q, k) => (o += code(q[0], q[1], s, k * 7 + 3, '#0A0D0E', '#F2F5F4')))
  return o + `<rect x="${W / 2 - 36}" y="${H / 2 - 24}" width="72" height="48" rx="6" fill="none" stroke="#9AA6A4" stroke-dasharray="3 3"/><text x="${W / 2}" y="${H / 2 + 4}" text-anchor="middle" font-size="9" font-family="Geist Mono,monospace" fill="var(--ink-3)">part here</text><path d="M14 ${H + 10}H${W - 14}M14 ${H + 6}v8M${W - 14} ${H + 6}v8" stroke="#3DDBC3" stroke-width=".8"/><text x="${W / 2}" y="${H + 21}" text-anchor="middle" font-size="8" font-family="Geist Mono,monospace" fill="#3DDBC3">297 mm</text><path d="M14 -4h${s}M14 -7v6M${14 + s} -7v6" stroke="#3DDBC3" stroke-width=".8"/><text x="${14 + s + 5}" y="-1" font-size="7" font-family="Geist Mono,monospace" fill="#3DDBC3">22.0 mm</text></svg>`
})()

export const ART = [
  `<svg viewBox="0 0 260 150"><path d="M40 108L130 136L220 108L130 82Z" fill="var(--panel-2)" stroke="var(--line-2)"/>${[[64, 108], [96, 120], [164, 120], [196, 108], [130, 92]].map((q, k) => code(q[0] - 5, q[1] - 4, 9, k + 9)).join('')}<path d="M115 102l15 6 15-6-15-6z" fill="var(--line)" stroke="var(--ink-3)"/><ellipse cx="130" cy="92" rx="100" ry="30" fill="none" stroke="${A}" stroke-dasharray="3 5" opacity=".7"/><g><rect x="-9" y="-6" width="18" height="12" rx="3" fill="${A}"/><animateMotion dur="7s" repeatCount="indefinite" path="M230 92A100 30 0 1 1 30 92A100 30 0 1 1 230 92" ${reduce ? 'begin="indefinite"' : ''}/></g></svg>`,
  `<svg viewBox="0 0 260 150">${[[22, 24, -5], [140, 20, 5], [80, 70, 0]].map((f, k) => `<g transform="translate(${f[0]} ${f[1]}) rotate(${f[2]})"><rect width="100" height="62" rx="7" fill="var(--panel-2)" stroke="var(--line-2)"/><path d="M24 44l28-22 28 8-4 16-26 6z" fill="${MB}" fill-opacity=".14" stroke="${MB}" stroke-width="1.8" stroke-dasharray="130"><animate attributeName="stroke-dashoffset" values="130;0;0" dur="3s" begin="${k * 0.5}s" repeatCount="indefinite"/></path></g>`).join('')}</svg>`,
  `<svg viewBox="0 0 260 150">${Array.from({ length: 10 }, (_, i) => Array.from({ length: 6 }, (_, j) => { const inn = i > 0 && i < 9 && j > 0 && j < 5 && !(i > 2 && i < 6 && j > 1 && j < 4); return `<rect x="${32 + i * 20}" y="${16 + j * 20}" width="17" height="17" rx="3" fill="${inn ? A : 'var(--line)'}" opacity="${inn ? 0.85 : 0.6}"/>` }).join('')).join('')}</svg>`,
  `<svg viewBox="0 0 260 150"><rect x="24" y="26" width="212" height="44" rx="12" fill="rgba(107,149,255,.12)" stroke="rgba(107,149,255,.4)"/><text x="40" y="53" font-size="12" font-family="Geist,sans-serif" fill="var(--ink)">How deep are the 4 slots?</text><rect x="120" y="86" width="116" height="40" rx="12" fill="rgba(61,219,195,.12)" stroke="rgba(61,219,195,.45)"/><text x="178" y="111" text-anchor="middle" font-size="13" font-family="Geist Mono,monospace" fill="${A}">3.7 mm</text></svg>`,
]

export const SLOTS = [
  { k: '1 · High ring', t: 'High ring.', l: 'Circle the part once from about 45° above.', h: -34, flip: false },
  { k: '2 · Low ring', t: 'Low ring.', l: 'Circle once more, about 20° above the table.', h: -14, flip: false },
  { k: '3 · Flipped', t: 'Part flipped.', l: 'Turn the part over and circle once more, so the underside is seen.', h: -26, flip: true },
]

export function guide(i: number) {
  const s = SLOTS[i], cy = 170 + s.h, ry = 46 + s.h / 4
  return `<svg viewBox="0 0 400 250"><path d="M70 190L200 236L330 190L200 146Z" fill="var(--panel-2)" stroke="var(--line-2)"/>${[[100, 190], [150, 208], [250, 208], [300, 190], [150, 168], [250, 168]].map((q, k) => code(q[0] - 7, q[1] - 5, 13, k + 21)).join('')}<path d="M176 186l24 9 24-9-24-9z" fill="${s.flip ? 'rgba(246,180,76,.25)' : 'var(--line)'}" stroke="var(--ink-3)"/>${s.flip ? `<path d="M186 160q14-20 28 0" fill="none" stroke="${AM}" stroke-width="2"/>` : ''}<ellipse cx="200" cy="${cy}" rx="150" ry="${ry}" fill="none" stroke="${A}" stroke-dasharray="5 6" stroke-width="1.4"/><g><rect x="-12" y="-8" width="24" height="16" rx="4" fill="${A}"/><circle r="4" fill="#03201B"/><animateMotion dur="8s" repeatCount="indefinite" path="M350 ${cy}A150 ${ry} 0 1 1 50 ${cy}A150 ${ry} 0 1 1 350 ${cy}" ${reduce ? 'begin="indefinite"' : ''}/></g><text x="20" y="28" font-size="12" font-family="Geist Mono,monospace" fill="var(--ink-3)">${s.flip ? 'part flipped · same orbit' : s.h < -20 ? 'about 45° above' : 'about 20° above'}</text></svg>`
}

/** Coverage ring from the capture check: one segment per azimuth sector, teal when the video has frames there. */
export function ring(covered: boolean[]) {
  const n = covered.length
  let o = '<svg viewBox="0 0 132 132" role="img" aria-label="Angles filmed around the part">'
  covered.forEach((v, k) => {
    const a0 = (k / n) * 6.2832 - 1.5708, a1 = ((k + 0.8) / n) * 6.2832 - 1.5708, R = 60, r = 46
    const P = (t: number, rr: number) => [66 + rr * Math.cos(t), 66 + rr * Math.sin(t)]
    const [x0, y0] = P(a0, R), [x1, y1] = P(a1, R), [x2, y2] = P(a1, r), [x3, y3] = P(a0, r)
    o += `<path d="M${x0} ${y0}A${R} ${R} 0 0 1 ${x1} ${y1}L${x2} ${y2}A${r} ${r} 0 0 0 ${x3} ${y3}Z" fill="${v ? A : AM}"/>`
  })
  return o + `<text x="66" y="71" text-anchor="middle" font-size="15" font-family="Geist Mono,monospace" fill="var(--ink)">${Math.round((covered.filter(Boolean).length / n) * 100)}%</text></svg>`
}

export const SUN = '<circle cx="8" cy="8" r="3" stroke="currentColor" stroke-width="1.4" fill="none"/><path d="M8 1v1.6M8 13.4V15M1 8h1.6M13.4 8H15M3 3l1.1 1.1M11.9 11.9L13 13M3 13l1.1-1.1M11.9 4.1L13 3" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/>'
export const MOON = '<path d="M13.5 9.5A6 6 0 0 1 6.5 2.5a6 6 0 1 0 7 7z" stroke="currentColor" stroke-width="1.4" fill="none" stroke-linejoin="round"/>'
