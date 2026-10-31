// Browser-side media: pull sharp JPEG frames from a phone video, and record speech as WAV (mono, 16-bit, 16 kHz).

const once = (el: EventTarget, ev: string) => new Promise<void>(ok => el.addEventListener(ev, () => ok(), { once: true }))

function sharpness(ctx: CanvasRenderingContext2D, w: number, h: number): number {
  const d = ctx.getImageData(0, 0, w, h).data, g = new Float32Array(w * h)
  for (let i = 0; i < w * h; i++) g[i] = d[i * 4] * 0.299 + d[i * 4 + 1] * 0.587 + d[i * 4 + 2] * 0.114
  let s = 0, s2 = 0, n = 0
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    const i = y * w + x, l = g[i - 1] + g[i + 1] + g[i - w] + g[i + w] - 4 * g[i]
    s += l; s2 += l * l; n++
  }
  return n ? s2 / n - (s / n) ** 2 : 0                       // variance of the Laplacian
}

/** Samples `candidates` evenly spaced frames, keeps the `keep` sharpest (in time order), each a JPEG at most `longEdge` px. */
export async function extractFrames(file: Blob, candidates: number, keep: number, longEdge: number,
  progress: (done: number, of: number) => void): Promise<Blob[]> {
  const v = document.createElement('video')
  v.muted = true; v.playsInline = true; v.preload = 'auto'; v.src = URL.createObjectURL(file)
  try {
    await Promise.race([once(v, 'loadedmetadata'), once(v, 'error').then(() => { throw new Error('This video could not be opened in the browser. Try an MP4 or WebM.') })])
    if (!Number.isFinite(v.duration)) {                       // recorder files without a duration: seek far to make the browser find it
      v.currentTime = 1e101; await once(v, 'durationchange'); v.currentTime = 0; await once(v, 'seeked')
    }
    const W = v.videoWidth, H = v.videoHeight, k = Math.min(1, longEdge / Math.max(W, H))
    const cw = Math.round(W * k), ch = Math.round(H * k)
    if (Math.min(cw, ch) < 240) throw new Error('This video is too small. Film at 720p or more.')
    const big = document.createElement('canvas'); big.width = cw; big.height = ch
    const sw = 160, sh = Math.max(2, Math.round((160 * ch) / cw))
    const small = document.createElement('canvas'); small.width = sw; small.height = sh
    const bctx = big.getContext('2d')!, sctx = small.getContext('2d', { willReadFrequently: true })!
    const got: { t: number; s: number; b: Blob }[] = []
    for (let i = 0; i < candidates; i++) {
      v.currentTime = (v.duration * (i + 0.5)) / candidates
      await once(v, 'seeked')
      bctx.drawImage(v, 0, 0, cw, ch); sctx.drawImage(v, 0, 0, sw, sh)
      const b = await new Promise<Blob | null>(ok => big.toBlob(ok, 'image/jpeg', 0.9))
      if (b) got.push({ t: i, s: sharpness(sctx, sw, sh), b })
      progress(i + 1, candidates)
    }
    return got.sort((a, b) => b.s - a.s).slice(0, keep).sort((a, b) => a.t - b.t).map(x => x.b)
  } finally { URL.revokeObjectURL(v.src) }
}

export type Recorder = { stop: () => Promise<Blob>; cancel: () => void }

export async function record(): Promise<Recorder> {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } })
  const ctx = new AudioContext()
  const src = ctx.createMediaStreamSource(stream)
  // ponytail: ScriptProcessorNode is deprecated but works in every browser; move to an AudioWorklet if it is ever removed.
  const proc = ctx.createScriptProcessor(4096, 1, 1), chunks: Float32Array[] = []
  proc.onaudioprocess = e => chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)))
  src.connect(proc); proc.connect(ctx.destination)
  const close = () => { proc.disconnect(); src.disconnect(); stream.getTracks().forEach(t => t.stop()); void ctx.close() }
  return {
    cancel: close,
    stop: async () => {
      close()
      const all = new Float32Array(chunks.reduce((n, c) => n + c.length, 0)); let o = 0
      for (const c of chunks) { all.set(c, o); o += c.length }
      return wav(resample(all, ctx.sampleRate, 16000), 16000)
    },
  }
}

function resample(x: Float32Array, from: number, to: number): Float32Array {
  if (from === to) return x
  const n = Math.floor((x.length * to) / from), y = new Float32Array(n), r = from / to
  for (let i = 0; i < n; i++) {                              // linear interpolation; speech is band-limited well below 8 kHz
    const p = i * r, j = Math.floor(p), f = p - j
    y[i] = x[j] * (1 - f) + (x[j + 1] ?? x[j]) * f
  }
  return y
}

export function wav(x: Float32Array, rate: number): Blob {
  const b = new DataView(new ArrayBuffer(44 + x.length * 2)), s = (o: number, t: string) => [...t].forEach((c, i) => b.setUint8(o + i, c.charCodeAt(0)))
  s(0, 'RIFF'); b.setUint32(4, 36 + x.length * 2, true); s(8, 'WAVE'); s(12, 'fmt ')
  b.setUint32(16, 16, true); b.setUint16(20, 1, true); b.setUint16(22, 1, true); b.setUint32(24, rate, true)
  b.setUint32(28, rate * 2, true); b.setUint16(32, 2, true); b.setUint16(34, 16, true); s(36, 'data'); b.setUint32(40, x.length * 2, true)
  for (let i = 0; i < x.length; i++) b.setInt16(44 + i * 2, Math.max(-1, Math.min(1, x[i])) * 0x7fff, true)
  return new Blob([b.buffer], { type: 'audio/wav' })
}
