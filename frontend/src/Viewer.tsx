import { useEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react'
import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { raw } from './api'

const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches
const isLight = () => {
  const t = document.documentElement.getAttribute('data-theme')
  return t ? t === 'light' : matchMedia('(prefers-color-scheme: light)').matches
}

export type Pick = { point: number[]; normal: number[] }

/** three.js view of the part's preview GLB (positions in mm, node scale 0.001, z-up rotated to y-up). Camera positions, if given, are
 * in the part frame in mm and get the same transform. With `pick`, a click on the model reports the point and normal in the part frame
 * (mm), and `mark` shows a marker there. `src` is an API path (/v1/...) or a static file of this site. */
export function Viewer({ src, cameras, spin, style, children, onDraft, pick, onPick, mark, dragExtra, fallback }: {
  src: string | null; cameras?: number[][]; spin?: boolean; style?: CSSProperties; children?: ReactNode; onDraft?: (d: boolean) => void
  pick?: boolean; onPick?: (p: Pick) => void; mark?: number[] | null; dragExtra?: ReactNode; fallback?: string
}) {
  const el = useRef<HTMLDivElement>(null)
  const three = useRef<{ content: THREE.Group; fit: () => void; flip: (under: boolean) => void; marker: THREE.Mesh; part: () => THREE.Mesh | null; setPart: (m: THREE.Mesh | null) => void } | null>(null)
  const [state, setState] = useState<'idle' | 'loading' | 'ok' | 'error'>('idle')
  const [draft, setDraft] = useState(false)
  const [err, setErr] = useState('')
  const [under, setUnder] = useState(false)
  const pickRef = useRef({ pick, onPick }); pickRef.current = { pick, onPick }

  useEffect(() => {
    const host = el.current!
    const r = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    r.setPixelRatio(Math.min(2, devicePixelRatio)); r.toneMapping = THREE.ACESFilmicToneMapping; r.toneMappingExposure = 0.92
    host.prepend(r.domElement)
    const sc = new THREE.Scene(), cam = new THREE.PerspectiveCamera(28, 1, 0.001, 100)
    sc.add(new THREE.HemisphereLight(0xdff7f2, 0x0b1214, 0.55 * Math.PI))
    for (const [c, i, p] of [[0xffffff, 1.25, [80, 140, 110]], [0x6be8d5, 0.9, [-120, 40, -90]], [0x6b95ff, 0.55, [120, -20, -60]]] as const) {
      const l = new THREE.DirectionalLight(c, i * Math.PI); l.position.set(p[0], p[1], p[2]); sc.add(l)
    }
    const content = new THREE.Group(); sc.add(content)
    const grid = new THREE.GridHelper(1, 44); (grid.material as THREE.Material).transparent = true; (grid.material as THREE.Material).opacity = 0.55; sc.add(grid)
    const gridTheme = () => {
      const c = new THREE.Color(isLight() ? 0xd5dfdc : 0x162022), cs = grid.geometry.attributes.color
      for (let i = 0; i < cs.count; i++) cs.setXYZ(i, c.r, c.g, c.b)
      cs.needsUpdate = true
    }
    gridTheme()
    const mo = new MutationObserver(gridTheme); mo.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] })
    const ctl = new OrbitControls(cam, r.domElement)
    ctl.enableDamping = true; ctl.autoRotate = !!spin && !reduce; ctl.autoRotateSpeed = 0.8; ctl.maxPolarAngle = Math.PI                          // allow turning the part over to see its underside
    ctl.addEventListener('start', () => (ctl.autoRotate = false))
    const fit = () => {
      const b = new THREE.Box3().setFromObject(content)
      if (b.isEmpty()) return
      const s = b.getSize(new THREE.Vector3()), size = Math.max(s.x, s.y, s.z), c = b.getCenter(new THREE.Vector3())
      grid.scale.setScalar(size * 4); grid.position.set(c.x, b.min.y, c.z)
      ctl.target.copy(c)
      const d = (size / Math.tan((28 * Math.PI) / 360)) * (host.clientWidth < 560 ? 1.15 : 0.85)
      cam.position.set(c.x - d * 0.55, c.y + d * 0.5, c.z + d * 0.65); cam.near = size / 100; cam.far = size * 100; cam.updateProjectionMatrix()
    }
    // Turn over: swing the camera to the mirror position below the part (phi -> pi - phi), eased unless motion is reduced.
    let flipTo: number | null = null
    const flip = (u: boolean) => {
      ctl.autoRotate = false
      const ph = new THREE.Spherical().setFromVector3(cam.position.clone().sub(ctl.target)).phi
      flipTo = u === ph > Math.PI / 2 ? null : Math.PI - ph
    }
    const step = () => {
      if (flipTo === null) return
      const off = cam.position.clone().sub(ctl.target), sp = new THREE.Spherical().setFromVector3(off)
      sp.phi += (flipTo - sp.phi) * (reduce ? 1 : 0.12)
      if (Math.abs(flipTo - sp.phi) < 0.003) { sp.phi = flipTo; flipTo = null }
      cam.position.copy(ctl.target).add(off.setFromSpherical(sp))
    }
    // Picking: a click (not a drag) on the model, converted into the mesh's own frame, which is the part frame in mm.
    let partMesh: THREE.Mesh | null = null
    const marker = new THREE.Mesh(new THREE.SphereGeometry(1.4, 16, 12), new THREE.MeshBasicMaterial({ color: 0xf6b44c, depthTest: false }))
    marker.renderOrder = 10; marker.visible = false
    const ray = new THREE.Raycaster(), down = { x: 0, y: 0 }
    const pd = (e: PointerEvent) => { down.x = e.clientX; down.y = e.clientY }
    const pu = (e: PointerEvent) => {
      const { pick: on, onPick: cb } = pickRef.current
      if (!on || !cb || !partMesh || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 5) return
      const b = r.domElement.getBoundingClientRect()
      ray.setFromCamera(new THREE.Vector2(((e.clientX - b.left) / b.width) * 2 - 1, -((e.clientY - b.top) / b.height) * 2 + 1), cam)
      const hit = ray.intersectObject(partMesh, false)[0]
      if (!hit) return
      const p = partMesh.worldToLocal(hit.point.clone()), n = hit.face?.normal ?? new THREE.Vector3(0, 0, 1)
      cb({ point: [p.x, p.y, p.z].map(v => +v.toFixed(3)), normal: [n.x, n.y, n.z].map(v => +v.toFixed(3)) })
    }
    r.domElement.addEventListener('pointerdown', pd); r.domElement.addEventListener('pointerup', pu)
    three.current = { content, fit, flip, marker, part: () => partMesh, setPart: m => { partMesh = m } }
    const rs = () => { r.setSize(host.clientWidth, host.clientHeight); cam.aspect = host.clientWidth / Math.max(1, host.clientHeight); cam.updateProjectionMatrix() }
    rs(); const ro = new ResizeObserver(rs); ro.observe(host)
    let raf = 0
    const loop = () => { raf = requestAnimationFrame(loop); step(); ctl.update(); grid.visible = cam.position.y >= grid.position.y; r.render(sc, cam) }   // hide the floor grid when looking from below
    loop()
    return () => {
      r.domElement.removeEventListener('pointerdown', pd); r.domElement.removeEventListener('pointerup', pu)
      cancelAnimationFrame(raf); ro.disconnect(); mo.disconnect(); ctl.dispose()
      sc.traverse(o => { const m = o as THREE.Mesh; m.geometry?.dispose?.(); (m.material as THREE.Material | undefined)?.dispose?.() })
      r.dispose(); r.domElement.remove(); three.current = null
    }
  }, [spin])

  useEffect(() => {
    if (!src) return
    let live = true
    setState('loading')
    ;(async () => {
      try {
        const res = src.startsWith('/v1/') ? await raw(src) : await fetch(src)
        if (!res.ok) throw new Error(fallback ?? 'The model could not be loaded.')
        const d = res.headers.get('X-SnapPart-Draft') === 'true'
        const buf = await res.arrayBuffer()
        if (new TextDecoder().decode(buf.slice(0, 4)) !== 'glTF') throw new Error(fallback ?? 'The model could not be loaded.')   // a dev server answers a missing file with the page
        const g = await new GLTFLoader().parseAsync(buf, '')
        if (!live || !three.current) return
        const mat = new THREE.MeshPhysicalMaterial({ color: 0x9faaa8, roughness: 0.55, metalness: 0, clearcoat: 0.25, clearcoatRoughness: 0.6, side: THREE.DoubleSide })
        g.scene.traverse(o => { if ((o as THREE.Mesh).isMesh) (o as THREE.Mesh).material = mat })
        const { content, fit } = three.current
        content.clear(); content.add(g.scene)
        let first: THREE.Mesh | null = null
        g.scene.traverse(o => { if (!first && (o as THREE.Mesh).isMesh) first = o as THREE.Mesh })
        three.current.setPart(first)
        if (cameras?.length) {
          const cg = new THREE.Group(); cg.scale.setScalar(0.001); cg.rotation.x = -Math.PI / 2   // same frame as the GLB root node
          const m = new THREE.MeshBasicMaterial({ color: 0x3ddbc3 })
          for (const p of cameras) { const s = new THREE.Mesh(new THREE.SphereGeometry(2.2, 10, 8), m); s.position.set(p[0], p[1], p[2]); cg.add(s) }
          content.add(cg)
        }
        fit(); setDraft(d); onDraft?.(d); setUnder(false); setState('ok')
      } catch (e) {
        if (live) { setErr(e instanceof Error ? e.message : 'The model could not be loaded.'); setState('error') }
      }
    })()
    return () => { live = false }
  }, [src, cameras, onDraft, fallback])

  // The marker lives in the part mesh's frame, so its position is the picked point in mm.
  useEffect(() => {
    const t = three.current, m = t?.part()
    if (!t || !m) return
    if (mark) { m.add(t.marker); t.marker.position.set(mark[0], mark[1], mark[2]); t.marker.visible = true } else t.marker.visible = false
  }, [mark, state])

  const turn = () => { three.current?.flip(!under); setUnder(!under) }

  return (
    <div className="stage-sm" ref={el} style={{ ...style, cursor: pick ? 'crosshair' : undefined }}>
      {children}
      {draft && <span className="draft" title="This model failed a quality check. Downloads are labelled DRAFT.">DRAFT</span>}
      {state !== 'ok' && <div className="empty">{state === 'loading' ? 'Loading the model…' : state === 'error' ? err : 'No model yet.'}</div>}
      {state === 'ok' && <div className="drag">Drag to turn · <button className="link tiny" style={{ pointerEvents: 'auto' }} onClick={turn}>{under ? 'Turn back' : 'Turn over'}</button>{dragExtra}</div>}
    </div>
  )
}
