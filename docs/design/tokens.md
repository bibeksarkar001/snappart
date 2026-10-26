# SnapPart design tokens

Apple-style, light, calm. Source of truth for Claude Code. Values come from the mockup in `docs/design/prototype/`. Put these in `tailwind.config` and CSS variables; never hard-code hexes in components.

## Color
| Token | Hex | Use |
|---|---|---|
| `bg` | #F5F5F7 | Page background |
| `card` | #FFFFFF | Cards, sheets |
| `ink` | #1D1D1F | Main text |
| `ink-2` | #5E5E64 | Secondary text, captions |
| `ink-3` | #3C3C43 | Body text on cards |
| `muted` | #8E8E93 | Disabled/waiting text |
| `accent` | #0A63D8 | Primary buttons, links, selected borders |
| `accent-tint` | #E8F0FD | Model badges, icon tiles, selected fills |
| `fill` | #EDEDF0 | Secondary buttons, steppers, chips |
| `track` | #E3E3E8 | Segmented control background |
| `line` | #EDEDF0 | Dividers |
| `ok` | #34C759 | Done steps, switch on |
| `ok-text` | #1E7B34 | Success text |
| `danger` | #B3261E | Errors, destructive buttons |
| `disabled` | #C7C7CC | Disabled button fill (text `ink-2`) |
| `camera-bg` | #1C1C1E to #2A2A2E | Camera screen |
| `camera-accent` | #0A84FF | Part box, current thumbnail |
| `viewer-top` / `-side` / `-hole` / `-stroke` | #E3EBF7 / #C9D6EA / #F5F5F7 / #9DB3D6 | 3D viewer solid colors. Wireframe: no fill, stroke #0A63D8 |
| `code-bg` | #1D1D1F | Script blocks (text #E5E5EA) |

Text on `accent` is white. Keep text at least 4.5:1 contrast.

## Type
Stack: `-apple-system, BlinkMacSystemFont, "SF Pro Text", Inter, system-ui, sans-serif` (load Inter 400/500/600/700/800).
| Role | Size / weight / tracking |
|---|---|
| Display (home, About) | 40 to 56px / 700 / -0.035em |
| Page title | 32px / 700 / -0.03em |
| Card title | 18 to 20px / 600 / -0.02em |
| Body | 15 to 17px / 400 to 500 |
| Caption | 12 to 14px / 400, color `ink-2` |
| Eyebrow | 13px / 600 / uppercase / +0.04em / `ink-2` |
| Code | ui-monospace, 11.5 to 12px |
Keychain preview fonts: Rounded (`"SF Pro Rounded"`, 700), Serif (Georgia, 600), Bold (Inter, 800).

## Shape and spacing
- Card radius 22px; sheet radius 26px; buttons 14px; chips and pills fully rounded; small controls 8 to 12px.
- Page padding 20px (phone), 32px (desktop, max width 1440px). Gaps 12 to 24px.
- Touch targets at least 44px. Primary button height 50 to 52px.
- Phone frame 390 x 844 for design checks. Desktop breakpoint: three columns above 1100px.

## Components
- **Segmented control:** track `track`, selected segment white, weight 600.
- **Stepper:** minus and plus buttons in `fill`, value centered in 600 weight.
- **Switch:** 51 x 31, on `ok`, off `disabled`.
- **Model badge:** 12px / 600, `accent` on `accent-tint` for AI steps; `ink-2` on `fill` for non-AI steps (OpenCV, CadQuery, font engine).
- **Bottom sheet:** dark scrim 40%, white card, radius 26px, 8px inset from the edges.
- **Toast:** dark pill near the bottom, auto-dismiss about 2.4s.
- **Motion:** 150 to 250ms ease-out for state changes; spinner for the running step; respect `prefers-reduced-motion`.
- **Icons:** inline stroke SVG, 1.8px stroke, round caps. No emoji.
