# Nova Magic-Light Cue System — v300 (from-scratch rebuild)

A single shared engine renders the on-camera cue + feedback layer for all three
games as a **Pixar wand-trail that flows along the kid's moving limb** — not a
debug overlay. Built per `CLI-BUILD-FULL.md` + `nova-magic-visuals/SKILL.md`.

## Files
| File | Role |
|---|---|
| `nova-light.js` | **The engine.** Confidence-gated, body-relative, 3-component cue (focus orb + top-strip chevron + undulating ribbon), additive bloom, quality color crossfade, velocity sparkles. One module, imported by all games. |
| `nova-game.js` | Shared shell: webcam + MoveNet → mirrored keypoints → engine + per-game cue scheduler. `?mock=1` previews without a camera. |
| `nova-game.css` | Shared kid-view layout (mirrored cam, FX canvas, Nova orb ≤12vw, banner). |
| `up-groove.html` `hand-wave.html` `hello-hello.html` | The three games — config only (sequence + light overrides). |
| `test-harness.html` | Camera-less synthetic-pose driver for visual self-test. |
| `qa.mjs` / `qa-games.mjs` | Headless Playwright self-test (pixel-judged PASS/FAIL). |

## The precision rule (what broke the last build, now fixed)
- **Confidence gate ≥ 0.5.** Only ever light a keypoint MoveNet trusts.
- **No fallback, no nose/face/corner blob.** If the cued part isn't confident,
  the frame draws **nothing** — verified: hidden arm → 0 bright px, 0 corner px.
- Smoothing is cosmetic (lerp); it never invents position when confidence dies.

## "Add ideas" shipped (and why)
- **Anticipation pre-glow** — faint orb on the target ~400–600ms before the cue
  fires (`setCue` `leadMs`), so Nova "points" before she asks.
- **Success bloom** — clean hit flares the ribbon + releases a sparkle burst
  ~420ms, then settles. Reward is light, not a number. (`celebrate()`)
- **Breathing idle shimmer** — barely-there warm pulse at torso center between
  cues so the screen never feels dead; killed the instant a cue is active.
- **Velocity hue shimmer** — faster motion pushes the core whiter/hotter.

## Per-game config
- **Up Groove** — shoulders → hips, top→down bounce. (Build doc said "head" as the
  top anchor; the NEVER list forbids face/head light, so it starts at shoulders.)
- **Hand Wave** — wrist → elbow → shoulder, then cross to the opposite side; sky-blue banner.
- **Hello Hello** — ages 4–7, calmest: shorter ribbon, gentlest sparkle, softest chevron; pink banner.

## Run with a real camera
`getUserMedia` needs a secure context, so serve over `localhost` (not `file://`):

```
cd nova-magic
python -m http.server 8848      # or: npx --yes serve -l 8848
```
Then open: `http://localhost:8848/up-groove.html`
(append `?mock=1` to any URL to preview the light without a camera).

## QA
```
node qa.mjs        # renderer self-test → 11/11, screenshots in ./shots
node qa-games.mjs  # three games boot + render → 9/9
```

## Hook up the real scorer
The shell ships a placeholder quality estimate (cued-part travel vs. cue
direction) so color reacts to movement out of the box. Override from your
50/30/20 scorer any frame:
```js
NovaGame.setQuality(score01);   // 0..1 → live mint-gold ↔ amber-red crossfade
```
