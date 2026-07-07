/* ============================================================================
 * nova-light.js  —  Nova Magic-Light Cue Engine  (shared, self-contained)
 * ----------------------------------------------------------------------------
 * One engine. Three games import it (Up Groove / Hand Wave / Hello Hello) and
 * pass a per-game config. No copy-paste duplication across games.
 *
 * The bar: a Pixar wand-trail that flows along the kid's moving limb — Tangled's
 * lantern, Rapunzel's hair, Brave's wisps. Never a debug overlay.
 *
 * THE THREE COMPONENTS of every cue (drawn in this order, one part at a time):
 *   1. Focus point   — soft pulsing comet orb ON the trusted keypoint.
 *   2. Direction arrow — glowing chevron in the top-15% strip, never on the body.
 *   3. Live motion ribbon — the hero: spawns at the keypoint, flows in the travel
 *      direction, undulates like a flag, tapers + fades behind.
 *
 * THE PRECISION RULE (this is what broke the last build):
 *   • Confidence gate >= 0.5. Only ever light a keypoint MoveNet actually trusts.
 *   • NO anatomical fallback. If the cued part isn't confident, draw NOTHING this
 *     frame. A dark frame is correct; a misplaced blob is a bug.
 *   • NO nose / face / corner anchor, ever. Light rides limbs only.
 *   • Smoothing is cosmetic (lerp) — it never invents position when confidence dies.
 *
 * "ADD IDEAS" SHIPPED (and why):
 *   ✓ Anticipation pre-glow — a faint orb appears on the target part ~400ms before
 *     the cue fires, so the kid feels Nova "point" before she asks. (setCue leadMs)
 *   ✓ Success bloom — on a clean hit the ribbon flares + a sparkle burst releases
 *     ~420ms then settles. Reward is light, not a number. (celebrate())
 *   ✓ Breathing idle shimmer — between cues, a barely-there warm pulse at torso
 *     center so the screen never feels dead. Killed the instant a cue is active.
 *   ✓ Velocity hue shimmer — faster motion pushes the core whiter/hotter; effort
 *     reads as heat. (core pass only, so it never tints the whole ribbon.)
 *   ✗ Skipped nothing — all four serve "the kid feels seen."
 *
 * PUBLIC API (window.NovaLight):
 *   const fx = NovaLight.create(fxCanvasEl, gameConfig);
 *   fx.setCue(partName, dir [, {leadMs}]);   // dir: 'up'|'down'|'left'|'right' or {x,y} unit or radians
 *   fx.setQuality(0..1);                     // running score → live color crossfade
 *   fx.celebrate();                          // success bloom + sparkle burst
 *   fx.clearCue();                           // no active cue → idle shimmer
 *   fx.render(keypoints, nowMs);             // call every rAF. keypoints: {name:{x,y,score}}
 *   fx.resize();                             // match canvas backing store to CSS size
 * ========================================================================== */
(function (global) {
  'use strict';

  // ---- palette (exact values from SKILL.md — no improvising) ----------------
  const COL = {
    core:  '#FFFDF5', // hot center — near white
    gold:  '#FFD27A', // body of the glow
    amber: '#FFA63D', // outer falloff / bloom edge
    deep:  '#F2730C', // deepest tail, low alpha only
    clean: '#BFFFC8', // clean-iso mint-gold tint, mixed into core
    messy: '#FF7A3D', // off-axis amber-red — reads "almost", never "wrong"
  };

  // ---- tunable defaults (a game config overrides any of these) --------------
  const DEFAULTS = {
    refShoulder: 220,       // reference shoulder width (px) → body-relative unit
    confidence:  0.5,       // the gate. below this, the part is not lit. period.
    trail: {
      nodes:     22,        // ribbon length in samples
      smooth:    0.35,      // node-follow lerp; lower = silkier lag
      waveAmp:   14,        // px undulation at the tail (× body unit k)
      waveSpeed: 6.0,       // undulation frequency over time
      wavePhase: 0.55,      // phase shift per node — the "snake" tightness
      headWidth: 16,        // px, widest stroke at the comet head
      tailWidth: 1.5,       // px at the tail
      fade:      0.18,      // per-frame trail persistence (lower = longer ghost)
    },
    spark: {
      spawnPerVel: 0.6,     // sparkles ∝ keypoint speed (still limb = none)
      ttl:       [400, 900],// ms lifespan range
      drift:     0.4,       // upward drift px/frame — magic floats up
      twinkle:   true,
      size:      [1, 3],    // px radius range
      gentleness: 1.0,      // 1 = normal, <1 = calmer (Hello Hello)
    },
    arrow: { len: 46, width: 5, pulse: 1.8 },
    idleShimmer: true,      // breathing warm pulse between cues
    topStripFrac: 0.15,     // arrows live in the top 15% strip
  };

  // ---- tiny math/color helpers ----------------------------------------------
  function lerp(a, b, t) { return a + (b - a) * t; }
  function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }
  function hexToRgb(h) {
    h = h.replace('#', '');
    return { r: parseInt(h.slice(0, 2), 16), g: parseInt(h.slice(2, 4), 16), b: parseInt(h.slice(4, 6), 16) };
  }
  function rgbLerp(c1, c2, t) {
    return { r: lerp(c1.r, c2.r, t), g: lerp(c1.g, c2.g, t), b: lerp(c1.b, c2.b, t) };
  }
  function rgba(c, a) { return 'rgba(' + (c.r | 0) + ',' + (c.g | 0) + ',' + (c.b | 0) + ',' + a + ')'; }

  // normalize a direction (string compass | radians | {x,y}) → unit vector
  function dirVec(dir) {
    if (dir == null) return { x: 1, y: 0 };
    if (typeof dir === 'number') return { x: Math.cos(dir), y: Math.sin(dir) };
    if (typeof dir === 'string') {
      switch (dir) {
        case 'up':    return { x: 0,  y: -1 };
        case 'down':  return { x: 0,  y:  1 };
        case 'left':  return { x: -1, y:  0 };
        case 'right': return { x: 1,  y:  0 };
        case 'upleft':    return { x: -0.707, y: -0.707 };
        case 'upright':   return { x:  0.707, y: -0.707 };
        case 'downleft':  return { x: -0.707, y:  0.707 };
        case 'downright': return { x:  0.707, y:  0.707 };
      }
      return { x: 1, y: 0 };
    }
    const m = Math.hypot(dir.x, dir.y) || 1;
    return { x: dir.x / m, y: dir.y / m };
  }

  // ---- the engine instance ---------------------------------------------------
  function create(canvas, userCfg) {
    const cfg = Object.assign({}, DEFAULTS, userCfg || {});
    cfg.trail = Object.assign({}, DEFAULTS.trail, (userCfg && userCfg.trail) || {});
    cfg.spark = Object.assign({}, DEFAULTS.spark, (userCfg && userCfg.spark) || {});
    cfg.arrow = Object.assign({}, DEFAULTS.arrow, (userCfg && userCfg.arrow) || {});

    const fx = canvas.getContext('2d');

    // per-instance state
    let spine = [];          // [{x,y}] newest first — the ribbon spine
    let sparks = [];         // [{x,y,vx,vy,born,ttl,r}]
    let curCol = hexToRgb(COL.gold);   // live (crossfading) body color
    let quality = 0.5;
    let cue = null;          // {part, vec}
    let cueLiveAt = 0;       // ms — when the full cue becomes active (after lead)
    let preGlowPart = null;  // part shown faintly during the lead window
    let bloomUntil = 0;      // ms — success bloom active until
    let burstPending = false;
    let lastK = 1;           // last good body-relative unit
    let kpCache = null;      // last keypoints (for idle torso center)

    function resize() {
      // match backing store to the CSS box (the FX canvas is sized by the page)
      const r = canvas.getBoundingClientRect();
      const dpr = Math.min(global.devicePixelRatio || 1, 2);
      canvas.width = Math.max(2, Math.round(r.width * dpr));
      canvas.height = Math.max(2, Math.round(r.height * dpr));
      fx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    // ---- cue control ----
    function setCue(part, dir, opts) {
      const leadMs = (opts && opts.leadMs) || 0;
      cue = { part: part, vec: dirVec(dir) };
      cueLiveAt = (opts && opts.now != null ? opts.now : performance.now()) + leadMs;
      preGlowPart = leadMs > 0 ? part : null;
      spine = [];            // start a fresh ribbon for the new part
    }
    function clearCue() { cue = null; preGlowPart = null; spine = []; }
    function setQuality(q) { quality = clamp(q, 0, 1); }
    function celebrate() { bloomUntil = performance.now() + 420; burstPending = true; }

    // ---- target color from running quality + velocity ----
    function qualityColor(q) {
      const gold = hexToRgb(COL.gold);
      if (q > 0.66) return rgbLerp(gold, hexToRgb(COL.clean), clamp((q - 0.66) / 0.34, 0, 1));
      if (q < 0.40) return rgbLerp(gold, hexToRgb(COL.messy), clamp((0.40 - q) / 0.40, 0, 1));
      return gold;
    }

    // ---- radial-gradient orb (never a hard arc fill) ----
    function glowOrb(x, y, r, col, alpha) {
      if (r <= 0) return;
      const g = fx.createRadialGradient(x, y, 0, x, y, r);
      g.addColorStop(0, rgba(col, 1));
      g.addColorStop(0.4, rgba(col, 0.8));
      g.addColorStop(1, rgba(col, 0));
      fx.globalAlpha = alpha == null ? 1 : alpha;
      fx.fillStyle = g;
      fx.beginPath(); fx.arc(x, y, r, 0, 7); fx.fill();
      fx.globalAlpha = 1;
    }

    // ---- glowing chevron in the top strip (never on the body) ----
    function glowChevron(cx, dir, k, col, now) {
      const W = canvas.clientWidth || canvas.width;
      const stripY = (canvas.clientHeight || canvas.height) * (cfg.topStripFrac * 0.6);
      const x = clamp(cx, 60, W - 60);
      const len = cfg.arrow.len * k;
      const pulse = 1 + 0.2 * Math.sin(now / 300);
      const ang = Math.atan2(dir.y, dir.x);
      // two short additive strokes forming a ">" rotated toward dir
      function leg(sign) {
        const a1 = ang + sign * (Math.PI * 0.72);
        fx.beginPath();
        fx.moveTo(x, stripY);
        fx.lineTo(x + Math.cos(a1) * len, stripY + Math.sin(a1) * len);
        fx.stroke();
      }
      fx.lineCap = 'round';
      // bloom + core pass
      fx.strokeStyle = rgba(hexToRgb(COL.amber), 0.5);
      fx.lineWidth = cfg.arrow.width * pulse * 2.6 * k; leg(1); leg(-1);
      fx.strokeStyle = rgba(col, 0.95);
      fx.lineWidth = cfg.arrow.width * pulse * k; leg(1); leg(-1);
    }

    // ---- the ribbon: 3 additive passes, tapered + sine-waved ----
    function strokeRibbon(t, k, widthMul, alpha, col, headCol) {
      if (spine.length < 2) return;
      fx.beginPath();
      for (let i = 0; i < spine.length; i++) {
        const a = spine[i];
        const b = spine[i + 1] || spine[i - 1] || a;
        const ang = Math.atan2(b.y - a.y, b.x - a.x) + Math.PI / 2; // perpendicular
        const taper = i / spine.length;                            // 0 head → 1 tail
        const amp = cfg.trail.waveAmp * k * taper;                 // 0 at head, grows to tail
        const off = Math.sin(t * cfg.trail.waveSpeed + i * cfg.trail.wavePhase) * amp;
        const x = a.x + Math.cos(ang) * off;
        const y = a.y + Math.sin(ang) * off;
        i ? fx.lineTo(x, y) : fx.moveTo(x, y);
      }
      const head = spine[0], tail = spine[spine.length - 1];
      const g = fx.createLinearGradient(head.x, head.y, tail.x, tail.y);
      g.addColorStop(0, rgba(headCol || col, 1));
      g.addColorStop(0.6, rgba(col, 0.6));
      g.addColorStop(1, rgba(hexToRgb(COL.deep), 0)); // fade to nothing at the tail
      fx.strokeStyle = g;
      fx.lineWidth = Math.max(cfg.trail.tailWidth, cfg.trail.headWidth * widthMul * k);
      fx.lineCap = 'round'; fx.lineJoin = 'round';
      fx.globalAlpha = alpha; fx.stroke(); fx.globalAlpha = 1;
    }

    // ---- main per-frame entry ----
    function render(kp, now) {
      const W = canvas.clientWidth || canvas.width;
      const H = canvas.clientHeight || canvas.height;
      kpCache = kp || kpCache;

      // 1) fade old light — NOT clearRect (that kills the trail). FX canvas only.
      fx.globalCompositeOperation = 'source-over';
      fx.fillStyle = rgba({ r: 0, g: 0, b: 0 }, cfg.trail.fade);
      fx.fillRect(0, 0, W, H);

      // body-relative unit k from shoulders (cosmetic scaling — fall back to last good)
      let k = lastK;
      if (kp && kp.left_shoulder && kp.right_shoulder &&
          kp.left_shoulder.score >= cfg.confidence && kp.right_shoulder.score >= cfg.confidence) {
        const unit = Math.hypot(kp.right_shoulder.x - kp.left_shoulder.x,
                                kp.right_shoulder.y - kp.left_shoulder.y);
        if (unit > 1) { k = unit / cfg.refShoulder; lastK = k; }
      }

      fx.globalCompositeOperation = 'lighter'; // ADD blend for everything glowing

      // ---- no active cue → breathing idle shimmer, then done ----
      if (!cue) {
        if (cfg.idleShimmer && kp && kp.left_shoulder && kp.right_shoulder &&
            kp.left_hip && kp.right_hip &&
            kp.left_shoulder.score >= cfg.confidence && kp.right_shoulder.score >= cfg.confidence) {
          const cxx = (kp.left_shoulder.x + kp.right_shoulder.x) / 2;
          const cyy = (kp.left_shoulder.y + kp.right_shoulder.y) / 2;
          // barely-there: kept dim so additive accumulation never builds past a whisper
          const breathe = 0.013 + 0.011 * (0.5 + 0.5 * Math.sin(now / 1400));
          glowOrb(cxx, cyy, cfg.refShoulder * k * 0.7, hexToRgb(COL.gold), breathe);
        }
        fx.globalCompositeOperation = 'source-over';
        return;
      }

      // ---- lead window: anticipation pre-glow only (no ribbon yet) ----
      const live = now >= cueLiveAt;
      const tp = kp && kp[cue.part];
      const trusted = tp && tp.score >= cfg.confidence;

      if (!live) {
        if (preGlowPart && trusted) {
          const pul = 0.18 + 0.12 * (0.5 + 0.5 * Math.sin(now / 260));
          glowOrb(tp.x, tp.y, 14 * k, qualityColor(quality), pul);
        }
        fx.globalCompositeOperation = 'source-over';
        return;
      }

      // ---- THE GATE: cued part not trusted → draw NOTHING this frame ----
      // (no nose, no corner, no fallback. a dark frame is correct.)
      if (!trusted) {
        fx.globalCompositeOperation = 'source-over';
        return;
      }

      // 2) update spine with the trusted keypoint, then smooth (cosmetic only)
      spine.unshift({ x: tp.x, y: tp.y });
      if (spine.length > cfg.trail.nodes) spine.pop();
      for (let i = 1; i < spine.length; i++) {
        spine[i].x = lerp(spine[i].x, spine[i - 1].x, cfg.trail.smooth);
        spine[i].y = lerp(spine[i].y, spine[i - 1].y, cfg.trail.smooth);
      }

      // velocity of the head (for sparkles + hue shimmer)
      const vel = spine[1] ? Math.hypot(tp.x - spine[1].x, tp.y - spine[1].y) : 0;

      // crossfade live color toward quality target (~200ms)
      const target = qualityColor(quality);
      curCol = rgbLerp(curCol, target, 0.12);
      // quality color is carried through ALL three passes so it actually reads:
      //   bloom leans amber→quality, core stays near-white but takes a hint of it.
      const bloomCol = rgbLerp(hexToRgb(COL.amber), curCol, 0.6);
      // velocity hue shimmer then pushes the CORE toward hot white (effort = heat)
      const heat = clamp(vel / (40 * k), 0, 1) * 0.35;
      let coreCol = rgbLerp(hexToRgb(COL.core), curCol, 0.30);
      coreCol = rgbLerp(coreCol, { r: 255, g: 255, b: 255 }, heat);

      // success bloom multipliers
      const blooming = now < bloomUntil;
      const bm = blooming ? 1.5 : 1.0;
      const ba = blooming ? 1.0 : 1.0;

      const t = now / 1000;
      // 3) ribbon — 3 additive passes (bloom → body → core), widest+dimmest first
      strokeRibbon(t, k, 3.5 * bm, 0.14 * ba, bloomCol);
      strokeRibbon(t, k, 1.8 * bm, 0.45 * ba, curCol, curCol);
      strokeRibbon(t, k, 0.6 * bm, 0.95 * ba, coreCol, coreCol);

      // 4) focus orb — comet head, the brightest thing on screen
      const orbPulse = 1 + 0.12 * Math.sin(now / 220);
      glowOrb(tp.x, tp.y, 18 * k * orbPulse * (blooming ? 1.4 : 1), curCol, 0.9);
      glowOrb(tp.x, tp.y, 7 * k * orbPulse, coreCol, 1);

      // 5) sparkles ∝ velocity (still limb = none; fast move = eruption)
      const rate = vel * cfg.spark.spawnPerVel * cfg.spark.gentleness * 0.05;
      if (Math.random() < rate) spawnSpark(tp, k, now, 1);
      if (burstPending) { for (let i = 0; i < 14 * cfg.spark.gentleness; i++) spawnSpark(tp, k, now, 2.2); burstPending = false; }
      sparks = sparks.filter(function (s) {
        const age = (now - s.born) / s.ttl;
        if (age >= 1) return false;
        s.x += s.vx; s.y += s.vy; s.vy -= 0.01; // float up
        const tw = cfg.spark.twinkle ? 0.6 + 0.4 * Math.sin(now / 80 + s.x) : 1;
        glowOrb(s.x, s.y, s.r * (1 - age * 0.5), coreCol, (1 - age) * tw);
        return true;
      });

      // 6) direction arrow — glowing chevron, top strip, pointing where to travel
      glowChevron(tp.x, cue.vec, k, curCol, now);

      fx.globalCompositeOperation = 'source-over';
    }

    function spawnSpark(p, k, now, energy) {
      sparks.push({
        x: p.x, y: p.y,
        vx: (Math.random() - 0.5) * 1.2 * energy,
        vy: -Math.random() * cfg.spark.drift * energy - 0.2,
        born: now,
        ttl: lerp(cfg.spark.ttl[0], cfg.spark.ttl[1], Math.random()),
        r: lerp(cfg.spark.size[0], cfg.spark.size[1], Math.random()) * k,
      });
    }

    return { setCue, clearCue, setQuality, celebrate, render, resize,
             get config() { return cfg; } };
  }

  global.NovaLight = { create, COL, DEFAULTS };
})(typeof window !== 'undefined' ? window : this);
