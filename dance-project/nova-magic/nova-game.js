/* ============================================================================
 * nova-game.js — shared game shell for the three Nova dance games.
 * ----------------------------------------------------------------------------
 * Wires webcam + MoveNet (TF.js pose-detection) → mirrored keypoints → the
 * nova-light.js engine, and runs the per-game cue sequence. ZERO duplication:
 * Up Groove / Hand Wave / Hello Hello each just pass a config object.
 *
 * The kid's real scorer (50% dir / 30% iso / 20% timing) lives elsewhere; this
 * shell ships a *placeholder* quality estimate (how well the cued part travels
 * in the cued direction) so the light's color reacts to real movement out of the
 * box. Call NovaGame.setQuality(0..1) from the real scorer to override it.
 *
 * ?mock=1  → drive synthetic poses (no camera) so the page previews anywhere.
 * ========================================================================== */
(function (global) {
  'use strict';

  const QS = new URLSearchParams(location.search);
  const MOCK = QS.get('mock') === '1';

  function log(state, msg) {
    const el = document.getElementById('status');
    if (el) el.textContent = msg;
    (global.__novaLog = global.__novaLog || []).push(msg);
    if (state) console.log('[nova-game]', msg);
  }

  async function start(opts) {
    const canvas = document.getElementById(opts.canvasId || 'fx');
    const video = document.getElementById(opts.videoId || 'cam');
    const cfg = opts.config;
    const light = global.NovaLight.create(canvas, cfg.light || {});
    global.__nova = light;

    function fit() {
      const r = canvas.getBoundingClientRect();
      if (video) { video.style.width = r.width + 'px'; video.style.height = r.height + 'px'; }
      light.resize();
    }
    global.addEventListener('resize', fit);
    fit();

    // ---- quality placeholder + scheduler state ----
    let quality = 0.6, lastCuePos = null, celebrated = false;
    global.NovaGame.setQuality = q => { quality = q; };

    const seq = cfg.sequence;
    let idx = -1, cueAt = 0;
    function nextCue(now) {
      idx = (idx + 1) % seq.length;
      const c = seq[idx];
      light.setCue(c.part, c.dir, { leadMs: c.leadMs || 400, now });
      cueAt = now + (c.holdMs || 2600);
      lastCuePos = null; celebrated = false;
    }

    // ---- estimate quality from cued-part travel direction (placeholder) ----
    function estimate(kp, now) {
      const c = seq[idx]; if (!c) return;
      const p = kp[c.part];
      if (!p || p.score < 0.5) { quality = quality * 0.94 + 0.5 * 0.06; return; }
      if (lastCuePos) {
        const vx = p.x - lastCuePos.x, vy = p.y - lastCuePos.y;
        const sp = Math.hypot(vx, vy);
        const dir = dirVec(c.dir);
        if (sp > 1.2) {
          const align = (vx / sp) * dir.x + (vy / sp) * dir.y;   // -1..1
          const target = 0.5 + 0.5 * align;                      // 0..1
          quality = quality * 0.85 + target * 0.15;
        }
        if (quality > 0.82 && !celebrated && sp > 2) { light.celebrate(); celebrated = true; }
      }
      lastCuePos = { x: p.x, y: p.y };
    }
    function dirVec(d) {
      const m = { up:[0,-1], down:[0,1], left:[-1,0], right:[1,0],
                  upleft:[-.707,-.707], upright:[.707,-.707],
                  downleft:[-.707,.707], downright:[.707,.707] };
      const v = m[d] || [1,0]; return { x: v[0], y: v[1] };
    }

    // ---- keypoint mapping: model space → mirrored display (selfie) ----
    // The <video> is object-fit:cover (fills the screen, center-cropped) AND
    // CSS-mirrored. We must replicate cover here or the light lands off the limb.
    function mapKeypoints(kps, vw, vh) {
      const r = canvas.getBoundingClientRect();
      const scale = Math.max(r.width / vw, r.height / vh);   // cover, not stretch
      const offX = (r.width  - vw * scale) / 2;
      const offY = (r.height - vh * scale) / 2;
      const out = {};
      for (const k of kps) {
        if (!k.name) continue;
        const sx = k.x * scale + offX;
        const sy = k.y * scale + offY;
        out[k.name] = { x: r.width - sx, y: sy, score: k.score == null ? k.confidence : k.score };
      }
      return out;
    }

    function loop(detector) {
      let raf, lastSeen = 0, hinting = false;
      async function step(now) {
        let kp = null;
        if (detector && video.readyState >= 2) {
          try {
            const poses = await detector.estimatePoses(video, { flipHorizontal: false });
            if (poses[0]) kp = mapKeypoints(poses[0].keypoints, video.videoWidth, video.videoHeight);
          } catch (e) { /* transient — skip frame */ }
        } else if (MOCK || !detector) {
          kp = mockPose(now, canvas, seq[Math.max(0, idx)]);
        }
        // "can I see you?" feedback — a still arm lighting nothing is correct,
        // but no person at all should TELL the kid, not just sit dark.
        if (detector) {
          const seen = kp && Object.keys(kp).some(n => kp[n].score >= 0.5);
          if (seen) lastSeen = now;
          const lost = now - lastSeen > 1400;
          if (lost && !hinting) { hinting = true; log(false, 'Step back so I can see you ✨'); }
          if (!lost && hinting) { hinting = false; log(false, 'dance! ✨'); }
        }
        if (idx < 0 || now >= cueAt) nextCue(now);
        if (kp) estimate(kp, now);
        light.setQuality(quality);
        light.render(kp || {}, now);
        raf = requestAnimationFrame(step);
      }
      raf = requestAnimationFrame(step);
    }

    // ---- boot: camera + MoveNet, or mock ----
    if (MOCK) { log(true, 'mock mode — synthetic dancer'); global.__ready = true; return loop(null); }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: 640, height: 480 }, audio: false });
      video.srcObject = stream; await video.play();
      log(true, 'camera on — loading pose model…');
      await tfReady();
      const detector = await poseDetection.createDetector(
        poseDetection.SupportedModels.MoveNet,
        { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING });
      log(true, 'dance! ✨');
      global.__ready = true;
      fit();
      loop(detector);
    } catch (e) {
      log(true, 'no camera → mock preview (' + e.message + ')');
      global.__ready = true;
      loop(null);
    }
  }

  async function tfReady() {
    await tf.setBackend('webgl').catch(() => tf.setBackend('cpu'));
    await tf.ready();
  }

  // synthetic dancer so the page is alive without a camera
  function mockPose(now, canvas, cue) {
    const r = canvas.getBoundingClientRect();
    const W = r.width, H = r.height, T = now / 1000;
    const cx = W / 2, sy = H * 0.42, hipY = H * 0.70, sh = W * 0.11;
    const kp = {
      left_shoulder:  { x: cx - sh, y: sy, score: .95 },
      right_shoulder: { x: cx + sh, y: sy, score: .95 },
      left_hip:  { x: cx - sh * 0.7, y: hipY, score: .9 },
      right_hip: { x: cx + sh * 0.7, y: hipY, score: .9 },
      left_elbow:  { x: cx - sh * 1.4, y: sy + 50, score: .9 },
      right_elbow: { x: cx + sh * 1.4, y: sy + 50, score: .9 },
      head: { x: cx, y: sy - sh * 1.1, score: .9 },
    };
    const part = (cue && cue.part) || 'right_wrist';
    const swing = Math.sin(T * 1.7);
    kp.right_wrist = { x: cx + sh * 1.5 + swing * sh * 0.9, y: sy + 70 - Math.abs(swing) * 90, score: .92 };
    kp.left_wrist  = { x: cx - sh * 1.5 - swing * sh * 0.9, y: sy + 70 - Math.abs(swing) * 90, score: .92 };
    if (!kp[part]) kp[part] = kp.right_wrist;
    return kp;
  }

  global.NovaGame = { start, setQuality: function () {} };
})(window);
