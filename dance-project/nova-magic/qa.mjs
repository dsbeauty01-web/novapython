// QA harness — headless Playwright self-test of the magic-light renderer.
// Drives the synthetic test harness, screenshots each scenario, samples the FX
// canvas pixels, and judges PASS/FAIL against the spec (esp. the confidence-gate
// "no blob when the part is hidden" bug). Run: node qa.mjs
import { chromium } from 'playwright';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { mkdirSync } from 'fs';

const here = dirname(fileURLToPath(import.meta.url));
const shots = join(here, 'shots');
mkdirSync(shots, { recursive: true });
const url = 'file://' + join(here, 'test-harness.html').replace(/\\/g, '/');

const results = [];
function judge(name, pass, detail) {
  results.push({ name, pass, detail });
  console.log(`${pass ? 'PASS' : 'FAIL'}  ${name}  —  ${detail}`);
}

// in-page: sample the fx canvas; count bright pixels per region + avg bright color
const SAMPLE = `(() => {
  const c = document.getElementById('fx');
  const g = c.getContext('2d');
  const W = c.width, H = c.height;
  const img = g.getImageData(0,0,W,H).data;
  const regions = { all:[0,0,1,1], topStrip:[0,0,1,0.15], corners:null, center:[0.30,0.30,0.70,0.62] };
  let brightAll=0, sumR=0,sumG=0,sumB=0, brightTop=0, brightCenter=0, brightCorners=0;
  let satN=0, satR=0,satG=0,satB=0;   // saturated (non-white) bright pixels → the hue carrier
  for (let y=0;y<H;y++){
    for (let x=0;x<W;x++){
      const i=(y*W+x)*4;
      const a=img[i+3]; if(a<12) continue;
      const r=img[i],gg=img[i+1],b=img[i+2];
      if (r+gg+b < 120) continue;
      brightAll++; sumR+=r; sumG+=gg; sumB+=b;
      const sat = Math.max(r,gg,b) - Math.min(r,gg,b);
      if (sat > 35) { satN++; satR+=r; satG+=gg; satB+=b; }
      const fx=x/W, fy=y/H;
      if (fy<0.15) brightTop++;
      if (fx>0.30&&fx<0.70&&fy>0.30&&fy<0.62) brightCenter++;
      if ((fx<0.12||fx>0.88)&&(fy<0.12||fy>0.88)) brightCorners++;
    }
  }
  const n=Math.max(1,brightAll), sn=Math.max(1,satN);
  return { W,H, brightAll, brightTop, brightCenter, brightCorners, satN,
           avg:{r:Math.round(sumR/n),g:Math.round(sumG/n),b:Math.round(sumB/n)},
           sat:{r:Math.round(satR/sn),g:Math.round(satG/sn),b:Math.round(satB/sn)} };
})()`;

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 720, height: 760 } });
const errors = [];
page.on('pageerror', e => errors.push(e.message));
page.on('console', m => { if (m.type() === 'error') errors.push('console:' + m.text()); });

await page.goto(url);
await page.waitForFunction('window.__ready === true', { timeout: 8000 });

// --- structural checks ---
const banner = (await page.textContent('#banner')).trim();
judge('banner correct', /NOVA-LIGHT/.test(banner) && /v300/.test(banner), `"${banner}"`);
const logHasInit = await page.evaluate(() => (window.__novaLog || []).some(l => /logger init/.test(l)));
judge('logger init', logHasInit, 'log line present');
const novaVW = await page.evaluate(() => {
  const el = document.getElementById('nova');
  return (el.getBoundingClientRect().width / window.innerWidth) * 100;
});
judge('Nova orb ≤12vw', novaVW <= 12.5, `orb = ${novaVW.toFixed(1)}vw`);

async function scene(name, warmMs) {
  await page.evaluate(s => window.__setScenario(s), name);
  await page.waitForTimeout(warmMs);
  const m = await page.evaluate(SAMPLE);
  await page.screenshot({ path: join(shots, name + '.png'), clip: { x: 0, y: 0, width: 720, height: 520 } });
  return m;
}

// 1) arm raise → light flows ON the arm (substantial bright pixels)
const up = await scene('armUp', 1300);
judge('arm-raise renders ribbon', up.brightAll > 1500, `bright px = ${up.brightAll}`);
judge('arm-raise has top-strip chevron', up.brightTop > 50, `top-strip px = ${up.brightTop}`);

// 2) THE BUG TEST: hide arm (conf 0.2) → after fade, NOTHING drawn anywhere
const low = await scene('lowConf', 1600);
judge('hidden arm draws NOTHING', low.brightAll < 60, `bright px = ${low.brightAll} (must be ~0)`);
judge('no face/corner blob', low.brightCorners === 0 && low.brightCenter < 20, `corners=${low.brightCorners} center=${low.brightCenter}`);

// 3) color = quality: clean (mint-gold) vs messy (amber-red).
// Use the continuously-moving arm (never parks) + equal warm time so the only
// variable is quality. Judge hue on SATURATED pixels (the white core is excluded).
async function colorSample(qv, tag) {
  await page.evaluate(() => window.__setScenario('armSide'));
  await page.evaluate(v => { const e = document.getElementById('q'); e.value = v; e.dispatchEvent(new Event('input')); }, qv);
  await page.waitForTimeout(1200);
  const m = await page.evaluate(SAMPLE);
  await page.screenshot({ path: join(shots, tag + '.png'), clip: { x: 0, y: 0, width: 720, height: 520 } });
  return m;
}
const clean = await colorSample(0.95, 'color-clean');
const messy = await colorSample(0.12, 'color-messy');
const cleanRedBias = clean.sat.r - clean.sat.g;   // mint-gold → low / negative
const messyRedBias = messy.sat.r - messy.sat.g;   // amber-red → clearly positive
judge('color reacts to quality', messyRedBias > cleanRedBias + 14,
      `sat redBias clean=${cleanRedBias} messy=${messyRedBias} (clean=${JSON.stringify(clean.sat)} messy=${JSON.stringify(messy.sat)})`);

// 4) body-relative: near (big) vs far (small) — both render, ratio ~ tracks body
const near = await scene('near', 1300);
const far = await scene('far', 1300);
judge('body-relative renders near & far', near.brightAll > 1000 && far.brightAll > 200,
      `near=${near.brightAll} far=${far.brightAll}`);

// 5) idle shimmer: dim, centered, never in the top corners
const idle = await scene('idle', 1200);
judge('idle shimmer is subtle & centered', idle.brightCorners === 0 && idle.brightAll < up.brightAll,
      `idle bright=${idle.brightAll} corners=${idle.brightCorners}`);

judge('no page errors', errors.length === 0, errors.length ? errors.join(' | ') : 'clean');

await browser.close();

const failed = results.filter(r => !r.pass);
console.log(`\n${results.length - failed.length}/${results.length} checks passed. shots → ${shots}`);
if (failed.length) { console.log('FAILURES:', failed.map(f => f.name).join(', ')); process.exit(1); }
