// Smoke-test the three game HTMLs in ?mock=1 (synthetic dancer) — verifies each
// page boots, the engine renders light, banner is right, and there are no JS
// exceptions. CDN resource-load failures (offline) are tolerated: mock mode
// never touches TF.js. Run: node qa-games.mjs
import { chromium } from 'playwright';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { mkdirSync } from 'fs';

const here = dirname(fileURLToPath(import.meta.url));
const shots = join(here, 'shots');
mkdirSync(shots, { recursive: true });

const SAMPLE = `(() => {
  const c = document.getElementById('fx'); const g = c.getContext('2d');
  const W=c.width,H=c.height,img=g.getImageData(0,0,W,H).data; let bright=0,top=0,corners=0;
  for(let y=0;y<H;y++)for(let x=0;x<W;x++){const i=(y*W+x)*4;if(img[i+3]<12)continue;
    const s=img[i]+img[i+1]+img[i+2];if(s<120)continue;bright++;const fx=x/W,fy=y/H;
    if(fy<0.15)top++; if((fx<0.12||fx>0.88)&&(fy<0.12||fy>0.88))corners++;}
  return {bright,top,corners};
})()`;

const games = [
  { file: 'up-groove.html',  banner: /UP\s*GROOVE/ },
  { file: 'hand-wave.html',  banner: /HAND\s*WAVE/ },
  { file: 'hello-hello.html',banner: /HELLO\s*HELLO/ },
];

const results = [];
const judge = (n, p, d) => { results.push({ n, p }); console.log(`${p ? 'PASS' : 'FAIL'}  ${n}  —  ${d}`); };

const browser = await chromium.launch();
for (const g of games) {
  const page = await browser.newPage({ viewport: { width: 720, height: 560 } });
  const errors = [];
  page.on('pageerror', e => errors.push(e.message)); // real JS exceptions only
  const url = 'file://' + join(here, g.file).replace(/\\/g, '/') + '?mock=1';
  await page.goto(url);
  await page.waitForFunction('window.__ready === true', { timeout: 8000 }).catch(() => {});
  await page.waitForTimeout(1800);
  const banner = (await page.textContent('#banner')).replace(/ /g, ' ').trim();
  const m = await page.evaluate(SAMPLE);
  await page.screenshot({ path: join(shots, g.file.replace('.html', '') + '.png'), clip: { x: 0, y: 0, width: 720, height: 540 } });
  judge(g.file + ' banner', g.banner.test(banner), `"${banner}"`);
  judge(g.file + ' renders light', m.bright > 800, `bright=${m.bright} top=${m.top} corners=${m.corners}`);
  judge(g.file + ' no JS errors', errors.length === 0, errors.length ? errors.join(' | ') : 'clean');
  await page.close();
}
await browser.close();

const failed = results.filter(r => !r.p);
console.log(`\n${results.length - failed.length}/${results.length} checks passed.`);
if (failed.length) process.exit(1);
