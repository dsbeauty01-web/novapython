// Reproduce the REAL game path: fake-but-real getUserMedia + real MoveNet from
// CDN. Tells us if the model loads, estimatePoses runs, and what status we reach.
import { chromium } from 'playwright';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const here = dirname(fileURLToPath(import.meta.url));
const url = 'http://localhost:8848/up-groove.html';

const browser = await chromium.launch({
  args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream'],
});
const ctx = await browser.newContext({ permissions: ['camera'], viewport: { width: 720, height: 560 } });
const page = await ctx.newPage();
const logs = [], errs = [], neterr = [];
page.on('console', m => logs.push(m.type() + ': ' + m.text()));
page.on('pageerror', e => errs.push(e.message));
page.on('requestfailed', r => neterr.push(r.url() + ' — ' + (r.failure()?.errorText || '')));

await page.goto(url);
await page.waitForTimeout(9000); // give CDN + model time

const status = await page.textContent('#status').catch(() => '(no #status)');
const diag = await page.evaluate(() => ({
  tf: typeof tf, pd: typeof poseDetection,
  ready: window.__ready === true,
  vw: document.getElementById('cam')?.videoWidth,
  vh: document.getElementById('cam')?.videoHeight,
}));
console.log('status pill :', status);
console.log('diag        :', JSON.stringify(diag));
console.log('netfail     :', neterr.length ? neterr.join('\n              ') : 'none');
console.log('pageerrors  :', errs.length ? errs.join(' | ') : 'none');
console.log('console tail:', logs.slice(-8).join('\n              '));
await browser.close();
