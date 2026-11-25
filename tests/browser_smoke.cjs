// Run against the local app launched with --realtime --warmup.
const assert = require('node:assert/strict');
const { spawn } = require('node:child_process');
const { once } = require('node:events');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const [url, browserPath, imagePath, screenshots] = process.argv.slice(2);
if (!url || !browserPath || !imagePath || typeof WebSocket === 'undefined') {
  console.error('Usage: node browser_smoke.cjs URL CHROMIUM_EXE IMAGE [SCREENSHOT_DIR] (Node with WebSocket support)');
  process.exit(2);
}

const prefix = path.join(os.tmpdir(), 'zoom-browser-');
const profile = fs.mkdtempSync(prefix);
const browser = spawn(browserPath, [
  '--headless=new', '--remote-debugging-port=0', `--user-data-dir=${profile}`,
  '--no-first-run', '--no-default-browser-check', '--disable-background-networking',
  '--disable-extensions', '--disable-sync', '--disable-component-update',
  '--window-size=1240,900', 'about:blank'
], {windowsHide: true, stdio: 'ignore'});
const browserExit = once(browser, 'exit');
browserExit.catch(() => {});

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
async function until(what, read, timeout = 12000) {
  const end = Date.now() + timeout;
  while (Date.now() < end) {
    const result = await read();
    if (result) return result;
    await sleep(50);
  }
  throw new Error(`Timed out waiting for ${what}`);
}

async function main() {
  const portFile = path.join(profile, 'DevToolsActivePort');
  const port = await until('the test browser', () =>
    fs.existsSync(portFile) && fs.readFileSync(portFile, 'utf8').split(/\r?\n/)[0]);
  const tabs = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
  const ws = new WebSocket(tabs.find(tab => tab.type === 'page').webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    ws.addEventListener('open', resolve, {once: true});
    ws.addEventListener('error', reject, {once: true});
  });
  const pending = new Map();
  const networkPaths = [];
  let id = 0;
  const send = (method, params = {}) => new Promise((resolve, reject) => {
    pending.set(++id, {resolve, reject});
    ws.send(JSON.stringify({id, method, params}));
  });
  ws.addEventListener('message', event => {
    const message = JSON.parse(event.data);
    if (message.method === 'Network.requestWillBeSent') {
      const requestUrl = message.params.request.url;
      if (requestUrl.startsWith(url)) networkPaths.push(new URL(requestUrl).pathname);
    }
    if (message.method === 'Fetch.requestPaused') {
      send('Fetch.failRequest', {requestId: message.params.requestId, errorReason: 'Failed'}).catch(() => {});
    }
    const callbacks = pending.get(message.id);
    if (!callbacks) return;
    pending.delete(message.id);
    if (message.error) callbacks.reject(new Error(message.error.message));
    else callbacks.resolve(message.result);
  });

  async function evaluate(expression) {
    const result = await send('Runtime.evaluate', {expression, returnByValue: true, awaitPromise: true});
    if (result.exceptionDetails) throw new Error(result.exceptionDetails.exception?.description || result.exceptionDetails.text);
    return result.result.value;
  }
  const text = selector => evaluate(`document.querySelector('${selector}')?.textContent`);
  const isDisabled = selector => evaluate(`document.querySelector('${selector}').disabled`);
  const canvasPixels = () => evaluate("document.querySelector('#canvas').toDataURL()");

  async function pointAt(x, y) {
    return evaluate(`(() => {
      const bounds = document.querySelector('#canvas').getBoundingClientRect();
      return {x: bounds.left + bounds.width * ${x}, y: bounds.top + bounds.height * ${y}};
    })()`);
  }
  async function click(x, y) {
    const point = await pointAt(x, y);
    await send('Input.dispatchMouseEvent', {...point, type: 'mousePressed', button: 'left', clickCount: 1});
    await send('Input.dispatchMouseEvent', {...point, type: 'mouseReleased', button: 'left', clickCount: 1});
  }
  async function key(name) {
    await send('Input.dispatchKeyEvent', {type: 'keyDown', key: name});
    await send('Input.dispatchKeyEvent', {type: 'keyUp', key: name});
  }
  async function snapshot(name) {
    if (!screenshots) return;
    fs.mkdirSync(screenshots, {recursive: true});
    const image = await send('Page.captureScreenshot', {format: 'png'});
    fs.writeFileSync(path.join(screenshots, name), Buffer.from(image.data, 'base64'));
  }
  async function waitForZoom(scale) {
    return until(`a generated result at ${scale}`, async () =>
      await text('#zoomScale') === scale && !(await isDisabled('#seed')) && await text('#timings'), 45000);
  }
  const setBackend = value => evaluate(`(() => {
    const select = document.querySelector('#backend');
    select.value = '${value}';
    select.dispatchEvent(new Event('change'));
  })()`);
  const setPromptMode = value => evaluate(`(() => {
    const select = document.querySelector('#promptMode');
    select.value = '${value}';
    select.dispatchEvent(new Event('change'));
  })()`);

  try {
    await send('Network.enable');
    await send('Page.navigate', {url});
    await until('OpenVINO defaults', () => evaluate(
      "document.querySelector('#backend')?.value === 'openvino' && document.querySelector('#promptMode').value === 'crop'"
    ));
    const {root} = await send('DOM.getDocument');
    const {nodeId} = await send('DOM.querySelector', {nodeId: root.nodeId, selector: '#upload'});
    await send('DOM.setFileInputFiles', {nodeId, files: [path.resolve(imagePath)]});
    await until('image upload', () => evaluate("document.querySelector('#stage').hasAttribute('data-image')"));
    assert.equal(await evaluate("document.querySelector('#prompt').value"), '');
    const original = await canvasPixels();
    const corner = await pointAt(.05, .08);
    await send('Input.dispatchMouseEvent', {...corner, type: 'mouseMoved'});
    assert.deepEqual(await evaluate(`(() => {
      const guide = document.querySelector('#cropGuide');
      return [guide.hidden, guide.style.left, guide.style.top, guide.style.width];
    })()`), [false, '0%', '0%', '50%']);
    let startupPreview = 0;
    if (await evaluate("!document.querySelector('#warmupStatus').hidden")) {
      await click(.05, .08);
      assert.equal(await text('#zoomScale'), '1×');
      assert.match(await text('#status'), /still preparing/);
      await setBackend('upscale');
      await click(.05, .08);
      await waitForZoom('2×');
      assert.equal(await text('#status'), 'Lanczos zoom complete.');
      const aimedX = await evaluate("parseFloat(document.querySelector('#clickGuide').style.left)");
      assert.ok(aimedX > 8 && aimedX < 12, 'An edge click stays aimed at its location in the clamped crop.');
      await evaluate("document.querySelector('#resetView').click()");
      await setBackend('openvino');
      startupPreview = 1;
      console.log('Lanczos remained usable while OpenVINO warmed up.');
    }
    await until('model ready', () => evaluate("document.querySelector('#warmupStatus').hidden"), 120000);

    await click(.30, .49);
    const first = await waitForZoom('2×');
    const firstFrame = await canvasPixels();
    assert.notEqual(firstFrame, original);
    assert.equal(await text('#detailHeading'), 'Image crop only');
    console.log(`2× ${first}`);
    await snapshot('desktop.png');
    await click(.51, .55);
    console.log(`4× ${await waitForZoom('4×')}`);
    const secondFrame = await canvasPixels();
    await evaluate("document.querySelector('#canvas').focus()");
    await key('ArrowRight');
    assert.equal(await evaluate("document.querySelector('#clickGuide').style.left"), '54%');
    await key('Enter');
    console.log(`8× ${await waitForZoom('8×')}`);
    assert.equal(networkPaths.filter(route => route === '/api/zoom').length, 3 + startupPreview);
    assert.equal(networkPaths.filter(route => route === '/api/director').length, 0);

    await key('Backspace');
    assert.equal(await text('#zoomScale'), '4×');
    assert.equal(await canvasPixels(), secondFrame);
    assert.equal(await evaluate("document.querySelector('#clickGuide').style.left"), '54%');
    await evaluate(`(() => {
      const select = document.querySelector('#view');
      select.value = 'baseline';
      select.dispatchEvent(new Event('change'));
    })()`);
    await until('the comparison image', async () => await canvasPixels() !== secondFrame);

    await send('Fetch.enable', {patterns: [{urlPattern: '*api/zoom'}]});
    await click(.5, .5);
    await until('render error', async () => (await text('#status')).includes('request to the zoom server failed'));
    await send('Fetch.disable');
    assert.equal(await text('#zoomScale'), '4×');
    assert.equal(await canvasPixels(), secondFrame);
    assert.equal(await isDisabled('#back'), false);

    await evaluate("document.querySelector('#back').click()");
    assert.equal(await text('#zoomScale'), '2×');
    assert.equal(await canvasPixels(), firstFrame);
    await evaluate("document.querySelector('#resetView').click()");
    assert.equal(await text('#zoomScale'), '1×');
    assert.equal(await canvasPixels(), original);
    assert.equal(await isDisabled('#back'), true);
    assert.equal(await isDisabled('#view'), true);

    await evaluate(`(() => {
      const fetchLocal = window.fetch.bind(window);
      window.promptRequests = [];
      window.fetch = async (url, options) => {
        if (url === 'api/director') {
          window.promptRequests.push(JSON.parse(options.body));
          return new Response(JSON.stringify({observations: 'Visible material.', detail: 'Fine texture.',
            prompt: 'Fine surface texture.', seconds: 0.01}), {status: 200});
        }
        if (url === 'api/zoom') {
          const request = JSON.parse(options.body);
          window.lastRenderPrompt = request.prompt;
          window.lastRenderImage = request.image;
        }
        return fetchLocal(url, options);
      };
    })()`);
    await setPromptMode('wait');
    await click(.28, .52);
    await waitForZoom('2×');
    assert.equal(await evaluate("window.promptRequests[0].initial_prompt ?? null"), null);
    assert.equal(await evaluate('window.promptRequests[0].zoom_depth'), 0);
    assert.equal(await evaluate('window.lastRenderPrompt'), 'Fine surface texture.');
    assert.equal(await text('#usedPrompt'), 'Fine surface texture.');
    await evaluate("document.querySelector('#resetView').click()");
    await evaluate(`(() => {
      const fetchLocal = window.fetch.bind(window);
      window.fetch = (url, options) => url === 'api/director'
        ? Promise.resolve(new Response(JSON.stringify({detail: 'Test prompt service unavailable.'}), {status: 503}))
        : fetchLocal(url, options);
    })()`);
    await click(.28, .52);
    await waitForZoom('2×');
    assert.equal(await evaluate('window.lastRenderPrompt'), '');
    assert.equal(await text('#status'), 'Zoom complete using the image crop.');
    assert.equal(await text('#detailDirection'), 'Test prompt service unavailable.');
    console.log('Blank first zoom, crop-to-prompt routing, and fallback passed.');

    await evaluate(`(() => {
      document.querySelector('#resetView').click();
      const fetchLocal = window.fetch.bind(window);
      window.backgroundRequests = [];
      window.fetch = (url, options) => {
        if (url !== 'api/director') return fetchLocal(url, options);
        window.backgroundRequests.push(JSON.parse(options.body));
        return new Promise(resolve => {
          window.finishBackgroundPrompt = () => resolve(new Response(JSON.stringify({
            observations: 'A visible branch.', detail: 'Fine grain.', prompt: 'Fine bark grain.', seconds: 12
          }), {status: 200}));
        });
      };
    })()`);
    await setPromptMode('background');
    const aim = await pointAt(.5, .5);
    await send('Input.dispatchMouseEvent', {...aim, type: 'mouseMoved'});
    await until('Qwen prefetch while aiming', () => evaluate('window.backgroundRequests.length === 1'));
    await click(.5, .5);
    const firstBackgroundZoom = await waitForZoom('2×');
    assert.equal(await evaluate('window.lastRenderPrompt'), '');
    assert.equal(await evaluate('window.backgroundRequests[0].image === window.lastRenderImage'), true);
    if (process.env.ZOOM_EXPECT_HIGH_RES === '1') {
      const originalSize = await evaluate(`new Promise(resolve => {
        const source = new Image();
        source.onload = () => resolve(source.naturalWidth);
        source.src = window.lastRenderImage;
      })`);
      assert.equal(originalSize, 2048);
    }
    assert.match(await text('#promptStatus'), /keep zooming/);
    await evaluate('window.finishBackgroundPrompt()');
    await until('a ready prompt', async () => (await text('#promptStatus')).includes('ready'));
    await click(.5, .5);
    const promptedBackgroundZoom = await waitForZoom('4×');
    assert.equal(await evaluate('window.lastRenderPrompt'), 'Fine bark grain.');
    assert.equal(await evaluate("window.lastRenderImage.startsWith('data:image/png;base64,')"), true);
    assert.equal(await evaluate('window.backgroundRequests.length'), 1);
    assert.match(promptedBackgroundZoom, /Background Qwen used/);
    await evaluate(`(() => {
      const zoom = document.querySelector('#zoomFactor');
      zoom.value = '4';
      zoom.dispatchEvent(new Event('change'));
    })()`);
    await click(.5, .5);
    await waitForZoom('16×');
    assert.equal(await evaluate('window.lastRenderPrompt'), '');
    assert.equal(await evaluate('window.backgroundRequests.length'), 2);
    await evaluate(`(() => {
      document.querySelector('#resetView').click();
      document.querySelector('#zoomFactor').value = '2';
      window.finishBackgroundPrompt();
    })()`);
    await until('the stale prompt to be ignored', async () => (await text('#promptStatus')).startsWith('Aim'));
    console.log(`Background zoom without a ready prompt: ${firstBackgroundZoom}`);
    console.log(`Background zoom with a ready prompt: ${promptedBackgroundZoom}`);

    await evaluate(`(() => {
      const preset = document.querySelector('#preset');
      preset.value = 'natural';
      preset.dispatchEvent(new Event('change'));
    })()`);
    assert.match(await evaluate("document.querySelector('#prompt').value"), /fern fronds/);

    await send('Emulation.setDeviceMetricsOverride', {mobile: true, width: 390, height: 844, deviceScaleFactor: 1});
    assert.equal(await evaluate('document.documentElement.scrollWidth <= innerWidth'), true);
    await snapshot('mobile.png');
    await send('Emulation.setEmulatedMedia', {features: [{name: 'prefers-reduced-motion', value: 'reduce'}]});
    await setBackend('upscale');
    await click(.45, .52);
    await waitForZoom('2×');
    assert.equal(await text('#status'), 'Lanczos zoom complete.');
    await evaluate(`(async () => {
      const canvas = document.querySelector('#canvas');
      const png = await new Promise(resolve => canvas.toBlob(resolve));
      const transfer = new DataTransfer();
      transfer.items.add(new File([png], 'dropped.png', {type: 'image/png'}));
      document.querySelector('#stage').dispatchEvent(new DragEvent('drop', {dataTransfer: transfer}));
    })()`);
    await until('the dropped image', async () => await text('#zoomScale') === '1×' && !(await isDisabled('#seed')));
    assert.equal(await isDisabled('#back'), true);
    assert.equal(await evaluate("document.querySelector('#prompt').value"), '');
    assert.equal(await evaluate("document.querySelector('#preset').value"), '');
    console.log('Crop guide, zoom, compare, history, error recovery, mobile, reduced motion, and drop passed.');
  } finally {
    ws.send(JSON.stringify({id: ++id, method: 'Browser.close'}));
    ws.close();
  }
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
}).finally(async () => {
  await Promise.race([browserExit, sleep(1500)]).catch(() => {});
  if (browser.exitCode === null) browser.kill();
  if (path.resolve(profile).startsWith(path.resolve(prefix))) {
    await fs.promises.rm(profile, {recursive: true, force: true, maxRetries: 3}).catch(() => {});
  }
});
