import {maxExtraDepth, promptApplies} from './prompting.mjs';

const $ = id => document.getElementById(id);
const canvas = $('canvas'), ctx = canvas.getContext('2d');
const stage = $('stage');
const zoomHistory = [];
const historyLimit = 6;
const maxEncodedFrameLength = 8_000_000;
const centerPoint = [0.5, 0.5];
let frame = null, original = null, focus = centerPoint, showGuide = false, busy = false;
let warmup = {state: 'idle'}, backendChosen = false;
let nextFrameId = 0;
let promptModeChosen = false;
let aimTimer;
const background = {revision: 0, pendingRevision: null, ready: null, error: '', retryAt: 0};

const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
function cropBounds(point, zoomFactor) {
  const size = canvas.width;
  const width = size / zoomFactor;
  const [left, top] = point.map(value => clamp(Math.round(value * size) - width / 2, 0, size - width));
  return [left, top, left + width, top + width];
}

function updateGuide() {
  const visible = Boolean(frame && showGuide && !busy);
  $('cropGuide').hidden = $('clickGuide').hidden = !visible;
  if (!visible) return;
  const [left, top, right] = cropBounds(focus, Number($('zoomFactor').value));
  const percent = value => `${100 * value / canvas.width}%`;
  $('cropGuide').style.left = percent(left);
  $('cropGuide').style.top = percent(top);
  $('cropGuide').style.width = percent(right - left);
  $('clickGuide').style.left = `${focus[0] * 100}%`;
  $('clickGuide').style.top = `${focus[1] * 100}%`;
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const image = frame?.decodedViews?.[$('view').value] || frame?.image;
  if (image) ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
  updateGuide();
}

function showDetail(detail) {
  $('promptDetails').hidden = !detail;
  $('promptSummary').textContent = detail?.summary || 'Prompt used for the last image';
  $('detailHeading').textContent = detail?.heading || '';
  $('detailDirection').textContent = detail?.direction || '';
  $('cropObservation').textContent = detail?.observation || '';
  $('usedPrompt').textContent = detail?.prompt || '';
}

function waitingForModel() {
  return warmup.state === 'warming' && $('backend').value !== 'upscale';
}

function promptMode() {
  return $('backend').value === 'upscale' ? 'crop' : $('promptMode').value;
}

function promptTarget(source) {
  const zoomFactor = Number($('zoomFactor').value);
  return {
    frameId: source.id, depth: source.depth, trail: source.trail, center: focus, zoomFactor,
    bounds: cropBounds(focus, zoomFactor).map(value => value / canvas.width),
    settings: JSON.stringify([$('prompt').value.trim(), $('variation').value, Number($('seed').value)])
  };
}

function directorBody(source, target) {
  const [hint, variation, seed] = JSON.parse(target.settings);
  return {
    image: source.referenceImage || source.image.src, center: target.center, zoom_factor: target.zoomFactor,
    ...(hint ? {initial_prompt: hint} : {}), step: source.detailStep + 1,
    zoom_depth: source.depth, variation, seed
  };
}

function cachedPrompt(target) {
  const entry = background.ready;
  return entry && promptApplies(entry.target, target) ? entry.detail : null;
}

function updatePromptStatus() {
  const status = $('promptStatus');
  status.hidden = promptMode() !== 'background' || !frame;
  if (status.hidden) return;
  let message = 'Aim at a detail or click to start preparing a prompt.';
  const ready = cachedPrompt(promptTarget(frame));
  if (ready) message = `Qwen prompt ready for this area · ${ready.seconds.toFixed(1)}s analysis.`;
  else if (background.pendingRevision === background.revision) message = 'Qwen is preparing a prompt. You can keep zooming.';
  else if (background.pendingRevision !== null) message = 'An earlier Qwen request is finishing. You can keep zooming.';
  else if (Date.now() < background.retryAt) message = background.error;
  if (status.textContent !== message) status.textContent = message;
}

function clearBackgroundPrompt() {
  clearTimeout(aimTimer);
  background.revision++;
  background.ready = null;
  background.error = '';
  background.retryAt = 0;
  updatePromptStatus();
}

async function preparePrompt(source, target) {
  if (background.pendingRevision !== null || Date.now() < background.retryAt || cachedPrompt(target)) return;
  const revision = background.revision;
  background.pendingRevision = revision;
  updatePromptStatus();
  try {
    const detail = await requestDetail(directorBody(source, target));
    if (revision === background.revision) background.ready = {target, detail};
  } catch (error) {
    if (revision === background.revision) {
      background.retryAt = Date.now() + (error.status === 409 ? 5000 : 30000);
      background.error = `Background prompt unavailable. ${error.message}`;
    }
  } finally {
    background.pendingRevision = null;
    updatePromptStatus();
  }
}

function schedulePrompt() {
  clearTimeout(aimTimer);
  updatePromptStatus();
  if (!frame || promptMode() !== 'background' || waitingForModel() || !$('seed').checkValidity()) return;
  aimTimer = setTimeout(() => {
    if (!busy && showGuide) void preparePrompt(frame, promptTarget(frame));
  }, 650);
}

function updateControls() {
  for (const element of document.querySelectorAll('input, textarea, select, button')) {
    element.disabled = busy;
  }
  $('back').disabled = busy || !zoomHistory.length;
  $('resetView').disabled = busy || frame === original;
  $('view').disabled = busy || !frame?.views;
  const lanczos = $('backend').value === 'upscale';
  $('refinementSteps').disabled = $('preserveStructure').disabled = busy || lanczos;
  $('preset').disabled = $('prompt').disabled = busy || lanczos || !frame;
  $('promptMode').disabled = busy || lanczos;
  $('variationRow').hidden = promptMode() === 'crop';
  $('warmupStatus').hidden = warmup.state !== 'warming' && warmup.state !== 'failed';
  if (warmup.state === 'warming') {
    $('warmupStatus').textContent = `${warmup.message} You can open an image or use Lanczos.`;
  } else if (warmup.state === 'failed') {
    $('warmupStatus').textContent = `Warm-up did not finish. ${warmup.message}`;
  }
  stage.toggleAttribute('data-image', Boolean(frame));
  stage.toggleAttribute('data-busy', busy || waitingForModel());
  canvas.setAttribute('aria-busy', String(busy));
  $('emptyView').hidden = Boolean(frame);
  updateGuide();
  updatePromptStatus();
}

function showFrame(next, point = centerPoint) {
  clearTimeout(aimTimer);
  frame = next;
  focus = point;
  $('view').value = 'image';
  $('zoomScale').textContent = frame.depth < 20 ? `${2 ** frame.depth}×` : `2^${frame.depth}×`;
  $('status').textContent = frame.status;
  $('timings').textContent = frame.timings || '';
  showDetail(frame.detail);
  updateControls();
  draw();
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('Could not open this image. Try a PNG, JPEG, or WebP.'));
    image.src = url;
  });
}

function squarePreview(source, resolution) {
  const size = Math.min(source.naturalWidth, source.naturalHeight);
  const square = document.createElement('canvas');
  square.width = square.height = resolution;
  const squareContext = square.getContext('2d', {alpha: false});
  squareContext.fillStyle = '#191d20';
  squareContext.fillRect(0, 0, resolution, resolution);
  squareContext.drawImage(source, (source.naturalWidth - size) / 2, (source.naturalHeight - size) / 2,
    size, size, 0, 0, resolution, resolution);
  return square;
}

async function openImage(file) {
  if (!file || busy) return;
  const url = URL.createObjectURL(file);
  busy = true;
  updateControls();
  $('status').textContent = 'Opening image…';
  try {
    const source = await loadImage(url);
    const size = Math.min(source.naturalWidth, source.naturalHeight);
    const reference = size >= 1.5 * canvas.width
      ? squarePreview(source, 2 * canvas.width).toDataURL('image/jpeg', 0.96) : null;
    const referenceImage = reference?.length <= maxEncodedFrameLength ? reference : null;
    const image = await loadImage(squarePreview(source, canvas.width).toDataURL());
    original = {image, referenceImage, id: ++nextFrameId, trail: [], depth: 0, detailStep: 0,
      status: 'Choose a point to zoom in.'};
    zoomHistory.length = 0;
    $('preset').value = $('prompt').value = '';
    clearBackgroundPrompt();
    showFrame(original);
  } catch (error) {
    $('status').textContent = error.message;
  } finally {
    URL.revokeObjectURL(url);
    busy = false;
    updateControls();
  }
}

$('openImage').onclick = () => $('upload').click();
$('upload').onchange = event => {
  const file = event.target.files[0];
  event.target.value = '';
  openImage(file);
};
stage.ondragover = event => {
  if (!event.dataTransfer.types.includes('Files')) return;
  event.preventDefault();
  stage.classList.toggle('dragging', !busy);
};
stage.ondragleave = () => stage.classList.remove('dragging');
stage.ondrop = event => {
  event.preventDefault();
  stage.classList.remove('dragging');
  openImage(event.dataTransfer.files[0]);
};

function pointerFocus(event) {
  const rect = canvas.getBoundingClientRect();
  return [clamp((event.clientX - rect.left) / rect.width, 0, 1),
    clamp((event.clientY - rect.top) / rect.height, 0, 1)];
}

canvas.onpointermove = event => {
  if (busy || event.pointerType === 'touch') return;
  focus = pointerFocus(event);
  showGuide = true;
  updateGuide();
  schedulePrompt();
};
canvas.onpointerleave = canvas.onblur = () => {
  showGuide = false;
  clearTimeout(aimTimer);
  updateGuide();
};
canvas.onfocus = () => {
  showGuide = true;
  updateGuide();
};
canvas.onpointerup = event => {
  if (busy || !frame || !event.isPrimary || event.button !== 0) return;
  focus = pointerFocus(event);
  generate();
};
canvas.onkeydown = event => {
  if (busy || !frame || event.ctrlKey || event.metaKey || event.altKey) return;
  const movement = {ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1]}[event.key];
  if (movement) {
    event.preventDefault();
    const amount = event.shiftKey ? 0.01 : 0.04;
    focus = focus.map((value, axis) => clamp(value + movement[axis] * amount, 0, 1));
    showGuide = true;
    updateGuide();
    schedulePrompt();
  } else if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault();
    if (!event.repeat) generate();
  } else if (event.key === 'Backspace') {
    event.preventDefault();
    if (!event.repeat) goBack();
  }
};
function goBack() {
  if (busy || !zoomHistory.length) return;
  const step = zoomHistory.pop();
  $('zoomFactor').value = String(step.zoomFactor);
  showFrame(step.frame, step.center);
  canvas.focus({preventScroll: true});
}
$('back').onclick = goBack;
$('resetView').onclick = () => {
  if (busy || !original) return;
  zoomHistory.length = 0;
  clearBackgroundPrompt();
  showFrame(original);
};
$('view').onchange = async () => {
  const selected = $('view').value;
  const target = frame;
  const encoded = target?.views?.[selected];
  try {
    if (encoded && !target.decodedViews[selected]) {
      target.decodedViews[selected] = await loadImage(encoded);
    }
    if (frame === target && $('view').value === selected && !busy) draw();
  } catch (error) {
    if (frame !== target || $('view').value !== selected || busy) return;
    $('view').value = 'image';
    $('status').textContent = error.message;
    draw();
  }
};
$('zoomFactor').onchange = () => {
  clearTimeout(aimTimer);
  updateGuide();
  updatePromptStatus();
};

function previewZoom(source, bounds) {
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const size = canvas.width;
  const [previewLeft, previewTop, previewRight] = bounds;
  const cropSize = previewRight - previewLeft;
  const previewDuration = 260;
  const minimumRevealDuration = 160;
  const extraTravelDuration = 150;
  let frame;
  let progress = 0;
  let previewStart;
  const ease = t => t * t * (3 - 2 * t);

  function render(left, top, width, image, opacity = 0) {
    const span = size + (width - size) * progress;
    const viewLeft = left * progress;
    const viewTop = top * progress;
    ctx.drawImage(source, viewLeft, viewTop, span, span, 0, 0, size, size);
    if (image) {
      const scale = size / span;
      ctx.globalAlpha = opacity;
      ctx.drawImage(image, (left - viewLeft) * scale, (top - viewTop) * scale,
        width * scale, width * scale);
      ctx.globalAlpha = 1;
    }
  }

  function preview(now) {
    previewStart ??= now;
    const elapsed = clamp((now - previewStart) / previewDuration, 0, 1);
    // Fill the view with the exact crop immediately while the new detail renders.
    progress = 1 - (1 - elapsed) ** 3;
    render(previewLeft, previewTop, cropSize);
    if (elapsed < 1) frame = requestAnimationFrame(preview);
  }

  if (reducedMotion) {
    progress = 1;
    render(previewLeft, previewTop, cropSize);
  } else {
    frame = requestAnimationFrame(preview);
  }
  const cancel = () => cancelAnimationFrame(frame);
  return {
    cancel,
    finish(image, [left, top, right]) {
      if (reducedMotion) return;
      cancel();
      const from = progress;
      const duration = minimumRevealDuration + extraTravelDuration * (1 - from);
      let revealStart;
      return new Promise(resolve => {
        function reveal(now) {
          revealStart ??= now;
          const t = clamp((now - revealStart) / duration, 0, 1);
          const blend = ease(t);
          progress = from + (1 - from) * blend;
          render(left, top, right - left, image, blend);
          if (t < 1) frame = requestAnimationFrame(reveal);
          else resolve();
        }
        frame = requestAnimationFrame(reveal);
      });
    }
  };
}
async function requestDetail(body, trackStatus = false) {
  const start = performance.now();
  const controller = new AbortController();
  let timer;
  async function showStage() {
    try {
      const response = await fetch('api/director/status', {cache: 'no-store', signal: controller.signal});
      const {stage} = await response.json();
      if (stage && !controller.signal.aborted) {
        const seconds = Math.floor((performance.now() - start) / 1000);
        $('status').textContent = `${stage} · ${seconds}s elapsed`;
      }
    } catch {
      // The main request reports generation errors even when status polling fails.
    } finally {
      if (!controller.signal.aborted) timer = setTimeout(showStage, 3000);
    }
  }
  if (trackStatus) timer = setTimeout(showStage, 1000);
  try {
    const response = await fetch('api/director', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)
    });
    const detail = await response.json();
    if (!response.ok) {
      const error = new Error(typeof detail.detail === 'string' ? detail.detail : JSON.stringify(detail.detail));
      error.status = response.status;
      throw error;
    }
    return detail;
  } finally {
    controller.abort();
    clearTimeout(timer);
  }
}
async function generate() {
  if (!frame || busy) return;
  if (waitingForModel()) {
    $('status').textContent = 'The model is still preparing. You can use Lanczos in the meantime.';
    return;
  }
  for (const id of ['seed', 'sharpen', 'feather']) {
    if (!$(id).checkValidity()) {
      $('status').textContent = 'Check the seed, sharpening, and feather values in Advanced settings.';
      return;
    }
  }
  const started = performance.now();
  const previous = frame;
  const mode = promptMode();
  const target = promptTarget(previous);
  let nextDetail = mode === 'background' ? cachedPrompt(target) : null;
  let promptWarning = '';
  const payload = {
    image: previous.referenceImage || previous.image.src,
    center: focus,
    zoom_factor: Number($('zoomFactor').value),
    prompt: $('prompt').value.trim(),
    backend: $('backend').value,
    refinement_steps: Number($('refinementSteps').value),
    seed: Number($('seed').value),
    sharpen: Number($('sharpen').value),
    feather: Number($('feather').value),
    preserve_structure: $('preserveStructure').checked
  };
  const promptSource = payload.prompt ? 'Manual prompt' : 'Image crop only';
  const fallbackSource = payload.prompt ? 'your prompt hint' : 'the image crop';
  clearTimeout(aimTimer);
  busy = true;
  $('view').value = 'image';
  $('timings').textContent = '';
  updateControls();
  const zoom = previewZoom(previous.image, cropBounds(payload.center, payload.zoom_factor));
  try {
    if (mode === 'wait') {
      $('status').textContent = 'Reading the selected crop and writing a detail direction…';
      try {
        nextDetail = await requestDetail(directorBody(previous, target), true);
        payload.prompt = nextDetail.prompt;
      } catch (error) {
        promptWarning = error instanceof TypeError ? 'The prompt request lost its connection.' : error.message;
      }
      showDetail({
        summary: 'Prompt for this zoom attempt',
        heading: nextDetail ? 'Detail direction for this zoom' : promptSource,
        direction: nextDetail?.detail || promptWarning,
        observation: nextDetail ? `Visible in the analyzed crop: ${nextDetail.observations}` : '',
        prompt: payload.prompt
      });
    }
    if (mode === 'background' && nextDetail) payload.prompt = nextDetail.prompt;
    $('status').textContent = promptWarning
      ? `Qwen unavailable. Generating with ${fallbackSource}…`
      : 'Generating the image…';
    const rendering = fetch('api/zoom', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)
    });
    if (mode === 'background' && !nextDetail) void preparePrompt(previous, target);
    const response = await rendering;
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail));
    const image = await loadImage(data.image);
    await zoom.finish(image, data.crop);
    zoomHistory.push({frame: previous, center: target.center, zoomFactor: target.zoomFactor});
    if (zoomHistory.length > historyLimit) zoomHistory.shift();
    if (warmup.state === 'failed' && payload.backend !== 'upscale') warmup = {state: 'ready'};
    const detailStep = previous.detailStep + Number(Boolean(nextDetail));
    const lanczos = payload.backend === 'upscale';
    const timingParts = [];
    if (data.device) timingParts.push(data.device);
    if (!lanczos) {
      timingParts.push(`Inference ${data.inference_seconds.toFixed(2)}s`);
      if (data.load_seconds > 0) timingParts.push(`Model load ${data.load_seconds.toFixed(2)}s`);
      if (nextDetail) timingParts.push(mode === 'background' ? 'Background Qwen used' : `Qwen ${nextDetail.seconds.toFixed(2)}s`);
    }
    const totalTime = ((performance.now() - started) / 1000).toFixed(2);
    timingParts.push(`Click to result ${totalTime}s`);
    const nextFocus = target.center.map((value, axis) =>
      (Math.round(value * canvas.width) - data.crop[axis]) / (data.crop[2] - data.crop[0]));
    showFrame({
      image,
      id: ++nextFrameId,
      trail: [{frameId: target.frameId, center: target.center, zoomFactor: target.zoomFactor, bounds: target.bounds},
        ...previous.trail].slice(0, maxExtraDepth),
      views: lanczos ? null : {raw: data.raw, baseline: data.baseline},
      decodedViews: {},
      depth: previous.depth + Math.log2(payload.zoom_factor),
      detailStep,
      detail: lanczos ? null : {
        heading: nextDetail ? (mode === 'background' ? 'Background Qwen prompt' : `Detail direction · Step ${detailStep}`) : promptSource,
        direction: nextDetail?.detail || promptWarning,
        observation: nextDetail ? `Visible in the analyzed crop: ${nextDetail.observations}` : '',
        prompt: payload.prompt
      },
      status: lanczos ? 'Lanczos zoom complete.' : promptWarning
        ? `Zoom complete using ${fallbackSource}.` : 'Generated zoom complete.',
      timings: timingParts.join(' · ')
    }, nextFocus);
  } catch(e) {
    $('status').textContent = e instanceof TypeError
      ? 'The request to the zoom server failed. Check the Python terminal for the error.'
      : e.message;
  }
  finally {
    zoom.cancel();
    busy = false;
    updateControls();
    draw();
  }
}
const prompts = {
  natural: 'A close-up photograph of an intricate living ecosystem, interwoven roots, branching moss, tiny fern fronds, mineral grains and glistening water droplets, irregular organic structures at many scales, crisp fine detail throughout, natural diffuse light',
  architecture: 'An intricate architectural world of nested courtyards, narrow staircases, carved stone arches, small balconies and connected passages, structures within structures, weathered materials, coherent perspective, crisp detail throughout, soft daylight',
  surreal: 'A miniature world within an organic landscape, branching river channels surrounding tiny terraced gardens, intricate structures nested inside larger forms, coherent natural materials and illumination, richly varied small-scale detail, crisp focus throughout',
  material: 'An extreme close-up of layered natural material, branching fissures, interlocking crystalline grains, tiny pores and translucent mineral veins, irregular multiscale structure, crisp surface detail throughout, diffuse lighting'
};
$('preset').onchange = () => {
  $('prompt').value = prompts[$('preset').value] || '';
  clearBackgroundPrompt();
};
$('prompt').oninput = () => {
  $('preset').value = '';
  clearBackgroundPrompt();
};
$('variation').onchange = $('seed').oninput = clearBackgroundPrompt;
$('promptMode').onchange = () => {
  promptModeChosen = true;
  clearBackgroundPrompt();
  updateControls();
};

function configureBackend() {
  const steps = $('refinementSteps');
  const maxSteps = $('backend').value === 'z-image' ? 8 : 4;
  const selected = Math.min(Number(steps.value), maxSteps);
  steps.replaceChildren(...Array.from({length: maxSteps}, (_, index) => {
    const count = index + 1;
    return new Option(count === 1 ? '1 step' : count + ' steps', count);
  }));
  steps.value = selected;
  updateControls();
}

$('backend').onchange = () => {
  backendChosen = true;
  clearTimeout(aimTimer);
  configureBackend();
};

async function refreshConfiguration(initial = false) {
  try {
    const response = await fetch('api/config', {cache: 'no-store'});
    if (!response.ok) throw new Error('Configuration request failed');
    const config = await response.json();
    const wasWaiting = waitingForModel();
    warmup = config.warmup || {state: 'idle'};
    $('openvinoBackend').hidden = !config.openvino_available;
    $('backgroundOption').hidden = !config.background_prompting;
    if (initial && config.background_prompting && !promptModeChosen) $('promptMode').value = 'crop';
    if (initial && config.realtime && config.openvino_available && !backendChosen) {
      $('backend').value = 'openvino';
      configureBackend();
    } else {
      updateControls();
    }
    if (wasWaiting && warmup.state === 'ready' && frame && !busy) {
      $('status').textContent = 'Model ready. Choose a point to zoom in.';
    }
  } catch {
    // If polling was already active, retry a transient disconnection.
  }
  if (warmup.state === 'warming') setTimeout(refreshConfiguration, 750);
}

configureBackend();
refreshConfiguration(true);
