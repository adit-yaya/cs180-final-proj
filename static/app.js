const $ = id => document.getElementById(id);
const canvas = $('canvas'), ctx = canvas.getContext('2d');
let current = null, result = null, center = [0.5, 0.5], busy = false;
let detailStep = 0;
function resetDetails() {
  detailStep = 0;
  $('detailHeading').textContent = 'Detail direction appears after your next zoom.';
  $('detailDirection').textContent = ''; $('usedPrompt').textContent = '';
  $('cropObservation').textContent = '';
  $('promptDetails').open = false;
}
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const image = result?.[$('view').value] || current;
  if (image) ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
}
function loadImage(url) {
  return new Promise((resolve, reject) => {
    const im = new Image(); im.onload = () => resolve(im); im.onerror = () => reject(new Error('Cannot decode image')); im.src = url;
  });
}
function setBusy(value) {
  busy = value;
  document.querySelectorAll('input, textarea, select, button').forEach(el => el.disabled = value);
}
$('upload').onchange = async e => {
  const file = e.target.files[0]; if (!file) return;
  const url = URL.createObjectURL(file);
  try {
    const im = await loadImage(url), s = Math.min(im.width, im.height);
    const c = document.createElement('canvas'); c.width = c.height = 1024;
    c.getContext('2d').drawImage(im, (im.width-s)/2, (im.height-s)/2, s, s, 0, 0, 1024, 1024);
    current = await loadImage(c.toDataURL()); result = null; center = [0.5,0.5]; resetDetails();
    $('view').value = 'image'; draw();
    $('status').textContent = 'Choose a region and zoom. First diffusion call downloads/loads the model.';
  } catch (e) { $('status').textContent = e.message; }
  finally { URL.revokeObjectURL(url); }
};
canvas.onpointerdown = e => {
  if (busy || !current) return;
  const rect = canvas.getBoundingClientRect();
  center = [clamp((e.clientX-rect.left)/rect.width,0,1), clamp((e.clientY-rect.top)/rect.height,0,1)];
  generate();
};
$('view').onchange = () => draw();
function previewZoom(source, center) {
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const size = canvas.width;
  const cropSize = size / 2;
  const previewDuration = 6000;
  const previewLimit = 0.85;
  const minimumRevealDuration = 1200;
  const extraTravelDuration = 1100;
  // Match decode_zoom_crop so the pending preview and server result align.
  const [previewLeft, previewTop] = center.map(value =>
    clamp(Math.round(value * size) - cropSize / 2, 0, size - cropSize));
  let frame;
  let progress = 0;
  let lastPreviewFrame;
  let previewElapsed = 0;
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
    previewElapsed += Math.min(now - (lastPreviewFrame ?? now), 50);
    lastPreviewFrame = now;
    const elapsed = clamp(previewElapsed / previewDuration, 0, 1);
    // Move toward the crop while inference runs, leaving room for the reveal.
    progress = previewLimit * (1 - (1 - elapsed) ** 3);
    render(previewLeft, previewTop, cropSize);
    if (elapsed < 1) frame = requestAnimationFrame(preview);
  }

  if (!reducedMotion) frame = requestAnimationFrame(preview);
  const cancel = () => cancelAnimationFrame(frame);
  return {
    cancel,
    finish(image, [left, top, right]) {
      if (reducedMotion) return;
      cancel();
      const from = progress;
      const duration = minimumRevealDuration + extraTravelDuration * (1 - from);
      let lastRevealFrame;
      let revealElapsed = 0;
      return new Promise(resolve => {
        function reveal(now) {
          // Keep the blend visible across dropped frames or iframe throttling.
          revealElapsed += Math.min(now - (lastRevealFrame ?? now), 50);
          lastRevealFrame = now;
          const t = clamp(revealElapsed / duration, 0, 1);
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
async function requestDetail(body) {
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
  timer = setTimeout(showStage, 1000);
  try {
    return await fetch('api/director', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)
    });
  } finally {
    controller.abort();
    clearTimeout(timer);
  }
}
async function generate() {
  if (!current || busy) return;
  const source = current;
  let nextDetail = null;
  const payload = {image:source.src, center:center, prompt:$('prompt').value,
    backend:$('backend').value, refinement_steps:Number($('refinementSteps').value),
    seed:Number($('seed').value), sharpen:Number($('sharpen').value), feather:Number($('feather').value)};
  setBusy(true); $('view').value = 'image';
  $('status').textContent = 'Generating… (first model load can take several minutes)';
  const zoom = previewZoom(source, payload.center);
  try {
    if ($('evolvePrompt').checked && payload.backend !== 'upscale') {
      if (!payload.prompt.trim()) throw new Error('Enter a starting prompt or choose a preset.');
      $('status').textContent = 'Reading the selected crop and writing a detail direction…';
      const directorResponse = await requestDetail({
        image: payload.image, center: payload.center,
        initial_prompt: payload.prompt,
        step: detailStep + 1, variation: $('variation').value
      });
      const directed = await directorResponse.json();
      if (!directorResponse.ok) throw new Error(typeof directed.detail === 'string' ? directed.detail : JSON.stringify(directed.detail));
      nextDetail = directed; payload.prompt = directed.prompt;
    }
    $('status').textContent = 'Generating the image…';
    const response = await fetch('api/zoom', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)
    });
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail));
    const [image, raw, baseline] = await Promise.all([data.image, data.raw, data.baseline].map(loadImage));
    await zoom.finish(image, data.crop);
    current = image; result = {image, raw, baseline}; center = [0.5,0.5];
    if (nextDetail) {
      detailStep += 1;
      $('detailHeading').textContent = `Detail direction · Step ${detailStep}`;
      $('detailDirection').textContent = nextDetail.detail;
    } else {
      $('detailHeading').textContent = payload.backend === 'upscale' ? 'Lanczos only · detail refinement paused' : 'Fixed prompt · detail refinement paused';
      $('detailDirection').textContent = '';
    }
    $('usedPrompt').textContent = payload.prompt;
    $('cropObservation').textContent = nextDetail ? `Visible in selected crop: ${nextDetail.observations}` : '';
    $('promptDetails').open = false;
    $('status').textContent = `${data.denoising_steps} denoising steps from t=${data.start_timestep ?? "—"} · inference ${data.inference_seconds.toFixed(2)}s · load ${data.load_seconds.toFixed(2)}s · total ${data.seconds.toFixed(2)}s`;
  } catch(e) { $('status').textContent = e.message; }
  finally { zoom.cancel(); setBusy(false); draw(); }
};
draw();
const prompts = {
  natural: 'A close-up photograph of an intricate living ecosystem, interwoven roots, branching moss, tiny fern fronds, mineral grains and glistening water droplets, irregular organic structures at many scales, crisp fine detail throughout, natural diffuse light',
  architecture: 'An intricate architectural world of nested courtyards, narrow staircases, carved stone arches, small balconies and connected passages, structures within structures, weathered materials, coherent perspective, crisp detail throughout, soft daylight',
  surreal: 'A miniature world within an organic landscape, branching river channels surrounding tiny terraced gardens, intricate structures nested inside larger forms, coherent natural materials and illumination, richly varied small-scale detail, crisp focus throughout',
  material: 'An extreme close-up of layered natural material, branching fissures, interlocking crystalline grains, tiny pores and translucent mineral veins, irregular multiscale structure, crisp surface detail throughout, diffuse lighting'
};
$('preset').onchange = () => { if (prompts[$('preset').value]) { $('prompt').value = prompts[$('preset').value]; resetDetails(); } };
$('prompt').oninput = resetDetails;
$('resetDetails').onclick = resetDetails;
$('evolvePrompt').onchange = resetDetails;

$('backend').onchange = () => {
  const steps = $('refinementSteps');
  const maxSteps = $('backend').value === 'z-image' ? 8 : 4;
  const selected = Math.min(Number(steps.value), maxSteps);
  steps.replaceChildren(...Array.from({length: maxSteps}, (_, index) => {
    const count = index + 1;
    return new Option(count === 1 ? '1 step' : count + ' steps', count);
  }));
  steps.value = selected;
};
$('backend').onchange();
