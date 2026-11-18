const $ = id => document.getElementById(id);
const canvas = $('canvas'), ctx = canvas.getContext('2d');
let current = null, result = null, center = [0.5, 0.5], busy = false;
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
function draw(image = null, box = null) {
  ctx.clearRect(0, 0, 1024, 1024);
  image ||= result?.[$('view').value] || current;
  if (!image) return;
  if (box) ctx.drawImage(image, box[0]*1024, box[1]*1024, box[2]*1024, box[2]*1024, 0, 0, 1024, 1024);
  else ctx.drawImage(image, 0, 0, 1024, 1024);

}
function loadImage(url) {
  return new Promise((resolve, reject) => {
    const im = new Image(); im.onload = () => resolve(im); im.onerror = () => reject(new Error('Cannot decode image')); im.src = url;
  });
}
function dataURL(image) {
  const c = document.createElement('canvas'); c.width = c.height = 1024;
  c.getContext('2d').drawImage(image, 0, 0, 1024, 1024); return c.toDataURL('image/png');
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
    current = await loadImage(c.toDataURL()); result = null; center = [0.5,0.5];
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
// Reveal the generated patch in the enlarged context.
function animate(duration, frame) {
  return new Promise(resolve => {
    const start = performance.now();
    function tick(t) { const u = Math.min(1, (t-start)/duration); frame(u*u*(3-2*u)); if (u<1) requestAnimationFrame(tick); else resolve(); }
    requestAnimationFrame(tick);
  });
}
async function generate() {
  if (!current || busy) return;
  const source = current;
  const payload = {image:dataURL(source), center:center, prompt:$('prompt').value,
    backend:$('backend').value, refinement_steps:Number($('refinementSteps').value),
    seed:Number($('seed').value), sharpen:Number($('sharpen').value), feather:Number($('feather').value)};
  setBusy(true); $('view').value = 'image';
  $('status').textContent = 'Generating… (first model load can take several minutes)';
  try {
    // Keep the UI responsive while the server generates.
    const pending = fetch('/api/zoom', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)})
      .then(async r => {const data = await r.json(); if (!r.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail)); return data;})
      .then(data => ({data}), error => ({error}));
    // Keep the current frame while inference runs; no history is retained.
    const response = await pending; if (response.error) throw response.error;
    const data = response.data;
    const [image, raw, baseline] = await Promise.all([data.image, data.raw, data.baseline].map(loadImage));
    await animate(450, t => { draw(baseline, [0,0,1]); ctx.globalAlpha=t; ctx.drawImage(image,0,0,1024,1024); ctx.globalAlpha=1; });
    current = image; result = {image, raw, baseline}; center = [0.5,0.5];
    $('status').textContent = `${data.model} · ${data.denoising_steps} denoising steps from t=${data.start_timestep ?? "—"} · inference ${data.inference_seconds.toFixed(2)}s · load ${data.load_seconds.toFixed(2)}s · total ${data.seconds.toFixed(2)}s`;
  } catch(e) { $('status').textContent = e.message; }
  finally { setBusy(false); draw(); }
};
draw();
const prompts = {
  natural: 'A close-up photograph of an intricate living ecosystem, interwoven roots, branching moss, tiny fern fronds, mineral grains and glistening water droplets, irregular organic structures at many scales, crisp fine detail throughout, natural diffuse light',
  architecture: 'An intricate architectural world of nested courtyards, narrow staircases, carved stone arches, small balconies and connected passages, structures within structures, weathered materials, coherent perspective, crisp detail throughout, soft daylight',
  surreal: 'A miniature world within an organic landscape, branching river channels surrounding tiny terraced gardens, intricate structures nested inside larger forms, coherent natural materials and illumination, richly varied small-scale detail, crisp focus throughout',
  material: 'An extreme close-up of layered natural material, branching fissures, interlocking crystalline grains, tiny pores and translucent mineral veins, irregular multiscale structure, crisp surface detail throughout, diffuse lighting'
};
$('preset').onchange = () => { if (prompts[$('preset').value]) $('prompt').value = prompts[$('preset').value]; };

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
