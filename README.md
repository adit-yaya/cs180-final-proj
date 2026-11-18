# Minimal generative zoom

## Setup / launch (cs180 Conda environment)

```sh
conda activate cs180
cd ~/Documents/cs180/final_proj
python -m pip install -r requirements-model.txt
python app.py
```

Or `./run.sh` after installation; this explicitly uses ~/miniconda3/envs/cs180/bin/python.
Restart the server and reload http://127.0.0.1:8000 after changes. This environment
could download dependencies but could not write to the Conda environment; run the
install command in Terminal. No packages in cs180 were successfully installed by
that attempt. Imaging tests do run in cs180 using existing NumPy/Pillow.

Upload an image; click a point to generate. Prompt and seed remain fixed until
edited. No history, cache of generated images, or disk output. Lanczos only skips
diffusion. SD-Turbo downloads on its first diffusion request and stays in memory.
Direct StableDiffusionImg2ImgPipeline import avoids unrelated Qwen imports.
ZOOM_MODEL=stabilityai/sdxl-turbo optionally selects the XL pipeline. ZOOM_DEVICE
can be cpu, mps, or cuda; default auto-detects. GPU speed is not measured here.

## Geometry

The upload is center-cropped and resized to a 1024-square working frame. Each
click extracts a 1024-square context at the current pixel scale, replicating
pixels beyond the frame edges. Its central 512-square patch goes to diffusion
without any resize at that click. Insert the generated patch into the context
with border feathering, then enlarge the center by 2× using Lanczos for display
and the next click. The comparison views use the same framing.

## Border-only blending

Paste the generated 512×512 patch at [256:768,256:768] in the 1024×1024 context.
There are no pyramids, image blurs, or frequency corrections.

For distance d from the nearest patch edge, with border width w (default 32):

    t = clip(d / w, 0, 1)
    alpha = (1 - cos(pi * t)) / 2
    output = alpha * generated + (1-alpha) * original_context_patch

Only the border changes. The interior 448×448 pixels are copied directly from
the generation; everything outside the 512×512 patch is unchanged. Width 0 is
a hard paste. The raw comparison also starts with a hard paste, useful for checking
whether blur already exists in the generation. Next clicks use the composite.

A 450ms temporal fade reveals the result; the final image has no global fade.

## Prompts and first experiments

Choose a preset to populate the editable prompt; it stays the same across steps.
Presets are hypotheses to test, not demonstrated improvements. Concrete scene
content is more useful here than instructions such as "add more detail". A fixed
prompt encourages its named objects to recur; it cannot describe every semantic
scale or guarantee coherence.

- Living ecosystem: branching moss, fern fronds, mineral grains and water droplets.
- Nested architecture: courtyards, narrow stairs, carved arches, balconies, passages.
- World within a world: branching rivers around terraced gardens; more novelty,
  probably more drift.
- Material microstructure: fissures, crystalline grains, pores and mineral veins;
  generally a more conservative target than new miniature objects.

The default is three refinement steps with seed 42. Compare one, two, and three
steps on the same crop with the same prompt and seed. More steps start at a
higher noise level. Inspect Raw generation to see what the model produced.
Then compare border widths 0,16,32 with identical generation settings.

Avoid "bokeh", "shallow depth of field", and "tilt-shift" for a sharp zoom target.
Even "macro photograph" can encourage background blur. Try "crisp detail throughout"
or "deep focus" while naming specific structures. SD-Turbo uses guidance_scale=0;
negative prompts are not supported by this setup. Consider SDXL-Turbo if SD-Turbo
understands the desired scene poorly, accepting greater compute cost.

## Validation

```sh
python -m unittest discover -s tests -v
```

Border tests verify exact interior/exterior preservation, edge behavior, and zero-width
hard paste. JavaScript syntax is checked. Model generation and browser appearance
remain unverified here.

## Pre-noise sharpening and noise schedule

The 512px input patch now receives Pillow UnsharpMask (radius 1.5px, threshold 3,
amount 1.5 = 150%) before the pipeline's VAE encoder and latent noising. Amount 0
bypasses it. Blending still uses the untouched context. This boosts existing edge
contrast but cannot guarantee that diffusion preserves those edges.

The refinement dropdown selects the last K entries of the model's four-step
schedule. Diffusers adds noise at the first selected timestep and denoises
through all K entries. For SD-Turbo, three steps start at normalized time 0.75
(scheduler index 749), then use indices 499 and 249 before the clean endpoint.
The UI reports the actual step count and starting scheduler index.
