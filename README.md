# Minimal generative zoom

## Run in Google Colab

1. Open [Google Colab](https://colab.research.google.com/), choose **File → Upload notebook**, and select [Generative_Zoom.ipynb](Generative_Zoom.ipynb) from this project. The notebook includes the Python app, browser UI, and dependency files. You can share just the notebook with a partner.
2. Choose **Runtime → Change runtime type**, select a GPU (start with a T4), save, and connect. Choose **Runtime → Run all**. The setup cell prints the assigned GPU, installs the dependencies, and the final code cell displays the demo.
3. Upload a photo in the demo, enter a starting direction or choose a preset, and leave **Evolve prompt each zoom** enabled. Select **Diffusion · SD-Turbo** and click a detail in the image. The first click downloads the weights and loads both models. Later clicks reuse them.
4. Click the result to continue. The story panel shows each discovery, and **Prompt used for the last image** shows what Qwen sent to diffusion. Use **Restart story** to clear the text history while keeping the current image.
5. Keep the runtime connected while using the demo. When finished, choose **Runtime → Disconnect and delete runtime** to release the GPU. A fresh runtime downloads the weights again.

Colab runs Qwen3-VL-4B through Transformers with 4-bit weights and runs diffusion on the same GPU. Both models download directly from Hugging Face. The hosted demo uses Colab's built-in port proxy. For a quick UI check, select **Lanczos only** and click the image. If a larger image model exhausts GPU memory, rerun **Start the demo** and select SD-Turbo. To see server output, run `print(log.read_text()[-4000:])` in a new cell.

After editing the app, static assets, or requirements, run `python3 scripts/refresh_colab.py` and include the updated notebook when sharing your changes.

## Run locally

From the project directory on a Mac with the local environment installed, run:

```sh
./run.sh
```

Open http://127.0.0.1:8000. Press Ctrl-C in Terminal to stop the server.
Restart the server and reload the page after changes.

To install on a fresh machine, create the environment first:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-model.txt
./run.sh
```

The launcher prefers `.venv/bin/python`, falling back to `python3` if no local
virtual environment exists. Model weights download on first use into the normal
Hugging Face cache; GPU selection is automatic (MPS on Apple Silicon).

Upload an image; click a point to generate. With **Evolve prompt each zoom**
enabled, local Qwen3-VL-4B inspects the clicked crop and writes a new prompt and short
story beat before diffusion.
The starting prompt remains the anchor; the browser retains the last three text
beats. No image history or generated-image disk output is kept. The diffusion
seed stays fixed until edited. Lanczos only skips both prompt writing and diffusion. SD-Turbo downloads on its first diffusion request and stays in memory.
Direct StableDiffusionImg2ImgPipeline import avoids unrelated Qwen imports.
ZOOM_MODEL=stabilityai/sdxl-turbo optionally selects the XL pipeline. ZOOM_DEVICE
can be cpu, mps, or cuda; default auto-detects. GPU speed is not measured here.

## Local story director (Ollama)

Ollama must be installed and running (`ollama serve` if needed). Pull the small
local model once:

```sh
ollama pull qwen3-vl:4b-instruct
```

`./run.sh` starts the local Ollama service if necessary. Qwen runs on
`http://127.0.0.1:11434`; no cloud model or API key is used. Set
`ZOOM_PROMPT_MODEL` before launch to choose another installed Ollama model that
supports vision, structured output and non-thinking generation. Text-only models
are rejected explicitly; there is no silent text-only fallback.

- **Starting prompt / story direction** stays constant until you edit it.
- **Story variation** defaults to Medium; Gentle changes less, Surprising permits
  larger departures. The current crop takes priority over the initial theme and
  previous story; creative cues are optional and must fit the visible content.
- Each step sends the current 1024px frame and click coordinates to the backend.
  The backend extracts the same unsharpened 512px patch used by diffusion and
  supplies its lossless PNG to Qwen through Ollama's image input. It also supplies
  the initial direction and last three discoveries. The image is inspected again
  for every click, so moving to a new subject can override the earlier story.
- Qwen first writes factual crop observations (10-15 words), then a short story
  sentence (10-15 words) and a standalone diffusion prompt (30-40 words). These are
  prompt-level length targets, not hard truncation rules. Thinking is off; token
  sampling uses the selected temperature. Near-duplicate scenes
  trigger one retry; invalid or unavailable results stop the step with an error.
- Story context advances only after the resulting image succeeds. Uploading a
  new image, editing the starting prompt, or pressing **Restart story** resets it.
  Restarting the story keeps the current image. Disabling evolution uses the exact
  fixed prompt for controlled comparisons.
- The story panel shows the latest discovery and actual prompt used. The prompt
  dropdown starts collapsed and closes after each successful step or story reset.
  It includes the visible crop observations for checking what guided the prompt.

Image grounding improves prompt relevance, but does not guarantee object preservation
by diffusion. Refinement strength still controls how far the generated image may
change. Ollama 0.12.7 or newer is required for Qwen3-VL.

Prompt writing adds latency before diffusion. Its sampling is randomized independently
of the diffusion seed, so different prompts can produce different images even with a
fixed image seed. Both models share the Mac's memory; SD-Turbo is the tested default. Diffusion uses
float16 on MPS/CUDA (float32 on CPU), and releases unused Metal allocations after
each generation to leave room for the local prompt writer.

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

Choose a preset to populate the editable starting prompt; it stays the same across steps.
With evolution enabled, this is context for the prompt director rather than the
exact prompt sent to diffusion.
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
