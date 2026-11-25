# Minimal generative zoom

## Run in Google Colab

1. Open [Google Colab](https://colab.research.google.com/), choose **File → Upload notebook**, and select [Generative_Zoom.ipynb](Generative_Zoom.ipynb) from this project. The notebook includes the Python app, browser UI, and dependency files. You can share just the notebook with a partner.
2. Choose **Runtime → Change runtime type**, select a GPU (start with a T4), save, and connect. Choose **Runtime → Run all**. The setup cell prints the assigned GPU, installs the dependencies, and the final code cell displays the demo.
3. Upload a photo and click a detail with **Wait for Qwen** selected under **Prompt mode**. The prompt hint is optional. In **Backend**, choose **SD-Turbo** for a T4 or **Z-Image-Turbo · 1024×1024** on an L4, A100, or another GPU with native BF16 support and enough VRAM. The first click downloads and loads the selected models. Later clicks reuse them.
4. Click the result to continue. Open **Prompt used for the last image** to see what Qwen sent to diffusion. **Back one zoom** restores the previous frame, and **Original image** restarts the exploration.
5. Keep the runtime connected while using the demo. When finished, choose **Runtime → Disconnect and delete runtime** to release the GPU. A fresh runtime downloads the weights again.

Colab runs Qwen3-VL-4B through Transformers with 4-bit weights and runs diffusion on the same GPU. Both models download directly from Hugging Face. The hosted demo uses Colab's built-in port proxy. For a quick UI check, select **Lanczos only** and click the image. If a larger image model exhausts GPU memory, rerun **Start the demo** and select SD-Turbo. To see server output, run `print(log.read_text()[-4000:])` in a new cell.

After editing the app, static assets, or requirements, run `python3 scripts/refresh_colab.py` and include the updated notebook when sharing your changes.

## Run locally

### Interactive OpenVINO mode on Windows

With Python set up, install the optional runtime once, then launch the app:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-openvino.txt
.\.venv\Scripts\python.exe app.py --realtime --warmup
```

The first warm-up downloads SD-Turbo's OpenVINO files into the ignored
`data/openvino` directory and prepares the local model. Later starts reuse
the downloaded and compiled files. Warm-up runs one draft locally to initialize
the rendering kernels before the first generated click. Open http://127.0.0.1:8000
as soon as the server prints the URL. You can open an image and use Lanczos
while the model prepares. The browser selects **SD-Turbo · OpenVINO** and starts each
click with a short zoom animation. It then blends in the generated image as
soon as it arrives. By default, the image model uses the crop and any prompt
hint you enter. Choose **Background Qwen** to add visual prompting without
making a click wait for it.

OpenVINO uses an available Intel GPU by default, including integrated graphics,
and falls back to CPU if GPU compilation fails. Set `ZOOM_OPENVINO_DEVICE=CPU`
to use the CPU explicitly. The status line reports the active device and
click-to-result time. The model renders at 384×384 and the app aligns the result
with the 1024×1024 source crop.

### PyTorch mode

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

On Windows with the project environment set up and Ollama running:

```powershell
.\.venv\Scripts\python.exe app.py --warmup
```

`--warmup` prepares the default diffusion model in the background. Qwen loads
when requested. Omit `--warmup` to load the image model on demand as well.
The app checks the existing Hugging Face cache first and downloads missing model
files as needed. Ollama stores its weights separately. After the first download,
both reuse the files on disk, although a fresh process still has to load them
into memory. Set `HF_HUB_OFFLINE=1` before launch for a cache-only diffusion run.
On CPU, set `ZOOM_CPU_THREADS` to override PyTorch's default thread count.

Open or drop an image, then click a point to generate. The outline previews
the exact crop. With the image focused, arrow keys move the target, Enter
zooms, and Backspace returns to the previous frame and restores the same target.
Change the settings and press Enter to compare another result on the exact crop.
The app keeps the six most recent frames in this browser tab, plus the original.

Choose a **Prompt mode** after opening an image.

- **Crop / manual prompt** uses the selected crop and your optional text. This is the local default.
- **Background Qwen** prepares a prompt when you pause over a target or click. A click renders immediately with any relevant prompt that's already ready. The app can reuse a prompt near its original target for up to two further 2× doublings. Moving outside that area, zooming past that scale, or changing the prompt settings makes the next eligible target request fresh analysis. The mode is available with local Ollama.
- **Wait for Qwen** reads the exact crop before rendering. Use it to compare the result of fresh visual prompting. This is the Colab default, where both models share one GPU.

A prompt hint can steer the style when it fits the crop. Zoom history stays in the browser tab. The seed controls both Qwen sampling and diffusion and stays fixed until edited.
Lanczos only skips both prompt writing and diffusion. The active diffusion
pipeline stays in memory until switching models or stopping the server.
Direct StableDiffusionImg2ImgPipeline import avoids unrelated Qwen imports.
ZOOM_MODEL=stabilityai/sdxl-turbo optionally selects the XL pipeline. ZOOM_DEVICE
can be cpu, mps, or cuda; default auto-detects. GPU speed is not measured here.

## Local detail director (Ollama)

Ollama must be installed and running (`ollama serve` if needed). Pull the small
local model once:

```sh
ollama pull qwen3-vl:4b-instruct
```

For a lighter CPU prompt model, pull `qwen3-vl:2b-instruct` and set
`$env:ZOOM_PROMPT_MODEL = "qwen3-vl:2b-instruct"` in PowerShell before launching
the app. The 4B model remains the default.

`./run.sh` starts the local Ollama service if necessary. Qwen runs on
`http://127.0.0.1:11434`; no cloud model or API key is used. Set
`ZOOM_PROMPT_MODEL` before launch to choose another installed Ollama model that
supports vision, structured output and non-thinking generation. Text-only models
are rejected explicitly; there is no silent text-only fallback.

- **Prompt hint** starts blank for each image. Qwen names a visible object or
  feature and proposes one primary refinement that preserves its identity and
  the crop's composition. Enter a hint if you want to guide the style or detail.
- **Detail variation** defaults to Medium. Gentle refines a visible feature,
  Medium may reveal one small hidden detail, and Surprising may reveal a more
  intricate structure. A hidden detail or a new path to explore must follow
  from a visible feature, such as a crack, seam, branch, or opening.
- Qwen sees one reference image with the unsharpened crop and a small close-up
  of the clicked feature side by side. It also receives the click position inside the crop,
  the current zoom factor and depth, the variation, and any prompt hint.
  Previous generated prompts and detail directions are omitted.
- Qwen returns factual crop observations, a short detail direction, and the
  standalone prompt sent to diffusion. The detail panel shows the latest
  direction. Open **Prompt used for the last image** to see the exact prompt
  and crop observations.
- The detail panel shows Qwen's proposed prompt as soon as it is ready, then
  keeps the prompt that was used for the frame.
- If Qwen is off or the prompt request fails, diffusion uses the crop and any
  manually entered prompt. The detail panel shows the reason for a failed request.

Image grounding improves prompt relevance, but does not guarantee object preservation
by diffusion. Refinement strength still controls how far the generated image may
change. Ollama 0.12.7 or newer is required for Qwen3-VL.

The status line reports model load, inference, and click-to-result times. It
shows Qwen time in **Wait for Qwen** mode and marks any reused background prompt.
Ollama and the renderer can run concurrently, so speed depends on whether they
compete for CPU, graphics, or memory on your machine.
Both models share the machine's memory. Diffusion uses
float16 on MPS/CUDA (float32 on CPU), and releases unused Metal allocations after
each generation to leave room for the local prompt writer.

## Geometry

The browser displays a 1024-square working frame. For uploads at least 1536px
on both sides, the first zoom uses a 2048-square reference of the original
image so the crop retains more real source detail. Subsequent zooms work from
the last generated frame. Each click selects a square based on **Zoom per click**. The default **2×** uses half
the frame's width and height, and **4×** uses one quarter. Near the edges, the crop
shifts inward to stay within the frame. The aim follows the selected point into
the new crop.
The generator upscales that crop with Lanczos to 512×512 for SD-Turbo or
1024×1024 for Z-Image-Turbo before sharpening and latent noising. All comparison
views share the crop, and the blended result becomes the next working frame.

The view eases into the exact crop in about a quarter of a second. When the
generation arrives, the aligned result fades in over roughly 160–310
milliseconds. Reduced-motion mode switches directly to the crop and then to
the completed frame.

## Structure preservation and feathering

**Preserve source structure** is enabled by default. The displayed image starts
from the source crop and transfers a bounded amount of new texture, strongest
near the click. Areas that the model smooths retain more fine source texture.
Changes to broad color and lighting are attenuated. Toggle it
off to use the original feathered composite. **Raw generation** always shows
the unmodified model output at display resolution.

The generated image and Lanczos baseline are aligned at 1024×1024. The border
width control is measured at 512px scale, so its default value of 32 blends the
outer 64 pixels in the output. For distance d from the nearest edge and output
border width w:

    t = clip(d / w, 0, 1)
    alpha = (1 - cos(pi * t)) / 2
    output = alpha * generated + (1 - alpha) * baseline

Width 0 skips feathering. The structure preservation toggle still applies.

## Prompts and first experiments

Leave the hint blank to start from the pixels. A preset fills the editable
hint for the current image. With Qwen enabled, the hint is optional context for
the prompt director. With Qwen off, the hint goes straight to diffusion.
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

The default is one refinement step with seed 42. Compare one, two, and three
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
node --test tests/prompting.test.mjs
```

The unit suite covers crop geometry, prompt inputs, image compositing,
warm-up availability, invalid requests, and recovery after generation errors.
The optional browser smoke test uses the real OpenVINO backend. Start the app
with `--realtime --warmup`, then run this from a second terminal with Node.js 22
and a Chromium browser:

```text
node tests/browser_smoke.cjs http://127.0.0.1:8000/ CHROMIUM_EXE IMAGE_FILE
```

For Node 20, pass `--experimental-websocket` before the script name. The test
reports click-to-result times and checks keyboard zoom, comparison,
history, mobile layout, and background-prompt timing, reuse, and invalidation.

## Pre-noise sharpening and noise schedule

Pre-noise sharpening is off by default. A nonzero amount applies Pillow
UnsharpMask (radius 1.5px, threshold 3) before VAE encoding and latent noising.
Use a small amount when the input needs more edge contrast.

The refinement dropdown selects the last K entries of the model's schedule (four steps for SD-Turbo
and eight for Z-Image-Turbo). Diffusers adds noise at the first selected timestep and denoises
through all K entries. For SD-Turbo, three steps start at normalized time 0.75
(scheduler index 749), then use indices 499 and 249 before the clean endpoint.
The UI reports the actual step count and starting scheduler index.
