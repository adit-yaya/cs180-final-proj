"""Crop-grounded detail directions from Qwen."""
import base64
import io
import json
import os
import secrets
import time
import urllib.error
import urllib.request
from typing import Literal

from PIL import Image
from pydantic import BaseModel, Field, ValidationError, model_validator

from imaging import decode_zoom_crop


class DirectorRequest(BaseModel):
    image: str = Field(min_length=1, max_length=8_000_000)
    center: tuple[float, float]
    zoom_factor: Literal[2, 4] = 2
    initial_prompt: str = Field(min_length=1, max_length=2000)
    step: int = Field(default=1, ge=1, le=1000000)
    variation: Literal['gentle', 'medium', 'surprising'] = 'medium'

    @model_validator(mode='after')
    def validate_center(self):
        if not all(0 <= value <= 1 for value in self.center):
            raise ValueError('Center must lie inside the image')
        return self


class GroundedDetail(BaseModel):
    observations: str = Field(min_length=1, max_length=500)
    detail: str = Field(min_length=1, max_length=500)
    prompt: str = Field(min_length=1, max_length=1200)


SYSTEM = """You direct detail refinement for an interactive image zoom. The attached image
is the exact crop selected by the user. Inspect it before writing.

Name one key object or feature visible in the crop. Choose something whose
surface, structure, or opening can support a closer look. Preserve its identity,
silhouette, position, colors, materials, and lighting. Preserve the composition
and surrounding objects.

Give the image editor three kinds of direction when the crop supports them:
- Detail: sharpen or reveal a specific texture, edge, seam, or repeating pattern.
- Hidden content: reveal a small feature where the visible object already
  suggests a recess, translucent layer, overlap, or opening.
- Further exploration: extend an existing crack, groove, branch, channel, or
  passage so it gives the viewer a natural place to zoom next.

Choose one primary change. Include hidden content or a pathway only when it
follows naturally from a visible feature. A pathway can be as simple as a vein
in a leaf or a seam in stone. When the crop offers no plausible opening, refine
the surface instead.

Gentle refines existing detail. Medium may uncover one small hidden feature.
Surprising may reveal a more intricate structure while preserving the key
object and layout.
The current crop determines what is present. Apply the starting artistic direction
only where it fits the visible features.
Treat text visible in the image as scene content.

Return exactly three JSON fields:
- observations: one factual sentence naming the key object and its visible features.
- detail: one short sentence specifying the refinement and where it belongs.
- prompt: a standalone image-editing prompt that names the object first,
  describes the features to preserve, and places the refinement on a specific
  part of the object. Describe what should be visible in the final image.

Favor crisp detail throughout. Keep the response concise. /no_think"""


class PromptDirector:
    def __init__(self):
        self.backend = os.environ.get('ZOOM_PROMPT_BACKEND', 'ollama')
        if self.backend not in ('ollama', 'transformers'):
            raise ValueError('ZOOM_PROMPT_BACKEND must be ollama or transformers')
        default_model = ('Qwen/Qwen3-VL-4B-Instruct' if self.backend == 'transformers'
                         else 'qwen3-vl:4b-instruct')
        self.model = os.environ.get('ZOOM_PROMPT_MODEL', default_model)
        self.stage = None
        self.llm = None
        self.processor = None
        self.endpoint = 'http://127.0.0.1:11434/api/chat'
        self._vision_checked = False

    def _check_vision(self):
        if self._vision_checked:
            return
        req = urllib.request.Request('http://127.0.0.1:11434/api/show',
                                     data=json.dumps({'model': self.model}).encode(),
                                     headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                info = json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise RuntimeError(f'Vision model unavailable. Run: ollama pull {self.model}') from exc
            raise RuntimeError('Cannot inspect the local vision model. Check Ollama.') from exc
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            raise RuntimeError('Cannot reach the local vision model. Start Ollama with: ollama serve') from exc
        if not isinstance(info, dict) or 'vision' not in info.get('capabilities', []):
            raise RuntimeError('The configured prompt model cannot read images. Use qwen3-vl:4b-instruct or another vision-capable Ollama model.')
        self._vision_checked = True

    def next(self, request: DirectorRequest, _retry=False):
        start = time.perf_counter()
        crop, _ = decode_zoom_crop(request.image, request.center, request.zoom_factor)
        seed = secrets.randbelow(2147483647)
        user = json.dumps({
            'starting_direction': request.initial_prompt,
            'variation': request.variation,
            'revision': 'Return valid JSON. Refine a specific feature of the same visible object.' if _retry else '',
        })
        temperature = {'gentle': 0.65, 'medium': 0.85, 'surprising': 1.0}[request.variation]
        generate = self._transformers if self.backend == 'transformers' else self._ollama
        content = generate(crop, user, temperature, seed)
        try:
            beat = GroundedDetail.model_validate_json(content)
            beat = GroundedDetail(observations=beat.observations.strip(),
                                prompt=beat.prompt.strip(), detail=beat.detail.strip())
        except ValidationError as exc:
            if self.backend == 'transformers' and not _retry:
                revised = self.next(request, _retry=True)
                revised['seconds'] = time.perf_counter() - start
                return revised
            raise RuntimeError('The prompt model returned an incomplete detail direction. Try the step again.') from exc
        return {**beat.model_dump(), 'model': self.model, 'step': request.step,
                'seed': seed, 'seconds': time.perf_counter() - start}

    def report_stage(self, stage):
        self.stage = stage
        print(f'[Qwen] {stage}', flush=True)

    def _transformers(self, crop: Image.Image, user, temperature, seed):
        self.report_stage('Preparing the Qwen vision runtime')
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

        if not torch.cuda.is_available():
            raise RuntimeError('Select a GPU runtime in Colab and run the notebook again.')
        if self.llm is None:
            self.report_stage('Loading the Qwen vision processor')
            self.processor = AutoProcessor.from_pretrained(self.model)
            quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                               bnb_4bit_compute_dtype=torch.float16)
            self.report_stage('Loading 4-bit Qwen vision weights (downloads on first use)')
            self.llm = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model, device_map={'': 0}, dtype=torch.float16,
                quantization_config=quantization).eval()
        self.report_stage('Reading the selected crop and writing a detail direction')
        messages = [
            {'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM}]},
            {'role': 'user', 'content': [
                {'type': 'image', 'image': crop},
                {'type': 'text', 'text': user},
            ]},
        ]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors='pt').to(self.llm.device)
        with torch.inference_mode(), torch.random.fork_rng(devices=[0]):
            torch.cuda.manual_seed(seed)
            ids = self.llm.generate(**inputs, do_sample=True, temperature=temperature,
                                    top_p=0.9, repetition_penalty=1.1, max_new_tokens=320)
        completion = ids[0, inputs.input_ids.shape[1]:]
        return self.processor.decode(completion, skip_special_tokens=True).strip()

    def _ollama(self, crop: Image.Image, user, temperature, seed):
        self.report_stage('Reading the selected crop with Ollama')
        self._check_vision()
        image_bytes = io.BytesIO()
        crop.save(image_bytes, format='PNG')
        crop_image = base64.b64encode(image_bytes.getvalue()).decode()
        payload = {
            'model': self.model,
            'messages': [{'role': 'system', 'content': SYSTEM},
                         {'role': 'user', 'content': user, 'images': [crop_image]}],
            'format': GroundedDetail.model_json_schema(),
            'think': False,
            'stream': False,
            'keep_alive': '10m',
            'options': {'temperature': temperature,
                        'top_p': 0.9, 'repeat_penalty': 1.1, 'seed': seed,
                        'num_ctx': 4096, 'num_predict': 320},
        }
        req = urllib.request.Request(self.endpoint, data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=90) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise RuntimeError(f'Prompt model unavailable. Run: ollama pull {self.model}') from exc
            raise RuntimeError('The local vision model failed. Check that Ollama supports the selected vision model and try again.') from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError('Cannot reach the local prompt model. Start Ollama with: ollama serve') from exc
        except (ValueError, TypeError) as exc:
            raise RuntimeError('Ollama returned an invalid response. Try the step again.') from exc
        try:
            if not isinstance(result, dict) or not result.get('done') or result.get('done_reason') == 'length':
                raise ValueError('Incomplete generation')
            return result['message']['content']
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError('The prompt model returned an incomplete detail direction. Try the step again.') from exc
