"""Lazy model loading: the baseline needs neither torch nor model downloads."""
import os
import time
from pathlib import Path
from imaging import sharpen_input
from PIL import Image

Z_IMAGE = "Tongyi-MAI/Z-Image-Turbo"
OPENVINO_MODEL = 'rupeshs/sd-turbo-openvino'
OPENVINO_ROOT = Path(__file__).resolve().parent / 'data' / 'openvino'
OPENVINO_SCHEDULE = 4


class OpenVinoZoom:
    """Keep a compiled image-to-image pipeline for the local interactive path."""

    def __init__(self):
        self.pipe = None
        self.size = 384

    def warmup(self):
        if self.pipe is not None:
            return 0
        start = time.perf_counter()
        try:
            from openvino import Core
            from openvino_genai import Image2ImagePipeline
        except ImportError as exc:
            raise RuntimeError('Install the local OpenVINO runtime: pip install -r requirements-openvino.txt') from exc
        from huggingface_hub import snapshot_download

        model = OPENVINO_ROOT / 'sd-turbo'
        parts = ('text_encoder', 'unet', 'vae_encoder', 'vae_decoder')
        required = ['model_index.json', 'scheduler/scheduler_config.json',
                    'tokenizer/tokenizer_config.json', 'tokenizer/merges.txt', 'tokenizer/vocab.json']
        required.extend(f'{part}/{name}' for part in parts
                        for name in ('config.json', 'openvino_model.xml', 'openvino_model.bin'))
        if not all((model / name).is_file() for name in required):
            print('Downloading the OpenVINO SD-Turbo checkpoint to data/openvino...', flush=True)
            snapshot_download(OPENVINO_MODEL, revision='b5582a9', local_dir=model,
                              allow_patterns=['*.json', '*.xml', '*.bin', 'tokenizer/*.txt'], max_workers=4)

        tokenizer = model / 'tokenizer'
        tokenizer_ready = tokenizer / '.zoom-tokenizer-i32'
        if not tokenizer_ready.is_file():
            from openvino import Type, save_model
            from openvino_tokenizers import convert_tokenizer
            from transformers import CLIPTokenizerFast

            clip = CLIPTokenizerFast.from_pretrained(tokenizer, local_files_only=True)
            converted = convert_tokenizer(clip, tokenizer_output_type=Type.i32, use_max_padding=True)
            save_model(converted, tokenizer / 'openvino_tokenizer.xml')
            tokenizer_ready.touch()

        preference = os.environ.get('ZOOM_OPENVINO_DEVICE', 'AUTO').upper()
        device = preference
        if device == 'AUTO':
            device = 'GPU' if any(name.split('.')[0] == 'GPU' for name in Core().available_devices) else 'CPU'

        def compile_pipeline(target):
            print(f'Preparing OpenVINO SD-Turbo ({self.size}px, {target})...', flush=True)
            pipeline = Image2ImagePipeline(str(model))
            pipeline.reshape(1, self.size, self.size, 0.0)
            pipeline.compile(target, CACHE_DIR=str(OPENVINO_ROOT / 'compiled'), PERFORMANCE_HINT='LATENCY')
            return pipeline

        try:
            self.pipe = compile_pipeline(device)
        except RuntimeError as exc:
            if preference != 'AUTO' or device != 'GPU':
                raise
            print(f'OpenVINO GPU unavailable, using CPU: {exc}', flush=True)
            device = 'CPU'
            self.pipe = compile_pipeline(device)
        self.device = device
        return time.perf_counter() - start

    def generate(self, image, prompt, steps, seed, sharpen):
        load_time = self.warmup()
        import numpy as np
        from openvino import Tensor

        start = time.perf_counter()
        image = image.resize((self.size, self.size), Image.Resampling.LANCZOS)
        image = sharpen_input(image, sharpen)
        tensor = Tensor(np.asarray(image)[None, ...].copy())
        result = self.pipe.generate(prompt, tensor, height=self.size, width=self.size,
                                    num_inference_steps=OPENVINO_SCHEDULE, strength=steps / OPENVINO_SCHEDULE,
                                    guidance_scale=0.0, rng_seed=seed)
        return Image.fromarray(result.data[0].copy()), {
            'model': OPENVINO_MODEL, 'device': f'OpenVINO {self.device}',
            'load_seconds': load_time, 'denoising_steps': steps,
            'start_timestep': 250 * steps - 1, 'inference_seconds': time.perf_counter() - start,
        }


class Generator:
    def __init__(self):
        self.pipe = None
        self.loaded_model = None
        self.openvino = OpenVinoZoom()
        self.model_id = os.environ.get('ZOOM_MODEL', 'stabilityai/sd-turbo')
        if self.model_id not in ('stabilityai/sd-turbo', 'stabilityai/sdxl-turbo'):
            raise ValueError('ZOOM_MODEL must be stabilityai/sd-turbo or stabilityai/sdxl-turbo')

    def _load_pipeline(self, model_id):
        if self.loaded_model == model_id:
            return 0
        self.pipe = None
        self.loaded_model = None
        start = time.perf_counter()
        import torch
        from huggingface_hub.constants import HF_HUB_OFFLINE

        # Avoid AutoPipeline: it eagerly imports unrelated Qwen pipelines.
        if model_id == Z_IMAGE:
            from diffusers import ZImageImg2ImgPipeline as Pipeline
        elif model_id == 'stabilityai/sdxl-turbo':
            from diffusers import StableDiffusionXLImg2ImgPipeline as Pipeline
        else:
            from diffusers import StableDiffusionImg2ImgPipeline as Pipeline

        device = os.environ.get('ZOOM_DEVICE', 'auto')
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        dtype = torch.float16 if device in ('cuda', 'mps') else torch.float32
        kwargs = {'torch_dtype': dtype}
        if model_id == Z_IMAGE and device != 'cpu':
            kwargs['torch_dtype'] = torch.bfloat16
        if dtype == torch.float16 and model_id != Z_IMAGE:
            kwargs['variant'] = 'fp16'
        if device == 'cpu' and 'ZOOM_CPU_THREADS' in os.environ:
            torch.set_num_threads(int(os.environ['ZOOM_CPU_THREADS']))

        print(f'Loading {model_id} on {device} from the local cache...', flush=True)
        try:
            pipe = Pipeline.from_pretrained(model_id, local_files_only=True, **kwargs)
        except OSError:
            if HF_HUB_OFFLINE:
                raise
            print(f'Fetching missing model files for {model_id}...', flush=True)
            pipe = Pipeline.from_pretrained(model_id, **kwargs)
        pipe.to(device)
        if model_id != Z_IMAGE:
            pipe.enable_attention_slicing()
        self.pipe = pipe
        self.loaded_model = model_id
        return time.perf_counter() - start

    def warmup(self, backend='diffusion'):
        """Load the default diffusion pipeline before the first click."""
        if backend == 'openvino':
            if self.openvino.pipe is None:
                start = time.perf_counter()
                image = Image.new('RGB', (self.openvino.size, self.openvino.size), (120, 125, 130))
                self.openvino.generate(image, 'Natural detail', steps=1, seed=42, sharpen=0)
                return time.perf_counter() - start
            return 0
        return self._load_pipeline(self.model_id)

    def generate(self, image, prompt, refinement_steps, seed, backend, sharpen=0):
        if backend == 'upscale':
            return image.copy(), {'model': 'Lanczos baseline', 'load_seconds': 0, 'inference_seconds': 0, 'denoising_steps': 0, 'start_timestep': None}
        is_z_image = backend == 'z-image'
        model_id = Z_IMAGE if is_z_image else self.model_id
        schedule_steps = 8 if is_z_image else 4
        if not 1 <= refinement_steps <= schedule_steps:
            raise ValueError(f'Refinement steps must be between 1 and {schedule_steps}')
        if backend == 'openvino':
            self.pipe = self.loaded_model = None
            return self.openvino.generate(image, prompt, refinement_steps, seed, sharpen)
        self.openvino.pipe = None
        load_time = self._load_pipeline(model_id)
        import torch

        start = time.perf_counter()
        input_size = 1024 if is_z_image else 512
        image = image.resize((input_size, input_size), Image.Resampling.LANCZOS)
        image = sharpen_input(image, sharpen)
        used_timesteps = []
        def record_step(pipe, index, timestep, callback_kwargs):
            used_timesteps.append(float(timestep))
            return callback_kwargs
        # CPU RNG also works with MPS. Each request owns its seed.
        result = self.pipe(prompt=prompt, image=image, strength=refinement_steps / schedule_steps,
                           num_inference_steps=schedule_steps, guidance_scale=0.0,
                           generator=torch.Generator(device='cpu').manual_seed(seed),
                           callback_on_step_end=record_step)
        if getattr(result, 'nsfw_content_detected', None) and any(result.nsfw_content_detected):
            raise ValueError('The model filtered this result; try a different crop or prompt.')
        # Release unused temporary Metal allocations before the next local LLM call.
        if self.pipe.device.type == 'mps':
            torch.mps.empty_cache()
        return result.images[0].convert('RGB'), {
            'model': model_id, 'load_seconds': load_time,
            'denoising_steps': len(used_timesteps),
            'start_timestep': used_timesteps[0] if used_timesteps else None,
            'inference_seconds': time.perf_counter()-start}
