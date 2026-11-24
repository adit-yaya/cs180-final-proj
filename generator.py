"""Lazy model loading: the baseline needs neither torch nor model downloads."""
import os
import time
from imaging import sharpen_input
from PIL import Image

Z_IMAGE = "Tongyi-MAI/Z-Image-Turbo"

class Generator:
    def __init__(self):
        self.pipe = None
        self.model_id = os.environ.get('ZOOM_MODEL', 'stabilityai/sd-turbo')
        if self.model_id not in ('stabilityai/sd-turbo', 'stabilityai/sdxl-turbo'):
            raise ValueError('ZOOM_MODEL must be stabilityai/sd-turbo or stabilityai/sdxl-turbo')

    def generate(self, image, prompt, refinement_steps, seed, backend, sharpen=1.5):
        if backend == 'upscale':
            return image.copy(), {'model': 'Lanczos baseline', 'load_seconds': 0, 'inference_seconds': 0, 'denoising_steps': 0, 'start_timestep': None}
        import torch
        is_z_image = backend == 'z-image'
        model_id = Z_IMAGE if is_z_image else self.model_id
        schedule_steps = 8 if is_z_image else 4
        if not 1 <= refinement_steps <= schedule_steps:
            raise ValueError(f'Refinement steps must be between 1 and {schedule_steps}')
        if self.pipe is not None and self.loaded_model != model_id:
            self.pipe = None
        # Avoid AutoPipeline: it eagerly imports unrelated Qwen pipelines.
        if is_z_image:
            from diffusers import ZImageImg2ImgPipeline as Pipeline
        elif model_id == 'stabilityai/sdxl-turbo':
            from diffusers import StableDiffusionXLImg2ImgPipeline as Pipeline
        else:
            from diffusers import StableDiffusionImg2ImgPipeline as Pipeline
        load_time = 0
        if self.pipe is None:
            start = time.perf_counter()
            device = os.environ.get('ZOOM_DEVICE', 'auto')
            if device == 'auto':
                device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
            dtype = torch.float16 if device in ('cuda', 'mps') else torch.float32
            kwargs = {'torch_dtype': dtype}
            if is_z_image and device != 'cpu':
                kwargs['torch_dtype'] = torch.bfloat16
            if dtype == torch.float16 and not is_z_image:
                kwargs['variant'] = 'fp16'
            pipe = Pipeline.from_pretrained(model_id, **kwargs)
            pipe.to(device)
            if not is_z_image:
                pipe.enable_attention_slicing()
            self.pipe = pipe
            self.loaded_model = model_id
            load_time = time.perf_counter()-start
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
