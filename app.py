"""Local zoom server; the browser supplies bounded refinement context per request."""
import argparse
import base64
import io
import threading
import time
from importlib.util import find_spec
from pathlib import Path

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from PIL import Image
from pydantic import BaseModel, Field, ValidationError, model_validator
from typing import Literal, Tuple

from director import DirectorRequest, PromptDirector
from generator import Generator
from imaging import FRAME_SIZE, crop_focus, decode_zoom_crop, feather_patch, preserve_source_detail

ROOT = Path(__file__).resolve().parent
generator = Generator()
model_lock = threading.Lock()
ollama_lock = threading.Lock()
prompt_director = PromptDirector()
default_backend = 'diffusion'
warmup_status = {'state': 'idle', 'message': ''}
STATIC_FILES = {
    '/': ('index.html', 'text/html; charset=utf-8'),
    '/static/app.js': ('app.js', 'text/javascript; charset=utf-8'),
    '/static/prompting.mjs': ('prompting.mjs', 'text/javascript; charset=utf-8'),
    '/static/style.css': ('style.css', 'text/css; charset=utf-8'),
}


class ModelBusyError(RuntimeError):
    pass


def acquire_model():
    if warmup_status['state'] == 'warming' or not model_lock.acquire(blocking=False):
        raise ModelBusyError('The model is busy. You can use Lanczos while it finishes.')


def start_warmup():
    """Prepare the chosen models without blocking the page or the Lanczos path."""
    global warmup_status
    warmup_status = {'state': 'warming', 'message': 'Preparing the image model…'}

    def prepare():
        global warmup_status
        try:
            with model_lock:
                generator.warmup(default_backend)
        except Exception as exc:
            print(f'Model warm-up failed: {exc}', flush=True)
            warmup_status = {'state': 'failed', 'message': str(exc)}
        else:
            warmup_status = {'state': 'ready', 'message': 'Image model ready.'}

    worker = threading.Thread(target=prepare, daemon=True, name='model-warmup')
    worker.start()
    return worker


def direct(req: DirectorRequest):
    # Ollama runs in a separate process. Colab's Qwen and diffusion share one GPU.
    lock = ollama_lock if prompt_director.backend == 'ollama' else model_lock
    if lock is model_lock:
        acquire_model()
    elif not lock.acquire(blocking=False):
        raise ModelBusyError('Ollama is already reading a crop. Try again shortly.')
    try:
        return prompt_director.next(req)
    finally:
        prompt_director.stage = None
        lock.release()


class Request(BaseModel):
    image: str = Field(max_length=8_000_000)
    center: Tuple[float, float]
    zoom_factor: Literal[2, 4] = 2
    prompt: str = Field(default='', max_length=2000)
    backend: Literal['upscale', 'diffusion', 'z-image', 'openvino'] = 'diffusion'
    refinement_steps: int = Field(default=1, ge=1, le=8)
    seed: int = Field(default=42, ge=0, le=2147483647)
    sharpen: float = Field(default=0, ge=0, le=3)
    feather: int = Field(default=32, ge=0, le=128)
    preserve_structure: bool = True

    @model_validator(mode='after')
    def validate_settings(self):
        if not all(0 <= v <= 1 for v in self.center):
            raise ValueError('Center must lie inside the image')
        if self.backend in ('diffusion', 'openvino') and self.refinement_steps > 4:
            raise ValueError("SD-Turbo supports up to four refinement steps")
        return self

def encode(image, compress_level=6):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG', compress_level=compress_level)
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()

def zoom(req: Request):
    start = time.perf_counter()
    crop, bounds = decode_zoom_crop(req.image, req.center, req.zoom_factor)
    output_size = (FRAME_SIZE, FRAME_SIZE)
    if req.backend == 'upscale':
        view = encode(crop.resize(output_size, Image.Resampling.LANCZOS))
        return {'image': view, 'baseline': view, 'raw': view, 'crop': bounds,
                'seconds': time.perf_counter() - start, 'model': 'Lanczos baseline',
                'load_seconds': 0, 'inference_seconds': 0, 'denoising_steps': 0, 'start_timestep': None}

    acquire_model()
    try:
        try:
            raw, timing = generator.generate(crop, req.prompt, req.refinement_steps, req.seed, req.backend, req.sharpen)
        except Exception as exc:
            raise RuntimeError(f'Generation failed: {exc}') from exc
    finally:
        model_lock.release()

    baseline_view = crop.resize(output_size, Image.Resampling.LANCZOS)
    raw_view = raw.resize(output_size, Image.Resampling.LANCZOS)
    edited_view = raw_view
    if req.preserve_structure:
        edited_view = preserve_source_detail(baseline_view, raw_view, crop_focus(req.center, bounds))
    blended = feather_patch(baseline_view, edited_view, 2 * req.feather)
    compression = 1 if req.backend == 'openvino' else 6
    return {'image': encode(blended, compression), 'baseline': encode(baseline_view, compression),
            'raw': encode(raw_view, compression), 'crop': bounds,
            'seconds': time.perf_counter() - start, **timing}


class Handler(BaseHTTPRequestHandler):
    def log_request(self, code='-', size='-'):
        polling = self.path.partition('?')[0] in ('/api/config', '/api/director/status')
        if not (polling and code == 200):
            super().log_request(code, size)

    def respond(self, status, body, content_type):
        try:
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            if content_type == 'application/json':
                self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # A browser can leave a request while the model is still computing.
            self.close_connection = True

    def do_GET(self):
        path = self.path.partition('?')[0]
        if path == '/api/config':
            config = {'realtime': default_backend == 'openvino',
                      'openvino_available': find_spec('openvino_genai') is not None,
                      'background_prompting': prompt_director.backend == 'ollama',
                      'warmup': warmup_status}
            self.respond(200, json.dumps(config).encode(), 'application/json')
            return
        if path == '/api/director/status':
            self.respond(200, json.dumps({'stage': prompt_director.stage}).encode(), 'application/json')
            return
        if path == '/favicon.ico':
            self.respond(204, b'', 'image/x-icon')
            return
        if path not in STATIC_FILES:
            self.respond(404, b'Not found', 'text/plain')
            return
        name, mime = STATIC_FILES[path]
        self.respond(200, (ROOT/'static'/name).read_bytes(), mime)

    def do_POST(self):
        if self.path not in ('/api/zoom', '/api/director'):
            self.respond(404, b'Not found', 'text/plain')
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 8_100_000:
                raise ValueError('Invalid request size')
            body = self.rfile.read(length)
            if self.path == '/api/director':
                req = DirectorRequest.model_validate_json(body)
                output = direct(req)
            else:
                req = Request.model_validate_json(body)
                output = zoom(req)
            self.respond(200, json.dumps(output).encode(), 'application/json')
        except ModelBusyError as exc:
            self.respond(409, json.dumps({'detail': str(exc)}).encode(), 'application/json')
        except ValidationError as exc:
            error = exc.errors(include_url=False, include_input=False)[0]
            field = ' '.join(str(part).replace('_', ' ') for part in error['loc'])
            message = error['msg'].removeprefix('Value error, ')
            detail = f'{field.capitalize()} — {message}' if field else message
            self.respond(400, json.dumps({'detail': detail}).encode(), 'application/json')
        except Exception as exc:
            print(f'{self.path} failed: {exc}', flush=True)
            status = 400 if isinstance(exc, ValueError) else 500
            self.respond(status, json.dumps({'detail': str(exc)}).encode(), 'application/json')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the generative zoom server.')
    parser.add_argument('--warmup', action='store_true',
                        help='Prepare the selected pipeline in the background on startup.')
    parser.add_argument('--realtime', action='store_true',
                        help='Prefer the local OpenVINO backend in the browser.')
    args = parser.parse_args()
    if args.realtime:
        default_backend = 'openvino'
    with ThreadingHTTPServer(('127.0.0.1', 8000), Handler) as server:
        if args.warmup:
            start_warmup()
        print('Open http://127.0.0.1:8000', flush=True)
        server.serve_forever()
