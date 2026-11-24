"""Local zoom server; the browser supplies bounded refinement context per request."""
import base64
import io
import threading
import time
from pathlib import Path

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from PIL import Image
from pydantic import BaseModel, Field, model_validator
from typing import Literal, Tuple

from director import DirectorRequest, PromptDirector
from generator import Generator
from imaging import FRAME_SIZE, decode_zoom_crop, feather_patch

ROOT = Path(__file__).resolve().parent
generator = Generator()
model_lock = threading.Lock()
prompt_director = PromptDirector()

class Request(BaseModel):
    image: str = Field(max_length=8_000_000)
    center: Tuple[float, float]
    zoom_factor: Literal[2, 4] = 2
    prompt: str = Field(max_length=2000)
    backend: Literal['upscale', 'diffusion', 'z-image'] = 'diffusion'
    refinement_steps: int = Field(default=3, ge=1, le=8)
    seed: int = Field(default=42, ge=0, le=2147483647)
    sharpen: float = Field(default=1.5, ge=0, le=3)
    feather: int = Field(default=32, ge=0, le=128)

    @model_validator(mode='after')
    def validate_settings(self):
        if not all(0 <= v <= 1 for v in self.center):
            raise ValueError('Center must lie inside the image')
        if self.backend == "diffusion" and self.refinement_steps > 4:
            raise ValueError("SD-Turbo supports up to four refinement steps")
        return self

def encode(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()

def zoom(req: Request):
    if not model_lock.acquire(blocking=False):
        raise ValueError('A generation is already running')
    try:
        crop, bounds = decode_zoom_crop(req.image, req.center, req.zoom_factor)
        start = time.perf_counter()
        try:
            raw, timing = generator.generate(crop, req.prompt, req.refinement_steps, req.seed, req.backend, req.sharpen)
        except Exception as exc:
            raise RuntimeError(f'Generation failed: {exc}') from exc
        output_size = (FRAME_SIZE, FRAME_SIZE)
        baseline_view = crop.resize(output_size, Image.Resampling.LANCZOS)
        raw_view = raw.resize(output_size, Image.Resampling.LANCZOS)
        blended_view = feather_patch(baseline_view, raw_view, 2 * req.feather)
        return {'image': encode(blended_view), 'baseline': encode(baseline_view),
                'raw': encode(raw_view), 'crop': bounds,
                'seconds': time.perf_counter()-start, **timing}

    finally:
        model_lock.release()


class Handler(BaseHTTPRequestHandler):
    def respond(self, status, body, content_type):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        files = {'/': ('index.html', 'text/html; charset=utf-8'),
                 '/static/app.js': ('app.js', 'text/javascript; charset=utf-8')}
        path = self.path.partition('?')[0]
        if path == '/api/director/status':
            self.respond(200, json.dumps({'stage': prompt_director.stage}).encode(), 'application/json')
            return
        if path not in files:
            self.respond(404, b'Not found', 'text/plain')
            return
        name, mime = files[path]
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
                if not model_lock.acquire(blocking=False):
                    raise ValueError('A generation is already running')
                try:
                    output = prompt_director.next(req)
                finally:
                    prompt_director.stage = None
                    model_lock.release()
            else:
                req = Request.model_validate_json(body)
                output = zoom(req)
            self.respond(200, json.dumps(output).encode(), 'application/json')
        except Exception as exc:
            print(f'{self.path} failed: {exc}', flush=True)
            status = 400 if isinstance(exc, ValueError) else 500
            self.respond(status, json.dumps({'detail': str(exc)}).encode(), 'application/json')

if __name__ == '__main__':
    print('Open http://127.0.0.1:8000', flush=True)
    ThreadingHTTPServer(('127.0.0.1', 8000), Handler).serve_forever()
