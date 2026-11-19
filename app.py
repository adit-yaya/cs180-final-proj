"""Local zoom server; the browser supplies bounded story context per request."""
import base64
import io
import threading
import time
from pathlib import Path

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field, model_validator
from typing import Literal, Tuple

from director import DirectorRequest, PromptDirector
from generator import Generator
from imaging import context_at, insert_center, raw_center, zoom_center, feather_patch

ROOT = Path(__file__).resolve().parent
generator = Generator()
model_lock = threading.Lock()
prompt_director = PromptDirector()

class Request(BaseModel):
    image: str = Field(max_length=8_000_000)
    center: Tuple[float, float]
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
        try:
            data = base64.b64decode(req.image.split(',')[-1], validate=True)
            with Image.open(io.BytesIO(data)) as source:
                if source.size != (1024,1024):
                    raise ValueError('Expected a 1024 × 1024 input image')
                context = context_at(source.convert('RGB'), req.center)
                baseline = context.crop((256,256,768,768))
        except (ValueError, OSError, UnidentifiedImageError) as exc:
            raise ValueError('Invalid input image') from exc
        start = time.perf_counter()
        try:
            raw, timing = generator.generate(baseline, req.prompt, req.refinement_steps, req.seed, req.backend, req.sharpen)
        except Exception as exc:
            raise RuntimeError(f'Generation failed: {exc}') from exc
        baseline_view = zoom_center(context)
        if req.backend == 'z-image':
            raw_view = raw
            blended_view = feather_patch(baseline_view, raw_view, 2 * req.feather)
        else:
            raw_view = zoom_center(raw_center(context, raw))
            blended_view = zoom_center(insert_center(context, raw, req.feather))
        return {'image': encode(blended_view), 'baseline': encode(baseline_view),
                'raw': encode(raw_view), 'seconds': time.perf_counter()-start, **timing}

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
        if self.path not in files:
            self.respond(404, b'Not found', 'text/plain')
            return
        name, mime = files[self.path]
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
                output = prompt_director.next(req)
            else:
                req = Request.model_validate_json(body)
                output = zoom(req)
            self.respond(200, json.dumps(output).encode(), 'application/json')
        except Exception as exc:
            status = 400 if isinstance(exc, ValueError) else 500
            self.respond(status, json.dumps({'detail': str(exc)}).encode(), 'application/json')

if __name__ == '__main__':
    print('Open http://127.0.0.1:8000', flush=True)
    ThreadingHTTPServer(('127.0.0.1', 8000), Handler).serve_forever()
