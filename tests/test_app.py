"""Exercise the real HTTP routes without downloading or running a model."""
import base64
import http.client
import io
import json
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

from PIL import Image

import app


def data_image(value):
    return Image.open(io.BytesIO(base64.b64decode(value.split(',', 1)[1])))


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = app.ThreadingHTTPServer(('127.0.0.1', 0), app.Handler)
        cls.worker = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.worker.start()
        image = Image.new('RGB', (1024, 1024), 'red')
        image.paste('blue', (512, 0, 1024, 1024))
        cls.source = app.encode(image)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.worker.join(timeout=5)

    def request(self, method, path, data=None, headers=None):
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=5)
        try:
            connection.request(method, path, body=data, headers=headers or {})
            response = connection.getresponse()
            body = response.read()
            return response.status, response.headers, body
        finally:
            connection.close()

    def post(self, path='/api/zoom', **overrides):
        payload = {'image': self.source, 'center': [1, 0], 'zoom_factor': 4,
                   'prompt': 'Fine surface detail.', **overrides}
        status, _, body = self.request('POST', path, json.dumps(payload).encode(),
                                       {'Content-Type': 'application/json'})
        return status, json.loads(body)

    def test_static_assets_and_config_are_served(self):
        for path, content_type, marker in (
            ('/', 'text/html', b'Generative Zoom'),
            ('/static/app.js', 'text/javascript', b'cropBounds'),
            ('/static/prompting.mjs', 'text/javascript', b'promptApplies'),
            ('/static/style.css', 'text/css', b'.crop-guide'),
        ):
            with self.subTest(path=path):
                status, headers, body = self.request('GET', path)
                self.assertEqual(status, 200)
                self.assertIn(content_type, headers['Content-Type'])
                self.assertIn(marker, body)
        self.assertEqual(self.request('GET', '/favicon.ico')[0], 204)
        self.assertEqual(self.request('GET', '/missing')[0], 404)
        status, headers, body = self.request('GET', '/api/config?refresh=1')
        self.assertEqual(status, 200)
        self.assertEqual(headers['Cache-Control'], 'no-store')
        self.assertIn('state', json.loads(body)['warmup'])

    def test_ollama_can_analyze_a_crop_while_diffusion_renders(self):
        started = threading.Event()
        release = threading.Event()

        def read_crop(_):
            started.set()
            if not release.wait(5):
                raise RuntimeError('test did not release Ollama')
            return {'prompt': 'A fine texture.'}

        output = Image.new('RGB', (256, 256), 'green')
        with (ThreadPoolExecutor(max_workers=1) as executor,
              patch.object(app.prompt_director, 'backend', 'ollama'),
              patch.object(app.prompt_director, 'next', side_effect=read_crop),
              patch.object(app.generator, 'generate', return_value=(output, {}))):
            pending = executor.submit(self.post, '/api/director')
            try:
                self.assertTrue(started.wait(2))
                self.assertEqual(self.post('/api/director')[0], 409)
                self.assertEqual(self.post(backend='openvino')[0], 200)
                self.assertFalse(pending.done())
            finally:
                release.set()
            self.assertEqual(pending.result(timeout=5), (200, {'prompt': 'A fine texture.'}))

    def test_colab_prompting_still_uses_the_shared_model_lock(self):
        with (patch.object(app.prompt_director, 'backend', 'transformers'),
              patch.object(app.prompt_director, 'next') as director,
              app.model_lock):
            self.assertEqual(self.post('/api/director')[0], 409)
            director.assert_not_called()

    def test_lanczos_is_available_while_model_warms(self):
        started = threading.Event()
        release = threading.Event()

        def warm(_):
            started.set()
            if not release.wait(5):
                raise RuntimeError('test did not release warm-up')

        with (patch.object(app, 'default_backend', 'openvino'),
              patch.object(app, 'warmup_status', {'state': 'idle', 'message': ''}),
              patch.object(app.generator, 'warmup', side_effect=warm),
              patch.object(app.generator, 'generate') as model):
            worker = app.start_warmup()
            try:
                self.assertTrue(started.wait(2))
                status, _, body = self.request('GET', '/api/config')
                self.assertEqual(status, 200)
                self.assertEqual(json.loads(body)['warmup']['state'], 'warming')
                status, error = self.post(backend='openvino')
                self.assertEqual(status, 409)
                self.assertIn('Lanczos', error['detail'])
                status, result = self.post(backend='upscale')
                self.assertEqual(status, 200)
                self.assertEqual(result['crop'], [768, 0, 1024, 256])
                self.assertEqual(result['image'], result['raw'])
                self.assertEqual(result['image'], result['baseline'])
                with data_image(result['image']) as image:
                    self.assertEqual(image.size, (1024, 1024))
                    self.assertEqual(image.getpixel((512, 512)), (0, 0, 255))
                model.assert_not_called()
            finally:
                release.set()
                worker.join(timeout=5)
            self.assertEqual(app.warmup_status['state'], 'ready')

    def test_model_failure_releases_the_lock_for_the_next_click(self):
        output = Image.new('RGB', (256, 256), 'green')
        timing = {'model': 'test generator', 'denoising_steps': 1,
                  'start_timestep': 249, 'load_seconds': 0, 'inference_seconds': 0.1}
        with patch.object(app.generator, 'generate',
                          side_effect=[RuntimeError('device lost'), (output, timing)]):
            status, error = self.post(backend='openvino')
            self.assertEqual(status, 500)
            self.assertIn('device lost', error['detail'])
            status, result = self.post(backend='openvino', preserve_structure=False, feather=0)
            self.assertEqual(status, 200)
            self.assertEqual(result['image'], result['raw'])
            self.assertNotEqual(result['image'], result['baseline'])

    def test_image_only_zoom_passes_empty_text_to_the_generator(self):
        output = Image.new('RGB', (256, 256), 'green')
        payload = {'image': self.source, 'center': [1, 0], 'zoom_factor': 4, 'backend': 'openvino'}
        with patch.object(app.generator, 'generate', return_value=(output, {})) as model:
            status, _, body = self.request('POST', '/api/zoom', json.dumps(payload).encode(),
                                           {'Content-Type': 'application/json'})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)['crop'], [768, 0, 1024, 256])
        crop, prompt = model.call_args.args[:2]
        self.assertEqual(crop.getpixel((128, 128)), (0, 0, 255))
        self.assertEqual(prompt, '')

    def test_bad_requests_are_rejected_before_model_use(self):
        with patch.object(app.generator, 'generate') as model:
            for overrides in (
                {'image': 'invalid'},
                {'center': [-0.1, 0.5]},
                {'backend': 'openvino', 'refinement_steps': 5},
            ):
                with self.subTest(overrides=overrides):
                    status, body = self.post(**overrides)
                    self.assertEqual(status, 400)
                    self.assertIn('detail', body)
                    self.assertNotIn('input_value', body['detail'])
            status, _, _ = self.request('POST', '/api/zoom', b'', {'Content-Length': '8100001'})
            self.assertEqual(status, 400)
            model.assert_not_called()

    def test_disconnected_browser_does_not_break_the_response_handler(self):
        handler = app.Handler.__new__(app.Handler)
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.wfile = MagicMock()
        handler.wfile.write.side_effect = BrokenPipeError
        handler.respond(200, b'{}', 'application/json')
        self.assertTrue(handler.close_connection)
        handler.send_response.assert_called_once_with(200)
