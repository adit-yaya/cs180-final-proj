import base64
import io
import json
import unittest
import urllib.error
from unittest.mock import patch

from PIL import Image
from pydantic import ValidationError

from director import DirectorRequest, PromptDirector


def frame():
    image = Image.new('RGB', (1024, 1024), 'red')
    image.paste('blue', (512, 0, 1024, 1024))
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()


def response(body):
    stream = io.BytesIO(json.dumps(body).encode())
    return stream


class DirectorTests(unittest.TestCase):
    def request(self, **overrides):
        values = dict(image=frame(), center=(0.25, 0.5),
                      initial_prompt='An ancient forest of moss and ferns',
                      step=7)
        values.update(overrides)
        return DirectorRequest(**values)

    def mock_ollama(self, send, beats=None):
        bodies = iter(beats or [{'observations': 'A red surface fills the crop.',
                                'detail': 'A fine amber grain glints on the red surface.',
                                'prompt': 'Fine amber grains glint across a crimson surface in warm light.'}])
        calls = []
        def reply(req, **kwargs):
            if req.full_url.endswith('/api/show'):
                return response({'capabilities': ['completion', 'vision']})
            calls.append(json.loads(req.data))
            beat = next(bodies)
            return response({'done': True, 'done_reason': 'stop',
                             'message': {'content': json.dumps(beat)}})
        send.side_effect = reply
        return calls

    @patch('director.urllib.request.urlopen')
    def test_local_structured_prompt_uses_current_crop_and_starting_direction(self, send):
        calls = self.mock_ollama(send)
        out = PromptDirector().next(self.request())
        payload = calls[0]
        context = json.loads(payload['messages'][1]['content'])
        self.assertEqual(context['starting_direction'], self.request().initial_prompt)
        self.assertEqual(context['variation'], 'medium')
        self.assertFalse(payload['think'])
        self.assertFalse(payload['stream'])
        self.assertEqual(payload['format']['required'], ['observations', 'detail', 'prompt'])
        crop = Image.open(io.BytesIO(base64.b64decode(payload['messages'][1]['images'][0])))
        self.assertEqual(crop.size, (512, 512))
        self.assertEqual(crop.getpixel((256, 256)), (255, 0, 0))
        self.assertEqual(out['observations'], 'A red surface fills the crop.')

    @patch('director.urllib.request.urlopen')
    def test_changing_click_changes_pixels_given_to_model(self, send):
        beat = {'observations': 'A colored surface.', 'detail': 'A delicate texture.', 'prompt': 'Fine surface texture.'}
        calls = self.mock_ollama(send, [beat, beat])
        director = PromptDirector()
        director.next(self.request(center=(0.25, 0.5)))
        director.next(self.request(center=(0.75, 0.5), zoom_factor=4))
        crops = [Image.open(io.BytesIO(base64.b64decode(c['messages'][1]['images'][0]))) for c in calls]
        self.assertEqual(crops[0].getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(crops[1].size, (256, 256))
        self.assertEqual(crops[1].getpixel((0, 0)), (0, 0, 255))
        # Verify the generator receives the same source pixels before upscaling.
        import app
        import numpy as np
        for center, factor, crop in zip([(0.25, 0.5), (0.75, 0.5)], [2, 4], crops):
            with patch.object(app.generator, 'generate', return_value=(crop, {})) as generate:
                output = app.zoom(app.Request(image=frame(), center=center, zoom_factor=factor, prompt='test'))
                for key in ('image', 'raw', 'baseline'):
                    view = Image.open(io.BytesIO(base64.b64decode(output[key].split(',')[1])))
                    self.assertEqual(view.size, (1024, 1024))
                np.testing.assert_array_equal(np.asarray(generate.call_args.args[0]), np.asarray(crop))

    @patch('director.urllib.request.urlopen')
    def test_legacy_history_is_ignored_even_if_detail_repeats(self, send):
        prior = {'detail': 'Finer grain along the edge.', 'prompt': 'A red edge with finer grain.'}
        result = {'observations': 'A red surface.', **prior}
        calls = self.mock_ollama(send, [result])
        out = PromptDirector().next(self.request(history=[prior]))
        self.assertEqual(out['detail'], prior['detail'])
        self.assertEqual(len(calls), 1)
        context = json.loads(calls[0]['messages'][1]['content'])
        self.assertEqual(set(context), {'starting_direction', 'variation', 'revision'})
        self.assertNotIn(prior['detail'], json.dumps(calls[0]))

    @patch('director.urllib.request.urlopen')
    def test_text_only_model_is_rejected_instead_of_ignoring_image(self, send):
        send.return_value = response({'capabilities': ['completion']})
        with self.assertRaisesRegex(RuntimeError, 'cannot read images'):
            PromptDirector().next(self.request())

    @patch('director.urllib.request.urlopen')
    def test_invalid_image_is_rejected_before_model_call(self, send):
        with self.assertRaisesRegex(ValueError, 'Invalid input image'):
            PromptDirector().next(self.request(image='bad image'))
        send.assert_not_called()

    def test_invalid_coordinates_are_rejected(self):
        for center in [(-0.1, 0.5), (1.1, 0.5), (float('nan'), 0.5), (0.5, float('inf'))]:
            with self.subTest(center=center), self.assertRaises(ValidationError):
                self.request(center=center)

    @patch('director.urllib.request.urlopen', side_effect=urllib.error.URLError('refused'))
    def test_connection_failure_is_actionable(self, _):
        with self.assertRaisesRegex(RuntimeError, 'ollama serve'):
            PromptDirector().next(self.request())

    @patch('director.urllib.request.urlopen')
    def test_missing_model_is_actionable(self, send):
        send.side_effect = urllib.error.HTTPError('local', 404, 'not found', {}, io.BytesIO())
        with self.assertRaisesRegex(RuntimeError, 'ollama pull'):
            PromptDirector().next(self.request())

    @patch('director.urllib.request.urlopen')
    def test_rejects_incomplete_or_invalid_directions(self, send):
        for body in [
            {'done': True, 'done_reason': 'length', 'message': {'content': '{}'}},
            {'done': True, 'message': {'content': '{"prompt": "only a prompt"}'}},
            {'done': True, 'message': {'content': '{"observations":" ","prompt":" ","detail":" "}'}},
        ]:
            with self.subTest(body=body):
                send.side_effect = lambda req, **kw: response({'capabilities': ['vision']} if req.full_url.endswith('/api/show') else body)
                with self.assertRaisesRegex(RuntimeError, 'incomplete detail direction'):
                    PromptDirector().next(self.request())


if __name__ == '__main__':
    unittest.main()
