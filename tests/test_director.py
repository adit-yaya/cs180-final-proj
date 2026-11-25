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


def reference_view(message):
    return Image.open(io.BytesIO(base64.b64decode(message['images'][0])))


def full_crop(reference):
    return reference.crop((0, 0, reference.height, reference.height))


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
        self.assertEqual(context['focus_percent'], [50, 50])
        self.assertEqual(context['zoom_factor'], 2)
        self.assertEqual(context['zoom_depth'], 0)
        self.assertFalse(payload['think'])
        self.assertFalse(payload['stream'])
        self.assertEqual(payload['format']['required'], ['observations', 'detail', 'prompt'])
        crop = full_crop(reference_view(payload['messages'][1]))
        self.assertEqual(crop.size, (320, 320))
        self.assertEqual(crop.getpixel((160, 160)), (255, 0, 0))
        self.assertEqual(out['observations'], 'A red surface fills the crop.')

    @patch('director.urllib.request.urlopen')
    def test_first_crop_can_be_directed_without_a_starting_prompt(self, send):
        beat = {'observations': 'A red surface.', 'detail': 'Fine grain.', 'prompt': 'Fine red grain.'}
        calls = self.mock_ollama(send, [beat, beat])
        director = PromptDirector()
        for hint in ({}, {'initial_prompt': '  '}):
            result = director.next(DirectorRequest(image=frame(), center=(0.25, 0.5), **hint))
            self.assertEqual(result['prompt'], beat['prompt'])
            message = calls[-1]['messages'][1]
            self.assertNotIn('starting_direction', json.loads(message['content']))
            self.assertEqual(full_crop(reference_view(message)).getpixel((160, 160)), (255, 0, 0))

    @patch('director.urllib.request.urlopen')
    def test_edge_clicks_send_different_closeups_even_when_the_crop_is_identical(self, send):
        image = Image.new('RGB', (1024, 1024), 'red')
        image.paste('blue', (128, 0, 1024, 1024))
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        encoded = base64.b64encode(buffer.getvalue()).decode()
        detail = {'observations': 'Visible surface.', 'detail': 'Fine grain.', 'prompt': 'Fine grain.'}
        calls = self.mock_ollama(send, [detail, detail])
        director = PromptDirector()
        for x in (0, 0.24):
            director.next(self.request(image=encoded, center=(x, 0.5), zoom_depth=5, seed=17))
        first, second = (call['messages'][1] for call in calls)
        self.assertEqual(full_crop(reference_view(first)).tobytes(),
                         full_crop(reference_view(second)).tobytes())
        for message, color, x in ((first, (255, 0, 0), 0), (second, (0, 0, 255), 48)):
            reference = reference_view(message)
            self.assertEqual(len(message['images']), 1)
            self.assertEqual(reference.getpixel((reference.width - 64, reference.height // 2)), color)
            context = json.loads(message['content'])
            self.assertEqual(context['focus_percent'], [x, 50])
            self.assertEqual(context['zoom_depth'], 5)
        self.assertTrue(all(call['options']['seed'] == 17 for call in calls))

    @patch('director.urllib.request.urlopen')
    def test_changing_click_changes_pixels_given_to_model(self, send):
        beat = {'observations': 'A colored surface.', 'detail': 'A delicate texture.', 'prompt': 'Fine surface texture.'}
        calls = self.mock_ollama(send, [beat, beat])
        director = PromptDirector()
        director.next(self.request(center=(0.25, 0.5)))
        director.next(self.request(center=(0.75, 0.5), zoom_factor=4))
        crops = [full_crop(reference_view(call['messages'][1])) for call in calls]
        self.assertEqual(crops[0].getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(crops[1].size, (256, 256))
        self.assertEqual(crops[1].getpixel((0, 0)), (0, 0, 255))
        # Both paths select the same source region before their model-specific resize.
        import app
        import numpy as np
        for center, factor, crop in zip([(0.25, 0.5), (0.75, 0.5)], [2, 4], crops):
            with patch.object(app.generator, 'generate', return_value=(crop, {})) as generate:
                output = app.zoom(app.Request(image=frame(), center=center, zoom_factor=factor, prompt='test'))
                for key in ('image', 'raw', 'baseline'):
                    view = Image.open(io.BytesIO(base64.b64decode(output[key].split(',')[1])))
                    self.assertEqual(view.size, (1024, 1024))
                selected = generate.call_args.args[0].resize(crop.size, Image.Resampling.LANCZOS)
                np.testing.assert_array_equal(np.asarray(selected), np.asarray(crop))

    @patch('director.urllib.request.urlopen')
    def test_legacy_history_is_ignored_even_if_detail_repeats(self, send):
        prior = {'detail': 'Finer grain along the edge.', 'prompt': 'A red edge with finer grain.'}
        result = {'observations': 'A red surface.', **prior}
        calls = self.mock_ollama(send, [result])
        out = PromptDirector().next(self.request(history=[prior]))
        self.assertEqual(out['detail'], prior['detail'])
        self.assertEqual(len(calls), 1)
        context = json.loads(calls[0]['messages'][1]['content'])
        self.assertNotIn('history', context)
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
