import io
import json
import unittest
import urllib.error
from unittest.mock import patch

from director import DirectorRequest, PromptDirector, StoryBeat


class DirectorTests(unittest.TestCase):
    def request(self):
        return DirectorRequest(initial_prompt='An ancient forest of moss and ferns',
                               history=[StoryBeat(prompt=f'forest scene {i}', story=f'beat {i}') for i in range(6)],
                               step=7)

    @patch('director.urllib.request.urlopen')
    def test_local_structured_prompt_has_anchor_and_bounded_history(self, send):
        send.return_value.__enter__.return_value = io.BytesIO(json.dumps({
            'done': True, 'done_reason': 'stop',
            'message': {'content': json.dumps({'story': 'A vein of amber lights the roots.',
                                               'prompt': 'Golden amber veins thread through mossy roots.'})}
        }).encode())
        out = PromptDirector().next(self.request())
        http_request = send.call_args.args[0]
        self.assertEqual(http_request.full_url, 'http://127.0.0.1:11434/api/chat')
        payload = json.loads(http_request.data)
        context = json.loads(payload['messages'][1]['content'])
        self.assertEqual(context['starting_direction'], self.request().initial_prompt)
        self.assertEqual(context['recent_discoveries_do_not_repeat'], ['beat 3', 'beat 4', 'beat 5'])
        self.assertEqual(context['current_discovery'], 'beat 5')
        self.assertTrue(context['creative_cue'])
        self.assertFalse(payload['think'])
        self.assertFalse(payload['stream'])
        self.assertEqual(payload['format']['required'], ['story', 'prompt'])
        self.assertEqual(payload['options']['temperature'], 0.85)
        self.assertEqual(out['step'], 7)
        self.assertIn('amber', out['prompt'])

    @patch('director.urllib.request.urlopen')
    def test_repeated_scene_retries_then_uses_new_scene(self, send):
        old = {'story': 'beat 5', 'prompt': 'forest scene 5'}
        new = {'story': 'An amber bridge shelters miniature gardens.',
               'prompt': 'A translucent amber bridge arches over tiny terraced gardens among jade leaves.'}
        send.return_value.__enter__.side_effect = [
            io.BytesIO(json.dumps({'done': True, 'message': {'content': json.dumps(beat)}}).encode())
            for beat in (old, new)
        ]
        out = PromptDirector().next(self.request())
        self.assertEqual(out['story'], new['story'])
        self.assertEqual(send.call_count, 2)

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
    def test_rejects_incomplete_or_invalid_scenes(self, send):
        for body in [
            {'done': True, 'done_reason': 'length', 'message': {'content': '{}'}},
            {'done': True, 'message': {'content': '{"prompt": "only a prompt"}'}},
            {'done': True, 'message': {'content': '{"prompt":" ","story":" "}'}},
        ]:
            with self.subTest(body=body):
                send.return_value.__enter__.return_value = io.BytesIO(json.dumps(body).encode())
                with self.assertRaisesRegex(RuntimeError, 'incomplete scene'):
                    PromptDirector().next(self.request())


if __name__ == '__main__':
    unittest.main()
