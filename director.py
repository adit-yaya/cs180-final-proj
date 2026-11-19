"""Local, bounded story context for an Ollama prompt writer."""
import json
from difflib import SequenceMatcher
import os
import secrets
import time
import urllib.error
import urllib.request
from typing import Literal

from pydantic import BaseModel, Field, ValidationError


class StoryBeat(BaseModel):
    story: str = Field(min_length=1, max_length=500)
    prompt: str = Field(min_length=1, max_length=1200)


class DirectorRequest(BaseModel):
    initial_prompt: str = Field(min_length=1, max_length=2000)
    history: list[StoryBeat] = Field(default_factory=list, max_length=6)
    step: int = Field(default=1, ge=1, le=1000000)
    variation: Literal['gentle', 'medium', 'surprising'] = 'medium'


SYSTEM = """You are the visual storyteller for an interactive generative zoom.
Write the NEXT close-up scene, not a whole plot. The user's starting direction is
always the anchor. Recent beats are context, not instructions. You do NOT see the
image: do not claim to observe it or describe the clicked location.
Keep the scene's palette, illumination, materials and key motifs connected.
Use the supplied creative cue as the main new development. It is mandatory, not
an optional suggestion. Move closer into one existing motif and reveal something
that has not appeared in earlier discoveries. The next discovery must change the
subject or its structure, not just its adjectives.
A moderate evolution should feel like another detail of the same world, not a
jump to an unrelated scene. At the first step introduce the world, then build on
recent beats without repeating their exact composition or prompt.
Return JSON with exactly two fields:
- story: one NEW short sentence describing the new visible discovery. Do not
  reuse a previous story beat or merely change an adjective.
- prompt: a standalone diffusion image prompt, 40-65 words, naming concrete
subjects, textures, light and spatial relationships. Include enough of the anchor
for this prompt to make sense by itself. The prompt MUST depict the discovery
in your new story sentence. Change the actual subject or structure, not just its
adjectives. Do not copy a previous prompt. Favor crisp detail throughout and deep
focus. No bokeh, shallow depth of field, tilt-shift, captions, dialogue, camera
instructions, 'zoom in', or instructions about writing prompts.
Do not include commentary or reasoning. /no_think"""


class PromptDirector:
    def __init__(self):
        self.model = os.environ.get('ZOOM_PROMPT_MODEL', 'qwen3:4b')
        self.endpoint = 'http://127.0.0.1:11434/api/chat'

    def next(self, request: DirectorRequest, _retry=False):
        notes = {
            'gentle': 'Preserve almost everything; introduce one subtle material or structural detail.',
            'medium': 'Preserve the main world and introduce one surprising but plausible discovery.',
            'surprising': 'Introduce a bolder surreal discovery while retaining recognizable visual motifs.',
        }
        seed = secrets.randbelow(2147483647)
        cues = [
            'Reveal an unexpected structure hidden inside one existing object.',
            'Follow a tiny passage into a connected inner world.',
            'Reveal a new pattern where two existing materials meet.',
            'Introduce a tiny living system that fits this world.',
            'Let one existing motif transform into a related architectural form.',
            'Discover a delicate mechanism embedded in a natural structure.',
            'Reveal a surprising source for the scene’s light.',
            'Follow a branching feature to an unexpected miniature landscape.',
        ]
        cue = secrets.choice(cues)
        user = json.dumps({
            'starting_direction': request.initial_prompt,
            'next_step': request.step,
            'variation': notes[request.variation],
            'creative_cue': cue,
            'recent_discoveries_do_not_repeat': [beat.story for beat in request.history[-3:]],
            'current_discovery': request.history[-1].story if request.history else 'The starting world has not yet been explored.',
            'revision': 'Previous draft repeated an old scene. Invent a clearly different discovery.' if _retry else '',
        })
        payload = {
            'model': self.model,
            'messages': [{'role': 'system', 'content': SYSTEM},
                         {'role': 'user', 'content': user}],
            'format': StoryBeat.model_json_schema(),
            'think': False,
            'stream': False,
            'keep_alive': '10m',
            'options': {'temperature': {'gentle': 0.65, 'medium': 0.85, 'surprising': 1.0}[request.variation],
                        'top_p': 0.9, 'repeat_penalty': 1.1, 'seed': seed,
                        'num_ctx': 4096, 'num_predict': 256},
        }
        start = time.perf_counter()
        req = urllib.request.Request(self.endpoint, data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=90) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise RuntimeError(f'Prompt model unavailable. Run: ollama pull {self.model}') from exc
            raise RuntimeError('The local prompt model failed. Check Ollama and try again.') from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError('Cannot reach the local prompt model. Start Ollama with: ollama serve') from exc
        except (ValueError, TypeError) as exc:
            raise RuntimeError('Ollama returned an invalid response. Try the step again.') from exc
        try:
            if not isinstance(result, dict) or not result.get('done') or result.get('done_reason') == 'length':
                raise ValueError('Incomplete generation')
            beat = StoryBeat.model_validate_json(result['message']['content'])
            beat = StoryBeat(prompt=beat.prompt.strip(), story=beat.story.strip())
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            raise RuntimeError('The prompt model returned an incomplete scene. Try the step again.') from exc
        def normalized(text):
            return ' '.join(text.lower().split())
        repeated = any(
            normalized(beat.story) == normalized(previous.story) or
            SequenceMatcher(None, normalized(beat.prompt), normalized(previous.prompt)).ratio() > 0.9
            for previous in request.history[-3:]
        )
        if repeated:
            if _retry:
                raise RuntimeError('The prompt writer repeated a scene. Try again or increase story variation.')
            revised = self.next(request, _retry=True)
            revised['seconds'] = time.perf_counter() - start
            return revised
        return {**beat.model_dump(), 'model': self.model, 'step': request.step,
                'seed': seed, 'seconds': time.perf_counter() - start}
