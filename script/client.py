"""OpenRouter image chat using a single API key."""
import base64
import io
import json
import os
import time
from pathlib import Path
import requests


DEFAULT_REASONING_EFFORT = 'xhigh'
DEFAULT_IMAGE_ENCODING = 'jpeg'
DEFAULT_RECENT_TURNS = 2
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 4096


def default_max_tokens(model):
    return DEFAULT_MAX_TOKENS


CHAT_CONNECTION_MODES = ('separate', 'continuous', 'first-recent')
# Long image histories with xhigh reasoning can outlast a three-minute read.
HTTP_TIMEOUT_SECONDS = (15, 600)


class VisionClient:
    def __init__(self, model, base_url, keys_file=None, max_tokens=None,
                 chat_connection_mode='continuous', chat_recent_turns=DEFAULT_RECENT_TURNS, image_encoding=DEFAULT_IMAGE_ENCODING, jpeg_quality=90, reasoning_effort=DEFAULT_REASONING_EFFORT, system_prompt=None, label_images=False):
        if chat_connection_mode not in CHAT_CONNECTION_MODES:
            raise ValueError('Unknown chat connection mode: ' + chat_connection_mode)
        if chat_recent_turns < 1:
            raise ValueError('chat_recent_turns must be at least 1')
        if keys_file:
            keys = json.loads(Path(keys_file).read_text())
            if not isinstance(keys, list) or len(keys) != 1:
                raise ValueError('--keys-file must contain a JSON list with exactly one API key')
            api_key = keys[0]
        else:
            api_key = os.environ.get('OPENROUTER_API_KEY')
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError('Provide one nonempty API key via OPENROUTER_API_KEY or --keys-file')
        self.api_key = api_key.strip()
        if image_encoding not in ('png', 'jpeg') or not 1 <= jpeg_quality <= 100:
            raise ValueError('Invalid image encoding or JPEG quality')
        self.image_encoding, self.jpeg_quality = image_encoding, jpeg_quality
        if reasoning_effort not in ('none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'):
            raise ValueError('Invalid reasoning effort')
        self.reasoning_effort = reasoning_effort
        self.system_prompt = system_prompt
        self.label_images = label_images
        self.model, self.url = model, base_url.rstrip('/') + '/chat/completions'
        max_tokens = default_max_tokens(model) if max_tokens is None else max_tokens
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.max_tokens = max_tokens
        self.chat_connection_mode = chat_connection_mode
        self.chat_recent_turns = chat_recent_turns
        self.history = []  # Full user content (including images) and assistant replies.
        self.turn_records = []  # Paths and task step IDs; no credentials or base64.
        self.usage = []

    def _user_message(self, prompt, images):
        content = [{'type': 'text', 'text': prompt}]
        for image in images:
            if self.label_images:
                path = Path(image)
                content.append({'type': 'text', 'text': f'Image source: observation {path.parent.name}, camera file {path.name}. Match this ID to the CURRENT OBSERVATION heading.'})
            data = Path(image).read_bytes()
            if self.image_encoding == 'jpeg':
                from PIL import Image
                buffer = io.BytesIO()
                with Image.open(io.BytesIO(data)) as source:
                    source.convert('RGB').save(buffer, format='JPEG', quality=self.jpeg_quality)
                data = buffer.getvalue()
            encoded = base64.b64encode(data).decode('ascii')
            content.append({'type': 'image_url', 'image_url': {
                'url': f'data:image/{self.image_encoding};base64,' + encoded, 'detail': 'low'}})
        return {'role': 'user', 'content': content}

    def _prepare_context(self, prompt, images, context_path, step_index, source, response_path=None):
        user = self._user_message(prompt, images)
        count = len(self.turn_records)
        mode = self.chat_connection_mode
        if mode == 'separate':
            selected = []
        elif mode == 'continuous':
            selected = list(range(count))
        else:
            selected = sorted({0, *range(max(0, count-self.chat_recent_turns), count)}) if count else []
        folder = Path(context_path).resolve().parent if context_path is not None else None
        current = {'step': step_index if step_index is not None else count+1,
                   'prompt': str(folder / 'prompt.txt') if folder else None,
                   'response': str(Path(response_path).resolve()) if response_path else
                               (str(folder / 'response.txt') if folder else None),
                   'images': [str(Path(image).resolve()) for image in images], 'source': source}
        messages = ([{'role': 'system', 'content': self.system_prompt}] if self.system_prompt else [])
        for index in selected:
            historical = self.history[2*index]
            if mode == 'first-recent':
                role = 'FIRST' if index == 0 else 'RECENT'
                label = (f'[History: {role} task conversation; step {self.turn_records[index]["step"]}; '
                         'these images are historical observations, not current images.]')
                historical = {**historical, 'content': [{'type': 'text', 'text': label}, *historical['content']]}
            messages.extend([historical, self.history[2*index+1]])
        current_message = user
        if mode == 'first-recent':
            retained = [self.turn_records[i]['step'] for i in selected]
            omitted = [r['step'] for i, r in enumerate(self.turn_records) if i not in selected]
            label = (f'[CURRENT task conversation: step {current["step"]}. Keep the FIRST conversation '
                     f'plus the latest {self.chat_recent_turns} completed conversation(s), with images and replies, '
                     f'without duplicates. Retained previous steps: {retained}. Omitted previous steps: {omitted}. '
                     'Images in THIS message are current. Historical model answers are proposed actions, '
                     'not proof of successful execution. Follow the requested action output format.]')
            current_message = {**user, 'content': [{'type': 'text', 'text': label}, *user['content']]}
        messages.append(current_message)
        context = {'system_prompt_present': bool(self.system_prompt), 'image_source_labels': self.label_images, 'reasoning_effort': self.reasoning_effort, 'image_encoding': self.image_encoding, 'jpeg_quality': self.jpeg_quality if self.image_encoding == 'jpeg' else None,
                   'image_resize': False, 'mode': mode, 'recent_turns': self.chat_recent_turns if mode == 'first-recent' else None,
                   'history': [self.turn_records[i] for i in selected], 'current': current,
                   'history_labels_added': mode == 'first-recent',
                   'request_kind': 'online' if source == 'online' else 'replay_context_only',
                   'http_timeout_seconds': {'connect': HTTP_TIMEOUT_SECONDS[0], 'read': HTTP_TIMEOUT_SECONDS[1]} if source == 'online' else None,
                   'message_count': len(messages),
                   'image_count': sum(len(r['images']) for r in [*[self.turn_records[i] for i in selected], current])}
        if context_path is not None:
            Path(context_path).write_text(json.dumps(context, indent=2) + '\n')
        return user, messages, current

    def _remember(self, user, answer, record):
        if self.chat_connection_mode != 'separate':
            self.history.extend([user, {'role': 'assistant', 'content': answer}])
        self.turn_records.append(record)

    def remember_turn(self, prompt, images, response, *, context_path=None, step_index=None, response_path=None):
        """Restore a physically replayed step with its fresh observation; no API call."""
        user, _, record = self._prepare_context(prompt, images, context_path, step_index,
                                                'replay', response_path=response_path)
        self._remember(user, response, record)

    def chat(self, prompt, images, *, context_path=None, step_index=None):
        user, messages, record = self._prepare_context(prompt, images, context_path, step_index, 'online')
        payload = {'model': self.model, 'messages': messages,
                   'max_tokens': self.max_tokens, 'temperature': DEFAULT_TEMPERATURE, 'reasoning': {'effort': self.reasoning_effort}}
        for attempt in range(12):
            start = time.time()
            try:
                response = requests.post(self.url, headers={'Authorization': 'Bearer ' + self.api_key}, json=payload, timeout=HTTP_TIMEOUT_SECONDS)
            except (requests.Timeout, requests.ConnectionError, requests.exceptions.ChunkedEncodingError) as error:
                # Transport errors must not commit or duplicate a conversation turn.
                print(f'OpenRouter transport {type(error).__name__}; retry {attempt + 1}/12', flush=True)
                time.sleep(min(10, 2 ** attempt))
                continue
            if response.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(10, 2 ** attempt))
                continue
            if not response.ok:
                detail = response.text.replace(self.api_key, '[redacted]')[:500]
                raise RuntimeError(f'OpenRouter HTTP {response.status_code}: {detail}')
            try:
                data = response.json()
            except requests.exceptions.JSONDecodeError:
                # An HTTP-success response can still contain truncated/non-JSON
                # data. Retry the same turn, without committing chat history.
                # Record metadata only: response bodies may echo private input.
                if context_path is not None:
                    diagnostic = {'attempt': attempt + 1, 'max_attempts': 12,
                                  'error': 'invalid_response_json',
                                  'http_status': response.status_code,
                                  'response_bytes': len(response.content)}
                    (Path(context_path).parent / f'api_error_attempt_{attempt + 1:02d}.json').write_text(
                        json.dumps(diagnostic, indent=2))
                print(f'OpenRouter invalid response JSON; attempt {attempt + 1}/12', flush=True)
                if attempt < 11:
                    time.sleep(min(10, 2 ** attempt))
                continue
            if 'error' in data:
                detail = str(data['error'].get('message', 'unknown')).replace(self.api_key, '[redacted]')
                raise RuntimeError('OpenRouter returned an error: ' + detail)
            if context_path is not None:
                # Preserve the model's returned message for diagnosis, not the request or credentials.
                (Path(context_path).parent / 'model_message.json').write_text(
                    json.dumps(data['choices'][0]['message'], ensure_ascii=False, indent=2))
            text = data['choices'][0]['message']['content']
            self.usage.append({'model': data.get('model'), 'id': data.get('id'), 'usage': data.get('usage'), 'seconds': time.time() - start})
            if not text:
                print('Empty content; finish reason:', data['choices'][0].get('finish_reason'), flush=True)
                self.max_tokens = min(max(self.max_tokens * 2, self.max_tokens + 500), max(8192, self.max_tokens))
                payload['max_tokens'] = self.max_tokens
                continue
            self._remember(user, text, record)
            return text
        raise RuntimeError('OpenRouter temporary errors exhausted retries')
