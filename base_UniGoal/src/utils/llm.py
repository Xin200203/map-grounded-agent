import base64
import time
import logging
from openai import OpenAI
from io import BytesIO

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_DELAYS = [2, 5, 10]


def _retry_api_call(fn, description="LLM call"):
    """Retry an API call with exponential backoff. Returns empty string on failure."""
    for attempt in range(_MAX_RETRIES):
        try:
            return fn()
        except Exception as e:
            delay = _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)]
            logger.warning(f"{description} attempt {attempt+1}/{_MAX_RETRIES} failed: {e}. "
                           f"Retrying in {delay}s...")
            if attempt < _MAX_RETRIES - 1:
                time.sleep(delay)
            else:
                logger.error(f"{description} failed after {_MAX_RETRIES} attempts: {e}")
                return ""


class LLM:
    def __init__(self, base_url, api_key, llm_model):
        self.base_url = base_url
        self.api_key = api_key
        self.llm_model = llm_model

    def __call__(self, prompt):
        def _call():
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                    }
                ],
                model=self.llm_model,
            )
            return chat_completion.choices[0].message.content
        return _retry_api_call(_call, "LLM")


class VLM:
    def __init__(self, base_url, api_key, vlm_model):
        self.base_url = base_url
        self.api_key = api_key
        self.vlm_model = vlm_model

    def __call__(self, prompt, image):
        buffered = BytesIO()
        image.save(buffered, format='PNG')
        image_bytes = base64.b64encode(buffered.getvalue())
        image_str = str(image_bytes, 'utf-8')

        def _call():
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        'role': 'user',
                        'content': [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image_str}}
                        ]
                    }
                ],
                model=self.vlm_model,
            )
            return chat_completion.choices[0].message.content
        return _retry_api_call(_call, "VLM")