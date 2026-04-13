import base64
import json
import logging
import time
import urllib.error
import urllib.request
from io import BytesIO

try:
    from anthropic import Anthropic
except ImportError:  # pragma: no cover - exercised via runtime fallback.
    Anthropic = None

logger = logging.getLogger(__name__)

SUPPORTED_API_PROVIDERS = ("anthropic", "openai")
SUPPORTED_API_PROTOCOLS = (
    "anthropic-messages",
    "openai-responses",
    "openai-chat-completions",
)

_DEFAULT_API_PROVIDER = "anthropic"
_DEFAULT_API_PROTOCOL = "anthropic-messages"
_DEFAULT_PROTOCOL_BY_PROVIDER = {
    "anthropic": "anthropic-messages",
    "openai": "openai-responses",
}
_PROVIDER_BY_PROTOCOL = {
    "anthropic-messages": "anthropic",
    "openai-responses": "openai",
    "openai-chat-completions": "openai",
}

_MAX_RETRIES = 3
_RETRY_DELAYS = [2, 5, 10]
_REQUEST_TIMEOUT_SECONDS = 120
_DEFAULT_MAX_TOKENS = 1024


def resolve_provider_protocol(api_provider="", api_protocol=""):
    provider = (api_provider or "").strip().lower()
    protocol = (api_protocol or "").strip().lower()

    if not provider and not protocol:
        return _DEFAULT_API_PROVIDER, _DEFAULT_API_PROTOCOL

    if provider and provider not in SUPPORTED_API_PROVIDERS:
        raise RuntimeError(
            f"Unsupported API provider '{api_provider}'. "
            f"Expected one of: {', '.join(SUPPORTED_API_PROVIDERS)}."
        )

    if protocol and protocol not in SUPPORTED_API_PROTOCOLS:
        raise RuntimeError(
            f"Unsupported API protocol '{api_protocol}'. "
            f"Expected one of: {', '.join(SUPPORTED_API_PROTOCOLS)}."
        )

    if not provider:
        provider = _PROVIDER_BY_PROTOCOL[protocol]
    if not protocol:
        protocol = _DEFAULT_PROTOCOL_BY_PROVIDER[provider]

    expected_provider = _PROVIDER_BY_PROTOCOL[protocol]
    if provider != expected_provider:
        raise RuntimeError(
            f"API provider/protocol mismatch: provider='{provider}' "
            f"is incompatible with protocol='{protocol}'."
        )

    return provider, protocol


def _retry_api_call(fn, description="LLM call"):
    """Retry an API call with exponential backoff. Returns empty string on failure."""
    for attempt in range(_MAX_RETRIES):
        try:
            return fn()
        except Exception as exc:
            delay = _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)]
            logger.warning(
                f"{description} attempt {attempt + 1}/{_MAX_RETRIES} failed: {exc}. "
                f"Retrying in {delay}s..."
            )
            if attempt < _MAX_RETRIES - 1:
                time.sleep(delay)
            else:
                logger.error(f"{description} failed after {_MAX_RETRIES} attempts: {exc}")
                return ""


def _append_endpoint(base_url, suffix):
    base = (base_url or "").rstrip("/")
    suffix = suffix if suffix.startswith("/") else f"/{suffix}"
    if base.endswith(suffix):
        return base
    return f"{base}{suffix}"


def _endpoint_for_protocol(base_url, protocol):
    if protocol == "anthropic-messages":
        if base_url.rstrip("/").endswith("/v1"):
            return _append_endpoint(base_url, "/messages")
        return _append_endpoint(base_url, "/v1/messages")
    if protocol == "openai-responses":
        if base_url.rstrip("/").endswith("/v1"):
            return _append_endpoint(base_url, "/responses")
        return _append_endpoint(base_url, "/v1/responses")
    if protocol == "openai-chat-completions":
        if base_url.rstrip("/").endswith("/v1"):
            return _append_endpoint(base_url, "/chat/completions")
        return _append_endpoint(base_url, "/v1/chat/completions")
    raise RuntimeError(f"Unsupported API protocol '{protocol}'.")


def _anthropic_headers(api_key, use_auth_header):
    headers = {
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    if use_auth_header:
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        headers["x-api-key"] = api_key
    return headers


def _openai_headers(api_key):
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _post_json(endpoint, payload, headers):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=data,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT_SECONDS) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        excerpt = " ".join(body.split())[:400]
        raise RuntimeError(f"HTTP {exc.code} from {endpoint}: {excerpt}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Request to {endpoint} failed: {exc}") from exc

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Non-JSON response from {endpoint}: {' '.join(body.split())[:400]}"
        ) from exc


def _build_anthropic_message_payload(prompt, model, max_tokens, image_str=None):
    content = [{"type": "text", "text": prompt}]
    if image_str:
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_str,
                },
            }
        )
    return {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": content}],
    }


def _build_openai_responses_payload(prompt, model, max_tokens, image_str=None):
    content = [{"type": "input_text", "text": prompt}]
    if image_str:
        content.append(
            {
                "type": "input_image",
                "image_url": f"data:image/png;base64,{image_str}",
            }
        )
    return {
        "model": model,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": max_tokens,
    }


def _build_openai_chat_payload(prompt, model, image_str=None):
    if image_str:
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_str}"}},
        ]
    else:
        content = prompt
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
    }


def _extract_text_from_anthropic_response(response):
    if hasattr(response, "content"):
        parts = []
        for block in getattr(response, "content", []) or []:
            if getattr(block, "type", None) == "text" and getattr(block, "text", ""):
                parts.append(block.text)
        return "\n".join(parts).strip()

    parts = []
    for block in response.get("content", []):
        if block.get("type") == "text" and block.get("text"):
            parts.append(block["text"])
    return "\n".join(parts).strip()


def _extract_text_from_openai_responses(response):
    if isinstance(response.get("output_text"), str) and response["output_text"].strip():
        return response["output_text"].strip()

    parts = []
    for item in response.get("output", []):
        for block in item.get("content", []):
            block_type = block.get("type")
            if block_type in {"output_text", "text"} and block.get("text"):
                parts.append(block["text"])
    if parts:
        return "\n".join(parts).strip()
    return _extract_text_from_chat_response(response)


def _extract_text_from_chat_response(response):
    choices = response.get("choices", [])
    if not choices:
        return ""
    content = choices[0].get("message", {}).get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if block.get("type") in {"text", "output_text"} and block.get("text"):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    return ""


def _extract_text(protocol, response):
    if protocol == "anthropic-messages":
        return _extract_text_from_anthropic_response(response)
    if protocol == "openai-responses":
        return _extract_text_from_openai_responses(response)
    if protocol == "openai-chat-completions":
        return _extract_text_from_chat_response(response)
    raise RuntimeError(f"Unsupported API protocol '{protocol}'.")


def _call_model(base_url, api_key, model, provider, protocol, prompt, max_tokens, image_str=None):
    del provider  # provider is validated before this point and carried for observability.

    if protocol == "anthropic-messages":
        if Anthropic is None:
            raise RuntimeError(
                "anthropic package is required for anthropic-messages protocol but is not installed."
            )

        payload = _build_anthropic_message_payload(prompt, model, max_tokens, image_str)
        client = Anthropic(api_key=api_key, base_url=(base_url or "").rstrip("/"))
        response = client.messages.create(**payload)
        return _extract_text(protocol, response)

    endpoint = _endpoint_for_protocol(base_url, protocol)

    if protocol == "openai-responses":
        payload = _build_openai_responses_payload(prompt, model, max_tokens, image_str)
        response = _post_json(endpoint, payload, _openai_headers(api_key))
        return _extract_text(protocol, response)

    if protocol == "openai-chat-completions":
        payload = _build_openai_chat_payload(prompt, model, image_str)
        response = _post_json(endpoint, payload, _openai_headers(api_key))
        return _extract_text(protocol, response)

    raise RuntimeError(f"Unsupported API protocol '{protocol}'.")


class LLM:
    def __init__(
        self,
        base_url,
        api_key,
        llm_model,
        api_provider="",
        api_protocol="",
        max_tokens=_DEFAULT_MAX_TOKENS,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.llm_model = llm_model
        self.api_provider, self.api_protocol = resolve_provider_protocol(
            api_provider,
            api_protocol,
        )
        self.max_tokens = max_tokens

    def __call__(self, prompt):
        def _call():
            return _call_model(
                self.base_url,
                self.api_key,
                self.llm_model,
                self.api_provider,
                self.api_protocol,
                prompt,
                self.max_tokens,
            )

        return _retry_api_call(_call, "LLM")


class VLM:
    def __init__(
        self,
        base_url,
        api_key,
        vlm_model,
        api_provider="",
        api_protocol="",
        max_tokens=_DEFAULT_MAX_TOKENS,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.vlm_model = vlm_model
        self.api_provider, self.api_protocol = resolve_provider_protocol(
            api_provider,
            api_protocol,
        )
        self.max_tokens = max_tokens

    def __call__(self, prompt, image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        def _call():
            return _call_model(
                self.base_url,
                self.api_key,
                self.vlm_model,
                self.api_provider,
                self.api_protocol,
                prompt,
                self.max_tokens,
                image_str=image_str,
            )

        return _retry_api_call(_call, "VLM")
