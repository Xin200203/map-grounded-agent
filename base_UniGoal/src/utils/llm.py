import base64
import json
import logging
import os
import socket
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from io import BytesIO
from urllib.parse import urlparse

import httpx

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

_REQUEST_TIMEOUT_SECONDS = 120
_DEFAULT_MAX_TOKENS = 1024


def _int_from_env(name, default):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; falling back to %s.", name, raw, default)
        return default
    return max(1, value)


def _float_from_env(name, default, *, minimum=None):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; falling back to %s.", name, raw, default)
        return default
    if minimum is not None:
        value = max(float(minimum), value)
    return value


def _bool_from_env(name, default=False):
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on", "y"}


def _retry_delays_from_env(name, default):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return list(default)

    delays = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            value = float(piece)
        except ValueError:
            logger.warning("Invalid retry delay %r in %s=%r; using %s.", piece, name, raw, default)
            return list(default)
        delays.append(max(0.0, value))

    if not delays:
        logger.warning("Empty %s=%r; falling back to %s.", name, raw, default)
        return list(default)
    return delays


_MAX_RETRIES = _int_from_env("SMOOTHNAV_LLM_MAX_RETRIES", 3)
_RETRY_DELAYS = _retry_delays_from_env("SMOOTHNAV_LLM_RETRY_DELAYS", [2, 5, 10])
_LLM_TEMPERATURE = _float_from_env(
    "SMOOTHNAV_LLM_TEMPERATURE",
    None,
    minimum=0.0,
)


def _pinned_dns_from_env():
    raw = os.environ.get("SMOOTHNAV_PINNED_DNS", "").strip()
    pinned = {}
    if not raw:
        return pinned

    for entry in raw.split(","):
        entry = entry.strip()
        if not entry or "=" not in entry:
            continue
        host, ip_list = entry.split("=", 1)
        host = host.strip().lower()
        ips = [ip.strip() for ip in ip_list.split("|") if ip.strip()]
        if host and ips:
            pinned[host] = ips
    return pinned


@contextmanager
def _dns_override_for_endpoint(endpoint):
    hostname = urlparse(endpoint).hostname or ""
    pinned_ips = _pinned_dns_from_env().get(hostname.lower())
    if not pinned_ips:
        yield
        return

    original_getaddrinfo = socket.getaddrinfo

    def _patched_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        if (host or "").lower() != hostname.lower():
            return original_getaddrinfo(host, port, family, type, proto, flags)

        results = []
        socktype = type or socket.SOCK_STREAM
        proto_value = proto or 0
        for ip in pinned_ips:
            ip_family = socket.AF_INET6 if ":" in ip else socket.AF_INET
            if family not in (0, socket.AF_UNSPEC, ip_family):
                continue
            sockaddr = (ip, port, 0, 0) if ip_family == socket.AF_INET6 else (ip, port)
            results.append((ip_family, socktype, proto_value, "", sockaddr))

        if results:
            return results
        return original_getaddrinfo(host, port, family, type, proto, flags)

    socket.getaddrinfo = _patched_getaddrinfo
    try:
        logger.info("Applying pinned DNS for %s -> %s", hostname, ",".join(pinned_ips))
        yield
    finally:
        socket.getaddrinfo = original_getaddrinfo


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


def _retry_api_call(fn, description="LLM call", on_final_error=None):
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
                if on_final_error is not None:
                    on_final_error(str(exc))
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
        with _dns_override_for_endpoint(endpoint):
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


def _post_json_httpx(endpoint, payload, headers):
    try:
        with _dns_override_for_endpoint(endpoint):
            with httpx.Client(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
                response = client.post(endpoint, headers=headers, json=payload)
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Request to {endpoint} failed: {exc}") from exc

    body = response.text
    if response.status_code >= 400:
        excerpt = " ".join(body.split())[:400]
        raise RuntimeError(f"HTTP {response.status_code} from {endpoint}: {excerpt}")

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Non-JSON response from {endpoint}: {' '.join(body.split())[:400]}"
        ) from exc


def _build_anthropic_message_payload(
    prompt, model, max_tokens, image_str=None, temperature=None
):
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
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": content}],
    }
    if temperature is not None:
        payload["temperature"] = float(temperature)
    return payload


def _create_anthropic_client(base_url, api_key):
    if Anthropic is None:
        raise RuntimeError("anthropic package is not installed")

    client_kwargs = {
        "base_url": (base_url or "").rstrip("/"),
        "timeout": _REQUEST_TIMEOUT_SECONDS,
    }
    try:
        return Anthropic(auth_token=api_key, **client_kwargs)
    except TypeError:
        return Anthropic(api_key=api_key, **client_kwargs)


def _call_anthropic_streaming(base_url, api_key, payload):
    client = _create_anthropic_client(base_url, api_key)
    endpoint = _endpoint_for_protocol(base_url, "anthropic-messages")
    try:
        with _dns_override_for_endpoint(endpoint):
            with client.messages.stream(
                model=payload["model"],
                max_tokens=payload["max_tokens"],
                messages=payload["messages"],
                **(
                    {"temperature": payload["temperature"]}
                    if payload.get("temperature") is not None
                    else {}
                ),
            ) as stream:
                text = "".join(chunk for chunk in stream.text_stream).strip()
        if not text:
            raise RuntimeError("Anthropic stream returned empty text")
        return text
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


def _build_openai_responses_payload(
    prompt, model, max_tokens, image_str=None, temperature=None
):
    content = [{"type": "input_text", "text": prompt}]
    if image_str:
        content.append(
            {
                "type": "input_image",
                "image_url": f"data:image/png;base64,{image_str}",
            }
        )
    payload = {
        "model": model,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": max_tokens,
    }
    if temperature is not None:
        payload["temperature"] = float(temperature)
    return payload


def _openai_chat_extra_body_from_env():
    extra = {}
    raw = os.environ.get("SMOOTHNAV_OPENAI_CHAT_EXTRA_BODY_JSON", "").strip()
    if raw:
        try:
            loaded = json.loads(raw)
            if isinstance(loaded, dict):
                extra.update(loaded)
            else:
                logger.warning(
                    "Ignoring SMOOTHNAV_OPENAI_CHAT_EXTRA_BODY_JSON because it is not an object."
                )
        except json.JSONDecodeError as exc:
            logger.warning(
                "Ignoring invalid SMOOTHNAV_OPENAI_CHAT_EXTRA_BODY_JSON: %s",
                exc,
            )
    if _bool_from_env("SMOOTHNAV_OPENAI_CHAT_ENABLE_THINKING", False):
        extra["enable_thinking"] = True
    return extra


def _build_openai_chat_payload(
    prompt,
    model,
    image_str=None,
    temperature=None,
    max_tokens=None,
    extra_body=None,
):
    if image_str:
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_str}"}},
        ]
    else:
        content = prompt
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
    }
    if temperature is not None:
        payload["temperature"] = float(temperature)
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)
    for key, value in dict(extra_body or {}).items():
        if key not in payload:
            payload[key] = value
    return payload


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


def _anthropic_httpx_headers(api_key, base_url):
    if Anthropic is not None:
        try:
            client = _create_anthropic_client(base_url, api_key)
            try:
                headers = dict(client.default_headers)
                headers.update(client.auth_headers)
                return headers
            finally:
                close = getattr(client, "close", None)
                if callable(close):
                    close()
        except Exception:
            pass

    headers = _anthropic_headers(api_key, use_auth_header=True)
    headers.setdefault("User-Agent", "SmoothNav/anthropic-httpx")
    headers.setdefault("x-stainless-lang", "python")
    headers.setdefault("x-stainless-package-version", "fallback")
    return headers


def _call_model(
    base_url,
    api_key,
    model,
    provider,
    protocol,
    prompt,
    max_tokens,
    image_str=None,
    temperature=None,
):
    del provider  # provider is validated before this point and carried for observability.

    if protocol == "anthropic-messages":
        payload = _build_anthropic_message_payload(
            prompt,
            model,
            max_tokens,
            image_str,
            temperature=temperature,
        )
        if Anthropic is not None:
            try:
                return _call_anthropic_streaming(base_url, api_key, payload)
            except Exception as exc:
                logger.warning(
                    "Anthropic streaming path failed for %s: %s. Falling back to non-stream request.",
                    model,
                    exc,
                )

        endpoint = _endpoint_for_protocol(base_url, protocol)
        response = _post_json_httpx(
            endpoint,
            payload,
            _anthropic_httpx_headers(api_key, base_url),
        )
        return _extract_text(protocol, response)

    endpoint = _endpoint_for_protocol(base_url, protocol)

    if protocol == "openai-responses":
        payload = _build_openai_responses_payload(
            prompt,
            model,
            max_tokens,
            image_str,
            temperature=temperature,
        )
        response = _post_json(endpoint, payload, _openai_headers(api_key))
        return _extract_text(protocol, response)

    if protocol == "openai-chat-completions":
        payload = _build_openai_chat_payload(
            prompt,
            model,
            image_str,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=_openai_chat_extra_body_from_env(),
        )
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
        temperature=_LLM_TEMPERATURE,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.llm_model = llm_model
        self.api_provider, self.api_protocol = resolve_provider_protocol(
            api_provider,
            api_protocol,
        )
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.last_error = ""

    def __call__(self, prompt):
        self.last_error = ""

        def _call():
            return _call_model(
                self.base_url,
                self.api_key,
                self.llm_model,
                self.api_provider,
                self.api_protocol,
                prompt,
                self.max_tokens,
                temperature=self.temperature,
            )

        return _retry_api_call(
            _call,
            "LLM",
            on_final_error=lambda message: setattr(self, "last_error", message),
        )


class VLM:
    def __init__(
        self,
        base_url,
        api_key,
        vlm_model,
        api_provider="",
        api_protocol="",
        max_tokens=_DEFAULT_MAX_TOKENS,
        temperature=_LLM_TEMPERATURE,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.vlm_model = vlm_model
        self.api_provider, self.api_protocol = resolve_provider_protocol(
            api_provider,
            api_protocol,
        )
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.last_error = ""

    def __call__(self, prompt, image):
        self.last_error = ""
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
                temperature=self.temperature,
            )

        return _retry_api_call(
            _call,
            "VLM",
            on_final_error=lambda message: setattr(self, "last_error", message),
        )
