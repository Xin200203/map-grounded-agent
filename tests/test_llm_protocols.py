"""Unit tests for protocol-specific LLM/VLM payload and parsing helpers."""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "base_UniGoal"))

from src.utils.llm import (  # noqa: E402
    LLM,
    _call_anthropic_streaming,
    _call_model,
    _build_anthropic_message_payload,
    _build_openai_chat_payload,
    _build_openai_responses_payload,
    _endpoint_for_protocol,
    _extract_text_from_anthropic_response,
    _extract_text_from_openai_responses,
    _float_from_env,
    _retry_delays_from_env,
    _int_from_env,
    resolve_provider_protocol,
)


class ResolveProviderProtocolTests(unittest.TestCase):
    def test_defaults_to_anthropic_messages(self):
        provider, protocol = resolve_provider_protocol("", "")
        self.assertEqual(provider, "anthropic")
        self.assertEqual(protocol, "anthropic-messages")

    def test_infers_protocol_from_provider(self):
        provider, protocol = resolve_provider_protocol("openai", "")
        self.assertEqual(provider, "openai")
        self.assertEqual(protocol, "openai-responses")

    def test_rejects_provider_protocol_mismatch(self):
        with self.assertRaisesRegex(RuntimeError, "mismatch"):
            resolve_provider_protocol("anthropic", "openai-responses")


class RetryConfigTests(unittest.TestCase):
    def test_retry_count_reads_env_override(self):
        original = os.environ.get("SMOOTHNAV_TEST_RETRY_COUNT")
        try:
            os.environ["SMOOTHNAV_TEST_RETRY_COUNT"] = "2"
            self.assertEqual(_int_from_env("SMOOTHNAV_TEST_RETRY_COUNT", 3), 2)
        finally:
            if original is None:
                os.environ.pop("SMOOTHNAV_TEST_RETRY_COUNT", None)
            else:
                os.environ["SMOOTHNAV_TEST_RETRY_COUNT"] = original

    def test_retry_delays_reads_csv_env_override(self):
        original = os.environ.get("SMOOTHNAV_TEST_RETRY_DELAYS")
        try:
            os.environ["SMOOTHNAV_TEST_RETRY_DELAYS"] = "1, 3, 5"
            self.assertEqual(
                _retry_delays_from_env("SMOOTHNAV_TEST_RETRY_DELAYS", [2, 5, 10]),
                [1.0, 3.0, 5.0],
            )
        finally:
            if original is None:
                os.environ.pop("SMOOTHNAV_TEST_RETRY_DELAYS", None)
            else:
                os.environ["SMOOTHNAV_TEST_RETRY_DELAYS"] = original

    def test_temperature_reads_env_override(self):
        original = os.environ.get("SMOOTHNAV_TEST_TEMPERATURE")
        try:
            os.environ["SMOOTHNAV_TEST_TEMPERATURE"] = "0"
            self.assertEqual(
                _float_from_env("SMOOTHNAV_TEST_TEMPERATURE", None, minimum=0.0),
                0.0,
            )
        finally:
            if original is None:
                os.environ.pop("SMOOTHNAV_TEST_TEMPERATURE", None)
            else:
                os.environ["SMOOTHNAV_TEST_TEMPERATURE"] = original

    def test_llm_records_final_error_when_retry_exhausts(self):
        llm = LLM(
            "https://clauddy.com",
            "sk-test",
            "claude-haiku-4-5-20251001",
            api_provider="anthropic",
            api_protocol="anthropic-messages",
        )

        with mock.patch("src.utils.llm._MAX_RETRIES", 1), mock.patch(
            "src.utils.llm._RETRY_DELAYS",
            [0],
        ), mock.patch(
            "src.utils.llm._call_model",
            side_effect=RuntimeError("HTTP 403 from https://clauddy.com/v1/messages"),
        ):
            response = llm("Reply with JSON")

        self.assertEqual(response, "")
        self.assertIn("HTTP 403", llm.last_error)


class EndpointResolutionTests(unittest.TestCase):
    def test_anthropic_endpoint_appends_messages(self):
        self.assertEqual(
            _endpoint_for_protocol("https://clauddy.com", "anthropic-messages"),
            "https://clauddy.com/v1/messages",
        )

    def test_openai_endpoint_appends_responses(self):
        self.assertEqual(
            _endpoint_for_protocol("https://clauddy.com/v1", "openai-responses"),
            "https://clauddy.com/v1/responses",
        )


class PayloadBuilderTests(unittest.TestCase):
    def test_builds_anthropic_multimodal_payload(self):
        payload = _build_anthropic_message_payload(
            "describe object",
            "claude-haiku-4-5-20251001",
            128,
            image_str="ZmFrZQ==",
            temperature=0.0,
        )
        self.assertEqual(payload["messages"][0]["content"][0]["type"], "text")
        self.assertEqual(payload["messages"][0]["content"][1]["type"], "image")
        self.assertEqual(
            payload["messages"][0]["content"][1]["source"]["data"],
            "ZmFrZQ==",
        )
        self.assertEqual(payload["temperature"], 0.0)

    def test_builds_openai_responses_multimodal_payload(self):
        payload = _build_openai_responses_payload(
            "describe object",
            "gpt-5.4-mini",
            128,
            image_str="ZmFrZQ==",
            temperature=0.0,
        )
        self.assertEqual(payload["input"][0]["content"][0]["type"], "input_text")
        self.assertEqual(payload["input"][0]["content"][1]["type"], "input_image")
        self.assertTrue(
            payload["input"][0]["content"][1]["image_url"].startswith("data:image/png;base64,")
        )
        self.assertEqual(payload["temperature"], 0.0)

    def test_builds_openai_chat_payload_with_dashscope_extra_body(self):
        payload = _build_openai_chat_payload(
            "choose branch",
            "deepseek-v4-pro",
            image_str="ZmFrZQ==",
            temperature=0.0,
            max_tokens=128,
            extra_body={"enable_thinking": True},
        )

        self.assertEqual(payload["model"], "deepseek-v4-pro")
        self.assertEqual(payload["max_tokens"], 128)
        self.assertTrue(payload["enable_thinking"])
        self.assertEqual(payload["messages"][0]["content"][0]["type"], "text")
        self.assertEqual(payload["messages"][0]["content"][1]["type"], "image_url")


class ResponseParsingTests(unittest.TestCase):
    def test_extracts_text_from_anthropic_messages(self):
        response = {
            "content": [
                {"type": "text", "text": "north room"},
                {"type": "text", "text": "then inspect sink"},
            ]
        }
        self.assertEqual(
            _extract_text_from_anthropic_response(response),
            "north room\nthen inspect sink",
        )

    def test_extracts_text_from_anthropic_sdk_objects(self):
        class FakeBlock:
            def __init__(self, text):
                self.type = "text"
                self.text = text

        class FakeResponse:
            def __init__(self):
                self.content = [FakeBlock("north room"), FakeBlock("then inspect sink")]

        self.assertEqual(
            _extract_text_from_anthropic_response(FakeResponse()),
            "north room\nthen inspect sink",
        )

    def test_extracts_text_from_openai_responses(self):
        response = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "go left"},
                        {"type": "output_text", "text": "look for sofa"},
                    ],
                }
            ]
        }
        self.assertEqual(
            _extract_text_from_openai_responses(response),
            "go left\nlook for sofa",
        )


class AnthropicStreamingTests(unittest.TestCase):
    def test_call_anthropic_streaming_reads_text_stream(self):
        payload = _build_anthropic_message_payload(
            "Reply with exactly: PING",
            "claude-haiku-4-5-20251001",
            64,
            temperature=0.0,
        )

        class FakeStreamContext:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            @property
            def text_stream(self):
                return iter(["PI", "NG"])

        fake_client = mock.Mock()
        fake_client.messages.stream.return_value = FakeStreamContext()

        with mock.patch("src.utils.llm._create_anthropic_client", return_value=fake_client):
            text = _call_anthropic_streaming("https://clauddy.com", "sk-test", payload)

        self.assertEqual(text, "PING")
        fake_client.messages.stream.assert_called_once_with(
            model="claude-haiku-4-5-20251001",
            max_tokens=64,
            messages=payload["messages"],
            temperature=0.0,
        )
        fake_client.close.assert_called_once()

    def test_call_model_prefers_streaming_for_anthropic_messages(self):
        with mock.patch("src.utils.llm.Anthropic", new=object()), mock.patch(
            "src.utils.llm._call_anthropic_streaming",
            return_value="PING",
        ) as stream_call, mock.patch("src.utils.llm._post_json_httpx") as post_json:
            text = _call_model(
                "https://clauddy.com",
                "sk-test",
                "claude-haiku-4-5-20251001",
                "anthropic",
                "anthropic-messages",
                "Reply with exactly: PING",
                64,
                temperature=0.0,
            )

        self.assertEqual(text, "PING")
        stream_call.assert_called_once()
        post_json.assert_not_called()

    def test_call_model_falls_back_to_non_stream_when_stream_fails(self):
        with mock.patch("src.utils.llm.Anthropic", new=object()), mock.patch(
            "src.utils.llm._call_anthropic_streaming",
            side_effect=RuntimeError("stream failed"),
        ), mock.patch(
            "src.utils.llm._anthropic_httpx_headers",
            return_value={"Authorization": "Bearer sk-test"},
        ), mock.patch(
            "src.utils.llm._post_json_httpx",
            return_value={"content": [{"type": "text", "text": "PING"}]},
        ) as post_json:
            text = _call_model(
                "https://clauddy.com",
                "sk-test",
                "claude-haiku-4-5-20251001",
                "anthropic",
                "anthropic-messages",
                "Reply with exactly: PING",
                64,
                temperature=0.0,
            )

        self.assertEqual(text, "PING")
        post_json.assert_called_once()


if __name__ == "__main__":
    unittest.main()
