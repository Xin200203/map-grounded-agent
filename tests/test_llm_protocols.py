"""Unit tests for protocol-specific LLM/VLM payload and parsing helpers."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "base_UniGoal"))

from src.utils.llm import (  # noqa: E402
    _build_anthropic_message_payload,
    _build_openai_responses_payload,
    _endpoint_for_protocol,
    _extract_text_from_anthropic_response,
    _extract_text_from_openai_responses,
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
        )
        self.assertEqual(payload["messages"][0]["content"][0]["type"], "text")
        self.assertEqual(payload["messages"][0]["content"][1]["type"], "image")
        self.assertEqual(
            payload["messages"][0]["content"][1]["source"]["data"],
            "ZmFrZQ==",
        )

    def test_builds_openai_responses_multimodal_payload(self):
        payload = _build_openai_responses_payload(
            "describe object",
            "gpt-5.4-mini",
            128,
            image_str="ZmFrZQ==",
        )
        self.assertEqual(payload["input"][0]["content"][0]["type"], "input_text")
        self.assertEqual(payload["input"][0]["content"][1]["type"], "input_image")
        self.assertTrue(
            payload["input"][0]["content"][1]["image_url"].startswith("data:image/png;base64,")
        )


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


if __name__ == "__main__":
    unittest.main()
