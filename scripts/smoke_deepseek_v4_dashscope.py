#!/usr/bin/env python3
"""Smoke test DashScope deepseek-v4-pro through SmoothNav's OpenAI-chat path.

Credentials are read from DASHSCOPE_API_KEY by default and are never printed.
The default base URL is DashScope's OpenAI-compatible endpoint.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
BASE_UNIGOAL = ROOT / "base_UniGoal"
sys.path.insert(0, str(BASE_UNIGOAL))

from src.utils.llm import LLM  # noqa: E402


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "deepseek-v4-pro"


def _safe_host(url: str) -> str:
    parsed = urlparse(url if "://" in url else f"https://{url}")
    return parsed.netloc or "<missing>"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--base-url", default=os.environ.get("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--prompt", default="请只回复 OK")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=os.environ.get("SMOOTHNAV_OPENAI_CHAT_ENABLE_THINKING", "").lower()
        in {"1", "true", "yes", "on"},
    )
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env, "").strip()
    print(
        json.dumps(
            {
                "kind": "dashscope_deepseek_context",
                "model": args.model,
                "base_url_host": _safe_host(args.base_url),
                "api_key_env": args.api_key_env,
                "api_key_present": bool(api_key),
                "api_key_length": len(api_key) if api_key else 0,
                "enable_thinking": bool(args.enable_thinking),
                "api_provider": "openai",
                "api_protocol": "openai-chat-completions",
            },
            ensure_ascii=False,
        )
    )
    if not api_key:
        print(json.dumps({"kind": "smoke_error", "ok": False, "reason": "missing_api_key"}, ensure_ascii=False))
        return 2

    old_thinking = os.environ.get("SMOOTHNAV_OPENAI_CHAT_ENABLE_THINKING")
    try:
        os.environ["SMOOTHNAV_OPENAI_CHAT_ENABLE_THINKING"] = "1" if args.enable_thinking else "0"
        llm = LLM(
            args.base_url,
            api_key,
            args.model,
            api_provider="openai",
            api_protocol="openai-chat-completions",
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        text = llm(args.prompt)
    finally:
        if old_thinking is None:
            os.environ.pop("SMOOTHNAV_OPENAI_CHAT_ENABLE_THINKING", None)
        else:
            os.environ["SMOOTHNAV_OPENAI_CHAT_ENABLE_THINKING"] = old_thinking

    ok = bool(str(text or "").strip())
    print(
        json.dumps(
            {
                "kind": "dashscope_deepseek_probe",
                "ok": ok,
                "text": str(text or "").strip()[:120],
                "last_error": str(getattr(llm, "last_error", "") or "")[:400],
            },
            ensure_ascii=False,
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
