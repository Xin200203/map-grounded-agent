#!/usr/bin/env python3
"""Minimal SmoothNav LLM API smoke test.

This probes the exact `base_UniGoal.src.utils.llm.LLM` path used by SmoothNav
without printing credentials.  It is intended for checking API keys, proxy base
URLs, protocol compatibility, and model availability before launching Habitat.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
BASE_UNIGOAL = ROOT / "base_UniGoal"
sys.path.insert(0, str(BASE_UNIGOAL))

from src.utils.llm import LLM  # noqa: E402


DEFAULT_CLAUDE_CHEAP_MODELS = [
    "claude-haiku-4-5-20251001",
    "claude-haiku-4-5",
    "claude-3-5-haiku-20241022",
]


def _safe_host(base_url: str) -> str:
    parsed = urlparse(base_url if "://" in base_url else f"https://{base_url}")
    return parsed.netloc or "<missing>"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        help="Model to probe. Repeat to test multiple models.",
    )
    parser.add_argument("--api-provider", default="anthropic")
    parser.add_argument("--api-protocol", default="anthropic-messages")
    parser.add_argument("--api-key-env", default="SMOOTHNAV_API_KEY")
    parser.add_argument("--base-url-env", default="SMOOTHNAV_BASE_URL")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prompt", default="Reply exactly OK")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.ERROR)

    base_url = os.environ.get(args.base_url_env, "").strip()
    api_key = os.environ.get(args.api_key_env, "").strip()
    models = args.models or DEFAULT_CLAUDE_CHEAP_MODELS

    header = {
        "base_url_host": _safe_host(base_url),
        "api_key_env": args.api_key_env,
        "api_key_present": bool(api_key),
        "api_key_length": len(api_key) if api_key else 0,
        "api_provider": args.api_provider,
        "api_protocol": args.api_protocol,
    }
    print(json.dumps({"kind": "smoke_context", **header}, ensure_ascii=False))

    if not base_url or not api_key:
        print(
            json.dumps(
                {
                    "kind": "smoke_error",
                    "ok": False,
                    "reason": "missing_base_url_or_api_key",
                },
                ensure_ascii=False,
            )
        )
        return 2

    any_ok = False
    for model in models:
        llm = LLM(
            base_url,
            api_key,
            model,
            api_provider=args.api_provider,
            api_protocol=args.api_protocol,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        text = llm(args.prompt)
        ok = bool(str(text or "").strip())
        any_ok = any_ok or ok
        print(
            json.dumps(
                {
                    "kind": "model_probe",
                    "model": model,
                    "ok": ok,
                    "text": str(text or "").strip()[:80],
                    "last_error": str(getattr(llm, "last_error", "") or "")[:320],
                },
                ensure_ascii=False,
            )
        )

    return 0 if any_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
