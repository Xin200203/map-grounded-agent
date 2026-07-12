"""D1 instance-gated-stop: verdict->allow mapping and throttle reuse.

The VLM call itself needs a live channel; here we lock down the pure decision
logic that gates the stop: parse the verifier response, and map verdicts to
allow/reject with a fail-open bias (only a confident 'no' rejects a stop).
"""

from smoothnav.executor_adoption import (
    parse_verify_response,
    resolve_cached_verify_verdict,
)


def allow_stop(verdict):
    """Mirror of agent._instance_stop_allowed's verdict->bool mapping."""
    return verdict != "no"


def test_only_confident_no_rejects_stop():
    assert allow_stop("no") is False
    assert allow_stop("yes") is True
    assert allow_stop("unsure") is True


def test_failopen_parse_defaults_to_unsure_then_allows():
    # unparseable / empty / non-JSON -> 'unsure' -> stop allowed (never worse
    # than baseline when the verifier is down)
    for raw in ["", None, "the model rambled", '{"visible": true}']:
        verdict, _ = parse_verify_response(raw)
        assert verdict == "unsure"
        assert allow_stop(verdict) is True


def test_confident_no_parsed_and_rejects():
    verdict, reason = parse_verify_response(
        '{"match": "no", "reason": "brown chair but goal is a white sofa"}'
    )
    assert verdict == "no"
    assert allow_stop(verdict) is False


def test_yes_parsed_and_allows():
    verdict, _ = parse_verify_response('{"match": "yes", "reason": "matches"}')
    assert allow_stop(verdict) is True


def test_throttle_cache_separate_and_sticky_after_cap():
    cache = {}
    # first call is due
    assert resolve_cached_verify_verdict(cache, timestep=0, interval=3, max_calls=12) is None
    cache["calls"] = 1
    cache["last"] = {"verdict": "no", "step": 0}
    # within interval -> sticky
    assert resolve_cached_verify_verdict(cache, timestep=1, interval=3, max_calls=12) == "no"
    # past interval -> due again
    assert resolve_cached_verify_verdict(cache, timestep=5, interval=3, max_calls=12) is None
    # cap reached -> sticky last verdict (bounds VLM cost per episode)
    cache["calls"] = 12
    assert resolve_cached_verify_verdict(cache, timestep=999, interval=3, max_calls=12) == "no"
