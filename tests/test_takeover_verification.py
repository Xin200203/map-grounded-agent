"""C7' takeover-verification pure logic: throttle cache + response parsing."""

from smoothnav.executor_adoption import (
    parse_verify_response,
    resolve_cached_verify_verdict,
)


def test_empty_cache_requests_fresh_call():
    assert resolve_cached_verify_verdict({}, timestep=5, interval=10, max_calls=8) is None


def test_within_interval_returns_last_verdict():
    cache = {"calls": 1, "last": {"verdict": "no", "step": 100}}
    assert (
        resolve_cached_verify_verdict(cache, timestep=105, interval=10, max_calls=8)
        == "no"
    )


def test_after_interval_requests_fresh_call():
    cache = {"calls": 1, "last": {"verdict": "no", "step": 100}}
    assert (
        resolve_cached_verify_verdict(cache, timestep=110, interval=10, max_calls=8)
        is None
    )


def test_cap_reached_sticks_with_last_verdict():
    cache = {"calls": 8, "last": {"verdict": "yes", "step": 100}}
    assert (
        resolve_cached_verify_verdict(cache, timestep=500, interval=10, max_calls=8)
        == "yes"
    )


def test_cap_reached_without_history_fails_open():
    cache = {"calls": 8}
    assert (
        resolve_cached_verify_verdict(cache, timestep=500, interval=10, max_calls=8)
        == "unsure"
    )


def test_parse_clean_json():
    verdict, reason = parse_verify_response(
        '{"match": "yes", "reason": "blue sofa matches"}'
    )
    assert verdict == "yes"
    assert reason == "blue sofa matches"


def test_parse_json_with_surrounding_prose():
    raw = 'Looking at the crop... {"match": "no", "reason": "this is a table"} done'
    verdict, reason = parse_verify_response(raw)
    assert verdict == "no"
    assert reason == "this is a table"


def test_parse_case_insensitive_and_unquoted():
    verdict, _ = parse_verify_response('{"MATCH": Yes}')
    assert verdict == "yes"


def test_unparseable_fails_open_to_unsure():
    for raw in ["", None, "I cannot tell.", '{"visible": true}']:
        verdict, _ = parse_verify_response(raw)
        assert verdict == "unsure"


def test_reason_truncated_to_60_chars():
    long_reason = "x" * 200
    _, reason = parse_verify_response(
        '{"match": "unsure", "reason": "%s"}' % long_reason
    )
    assert len(reason) == 60
