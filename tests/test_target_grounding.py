"""R1 target-text grounding phrase construction (pure logic)."""

from smoothnav.target_grounding import (
    augment_node_space,
    build_target_grounding_phrase,
)

BASE = "table. tv. chair. cabinet. sofa. bed"


def test_dict_goal_uses_intrinsic_only():
    goal = {
        "intrinsic_attributes": "a white porcelain toilet",
        "extrinsic_attributes": "next to a sink and a bathtub",
    }
    phrase = build_target_grounding_phrase(goal)
    # toilet category noun + intrinsic description; surroundings excluded
    assert "toilet" in phrase
    assert "white porcelain toilet" in phrase
    assert "sink" not in phrase and "bathtub" not in phrase


def test_missing_category_toilet_is_surfaced():
    # toilet is absent from the base node_space; R1 must add it
    goal = {"intrinsic_attributes": "a toilet", "extrinsic_attributes": ""}
    augmented, phrase = augment_node_space(BASE, goal)
    assert "toilet" in augmented
    assert augmented.startswith(BASE + ". ")


def test_television_maps_to_tv_noun():
    goal = {"intrinsic_attributes": "a large flat television", "extrinsic_attributes": ""}
    phrase = build_target_grounding_phrase(goal)
    assert phrase.split(". ")[0] == "tv"


def test_leading_article_dropped():
    phrase = build_target_grounding_phrase(
        {"intrinsic_attributes": "the brown leather sofa", "extrinsic_attributes": ""}
    )
    assert "the brown leather sofa" not in phrase
    assert "brown leather sofa" in phrase


def test_description_truncated_to_max_words():
    long = "a " + " ".join(["word%d" % i for i in range(30)])
    phrase = build_target_grounding_phrase(
        {"intrinsic_attributes": long, "extrinsic_attributes": ""}, max_words=5
    )
    # the description clause has at most max_words tokens
    desc_clause = phrase.split(". ")[-1]
    assert len(desc_clause.split()) <= 5


def test_plain_string_goal():
    phrase = build_target_grounding_phrase("a potted plant")
    assert "plant" in phrase


def test_empty_goal_yields_empty_phrase_and_untouched_prompt():
    assert build_target_grounding_phrase("") == ""
    assert build_target_grounding_phrase({}) == ""
    augmented, phrase = augment_node_space(BASE, "")
    assert phrase == "" and augmented == BASE


def test_no_exact_duplicate_pieces():
    # category noun equals a description token but pieces are dot-separated;
    # only EXACT duplicate clauses are dropped
    goal = {"intrinsic_attributes": "chair", "extrinsic_attributes": ""}
    phrase = build_target_grounding_phrase(goal)
    assert phrase == "chair"


def test_augment_is_idempotent_given_stable_base():
    goal = {"intrinsic_attributes": "a red chair", "extrinsic_attributes": ""}
    once, _ = augment_node_space(BASE, goal)
    twice, _ = augment_node_space(BASE, goal)  # rebuild from base, not from `once`
    assert once == twice
