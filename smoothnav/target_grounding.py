"""Pure helper for R1 target-text active grounding.

The scene graph's GroundingDINO detection prompt (`node_space`) is a fixed
15-term class list. This leaves two recall holes for text goals:

1. Some goal categories are absent from the list entirely (e.g. toilet), so
   the open-vocabulary detector is never asked to find them.
2. Even when the category is present, a detection's caption carries only the
   generic class token, so a described target instance's caption relevance
   can fall below the candidate threshold (detected_no_anchor).

R1 appends a target-specific phrase — the canonical category noun plus the
intrinsic object description (surroundings deliberately excluded, so we do
not ground the context objects) — to the prompt. GroundingDINO then actively
grounds the target instance and captions it with target tokens.
"""

from smoothnav.target_matching import normalize_text, primary_goal_categories

# GroundingDINO-friendly noun for internal category keys that differ from a
# natural detection phrase; unlisted categories use the key verbatim.
_CATEGORY_NOUN = {
    "television": "tv",
}

_LEADING_ARTICLES = {"a", "an", "the"}


def _split_goal_text(goal_text):
    """Return (intrinsic_description, full_text). Surroundings feed category
    detection (robustness) but never the grounding description."""
    if isinstance(goal_text, dict):
        intrinsic = str(goal_text.get("intrinsic_attributes", "") or "")
        extrinsic = str(goal_text.get("extrinsic_attributes", "") or "")
        return intrinsic, (intrinsic + " " + extrinsic).strip()
    text = str(goal_text or "")
    return text, text


def build_target_grounding_phrase(goal_text, max_words=12):
    """Build a dot-separated GroundingDINO phrase for the goal, or '' if the
    goal text yields nothing usable.

    Example: {"intrinsic_attributes": "a white porcelain toilet", ...}
      -> "toilet. white porcelain toilet"
    """
    intrinsic, full_text = _split_goal_text(goal_text)
    nouns = [
        _CATEGORY_NOUN.get(cat, cat)
        for cat in sorted(primary_goal_categories(full_text))
    ]

    words = [w for w in normalize_text(intrinsic).split() if w]
    if words and words[0] in _LEADING_ARTICLES:
        words = words[1:]
    desc_phrase = " ".join(words[:max_words]).strip()

    pieces = []
    seen = set()
    for piece in nouns + ([desc_phrase] if desc_phrase else []):
        piece = piece.strip()
        if piece and piece not in seen:
            seen.add(piece)
            pieces.append(piece)
    return ". ".join(pieces)


def augment_node_space(base_node_space, goal_text, max_words=12):
    """Append the target grounding phrase to the base prompt (idempotent given
    a stable base). Returns (augmented_prompt, phrase)."""
    phrase = build_target_grounding_phrase(goal_text, max_words=max_words)
    base = str(base_node_space or "").strip()
    if not phrase:
        return base, ""
    return (base + ". " + phrase) if base else phrase, phrase
