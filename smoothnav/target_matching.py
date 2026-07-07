"""Lightweight target/caption matching for text-goal evidence uptake.

The helpers in this module intentionally avoid model calls and new dependencies.
They provide a conservative, inspectable score for deciding whether a newly
observed caption is likely to be the task target rather than just generic scene
context.
"""

import re
from typing import Any, Dict, Iterable, List, Sequence, Set


# Canonical object groups. Keep this deliberately small and common for Habitat
# indoor object-nav targets; unmatched captions still fall back to token overlap.
CATEGORY_ALIASES: Dict[str, Set[str]] = {
    "television": {"television", "tv", "monitor", "screen", "display"},
    "chair": {"chair", "chairs", "stool", "seat", "armchair"},
    "sofa": {"sofa", "couch", "loveseat", "settee"},
    "table": {"table", "desk", "coffee table", "dining table", "side table"},
    "bed": {"bed", "beds", "mattress"},
    "cabinet": {"cabinet", "cupboard", "wardrobe", "dresser", "drawer"},
    "window": {"window", "windows"},
    "curtain": {"curtain", "curtains", "drape", "drapes"},
    "painting": {"painting", "picture", "art", "artwork", "poster"},
    "mirror": {"mirror"},
    "plant": {"plant", "potted plant", "vase"},
    "sink": {"sink", "basin"},
    "toilet": {"toilet"},
    "bathtub": {"bathtub", "tub", "bath"},
    "refrigerator": {"refrigerator", "fridge"},
    "oven": {"oven", "stove", "cooktop"},
    "microwave": {"microwave"},
    "lamp": {"lamp", "light"},
    "bookshelf": {"bookshelf", "bookcase", "shelf", "shelves"},
}

_ALIAS_TO_CATEGORY: Dict[str, str] = {
    alias: category
    for category, aliases in CATEGORY_ALIASES.items()
    for alias in aliases
}

STOPWORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "below", "by", "for", "from",
    "has", "have", "in", "inside", "is", "it", "its", "near", "next", "of",
    "on", "or", "the", "there", "this", "to", "with", "without", "image",
    "picture", "photo", "shows", "show", "side", "upper", "lower", "half",
    "object", "objects", "room", "scene", "target", "single", "provide",
    "information", "based", "what", "seen", "can", "determined",
}

ATTRIBUTE_WORDS: Set[str] = {
    "black", "white", "gray", "grey", "silver", "brown", "brownish", "yellow",
    "blue", "green", "red", "orange", "wood", "wooden", "leather", "metal",
    "glass", "transparent", "light", "dark", "frame", "cushion", "seat",
}


def normalize_text(value: Any) -> str:
    text = str(value or "").lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(value: Any) -> List[str]:
    return [token for token in normalize_text(value).split() if token]


def _phrase_positions(text: str, aliases: Iterable[str]) -> List[int]:
    positions: List[int] = []
    for alias in aliases:
        alias_norm = normalize_text(alias)
        if not alias_norm:
            continue
        match = re.search(rf"(?<![a-z0-9]){re.escape(alias_norm)}(?![a-z0-9])", text)
        if match:
            positions.append(match.start())
    return positions


def categories_in_text(value: Any) -> Dict[str, int]:
    """Return canonical object categories mentioned in text and earliest offset."""

    text = normalize_text(value)
    found: Dict[str, int] = {}
    if not text:
        return found
    for category, aliases in CATEGORY_ALIASES.items():
        positions = _phrase_positions(text, aliases | {category})
        if positions:
            found[category] = min(positions)
    return found


def caption_categories(caption: Any) -> Set[str]:
    text = normalize_text(caption)
    found = set(categories_in_text(text))
    tokens = set(tokenize(text))
    for token in tokens:
        category = _ALIAS_TO_CATEGORY.get(token)
        if category:
            found.add(category)
    return found


def primary_goal_categories(goal_text: Any) -> Set[str]:
    """Estimate the target category from the earliest object mention.

    Text goals often include contextual objects after the first sentence. Using
    the earliest known object category keeps those contextual objects from being
    promoted as direct targets.
    """

    text = str(goal_text or "")
    normalized = normalize_text(text)
    found = categories_in_text(normalized)
    if not found:
        return set()

    first_sentence = normalize_text(re.split(r"[.!?]", text, maxsplit=1)[0])
    first_sentence_found = categories_in_text(first_sentence)
    candidates = first_sentence_found or found
    earliest = min(candidates.values())
    return {category for category, pos in candidates.items() if pos == earliest}


def content_tokens(value: Any) -> Set[str]:
    return {
        token
        for token in tokenize(value)
        if token not in STOPWORDS and len(token) > 2
    }


def caption_matches_label(label: Any, caption: Any) -> bool:
    """Return True when a planner choice label can resolve to a node caption."""

    label_text = normalize_text(label)
    caption_text = normalize_text(caption)
    if not label_text or not caption_text:
        return False
    if label_text == caption_text or label_text in caption_text or caption_text in label_text:
        return True
    return bool(caption_categories(label_text) & caption_categories(caption_text))


def score_caption_against_goal(caption: Any, goal_text: Any) -> Dict[str, Any]:
    """Score how likely a caption is to be the primary text-goal target.

    Score scale is intentionally coarse:
    - >= 0.75: likely primary target / alias; safe to trigger uptake
    - 0.35-0.70: contextually relevant object; useful as evidence, not direct target
    - 0.0: currently irrelevant
    """

    caption_text = normalize_text(caption)
    goal_norm = normalize_text(goal_text)
    cap_categories = caption_categories(caption_text)
    goal_categories = set(categories_in_text(goal_norm))
    primary_categories = primary_goal_categories(goal_norm)
    canonical_overlap = sorted(cap_categories & goal_categories)
    primary_overlap = sorted(cap_categories & primary_categories)

    caption_tokens = content_tokens(caption_text)
    goal_tokens = content_tokens(goal_norm)
    token_overlap = sorted(caption_tokens & goal_tokens)
    attr_overlap = sorted((caption_tokens & goal_tokens) & ATTRIBUTE_WORDS)

    score = 0.0
    reason = "no_match"
    if primary_overlap:
        score = 1.0
        reason = "primary_category_alias"
    elif canonical_overlap:
        score = 0.55
        reason = "context_category_alias"
    elif token_overlap:
        score = 0.35
        reason = "token_overlap"

    if attr_overlap and score > 0.0:
        score = min(1.0, score + 0.05 * len(attr_overlap))

    return {
        "caption": str(caption or ""),
        "score": float(score),
        "reason": reason,
        "primary_categories": sorted(primary_categories),
        "caption_categories": sorted(cap_categories),
        "goal_categories": sorted(goal_categories),
        "canonical_overlap": canonical_overlap,
        "primary_overlap": primary_overlap,
        "token_overlap": token_overlap,
        "attribute_overlap": attr_overlap,
    }


def is_target_candidate(caption: Any, goal_text: Any, *, threshold: float = 0.75) -> bool:
    return score_caption_against_goal(caption, goal_text)["score"] >= float(threshold)


def target_candidate_details(nodes: Sequence[Any], goal_text: Any, *, threshold: float = 0.75) -> List[Dict[str, Any]]:
    details: List[Dict[str, Any]] = []
    for node in nodes or []:
        caption = getattr(node, "caption", "")
        if not caption:
            continue
        match = score_caption_against_goal(caption, goal_text)
        if match["score"] < float(threshold):
            continue
        center = getattr(node, "center", None)
        item = dict(match)
        if center is not None:
            try:
                item["center"] = [int(center[0]), int(center[1])]
            except Exception:
                item["center"] = None
        try:
            item["num_detections"] = int(
                (getattr(node, "object", {}) or {}).get("num_detections", 0) or 0
            )
        except Exception:
            item["num_detections"] = 0
        details.append(item)
    return details
