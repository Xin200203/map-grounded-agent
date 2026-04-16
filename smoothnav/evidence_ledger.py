"""Evidence ledger with normalized propositions and freshness tracking."""

from typing import Any, Dict, Iterable, List, Optional

from smoothnav.types import Evidence, VerificationLevel
from smoothnav.writer_guards import BELIEF_WRITER, WriterToken, require_writer


VALID_PROPOSITION_PREFIXES = (
    "exists(",
    "contains(",
    "candidate_match(",
    "region_exhausted(",
    "constraint_satisfied(",
    "constraint_violated(",
    "stop_condition_satisfied(",
    "grounding_failure(",
    "executor_feedback(",
)


def validate_proposition(proposition: str) -> None:
    if not proposition or not proposition.endswith(")"):
        raise ValueError("Evidence proposition must use a normalized template")
    if not proposition.startswith(VALID_PROPOSITION_PREFIXES):
        raise ValueError(f"Unsupported evidence proposition: {proposition}")


class EvidenceLedger:
    """Cold audit log for evidence. Only the belief updater may write."""

    def __init__(self, writer_token: Optional[WriterToken] = None):
        self.writer_token = writer_token or WriterToken(BELIEF_WRITER)
        require_writer(BELIEF_WRITER, self.writer_token, "EvidenceLedger")
        self._items: Dict[str, Evidence] = {}
        self._counter = 0

    def add_observation(
        self,
        proposition: str,
        *,
        source: str,
        scope: str,
        confidence: float,
        timestamp: int,
        entity_bindings: Optional[Dict[str, Any]] = None,
        supports: Optional[List[str]] = None,
        revocable: bool = True,
        verification_level: VerificationLevel = VerificationLevel.OBSERVED,
        coverage_witness: Optional[Dict[str, Any]] = None,
        staleness_policy: str = "decay",
    ) -> Evidence:
        require_writer(BELIEF_WRITER, self.writer_token, "EvidenceLedger")
        validate_proposition(proposition)
        evidence = Evidence(
            evidence_id=self._next_id(),
            proposition=proposition,
            source=source,
            scope=scope,
            confidence=max(0.0, min(1.0, float(confidence))),
            timestamp=int(timestamp),
            revocable=bool(revocable),
            verification_level=verification_level,
            entity_bindings=dict(entity_bindings or {}),
            supports=list(supports or []),
            freshness=1.0,
            coverage_witness=coverage_witness,
            staleness_policy=staleness_policy,
        )
        self._items[evidence.evidence_id] = evidence
        return evidence

    def add_derived(
        self,
        proposition: str,
        *,
        source: str,
        scope: str,
        confidence: float,
        timestamp: int,
        derived_from: Iterable[str],
        supports: Optional[List[str]] = None,
        entity_bindings: Optional[Dict[str, Any]] = None,
        verification_level: VerificationLevel = VerificationLevel.HYPOTHESIZED,
    ) -> Evidence:
        derived_ids = list(derived_from)
        missing = [evidence_id for evidence_id in derived_ids if evidence_id not in self._items]
        if missing:
            raise KeyError(f"Cannot derive evidence from missing ids: {missing}")
        evidence = self.add_observation(
            proposition,
            source=source,
            scope=scope,
            confidence=confidence,
            timestamp=timestamp,
            entity_bindings=entity_bindings,
            supports=supports,
            verification_level=verification_level,
        )
        evidence.derived_from = derived_ids
        return evidence

    def mark_contradicted(self, evidence_id: str, *, timestamp: int) -> None:
        require_writer(BELIEF_WRITER, self.writer_token, "EvidenceLedger")
        evidence = self._items[evidence_id]
        evidence.verification_level = VerificationLevel.CONTRADICTED
        evidence.freshness = 0.0
        evidence.timestamp = int(timestamp)
        for dependent in self._items.values():
            if evidence_id in dependent.derived_from:
                dependent.verification_level = VerificationLevel.CONTRADICTED
                dependent.freshness = 0.0

    def refresh_freshness(self, step_idx: int, *, horizon: int = 100) -> None:
        require_writer(BELIEF_WRITER, self.writer_token, "EvidenceLedger")
        horizon = max(1, int(horizon))
        for evidence in self._items.values():
            if not evidence.revocable or evidence.verification_level == VerificationLevel.VERIFIED:
                evidence.freshness = 1.0
                continue
            age = max(0, int(step_idx) - int(evidence.timestamp))
            evidence.freshness = max(0.0, 1.0 - float(age) / float(horizon))

    def is_valid(self, evidence_id: str) -> bool:
        evidence = self._items.get(evidence_id)
        if evidence is None:
            return False
        return (
            evidence.verification_level != VerificationLevel.CONTRADICTED
            and evidence.freshness > 0.0
        )

    def get(self, evidence_id: str) -> Optional[Evidence]:
        return self._items.get(evidence_id)

    def recent_ids(self, limit: int = 8) -> List[str]:
        items = sorted(self._items.values(), key=lambda item: item.timestamp, reverse=True)
        return [item.evidence_id for item in items[:limit]]

    def by_level(self, level: VerificationLevel) -> List[Evidence]:
        return [item for item in self._items.values() if item.verification_level == level]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": len(self._items),
            "recent_ids": self.recent_ids(),
            "items": [item.to_dict() for item in sorted(self._items.values(), key=lambda x: x.evidence_id)],
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "count": len(self._items),
            "recent_ids": self.recent_ids(),
            "contradicted_ids": [
                item.evidence_id
                for item in self._items.values()
                if item.verification_level == VerificationLevel.CONTRADICTED
            ],
        }

    def _next_id(self) -> str:
        self._counter += 1
        return f"ev_{self._counter:06d}"
