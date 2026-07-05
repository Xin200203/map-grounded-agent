# SmoothNav Top-tier Execution Plan (Phased, Actionable)

Date: 2026-04-22
Updated: 2026-04-23
Status: execution-ready plan derived from current experiment evidence and deep-interview clarification

## 0. Goal

Deliver a top-tier-paper-ready research package centered on the SmoothNav system story.

## 1. Guiding Principle

The immediate objective is **not** to keep changing the controller blindly.
The immediate objective is to remove the blockers that prevent a trustworthy top-tier readiness judgment.

## 2. Phase Map

### Current effective phase (2026-04-23)

- Phase A: complete
- Phase B: complete
- Phase C: active
- Phase D: partially active
- Phase E: conditional and likely next if repaired cross-scene aggregates continue to underperform
- Phase F: not started

### Phase A — Research Governance Baseline
Goal: stabilize execution norms and documentation.

Tasks:
- add repo-local `AGENTS.md`
- keep diagnosis / paper-gap / planning docs current
- standardize experiment aggregation and audit scripts

Exit gate:
- repo guidance exists
- current evidence is documented in one place

### Phase B — Dataset Integrity Repair
Goal: make multi-scene validation possible.

Tasks:
- acquire valid Matterport credentials
- use official Habitat downloader to fetch a clean `hm3d_val_habitat_v0.2` or `hm3d_val_full`
- install into a clean location rather than overlaying the broken bundle blindly
- run loader-only audit over scenes

Artifacts:
- repaired scene bundle path
- loader audit JSON report
- at least 2 known-good cross-scene candidates beyond `MHPLjHsuG27`

Exit gate:
- at least 2 additional scenes pass loader-only validation

### Phase C — Cross-scene Canary Validation
Goal: test whether the current ranking survives outside the current intact scene.

Profiles:
- `baseline-periodic`
- `smoothnav-no-monitor`
- `smoothnav-full`

Tasks:
- pick 1-2 matched episodes per new scene
- run explicit canaries
- compare traces and summaries

Exit gate:
- at least one clean cross-scene comparison set exists
- no loader/data errors remain in the active validation set

### Phase D — Cross-scene Dev Suite
Goal: move from canaries to a broader explicit suite.

Tasks:
- create a cross-scene explicit `devN` list
- run the three main profiles on the same list
- aggregate and compare `SR`, `SPL`, trace-derived control metrics

Exit gate:
- one completed cross-scene aggregate exists
- advantage ranking is stable enough to support a paper claim, or failure modes are clearly localized

### Phase E — Second Full-specific Patch Round (Conditional)
Only enter if cross-scene validation shows `full` losing or becoming unstable.

Tasks:
- use preserved matched divergences (e.g. `ep288`-type cases)
- identify whether the next issue is:
  - planner randomness,
  - pending/prefetch policy,
  - semantic grounding,
  - executor recovery,
  - budget handling
- patch narrowly
- rerun matched cases before re-running full suites

Exit gate:
- repaired matched case(s) and no regression on already-good suites

### Phase F — Paper Asset Consolidation
Goal: convert the now-validated system into submission assets.

Tasks:
- freeze the claim hierarchy
- prepare final experiment tables
- draft figures and failure-taxonomy visuals
- write the English manuscript

Exit gate:
- complete paper draft
- evidence-backed top-tier readiness reassessment

## 3. Recommended Immediate Next Actions

1. Treat repaired cross-scene ranking as the top blocker.
2. Preserve current completed `dev5` and `dev15` intact-scene results as the baseline evidence pack.
3. Only spend the next cycle on controller tweaks when a repaired cross-scene matched comparison isolates a concrete failure mode.

## 4. Minimal Validation Experiments For The Next Stage

1. Finish one deduplicated repaired cross-scene explicit suite with `baseline-periodic`, `smoothnav-no-monitor`, and `smoothnav-full`.
2. Identify the dominant cross-scene divergence mode if `full` still trails or ties.
3. Run one narrow full-specific patch loop and rerun the same explicit repaired suite.
