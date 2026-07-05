# PRD: Cross-scene SmoothNav Advantage Recovery

## Objective
Establish that `smoothnav-full` retains a clear advantage over `baseline-periodic` on repaired multi-scene evaluation, not just on the intact single-scene block.

## User stories
- As a researcher, I need a clean repaired cross-scene explicit suite so that I can evaluate whether the system claim generalizes beyond one scene.
- As a system designer, I need to know whether cross-scene degradation comes from implementation bugs or from method-level design choices so that I can patch the right layer.
- As a paper author, I need repaired multi-scene evidence that is strong enough to support a top-tier system-paper claim.

## Acceptance criteria
1. A grounded diagnosis exists for the dominant repaired cross-scene failure mode, with concrete code references.
2. A literature/code survey identifies transferable mitigation patterns from mature navigation systems.
3. At least one narrow code patch with regression tests is implemented.
4. A repaired cross-scene matched suite is rerun on `baseline-periodic`, `smoothnav-no-monitor`, and `smoothnav-full`.
5. The new aggregate shows positive numeric movement for `smoothnav-full` on the repaired multi-scene setting, or a clearly proven blocker is documented.
