# Test Spec: Cross-scene SmoothNav Advantage Recovery

## Regression protections
- Existing intact-scene controller tests remain green.
- Any new logic in text-goal visible-target gating must be covered by unit tests.
- Planner/object fallback changes must be covered by deterministic unit tests where possible.

## Verification commands
1. Focused unit/regression tests for touched modules.
2. Repaired cross-scene matched suite rerun on:
   - `baseline-periodic`
   - `smoothnav-no-monitor`
   - `smoothnav-full`
3. Aggregate comparison against prior repaired suite and prior intact-scene references.

## Success criteria
- No regression on focused local tests.
- Repaired cross-scene `smoothnav-full` shows positive numeric improvement vs the previous repaired suite and moves toward or above baseline.
- Intact-scene previously completed evidence remains the preserved baseline pack.
