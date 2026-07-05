# Test Spec — MLLM Planner BEV Decision State

## Static checks
- `python -m py_compile` on changed SmoothNav modules and scripts.

## Unit tests
- Coordinate contract: prompt/state exposes only canonical branch direction fields.
- Room split: unlocalized room priors do not become spatial masks or direct evidence.
- Target value map: synthetic direct/anchor/room-prior/geometry-only cases produce expected evidence levels.
- Prompt contract: evidence level is required; invalid branch IDs rejected.
- Evaluator: missing semantic/value/branch fields are flagged.

## Replay checks
- Replay known capsule steps 31, 43, 98, 145.
- Confirm output artifacts: decision JSON, planner image, evaluator report.
- Confirm step 31 does not claim direct semantic grounding if target/anchor absent.

## Online checks
- Only after replay passes, run one previously failed scene/step, not a full suite.
