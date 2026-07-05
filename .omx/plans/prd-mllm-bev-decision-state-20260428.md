# PRD — MLLM Planner BEV Decision State

## Problem
The current planner BEV is a debug visualization rather than a stable decision-state representation. It mixes geometry, sparse semantic labels, room text priors, branch labels, and warnings in a way that makes MLLM branch selection fragile and hard to audit.

## Goals
1. Offline replay can isolate planner-state failures without full online reruns.
2. Planner input uses one canonical coordinate frame.
3. Planner BEV/state separates geometry, semantics, target-conditioned value, room spatialization, and branch candidates.
4. Planner output carries explicit evidence levels and can be safely coupled to frontier scoring.
5. Known problematic replay steps expose whether decisions are direct/anchor/room-prior/geometry-only.

## Non-goals
- No full online suite expansion in this implementation pass.
- No new dependencies unless unavoidable.
- No training of external PONI/SemExp-style models.
- No egocentric RGB/panorama integration yet.

## Users
- Experiment lead debugging planner failures.
- Human reviewer inspecting MLLM planner inputs.
- Future online experiment runner deciding whether replay evidence is sufficient.

## Functional Requirements
- Save decision-state JSON for replayed planner frames.
- Render multi-panel BEV for human/MLLM inspection.
- Compute target-conditioned value and branch evidence levels.
- Validate MLLM branch contract with evidence level and target visibility.
- Evaluate replay frames with a scorecard.

## Acceptance
- Steps 31/43/98/145 replay produce decision JSON, image, and evaluator output.
- No mixed coordinate fields enter planner prompt.
- Geometry-only cases cannot hard gate target branch selection.
