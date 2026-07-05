# s23 Semantic BEV Capsule Replay Review

Offline replay only: no Habitat, no LLM API call.

| Step | dry | MLLM | final | agent ok | obj loc | rooms | branch hints | new objs next | warnings |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 31 | B2 |  | B2 | True | 2 | 1 | 3 | 0 |  |
| 43 | B2 | B1 | B1 | True | 2 | 1 | 2 | 1 |  |
| 98 | B1 | B3 | B3 | False | 3 | 1 | 3 | 0 | agent_not_on_known_or_near_known |
| 145 | B4 | B4 | B4 | False | 3 | 1 | 2 | 0 | agent_not_on_known_or_near_known |

## Per-step artifacts

### Step 31

- Frame JSON: `step_000031/semantic_bev_frame.json`
- Planner image: `step_000031/semantic_annotated_bev_planner.png`
- Debug image: `step_000031/semantic_annotated_bev_debug.png`
- Legacy image: `step_000031/legacy_bev_annotated_branches.png`

### Step 43

- Frame JSON: `step_000043/semantic_bev_frame.json`
- Planner image: `step_000043/semantic_annotated_bev_planner.png`
- Debug image: `step_000043/semantic_annotated_bev_debug.png`
- Legacy image: `step_000043/legacy_bev_annotated_branches.png`

### Step 98

- Frame JSON: `step_000098/semantic_bev_frame.json`
- Planner image: `step_000098/semantic_annotated_bev_planner.png`
- Debug image: `step_000098/semantic_annotated_bev_debug.png`
- Legacy image: `step_000098/legacy_bev_annotated_branches.png`

### Step 145

- Frame JSON: `step_000145/semantic_bev_frame.json`
- Planner image: `step_000145/semantic_annotated_bev_planner.png`
- Debug image: `step_000145/semantic_annotated_bev_debug.png`
- Legacy image: `step_000145/legacy_bev_annotated_branches.png`

