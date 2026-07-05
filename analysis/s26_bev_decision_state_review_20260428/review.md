# s23 Semantic BEV Capsule Replay Review

Offline replay only: no Habitat, no LLM API call.

| Step | dry | MLLM | final | agent ok | sem cats | sem regions | target heat | branch hints | new objs next | warnings |
| ---: | --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| 31 | B2 |  | B2 | True | 0 | 0 | none:0 | 3 | 0 | agent_display_coord_repaired_from_map_current_channel, raw_agent_coord_disagrees_with_map_current_channel:25.0, semantic_channels_present_but_empty, direct_target_semantic_channel_empty, target_value_geometry_only |
| 43 | B2 | B1 | B1 | True | 0 | 0 | none:0 | 2 | 1 | agent_display_coord_repaired_from_map_current_channel, raw_agent_coord_disagrees_with_map_current_channel:25.0, semantic_channels_present_but_empty, direct_target_semantic_channel_empty, target_value_geometry_only |
| 98 | B1 | B3 | B3 | True | 1 | 1 | none:0 | 3 | 0 | agent_display_coord_repaired_from_map_current_channel, raw_agent_coord_disagrees_with_map_current_channel:113.0, direct_target_semantic_channel_empty, target_value_geometry_only |
| 145 | B4 | B4 | B4 | True | 4 | 10 | semantic_prior_only:289 | 6 | 0 | agent_display_coord_repaired_from_map_current_channel, raw_agent_coord_disagrees_with_map_current_channel:93.0, direct_target_semantic_channel_empty, target_heatmap_uses_prior_not_direct_target |

## Target heatmap branch rankings

- Step 31: B1:H=0.00/A=0.00, B2:H=0.00/A=0.00, B3:H=0.00/A=0.00
- Step 43: B1:H=0.00/A=0.25, B2:H=0.00/A=0.10
- Step 98: B3:H=0.00/A=0.25, B2:H=0.00/A=0.22, B1:H=0.00/A=0.09
- Step 145: B5:H=0.90/A=0.90, B6:H=0.72/A=0.82, B4:H=0.72/A=0.73, B3:H=0.00/A=0.23, B1:H=0.00/A=0.09, B2:H=0.00/A=0.09

## Target-conditioned value branch rankings

- Step 31: B1:V=1.00/geometry_only, B2:V=1.00/geometry_only, B3:V=0.23/geometry_only; counts={'direct': 0, 'anchor': 0, 'room_prior': 0, 'geometry_only': 3}
- Step 43: B1:V=1.00/geometry_only, B2:V=1.00/geometry_only; counts={'direct': 0, 'anchor': 0, 'room_prior': 0, 'geometry_only': 2}
- Step 98: B2:V=1.00/geometry_only, B3:V=1.00/geometry_only, B1:V=0.05/geometry_only; counts={'direct': 0, 'anchor': 0, 'room_prior': 0, 'geometry_only': 3}
- Step 145: B6:V=1.00/anchor, B4:V=0.48/anchor, B3:V=0.33/geometry_only, B5:V=0.33/geometry_only, B2:V=0.01/geometry_only, B1:V=0.00/geometry_only; counts={'direct': 0, 'anchor': 2, 'room_prior': 0, 'geometry_only': 4}

## Per-step artifacts

### Step 31

- Frame JSON: `step_000031/semantic_bev_frame.json`
- Decision state JSON: `step_000031/planner_decision_state.json`
- Decision state fields: `step_000031/planner_decision_state_fields.npz`
- Planner image: `step_000031/semantic_annotated_bev_planner.png`
- Decision-state image: `step_000031/decision_state_bev_planner.png`
- Semantic field panel: `step_000031/semantic_field_panel.png`
- Target value field panel: `step_000031/target_value_field_panel.png`
- Debug image: `step_000031/semantic_annotated_bev_debug.png`
- Legacy image: `step_000031/legacy_bev_annotated_branches.png`

### Step 43

- Frame JSON: `step_000043/semantic_bev_frame.json`
- Decision state JSON: `step_000043/planner_decision_state.json`
- Decision state fields: `step_000043/planner_decision_state_fields.npz`
- Planner image: `step_000043/semantic_annotated_bev_planner.png`
- Decision-state image: `step_000043/decision_state_bev_planner.png`
- Semantic field panel: `step_000043/semantic_field_panel.png`
- Target value field panel: `step_000043/target_value_field_panel.png`
- Debug image: `step_000043/semantic_annotated_bev_debug.png`
- Legacy image: `step_000043/legacy_bev_annotated_branches.png`

### Step 98

- Frame JSON: `step_000098/semantic_bev_frame.json`
- Decision state JSON: `step_000098/planner_decision_state.json`
- Decision state fields: `step_000098/planner_decision_state_fields.npz`
- Planner image: `step_000098/semantic_annotated_bev_planner.png`
- Decision-state image: `step_000098/decision_state_bev_planner.png`
- Semantic field panel: `step_000098/semantic_field_panel.png`
- Target value field panel: `step_000098/target_value_field_panel.png`
- Debug image: `step_000098/semantic_annotated_bev_debug.png`
- Legacy image: `step_000098/legacy_bev_annotated_branches.png`

### Step 145

- Frame JSON: `step_000145/semantic_bev_frame.json`
- Decision state JSON: `step_000145/planner_decision_state.json`
- Decision state fields: `step_000145/planner_decision_state_fields.npz`
- Planner image: `step_000145/semantic_annotated_bev_planner.png`
- Decision-state image: `step_000145/decision_state_bev_planner.png`
- Semantic field panel: `step_000145/semantic_field_panel.png`
- Target value field panel: `step_000145/target_value_field_panel.png`
- Debug image: `step_000145/semantic_annotated_bev_debug.png`
- Legacy image: `step_000145/legacy_bev_annotated_branches.png`

