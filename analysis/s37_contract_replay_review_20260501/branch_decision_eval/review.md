# Semantic BEV Branch Decision Evaluation

Offline evidence audit only: no Habitat and no LLM calls.

| step | final | top heat/value/exec | final rank | sem cats | sem regions | heat source | support | issues |
| ---: | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 31 | B2 | B1/B1/B2 | 2/2/1 | 0 | 0 | none Vpeak=1.00 | geometry_only_or_graph_only (geometry_only) | agent_display_repaired, no_active_semantic_channels, direct_target_channel_empty, no_target_heatmap |
| 43 | B1 | B1/B1/B1 | 1/1/1 | 0 | 0 | none Vpeak=1.00 | geometry_only_or_graph_only (geometry_only) | agent_display_repaired, no_active_semantic_channels, direct_target_channel_empty, no_target_heatmap, geometry_only_branch_hard_gated |
| 98 | B3 | B3/B2/B3 | 1/2/1 | 1 | 1 | none Vpeak=1.00 | semantic_context_only (geometry_only) | agent_display_repaired, direct_target_channel_empty, no_target_heatmap, geometry_only_branch_hard_gated |
| 145 | B4 | B6/B6/B4 | 2/2/1 | 3 | 9 | semantic_prior_only Vpeak=1.00 | anchor_supported_exploration (anchor) | agent_display_repaired, direct_target_channel_empty, target_heatmap_prior_only |

## Interpretation guide

- `direct_target_channel_empty`: target must be treated as not visible.
- `semantic_prior_only`: target heatmap can guide exploration but is not target grounding.
- `final_branch_not_top_heatmap_branch`: recorded final branch disagrees with target-heat branch value.
- `selection_contract`: explains target-value vs Graph.get_goal final-score drift when executable scorer/actionability overrides semantic value.
- `geometry_only_branch_hard_gated`: MLLM/scorer coupling used a planner prior despite geometry-only evidence.
