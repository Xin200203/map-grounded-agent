# Semantic BEV Branch Decision Evaluation

Offline evidence audit only: no Habitat and no LLM calls.

| step | final | top heat | final rank | sem cats | sem regions | heat source | support | issues |
| ---: | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 31 | B2 | B1 | 2 | 0 | 0 | none | geometry_only_or_graph_only | agent_display_repaired, no_active_semantic_channels, direct_target_channel_empty, no_target_heatmap |
| 43 | B1 | B1 | 1 | 0 | 0 | none | geometry_only_or_graph_only | agent_display_repaired, no_active_semantic_channels, direct_target_channel_empty, no_target_heatmap |
| 98 | B3 | B3 | 1 | 1 | 1 | none | semantic_context_only | agent_display_repaired, direct_target_channel_empty, no_target_heatmap |
| 145 | B4 | B5 | 3 | 4 | 10 | semantic_prior_only | prior_supported_exploration | agent_display_repaired, direct_target_channel_empty, target_heatmap_prior_only, final_branch_not_top_heatmap_branch, mllm_branch_not_top_heatmap_branch |

## Interpretation guide

- `direct_target_channel_empty`: target must be treated as not visible.
- `semantic_prior_only`: target heatmap can guide exploration but is not target grounding.
- `final_branch_not_top_heatmap_branch`: recorded final branch disagrees with target-heat branch value.
