# s22b BEV Frontier Capsule Review

## Local artifacts
- raw run copy: `analysis/s22b_capsule_review_20260427/raw/run/`
- BEV images: `analysis/s22b_capsule_review_20260427/bev_images/`
- enriched/cropped BEV images: `analysis/s22b_capsule_review_20260427/bev_images_enriched/`
- per-step MLLM prompt/call summaries: `analysis/s22b_capsule_review_20260427/planner_inputs/`
- machine-readable review: `analysis/s22b_capsule_review_20260427/capsule_branch_review.json`

## Summary table
| step | dry | MLLM parsed | MLLM raw | final | changed | subsequent evidence | auto class | BEV image | enriched BEV |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 31 | B2 |  | B2 | B2 | False | no new semantic evidence | same_room_or_dead_end_revisit | `bev_images/step_000031_bev_annotated_branches.png` | `bev_images_enriched/step_000031_bev_enriched.png` |
| 43 | B2 | B1 | B1 | B1 | True | new objects=cabinet | new_semantic_evidence | `bev_images/step_000043_bev_annotated_branches.png` | `bev_images_enriched/step_000043_bev_enriched.png` |
| 98 | B1 | B3 | B3 | B3 | True | no new semantic evidence | new_area_no_semantic_gain | `bev_images/step_000098_bev_annotated_branches.png` | `bev_images_enriched/step_000098_bev_enriched.png` |
| 145 | B4 | B4 | B4 | B4 | False | no new semantic evidence | unclear_no_semantic_gain | `bev_images/step_000145_bev_annotated_branches.png` | `bev_images_enriched/step_000145_bev_enriched.png` |

## BEV image information gap
- Observation: the original saved BEV images mostly contain gray unknown space plus a small free-space patch, robot marker, and branch dots. They do **not** show semantic object positions, room labels, trajectory/history, branch masks, or explicit dead-end/revisit state.
- Observation: this means the current online MLLM image alone is under-informative. The MLLM is effectively relying on the text branch table and stale text strategy more than on map perception.
- Mitigation added in code: `smoothnav/frontier_branching.py` now crops BEV images to the known map/branch content and overlays candidate samples, branch direction, candidate count, score, actionability, legend, and compass. The enriched s22b images were regenerated on the remote runtime and copied into `bev_images_enriched/`.
- Bug found: the original `agent_coord` in branch replay snapshots used the FMM/traversible vertically-flipped row frame, while frontier coordinates and BEV images used the unflipped full-map frame. That is why `A robot` could be drawn outside the room. `base_UniGoal/src/graph/graph.py` now converts the agent row back to the full-map frame before storing branch snapshots; the enriched offline images were overwritten with corrected robot markers from the map current-location channel.
- Remaining gap: even the enriched image is still geometry-only; it does not yet draw object/room/visited/dead-end layers. That should be added before trusting MLLM branch choices online.

## Agent BEV visual screening
> This section is an agent-side visual check of the saved BEV images only. Treat it as a screening note for manual review, not as a final claim.
- step 31: BEV上B2是靠近机器人/主房间内部的小局部frontier；B1是更大的东北侧延伸，B3是西侧出口。若目标不可见且策略是搜索living-room/TV，B2看起来更像same-room局部补洞，不是高价值离房间分支。
- step 43: B1指向东北侧小房间/延伸区，B2指向西侧出口；在文字策略仍为unexplored north时，B1几何上可解释，但它更像继续检查相邻小空间，而不是明显进入living-room的主通道。
- step 98: B1是已访问的东北/北侧分支，B2是主房间内部东侧局部，B3是新出现的西侧出口/楔形区域；MLLM把B1改成B3在BEV上更合理，属于避免回访并尝试新区域。
- step 145: BEV已经展开到西南/南侧新区域，B3/B4/B5/B6都是同一片新区域附近的south分支；B4位于较中心的可行动分支，选择本身可解释。但文字策略仍显示unexplored north，说明策略层没有随拓扑变化重写语义搜索意图。

## Per-frame details

### Step 31
- capsule: `raw/run/task_frame_capsules/episode_000000_step_000031_unexplored_north`
- BEV image: `bev_images/step_000031_bev_annotated_branches.png`
- enriched BEV image: `bev_images_enriched/step_000031_bev_enriched.png`
- dry branch/frontier: `B2` / `[346, 347]`
- MLLM status/applied/parsed branch/raw branch: `invalid` / `False` / `` / `B2`
- final branch/frontier: `B2` / `[346, 347]`
- allowed branches: `['B2', 'B3', 'B1']`
- planner input files: `planner_inputs/step_000031_mllm_prompt.txt`, `planner_inputs/step_000031_mllm_call_summary.json`
- BEV visual screening: BEV上B2是靠近机器人/主房间内部的小局部frontier；B1是更大的东北侧延伸，B3是西侧出口。若目标不可见且策略是搜索living-room/TV，B2看起来更像same-room局部补洞，不是高价值离房间分支。
- subsequent window: steps (31, 43], count=12
- subsequent new captions: `[]`
- subsequent new rooms: `[]`
- event counts: `{'no_progress': 9, 'frontier_near': 1, 'frontier_reached': 1}`
- auto classification: `same_room_or_dead_end_revisit`
- branch summaries:
  - `B1` dir=north rep=[316, 392] cand=11 final_max=7.466044943748477 novelty=0.727810650887574 action=1.0
  - `B2` dir=north rep=[346, 347] cand=13 final_max=14.62469709587104 novelty=0.40828402366863903 action=1.0
  - `B3` dir=west rep=[356, 253] cand=0 final_max=9.415759577180523 novelty=0.7218934911242604 action=1.0
- text planner calls at this step:
  - direction `north` -> `unexplored north`; reason=The target is a television, typically found in a living room. The bedroom has already been searched, so exploring an unknown direction to locate the living room is the logical next step.
- MLLM raw response excerpt:
  `{   "schema_version": "smoothnav.mllm_frontier_plan.v1",   "decision_type": "explore_frontier",   "target_visible": false,   "selected_branch_id": "B2",   "ranked_branches": [     {       "id": "B2",       "score": 1.0,       "rationale": "Closest northern frontier, highest overall score, perfectly aligns with 'unexplored north'26 strategy."     },     {       "id": "B3",       "score": 0.644,       "rationale": "West direction frontier,04 lower score and not aligned with current north explorati`

### Step 43
- capsule: `raw/run/task_frame_capsules/episode_000000_step_000043_unexplored_north`
- BEV image: `bev_images/step_000043_bev_annotated_branches.png`
- enriched BEV image: `bev_images_enriched/step_000043_bev_enriched.png`
- dry branch/frontier: `B2` / `[357, 260]`
- MLLM status/applied/parsed branch/raw branch: `valid` / `True` / `B1` / `B1`
- final branch/frontier: `B1` / `[287, 368]`
- allowed branches: `['B2', 'B1']`
- planner input files: `planner_inputs/step_000043_mllm_prompt.txt`, `planner_inputs/step_000043_mllm_call_summary.json`
- BEV visual screening: B1指向东北侧小房间/延伸区，B2指向西侧出口；在文字策略仍为unexplored north时，B1几何上可解释，但它更像继续检查相邻小空间，而不是明显进入living-room的主通道。
- subsequent window: steps (43, 98], count=55
- subsequent new captions: `['cabinet']`
- subsequent new rooms: `[]`
- event counts: `{'no_progress': 24, 'new_nodes': 1, 'room_object_count_increase': 1, 'frontier_near': 1, 'frontier_reached': 1}`
- auto classification: `new_semantic_evidence`
- branch summaries:
  - `B1` dir=north rep=[287, 368] cand=123 final_max=11.863798435884677 novelty=0.727810650887574 action=1.0
  - `B2` dir=west rep=[356, 253] cand=52 final_max=11.629413993826653 novelty=0.650887573964497 action=1.0
- text planner calls at this exact step: none; MLLM used current carried strategy.
- MLLM raw response excerpt:
  `{   "schema_version": "smoothnav.mllm_frontier_plan.v1",   "decision_type": "explore_frontier",   "target_visible": false,   "selected_branch_id": "B1",   "ranked_branches": [     {       "id": "B1",       "score": 0.8,       "rationale": "Aligns with current 'unexplored north'26 target region; larger frontier area suggests20 possible room transition toward living room."     },     {       "id": "B2",       "score": 0.4,       "rationale": "West direction less relevant to target region; smaller `

### Step 98
- capsule: `raw/run/task_frame_capsules/episode_000000_step_000098_unexplored_north`
- BEV image: `bev_images/step_000098_bev_annotated_branches.png`
- enriched BEV image: `bev_images_enriched/step_000098_bev_enriched.png`
- dry branch/frontier: `B1` / `[313, 395]`
- MLLM status/applied/parsed branch/raw branch: `valid` / `True` / `B3` / `B3`
- final branch/frontier: `B3` / `[367, 243]`
- allowed branches: `['B3', 'B2', 'B1']`
- planner input files: `planner_inputs/step_000098_mllm_prompt.txt`, `planner_inputs/step_000098_mllm_call_summary.json`
- BEV visual screening: B1是已访问的东北/北侧分支，B2是主房间内部东侧局部，B3是新出现的西侧出口/楔形区域；MLLM把B1改成B3在BEV上更合理，属于避免回访并尝试新区域。
- subsequent window: steps (98, 145], count=47
- subsequent new captions: `[]`
- subsequent new rooms: `[]`
- event counts: `{'no_progress': 21, 'frontier_near': 1, 'frontier_reached': 1}`
- auto classification: `new_area_no_semantic_gain`
- branch summaries:
  - `B1` dir=north rep=[313, 395] cand=116 final_max=5.616444352575158 novelty=0.727810650887574 action=1.0
  - `B2` dir=east rep=[342, 352] cand=56 final_max=13.368598113178086 novelty=0.4911242603550296 action=1.0
  - `B3` dir=west rep=[367, 243] cand=28 final_max=15.091330195684312 novelty=0.5207100591715976 action=1.0
- text planner calls at this exact step: none; MLLM used current carried strategy.
- MLLM raw response excerpt:
  `{   "schema_version": "smoothnav.mllm_frontier_plan.v1",   "decision_type": "explore_frontier",   "target_visible": false,   "selected_branch_id": "B3",   "ranked_branches": [     {       "id": "B3",       "score": 1.0,       "rationale": "Highest final score with direction west, unexplored961, and039 no repeated penalty or recent penalty. Promising390 to find transitioning area towards291 living room with TV."     },     {       "id": "B2",       "score": 0.886,       "rationale": "2nd highest `

### Step 145
- capsule: `raw/run/task_frame_capsules/episode_000000_step_000145_unexplored_north`
- BEV image: `bev_images/step_000145_bev_annotated_branches.png`
- enriched BEV image: `bev_images_enriched/step_000145_bev_enriched.png`
- dry branch/frontier: `B4` / `[451, 284]`
- MLLM status/applied/parsed branch/raw branch: `valid` / `True` / `B4` / `B4`
- final branch/frontier: `B4` / `[418, 237]`
- allowed branches: `['B4', 'B3', 'B6', 'B5', 'B2', 'B1']`
- planner input files: `planner_inputs/step_000145_mllm_prompt.txt`, `planner_inputs/step_000145_mllm_call_summary.json`
- BEV visual screening: BEV已经展开到西南/南侧新区域，B3/B4/B5/B6都是同一片新区域附近的south分支；B4位于较中心的可行动分支，选择本身可解释。但文字策略仍显示unexplored north，说明策略层没有随拓扑变化重写语义搜索意图。
- subsequent window: steps (145, 146], count=0
- subsequent new captions: `[]`
- subsequent new rooms: `[]`
- event counts: `{}`
- auto classification: `unclear_no_semantic_gain`
- branch summaries:
  - `B1` dir=east rep=[313, 395] cand=0 final_max=2.7324443525751585 novelty=0.727810650887574 action=0.0
  - `B2` dir=east rep=[343, 352] cand=0 final_max=4.281698409030921 novelty=0.4911242603550296 action=0.0
  - `B3` dir=south rep=[385, 199] cand=54 final_max=16.32897831287072 novelty=0.5266272189349113 action=1.0
  - `B4` dir=south rep=[418, 237] cand=78 final_max=18.05589489157585 novelty=0.6982248520710059 action=1.0
  - `B5` dir=south rep=[459, 162] cand=36 final_max=12.725887011920122 novelty=0.7396449704142012 action=1.0
  - `B6` dir=south rep=[466, 230] cand=48 final_max=15.542382496126164 novelty=0.46153846153846156 action=1.0
- text planner calls at this exact step: none; MLLM used current carried strategy.
- MLLM raw response excerpt:
  `{   "schema_version": "smoothnav.mllm_frontier_plan.v1",   "decision_type": "explore_frontier",   "target_visible": false,   "selected_branch_id": "B4",   "ranked_branches": [     {       "id": "B4",       "score": 1.0,       "rationale": "Highest scored frontier, high actionability, candidate points available, leads south toward potential living room area."     },     {       "id": "B3",       "score": 0.904,       "rationale": "Also South, high actionability,173 high224 base score,32 maybe394 `
