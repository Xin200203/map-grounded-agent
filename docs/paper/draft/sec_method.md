# Method 节草稿 v0

## 3. System

SmoothNav layers an event-driven semantic controller over a frozen zero-shot navigation backbone (UniGoal: BEV mapping, incremental scene graph, FMM local executor). The design premise: LLM/VLM outputs are *hypotheses with evidence*, never commands — they bias map-grounded value, and every authority they receive is gated, traced, and revocable.

### 3.1 Layered controller

Six single-writer layers, each with an explicit contract (Fig. 1):

- **L0 WorldState** — per-step assembly of pose, frontier set, graph delta (new nodes/rooms/caption changes/target-like candidates), executor state; monotone `world_epoch`.
- **L1 Mission/Belief** — task spec, evidence ledger with confidence/freshness, stale-plan rejection by epoch mismatch.
- **L2 Semantic planner** — an LLM chooses from a *resolvable menu* (observed objects filtered by target relevance ≥ 0.75, rooms with computable centroids, cardinal directions). Output is a structured strategy (type, id, reasoning), never coordinates. Deterministic fallbacks cover empty/unparseable responses (target-aware first, geometric second).
- **L3 Tactical arbiter** — heuristic-first mode selection from graph-delta events: initial-plan, target-candidate, frontier-reached, pending adoption, recovery (stuck), hold (no frontiers). Emits replan/monitor triggers with reasons; all triggers are budget-governed (8 planner calls / 100 steps).
- **L4 Geometric grounder** — projects a strategy onto the map: direct object execution only if relevance and local-window projectability both pass; otherwise the object becomes a *search anchor* whose deterministic target-progress term (a PONI-lite potential with FMM-with-Euclidean-fallback distances) keeps frontier scoring target-conditioned; rooms ground to object-cluster centroids; directions to frontier-cluster means. Every failure is coded (`out_of_local_window`, `get_goal_none`, `no_frontiers`, noop types) and mapped to a recovery policy (window re-centering + re-projection; hold; replan after noop thresholds).
- **L5 Executor adapter** — wraps the unmodified FMM executor; strategy epochs clear stale temporary/stuck goals; recovery overrides are permission-gated by the controller; every adoption is traced (source, override duration, escalation flags).

Frontier value composes map-derived terms with semantic biases:
score = base + w_bias·semantic-bias + w_tp·target-progress + w_novel·unknown-density + w_act·local-actionability − w_rep·repeat − w_recent·recent, with raw-frontier and relaxed-distance fallbacks guaranteeing liveness.

### 3.2 Graded semantic authority

A single principle applied at four surfaces: (i) the planner menu admits only relevance-gated, resolvable choices; (ii) direct execution demands relevance ∧ projectability, else authority downgrades to a bias; (iii) anchors must demonstrate progress — a stall counter decommits anchors that stop improving (patience 4); (iv) optional VLM branch priors are scaled by declared evidence level (direct 1.0 / anchor 0.85 / room-prior 0.55 / geometry-only 0) and may only re-rank deterministic branch IDs, never emit coordinates. Removing any single gate is absorbed by the layers beneath (§4.5) — the architecture is deliberately defense-in-depth.

### 3.3 Observability

Every layer writes structured traces (planner/monitor calls with prompts and parse outcomes, grounding snapshots replayable without simulator or LLM, frozen task-frame capsules, per-step layered state). A 14-layer contract auditor and counterfactual replay tools attribute any failure to a specific boundary. All Findings in §4 are computed from these artifacts; the tooling is released.

### 3.4 Ablation surface

Each mechanism has an independent switch (scheduling policy, recovery, monitor policy, prefetch, four gates), yielding the matched profile family evaluated in §4 — the point of the architecture is not any single novel mechanism but that *every mechanism's contribution is measurable*.
