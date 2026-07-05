# Deep Interview Spec: SmoothNav Progress, Paper Gap, and Phased Plan

## Metadata
- Profile: standard
- Rounds: 6
- Final ambiguity: 0.19
- Threshold: 0.20
- Context type: brownfield
- Context snapshot: /Users/xin/Research/Code/research/SmoothNav/.omx/context/progress-paper-gap-plan-20260422T031840Z.md
- Transcript: /Users/xin/Research/Code/research/SmoothNav/.omx/interviews/progress-paper-gap-plan-20260422T033303Z.md

## Clarity Breakdown
- Intent clarity: high
- Outcome clarity: high
- Scope clarity: medium-high
- Constraint clarity: high
- Success-criteria clarity: medium-high
- Context clarity: high

## Intent (why this is wanted)
The immediate need is to convert the current research/engineering progress into a top-tier-paper-oriented gap assessment and execution roadmap, instead of continuing ad hoc experimentation without a shared definition of “submission-ready”.

## Desired Outcome
Produce a grounded, brownfield-aware package that does all of the following:
1. summarizes the latest project status accurately;
2. assesses how far the current state is from a top-tier-paper bar;
3. identifies the hard blockers vs optional improvements;
4. lays out a detailed phased, executable roadmap;
5. creates a repo-local AGENTS.md / agent-governance artifact to stabilize future execution and collaboration norms.

## In Scope
- Review current code / docs / experimental evidence.
- Create or update a **repo-local AGENTS.md** at the project root.
- Create a detailed paper-gap document.
- Create a detailed phased execution plan.
- Use the current experiment evidence as the basis for the paper-gap judgment.
- Treat dataset integrity and validation coverage as first-class research blockers.

## Out of Scope / Non-goals
- Do **not** draft the full English paper manuscript in this pass.
- Do not treat cross-scene data corruption as ignorable.
- Do not silently redefine the project as a pure benchmark-chasing effort.

## Decision Boundaries
The next execution lane may decide, without asking again:
- the exact filename and structure of the repo-local AGENTS.md, as long as it lives at the project root and governs experiment / writing workflow;
- the exact filenames for the paper-gap and phased-plan docs;
- the structure of the phased plan, provided it is detailed, staged, and actionable.

The next execution lane should **not** decide without surfacing clearly in the resulting docs:
- that cross-scene data repair is a hard blocker for top-tier-readiness assessment;
- whether current evidence is sufficient for a top-tier submission without cross-scene repair (the default assumption here is **no**).

## Constraints
- Evaluate against a **top-tier paper** standard, not only a workshop or rough-draft bar.
- Cross-scene HM3D data corruption is a **must-fix** blocker before claiming serious submission readiness.
- Keep the current pass focused on planning, governance, and readiness assessment rather than more algorithm implementation.
- Preserve brownfield evidence and cite concrete local artifacts.

## Core Claim Preference
The preferred paper center-of-gravity is the **engineering/system claim**, not purely a controller metric claim.

Interpretation:
- The paper should be organized around a system story: a map-grounded, observable, reproducible, validated semantic-replanning system.
- Controller gains matter, but they support the system paper rather than replacing it.

## Testable Acceptance Criteria
A successful next-pass deliverable must include all of the following:
1. A repo-local AGENTS.md exists and reflects the current research execution contract.
2. A paper-gap document exists and explicitly answers:
   - what is already strong enough,
   - what is still missing for top-tier submission,
   - what is blocked by dataset integrity,
   - what evidence currently supports the system claim.
3. A phased implementation / experiment / writing plan exists with:
   - stage goals,
   - dependencies,
   - blockers,
   - gating criteria,
   - recommended immediate next steps.
4. The documents use current experiment artifacts (dev5/dev15 and cross-scene blocker evidence) rather than generic statements.

## Assumptions Exposed + Resolutions
- Assumption: “submission-ready” might mean workshop or arXiv readiness.
  - Resolved: user wants a top-tier-paper bar.
- Assumption: the main contribution should be a controller gain claim.
  - Resolved by pressure pass: preferred framing is a system / engineering claim.
- Assumption: agent/governance docs might be optional.
  - Resolved: repo-local AGENTS.md should be created/updated in this pass.
- Assumption: cross-scene data issues might be treated as a later cleanup item.
  - Resolved: must-fix blocker before serious top-tier-readiness assessment.

## Brownfield Evidence vs Inference Notes
### Evidence-backed
- smoothnav-full currently exceeds baseline on completed explicit dev5 and dev15 intact-scene suites.
- cross-scene validation is currently blocked by corrupted HM3D val scene assets and a truncated local fallback archive.
- official Habitat dataset downloader exists and supports HM3D val bundles.

### Inference-backed
- the paper should be framed as a system paper rather than a pure benchmark paper, because that best matches the user’s chosen core-claim preference and the current asset mix.

## Recommended Handoff
Primary recommendation: **$ralplan**

Reason:
- requirements are now explicit enough;
- the next task is document/planning heavy;
- we need a consensus-quality paper-gap and staged plan before more execution branches.

Suggested downstream contract:
- input artifact: /Users/xin/Research/Code/research/SmoothNav/.omx/specs/deep-interview-progress-paper-gap-plan.md
- required outputs:
  - repo-local AGENTS.md
  - paper-gap assessment doc
  - phased implementation / experiment / writing roadmap
