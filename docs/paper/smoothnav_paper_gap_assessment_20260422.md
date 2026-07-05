# SmoothNav Paper Gap Assessment (Top-tier Standard)

Date: 2026-04-22
Updated: 2026-04-23
Scope: Assess how far the current SmoothNav project is from a top-tier submission bar, based on the latest explicit intact-scene evidence and repaired cross-scene status.

## 1. Executive Summary

### Observation
Current completed explicit-suite results show that `smoothnav-full` exceeds the semantic baseline on both the completed intact-scene `dev5` and `dev15` comparisons.

### Claim
At the current implementation state, the project has already crossed the threshold from “proof of concept” to “evidence-backed system paper candidate”.

### Constraint
The project is **not yet submission-ready for a top-tier venue**, because although HM3D `val` assets have now been repaired, the core method advantage has not yet been established on a clean repaired cross-scene comparison set.

### Bottom line
- **System viability:** yes
- **Baseline-beating evidence:** yes
- **Top-tier submission readiness:** not yet
- **Primary blocker:** cross-scene advantage is not yet established

## 2. Strongest Current Evidence

### 2.1 Explicit intact-scene dev5
Source: `/mnt/sdd/xxy/SmoothNav/results/phase2_revalidation/s2_dev5_intact_scene/suite_summary.json`

- `baseline-periodic`: `SR=0.4`, `SPL≈0.2071`
- `smoothnav-full`: `SR=0.6`, `SPL≈0.2636`

### 2.2 Explicit intact-scene dev15
Source: `/mnt/sdd/xxy/SmoothNav/results/phase2_revalidation/s3_dev15_intact_scene/suite_summary.json`

- `baseline-periodic`: `SR≈0.5333`, `SPL≈0.2353`
- `smoothnav-no-monitor`: `SR≈0.4667`, `SPL≈0.2416`
- `smoothnav-full`: `SR=0.6`, `SPL≈0.2678`

### 2.3 Interpretation

#### Claim
`Smoothnav-full` is currently the strongest controller-family variant in the completed explicit intact-scene evaluations available so far.

#### Why this matters
This is stronger than a one-off success case:
- it survives beyond the repaired episode-296 loop;
- it survives beyond the smallest explicit suite;
- it persists at the `dev15` scale on the current known-good scene block.

## 3. What Is Already Strong Enough For A Paper

### 3.1 Contribution candidate
The strongest paper framing is currently a **system / engineering contribution**, not a pure benchmark or pure controller-theory paper.

Suggested claim shape:
- a map-grounded semantic replanning system;
- explicit controller observability and diagnosability;
- dual-timescale control structure;
- empirically stronger than the semantic baseline under matched backbone conditions.

### 3.2 What is already compelling
- Strong brownfield execution story: the system was instrumented, audited, repaired, and revalidated.
- Matched ablation setup: same backbone, different controller profiles.
- Explicit traceability: `summary.json`, step traces, planner traces, monitor traces.
- Completed `dev5` and `dev15` intact-scene comparisons with baseline-beating evidence.

## 4. What Is Missing For A Top-tier Submission

### 4.1 Hard blocker: cross-scene advantage

#### Observation
The HM3D `val` asset blocker has been removed with a repaired local archive and successful loader audit on multiple new scenes.

#### Observation
Repaired cross-scene clean suites are now running, but the intact-scene advantage has not yet been shown to transfer robustly outside `MHPLjHsuG27`.

#### Observation
The first repaired clean cross-scene suite (`229`, `527`, `859`) does not yet show a clear `smoothnav-full` lead on `SR`/`SPL`.

#### Claim
The current top-tier blocker is no longer data integrity itself; it is the lack of a completed, clean, repaired multi-scene aggregate demonstrating that `smoothnav-full` remains at least non-inferior, and ideally superior, to the semantic baseline.

### 4.2 Missing evidence layers
Even after the dataset repair, a top-tier paper still needs stronger evidence in at least some of the following dimensions:
- more scenes
- more episodes per scene
- stronger matched cross-scene comparisons
- tighter failure taxonomy at scale
- a clean, narrative-ready experiment table set
- a clearer articulation of why the system framing matters beyond metric deltas

### 4.3 Writing assets not yet ready
This pass intentionally did **not** draft the full English manuscript.
What is still missing on the writing side:
- abstract / intro / method draft
- final experiment tables / figures
- polished limitations section
- final related-work framing against relevant navigation-system papers

## 5. Risk Register

### Risk 1: cross-scene repaired evidence remains negative or inconclusive
Impact: weakens the main system-level claim and could force a narrower submission framing.

### Risk 2: explicit intact-scene advantage may not survive broader repaired multi-scene validation
Impact: could collapse the current margin or flip the preferred profile from `full` to an ablation.

### Risk 3: paper story could become too metric-centric
Impact: weakens the preferred system-paper framing.

## 6. Submission-readiness Verdict

### Claim
Current status is best described as:

**“Promising top-tier system-paper candidate with baseline-beating evidence on intact-scene suites and repaired multi-scene execution, but not yet top-tier ready because cross-scene advantage is still unproven.”**

## 7. Immediate Next Gate

A top-tier-readiness reassessment should only happen after this gate is passed:

1. Finish a clean repaired cross-scene explicit suite with matched profiles
2. Confirm that the current `smoothnav-full` advantage does not collapse outside `MHPLjHsuG27`
3. Convert the resulting evidence into paper-ready tables / figures / claim hierarchy
