# Guided Objective Optimization for the nvMolKit + Nemotron Demo

**Date:** 2026-08-06  
**Status:** Approved design; awaiting written-spec review

## Purpose

Extend the existing objective challenge so Nemotron visibly reasons from quantitative feedback and reliably improves a molecular-diversity panel. The six-stage RDKit-to-MMFF94 workflow remains unchanged. The extension begins only after the current objective card is ready.

The audience must be able to see this scientific loop without reading implementation details:

> observe the limiting pair → compare legal interventions → choose and justify one → execute it → measure improvement → reach or honestly miss the target

Python remains the source of truth for constraints and scores. Nemotron determines the proposed panel and therefore the measured outcome.

## Considered Approaches

### A. Ranked legal counterfactuals — selected

After a missed attempt, Python returns up to three strictly improving, one-molecule swaps with their predicted post-swap score and score delta. Nemotron chooses one and explains it.

This is the best balance of agent decision-making and demonstration reliability. It resembles a scientist using a validated decision-support calculation: the calculation narrows the feasible experiment set, while the scientist chooses the intervention.

### B. Raw similarity evidence only

Nemotron would receive the distance matrix, cluster membership, and limiting pair and derive swaps without computed counterfactuals. This maximizes unconstrained model reasoning but performed poorly in the live notebook: the model repeated the same 0.767 panel three times. It is too fragile for a bounded conference demonstration.

### C. Automatically execute the best swap

Python would select and apply the highest-scoring intervention. This is reliable but makes the outcome primarily algorithmic; Nemotron would merely narrate a choice already made. It does not adequately demonstrate agentic scientific decision-making.

## Scientific Objective and Preserved Boundaries

The objective remains the existing four-compound maximin diversity score:

\[
D_{\min}(P)=\min_{i<j,\;i,j\in P}(1-T_{ij})
\]

The fixed constraints remain:

- exactly four unique supplied molecule IDs;
- every molecule belongs to the bounded eight-candidate pool;
- one molecule per fused Butina cluster;
- all candidates passed RDKit MMFF94 parameter eligibility;
- the retained Morgan/Tanimoto evidence is reused unchanged; and
- at most three accepted agent proposals.

This is structural-library optimization. It does not claim improvement in potency, selectivity, ADMET, efficacy, safety, synthesizability, clinical relevance, or experimental conformation.

## Dataset Calibration

The design was calibrated against the current qualified Brev L4 dataset and retained objective state:

| Quantity | Measured value |
|---|---:|
| Valid four-member panels | 70 |
| Below-target panels | 35 |
| Current observed panel score | 0.7674418688 |
| Best first one-swap score from that panel | 0.8279569894 |
| Target score | 0.8312661529 |
| Best second one-swap score | 0.8472222239 |
| Attainable benchmark | 0.8472222239 |

All 35 below-target starting panels have a target-reaching path within two improving one-molecule swaps. Every one of the top three first moves from each below-target panel preserves at least one target-reaching second move among the next top-three candidates.

These measurements are a dataset qualification receipt, not hard-coded runtime answers. Runtime suggestions must always be recomputed from the current objective context.

## Counterfactual Ranking

Add a pure function in `objective_challenge.py` that accepts the immutable objective context and the latest accepted attempt and returns a tuple of immutable swap suggestions.

Each suggestion contains:

- `replace_id`;
- `replacement_id`;
- the resulting four-molecule panel;
- predicted `D_min`;
- score improvement over the current panel; and
- predicted limiting pair.

The function enumerates every one-molecule replacement, applies the existing panel constraints, evaluates each candidate with the existing score function, and removes moves that do not strictly improve the current score. Results use deterministic ordering: descending predicted score, then replacement identifiers and resulting panel identifiers.

The returned shortlist contains at most three suggestions. If target-reaching moves exist, the shortlist is drawn exclusively from those moves; otherwise it contains the three highest-scoring strictly improving moves. No separate hidden recommendation flag is added. The benchmark panel is never exposed.

No molecule ID, target-reaching panel, or dataset-specific score may appear as a production-code constant.

## Agent Contract and Data Flow

The initial Nemotron proposal remains unconstrained beyond the current objective schema and scientific constraints. This preserves genuine agent ownership of the starting outcome.

After Python evaluates a missed attempt, the tool result sent to the same conversation contains:

- accepted panel and measured score;
- target score;
- measured limiting pair;
- attempts remaining;
- a `legal_improving_swaps` array containing the computed shortlist; and
- an instruction to choose exactly one listed resulting panel and justify the choice using the limiting pair, predicted score, and target.

The next hosted proposal retains the existing public schema:

```json
{
  "selected_ids": ["CHEMBL...", "CHEMBL...", "CHEMBL...", "CHEMBL..."],
  "decision_basis": "Replace the limiting member with the candidate predicted to raise D_min toward the target."
}
```

The controller validates that a revision exactly matches one listed resulting panel before scientific evaluation. Nemotron does not execute Python and cannot alter the similarity matrix, constraints, score function, target, or shortlist.

## Duplicate and Invalid Proposals

A proposal is not an accepted scientific attempt when it:

- repeats any previously accepted panel, regardless of ID order;
- does not match a supplied improving swap after feedback is available;
- contains duplicate, unknown, or cluster-conflicting IDs; or
- fails the strict hosted schema.

Such a proposal receives concise corrective tool feedback and may be retried within a separate bounded hosted-response allowance. It does not consume one of the three accepted scientific attempts and does not appear in the scientific attempt ledger.

The retry allowance must be explicit and finite. If the model never produces a valid revision, the workflow stops safely and reports that no additional scientific attempt was executed. It must not substitute Python's preferred panel or mislabel a rejected proposal as agent reasoning.

## Decision-Ladder Presentation

The user selected the **Decision ladder** visual. It replaces no existing six-stage content and remains inside the current objective card.

Each accepted attempt adds one compact row:

- numbered attempt marker;
- **Observe:** measured limiting pair and prior score;
- **Agent action:** replacement and concise decision basis;
- **Measure:** new `D_min`, signed delta, and target comparison; and
- **Outcome:** `Revise` or `Goal achieved`.

The current row is expanded and earlier rows collapse, preserving the existing interaction pattern. The validated tool-call receipt and deterministic evaluator receipt remain visible in the expanded detail. The existing trajectory, final structures, heatmap, O01 evidence, and evidence-backed conclusion remain unchanged except that the attempt delta and selected counterfactual are added to their inputs where appropriate.

Colors supplement text: neutral gray for the initial state, amber for improvement below target, NVIDIA green for target attainment, and red only for invalid or stopped states.

## State and Termination

The objective loop preserves the existing three accepted-attempt limit. The first proposal may already reach the target. Otherwise, each accepted revision must be strictly improving because it must match the current shortlist.

The loop terminates when:

1. an accepted panel reaches the target;
2. three accepted attempts are exhausted; or
3. bounded hosted retries cannot produce a valid proposal.

The final conclusion preserves the exact termination reason. The controller never relabels an attempt-limit or hosted-response failure as target achievement.

## Files and Responsibilities

- `objective_challenge.py`: immutable swap-suggestion record, deterministic enumeration/ranking, and evidence serialization.
- `demo_agent.py`: shortlist feedback, revision-membership validation, bounded corrective retries, and strict session accounting.
- `interactive_workflow.py`: decision-ladder rows and guarded rendering.
- `objective_receipts.py`: stable receipt for the exact selected intervention and evaluator invocation.
- `tests/test_objective_challenge.py`: ranking, invariants, reachability, and no-hard-coded-answer tests.
- `tests/test_demo_agent.py`: prompt/tool feedback, duplicates, listed-panel enforcement, retries, and turn bounds.
- `tests/test_interactive_workflow.py`: decision-ladder contents, ordering, deltas, and status rendering.
- `tests/test_notebook.py`: unchanged eight-cell notebook contract.

No new notebook code cell, dashboard, chemistry stage, or dependency is introduced.

## Verification Strategy

Development follows red-green-refactor. Tests must fail for the missing behavior before production code changes.

CPU tests verify:

- legal one-swap enumeration is exhaustive and deterministic;
- every suggestion is constraint-valid and strictly improving;
- scores and limiting pairs equal direct evaluation;
- returned suggestions are limited to three;
- no production constant contains a candidate molecule ID or winning panel;
- all below-target panels in the fixed test fixture have a bounded target path;
- repeated panels are canonicalized and rejected without consuming an attempt;
- post-feedback proposals must match a listed resulting panel;
- corrective hosted retries are bounded separately from accepted attempts;
- only accepted panels enter O01 and the decision ladder;
- decision-ladder rows show observation, action, score, delta, target, and outcome;
- success stops immediately and failures remain explicit; and
- all existing workflow, notebook, secret-safety, and evidence tests remain green.

GPU acceptance on the selected Brev L4 environment verifies:

- the complete suite passes with `RUN_GPU_TESTS=1`;
- a fresh real Nemotron run does not repeat an accepted panel;
- every revision matches a controller-supplied legal improving swap;
- the score improves after each accepted miss;
- the target is reached within three accepted attempts on the qualified dataset;
- the live decision ladder updates after each attempt; and
- the final E01-E06 plus O01 conclusion reports the measured outcome accurately.

One successful live run is necessary for demo acceptance but is not evidence of model-wide reliability. The acceptance receipt records model, dataset, environment, target, accepted proposals, scores, and termination reason.

## Acceptance Criteria

The change is complete when the existing notebook runs its six molecular stages unchanged, then visibly shows Nemotron use computed counterfactual evidence to choose a non-duplicate intervention, improve the measured `D_min`, and reach the fixed target within three accepted attempts on the qualified Brev dataset. Python must independently validate every proposal and remain the sole source of quantitative truth. No winning molecule IDs may be hard-coded, and any failure must remain explicit rather than being converted into a scripted success.
