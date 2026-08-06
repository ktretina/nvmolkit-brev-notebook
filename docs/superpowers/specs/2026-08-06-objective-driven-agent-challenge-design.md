# Objective-Driven Agent Challenge for the nvMolKit + Nemotron Demo

**Date:** 2026-08-06  
**Status:** Approved direction; implementation boundary awaiting document review

## Objective

Extend the existing notebook with one compact final challenge that demonstrates scientific agency after tool competence. The current six-stage workflow remains visible and unchanged in purpose. After it finishes, Nemotron must use the retained molecular evidence to improve a four-compound follow-up panel against a quantitative structural-diversity objective.

The audience should be able to distinguish four things without reading implementation details:

1. the scientific objective and constraints;
2. Nemotron's structured proposal and concise decision summary;
3. the deterministic Python evaluation that actually ran; and
4. the measured progression from baseline through attempts to a pass or bounded failure.

## Audience Narrative

The notebook has two movements:

1. **Molecular Evidence Generation** — inspect the library, generate Morgan fingerprints, measure Tanimoto similarity, discover fused Butina clusters, embed representative conformers with ETKDGv3, and optimize them with MMFF94.
2. **Objective-Driven Agent Challenge** — use that evidence to improve a constrained experimental follow-up panel.

The final output is labeled **Evidence-Backed Conclusion**. The word "synthesis" is not used as a standalone UI label because an ACS audience could reasonably interpret it as chemical synthesis or route planning.

This is a bounded cheminformatics demonstration. It does not claim optimization of potency, selectivity, ADMET, efficacy, safety, synthesizability, clinical relevance, or experimental structure.

## Scientific Objective

The challenge asks:

> Given a budget of four compounds, select a structurally diverse, MMFF94-parameter-eligible panel for experimental follow-up.

For selected molecule indices \(P\), the diversity score is the minimum pairwise Morgan/Tanimoto distance:

\[
D_{\min}(P)=\min_{i<j,\;i,j\in P}\left(1-T_{ij}\right)
\]

This maximin score makes the closest pair the limiting pair. It has an immediate chemical interpretation: a higher value reduces the worst structural redundancy in the panel.

Constraints are deterministic and fail closed:

- exactly four unique molecule IDs;
- every ID belongs to the controller-supplied candidate pool;
- one candidate per fused Butina cluster;
- every candidate passed RDKit MMFF94 parameter eligibility;
- the existing fingerprint definition and Tanimoto matrix are reused unchanged; and
- at most three hosted agent attempts are accepted.

## Candidate Pool and Baseline

The controller constructs a fixed pool of eight candidates from the eight largest MMFF94-eligible clusters. It takes the same deterministic first-by-source-row eligible representative already used by the current representative-selection policy. Restricting the challenge to eight candidates keeps the proposal schema, pairwise evidence, and final structures readable while still presenting 70 possible four-member panels.

If eight eligible distinct clusters are unavailable, challenge construction fails closed with a specific scientific-eligibility message. MMFF94 parameter eligibility is only a precondition for subsequent conformer work; it does not guarantee embedding, convergence, experimental stability, or biological utility.

The baseline is the first four candidates under the existing `largest_clusters_first` policy. This is a legitimate current-policy baseline, not a deliberately corrupted answer.

For the fixed eight-candidate pool, Python enumerates all 70 valid panels to calculate the attainable maximin benchmark. The benchmark panel itself is never sent to Nemotron. The challenge target is:

\[
D_{target}=D_{baseline}+0.8\left(D_{benchmark}-D_{baseline}\right)
\]

The UI describes this as **80% of attainable improvement over the current policy baseline** and also displays the absolute target score. This makes the target reproducible and feasible across the allowed upstream fingerprint and clustering parameters without inventing an uncalibrated absolute number.

If the baseline already equals the benchmark, the challenge must report that no diversity improvement is available for the bounded pool; it must not manufacture an iteration. GPU acceptance on the fixed launchable dataset must verify that the default demo path has a positive benchmark gap and exercises at least one agent attempt.

## Agent Attempt Loop

After all six stages complete, the UI shows one **Run Objective Challenge** button. This is the only additional presenter action. The challenge then runs automatically for at most three accepted attempts.

Before the first attempt, the same Nemotron conversation receives:

- the objective definition and target;
- the eight candidate IDs and cluster IDs;
- the baseline panel and score;
- the candidate-pool pairwise Tanimoto-distance table; and
- the rule that only concise decision summaries are permitted.

Each accepted proposal uses a strict schema:

```json
{
  "selected_ids": ["CHEMBL...", "CHEMBL...", "CHEMBL...", "CHEMBL..."],
  "decision_basis": "Replace the member of the limiting pair with a candidate farther from the retained panel."
}
```

Python validates the proposal, calculates the exact score, identifies the limiting pair, checks all constraints, and returns the measurement to the same conversation. If the target is not met and attempts remain, Nemotron must propose another panel using that feedback. The loop stops immediately on success. If three accepted attempts do not meet the target, the notebook displays **Objective not achieved within attempt limit** and continues to an evidence-backed conclusion that preserves that outcome.

Malformed, duplicate, out-of-pool, or cluster-duplicating proposals are rejected without scientific execution. Hosted-model retries remain bounded and do not count as accepted scientific attempts. Unexpected internal failures stop the workflow with the existing secret-safe behavior.

## Visible Command and Executed Code

Every attempt shows two separate deterministic receipts:

1. **Validated Nemotron proposal**

```python
select_diverse_panel(selected_ids=[...])
```

2. **Evaluation executed by Python**

```python
result = evaluate_diverse_panel(
    selected_ids=proposal.selected_ids,
    candidate_pool=candidate_pool,
    similarity_matrix=similarity_matrix,
)
```

Nemotron does not generate or execute arbitrary Python. Both receipts are rendered from the validated proposal object and stable templates; tests tie the displayed invocation to the evaluator called by the controller.

## Compact Visual Design

The appended challenge is one card using the existing widget styling. Its first view contains:

- objective statement;
- four constraints;
- baseline score;
- target score and target definition; and
- **Run Objective Challenge**.

While running, the same card updates with:

1. a horizontal score trajectory containing baseline, accepted attempts, and a target line;
2. a compact attempt ledger with columns `Step`, `Panel change`, `D_min`, `Limiting pair`, and `Result`;
3. one expandable attempt detail containing the decision summary, validated proposal, and executed Python receipt; and
4. on completion, four RDKit 2D structure thumbnails for the winning or best observed panel plus a small 4-by-4 Tanimoto heatmap.

The ledger retains all accepted attempts so the progression remains auditable. Only the current attempt detail is expanded; prior details are collapsed. Invalid hosted responses never appear as scientific attempts.

Colors supplement rather than replace text:

- neutral gray for the baseline;
- amber for a valid attempt below target;
- NVIDIA green for a valid attempt at or above target; and
- red only for stopped or invalid states.

No separate dashboard, navigation system, editable code cell, molecular viewer, or new application is added.

## Evidence Model and Conclusion

E01 through E06 retain their current meanings and provenance. The challenge adds one immutable record:

- **O01 — Objective-driven panel selection**

O01 contains:

- objective name and score definition;
- candidate-pool derivation;
- panel size and attempt limit;
- baseline panel and score;
- benchmark score, but not the hidden benchmark panel;
- target rule and target score;
- every accepted proposal, decision summary, score, limiting pair, and constraint result;
- termination reason;
- final/best panel and score; and
- whether the target was achieved.

The final schema-checked conclusion receives E01-E06 and O01. It adds an `objective_driven_selection` section grounded in O01 and must distinguish library coverage from any biological or experimental claim. The UI title becomes **Evidence-Backed Conclusion**.

## State and Turn Bounds

The existing six workflow phases and stage order remain unchanged. Objective state is maintained separately by the controller after `WorkflowPhase.OPTIMIZED` and before conclusion generation.

The accepted hosted-call budget is:

- one workflow plan;
- six stage proposals;
- zero to three objective proposals; and
- one final conclusion.

Thus the maximum accepted hosted turn count is eleven. The controller must require a completed objective state, including the explicit no-improvement case, before requesting the conclusion.

## Files and Responsibilities

- `chemistry_workflow.py` retains the six scientific stages and exposes the validated host similarity matrix and MMFF-eligible cluster candidates needed by the challenge.
- `objective_challenge.py` owns candidate-pool construction, exact scoring, benchmark/target calculation, immutable attempt records, O01 construction, and objective figures.
- `demo_agent.py` owns the strict proposal schema, bounded hosted attempt loop, conversation feedback, turn limits, and conclusion integration.
- `objective_receipts.py` renders the validated proposal and deterministic evaluator invocation.
- `interactive_workflow.py` appends the one-card challenge, live ledger, trajectory, details, final panel, and renamed conclusion.
- `notebooks/nvmolkit_nemotron_demo.ipynb` updates only explanatory copy; the existing public launch call remains the notebook entry point.

## Verification

CPU tests must verify:

- candidate-pool derivation is deterministic and cluster-distinct;
- the score equals the minimum pairwise Tanimoto distance;
- the limiting pair is deterministic under ties;
- the 70-panel benchmark is exact for an eight-candidate pool;
- the target calculation is finite, bounded by baseline and benchmark, and reproducible;
- invalid and duplicate IDs fail before evaluation;
- O01 is canonical JSON and contains no hidden benchmark panel;
- the agent receives only the bounded pool, distance table, baseline, target, and prior attempt feedback;
- every accepted attempt is evaluated once and appended once;
- success stops the loop immediately;
- failure stops after three accepted attempts;
- conclusion cannot start before objective termination;
- receipts contain the exact validated IDs and stable evaluator invocation;
- the widget renders command, code, score, limiting pair, status, structures, and heatmap;
- prior six-stage controls, cards, figures, retries, and evidence still behave as before; and
- notebook structure and copy remain compact.

GPU acceptance on Brev must additionally verify:

- the default fixed dataset produces eight eligible distinct-cluster candidates;
- the default baseline has a positive gap to the attainable benchmark;
- at least one real hosted Nemotron objective proposal is evaluated;
- the challenge reaches the target within three attempts or fails closed without being presented as success;
- the live widget visibly updates between attempts; and
- the final conclusion accurately reports the measured objective outcome.

## Acceptance Criteria

The change is complete when a presenter can run the existing six-stage demonstration unchanged in purpose, click one additional objective-challenge button, and visibly observe Nemotron propose, receive quantitative feedback, revise if needed, and either achieve the target or exhaust the stated bound. Every accepted attempt must expose its validated proposal, deterministic executed-code receipt, measured score, limiting pair, and status. The final panel must be visible as chemical structures, and the conclusion must preserve the exact outcome without extending the scientific claims beyond structural diversity and sampled force-field tractability.
