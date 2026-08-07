# Constrained Decision Ladder and Evidence-Controlled Conclusion

**Date:** 2026-08-07
**Status:** Approved; quantized Step-0 edge clarified during implementation

## Purpose

Make the final nvMolKit + Nemotron challenge both visibly agentic and consistently achievable by the hosted LLM. The audience must see the model maintain state across a measured optimization loop, acknowledge the supplied bottleneck, select the evidence-optimal action from an unranked deterministically evaluated menu, respond to feedback, and stop after measured success. The model must not author numerical or mechanistic claims that can contradict the retained evidence.

This design is a narrow refinement of `2026-08-06-objective-driven-agent-challenge-design.md`. It preserves the six-stage molecular evidence workflow, maximin Morgan/Tanimoto objective, eight-candidate pool, deterministic evaluator, attempt visuals, O01 evidence record, and one-card notebook extension. It supersedes that document's free-form objective `decision_basis` and free-form evidence-backed conclusion prose.

## Audience-Safe Scientific Claim

The success-state accomplishment is:

> Nemotron correctly follows a bounded evidence-optimization policy: at each measured state, it selects the maximal-scoring legal substitution from an unranked, deterministically evaluated menu and reaches the predefined minimum pairwise Morgan/Tanimoto-distance target.

This is constrained evidence-based action selection, not open-ended molecular design. Python has already computed the consequences of each offered action. Nemotron must attend to those results, select the argmax without position cues, preserve the current state across turns, and invoke the correct tool action. This claim appears only when a measured receipt proves target success. Correction exhaustion, evaluator failure, objective-action hosted failure, or attempt-limit failure receives a distinct bounded-failure conclusion.

The notebook does not claim potency, activity, selectivity, ADMET, efficacy, safety, synthesizability, route feasibility, experimental stability, or clinical relevance. MMFF94 eligibility remains a computational precondition, not evidence that a molecule can be made or is biologically useful.

## Responsibility Boundary

Nemotron is responsible for the bounded policy decision:

- acknowledge the complete set of current co-limiting pairs supplied by the evaluator;
- compare the supplied deterministic scores for up to three legal alternatives;
- select an offered action with maximal predicted `D_min`; and
- preserve and invoke the fixed rule `maximize_predicted_minimum_distance`.

Deterministic Python is responsible for scientific truth:

- construct and validate the candidate pool;
- compute every Morgan/Tanimoto distance;
- identify all current and resulting co-limiting pairs;
- enumerate, score, and validate legal one-molecule substitutions;
- determine the target comparison and stopping condition;
- render all numerical explanations; and
- construct every factual conclusion statement from retained evidence.

The model never generates arbitrary Python, numerical scores, deltas, target comparisons, convergence counts, cluster counts, molecule counts, molecule identifiers in prose, or force-field claims. The existing upstream parameter proposals continue to demonstrate tool selection and parameter use; the final challenge specifically demonstrates stateful policy adherence and action selection.

## Baseline and Objective

The scientific objective remains

\[
D_{\min}(P)=\min_{i<j,\;i,j\in P}(1-T_{ij})
\]

for a panel of four molecules. The baseline remains the first four eligible candidates under the existing `largest_clusters_first` policy. The target remains 80% of the improvement from that baseline to the exact best valid four-member panel in the bounded eight-candidate pool.

The baseline is displayed as **Step 0 — Measured baseline**. It is not counted as an LLM attempt. Before enabling the challenge, the controller runs the production action builder from the baseline, finds every accepted maximum in that exact offered set, and branches over every accepted `swap_id`. It repeats this process from each resulting panel for at most three accepted substitutions. The challenge is eligible only if every branch reaches the target within that bound. Canonical ordering controls traversal and display only; it never collapses accepted tied branches. Otherwise challenge construction stops with a specific eligibility message, and the target is never weakened after the model begins.

The default Brev dataset and allowed upstream parameter path must pass this reachability certificate during acceptance. A baseline that is already optimal retains the existing explicit no-improvement outcome and makes no hosted objective call. A distinct numerical edge is also handled explicitly: if the measured baseline and derived target have the same shared score key even though the attainable benchmark has a larger key, the run terminates as `target_achieved` at Step 0 with zero accepted substitutions and no hosted objective call. It is not mislabeled `baseline_already_optimal`; the measured baseline itself satisfies the predefined target at the declared decision precision.

## Shared Numerical Semantics

All decision logic uses one integer score key while retaining the original float64 value for evidence:

\[
q(x)=\left\lfloor x\times10^{12}+0.5\right\rfloor
\]

for finite scores in `[0, 1]`. The same `q` function governs every component:

- a substitution improves the panel only when `q(candidate) > q(current)`;
- two scores are tied only when their score keys are equal;
- an offered action is maximal when its key equals the largest offered key;
- the target is achieved when `q(score) >= q(target)`;
- action inclusion is ordered by descending score key and then canonical `swap_id`; and
- reachability certification calls the same production functions rather than reimplementing these rules.

Display precision must never visually reverse a score-key comparison. When score keys are equal, the UI explicitly labels the values **tied at 1e-12 decision precision**. When score keys differ, the UI renders enough digits to show their ordering and never labels them tied. Tests cover values immediately below, on, and above every comparison boundary.

Panel scoring retains every co-limiting pair whose pairwise distance has the minimum score key. Each molecule pair is ordered lexicographically, and the complete tuple of pairs is ordered lexicographically. A strictly improving one-molecule substitution must affect every prior co-limiting pair; otherwise an unchanged pair would preserve the prior minimum. The controller asserts this invariant rather than relying on one arbitrarily chosen pair.

## Offered Action Set

After each measured state below target, Python enumerates every legal one-molecule substitution from the current panel. A legal substitution must:

- retain exactly four unique supplied molecule IDs;
- retain four distinct fused Butina clusters;
- retain MMFF94 parameter eligibility;
- replace exactly one current panel member;
- strictly improve the current quantized `D_min` score key; and
- have canonical provenance back to the current measured panel.

The controller orders legal substitutions by descending score key and then canonical `swap_id`, and selects the first three when at least three exist. This gives deterministic inclusion when more than three actions share a boundary score. Before presentation, the selected actions are re-ordered by `swap_id`, not by score, so visual position does not reveal the answer. Accepted maxima are then computed from this exact offered set using the shared score key; certification branches over every offered action with that maximal key.

Each offered action shows factual, controller-generated fields:

- `swap_id`;
- molecule removed and molecule introduced;
- resulting panel;
- predicted `D_min`;
- predicted score change;
- predicted co-limiting pairs; and
- predicted target status.

These are exact evaluations of retained fingerprints, not model forecasts. The UI labels them **deterministically evaluated candidate actions** rather than experimental predictions.

Every action menu also has an immutable `state_id`, computed from canonical JSON containing the current panel, current score key, canonical co-limiting pairs, accepted-attempt number, and exact ordered offered `swap_id` values. The controller stores one pending state revision before the hosted request. A response is eligible only while that exact revision remains pending and current; a repeated substitution from a later panel cannot validate against an earlier `state_id`. Because `swap_id` uses the literal `replace_id->replacement_id` form, candidate molecule IDs containing the reserved `->` delimiter are rejected during context construction so action identity remains injective.

If fewer than three improving substitutions exist, all available substitutions are shown. If none exists before the target is reached, the challenge terminates truthfully as `no_legal_improving_swap`.

## Structured Agent Decision

The hosted tool is renamed conceptually from an open panel proposal to a constrained next-action selection. Its strict payload contains no free-form prose or floating-point values:

```json
{
  "state_id": "state-4f32...",
  "swap_id": "CHEMBL_A->CHEMBL_B",
  "observed_limiting_pairs": [["CHEMBL_C", "CHEMBL_D"]],
  "decision_rule": "maximize_predicted_minimum_distance"
}
```

Provider-visible schema constraints bind `state_id` to the one pending revision, bind `swap_id` to that revision's currently offered actions, bind `observed_limiting_pairs` to the exact canonical tuple of current measured co-limiting pairs, and bind `decision_rule` to its single supported literal. Local validation independently repeats those checks before any action is accepted.

An accepted scientific action must select an offered substitution whose predicted score key is maximal among the exact offered set. The selected substitution must affect every current co-limiting pair; this follows from strict improvement but is asserted explicitly for auditability.

The controller maintains five explicit counters:

- `accepted_attempt_count`: measured substitutions committed to the scientific ledger, bounded from zero through three;
- `rejected_selection_count`: hosted objective selections rejected before chemistry, bounded from zero through two; and
- `correction_prompts_sent`: retry prompts issued after a rejection, bounded from zero through one;
- `selection_response_count`: accepted plus rejected assistant selection responses, bounded from zero through five; and
- `provider_request_attempt_count`: network requests for objective selection, including a transport retry that produces no assistant response, bounded from zero through six.

If the model chooses a valid but nonmaximal action, reports stale or incorrect co-limiting pairs, selects an unavailable action, violates the decision-rule literal, calls the wrong tool, or returns malformed arguments, the controller appends a paired rejected tool result and increments `rejected_selection_count`. The first rejection sends one concise retry prompt containing only the same action table, current co-limiting pairs, fixed decision rule, and one remaining rejection. The second rejection appends its paired terminal result and stops immediately as `objective_correction_limit`; it sends no second retry prompt and permits no third rejected call. Neither rejection consumes an accepted scientific attempt.

Thus the only two-rejection terminal sequence is `invalid -> correction -> invalid -> stop`. Rejections may also be interleaved with accepted measured attempts, but the total rejection count never resets. Provider transport failures that produce no assistant tool call increment only `provider_request_attempt_count` and use at most one UI retry across the whole objective loop. Any assistant tool call with an invalid tool name or arguments increments both `selection_response_count` and `rejected_selection_count` because it must receive a paired tool result. Any accepted assistant selection increments `selection_response_count`; it increments `accepted_attempt_count` only after a measurement commits.

Any interim free-form objective-rationale validation is superseded by this design. Implementation must replace `decision_basis` for objective actions rather than extend or display model-authored rationale text. The six upstream stage rationales remain unchanged.

## Closed-Loop Execution

For every schema-valid, policy-valid action:

1. the notebook displays the exact structured Nemotron choice as **validated selection**;
2. the stable receipt displays the corresponding planned `select_next_panel_swap(...)` command;
3. deterministic Python computes a prospective attempt without mutating controller state;
4. Python validates the prospective measurement, O01 fragment, and paired success tool result;
5. the controller commits the measurement, increments `accepted_attempt_count`, and appends the success tool result in one guarded transition; and
6. the controller either stops on success or constructs the next action set.

If evaluation or prospective evidence construction fails before commit, no scientific attempt is consumed. The controller appends a paired error tool result, marks the validated selection **evaluation not completed**, and stops safely; it does not request a new model choice or claim that the command executed. A state-commit invariant failure also stops safely and must never leave a measured attempt without its paired success result. A rendering failure after commit does not rerun chemistry or the model: the immutable measurement remains authoritative and the UI may be reconstructed from it.

At most three accepted substitutions are evaluated. Success stops immediately. A run that remains below target after three accepted substitutions terminates as `attempt_limit_reached` and is never presented as success.

The model must select an evidence-optimal action from supplied evaluated alternatives; the controller never substitutes the correct `swap_id` on its behalf. Python has computed the action menu and determines whether the selected action worked. The notebook describes this division explicitly rather than presenting the model as the numerical optimizer.

## Visible Decision Ladder

The existing compact challenge card remains. Its ledger becomes:

| Step | Observe | Agent decision | Deterministic measurement | Outcome |
|---|---|---|---|---|
| Baseline | Current panel and co-limiting pairs | — | Baseline `D_min` | Below target or already optimal |
| Attempt 1 | Prior `D_min`, co-limiting pairs, up to three unranked actions | Selected `swap_id` and decision rule | New `D_min`, delta, new co-limiting pairs | Revise or achieved |
| Attempt 2 | Updated measured state and actions | Selected next `swap_id` | New measurement | Revise or achieved |
| Attempt 3 | Updated measured state and actions | Selected next `swap_id` | Final measurement | Achieved or bounded failure |

The expanded attempt card contains, in this order:

1. **Observe** — prior measured panel, score, target, and all co-limiting pairs;
2. **Candidate actions** — the compact unranked action table;
3. **Nemotron choice** — validated structured payload and rendered command;
4. **Execute** — stable Python evaluator receipt; and
5. **Measure** — actual score, delta, all co-limiting pairs, constraint status, and target comparison.

The renderer generates the explanatory sentence from validated objects, for example:

> Nemotron selected S2, an offered action affecting every measured co-limiting pair. Deterministic evaluation increased `D_min` from 0.767 to 0.828; the 0.831 target remains unmet.

No model-authored prose is interpolated into that sentence. A validated selection without a committed measurement is rendered separately as **evaluation not completed** and never receives a score or success color. Existing trajectory, molecule structures, final-panel heatmap, neutral/amber/green status colors, collapsed prior attempts, and single presenter button are retained.

## Evidence-Controlled Conclusion

E01 through E06 and O01 remain the evidence ledger. The final **Evidence-Backed Conclusion** no longer accepts free-form headline or section prose from the model.

Python constructs a catalog of immutable findings from the validated evidence. Each finding contains:

- a stable `finding_id`;
- one theme;
- required evidence keys;
- deterministic display text; and
- a testable predicate proving that the text is true for the current evidence.

The seven themes remain `dataset_scope`, `molecular_representation`, `similarity_structure`, `clustering`, `conformational_sampling`, `objective_driven_selection`, and `limitations_and_next_steps`. The one hosted finding-selection call uses this strict shape:

```json
{
  "ordered_finding_ids": ["D01", "M02", "S01", "C02", "F01", "O02", "L01"]
}
```

The schema item enum contains only predicate-true catalog IDs for the current evidence and requires exactly seven entries. Local validation requires seven unique IDs, exactly one ID from each theme, and no free-form fields; array order is the requested display order. Python repeats every finding predicate immediately before rendering.

Where the UI claims that Nemotron selected an emphasis, the default catalog must expose at least two predicate-true alternatives for that theme. A theme with only one valid finding is labeled **required measured finding** rather than agent selected; Nemotron may still place it in the display order. Acceptance requires at least four themes, including `objective_driven_selection`, to expose two or more valid alternatives on the fixed dataset.

All measured summaries are rendered independently of the model selection, including:

- valid molecule count;
- Morgan fingerprint radius and bit length;
- clustering cutoff, cluster count, and singleton count;
- candidate-pool size and distinct-cluster count;
- final-panel size and distinct-cluster count;
- final `D_min`, target, margin, and all co-limiting pairs;
- corresponding co-limiting-pair Tanimoto similarities;
- generated and converged conformer counts; and
- the qualification that MMFF94 energies are comparable only among sampled conformers of the same molecule.

Nemotron therefore determines which evidence-backed finding to foreground where multiple predicate-true alternatives exist and how to order the seven themes, while Python owns every factual statement. There is one finding-selection call and no semantic retry. A malformed, duplicate, omitted, cross-theme, stale, or predicate-invalid response receives a paired rejected tool result. A finding-selection transport failure without a tool call becomes `finding_selection_unavailable` and follows the same display fallback. The notebook then renders the complete deterministic measured summary, uses a deterministic success or bounded-failure headline from the measured objective receipt, and marks agent-selected emphasis unavailable. It does not discard the completed scientific workflow.

The success headline and accomplishment claim are available only when O01 records measured target achievement. `objective_correction_limit`, `evaluation_not_completed`, `attempt_limit_reached`, `no_legal_improving_swap`, `objective_provider_failure`, and baseline-already-optimal states each receive distinct deterministic wording that preserves the actual outcome. A failed conclusion call is instead `finding_selection_unavailable`: it never downgrades O01-proven success, changes the scientific headline, or removes measured results; it only removes the agent-selected-emphasis claim.

## State, Turn, and Failure Bounds

The six scientific phases remain unchanged. Objective state remains separate after `WorkflowPhase.OPTIMIZED`.

Hosted calls remain bounded to:

- one workflow plan;
- six upstream stage proposals;
- zero to three accepted objective-action selections;
- zero to two rejected objective selections, with only the first followed by a correction retry; and
- one evidence-finding selection.

The objective loop therefore permits at most five hosted assistant selection responses: three accepted plus two rejected. A second rejection always terminates the objective, so a third rejection cannot occur. Separately, it permits at most six provider request attempts because one transport failure may receive one UI retry without producing an assistant response. All five counters are checked independently against their bounds before every call and transition.

Rejected selections and validated-but-unevaluated selections do not enter the measured scientific attempt ledger. Every hosted assistant tool call retains a paired success, rejection, or evaluation-error tool result. Provider, schema, controller, evaluator, state-commit, rendering, and conclusion failures remain distinguishable and secret safe.

## File Responsibilities

- `objective_challenge.py` owns action enumeration, stable action identifiers, reachability certification, exact scoring, immutable attempts, and evidence predicates.
- `demo_agent.py` owns the strict action-selection and finding-selection schemas, bounded correction loop, turn accounting, and provider messages.
- `objective_receipts.py` renders the exact validated action and deterministic evaluator invocation.
- `interactive_workflow.py` renders the candidate-action table, decision ladder, deterministic explanations, measured summary, and catalog findings.
- `chemistry_workflow.py` remains the source of E01-E06 and validated retained scientific state.
- The notebook receives copy-only clarification; its launch call and six-stage flow remain unchanged.

No dashboard, service, database, route-planning feature, property model, arbitrary code execution surface, or additional presenter interaction is added.

## Verification

### Deterministic tests

Tests must prove:

- reachability certification invokes the production offered-set and comparator functions, branches across every accepted offered tie, and fails closed if any branch misses the target;
- every offered substitution is legal, improving, canonically derived from the current panel, and independent of display order;
- zero-, one-, two-, and three-action states render truthfully, and a three-action offered set contains the highest score keys with deterministic boundary inclusion;
- quantized comparison semantics are identical for improvement, ties, maximum selection, target attainment, reachability, ordering, and display at every boundary;
- all co-limiting pairs are canonical and an improving substitution affects every prior co-limiting pair;
- the strict hosted schema exposes only the current `state_id`, `swap_id`, co-limiting-pair, and decision-rule choices;
- stale responses from an earlier state revision are rejected even when their substitution IDs recur later;
- nonmaximal, stale, malformed, or mismatched choices are rejected without chemistry execution or attempt consumption;
- `invalid -> valid`, `invalid -> invalid -> stop`, rejections interleaved with accepted attempts, one transport retry, counter exhaustion, and a locally blocked third rejection preserve exact counters and message pairing;
- accepted choices execute exactly once and retain exact provenance;
- evaluator and prospective-evidence failures produce a paired error result, no measured attempt, and an **evaluation not completed** receipt;
- post-commit rendering failures reconstruct from retained state without rerunning the model or evaluator;
- target success stops immediately and bounded failure remains truthful;
- no objective model-authored text reaches the UI or O01;
- every rendered number is derived from current evidence;
- limiting distance and limiting similarity are labeled correctly as complements;
- candidate-pool cluster coverage is not described as final-panel coverage;
- conformer convergence and within-molecule energy qualifications are exact;
- finding IDs cannot cross themes, repeat, omit a theme, bypass predicates, or introduce prose;
- single-option themes are labeled deterministic and at least four default-data themes have multiple true alternatives;
- finding-selection failure receives a paired rejection when applicable, performs no semantic retry, and uses the deterministic fallback;
- deterministic measured summary remains available if finding selection fails;
- widget state retains the ledger and conclusion content after reopening or standalone export; and
- the existing six-stage workflow, controls, receipts, figures, retries, and evidence remain unchanged in purpose.

### Live Brev acceptance

On the exact L4 launch environment and fixed notebook dataset:

1. run the complete GPU-enabled test suite;
2. run 20 fresh objective-only hosted trials from the same retained deterministic scientific state at the production model settings, using temperature zero when the provider supports it;
3. require 20/20 trials to reach the target within at most three accepted substitutions, with correct argmax selection, no incorrect displayed claims, and no unpaired hosted messages; a transport-failure-then-success trial counts as completed but is labeled retry-assisted, and the clean first-request completion rate is reported separately;
4. run three fresh end-to-end notebook workflows and require 3/3 completion through the evidence-controlled conclusion; and
5. inspect a reopened or standalone-rendered widget to confirm that candidate actions, Nemotron choices, executed code, measurements, trajectory, final structures, heatmap, measured summary, and selected findings remain visible.

The acceptance report must separate deterministic test success, hosted decision reliability, first-request transport reliability, retry-assisted completion, GPU execution, and rendered UI persistence. If the 20/20 or 3/3 gate fails, the feature is not conference-ready; no cached successful trajectory may be substituted for a live agent run.

## Acceptance Criteria

The refinement is complete when an observer can identify, without reading source code:

1. the quantitative objective and unchanged target;
2. every measured co-limiting pair before each action;
3. the up to three legal actions available to Nemotron;
4. the exact action Nemotron selected;
5. the deterministic code and measurement produced by that selection;
6. why the controller revised or stopped; and
7. which final statements are measured facts, required deterministic findings, or agent-selected evidence-backed emphasis.

The result must be both meaningfully agentic and reproducible: the model performs bounded argmax action selection, maintains the multi-turn policy state, and chooses among predicate-true evidence emphases; validated code exclusively owns scientific calculation, action evaluation, factual text, and stopping conditions.
