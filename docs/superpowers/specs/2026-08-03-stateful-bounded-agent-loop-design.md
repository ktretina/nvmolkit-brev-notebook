# Stateful Bounded nvMolKit Agent Loop Design

## Goal

Reframe the notebook around one question:

> Can an AI chemistry agent use nvMolKit to autonomously execute a real cheminformatics workflow?

One high-level user request starts one persistent Nemotron conversation. Nemotron explains the required plan, selects bounded scientific parameters as results become available, issues the dependency-valid tool calls, receives structured tool results, and produces one evidence-linked conclusion. Python fixes the scientific workflow and dependencies. The notebook validates and executes the calls and presents the transcript and results; it does not prescribe six independent demonstrations in separate notebook sections.

This is bounded autonomy, not arbitrary autonomy. Scientific dependencies constrain the workflow, Python validates every transition, and Nemotron never supplies code, file paths, imports, or callable objects.

## Success Criterion

One high-level request produces one continuous, validated transcript in which:

- Nemotron presents a concise execution plan;
- every scientific computation is labeled with its actual nvMolKit entry point;
- Nemotron selects parameters within validated limits using the evidence available at that point;
- each successful tool result is returned to the same Nemotron conversation;
- invalid SMILES and partial scientific failures remain visible;
- the final narrative cites an immutable report built from computed artifacts; and
- exact quantitative results are rendered by Python rather than copied from free-form model text.

## Scope and Non-Goals

The project remains one lightweight, standalone Brev notebook. It does not add LangChain, LangGraph, MCP, a NeMo Agent Toolkit runtime, multiple runtime agents, arbitrary code execution, a database, a web application, a new dataset, or new deployment infrastructure.

It is not a performance benchmark, an nvMolKit-versus-RDKit comparison, or evidence of binding, biological activity, ADMET properties, efficacy, safety, synthesizability, clinical relevance, experimental conformations, or global-minimum conformations.

The phrase "Nemotron learns the skill" is replaced with "Nemotron is grounded with the pinned skill for this conversation." The notebook displays concise plan rationales and decisions, never hidden chain-of-thought.

## Presentation Design

The source notebook is reduced from 48 cells, 31 code cells, and about 1,280 visible code lines to no more than eight presentation cells and 150 visible code lines. The target is at least a 70 percent visible-code reduction.

The notebook contains:

1. **Question and boundary** - the single demonstration question, component responsibilities, and research-only boundary.
2. **Preflight** - one compact cell that locates the fixed repository assets, reads a hidden hosted API key, verifies Python, CUDA, the GPU, and pinned nvMolKit, and imports the workflow facade.
3. **User request** - one editable high-level request with the fixed bundled molecular-library path owned by Python, not the model.
4. **Agent run** - one call to the workflow facade. Its display callback renders three lightweight output headings in sequence: Nemotron plan, continuous execution, and checked conclusion.
5. **Boundary** - the research and scientific-interpretation limits.

The event stream uses this presentation vocabulary:

```text
Nemotron plan -> Inspect -> Represent -> Compare -> Cluster -> Embed -> Optimize
Nemotron -> inspect_library()                         [RDKit input validation]
Nemotron -> MorganFingerprintGenerator(...).GetFingerprints(molecules) [nvMolKit GPU]
Nemotron -> crossTanimotoSimilarity(...)             [nvMolKit GPU]
Nemotron -> fused_butina(...)                        [nvMolKit GPU]
Nemotron -> EmbedMolecules(...)                      [nvMolKit GPU]
Nemotron -> MMFFOptimizeMoleculesConfs(...)          [nvMolKit GPU]
```

The notebook does not contain per-stage interpretation calls. A successful tool call displays its compact result immediately, and the conversation advances. The plan, execution, and conclusion are three displayed phases of the same `run_workflow(...)` cell, not three separate workflow invocations.

## Component Boundaries

### `notebooks/nvmolkit_nemotron_demo.ipynb`

The notebook is a presentation surface only. It owns the user-visible question, one high-level request, the preflight display, and calls to the public workflow facade. It contains no tool schemas, scientific executors, state-machine implementation, validation boilerplate, or conformer-coordinate plumbing.

### `demo_agent.py`

This module owns hosted Nemotron communication:

- hosted endpoint and model constants;
- secret-safe API-key validation;
- the persistent OpenAI-compatible message history;
- strict plan, scientific-tool, and conclusion schemas;
- tool-call parsing and validation;
- eligible-tool selection for each workflow state;
- the agent-loop controller; and
- final evidence-reference validation.

It also exposes the single `run_workflow(...)` facade used by the notebook. It imports the deterministic domain operations from `chemistry_workflow.py`; the domain module does not import the hosted-agent module.

The old notebook-facing `request_tool_call`, `request_and_execute_step`, `request_brief_interpretation`, and independent `request_final_synthesis` interface is removed rather than retained alongside the loop.

The module retains zero SDK retries. Authentication, permission, malformed-response, empty-response, and hosted-request errors remain secret-safe.

### `chemistry_workflow.py`

This new module owns deterministic scientific work:

- `WorkflowState`, containing local scientific artifacts and completed-stage state;
- fixed-path sample loading and RDKit SMILES parsing;
- nvMolKit tool executors;
- dependency and transition validation;
- representative selection;
- compact result summaries and figures;
- `WorkflowReport` construction and scientific invariant checks.

The primary notebook interface is:

```python
result = run_workflow(
    user_goal=USER_GOAL,
    api_key=api_key,
    display_events=True,
)
```

Non-serializable RDKit molecules, nvMolKit results, CUDA tensors, similarity matrices, and coordinates remain local in `WorkflowState`. Only compact JSON-safe summaries cross the hosted-model boundary.

## Skill Grounding and Plan

The repository's pinned `skills/nvmolkit/SKILL.md` snapshot is read from the fixed local path and placed in the initial Nemotron context before planning. Its provenance and digest remain independently tested. The model cannot select a skill path or fetch replacement instructions.

The skill is not exposed as a tool call. The notebook displays its fixed provenance and the five nvMolKit entry points used by the workflow.

Nemotron first submits a structured explanation of the fixed dependency plan containing:

- an ordered list of the allowed workflow stages;
- one concise, presentation-safe rationale per stage; and
- no executable text or unvalidated scientific parameters.

The plan must cover input inspection, molecular representation, similarity, clustering, conformer embedding, and force-field optimization in dependency order. Scientific parameter choices are made later, when the relevant evidence exists. A missing, duplicated, unknown, or dependency-invalid stage stops before scientific execution. The notebook does not claim that Nemotron selected which scientific operations make up this fixed workflow.

## Persistent Agent Loop

The validated plan and all later calls share one message history. For each stage:

1. Python exposes only dependency-valid tool schemas.
2. Nemotron selects one exposed tool and supplies its bounded arguments.
3. Python validates exactly one non-empty function call, its name, ID, JSON shape, strict argument model, and workflow transition.
4. The deterministic executor runs exactly once.
5. Python validates and stores the local result.
6. A compact structured summary returns as a `role="tool"` message using the original call ID.
7. The notebook displays the call, selected parameters, result summary, and relevant figure.
8. The loop continues with the updated evidence.

No `eval`, dynamic import, model-selected path, arbitrary tool name, generated Python, callable serialization, or deterministic fallback plan is allowed. A rejected model call cannot cross the executor boundary.

The controller permits at most eight hosted response turns total: one plan, six execution stages including RDKit inspection, and one conclusion. Early textual completion, repeated stages, or turn-limit exhaustion fails closed.

## Tool Contract

### Input inspection boundary

`inspect_library` reads only the fixed bundled CSV. RDKit parses SMILES, preserves the expected 256-row accounting, excludes invalid entries visibly, and prepares molecules for display and nvMolKit. This operation is labeled RDKit input validation, not an nvMolKit capability.

The model chooses no path and no preview size. The notebook displays at most 24 valid molecules to remain compact.

### nvMolKit scientific tools

The hosted tool names and displays correspond directly to these entry points:

| Stage | Entry point | Model-selected arguments |
| --- | --- | --- |
| Represent | `MorganFingerprintGenerator(...).GetFingerprints(molecules)` | `radius` in `{2, 3}`; `fpSize` in `{1024, 2048}` |
| Compare | `crossTanimotoSimilarity(fingerprints)` | none |
| Cluster | `fused_butina(fingerprints, cutoff=...)` | `cutoff` from `0.40` through `0.60` |
| Embed | `EmbedMolecules(...)` | representative count from `3` through `6`; representative policy; conformers per representative from `3` through `8` |
| Optimize | `MMFFOptimizeMoleculesConfs(...)` | no free-form choice; fixed maximum of 500 iterations |

The clustering cutoff is chosen only after the similarity summary is available. Representative count, representative policy, and conformer count are chosen only after cluster results are available.

Representative policy is a strict enum with two deterministic behaviors. Before either policy runs, RDKit filters each cluster to MMFF94-eligible molecules. Clusters are ordered by original cluster size descending, then by the smallest bundled-data row index ascending. Members within a cluster are ordered by bundled-data row index ascending.

- `largest_clusters_first`: select the lowest-row-index eligible member from each ordered cluster;
- `include_singleton_if_available`: reserve one slot for the lowest-row-index eligible singleton, when one exists, exclude that singleton's cluster from the fill pass, and fill the remaining slots using `largest_clusters_first`.

If fewer eligible distinct clusters exist than requested, the executor selects every available eligible cluster and reports the shortfall. Fewer than three eligible distinct clusters stops before embedding. The implementation reports exact molecule, source-row, and cluster provenance. It does not call a selected molecule a medoid, centroid, cluster center, or algorithmic representative.

Representative selection remains part of the single hosted embed decision, but the display separates its execution boundary: `Python/RDKit -> select and eligibility-check representatives`, followed by `Nemotron -> EmbedMolecules(...) [nvMolKit GPU]`. `EmbedMolecules` and `MMFFOptimizeMoleculesConfs` remain separate visible calls so the audience sees both nvMolKit capabilities. ETKDG uses the fixed reproducible seed and supported random-coordinate initialization. MMFF94 eligibility, zero or partial embedding, flat coordinate indexing, convergence, and finite-energy checks remain fail-closed or visibly accounted for as appropriate.

## Working State and Adaptation

`WorkflowState` is the authoritative dependency record. It exposes compact evidence to Nemotron while retaining heavy artifacts locally.

The agent's consequential adaptive choices are:

- fingerprint radius and size before representation;
- clustering cutoff after observing the similarity distribution;
- representative count and policy after observing cluster sizes and singleton counts; and
- conformer count after observing cluster sizes, singleton counts, and representative eligibility information in the clustering summary.

The executor, not Nemotron, controls the fixed data path, CUDA device, allowed entry points, synchronization, scientific invariants, coordinate mapping, and rendering.

## Results and Figures

The workflow retains only figures that advance the continuous story:

- a compact 24-molecule preview after input inspection;
- a compact fingerprint summary followed by the all-pairs Tanimoto heatmap;
- a cluster-size plot after `fused_butina`;
- an MMFF94 energy and convergence plot; and
- static optimized representative structures.

The static figures are authoritative. An optional py3Dmol view may follow them, but a JavaScript or widget failure is non-fatal and cannot remove the static result.

There is no LLM-generated interpretation between figures. The final synthesis integrates the complete workflow.

## Canonical Report and Checked Conclusion

After scientific execution, Python builds an immutable `WorkflowReport` directly from computed artifacts, never from earlier model prose. Before synthesis it validates:

- raw, valid, and invalid input counts;
- fingerprint dimensions and selected parameters;
- similarity shape, symmetry, finite values, and the zero-to-one range;
- complete and unique cluster assignment;
- representative molecule and cluster provenance;
- requested, generated, attempted, converged, and unconverged conformer counts;
- conformer-to-molecule indexing;
- finite MMFF94 energies; and
- selected lowest-energy converged conformers within each molecule.

The report produces a stable evidence ledger. Each evidence item has a unique key, an exact Python-rendered value, units where relevant, and provenance naming the producing operation.

The top-level evidence groups are fixed for presentation and validation: `E01` input validity, `E02` fingerprints, `E03` similarity, `E04` clustering, `E05` embedding, and `E06` MMFF94 optimization.

Nemotron submits one strict structured conclusion. Each required theme contains qualitative prose and one or more evidence keys. Model-authored prose is prohibited from containing digit characters; fixed labels such as `3D`, `ETKDGv3`, and `MMFF94` are inserted by Python headings and evidence annotations rather than copied from the model. Computed quantities exist only in the referenced evidence records, and Python renders their canonical formatted values beside the prose.

Unknown keys, missing required themes, empty prose, or digit-bearing model prose invalidates the synthesis. This deliberately narrow grammar makes quantitative checking deterministic instead of attempting to infer which numbers in unrestricted prose are scientific claims.

If synthesis validation fails, the notebook displays the canonical report and a concise secret-safe validation error. It does not display the rejected narrative and does not invent a fallback interpretation. This validation establishes traceability of reported quantities; it does not guarantee that every qualitative scientific interpretation is correct.

The conclusion must address dataset scope, representation, similarity structure, clustering, conformational sampling, MMFF94 convergence, limitations, and appropriate next analyses. It must preserve the project's scientific claim boundaries.

## Error Behavior

- Preflight failures stop before hosted or scientific calls.
- Plan or tool-call validation failures stop before the relevant executor.
- A scientific invariant failure ends the loop with a visible failed event and partial canonical state; later dependent tools do not run.
- Invalid SMILES, ineligible representatives, and partial conformer generation are reported explicitly when the remaining result is still scientifically valid.
- A synthesis-validation failure preserves and displays the deterministic report but withholds model prose.
- An optional interactive-view failure is a quiet, non-fatal notice after the static figures.
- No failure path displays the API key, raw hosted exception, local tensors, coordinates, or arbitrary file contents.

## Files and Change Boundary

Expected implementation changes are limited to:

- create `chemistry_workflow.py`;
- modify `demo_agent.py`;
- rewrite `notebooks/nvmolkit_nemotron_demo.ipynb`;
- modify `tests/test_demo_agent.py`;
- create `tests/test_chemistry_workflow.py`;
- replace repetitive structural assertions in `tests/test_notebook.py` with thin-presentation assertions;
- modify `tests/test_gpu_acceptance.py` only where needed to exercise the separated runtime facade; and
- update `README.md`.

The bundled dataset, its provenance, pinned skill snapshot and provenance, requirements, setup script, Launchable fields, public repository boundary, and surrounding workbench remain unchanged unless implementation discovers a demonstrated incompatibility. Such an incompatibility requires a separate design decision rather than an incidental edit.

## Testing

Implementation follows test-driven development and fresh-agent review. Automated tests must prove:

- plan stages are complete, unique, allow-listed, and dependency-valid;
- only currently valid tools are exposed;
- every malformed, multiple, missing, wrong-name, wrong-type, extra-field, out-of-range, or empty call stops before execution;
- each valid call executes exactly once and returns a correctly linked `role="tool"` message;
- the same conversation history accumulates the plan and every result;
- each adaptive parameter is selected only after its required evidence exists;
- RDKit and nvMolKit responsibilities are labeled correctly;
- `WorkflowState` retains heavy artifacts locally and hosted payloads remain JSON-safe;
- the canonical report rejects invalid similarity, clustering, representative, coordinate, energy, and convergence states;
- conclusion evidence keys and exact quantities resolve to the report;
- a rejected synthesis is withheld while the deterministic report remains visible;
- the notebook has no more than eight cells and 150 visible code lines;
- the notebook contains one high-level request, one plan, one workflow invocation, one continuous transcript, and one conclusion;
- committed outputs and execution counts are clear;
- no API key or secret-bearing exception can enter source or output;
- the skill snapshot digest and provenance remain unchanged; and
- setup, Launchable, dependency, data, and surrounding-workbench files remain untouched.

## Acceptance Gates

1. **Local deterministic gate:** the complete test suite passes without claiming hosted or GPU execution.
2. **Brev GPU gate:** the existing opt-in L4 acceptance test passes under CPython 3.12 and exercises all five nvMolKit entry points.
3. **Hosted agent gate:** one valid Developer API key run shows one plan and one persistent sequence of validated calls with model-selected bounded parameters.
4. **Rendered notebook gate:** the organization-only Jupyter view shows the compact transcript, invalid-SMILES accounting, retained static figures, canonical metrics, and checked conclusion in natural presentation order.
5. **Independent review gate:** a fresh reviewer confirms that the narrative does not overstate autonomy, skill learning, nvMolKit attribution, GPU performance, or scientific meaning.

Local tests and the GPU gate do not by themselves prove hosted tool calling, adaptation, rendered figures, or conclusion quality. The existing Brev instance remains a deployment target, not part of the source design.
