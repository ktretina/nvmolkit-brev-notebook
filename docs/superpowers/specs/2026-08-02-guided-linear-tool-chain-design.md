# Guided Linear nvMolKit Tool Chain Design

## Goal

Redesign the notebook as a presentation-ready guided chemistry agent. Nemotron handles one bounded analysis at a time, the notebook visibly validates and executes the requested scientific function, the corresponding figure appears immediately, and Nemotron gives a brief interpretation before the narrative advances.

The implementation must remain a lightweight standalone Brev notebook. It must not add LangChain, LangGraph, MCP, a NeMo Agent Toolkit runtime, multiple agents, model-generated Python, a new dataset, or new deployment infrastructure.

## Why the Current Narrative Changes

The current notebook defines one long `analyze_molecule_library` function spanning fingerprints, similarity, clustering, conformers, optimization, visualization, and summary generation. The rendered result therefore separates Section 5 from its implementation, delays the heatmap until after Section 6 begins, and leaves the MMFF94 convergence statement without a reliable illustration. The selected `1 representative x 1 conformer` parameters also produce no useful energy distribution, while the only 3D illustration depends on py3Dmol JavaScript and failed in the reviewed presentation.

The redesign replaces that composite function with a guided linear chain of six allow-listed tools. Each tool has one purpose, one validated input schema, one local executor, one immediate result display, and one short interpretation.

## Notebook Outline

The headings remain numbered so the requested final heading stays `## 6. What the results mean`:

1. `## 1. Preflight`
2. `## 2. Nemotron learns the nvMolKit skill`
3. `## 3. Molecular sample`
4. `## 4. Mapping molecular similarity`
   - `### 4.1 Morgan fingerprints`
   - `### 4.2 All-pairs Tanimoto similarity`
   - `### 4.3 Fused Butina clusters`
5. `## 5. Conformers and MMFF94`
6. `## 6. What the results mean`

Sections 2 through 5 use the same visible cadence:

1. **Task** - a concise Markdown statement of the scientific question.
2. **Scientific function** - one code cell defining a small bounded function.
3. **Nemotron tool call** - one code cell requesting, validating, displaying, and executing the named call.
4. **Result** - the relevant static visual and compact numeric summary directly below the call.
5. **Nemotron interpretation** - two to four sentences grounded in the returned summary and a text description of the figure.

The model is text-only in this workflow. It does not see rendered notebook figures. Every explanation receives a compact `figure_context` object describing the plot's axes, scale, and salient numerical features; the notebook must not claim image understanding.

## Introduction Copy

The notebook begins with this text:

> # nvMolKit + Nemotron
>
> This notebook demonstrates a guided chemistry agent using NVIDIA Nemotron to call a small set of allow-listed scientific tools backed by nvMolKit on an NVIDIA GPU. Nemotron first reads the BioNeMo Agent Toolkit skill for nvMolKit, learning the library's supported operations, API boundaries, and GPU requirements. It then works through a molecular library one analysis at a time: validating the sample, generating Morgan fingerprints, measuring all-pairs Tanimoto similarity, identifying structural clusters, and generating and minimizing representative conformers.
>
> Each stage follows the same transparent pattern. The notebook defines a bounded scientific function; Nemotron requests that function through a structured tool call; the notebook validates and executes it; the result is visualized immediately; and Nemotron provides a short interpretation. A final synthesis combines the numerical results from every stage into a detailed scientific discussion.
>
> Brev supplies the GPU environment, nvMolKit performs the batched GPU chemistry operations, RDKit handles molecule parsing and display preparation, and the notebook enforces the execution and scientific-safety boundaries. Nemotron chooses validated tool parameters and explains results, but it does not execute arbitrary Python.
>
> This is a cheminformatics demonstration, not a benchmark or validated scientific study. Fingerprints, similarities, clusters, force-field energies, and candidate geometries are computational outputs. They do not establish binding, biological activity, ADMET properties, efficacy, safety, synthesizability, or clinical relevance.

## Agent and Executor Architecture

`demo_agent.py` generalizes the current one-tool request into a small reusable guided-step helper. A step provides:

- one tool name;
- one JSON schema derived from a strict Pydantic argument model;
- one user task prompt;
- one allow-listed local executor;
- one compact tool-result summary;
- one interpretation prompt.

Every request forces the single tool for that section. The response may cross the executor boundary only when all existing protections pass:

- exactly one tool call;
- `tool_call.type == "function"`;
- the expected allow-listed name;
- a non-empty call ID;
- valid JSON arguments;
- no missing or extra fields;
- values inside the step's Pydantic constraints;
- `source == "nemotron"`;
- Nemotron thinking disabled so the hosted endpoint returns parsed tool calls.

No `eval`, dynamic import, model-supplied path, arbitrary callable name, generated code, or automatic retry is permitted. A malformed, missing, wrong-name, wrong-type, multiple, API-error, or invalid-argument response stops that scientific step before execution.

An interpretation failure is non-fatal after a successful computation: the notebook displays `Interpretation unavailable`, retains the visual and numeric result, and can continue to the next deterministic stage.

## Tool 1: `read_nvmolkit_skill`

The repository vendors the exact public `library-skills/nvMolKit/SKILL.md` snapshot used by the notebook. Its provenance file records the public source URL, upstream commit selected during implementation, retrieval date, license, byte count, and SHA-256 digest. Vendoring makes the Launchable self-contained and prevents a live GitHub dependency.

The function takes no arguments. Its tool result returns the skill text to Nemotron once. The notebook displays a deterministic capability table with the operations used in the demonstration:

| Capability | Entry point | Role in notebook |
| --- | --- | --- |
| Morgan fingerprints | `MorganFingerprintGenerator` | Molecular representation |
| Tanimoto similarity | `crossTanimotoSimilarity` | Pairwise structural similarity |
| Butina clustering | `fused_butina` | Structural grouping |
| ETKDG embedding | `EmbedMolecules` | Candidate 3D conformers |
| MMFF94 optimization | `MMFFOptimizeMoleculesConfs` | Force-field minimization |

The result also states the skill's GPU-only, batched-operation boundary and RDKit's continuing role. Nemotron responds with two to four sentences identifying the capabilities it will use and their limitations. Subsequent step prompts include a bounded skill-grounding context rather than retransmitting unrelated conversation.

## Tool 2: `prepare_molecular_sample`

The function accepts only `preview_count: Literal[24]`. It always reads the fixed bundled `DATA_PATH`; the model cannot choose a path or upload data.

It:

- reads the 256-row bundled CSV;
- parses SMILES with RDKit;
- excludes invalid molecules while preserving a visible invalid-count report;
- raises only if the bundled file shape is unexpected or no valid molecules remain;
- returns the valid molecule objects and filtered metadata for local use;
- displays the first 24 valid molecules in the existing 6-column grid.

The model receives only JSON-safe metadata: raw rows, valid molecules, invalid molecules, excluded identifiers, preview count, and a figure description. It does not receive RDKit objects. Nemotron briefly explains what was validated and what a 24-molecule preview cannot establish.

## Tool 3: `compute_morgan_fingerprints`

The strict arguments are:

- `fingerprint_radius: Literal[2, 3]`;
- `fingerprint_size: Literal[1024, 2048]`.

The function uses nvMolKit `MorganFingerprintGenerator(...).GetFingerprints(mols)` and synchronizes before host-side summaries. Human comments explain why the result is GPU-resident, why synchronization is required before reading it on the host, and what an active hashed bit means at a high level.

The local result retains the nvMolKit fingerprint object for the next tools. The JSON-safe summary contains tensor shape, radius, size, molecule count, active-bit count distribution, and device. The static visual is a histogram of active fingerprint bits per molecule. Nemotron gives a two-to-four-sentence interpretation without treating fingerprint density as biological activity.

## Tool 4: `compute_tanimoto_similarity`

The function has no model-selected arguments. It consumes only the validated fingerprint artifact from the previous step and calls nvMolKit `crossTanimotoSimilarity`.

The returned matrix remains local. The model receives:

- matrix shape;
- median, first quartile, third quartile, and 90th percentile after excluding the diagonal;
- maximum off-diagonal similarity;
- identifiers for the most similar non-identical pair;
- `figure_context` describing the heatmap axes and 0-to-1 color scale.

The existing heatmap appears immediately in this subsection. It is not yet ordered by clusters because clustering has not executed. Nemotron briefly explains the similarity distribution and the meaning of the most-similar pair without inferring shared activity.

## Tool 5: `cluster_with_fused_butina`

The strict argument is `cluster_cutoff: float` constrained to `0.40 <= cutoff <= 0.60`. This presentation-specific range prevents the overly fragmented minimum-cutoff result seen in the reviewed run while retaining meaningful agent choice around the validated default of 0.50.

The function calls nvMolKit `fused_butina(fingerprints.torch(), cutoff=...)`, validates complete molecule assignment, and stores cluster membership locally. Human comments explain the cutoff's role and why singleton clusters are reported separately.

The model receives cluster count, singleton count and fraction, largest cluster sizes, cutoff, and a figure description. The static visual is a bar chart of the 15 largest clusters with singleton count called out in the caption. The similarity heatmap may also be redisplayed in cluster order only if this does not duplicate a full-sized figure; the required cluster visual is the size chart. Nemotron briefly discusses library fragmentation, structural diversity, and cutoff sensitivity.

## Tool 6: `generate_and_optimize_conformers`

The strict arguments are:

- `representative_count: int` constrained to 3 through 6;
- `conformers_per_representative: int` constrained to 3 through 8.

The prompt recommends four representatives and four conformers unless the prior cluster result justifies another valid choice. This prevents the uninformative one-by-one result while leaving a small, bounded decision to Nemotron.

The function preserves the existing scientific behavior:

- choose MMFF94-eligible representatives from distinct clusters;
- prefer lower heavy-atom count deterministically;
- add hydrogens;
- generate ETKDGv3 conformers with the fixed seed;
- report zero and partial embedding results;
- optimize with nvMolKit MMFF94 for at most 500 iterations;
- preserve device-coordinate handling;
- display only converged structures;
- compare energies only within a molecule.

Two static visuals are required:

1. a per-molecule dot plot of attempted conformer energies, distinguishing converged and unconverged attempts;
2. a static Matplotlib 3D rendering of the lowest-energy converged conformer for each representative, using the optimized coordinates and molecular bonds.

The existing py3Dmol view becomes optional and appears only after the static figures. JavaScript loading failure must be caught and displayed as a quiet notice; it cannot remove the required static illustration or interrupt the workflow.

The JSON-safe result includes requested, generated, attempted, converged, and unconverged counts; representative identifiers; per-conformer energies and convergence flags; selected conformer IDs; and figure descriptions. Nemotron explains convergence, sampling, and within-molecule energy ranking in two to four sentences.

## Section 6: Detailed Synthesis

The notebook builds one JSON-safe `analysis_summary` from the six tool summaries. It contains no fingerprint tensor, similarity matrix, RDKit molecule, raw coordinate tensor, API key, or notebook display object.

The final hosted call asks Nemotron for 450 to 650 words at a PhD scientific level while remaining readable in a presentation. The response is organized under these themes:

1. dataset validity and scope;
2. molecular representation;
3. pairwise similarity structure;
4. clustering and library diversity;
5. conformational sampling and MMFF94 convergence;
6. limitations and appropriate next analyses.

The prompt requires quantitative references to the supplied results and explicit discussion of the figures through their text descriptions. It forbids claims about binding, biological activity, ADMET, efficacy, safety, synthesizability, clinical relevance, or experimentally validated conformations. It must distinguish force-field minima within sampled conformers from global or experimental conformations.

The scientific-boundary notice appears once before the interpretation and once after it. The current raw Python dictionary display is replaced with a compact presentation table so Section 6 reads as synthesis rather than debug output.

## Code-Comment Standard

Notebook code cells add short `#` comments only where a scientifically literate reader may not understand the implementation boundary. Required comment topics are:

- fixed-path and invalid-SMILES handling;
- model-call validation before execution;
- GPU-resident results and synchronization;
- diagonal exclusion from pairwise statistics;
- Butina cutoff and singleton reporting;
- representative-selection rule;
- ETKDG partial/zero generation handling;
- device coordinates copied into RDKit conformers;
- within-molecule energy comparison;
- static visualization fallback before optional py3Dmol.

Comments must explain intent, not restate obvious syntax.

## Files and Scope

Expected changes are limited to:

- `demo_agent.py` for generic guided-step calls and interpretation requests;
- `notebooks/nvmolkit_nemotron_demo.ipynb` for the presentation narrative and scientific functions;
- `README.md` for the revised workflow description;
- `tests/test_demo_agent.py` and `tests/test_notebook.py` for agent and narrative contracts;
- `tests/test_gpu_acceptance.py` only if needed to cover the separated nvMolKit stages;
- a vendored public skill snapshot and one provenance file under a small `skills/nvmolkit/` directory.

No dependency, setup-script, sample-data, Launchable-field, or infrastructure change is expected.

## Testing and Acceptance

Implementation follows test-driven development and fresh-agent review. Automated checks must cover:

- six exact allow-listed tool names and strict schemas;
- wrong, missing, multiple, malformed, and invalid calls never execute;
- thinking remains disabled for Nemotron 3 Nano tool calls;
- authentication errors remain secret-safe;
- each section's function definition precedes its call;
- each call precedes its visual and brief interpretation;
- the Section 5 static figures exist independently of py3Dmol;
- intermediate prompts request two to four sentences;
- final synthesis receives every JSON-safe result and requests 450 to 650 words;
- all committed notebook outputs and execution counts are clear;
- the notebook remains valid nbformat 4 and every code cell parses;
- the vendored skill checksum and provenance match;
- no new dependencies or setup changes appear.

Local completion requires the full pytest suite and notebook structural checks. Live acceptance requires the existing Brev L4 GPU test plus a user-key run showing all six valid Nemotron tool calls, all required static figures, six brief interpretations, and the final detailed synthesis. Passing local and GPU tests alone does not prove hosted tool calling.

## Non-Goals

- autonomous tool selection or planning across arbitrary tools;
- multiple agents or a recursive agent loop;
- arbitrary file access or model-generated code;
- chemical-property prediction, docking, binding inference, ADMET inference, or wet-lab claims;
- a performance benchmark or nvMolKit-versus-RDKit comparison;
- production observability, persistence, retries, MCP serving, or release hardening.
