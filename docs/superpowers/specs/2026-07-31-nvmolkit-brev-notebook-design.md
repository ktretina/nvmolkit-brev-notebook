# Minimal nvMolKit Brev Notebook Design

## Goal

Create a small Brev Launchable that starts JupyterLab on one NVIDIA GPU and opens a single guided notebook. The notebook introduces nvMolKit and its BioNeMo Agent Toolkit skill, uses a hosted Nemotron model to produce a bounded chemistry-workflow plan, runs that plan with nvMolKit, and visualizes the computed outputs.

Success means a new user can launch the environment, run the notebook from top to bottom, see the agent choose valid task parameters, and obtain the expected GPU-computed visualizations. This is a demonstration, not a benchmark or scientific-validation product.

## Scope Boundary

Build only:

- one primary Jupyter notebook;
- one small, attributed sample SMILES file containing 256 molecules;
- one Brev setup script that installs pinned dependencies and starts JupyterLab;
- one dependency file;
- one concise README with local structure, Brev Console fields, and verification steps;
- small tests for agent-plan validation and notebook structure where practical.

Do not build a web application, chat interface, database, service mesh, Docker stack, autonomous code-execution system, benchmark suite, or production deployment framework. Do not modify or reuse the existing `nemoclaw-nvmolkit-agent-workbench` project.

## Architecture

Brev supplies a single modest NVIDIA GPU VM and runs the setup script. JupyterLab is the only user interface. nvMolKit and CUDA-enabled PyTorch execute locally on the Brev GPU. A hosted Nemotron model is called through NVIDIA's API using a masked `NVIDIA_API_KEY` Launchable parameter.

The notebook implements a planner-and-executor pattern:

1. Nemotron receives a compact description derived from the public nvMolKit skill plus a strict JSON schema containing the allowed workflow and parameter ranges.
2. Nemotron returns a plan; it does not write or execute arbitrary Python.
3. Notebook code validates the plan and dispatches only predefined nvMolKit functions.
4. Deterministic checks, not agent prose, establish whether each computation completed.
5. Nemotron receives a compact result summary and explains what was computed and what cannot be concluded.

If hosted inference is unavailable or returns an invalid plan, the notebook displays the failure and offers a clearly labeled fixed default plan. The user can still complete the nvMolKit portion without confusing fallback execution with successful agent planning.

## Notebook Story

The notebook progresses in one linear path:

1. **Orientation:** Explain the distinct roles of Brev, hosted Nemotron, the Agent Toolkit skill, nvMolKit, and RDKit.
2. **Preflight:** Display the GPU model and relevant package versions; verify CUDA-enabled PyTorch, nvMolKit import, one small GPU fingerprint operation, and API-key presence.
3. **Dataset:** Load the bundled sample, validate SMILES with RDKit, and show a small molecule grid.
4. **Agent plan:** Ask Nemotron to select conservative parameters for the fixed workflow and render its validated JSON plan.
5. **Fingerprints:** Compute batched Morgan fingerprints on the GPU.
6. **Similarity and clustering:** Compute pairwise Tanimoto similarity and Butina clusters, then show a similarity heatmap with cluster assignments.
7. **Conformers:** Select a few cluster representatives, generate ETKDG conformers, apply MMFF94 geometry optimization, and show selected conformers in an interactive 3D view.
8. **Interpretation:** Validate output counts, shapes, convergence indicators, and finite numeric values; ask Nemotron for a concise explanation bounded to those results.

The sample size should be large enough to demonstrate a real batched workload but small enough to finish interactively on a modest GPU. It is not used to claim speedup or comparative performance.

## Agent Contract

The agent may set only conservative parameters: fingerprint radius (`2` or `3`), fingerprint size (`1024` or `2048`), cluster-distance cutoff (`0.2`-`0.8`), representative count (`1`-`6`), and conformers per representative (`1`-`8`). The fixed defaults are radius `2`, size `1024`, cutoff `0.5`, four representatives, and four conformers per representative. Unknown keys, missing required fields, prose around the JSON payload, or out-of-range values cause validation failure and trigger the labeled default-plan option.

The prompt includes only the relevant public skill guidance. It emphasizes that nvMolKit is intended for GPU-accelerated batched operations and that RDKit remains appropriate for molecule parsing, display, and isolated CPU-side utilities.

## Error Handling

The notebook fails early and clearly when no compatible GPU is visible, CUDA-enabled PyTorch is unavailable, nvMolKit cannot import, or the initial GPU operation fails. It never silently substitutes CPU RDKit for an nvMolKit computation.

The API key is read from the environment, never printed, embedded, saved in notebook output, or committed. Hosted inference errors are isolated from GPU-computation errors. Invalid molecules are counted and excluded with a visible warning. Empty clusters, failed embeddings, and unconverged optimizations are reported rather than hidden.

## Visual Outputs

Use only three visuals:

- a small RDKit molecule grid to orient the user to the sample;
- a similarity heatmap annotated or ordered by Butina cluster;
- an interactive 3D view of a few optimized representative conformers.

No dashboard, custom frontend, persistent history, or downloadable evidence bundle is required.

## Scientific and Product Boundaries

Outputs are computational descriptors, similarity relationships, clusters, and generated/optimized conformers. They are not evidence of binding, biological activity, ADMET properties, efficacy, safety, synthesizability, or clinical relevance. Geometry optimization produces force-field-local computational structures, not experimentally validated conformations.

No performance claim will be made without a separate controlled benchmark. Exact component licenses and current hosted-model/API terms must be checked at their public source before commercial or production use.

## Launchable Configuration

The repository will include paste-ready Brev Console guidance rather than attempting unsupported Launchable authoring through the CLI. The expected configuration is:

- VM mode with one NVIDIA GPU meeting nvMolKit's published compatibility requirements;
- one masked, required `NVIDIA_API_KEY` parameter with no default;
- JupyterLab exposed through one Brev Secure Link port;
- the repository setup script as the Launchable setup command;
- no public unauthenticated service beyond the access policy selected in Brev.

The exact GPU type, hourly price, organization, access scope, and Launchable identifier remain deployment-time choices. No paid instance will be created without explicit cost and ownership approval.

## Verification

Local verification covers setup-script syntax, dependency metadata, notebook JSON validity, absence of embedded credentials, and agent-plan validation/fallback behavior. GPU acceptance requires one clean run on the exact Brev environment that proves:

- the expected GPU and CUDA runtime are active;
- nvMolKit completes the fingerprint, similarity, clustering, embedding, and optimization operations;
- the three visuals render;
- a valid Nemotron plan is accepted and an invalid response is rejected;
- the final explanation preserves the scientific boundaries;
- no credential appears in tracked files or saved notebook output.

Only after that run may the project be described as demo-ready. Launchable registration and publishing occur in the Brev Console; deployment acceptance follows once the user supplies the resulting Launchable or instance identifier.

## Public Sources

- nvMolKit repository and installation requirements: <https://github.com/NVIDIA-BioNeMo/nvMolKit>
- nvMolKit documentation: <https://nvidia-bionemo.github.io/nvMolKit/>
- BioNeMo Agent Toolkit nvMolKit skill: <https://github.com/NVIDIA-BioNeMo/bionemo-agent-toolkit/blob/main/library-skills/nvMolKit/SKILL.md>
