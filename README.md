# nvMolKit + Nemotron Notebook

This repository provides a bounded molecular-computing presentation: Brev hosts a GPU-backed JupyterLab session, hosted Nemotron emits six forced OpenAI-compatible tool calls, and nvMolKit performs the batched GPU chemistry operations. The notebook validates every requested tool and argument set before executing the corresponding allow-listed local function. Nemotron does not execute the notebook's Python code.

The public BioNeMo Agent Toolkit nvMolKit skill is pinned and vendored in the repository. The first guided call reads that skill at runtime, grounding later calls in its documented API and GPU boundaries. Hosted requests receive only the one skill text needed for grounding or compact JSON-safe summaries; local tensors, matrices, RDKit molecules, optimized coordinates, and credentials remain in the notebook process.

## Guided presentation

The six forced calls run in one visible order:

1. `read_nvmolkit_skill`
2. `prepare_molecular_sample`
3. `compute_morgan_fingerprints`
4. `compute_tanimoto_similarity`
5. `cluster_with_fused_butina`
6. `generate_and_optimize_conformers`

Each call is followed immediately by its numerical result and presentation visual: a molecule grid, fingerprint histogram, Tanimoto heatmap, cluster-size bar chart, conformer energy/convergence plot, and static 3D atom-and-bond conformer panels. The static Matplotlib figures are authoritative; py3Dmol is an optional enhancement after the static 3D result. Six brief interpretations keep each stage readable. A detailed final synthesis receives the six actual summaries and text-only `figure_context` descriptions, not figure pixels.

## What runs where

- **Brev:** Provisions the GPU VM and provides the organization-only Secure Link to JupyterLab.
- **Nemotron:** Runs as a hosted NVIDIA API model; it requests each bounded function and interprets returned summary evidence.
- **Notebook:** Owns the six-tool contract, validates each request, executes deterministic local functions, retains non-serializable artifacts, and renders results.
- **nvMolKit:** Runs on the Brev GPU for fingerprints, similarity, clustering, conformer generation, and MMFF94 geometry optimization.
- **RDKit:** Parses and hydrogenates molecules, screens MMFF94 eligibility, and supplies structures for reliable static rendering.

## Launch

Use Linux x86-64 with CPython 3.12 in VM mode. Enable Jupyter in the Brev Console so the Brev-managed Jupyter runtime is available; the setup script installs into that runtime and does not manage the Jupyter service.

1. Create the Launchable in the Brev web Console using [`launchable/fields.md`](launchable/fields.md).
2. Enable Brev-managed Jupyter and keep access set to **Only my organization** with a Secure Link on port `8888`; do not expose unrestricted public TCP.
3. Deploy, open JupyterLab through the Secure Link, and run `notebooks/nvmolkit_nemotron_demo.ipynb`.
4. From the Nemotron model page on build.nvidia.com, generate a hosted NVIDIA Developer API key (it starts with `nvapi-`). Supply `NVIDIA_API_KEY` to the notebook process or, when requested, paste only the bare key into the notebook's hidden prompt. This is distinct from an NGC personal key. Do not rely on setup-variable persistence for the key; never save it in the notebook or its outputs.

**Qualification:** This demo is designed for fresh deployment only; it is not yet live-qualified. GPU execution, hosted inference, rendered visuals, and Secure Link access require live acceptance. After a VM stop/start, verify the managed Jupyter service, rerun the notebook, and enter the key again if prompted; do not claim auto-restart without live evidence.

## Verify

These are separate evidence gates; passing one does not prove the others:

- **Local deterministic acceptance:** run `pytest` to validate notebook structure, safety contracts, serialization boundaries, and helper wiring without claiming GPU or hosted execution.
- **GPU acceptance:** on a compatible NVIDIA GPU, run `RUN_GPU_TESTS=1 pytest` and retain the result before calling the nvMolKit runtime GPU-accepted.
- **Hosted inference acceptance:** in the fresh Brev notebook, verify all six forced calls, six brief interpretations, and the detailed synthesis with a valid hosted Developer API key.
- **Rendered deployment acceptance:** inspect the immediate grid, histogram, heatmap, cluster, energy, and static 3D visuals through the organization-only Secure Link.

## Boundaries

This is a research and developer demonstration. It makes no performance claims and no claims about binding, biological activity, ADMET, efficacy, safety, synthesizability, clinical relevance, or experimentally validated conformations. ETKDGv3 samples candidate geometries, and MMFF94 ranks sampled force-field minima only within a molecule; those minima are not global or experimental conformations. The outputs require independent computational and experimental validation appropriate to the intended use.

## Official sources

- [NVIDIA Brev Launchables documentation](https://docs.nvidia.com/brev/concepts/launchables)
- [NVIDIA nvMolKit repository](https://github.com/NVIDIA-BioNeMo/nvMolKit)
- [NVIDIA nvMolKit documentation](https://nvidia-bionemo.github.io/nvMolKit/)
- [NVIDIA Nemotron 3 Nano 30B-A3B model card](https://build.nvidia.com/nvidia/nemotron-3-nano-30b-a3b/modelcard)
- [RDKit documentation](https://www.rdkit.org/docs/)
