# nvMolKit + Nemotron Notebook

This standalone three-notebook workshop demonstrates how agentic AI can support a bounded chemistry workflow. It starts with direct [nvMolKit](https://github.com/NVIDIA-BioNeMo/nvMolKit) calls, then uses hosted Nemotron for constrained policy selection, planning, and audit. Python owns validation and executable chemistry code. Nemotron receives an aggregate input profile for planning and an independently validated aggregate report snapshot for audit, but no raw molecule rows, similarity matrices, rendered code, credentials, or local visualization artifacts.

## Three-notebook workshop path

Start with **Module 1**. It is the recommended entry point and introduces the direct library calls before the agent exercises.

1. [`notebooks/01_direct_nvmolkit_reframe.ipynb`](notebooks/01_direct_nvmolkit_reframe.ipynb) — use nvMolKit directly for fingerprints, similarity, and clustering.
2. [`notebooks/02_agent_assisted_reframe_neighborhoods.ipynb`](notebooks/02_agent_assisted_reframe_neighborhoods.ipynb) — let Nemotron choose between bounded neighborhood-analysis policies while Python owns the executable implementation.
3. [`notebooks/03_full_agent_reframe_panel_design.ipynb`](notebooks/03_full_agent_reframe_panel_design.ipynb) — use a bounded planning, approval, execution, and audit loop to design a 24-compound panel.

**Hosted mode** (`interactive` in `NVMOLKIT_WORKSHOP_MODE`) is the attendee path for Modules 2 and 3. It uses `nvidia/nvidia/nemotron-3-nano-30b-a3b` at `https://inference-api.nvidia.com/v1` and the protected `NVIDIA_INFERENCE_API_KEY`. **Reference mode** is the deterministic recovery and local-acceptance path for Modules 2 and 3. It makes no hosted client call, needs no key, and is not evidence of hosted inference. Module 1 uses the library directly and does not call the hosted model.

## What runs where

- **Brev** provides the GPU VM and organization-only Secure Link to JupyterLab.
- **Hosted Nemotron** selects two bounded policies in Module 2. In Module 3, it returns one strict plan and one strict audit. It does not execute Python.
- **Python** owns the allowed choices, tool contract, validation, execution, artifact gate, and presentation.
- **RDKit** parses inputs, computes descriptors for bounded panel constraints, and supports rendering.
- **nvMolKit on the GPU** generates Morgan fingerprints, computes Tanimoto similarity, and runs fused Butina clustering.

## Interactive flow

Module 2 asks Nemotron for two bounded policy values. Python validates them, renders the allow-listed function locally, and runs the neighborhood analysis and checks. Module 3 asks Nemotron for a strict plan, waits for attendee approval, runs one of two allow-listed strategies, validates the artifacts, and requests a strict audit. The panel must match or improve the first-24-row baseline on descriptor coverage and minimum Tanimoto distance, with at least one strict improvement.

## Launch

Use Linux x86-64 with CPython 3.12 in VM mode. Enable Jupyter in the Brev Console so the Brev-managed Jupyter runtime is available; the setup script installs into that runtime and does not manage the Jupyter service.

1. Create or edit the Launchable in the Brev web Console using [`launchable/fields.md`](launchable/fields.md). Set the default disk storage to **75 GiB**, then paste the current contents of `launchable/setup.sh` into the Software configuration setup-script field. Updating the repository does not replace the script body already saved in a Launchable.
2. Keep only one Launch parameter: required Text parameter `NVIDIA_INFERENCE_API_KEY`, with no default. Remove `NVIDIA_API_KEY`, `NEMOTRON_MODEL`, and `JUPYTER_PORT` from Setup values.
3. Enable Jupyter and keep access set to **Only my organization** with a Secure Link on the fixed port `8888`; do not expose unrestricted public TCP. The hosted model is fixed to `nvidia/nvidia/nemotron-3-nano-30b-a3b` at `https://inference-api.nvidia.com/v1`.
4. The workshop organizer supplies an approved Inference Hub key beginning with `sk-`. Attendees do not create a personal NVIDIA API key. Enter the supplied value once in Setup values when you deploy, unless a separate runtime secret manager is available. The setup script stores it outside the repository in `${HOME}/.config/nvmolkit/NVIDIA_INFERENCE_API_KEY` with file mode `0600`, and notebook preflight loads it automatically without a prompt. Never expose the key in repository files, notebook outputs, logs, screenshots, or chat.
5. Open JupyterLab through the Secure Link and start with `notebooks/01_direct_nvmolkit_reframe.ipynb`.

**Qualification:** This demo is designed for fresh deployment only and is not yet live-qualified. GPU execution, hosted inference, rendered visuals, Secure Link access, and credential persistence each require live acceptance. The protected key file remains on the VM disk until the environment is deleted or the file is removed. After a VM stop/start, verify the managed Jupyter service, restart the notebook kernel, and rerun the notebook; do not claim automatic restart without live evidence.

## Verify

These are separate evidence gates; one does not prove the others:

CPU deterministic suite:

```bash
pytest -q
```

GPU suite on the task-owned compatible NVIDIA GPU:

```bash
RUN_GPU_TESTS=1 .venv/bin/python -m pytest -q
```

- **Local deterministic acceptance:** run `pytest` to validate notebook structure, scientific state transitions, serialization boundaries, and agent wiring without claiming GPU or hosted execution.
- **GPU acceptance receipt:** on a compatible NVIDIA GPU, run `RUN_GPU_TESTS=1 .venv/bin/python -m pytest -q` and retain the result before calling the nvMolKit runtime GPU-accepted.
- **Persistence receipt:** after a fresh stop/start, record Jupyter, kernel, notebook, and credential-reentry checks separately. Cross-reference it to the hosted and GPU receipts; do not merge their claims.
- **Hosted inference acceptance:** in a fresh Brev kernel, verify the Module 2 bounded policy response and the Module 3 strict plan and strict audit using the organizer-supplied Inference Hub key.
- **Rendered deployment acceptance:** inspect the Module 1 fingerprint, similarity, and cluster views; the Module 2 policy receipt and neighborhood comparison; and the Module 3 approval widget, authoritative receipt, and chemistry gallery through the organization-only Secure Link.

## Boundaries

This is a research and developer demonstration with bounded workflow autonomy. It makes no performance claims. Fingerprints, Tanimoto similarity, clusters, descriptors, and 2D drawings do not establish binding, biological activity, ADMET, efficacy, safety, synthesizability, clinical relevance, or experimental structure. Independent computational and experimental validation is required for any intended scientific use.

## Official sources

- [NVIDIA Brev Launchables documentation](https://docs.nvidia.com/brev/concepts/launchables)
- [NVIDIA nvMolKit repository](https://github.com/NVIDIA-BioNeMo/nvMolKit)
- [NVIDIA nvMolKit documentation](https://nvidia-bionemo.github.io/nvMolKit/)
- [BioNeMo Agent Toolkit nvMolKit skill](https://github.com/NVIDIA-BioNeMo/bionemo-agent-toolkit/blob/ce151c15470991c8cb9a0efdd531a124c346ca5b/library-skills/nvMolKit/SKILL.md)
- [NVIDIA Nemotron 3 Nano 30B-A3B model card](https://build.nvidia.com/nvidia/nemotron-3-nano-30b-a3b/modelcard)
- [RDKit documentation](https://www.rdkit.org/docs/)
