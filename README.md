# nvMolKit + Nemotron Notebook

This repository provides a bounded molecular-computing demo: Brev hosts a GPU-backed JupyterLab session, hosted Nemotron proposes validated workflow parameters and explains computed summaries, and nvMolKit performs the GPU chemistry operations.

## What runs where

- **Brev:** Provisions the GPU VM and provides the organization-only Secure Link to JupyterLab.
- **Nemotron:** Runs as a hosted NVIDIA API model; the notebook sends it bounded planning and explanation requests.
- **nvMolKit:** Runs on the Brev GPU for fingerprints, similarity, clustering, conformer generation, and geometry optimization.
- **RDKit:** Runs in the notebook environment for molecule parsing, preparation, and display.

## Launch

Use Linux x86-64 with CPython 3.12 in VM mode; the setup script rejects other runtime envelopes before creating the environment or installing packages.

1. Create the Launchable in the Brev web Console using [`launchable/fields.md`](launchable/fields.md).
2. Enter `NVIDIA_API_KEY` only in the required masked parameter field.
3. Keep access set to **Only my organization** and configure a Secure Link on port `8888`; do not expose unrestricted public TCP.
4. Deploy, open JupyterLab through the Secure Link, and run `notebooks/nvmolkit_nemotron_demo.ipynb`.

**Qualification:** This demo is designed for fresh deployment only; it is not yet live-qualified. GPU execution, hosted inference, rendered visuals, and Secure Link access require live acceptance. After a VM stop/start, perform a fresh redeploy and re-enter the masked `NVIDIA_API_KEY`; do not expect `nohup` or key persistence or claim auto-restart.

## Verify

Run the local deterministic checks with `pytest`. GPU acceptance remains a separate future gate: on a compatible NVIDIA GPU, run `RUN_GPU_TESTS=1 pytest` and retain the result before calling the Launchable GPU-accepted.

## Boundaries

This is a research and developer demonstration. It makes no performance claims and no claims about binding, biological activity, ADMET, efficacy, safety, synthesizability, or clinical suitability. Its outputs are computational candidates or summaries that require independent validation.

## Official sources

- [NVIDIA Brev Launchables documentation](https://docs.nvidia.com/brev/concepts/launchables)
- [NVIDIA nvMolKit repository](https://github.com/NVIDIA-BioNeMo/nvMolKit)
- [NVIDIA nvMolKit documentation](https://nvidia-bionemo.github.io/nvMolKit/)
- [NVIDIA Nemotron 3 Nano 30B-A3B model card](https://build.nvidia.com/nvidia/nemotron-3-nano-30b-a3b/modelcard)
- [RDKit documentation](https://www.rdkit.org/docs/)
