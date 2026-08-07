# nvMolKit + Nemotron Notebook

This standalone project asks one presentation-sized question: **can an AI chemistry agent use nvMolKit to autonomously execute a real cheminformatics workflow?** One stateful, bounded Nemotron conversation follows a fixed dependency chain:

`plan → RDKit inspect → MorganFingerprintGenerator → crossTanimotoSimilarity → fused_butina → EmbedMolecules → MMFFOptimizeMoleculesConfs → objective attempts → evidence-linked, schema-checked conclusion`

The pinned BioNeMo Agent Toolkit nvMolKit skill grounds the conversation's initial context. Reading it is not an agent tool call and does not permanently teach or modify the model. The compact presentation is one plan, six approvals, six completed command receipts/result cards, one objective challenge, and one conclusion. The user may override only bounded scientific parameters through the supplied dropdowns and sliders. Python validates and executes approved calls, renders results, and preserves exact artifact-grounded evidence records E01–E06 plus the objective ledger O01; credentials, RDKit molecules, tensors, matrices, and coordinates remain local. Python verifies the conclusion schema, evidence references, and exact rendered metrics; Nemotron's qualitative interpretation is not automatically fact-verified.

## What runs where

- **Brev** provides the GPU VM and organization-only Secure Link to JupyterLab.
- **Hosted Nemotron** creates one plan, makes six validated tool calls using prior evidence, proposes up to three objective panels using measured feedback, and drafts one evidence-linked, schema-checked conclusion. It does not execute Python.
- **Python** owns the tool contract, dependency order, validation, execution, exact evidence, and presentation.
- **RDKit** parses inputs, screens MMFF94 eligibility, and supports rendering.
- **nvMolKit on the GPU** generates Morgan fingerprints, computes all-pairs Tanimoto similarity, runs fused Butina clustering, embeds conformers, and performs MMFF94 optimization.

## Interactive flow

Run the final code cell to display the interface; the cell returns after the interface is displayed. Click **Start Agent** to request the plan, review the proposal and concise decision, and optionally adjust only the bounded controls. Then click **Approve & Run** on each stage in order. A completed card keeps its validated command receipt beside the corresponding result.

After all six approvals complete through MMFF94, click **Run Objective Challenge**. The bounded eight-candidate challenge asks Nemotron to improve a four-compound panel by maximizing its minimum pairwise Tanimoto distance. The target is 80% of the attainable improvement over the current largest-clusters-first baseline. Up to three attempts remain visible in one score trajectory and attempt ledger; every attempt shows the validated Nemotron proposal, the deterministic evaluation executed by Python, the limiting pair, and the result. The final structures and four-by-four similarity heatmap appear before the **Evidence-Backed Conclusion**. Guarded button failures stay inside the active card, leave it incomplete for review or retry, and do not mark the notebook cell failed.

## Launch

Use Linux x86-64 with CPython 3.12 in VM mode. Enable Jupyter in the Brev Console so the Brev-managed Jupyter runtime is available; the setup script installs into that runtime and does not manage the Jupyter service.

1. Create the Launchable in the Brev web Console using [`launchable/fields.md`](launchable/fields.md).
2. Enable Jupyter and keep access set to **Only my organization** with a Secure Link on port `8888`; do not expose unrestricted public TCP.
3. Deploy, open JupyterLab through the Secure Link, and run `notebooks/nvmolkit_nemotron_demo.ipynb`.
4. From the Nemotron model page on build.nvidia.com, generate a hosted NVIDIA Developer API key beginning with `nvapi-`. Provide `NVIDIA_API_KEY` to the kernel or paste only the bare key into the hidden prompt. This is distinct from an NGC personal key. Do not rely on setup-variable persistence, and never save the key in notebook outputs, files, logs, screenshots, or chat.

**Qualification:** This demo is designed for fresh deployment only and is not yet live-qualified. GPU execution, hosted inference, rendered visuals, and Secure Link access each require live acceptance. After a VM stop/start, verify the managed Jupyter service, restart the notebook kernel, rerun the notebook, and enter the key again if prompted; do not claim automatic restart without live evidence.

## Verify

These are separate evidence gates; one does not prove the others:

CPU deterministic suite:

```bash
pytest -q
```

GPU suite on the task-owned compatible NVIDIA GPU:

```bash
RUN_GPU_TESTS=1 pytest tests/test_gpu_acceptance.py -v
```

Hosted reliability qualification (20 isolated objective trials and three fresh end-to-end runs):

```bash
python scripts/run_objective_reliability.py --trials 20 --end-to-end-runs 3 --output objective-reliability-receipt.json
```

- **Local deterministic acceptance:** run `pytest` to validate notebook structure, scientific state transitions, serialization boundaries, and agent wiring without claiming GPU or hosted execution.
- **GPU acceptance:** on a compatible NVIDIA GPU, run `RUN_GPU_TESTS=1 pytest tests/test_gpu_acceptance.py -v` and retain the result before calling the nvMolKit runtime GPU-accepted.
- **Hosted inference acceptance:** in a fresh Brev kernel, verify one plan, six approvals, six completed command receipts/result cards, up to three objective proposals with measured feedback, and one evidence-linked, schema-checked conclusion using a valid hosted Developer API key; this does not fact-verify Nemotron's qualitative interpretation.
- **Rendered deployment acceptance:** inspect the 24-molecule RDKit preview and invalid-input report, fingerprint histogram, similarity heatmap, cluster chart, conformer-energy chart, static conformer views, objective score trajectory, attempt ledger, final four structures, and final-panel heatmap through the organization-only Secure Link.

## Boundaries

This is a research and developer demonstration with bounded workflow autonomy. It makes no performance claims and does not establish binding, biological activity, ADMET, efficacy, safety, synthesizability, clinical relevance, or experimentally validated conformations. ETKDGv3 samples candidate geometries, and MMFF94 compares sampled force-field minima only within each molecule; those minima are not global or experimental. Independent computational and experimental validation is required for any intended scientific use.

## Official sources

- [NVIDIA Brev Launchables documentation](https://docs.nvidia.com/brev/concepts/launchables)
- [NVIDIA nvMolKit repository](https://github.com/NVIDIA-BioNeMo/nvMolKit)
- [NVIDIA nvMolKit documentation](https://nvidia-bionemo.github.io/nvMolKit/)
- [BioNeMo Agent Toolkit nvMolKit skill](https://github.com/NVIDIA-BioNeMo/bionemo-agent-toolkit/blob/ce151c15470991c8cb9a0efdd531a124c346ca5b/library-skills/nvMolKit/SKILL.md)
- [NVIDIA Nemotron 3 Nano 30B-A3B model card](https://build.nvidia.com/nvidia/nemotron-3-nano-30b-a3b/modelcard)
- [RDKit documentation](https://www.rdkit.org/docs/)
