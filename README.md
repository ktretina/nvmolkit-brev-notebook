# nvMolKit + Nemotron Notebook

This standalone four-notebook workshop shows a progression from direct nvMolKit calls to two bounded Nemotron agent exercises and a compact companion demo. The companion demo asks one presentation-sized question: **can an AI chemistry agent use nvMolKit to execute a bounded cheminformatics workflow?** One stateful Nemotron conversation follows a fixed dependency chain:

`plan → RDKit inspect → MorganFingerprintGenerator → crossTanimotoSimilarity → fused_butina → EmbedMolecules → MMFFOptimizeMoleculesConfs → objective attempts → evidence-linked, schema-checked conclusion`

For the companion demo, the pinned BioNeMo Agent Toolkit nvMolKit skill grounds the conversation's initial context. Reading it is not an agent tool call and does not permanently teach or modify the model. The compact presentation is one plan, six approvals, six completed command receipts/result cards, one objective challenge, and one conclusion. The user may override only bounded scientific parameters through the supplied dropdowns and sliders. Python validates and executes approved calls, renders results, and preserves exact artifact-grounded evidence records E01–E06 plus the objective ledger O01; credentials, RDKit molecules, tensors, matrices, and coordinates remain local. The conclusion uses a Python-owned headline and facts with Nemotron-selected predicate-true emphasis.

## Four-notebook workshop path

Start with **Module 1**. It is the recommended entry point and introduces the direct library calls before the agent exercises.

1. [`notebooks/01_direct_nvmolkit_reframe.ipynb`](notebooks/01_direct_nvmolkit_reframe.ipynb) — use nvMolKit directly for fingerprints, similarity, and clustering.
2. [`notebooks/02_agent_assisted_reframe_neighborhoods.ipynb`](notebooks/02_agent_assisted_reframe_neighborhoods.ipynb) — let Nemotron choose between bounded neighborhood-analysis policies while Python owns the executable implementation.
3. [`notebooks/03_full_agent_reframe_panel_design.ipynb`](notebooks/03_full_agent_reframe_panel_design.ipynb) — use a bounded planning, approval, execution, and audit loop to design a 24-compound panel.
4. [`notebooks/nvmolkit_nemotron_demo.ipynb`](notebooks/nvmolkit_nemotron_demo.ipynb) — run the compact companion demo with the original six-stage workflow and objective challenge.

**Hosted mode** (`interactive` in `NVMOLKIT_WORKSHOP_MODE`) is the attendee path for the agent exercises in Modules 2 and 3. It uses the fixed `nvidia/nemotron-3-nano-30b-a3b` endpoint and the protected `NVIDIA_API_KEY`; the model selects only within Python-owned bounds. The companion demo also uses the same hosted model and key. **Reference mode** is the deterministic recovery and local-acceptance path for Modules 2 and 3. It makes no hosted client call, needs no API key, and is not evidence of hosted inference. Module 1 is a direct library exercise and does not call the hosted model.

## What runs where

- **Brev** provides the GPU VM and organization-only Secure Link to JupyterLab.
- **Hosted Nemotron** creates one plan, makes six validated tool calls using prior evidence, performs state-bound argmax action selection for up to three measured objective attempts, and selects predicate-true evidence emphasis for the conclusion. It does not execute Python.
- **Python** owns the tool contract, dependency order, validation, execution, exact evidence, and presentation.
- **RDKit** parses inputs, screens MMFF94 eligibility, and supports rendering.
- **nvMolKit on the GPU** generates Morgan fingerprints, computes all-pairs Tanimoto similarity, runs fused Butina clustering, embeds conformers, and performs MMFF94 optimization.

## Interactive flow

Run the final code cell to display the interface; the cell returns after the interface is displayed. Click **Start Agent** to request the plan, review the validated tool call and concise decision, and optionally adjust only the bounded controls. Then click **Approve & Run** on each stage in order. A completed card keeps its validated command receipt beside the corresponding result.

After all six approvals complete through MMFF94, click **Run Objective Challenge**. The bounded eight-candidate challenge asks Nemotron to improve a four-compound panel by maximizing its minimum pairwise Tanimoto distance. The target is 80% of the attainable improvement over the current largest-clusters-first baseline. At each of at most three attempts, the LLM receives the measured current panel, all co-limiting pairs, and up to four ranked legal replacements. It returns one state-bound argmax action selection; Python validates and executes that swap, then independently measures the result. Each visible attempt follows **Observe panel → Candidate menu → Agent chooses → Execute panel → Measure panel** with the actual retained RDKit molecules. A progress bar after every attempt shows the current D_min, remaining target gap, and fraction of required improvement achieved. The final structures and four-by-four similarity heatmap appear before the **Evidence-Backed Conclusion**. Guarded button failures stay inside the active card, leave it incomplete for review or retry, and do not mark the notebook cell failed.

## Launch

Use Linux x86-64 with CPython 3.12 in VM mode. Enable Jupyter in the Brev Console so the Brev-managed Jupyter runtime is available; the setup script installs into that runtime and does not manage the Jupyter service.

1. Create or edit the Launchable in the Brev web Console using [`launchable/fields.md`](launchable/fields.md). Set the default disk storage to **75 GiB**, then paste the current contents of `launchable/setup.sh` into the Software configuration setup-script field. Updating the repository does not replace the script body already saved in a Launchable.
2. Keep only one Launch parameter: required `NVIDIA_API_KEY`, with no default. Remove `NEMOTRON_MODEL` and `JUPYTER_PORT` from Setup values.
3. Enable Jupyter and keep access set to **Only my organization** with a Secure Link on the fixed port `8888`; do not expose unrestricted public TCP. The hosted model is fixed to `nvidia/nemotron-3-nano-30b-a3b`.
4. From the Nemotron model page on build.nvidia.com, generate a hosted NVIDIA Developer API key beginning with `nvapi-`. Enter it once in Setup values when you deploy. The setup script stores it outside the repository in `${HOME}/.config/nvmolkit/NVIDIA_API_KEY` with file mode `0600`, and notebook preflight loads it automatically without a prompt. This is distinct from an NGC personal key. Never expose the key in notebook outputs, logs, screenshots, or chat.
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

Hosted reliability qualification (20 isolated objective trials and three fresh end-to-end runs):

```bash
python scripts/run_objective_reliability.py --trials 20 --end-to-end-runs 3 --output objective-reliability-receipt.json
```

- **Local deterministic acceptance:** run `pytest` to validate notebook structure, scientific state transitions, serialization boundaries, and agent wiring without claiming GPU or hosted execution.
- **GPU acceptance receipt:** on a compatible NVIDIA GPU, run `RUN_GPU_TESTS=1 .venv/bin/python -m pytest -q` and retain the result before calling the nvMolKit runtime GPU-accepted.
- **Hosted reliability receipt:** retain the JSON emitted by `run_objective_reliability.py`; it proves only the bounded hosted trials recorded there. Keep it separate from the GPU acceptance receipt and cross-reference both by commit and run identifier.
- **Persistence receipt:** after a fresh stop/start, record Jupyter, kernel, notebook, and credential-reentry checks separately. Cross-reference it to the reliability and GPU receipts; do not merge their claims.
- **Hosted inference acceptance:** in a fresh Brev kernel, verify one plan, six approvals, six completed command receipts/result cards, up to three state-bound selections with measured feedback, and one evidence-controlled conclusion using a valid hosted Developer API key.
- **Rendered deployment acceptance:** inspect the 24-molecule RDKit preview and invalid-input report, fingerprint histogram, similarity heatmap, cluster chart, conformer-energy chart, static conformer views, molecule-aware objective attempts, per-attempt progress bars, final four structures, and final-panel heatmap through the organization-only Secure Link.

## Boundaries

This is a research and developer demonstration with bounded workflow autonomy. It makes no performance claims and does not establish binding, biological activity, ADMET, efficacy, safety, synthesizability, clinical relevance, or experimentally validated conformations. ETKDGv3 samples candidate geometries, and MMFF94 compares sampled force-field minima only within each molecule; those minima are not global or experimental. Independent computational and experimental validation is required for any intended scientific use.

## Official sources

- [NVIDIA Brev Launchables documentation](https://docs.nvidia.com/brev/concepts/launchables)
- [NVIDIA nvMolKit repository](https://github.com/NVIDIA-BioNeMo/nvMolKit)
- [NVIDIA nvMolKit documentation](https://nvidia-bionemo.github.io/nvMolKit/)
- [BioNeMo Agent Toolkit nvMolKit skill](https://github.com/NVIDIA-BioNeMo/bionemo-agent-toolkit/blob/ce151c15470991c8cb9a0efdd531a124c346ca5b/library-skills/nvMolKit/SKILL.md)
- [NVIDIA Nemotron 3 Nano 30B-A3B model card](https://build.nvidia.com/nvidia/nemotron-3-nano-30b-a3b/modelcard)
- [RDKit documentation](https://www.rdkit.org/docs/)
