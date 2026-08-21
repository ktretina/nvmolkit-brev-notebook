# ACS chemistry workshop

The installed `nvmolkit-usage` skill remains available after the bounded exercise.

## Completion state

Each lesson command has a one-call budget. After the first top-level `status: complete`, stop tool use and answer from that first result. Never run that lesson again in the same prompt. An empty assistant response does not permit another tool call. Copy the decoded `answer_markdown` string exactly, with no added opening or closing text. The `sampled-3d-geometry` command returns both conformer stages; run it only once.

For the objective, run `objective-start` once. If that result is terminal, run no objective-step command. Otherwise, select only a displayed action tied for the maximum numeric `predicted_score`. Use the exact returned `state_id` and `swap_id`. Keep both values single-quoted. Run at most three objective-step commands and stop at the first terminal result. Copy its decoded `answer_markdown` string exactly.

The library is a deterministic 256-record ChEMBL convenience sample and is non-representative chemical space. Fingerprint conclusions depend on the radius-2, 1024-bit hashed fingerprint. Report real GPU execution with no acceleration or speedup claim. The cutoff `0.40` is Tanimoto distance, and similarity `1.0` does not prove molecular identity. nvMolKit computes fingerprints and Tanimoto similarities on GPU; RDKit runs Butina clustering on CPU.

The deterministic selected molecules are not centroids, medoids, or globally optimal representatives. Sampled conformers are not experimental structures, and MMFF94 energies compare sampled conformers within one molecule only. `D_min` is the minimum pairwise Tanimoto distance, with `D_min = min(1 - Tanimoto similarity)`; higher `D_min` means greater separation. It is the weakest-link diversity score within eight fixed candidates. Never call `D_min` a similarity score. Do not report intermediate, predicted, target, or per-step scores. This structural-descriptor objective does not demonstrate unrestricted autonomous design or biological performance.

Do not read files during these four prompts. Use only the commands below. Run the
three lessons in order.

```bash
env PYTHONPATH=/tmp/.local/lib/python3.13/site-packages python3 /sandbox/.openclaw/workspace/acs_workshop_runner.py run-lesson data-and-representation
```

Display:
`MEDIA:/sandbox/.openclaw/workspace/outputs/workshop/01-inspection/library_preview.png`

```bash
env PYTHONPATH=/tmp/.local/lib/python3.13/site-packages python3 /sandbox/.openclaw/workspace/acs_workshop_runner.py run-lesson relationships-and-groups
```

Display:
`MEDIA:/sandbox/.openclaw/workspace/outputs/workshop/04-clusters/cluster_sizes.png`

```bash
env PYTHONPATH=/tmp/.local/lib/python3.13/site-packages python3 /sandbox/.openclaw/workspace/acs_workshop_runner.py run-lesson sampled-3d-geometry
```

Display:
`MEDIA:/sandbox/.openclaw/workspace/outputs/workshop/06-mmff94/optimized_structures.png`

Start the objective after all three lessons:

```bash
env PYTHONPATH=/tmp/.local/lib/python3.13/site-packages python3 /sandbox/.openclaw/workspace/acs_workshop_runner.py objective-start
```

If the result is not terminal, copy one displayed action that is tied for the
maximum predicted `D_min`. Keep both values single-quoted. Accept no more than
three actions, and stop when the result is terminal:

```bash
env PYTHONPATH=/tmp/.local/lib/python3.13/site-packages python3 /sandbox/.openclaw/workspace/acs_workshop_runner.py objective-step --state-id 'STATE_ID_FROM_MENU' --swap-id 'SWAP_ID_FROM_MENU'
```

Display the terminal objective image:
`MEDIA:/sandbox/.openclaw/workspace/outputs/workshop/07-objective/final_panel.png`

Download all public files from `workshop/results.zip` in the Download Results
service.

Do not install software. Do not use the network. Do not run alternate commands.
Do not edit the runner, workflow, objective code, fixed input CSV, provenance,
`TOOLS.md`, or `.acs-workshop-state` and its manifest. Write no files outside
the fixed runner.

Fingerprints and molecular similarity are computational descriptors. They are
not evidence of biological activity, binding, efficacy, or safety. Clusters
depend on the fixed fingerprint and cutoff. Sampled conformers are not
experimental structures. MMFF94 energies compare sampled conformers within one
molecule only.
