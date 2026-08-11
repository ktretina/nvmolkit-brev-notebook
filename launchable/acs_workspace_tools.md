# ACS chemistry workshop

Read the installed `nvmolkit-usage` skill once before the first lesson. Use only
the commands below. Run the three lessons in order.

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
