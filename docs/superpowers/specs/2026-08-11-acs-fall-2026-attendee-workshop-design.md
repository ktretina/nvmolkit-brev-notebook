# ACS Fall 2026 Attendee Workshop and OpenClaw Chemistry Runner

**Date:** 2026-08-11
**Status:** Approved design; implementation is blocked on written-spec review

## Objective

Make the two-part ACS Fall 2026 workshop usable by a chemist who has no prior
Brev, OpenClaw, or Python experience.

The first lab keeps the existing guided notebook. The second lab extends the
existing OpenClaw Launchable so attendees can reproduce the notebook's six
chemistry stages through a Nemotron conversation and then complete the bounded
four-compound diversity challenge. Every scientific result remains calculated
by validated Python, RDKit, and nvMolKit code. Nemotron selects only allowed
actions and explains measured outputs.

The final attendee handout is one concise GitHub Markdown file. It gives the
account, API-key, Launchable, prompt, artifact-download, troubleshooting, and
scientific-boundary instructions in one linear page.

## Audience and Experience Principles

The primary attendee is an ACS chemist, not a cloud or agent developer. The
experience must therefore have these properties:

1. The attendee uses account forms, two Launchable links, and the OpenClaw chat.
   A terminal is not part of the attendee workflow.
2. Every task starts with a copy-paste prompt that names the chemistry question
   before it names the command.
3. Each task returns a short scientific answer, one useful native image when a
   plot helps, and downloadable machine-readable artifacts.
4. The workflow uses one fixed, included 256-molecule dataset. It does not ask
   attendees to find, upload, or clean data during the workshop.
5. Errors stop at the affected task and give one clear next action. The agent
   must not install packages, use the network, or improvise another workflow.
6. Results are described as computational molecular descriptors and sampled
   force-field geometries. They are not presented as biological or experimental
   evidence.

The page will use Markdown only. A custom site, visual page builder, or new web
application would add maintenance work without improving the reference-sheet
use case.

## Selected Approach

Add one fixed command-line workshop runner to the existing sandbox and reuse the
existing scientific modules. OpenClaw and hosted Nemotron remain the only agent
and conversational layer.

This approach is smaller and safer than either alternative:

- A page-only release would be short, but the current OpenClaw sandbox cannot
  run the objective challenge or all six notebook stages.
- Porting `demo_agent.py` and `interactive_workflow.py` would create a second
  agent loop and a second UI inside OpenClaw. It would duplicate responsibilities
  and add avoidable failure modes.

The implementation must not upload or import `demo_agent.py`,
`interactive_workflow.py`, `objective_findings.py`, or notebook widget code.

## Runtime Architecture

### Existing components retained

- Brev provides one NVIDIA L4 VM and authenticated Secure Links.
- NemoClaw creates the OpenClaw sandbox and connects it to hosted
  `nvidia/nemotron-3-super-120b-a12b` through the `inference` provider.
- OpenClaw provides the attendee chat, tool transcript, and native image display.
- The installed `nvmolkit-usage` skill describes the supported nvMolKit use.
- `chemistry_workflow.py` remains the authority for the six ordered scientific
  stages.
- `objective_challenge.py` remains the authority for candidate construction,
  exact scores, allowed swaps, attempt limits, evidence, and figures.
- The existing artifact server on Secure Link port `8765` serves files below
  the workspace `outputs` directory.
- The raw OpenClaw dashboard stays loopback-only on port `18789`. Attendees use
  only the protected proxy on Secure Link port `18788`.

### New component

Add `acs_workshop_runner.py`. It has three responsibilities only:

1. run an exact dependency prefix through one named chemistry stage;
2. serialize fixed summaries and images into owned output directories; and
3. expose the existing objective challenge through a bounded, state-bound CLI.

The runner must not call an LLM, install software, accept Python source, accept
an arbitrary dataset path, or accept an arbitrary output path.

### Fixed files and trust boundary

The setup adds these setup-verified fixed files:

- `/sandbox/.openclaw/workspace/acs_workshop_runner.py`
- `/sandbox/.openclaw/workspace/objective_challenge.py`

It retains these setup-verified fixed inputs:

- `/sandbox/.openclaw/workspace/chemistry_workflow.py`
- `/sandbox/.openclaw/workspace/data/sample_molecules.csv`
- `/sandbox/.openclaw/workspace/TOOLS.md`

The setup script must require, remove stale copies of, upload, hash-check, and
post-turn verify the two new files in the same way as the existing fixed assets.
It writes a closed hash manifest outside `outputs`, sets the fixed files and
manifest read-only, and runs a non-GPU import and CLI-help smoke test before
writing its ready marker. Every runner invocation checks its own source and the
fixed companion files against that manifest before scientific work. A mismatch
stops the command.

The manifest path is:

```text
/sandbox/.openclaw/workspace/.acs-workshop-state/manifest.json
```

This is an integrity check for the bounded workshop, not a security boundary
against a malicious same-user process. OpenClaw tools run as the sandbox user,
which can change its own files and permissions. The prompts forbid those edits;
if the runner or manifest is deliberately changed, the accepted workshop
contract no longer applies. The existing threshold-0.80 seed task and its
acceptance checks remain unchanged.

`launchable/acs_workspace_tools.md` must replace its current generic instruction
to create source files and per-task ZIP files. It must instead list the exact
runner commands, state that only the runner creates workshop artifacts and the
complete ZIP, and state that the agent may not edit the runner, objective module,
workflow module, dataset, or integrity manifest. The setup-only seed prompt may
edit only its existing `acs_chemistry_task.py` edit point.

Add `launchable/acs_workshop_prompts.md` as the canonical seven-prompt acceptance
source. This prompt file stays in the repository; it is not needed in the
sandbox. The final attendee page copies these seven prompt blocks verbatim, and
a static test prevents the two copies from drifting.

## Fixed Scientific Profile

All stage commands use the same notebook profile:

- expected dataset rows: `256`;
- Morgan radius: `2`;
- Morgan fingerprint size: `1024` bits;
- fused Butina cutoff: `0.40`;
- representative policy: `largest_clusters_first`;
- representative count: `6`;
- conformers per representative: `5`;
- ETKDGv3 random seed: the existing fixed value `7`; and
- MMFF94 maximum iterations: the existing fixed value `500`.

These values are not attendee controls. Using one profile keeps both labs
comparable and makes artifact interpretation clear. The live L4 acceptance must
qualify this exact profile before the reference sheet is called attendee-ready.

## Stage CLI and Data Flow

The runner accepts one of six canonical stage names:

```text
inspect_library
generate_morgan_fingerprints
measure_tanimoto_similarity
discover_fused_butina_clusters
embed_representative_conformers
optimize_conformers_mmff94
```

The exact command form is:

```bash
env PYTHONPATH=/tmp/.local/lib/python3.13/site-packages \
  python3 /sandbox/.openclaw/workspace/acs_workshop_runner.py \
  run-stage inspect_library
```

The last argument changes to one of the other five canonical stage names for
prompts 2 through 6.

Each call creates a new `WorkflowState` and executes the exact dependency prefix
from library inspection through the requested stage. The fixed dataset is small
enough for this safe replay model. The design deliberately avoids pickle files,
long-running worker processes, and hidden Python object state.

The stage directories are fixed:

```text
outputs/workshop/01-inspection/
outputs/workshop/02-fingerprints/
outputs/workshop/03-similarity/
outputs/workshop/04-clusters/
outputs/workshop/05-conformers/
outputs/workshop/06-mmff94/
outputs/workshop/07-objective/
outputs/workshop/results.zip
```

Every completed stage directory contains:

- `README.md` with the question, method, result, and scientific limit;
- `summary.json` with schema version, dataset SHA-256, fixed profile, stage
  facts, GPU identity when applicable, and generated artifact names; and
- the stage figures already produced by `StageResult.figures`, saved with clear
  names.

The expected reusable artifacts are:

| Stage | Native image artifact | Downloadable chemistry data |
| --- | --- | --- |
| Inspection | `library_preview.png` | validated molecule records in `summary.json` |
| Fingerprints | `fingerprint_density.png` | fingerprint parameters and density facts in `summary.json` |
| Similarity | `similarity_heatmap.png` | `top_similarity_pairs.csv`, `similarity_matrix.csv` |
| Clusters | `cluster_sizes.png` | `cluster_assignments.csv` |
| Conformers | `embedding_counts.png` | representative and conformer counts in `summary.json` |
| MMFF94 | `conformer_energies.png`, `optimized_structures.png` | `mmff94_energies.csv`, `optimized_conformers.sdf` |

The runner must use separate, validated save adapters for Matplotlib figures and
PIL images. Stage 6 also writes `workflow_evidence.json` from the existing
E01-E06 workflow report so the final explanation can be checked against Python
facts. CSV rows use stable molecule identifiers and ordering. The SDF stores the
optimized coordinates and molecule, conformer, convergence, and MMFF94-energy
properties needed to relate each structure to `mmff94_energies.csv`.

After every successful command, the runner rebuilds
`outputs/workshop/results.zip` from completed public artifacts. ZIP members use
relative safe paths, fixed ordering, and deterministic metadata. The archive
never contains the inference key, gateway token, private objective checkpoint,
session transcript, raw model response, temporary files, or a prior copy of the
root archive itself.

The command prints one small JSON result envelope. It contains the status,
stage, concise summary, image paths, artifact directory, current ZIP filesystem
path, and the artifact-relative ZIP path `workshop/results.zip`. OpenClaw uses
only image paths in standalone `MEDIA:` lines. ZIP download uses the authenticated
**Download Results** Secure Link and never depends on `MEDIA:` handling.

## Bounded Objective Challenge

The objective is unchanged from the notebook: select a four-compound panel that
maximizes its minimum pairwise Morgan/Tanimoto distance.

The runner reuses the exact production contracts in `objective_challenge.py`:

- eight candidates from distinct eligible clusters;
- four unique compounds in a panel;
- an exact Python-calculated baseline, attainable benchmark, and target;
- no more than three accepted swaps (`MAX_ATTEMPTS = 3`);
- a state-bound menu of at most three legal improving swaps;
- acceptance of only a displayed tied-maximum action;
- exact recomputation of every selected result; and
- measured success or truthful bounded failure.

After a successful stage 6, that stage command creates the private checkpoint
below:

```text
/sandbox/.openclaw/workspace/.acs-workshop-state/context.json
/sandbox/.openclaw/workspace/.acs-workshop-state/history.json
```

The checkpoint contains only schema-checked data: dataset hash, fixed profile,
candidate provenance, the validated 8-by-8 distance matrix, objective scores,
and accepted `{state_id, swap_id}` selections. Files use mode `0600`, atomic
replacement, closed key sets, finite numeric values, and no pickle or executable
content. They are outside the artifact-server root and the final ZIP.

The objective commands are:

```bash
env PYTHONPATH=/tmp/.local/lib/python3.13/site-packages \
  python3 /sandbox/.openclaw/workspace/acs_workshop_runner.py objective-start

env PYTHONPATH=/tmp/.local/lib/python3.13/site-packages \
  python3 /sandbox/.openclaw/workspace/acs_workshop_runner.py objective-step \
  --state-id 'STATE_ID_FROM_MENU' --swap-id 'SWAP_ID_FROM_MENU'
```

The agent replaces both quoted values with the exact strings returned by the
current menu. Quoting the swap ID is mandatory because it contains `->`.

`objective-start` is read-only with respect to objective history. It validates
the checkpoint created by stage 6 and returns its current menu or terminal
result. It is idempotent and never creates, resets, or erases accepted history.
A repeated stage-6 command preserves a matching checkpoint and its history; it
fails on a conflicting checkpoint instead of resetting it.

For every `objective-step`, the runner reconstructs the exact context, replays
the complete accepted history through `build_action_menu`,
`resolve_menu_action`, and `evaluate_selected_swap`, and then validates the new
state and action. An invalid, stale, non-maximum, invented, or fourth selection
is rejected before history changes and does not consume an attempt.

When the objective terminates, the runner reconstructs the fixed library with
RDKit by running only `inspect_library` into a new in-memory state. This supplies
the molecule objects and candidate indices required by `objective_figures`
without repeating GPU stages or trusting serialized Python objects. It then
writes:

- `objective_summary.json`;
- `objective_evidence.json` using the existing O01 evidence builder;
- `score_trajectory.png`;
- `final_panel.png`;
- `final_similarity_heatmap.png`; and
- an updated complete `outputs/workshop/results.zip`.

Nemotron may explain why it chose one of the accepted maximum actions. Its prose
is not part of the scientific evidence record.

## Conversational Prompt Contract

The attendee page contains seven numbered, self-contained prompts used in one
new OpenClaw session:

1. inspect and preview the fixed library;
2. generate and summarize Morgan fingerprints on the L4;
3. calculate and interpret all-pairs Tanimoto similarity;
4. cluster the library with fused Butina;
5. generate representative ETKDGv3 conformers;
6. optimize and compare sampled conformers with MMFF94; and
7. complete the bounded four-compound diversity challenge.

Each stage prompt must:

- name the scientific question;
- direct the agent to read the installed nvMolKit skill before chemistry work;
- include the exact fixed runner command once;
- forbid package installation, network access, retries with other commands, and
  edits to setup-verified fixed inputs;
- require a short result and the relevant scientific limit;
- require the primary image as a standalone `MEDIA:` path; and
- report that the ZIP is available at `workshop/results.zip` through the
  authenticated **Download Results** link.

The objective prompt may call `objective-start` once and `objective-step` at
most three times. At each step, Nemotron must report the observed limiting pair,
choose only an exact displayed action with the highest predicted minimum
distance, submit the exact state and swap IDs, and stop immediately on a
terminal result. It then gives a short evidence-controlled conclusion grounded
only in `workflow_evidence.json`, `objective_summary.json`, and
`objective_evidence.json`. The runner, not the prompt, enforces the scientific
and attempt limits.

## Attendee Reference Sheet

Create `docs/acs-fall-2026-workshop.md` only after the extended Launchable passes
fresh live acceptance. The page has this order:

1. workshop title and one-sentence purpose;
2. before-the-workshop checklist;
3. Brev account steps with the official Brev link;
4. NVIDIA account and hosted Nemotron API-key steps using the correct
   `build.nvidia.com` site;
5. a warning to keep the `nvapi-...` key out of chat, screenshots, and files;
6. a short note that hosted endpoint access may be rate-limited and Brev compute
   is separately billable;
7. a small key-entry table: Lab 1 uses the deployment field `NVIDIA_API_KEY`,
   while Lab 2 uses `NVIDIA_INFERENCE_API_KEY`; the same private `nvapi-...`
   value is valid for both fields;
8. Lab 1 description and the guided-notebook Launchable link;
9. Lab 2 description and the OpenClaw Launchable link;
10. instructions to use the Launchable default hardware and, for Lab 2, verify
    the visible resource row shows one NVIDIA L4, x86-64, 4 CPUs, 16 GiB RAM,
    and 128 GiB disk; attendees are not required to find an explicit
    `g6.xlarge` label because Brev may show only the resource description;
11. instructions to wait for setup readiness, open the protected chat link, and
    start one new session;
12. the seven copy-paste prompts;
13. native-image and **Download Results** instructions, including that attendees
    use the authenticated artifact page rather than treating a
    `/sandbox/...` path as a browser URL;
14. concise troubleshooting;
15. scientific-use limits; and
16. official source links.

The page uses these current workshop links, subject to final link checks during
acceptance:

- repository: `https://github.com/ktretina/nvmolkit-brev-notebook`;
- guided notebook Launchable:
  `https://brev.nvidia.com/launchable/deploy/now?launchableID=env-3HJtJW3qHg4Dw1I3xt75BfpBmZW`;
- OpenClaw Launchable:
  `https://brev.nvidia.com/launchable/deploy/now?launchableID=env-3Hlp4pHBlTTlfDxfH41KkGhTeCV`;
- Brev account: `https://brev.nvidia.com/`;
- NVIDIA account: `https://developer.nvidia.com/login`; and
- hosted Nemotron model and key page:
  `https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b?nim=hosted`.

The page must not claim that an environment is free. It states that API access
for prototyping can be free and rate-limited, while Brev VM use consumes credits
or incurs compute charges. It tells attendees to stop or delete their Brev
environment after the workshop.

## Error Handling and Retry Rules

- A missing or non-L4 CUDA device stops GPU stages with a short safe error.
- An unsupported stage name exits nonzero and lists only the six allowed names.
- A stage failure leaves its prior complete directory unchanged. The runner
  writes to a temporary owned directory and publishes only after validation.
- A missing or mismatched stage-6 checkpoint blocks the objective and instructs
  the attendee to complete prompt 6.
- A malformed, provenance-mismatched, or internally inconsistent checkpoint,
  dataset hash, profile, state ID, swap ID, history sequence, or score blocks
  the objective without changing history.
- A hosted-model timeout stops the current prompt. The page allows one retry of
  the same prompt only; it does not tell attendees to change models or install
  software.
- Artifact paths remain below the fixed workshop output root. The runner must
  reject path traversal and must not follow a pre-existing symlink when replacing
  managed outputs or private state.
- Errors and artifacts must not contain credentials, gateway tokens, tokenized
  dashboard URLs, or raw provider responses.

## Verification

### Local deterministic gates

Tests must verify:

- only the six exact stage names are accepted;
- each stage executes the exact ordered dependency prefix with the fixed profile;
- no command accepts an arbitrary dataset or output path;
- every summary has closed keys, the fixed dataset hash, finite JSON values, and
  the expected stage/GPU fields;
- every expected PNG is valid and every ZIP member path is safe;
- Matplotlib and PIL images both use the correct save path and remain readable;
- every runner command stops before science when its manifest or a fixed-file
  hash no longer matches;
- similarity CSV dimensions and identifiers match the fixed library;
- cluster assignments cover every valid molecule exactly once;
- every SDF conformer and MMFF94 energy row has matching stable provenance;
- reruns are idempotent and a failed replacement preserves the prior complete
  artifacts;
- private state is not served or archived;
- stage 6 alone creates the initial objective context and empty history, a
  matching stage-6 rerun preserves accepted history, and `objective-start`
  cannot create, reset, or change them;
- objective context and history have closed schemas and atomic writes;
- objective replay rejects field mutations, stale state, invented actions,
  non-maximum actions, duplicate steps, and a fourth attempt;
- every accepted action is measured exactly once and terminal success stops the
  loop;
- the current seed-task, dashboard proxy, artifact server, upload cleanup,
  timeout, and secret-handling tests remain green;
- setup uploads and hash-checks both new files before readiness;
- all seven canonical prompt blocks name the exact runner command and correct
  output path.

Tests may replace only the GPU execution boundary for local runner artifact
tests. Objective-domain tests must use the real `objective_challenge.py` logic.
The existing full repository test suite, Ruff, strict scoped mypy, Bash syntax,
Node proxy tests, Python compilation, and secret scan must remain green.

### Fresh Brev L4 acceptance

The current running instance cannot qualify code that it does not contain. After
local review, the implementation requires a new public commit, an updated
exact-commit Console bootstrap, and a fresh deployment of the OpenClaw
Launchable.

On the fresh instance, acceptance must verify:

1. setup reaches the ready marker on exactly one NVIDIA L4;
2. only Secure Links `18788` and `8765` are exposed;
3. the hosted model is the configured Nemotron 3 Super model;
4. a new OpenClaw session completes all seven prompts in order;
5. every nvMolKit stage reports CUDA execution and produces its expected facts;
6. every required image renders natively in the OpenClaw chat;
7. objective history contains zero to three accepted, state-bound, exactly
   measured swaps and a truthful terminal reason;
8. `outputs/workshop/results.zip` downloads through the authenticated artifact
   link and passes archive and manifest checks;
9. the downloaded summaries agree with the displayed answers;
10. no key, gateway token, or tokenized URL appears in setup output, chat text,
    public artifacts, or the final repository diff; and
11. the measured setup and seven-prompt elapsed times are recorded so workshop
    instructions can set honest expectations.

The reference sheet is attendee-ready only after all eleven checks pass. A
local test pass, prior seed-task result, or successful artifact download does
not replace this fresh end-to-end acceptance.

### Post-acceptance reference-sheet gates

After the live checks pass and the final Markdown page is written, static tests
must verify:

- the page prompt blocks match `launchable/acs_workshop_prompts.md` verbatim;
- both exact Launchable links and all official account and API-key links are
  present;
- the visible hardware and API-key field instructions match the accepted live
  deployment; and
- no draft status, private instance URL, credential, token, or unverified timing
  claim remains.

## Release and Approval Boundaries

Implementation proceeds in this order:

1. add the runner, setup wiring, canonical prompt source, and deterministic
   tests;
2. run local verification and review the exact diff;
3. commit the reviewed implementation on the ACS branch;
4. obtain explicit approval before any public push or Brev deployment change;
5. update the Console bootstrap to the reviewed public commit and deploy one
   fresh L4 instance;
6. perform the live acceptance above;
7. write and locally verify `docs/acs-fall-2026-workshop.md`; and
8. leave publication of the final page to a separate explicit user request.

Deleting or stopping the acceptance instance is part of the approved live-run
cleanup. The implementation must not change Brev organizations.

## Non-Goals

- arbitrary attendee datasets or arbitrary chemistry code;
- a general-purpose molecular visualization application;
- a second agent/controller framework inside OpenClaw;
- package installation during attendee prompts;
- biological, pharmacological, synthetic, safety, or experimental conclusions;
- production inference guarantees or service-level claims;
- automatic Brev or NVIDIA account creation; and
- automatic publication of the Markdown page.

## Completion Criteria

The project goal is met when an ACS attendee can follow one concise Markdown
page, create the required accounts and key, open each Launchable, complete the
guided notebook lab, and then use seven copy-paste prompts in OpenClaw to run the
same six-stage chemistry workflow plus the bounded objective challenge. The
attendee must see useful chemistry images in chat and download a validated
artifact bundle, while Python remains the authority for every scientific value
and the final explanation stays within the stated computational limits.
