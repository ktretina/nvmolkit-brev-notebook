# ACS Prompt Reliability and Scientific UX Design

**Date:** 2026-08-21
**Status:** Approved design; awaiting written-spec review

## Goal

Make the four ACS Fall 2026 OpenClaw prompts easier for a chemist to understand
while preserving the bounded execution contract and preventing the live failures
observed on Brev instance `8id74izoa`.

Success requires all of the following:

1. each prompt begins with a clear scientific objective;
2. the agent uses only the fixed workshop commands and stops at the required
   state transition;
3. a repeated Prompt 3 command cannot turn a completed lesson into an
   attendee-visible failure;
4. every final answer uses the correct scientific definitions, limits, CPU/GPU
   attribution, image, and download location;
5. a committed verifier evaluates the complete live trajectory and artifact
   bundle; and
6. the local source, public workshop page, running instance, and future
   Launchable bootstrap remain synchronized through reviewed immutable commits.

## Evidence From the Live Failure

The published four-prompt sequence was run in an isolated QA session on one
NVIDIA L4 with `nvidia/nemotron-3-super-120b-a12b`.

- Prompt 1 completed its one command but described GPU execution as
  acceleration.
- Prompt 2 passed its execution and scientific-answer contracts.
- Prompt 3 returned top-level `status: complete`, valid stage results, and a
  valid ZIP. The assistant then emitted a blank text fragment and called the
  identical command again. The second call exited `2`, and the final answer
  discarded the first success.
- Prompt 4 selected a maximum predicted action at both steps and reached the
  target. Its answer called minimum Tanimoto distance a similarity score and
  reported an intermediate score that the prompt excluded.

The runner currently executes the complete workflow before checking whether a
matching Prompt 3 objective state already exists. The active `TOOLS.md` and
prompt blocks already forbid retries, so wording alone is not an adequate fix.

## Considered Approaches

### 1. Prompt wording and OpenClaw loop detection only

This is the smallest change, but it leaves the completed Prompt 3 result
vulnerable to one accidental duplicate. OpenClaw loop detection is disabled by
default and is a defense against repeated patterns, not a replacement for
idempotent scientific execution.

### 2. Validated runner replay, canonical answers, and live verification

This is the selected approach. It uses existing objective-state and archive
bindings, adds no new private receipt schema, and makes every scientific answer
deterministic while retaining the model's bounded tool use and Prompt 4 action
selection.

### 3. New OpenClaw enforcement plugin

A plugin could intercept duplicate tool calls and revise final answers, but it
would add a trusted runtime extension, hook permissions, installation and
restart behavior, and another failure surface. It is deferred unless the
selected approach fails repeated live acceptance.

## Selected Architecture

The attendee interaction remains:

```text
scientific objective -> fixed runner command -> validated JSON result
                     -> canonical answer_markdown -> image and ZIP
```

Nemotron still performs the bounded interaction:

- for Prompts 1-3, it invokes one exact lesson command;
- for Prompt 4, it starts the objective, selects only a displayed action tied
  for maximum predicted `D_min`, and stops at the terminal result; and
- after a completed result, it returns the runner's `answer_markdown` exactly.

Validated Python, RDKit, and nvMolKit remain the authority for every scientific
value. Model prose is no longer the authority for execution placement,
definitions, numerical facts, or scientific limits.

## Runner Response Contract

### Execution provenance

Every compact stage item gains a closed `execution` object:

```json
{
  "placement": "CPU",
  "software": "fixed software name",
  "operation": "fixed operation name",
  "upstream": null,
  "gpu": null
}
```

`placement` is exactly `CPU` or `GPU`. `upstream` is either `null` or a closed
object with the source stage ID, placement, software, and operation; it is
non-null only when a CPU stage consumes a prior GPU result. `gpu` is either
`null` or the existing verified `GpuIdentity`; it is non-null only when the
stage or its declared upstream provenance depends on that GPU execution. No
additional execution keys are permitted.

The fixed stage boundaries are:

- RDKit library parsing and validation on CPU;
- nvMolKit Morgan fingerprint generation on GPU;
- nvMolKit Tanimoto similarity on GPU;
- RDKit Butina clustering on CPU using nvMolKit GPU-computed Tanimoto
  distances;
- nvMolKit ETKDGv3 conformer embedding on GPU; and
- nvMolKit MMFF94 conformer optimization on GPU.

Execution placement is not a performance comparison. No canonical answer uses
`accelerated` or `acceleration`, and no answer claims speedup.

### Canonical answer

Each successful lesson result and each terminal objective result gains one
`answer_markdown` string. The runner builds it only from validated facts, fixed
scientific text, fixed execution labels, and fixed artifact paths.

Every canonical answer has exactly these headings in this order:

1. `Question`
2. `What ran`
3. `Measured result`
4. `Meaning`
5. `Scientific limit`
6. `Image and download location`

It ends with the exact prompt-specific `MEDIA:` line. It reports no more than
three measured result statements. The lesson result and a validated replay use
the same response constructor and contain no `replayed` marker, so an identical
duplicate returns the identical public result.

The objective returns `answer_markdown` only when `terminal` is `true`. Pending
menus contain no provisional answer.

## Prompt 3 Validated Replay

`run_lesson("sampled-3d-geometry")` checks for prior objective state after
manifest and argument validation but before any workflow executor call.

If both `context.json` and `history.json` are absent, the first execution
proceeds normally. Any other state enters fail-closed replay validation:

1. require both private files and reject symlinks or partial state;
2. validate the private root, canonical JSON, schemas, context hash, derived
   objective state, attempts, and cached result with the existing objective
   loader;
3. validate the current ZIP and require its canonical stage-only bytes to match
   `stage_results_zip_sha256` from the private context;
4. validate both `05-conformers` and `06-mmff94` directories;
5. require every public stage byte to equal its bound ZIP member; and
6. reconstruct the two compact stage items and canonical answer from those
   validated bytes without invoking chemistry or changing objective history.

A pending objective requires a stage-only ZIP with no objective members. A
terminal objective uses the existing terminal publication validation and
recovery contract. Invalid, partial, forged, unbound, or byte-divergent state
fails before the executor and is never converted to success.

This design addresses sequential duplicate calls, which matches the observed
failure. Concurrent first executions are outside the workshop interaction and
are not added to this scope.

## Human-Friendly Prompt Design

The four prompt blocks retain the exact workspace, command, network, file,
retry, response-heading, download, and media constraints. Their presentation
changes to this order:

1. the scientific question;
2. a short `Scientific objective` written for a chemist;
3. a compact `Execution contract` with the exact command and state transition;
4. the canonical-answer instruction; and
5. the prompt-specific scientific interpretation limits.

Prompts 1-3 state that the first top-level `status: complete` result is final,
consumes the command budget, and requires immediate final output with no blank
placeholder or further tool call.

Prompt 4 explains the bounded action loop in scientific terms. It defines
`D_min` as the minimum pairwise Tanimoto distance,
`min(1 - Tanimoto similarity)`, and calls it the weakest-link diversity score
within eight fixed candidates. Its final answer includes only baseline
`D_min`, final `D_min`, and their change. It excludes intermediate, predicted,
target, and per-step scores.

The prompt blocks in `docs/acs-fall-2026-workshop.md` and
`NVIDIA/digital-biology-examples:gh-pages/acsfall26/README.md` must remain
byte-identical. Updated prompt hashes replace the current byte locks.

## OpenClaw Workspace Guidance and Loop Detection

`launchable/acs_workspace_tools.md` remains the active read-only `TOOLS.md`.
It will describe the same state machine and tell the agent to return
`answer_markdown` byte-for-byte after success.

The setup enables `tools.loopDetection.enabled = true` through the existing
NemoClaw configuration command and reads the value back as JSON `true`. This is
defense in depth only. Validated runner replay remains the control that protects
the attendee result from one duplicate call.

No OpenClaw plugin is added in this change.

## Live Trajectory Verifier

Add a standard-library-first verifier under `scripts/` with focused tests. It
accepts an OpenClaw trajectory JSONL file and the final workshop ZIP. It emits a
small pass/fail receipt without prompts, tokens, credentials, or full model
responses.

It verifies:

- the exact four prompt hashes and order;
- one exact command for each of Prompts 1-3;
- the Prompt 4 start/step sequence, exact state IDs, displayed swap IDs, and
  maximum predicted-score choice;
- successful command results and terminal state;
- exact equality between assistant output and returned `answer_markdown`;
- the six headings, final media line, and download location;
- absence of forbidden acceleration language and incorrect `D_min` language;
- absence of Prompt 4 intermediate or predicted scores in the final answer;
- safe ZIP paths, deterministic member structure, and no corrupt member; and
- valid signatures and dimensions for the four required PNG files.

Any non-timeout contract failure fails live acceptance. A provider timeout may
be retried only as one complete prompt in a new session under the existing
workshop rule.

## Documentation Cleanup

- Rewrite `launchable/ACS_LAUNCHABLE_FIELDS.md` for the current four-prompt
  workshop. Remove the retired source-edit and one-agent-turn description.
- Delete the unused repository file `launchable/acs_task_prompt.txt`.
- Retain the setup cleanup entry for an old deployed `acs_task_prompt.txt`, so
  upgrades remove stale copies.
- Update prompt tests, setup tests, runner tests, and documentation references
  to the current contract.

## Test Strategy

Implementation follows test-driven development. Each new behavior first has a
focused test that fails for the expected reason.

Runner tests cover:

- exact execution metadata for all six stages;
- canonical answer headings, facts, scientific limits, and media lines;
- pending and terminal Prompt 3 replay without invoking the executor;
- identical replay response and unchanged stage, state, ZIP bytes, and mtimes;
- half-created private state, tampered ZIPs, and valid-looking but unbound stage
  artifacts failing before execution; and
- a true first Prompt 3 call executing exactly once.

Setup and prompt tests cover:

- loop detection set and read-back through NemoClaw;
- active `TOOLS.md` guidance and manifest protection;
- deletion of the obsolete repository prompt while retaining remote cleanup;
- human-readable objective-led prompt structure;
- exact command, state-transition, answer, download, and media contracts; and
- byte identity between the local and public workshop pages.

Verification runs one heavy command at a time on the user's Mac. It includes
focused tests, the complete relevant repository suite, Ruff, scoped mypy, Bash
syntax, the setup-script size gate, secret scanning, and `git diff --check`.

## Publication and Deployment

The source implementation is committed locally first. A separate local
bootstrap commit then pins `launchable/acs_console_bootstrap.sh` to the
immutable source implementation commit, not to the later bootstrap commit. The
rendered setup script must pass Bash syntax, contain one full reviewed SHA and
no placeholder, and remain at or below 16,384 UTF-8 bytes.

The reviewed source files are patched into the running instance from that exact
local source commit. Only after live acceptance passes are the source and
bootstrap commits pushed, in order, to `origin/acs-fall-2026-launchable`.

After live acceptance, the byte-identical workshop page is committed and pushed
to `NVIDIA/digital-biology-examples` on `gh-pages`.

Repository publication alone does not modify the saved Brev Launchable
definition. No supported callable Launchable-authoring interface is available
to this task. The exact paste-ready bootstrap is provided for the user to place
in Launchable `env-3Hlp4pHBlTTlfDxfH41KkGhTeCV` without changing its key field,
ports, hardware, or access settings.

## Approved Running-Instance Update

The user approved an in-place update of organization `agents-in-ls`, instance
`acs-fall-2026-gpu-chemistry-agent-lab-4e08c1` / `8id74izoa` on 2026-08-21.

The update contract is:

- keep the Brev instance and NemoClaw sandbox running;
- make no Brev lifecycle, organization, port, or visibility change;
- back up the current runner, `TOOLS.md`, manifest, workshop output, objective
  state, and prior loop-detection value to a task-owned mode-`0700` directory;
- install only the reviewed runner and `TOOLS.md` through atomic replacements,
  rebuild their manifest hashes, and restore read-only modes;
- enable loop detection with read-back verification, accepting one brief
  OpenClaw restart;
- reset only `outputs/workshop`, `context.json`, and `history.json` after the
  backup so the sequence starts clean;
- preserve the user's `agent:main:main` transcript and use new isolated QA
  session IDs;
- never print or transfer inference keys, gateway tokens, or tokenized URLs;
  and
- restore the exact backup and prior loop-detection value if patch validation
  fails.

Run the complete four-prompt sequence in three independent fresh QA sessions.
All three must pass the committed verifier. Also verify the final ZIP and four
PNG files directly. Browser auto-login, native image rendering, and clicking
**Download Results** remain separate human browser gates because this task
cannot attach to the user's authenticated browser session.

## Release Gates

The release is complete only when:

1. all local deterministic checks pass;
2. independent spec and code-quality reviews have no open Critical or Important
   findings;
3. the in-place patch passes rollback-aware validation;
4. three clean live QA sessions pass the committed verifier;
5. no credential or private URL appears in source, logs, receipts, answers, or
   artifacts;
6. the source and bootstrap commits are pushed in the correct pin order; and
7. the public workshop README is byte-identical to the locally accepted page.

## Non-Goals

- arbitrary chemistry commands, datasets, or output paths;
- biological, pharmacological, experimental, or performance conclusions;
- a new agent framework or OpenClaw plugin;
- concurrent first-run locking;
- automated use of an undocumented Brev Console endpoint; and
- claiming browser acceptance from terminal, trajectory, or artifact evidence.

## Approved Post-Rollback Stabilization Amendment

This section is authoritative over conflicting live-operation, QA-count, and
Launchable-handoff text above. The user approved this bounded continuation on
2026-08-21 after the first live patch attempt was rolled back and the trusted
backup was verified.

### One-process transaction lock and stale prepared recovery

The host patch process must acquire one fixed, non-blocking host lock before it
reads, reconciles, creates, or changes any operation journal. It holds that lock
until apply or rollback exits. A second cooperating operation, including one
that uses the same state directory, must fail before backup or workspace
mutation.

Only the current lock holder may recover a stale `prepared` reservation. It may
remove the reservation only when the journal proves that no authoritative
backup was committed and no workspace mutation began. The operating system
releases the lock after a hard kill, so the next exact apply can recover that
pre-mutation reservation. Any evidence of a committed backup, apply start, or
ambiguous state remains fail closed and requires the normal rollback path.

### No-replay Brev transport controller

State-changing patch execution must not expose the patch exit status directly
to `brev exec`, because the observed transport can repeat a command after a
nonzero remote exit. A small reviewed host controller accepts an exact
invocation ID and exact patch arguments, claims a private mode-`0700`
invocation directory atomically, and invokes the patch at most once.

The controller captures the patch's single closed JSON receipt and inner exit
code, validates their types and closure, and atomically writes one terminal
transport receipt. After the invocation is claimed, the controller returns
outer exit `0` for pass and fail outcomes. A duplicate controller invocation
with the same ID never calls the patch; it reports the existing terminal state
or `in_progress` and returns `0`. A separate read-only command retrieves the
terminal receipt. Pre-invocation validation failures are also recorded without
calling the patch.

The controller must not emit secrets, session IDs, tokenized URLs, backup
contents, or unrestricted paths. Its tests must prove one patch call across
duplicate invocations, faithful inner failure recording, closed receipts, and
safe rejection of malformed IDs and directories.

### Final local and live decision

After independent implementation and review, run the existing verifier suite
and live-operation suite once against the final `HEAD`. New non-Critical
findings are residual risks; a Critical finding blocks the canary.

Make one canary attempt on exact instance `8id74izoa` with a new state directory
and invocation ID. Preserve the already trusted backup and create a separate
trusted backup for the canary. Install only the reviewed runner, `TOOLS.md`,
manifest, patch, controller, QA driver, and verifier package. Run exactly one
fresh four-prompt QA trajectory. Verify that trajectory, the ZIP, the four PNG
files, installed hashes, modes, manifest, loop setting, GPU, and untouched main
session. Keep the patch only if all required checks pass; otherwise invoke the
reviewed rollback through the same no-replay controller and verify exact
restoration.

### Publication and future-deployment boundary

Only a validated canary permits source push, public workshop-page push, and
bootstrap repinning. Update saved Launchable
`env-3Hlp4pHBlTTlfDxfH41KkGhTeCV` only through a supported, callable,
authenticated Console, CLI, connector, or API. Do not automate an undocumented
private endpoint or reuse raw browser credentials. If no supported authoring
surface is callable, report that exact remaining manual action.

After the saved definition is confirmed, make at most one fresh deployment,
after read-only confirmation of its exact type and price and duplicate-create
protection. Browser auto-login, native image rendering, and clicked-download
hash verification are distinct required gates on that fresh deployment.

### Residual boundary

The host lock coordinates reviewed patch processes. It does not stop an
actively hostile same-UID process from writing the workspace while a patch is
running. Record that limitation as a residual risk unless the platform exposes
a supported quiesce boundary.

The final review also records these non-Critical residual risks without
expanding the frozen scope:

- the trusted patch has a closed 16 KiB receipt contract, but its temporary
  stdout and stderr files are not write-time bounded before validation;
- a true `SIGKILL` after controller claim can leave a permanent fail-closed
  `in_progress` record that requires operator inspection and a new invocation;
- descendants retain the host lock until the operation process tree exits, so
  a leaked descendant can wedge later operations instead of permitting unsafe
  overlap;
- the controller has no runtime timeout;
- same-UID replacement of reviewed pathnames, inodes, or the lock file remains
  outside the cooperative trust boundary; and
- the controller is staged and hash-verified separately because the existing
  closed patch bundle manifest contains exactly six files.
