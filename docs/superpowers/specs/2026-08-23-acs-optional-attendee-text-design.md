# ACS Optional Attendee Instructions and Prompt Design

**Date:** 2026-08-23
**Status:** Approved design; awaiting written-spec review

## Goal

Make the ACS Fall 2026 workshop text clear for a conference attendee with
limited command-line and AI-agent experience. Describe both the notebook and
OpenClaw journeys. Participation in either journey is optional, and neither is
a prerequisite for the other.

This design changes attendee-facing text and the prompt-verification bindings
that must follow that text. It does not change the OpenClaw interface.

## Selected Approach

Use short scientific prompts backed by the existing protected execution
contract.

Two smaller alternatives were rejected:

- Light editing would leave the prompts long and operational.
- One-line prompts would remove useful scientific intent and weaken reliable
  routing to the intended workshop step.

The selected approach keeps exact commands, private paths, state transitions,
one-run limits, canonical-answer rules, and media directives in the protected
workspace guidance. Attendees see scientific objectives and simple recovery
instructions only.

## Public Workshop Structure

The public guide has this order:

1. workshop overview and scientific limits;
2. preparation shared by both journeys;
3. an optional notebook journey;
4. an optional OpenClaw journey;
5. the four OpenClaw prompts;
6. download and shutdown instructions; and
7. official links.

The guide does not use `required`, `must complete`, or equivalent language to
describe attendance or participation. It explains that attendees may do either
journey, both journeys, or neither.

The notebook and OpenClaw environments remain distinct:

- The notebook guide explains its current Launchable setup, JupyterLab access,
  three numbered modules, and optional companion.
- The OpenClaw guide states that its deployment form has no Setup values or API
  key field, access is automatic, and the four prompts are used in one chat in
  order when an attendee chooses that tested path.

This text-only design does not change the notebook Launchable's current
credential behavior. The zero-input credential requirement in this design
applies to the OpenClaw Launchable.

## Notebook Journey Text

The notebook section keeps enough information to complete the current
`nvmolkit_nemotron_demo.ipynb` notebook without command-line work:

1. Open the notebook Launchable.
2. Enter its NVIDIA inference API key only in the named Launchable field when
   that field is present.
3. Deploy and wait for setup to finish.
4. Open JupyterLab through the port 8888 Secure Link.
5. Open `nvmolkit_nemotron_demo.ipynb` and run its cells from top to bottom.
6. In **Agent run**, select **Start Agent**, then select **Approve & Run** for each displayed stage.
7. After the MMFF94 stage, select **Run Objective Challenge**.
8. Download wanted files, then stop or delete the environment according to the
   Brev cost guidance.

The section keeps the notebook's current scientific purpose and removes stale
module, companion, and participation-mandate text.

## OpenClaw Journey Text

The OpenClaw section gives this attendee flow:

1. Open the OpenClaw Launchable.
2. Leave Setup empty and deploy with the default hardware.
3. Wait for setup to finish.
4. Open **Open Chemistry Agent**. It should open directly without requesting an
   API key, gateway token, or password.
5. Create one new chat.
6. Paste the four prompts one at a time and in order when following the tested
   journey.
7. Wait for the answer and image before sending the next prompt.
8. After Prompt 4, open **Download Results** and download `results.zip`.

The guide names the expected image after each prompt without exposing an agent
filesystem path.

## Four Attendee Prompts

### Prompt 1 — Library and representation

> Use the preinstalled ACS workshop workflow to inspect the fixed molecule
> library and explain how the molecules are represented for comparison. Run
> this workshop step once. Clearly distinguish RDKit work on the CPU from
> nvMolKit fingerprint generation on the GPU. Explain the measured result and
> its scientific limits, then show the result image and provide the workshop
> download. If this step has already completed in this chat, show the existing
> result without running it again.

### Prompt 2 — Relationships and groups

> Use the preinstalled ACS workshop workflow to find structurally similar
> molecules and group them using the fixed Butina rule. Run this workshop step
> once. Clearly distinguish GPU fingerprint and similarity calculations from
> CPU clustering. Explain what the groups mean, what they do not prove, and how
> the fingerprint and distance cutoff affect the result. Show the result image
> and provide the updated workshop download. If this step has already
> completed, show the existing result without running it again.

### Prompt 3 — Sampled 3D geometry

> Use the preinstalled ACS workshop workflow to generate and optimize the
> bounded set of sampled 3D conformers. Run this workshop step once. Explain
> which work ran on the GPU, what convergence and MMFF94 energy mean, and why
> these sampled conformers are not experimental structures. Show the result
> image and provide the updated workshop download. If this step has already
> completed, show the existing result without running it again.

### Prompt 4 — Bounded panel objective

> Use the preinstalled bounded workshop objective to improve the weakest-link
> structural diversity of the fixed four-molecule panel. Choose only permitted
> changes with the best predicted minimum pairwise Tanimoto distance, and stop
> when the objective finishes. Explain the starting panel, limiting pair,
> accepted changes, final result, and change in `D_min`. Make clear that this is
> a bounded descriptor exercise, not autonomous molecular design or evidence
> of biological performance. Show the final image and provide the complete
> workshop download. If the objective has already completed, show the existing
> result without running it again.

## Recovery Text

The attendee guide uses one recovery rule:

> Send each prompt only once. If OpenClaw is still working, wait. If you
> accidentally send the same prompt twice, do not try to repair anything; the
> workshop should return the existing completed result. If a request reports an
> error, take a screenshot and continue with another optional activity.

No terminal, shell, file-edit, state-ID, swap-ID, or administrator procedure is
shown to the attendee.

## Expected Results

The guide gives one simple visual check per prompt:

- Prompt 1: molecule-library image;
- Prompt 2: cluster-size image;
- Prompt 3: optimized-structure image;
- Prompt 4: final panel image; and
- every prompt: a CPU/GPU explanation and **Download Results**.

After Prompt 4, `results.zip` is the complete attendee download.

## Text Removed From Attendee Materials

The four prompt blocks and their surrounding attendee instructions do not show:

- `/sandbox/...` or `/tmp/...` paths;
- shell commands or `PYTHONPATH`;
- `MEDIA:` directives;
- `state_id`, `swap_id`, or command templates;
- top-level status handling, decoded `answer_markdown`, or one-call budget
  terminology;
- tool-discovery instructions;
- credential-file, endpoint, setup-script, repair, terminal, or administrator
  details; or
- instructions to modify code, files, or the environment.

## Reliability Boundary

The existing runner and chemistry workflow remain authoritative. The protected
`TOOLS.md` continues to hold the exact execution contract, scientific limits,
canonical-answer rule, image directives, and download location.

Prompt text changes require synchronized updates to:

- the four prompt hashes in the trajectory verifier and QA driver;
- prompt-loading rules that currently depend on the final `MEDIA:` line;
- focused prompt-page, verifier, live-operation, and fixture expectations; and
- the public `NVIDIA/digital-biology-examples` workshop README.

The local canonical prompt blocks and public prompt blocks remain byte-identical.
No runner, chemistry, artifact, secret, deployment, or UI redesign is part of
this text change.

## Verification

Focused tests prove:

- both optional journeys remain present and understandable;
- no participation language labels either journey as required;
- no private path, shell command, media directive, or internal state token
  appears in an attendee prompt;
- the four prompt hashes and order match the verifier and QA driver;
- the hidden execution commands and returned canonical answers remain enforced;
- duplicate-step guidance remains safe;
- the local and public prompt blocks are byte-identical; and
- the expected images and `results.zip` remain named in attendee instructions.

Run the existing verifier and relevant documentation/live-operation suites once
after implementation. A fresh four-prompt browser journey is still needed to
prove the revised text. It cannot prove that the unchanged OpenClaw operator UI
meets attendee-interface acceptance.

## Out of Scope

- changing the OpenClaw Control UI;
- hiding operator controls or tool cards;
- changing credential provisioning;
- changing notebook or chemistry execution;
- changing the workshop runner or artifact formats;
- adding a plugin or custom attendee application; and
- publishing or changing a saved Launchable before the revised text is tested.
