# ACS Launchable Console authoring sheet

**Status:** Instructions only. This does not mean that a Launchable was created or deployed.

## Create a new definition

- Brev cannot capture or convert the current Phase Zero VM into a Launchable.
- In the signed-in Brev Console, open **Launchables**, select **Create Launchable**, and enter the fields below.

The fields below apply only to a new definition.

## Details

- **Name:** `ACS Fall 2026 GPU Chemistry Agent Lab`
- **Description:** `Use a hosted Nemotron agent for four fixed, bounded chemistry prompts: molecular validation and fingerprints, similarity and clustering, sampled 3D geometry, and a state-bound diversity objective. View the scientific images and download the validated result bundle. Computational descriptors only; not evidence of biological activity.`

## Compute

- **Runtime mode:** VM
- **Cloud:** AWS
- **Instance type:** `g6.xlarge`
- **GPU:** One NVIDIA L4
- **System memory:** 16 GiB
- **Disk storage:** 128 GiB

## Setup script

- **Source:** Select **I don’t have any code files**. The setup bootstrap creates the authoritative checkout.
- **Public repository:** `https://github.com/ktretina/nvmolkit-brev-notebook.git`
- **Pinned commit:** `<REVIEWED_PUBLIC_COMMIT_SHA>`
- **Bootstrap template:** `launchable/acs_console_bootstrap.sh.in`
- Replace `@REVIEWED_PUBLIC_COMMIT_SHA@` in the template with the exact 40-character public source commit SHA.
- Generate `launchable/acs_console_bootstrap.sh` from the template. The generated file must be no larger than 16,384 bytes.
- Paste the bootstrap into the Console setup-script field.
- Do not paste `launchable/acs_nemoclaw_launchable_setup.sh` directly. Brev does not guarantee that a pasted Console script is stored beside an attached source checkout.

The bootstrap keeps the launch parameter out of clone and checkout child processes. It checks out the exact public commit in a private directory and then executes the unified setup from that checkout.

## Launch parameter

- **Name:** `NVIDIA_INFERENCE_API_KEY`
- **Type:** Text
- **Required:** Yes
- **Default:** None

Never put the API key in source, the setup script, a Launchable default, chat, screenshots, logs, or workshop files. Enter it only in the private required Console parameter at deployment time.

## Network

Add only these Secure Links:

| Secure Link name | Port |
| --- | ---: |
| `Open Chemistry Agent` | `18788` |
| `Download Results` | `8765` |

- Make `Open Chemistry Agent` the deployment-page call to action.
- Do not add a Secure Link for port `18789`.
- Do not expose raw TCP or UDP ports.

## Access

- **Acceptance:** `Only my organization`
- **ACS handoff, after acceptance:** `Anyone with the link`
- Do not publish it to the community catalog.

For a new definition only, select **Create Launchable** in the Console after review. Creating the definition does not deploy an instance; deployment and live acceptance are separate steps.

The setup installs a bounded workshop with four fixed prompts. The runner performs deterministic chemistry, returns canonical `answer_markdown`, protects its manifest and objective state, and publishes the fixed images and ZIP. The model may call only the documented runner commands and may choose only a displayed maximum-score objective action.

## Update the existing saved definition

Use this section for the existing saved Launchable `env-3Hlp4pHBlTTlfDxfH41KkGhTeCV`. Do not select **Create Launchable** for this update.

1. Validate the updated source in place on the task-owned existing instance before publication. This in-place pass validates the source and runtime, not the saved Launchable bootstrap. A source push alone does not update the saved Launchable definition.
2. After that in-place pass succeeds, publish the reviewed source commit. Generate `launchable/acs_console_bootstrap.sh` with that public source commit pinned.
3. Commit and push the generated bootstrap second.
4. Keep access set to **Only my organization** during this update. Paste the exact generated `launchable/acs_console_bootstrap.sh` into the existing saved Launchable `env-3Hlp4pHBlTTlfDxfH41KkGhTeCV`.
5. Create a future fresh deployment from the saved definition. Verify automatic OpenClaw sign-in, completion of all four prompts, display of all four images, and the **Download Results** service.
6. Only after this fresh-deployment pass may access change to **Anyone with the link**.

Preserve the existing API-key field, one NVIDIA L4, Secure Link ports `18788` and `8765`, and all other access settings during the update.
