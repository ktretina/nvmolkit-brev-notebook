# ACS Launchable Console authoring sheet

**Status:** Instructions only. This does not mean that a Launchable was created or deployed.

## Supported authoring path

- Brev cannot capture or convert the current Phase Zero VM into a Launchable.
- In the signed-in Brev Console, open **Launchables**, select **Create Launchable**, and enter the fields below.

## Details

- **Name:** `ACS Fall 2026 GPU Chemistry Agent Lab`
- **Description:** `Use a hosted Nemotron agent to edit a bounded chemistry task, run nvMolKit on an NVIDIA L4, view a similarity heatmap, and download the result files. Computational descriptors only; not evidence of biological activity.`

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
- Replace `@REVIEWED_PUBLIC_COMMIT_SHA@` in the template with the reviewed 40-character public commit SHA. Paste the bootstrap into the Console setup-script field.
- Do not paste `launchable/acs_nemoclaw_launchable_setup.sh` directly. Brev does not guarantee that a pasted Console script is stored beside an attached source checkout.

The bootstrap keeps the launch parameter out of clone and checkout child processes. It checks out the exact public commit in a private directory and then executes the unified setup from that checkout.

## Launch parameter

- **Name:** `NVIDIA_INFERENCE_API_KEY`
- **Type:** Text
- **Required:** Yes
- **Default:** None

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

After review, select **Create Launchable** in the Console. Creating the definition does not deploy an instance; deployment and live acceptance are separate steps.

The setup runs one time-bounded agent turn and validates the exact final state of the protected inputs, edited source, and artifacts. This does not prove the historical number of edits or commands inside the agent turn.
