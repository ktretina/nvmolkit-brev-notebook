# Brev Console fields

- **Name:** `nvMolKit + Nemotron Notebook`
- **Description:** A guided GPU workshop for nvMolKit fingerprints, similarity, clustering, neighborhood analysis, and bounded Nemotron-assisted panel design.
- **Source:** User-approved repository at the accepted release commit; do not assume public publication.
- Runtime: Linux x86-64 with CPython 3.12 in VM mode.
- **Hardware:** One NVIDIA GPU with compute capability 7.0 or newer and a driver compatible with CUDA 12.6 or newer; **75 GiB default disk storage**.
- **Setup script source:** `launchable/setup.sh` is a redacted operator template. Do not paste it as-is; use the renderer workflow below.
- **Brev-managed Jupyter:** Enable Jupyter in the Brev Console. The setup script installs into Brev's CPython 3.12 Jupyter runtime but does not start, stop, or replace the managed service.
- **Secure Link:** Fixed port `8888`; access `Only my organization`. Do not expose unrestricted public TCP.
- **Fixed hosted inference:** Model `nvidia/nvidia/nemotron-3-nano-30b-a3b` at `https://inference-api.nvidia.com/v1`. These values are fixed in `notebooks/workshop_llm_agent.py`.
- **Launch parameters:** **No Launch parameters or Setup values.** Remove `NVIDIA_INFERENCE_API_KEY`, `NVIDIA_API_KEY`, `NEMOTRON_MODEL`, and `JUPYTER_PORT`.
- **Lifecycle:** Designed for fresh deployment only; not yet live-qualified. GPU execution, hosted inference, rendered visuals, and Secure Link access all require live acceptance. After a VM stop/start, verify the managed Jupyter service and rerun the notebook; do not claim auto-restart without live evidence.

## Notebook order

Start with **Module 1**. It is the recommended entry point.

1. `notebooks/01_direct_nvmolkit_reframe.ipynb`
2. `notebooks/02_agent_assisted_reframe_neighborhoods.ipynb`
3. `notebooks/03_full_agent_reframe_panel_design.ipynb`

**Hosted mode** (`interactive` in `NVMOLKIT_WORKSHOP_MODE`) in Modules 2 and 3 uses the fixed Inference Hub model and the protected `NVIDIA_INFERENCE_API_KEY`. **Reference mode** in Modules 2 and 3 is a deterministic recovery and local-acceptance path. It makes no hosted client call and does not need a key. Module 1 uses [nvMolKit](https://github.com/NVIDIA-BioNeMo/nvMolKit) directly and does not call the hosted model. Keep the Secure Link on port `8888`, **75 GiB** disk storage, and access set to **Only my organization**.

The organizer preprovisions an approved workshop-only Inference Hub key. Attendees enter no API key and do not need an NVIDIA API account or key.

Author this Launchable in the Brev web Console only. From the repository, run `python3 launchable/render_setup.py /private/tmp/nvmolkit-workshop-setup.sh`. Enter the workshop-only key at the hidden prompt. The trusted offline renderer writes the private rendered file outside the repository with mode `0600` and refuses to overwrite an existing file or use an unsafe output path.

Paste only the rendered file's contents into the Brev Console saved setup body. Never commit, upload, attach, or share the rendered file. Delete the private rendered file after saving the Console body. A repository push does not replace the setup body already saved in the Console.

The saved setup body stores the key at `${HOME}/.config/nvmolkit/NVIDIA_INFERENCE_API_KEY`, outside the repository, with directory mode `0700` and file mode `0600`. The notebooks load it automatically and do not request it. A person who controls a deployed VM can recover this shared key. Use a workshop-only key, monitor it during the event, and rotate or revoke it afterward. Deleting a VM removes that VM's file but does not replace key revocation.
