# Brev Console fields

- **Name:** `nvMolKit + Nemotron Notebook`
- **Description:** A guided GPU workshop for nvMolKit fingerprints, similarity, clustering, neighborhood analysis, and bounded Nemotron-assisted panel design.
- **Source:** User-approved repository at the accepted release commit; do not assume public publication.
- Runtime: Linux x86-64 with CPython 3.12 in VM mode.
- **Hardware:** One NVIDIA GPU with compute capability 7.0 or newer and a driver compatible with CUDA 12.6 or newer; **75 GiB default disk storage**.
- **Setup script:** `launchable/setup.sh`
- **Brev-managed Jupyter:** Enable Jupyter in the Brev Console. The setup script installs into Brev's CPython 3.12 Jupyter runtime but does not start, stop, or replace the managed service.
- **Secure Link:** Fixed port `8888`; access `Only my organization`. Do not expose unrestricted public TCP.
- **Fixed hosted inference:** Model `nvidia/nvidia/nemotron-3-nano-30b-a3b` at `https://inference-api.nvidia.com/v1`. These values are fixed in `notebooks/workshop_llm_agent.py`.
- **Launch parameters:** Keep exactly one parameter: required Text parameter `NVIDIA_INFERENCE_API_KEY`, with no default. Remove `NVIDIA_API_KEY`, `NEMOTRON_MODEL`, and `JUPYTER_PORT` from Launch parameters.
- **Lifecycle:** Designed for fresh deployment only; not yet live-qualified. GPU execution, hosted inference, rendered visuals, and Secure Link access all require live acceptance. After a VM stop/start, verify the managed Jupyter service and rerun the notebook; do not claim auto-restart without live evidence.

## Notebook order

Start with **Module 1**. It is the recommended entry point.

1. `notebooks/01_direct_nvmolkit_reframe.ipynb`
2. `notebooks/02_agent_assisted_reframe_neighborhoods.ipynb`
3. `notebooks/03_full_agent_reframe_panel_design.ipynb`

**Hosted mode** (`interactive` in `NVMOLKIT_WORKSHOP_MODE`) in Modules 2 and 3 uses the fixed Inference Hub model and the protected `NVIDIA_INFERENCE_API_KEY`. **Reference mode** in Modules 2 and 3 is a deterministic recovery and local-acceptance path. It makes no hosted client call and does not need a key. Module 1 uses [nvMolKit](https://github.com/NVIDIA-BioNeMo/nvMolKit) directly and does not call the hosted model. Keep the Secure Link on port `8888`, **75 GiB** disk storage, and access set to **Only my organization**.

The workshop organizer supplies an approved Inference Hub key beginning with `sk-`; attendees do not create a personal NVIDIA API key. Attendees enter the supplied value once in Setup values unless a separate runtime secret manager is provided.

Author this Launchable in the Brev web Console only. In the Software configuration, paste the current contents of `launchable/setup.sh` into the saved setup-script field. A repository update does not replace that saved script body. The setup script copies `NVIDIA_INFERENCE_API_KEY` from the short-lived setup environment to `${HOME}/.config/nvmolkit/NVIDIA_INFERENCE_API_KEY`, outside the repository, with directory mode `0700` and file mode `0600`. It accepts `NVIDIA_API_KEY` only as a legacy variable name when that value is an `sk-` Inference Hub key. The notebooks load the protected file automatically and do not request the key. The key remains on the VM disk until the environment is deleted or the file is removed. Never put API keys in this file, the repository, notebook outputs, logs, screenshots, or chat.
