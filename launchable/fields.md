# Brev Console fields

- **Name:** `nvMolKit + Nemotron Notebook`
- **Description:** A guided GPU notebook for nvMolKit fingerprints, similarity, clustering, conformers, and MMFF94 optimization with bounded hosted Nemotron planning.
- **Source:** User-approved repository at the accepted release commit; do not assume public publication.
- Runtime: Linux x86-64 with CPython 3.12 in VM mode.
- **Hardware:** One NVIDIA GPU with compute capability 7.0 or newer and a driver compatible with CUDA 12.6 or newer; **75 GiB default disk storage**.
- **Setup script:** `launchable/setup.sh`
- **Brev-managed Jupyter:** Enable Jupyter in the Brev Console. The setup script installs into Brev's CPython 3.12 Jupyter runtime but does not start, stop, or replace the managed service.
- **Secure Link:** Fixed port `8888`; access `Only my organization`. Do not expose unrestricted public TCP.
- **Fixed Nemotron model:** `nvidia/nemotron-3-nano-30b-a3b`. This value is hardcoded in `demo_agent.py`.
- **Launch parameters:** Keep exactly one parameter: required Text parameter `NVIDIA_API_KEY`, with no default. Remove both `NEMOTRON_MODEL` and `JUPYTER_PORT` from Launch parameters.
- **Lifecycle:** Designed for fresh deployment only; not yet live-qualified. GPU execution, hosted inference, rendered visuals, and Secure Link access all require live acceptance. After a VM stop/start, verify the managed Jupyter service and rerun the notebook; do not claim auto-restart without live evidence.

## Notebook order

Start with **Module 1**. It is the recommended entry point.

1. `notebooks/01_direct_nvmolkit_reframe.ipynb`
2. `notebooks/02_agent_assisted_reframe_neighborhoods.ipynb`
3. `notebooks/03_full_agent_reframe_panel_design.ipynb`
4. `notebooks/nvmolkit_nemotron_demo.ipynb`

**Hosted mode** (`interactive` in `NVMOLKIT_WORKSHOP_MODE`) in Modules 2 and 3 uses the fixed `nvidia/nemotron-3-nano-30b-a3b` model and the protected `NVIDIA_API_KEY`. The companion demo uses the same hosted model and key. **Reference mode** in Modules 2 and 3 is a deterministic recovery and local-acceptance path. It makes no hosted client call and does not need an API key. Module 1 uses the library directly and does not call the hosted model. Keep the Secure Link on port `8888`, **75 GiB** disk storage, and access set to **Only my organization**.

Author this Launchable in the Brev web Console only. In the Software configuration, paste the current contents of `launchable/setup.sh` into the saved setup-script field. A repository update does not replace that saved script body. The setup script copies `NVIDIA_API_KEY` from the short-lived setup environment to `${HOME}/.config/nvmolkit/NVIDIA_API_KEY`, outside the repository, with directory mode `0700` and file mode `0600`. The notebook loads that file automatically and does not request the key. The key remains on the VM disk until the environment is deleted or the file is removed. Never put API keys in this file, the repository, notebook outputs, logs, screenshots, or chat.
