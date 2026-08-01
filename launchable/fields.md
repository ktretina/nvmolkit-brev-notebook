# Brev Console fields

- **Name:** `nvMolKit + Nemotron Notebook`
- **Description:** A guided GPU notebook for nvMolKit fingerprints, similarity, clustering, conformers, and MMFF94 optimization with bounded hosted Nemotron planning.
- **Source:** User-approved repository at the accepted release commit; do not assume public publication.
- Runtime: Linux x86-64 with CPython 3.12 in VM mode.
- **Hardware:** One NVIDIA GPU with compute capability 7.0 or newer and a driver compatible with CUDA 12.6 or newer; 50 GiB disk.
- **Setup script:** `launchable/setup.sh`
- **Brev-managed Jupyter:** Enable Jupyter in the Brev Console. The setup script installs into Brev's CPython 3.12 Jupyter runtime but does not start, stop, or replace the managed service.
- **Secure Link:** Port `8888`; access `Only my organization`. Do not expose unrestricted public TCP.
- **Lifecycle:** Designed for fresh deployment only; not yet live-qualified. GPU execution, hosted inference, rendered visuals, and Secure Link access all require live acceptance. After a VM stop/start, verify the managed Jupyter service and rerun the notebook; do not claim auto-restart without live evidence.

Author this Launchable in the Brev web Console only. Do not rely on setup-variable persistence for `NVIDIA_API_KEY`: the notebook requests it with a hidden prompt when it is absent from the kernel environment. Never put API keys in this file, the repository, notebook outputs, logs, screenshots, or chat.
