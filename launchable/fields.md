# Brev Console fields

- **Name:** `nvMolKit + Nemotron Notebook`
- **Description:** A guided GPU notebook for nvMolKit fingerprints, similarity, clustering, conformers, and MMFF94 optimization with bounded hosted Nemotron planning.
- **Source:** User-approved repository at the accepted release commit; do not assume public publication.
- **Runtime:** VM mode.
- **Hardware:** One NVIDIA GPU with compute capability 7.0 or newer and a driver compatible with CUDA 12.6 or newer; 50 GiB disk.
- **Setup script:** `launchable/setup.sh`
- **Secure Link:** Port `8888`; access `Only my organization`. Do not expose unrestricted public TCP.

## Launch parameters

| Name | Requirement | Default |
| --- | --- | --- |
| `NVIDIA_API_KEY` | Required; masked | Empty |
| `NEMOTRON_MODEL` | Optional | `nvidia/nemotron-3-nano-30b-a3b` |
| `JUPYTER_PORT` | Optional | `8888` |

Author this Launchable in the Brev web Console only. Never put API keys in this file, the repository, notebook outputs, logs, screenshots, or chat; enter `NVIDIA_API_KEY` only as the masked Console parameter.
