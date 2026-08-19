# ACS workshop: two required Launchables

## Goal

Update the ACS Fall 2026 attendee guide so that it accurately describes the
current notebook lessons and requires attendees to use both Brev Launchables.
The required order is the Notebook Launchable first and the Conversational
OpenClaw Launchable second. The integrated companion notebook remains optional.

## Required attendee path

The guide will present one numbered workshop path:

1. Deploy and open **nvMolKit + Nemotron Notebook**.
2. Complete numbered notebook Modules 1–3 in order.
3. Optionally run the integrated companion demo in the same Notebook
   Launchable.
4. Deploy and open the **Conversational OpenClaw Launchable**.
5. Complete the four existing conversational prompts in order.
6. Stop both Brev environments after the workshop.

The page must not call either Launchable optional, and Modules 1–3 are the
required Notebook Lab path. The integrated companion notebook, Module 1's
advanced 10,000-row run, and extra conversational exploration may remain
clearly marked optional activities inside the required environments.

## Page structure and content

### Introduction

The introduction will say that the workshop uses two required, complementary
agentic chemistry environments:

- The Notebook Launchable teaches direct nvMolKit work and bounded Nemotron
  assistance through three guided modules.
- The Conversational Launchable provides a separate sandboxed OpenClaw
  experience with four tested prompts.

The introduction will preserve the existing product boundary: the workshop
demonstrates the BioNeMo Agent Toolkit workflow pattern. It is not an
unrestricted or fully autonomous AI scientist.

### Preparation table

The preparation table will identify both required environments and keep their
configuration separate:

| Required environment | Launchable ID | API-key field | Model use |
| --- | --- | --- | --- |
| nvMolKit + Nemotron Notebook | `env-3HJtJW3qHg4Dw1I3xt75BfpBmZW` | `NVIDIA_API_KEY` | Module 1 uses no LLM. In hosted mode, Modules 2–3 and the companion use `nvidia/nemotron-3-nano-30b-a3b`. |
| Conversational OpenClaw | `env-3Hlp4pHBlTTlfDxfH41KkGhTeCV` | `NVIDIA_INFERENCE_API_KEY` | The four-prompt path uses Nemotron 3 Super 120B-A12B. |

The guide will use the existing deployment URLs for these exact Launchable
IDs. It will replace the dated public-HTTP status paragraph with evergreen
attendee checks. It will not claim that a saved Launchable or prior deployment
proves the health of a new environment.

### Required Lab 1: Notebook Launchable

The guide will tell attendees to enter `NVIDIA_API_KEY`, wait for setup to
finish, open JupyterLab through the existing port 8888 Secure Link, and confirm
that Module 1's initialization output reports the nvMolKit version and one CUDA
device without the CPU-fallback message. These checks apply to the attendee's
new environment and do not inherit success from an older deployment.

The required workshop path uses hosted mode for Modules 2–3. Reference mode is
a bounded recovery path when the instructor directs its use; it makes zero
hosted model calls and must not be presented as evidence that Nemotron ran.

Attendees will then complete these notebooks in order:

1. `notebooks/01_direct_nvmolkit_reframe.ipynb`
   - Run direct nvMolKit fingerprints, Tanimoto similarity, and clustering.
   - Compare bounded parameters and GPU results with clearly labeled RDKit CPU
     reference work.
   - State explicitly that no LLM is used in this module.
2. `notebooks/02_agent_assisted_reframe_neighborhoods.ipynb`
   - Use Nemotron to select two bounded failure policies.
   - Explain that Python renders, validates, binds, and executes the allow-listed
     implementation.
   - Review the 60-row neighborhood atlas and representation-sensitivity
     results.
3. `notebooks/03_full_agent_reframe_panel_design.ipynb`
   - Use Nemotron for a bounded plan and audit.
   - Review the two allow-listed strategies and approve one before execution.
   - Select a validated 24-compound panel from the fixed 96-row ReFRAME
     snapshot, then rerun Steps 5 and 6 for the receipt and gallery.

The guide will list `notebooks/nvmolkit_nemotron_demo.ipynb` as an optional
integrated companion demo. Its short description will cover the six approved
stages, objective challenge, and evidence-backed conclusion without making it
part of required completion.

### Required Lab 2: Conversational OpenClaw

The guide will retain the existing sandbox instructions, four-prompt order,
timeout recovery, artifact download, and cleanup guidance. It will make clear
that this is the second required lab. Attendees may adapt their questions and
interpretations within the configured tool and nvMolKit boundaries, while the
four unchanged prompts remain the tested workshop path.

## Product roles and scientific boundaries

The guide will use one consistent role statement:

> Nemotron plans and selects within validated choices. Python validates and
> executes deterministic chemistry. nvMolKit performs the configured GPU
> molecular operations. RDKit supports input handling, descriptors, CPU
> reference work, and visualization.

Notebook-specific copy will attribute GPU Morgan fingerprints, Tanimoto
similarity, and the configured accelerated operations to nvMolKit. It will not
claim that the model writes or executes arbitrary chemistry code. It will not
add generalized performance or speedup claims. Module 1 may report timing and
throughput observed during the attendee's exact run when the page labels the
hardware, inputs, parameters, and measurement as run-specific. It must not
turn those observations into a general acceleration conclusion.

The scientific-limit section will cover both datasets and workflows. Results
describe structural similarity, diversity, clustering, and sampled
force-field geometries. They do not prove binding, activity, ADMET, efficacy,
safety, synthesizability, clinical value, or experimental structure.

## Protected content and scope

Implementation is limited to:

- `docs/acs-fall-2026-workshop.md`
- `tests/test_acs_fall_2026_workshop_page.py`

The four marked conversational prompt blocks must remain byte-identical,
including their commands, wording, whitespace, markers, fences, and media
lines. The existing prompt SHA-256 locks remain authoritative.

This change will not edit notebooks, setup scripts, Launchable definitions,
runtime code, chemistry code, models, API-key handling, or Brev environments.
It will not merge the notebook and ACS branches.

## Verification and acceptance

Focused tests will require:

- both Launchables to be labeled required;
- the Notebook Launchable to appear before the Conversational Launchable;
- both exact Launchable IDs, URLs, API-key fields, and distinct model roles;
- all four notebook paths and accurate Module 1–3 summaries;
- both Launchables and Modules 1–3 to remain required while the companion,
  advanced 10,000-row run, and extra exploration remain optional;
- hosted mode to be the required Modules 2–3 path and reference mode to be
  labeled as a zero-hosted-call recovery path;
- the Notebook Launchable key, setup, port 8888, nvMolKit-version, one-CUDA-
  device, and no-CPU-fallback readiness checks;
- the Module 3 review, approval, and Steps 5–6 instructions;
- notebook-specific ReFRAME, 96-row, and 24-compound scientific context;
- the established product-role and scientific-limit boundaries;
- run-specific timing language without generalized performance or speedup
  claims;
- all four conversational prompt hashes to remain unchanged; and
- removal of the old optional-notebook and singular-required-lab wording.

Verification will run the focused attendee-page tests, Ruff check and format
check on the changed Python test, and `git diff --check`. The final diff must
contain only the two approved implementation files after this specification
and its implementation plan are excluded from the implementation-range check.

The page is accepted when an attendee can identify both required deployments,
their order, their separate credentials and model roles, the exact notebook
work, the optional companion, the unchanged conversational prompts, and the
cleanup action without consulting another document.
