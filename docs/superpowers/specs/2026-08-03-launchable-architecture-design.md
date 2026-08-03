# Launchable Architecture Diagram Design

## Purpose

Create one presentation-ready diagram that explains how the Brev launchable turns a user goal into a bounded, agent-guided nvMolKit workflow.

## Design

Use a 16:9 landscape composition with four primary elements: the Brev-hosted Jupyter/Python controller, hosted Nemotron inference, nvMolKit chemistry on the NVIDIA L4 GPU, and rendered notebook outputs. Show RDKit as a supporting local component. Directional arrows must distinguish model-selected structured tool calls from Python-executed GPU operations and show structured results returning to the persistent Nemotron conversation.

The diagram must name the five nvMolKit operations in execution order and state the autonomy boundary: Nemotron plans and selects bounded parameters; Python validates and executes; no arbitrary code execution is delegated to the model.

## Deliverables

- `docs/nvmolkit-launchable-architecture.svg`: editable source.
- `docs/nvmolkit-launchable-architecture.png`: presentation-ready export.

The notebook and application code remain unchanged.
