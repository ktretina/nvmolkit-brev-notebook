# ACS Fall 2026 workshop messaging update

## Goal

Reframe the attendee guide as a concise demonstration of agentic AI for
chemistry using the workflow pattern supported by NVIDIA BioNeMo Agent
Toolkit. Explain the distinct roles of Nemotron, the sandboxed OpenClaw
workspace, nvMolKit, RDKit, and validated Python without overstating autonomy,
scientific meaning, or performance.

## Copy changes

1. Replace the introduction with two short paragraphs that:
   - name agentic AI and link BioNeMo Agent Toolkit;
   - explain that Nemotron reasons and selects bounded actions;
   - explain that the OpenClaw and OpenShell environment limits execution;
   - link nvMolKit and state that it performs the configured GPU chemistry;
   - retain the no-local-setup message.
2. Rewrite the required OpenClaw Launchable description to present it as a
   sandboxed conversational space. The four prompts are tested starting points.
   Attendees may change their questions and requested interpretations, while
   executable work remains limited to the approved tools and configured
   nvMolKit capabilities.
3. Clarify that running all four prompts unchanged and in order is the tested
   workshop path.
4. Link nvMolKit in explanatory prose and add it to the resource list. Add the
   BioNeMo Agent Toolkit repository to that list.

## Accuracy boundaries

- Say that the workshop demonstrates the BioNeMo Agent Toolkit workflow
  pattern. Do not claim that it is an unrestricted or fully autonomous AI
  scientist.
- Attribute GPU Morgan fingerprints, Tanimoto similarity, conformer embedding,
  and MMFF94 optimization to nvMolKit. Attribute input handling, display, and
  the configured CPU Butina clustering step to RDKit and validated Python.
- Do not add performance or speedup claims.
- Do not imply biological, experimental, or clinical validation.

## Protected content

Keep all four marked prompt blocks byte-identical, including markers, fences,
commands, wording, whitespace, and media lines. Do not change setup, runtime,
Launchable, or chemistry code.

## Verification

- Add focused tests for the agentic AI, BioNeMo Agent Toolkit, sandboxed
  exploration, nvMolKit role, and official resource links.
- Preserve the existing prompt-parser and safety tests.
- Run the focused attendee-page test file, Ruff on the changed test, and
  `git diff --check`.
