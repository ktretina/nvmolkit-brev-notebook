# Launchable Architecture Diagram Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create an accurate, presentation-ready architecture diagram for the nvMolKit + Nemotron Brev launchable.

**Architecture:** Use one self-contained SVG as the editable source and render it to PNG. The diagram separates hosted inference from the Brev runtime and shows validated calls entering the ordered nvMolKit GPU tool chain.

**Tech Stack:** SVG, librsvg, XML validation, visual inspection.

---

### Task 1: Create and verify the diagram

**Files:**
- Create: `docs/nvmolkit-launchable-architecture.svg`
- Create: `docs/nvmolkit-launchable-architecture.png`

- [ ] **Step 1: Create the SVG**

Draw the approved four-part architecture, ordered nvMolKit tool chain, RDKit support role, result feedback, and bounded-autonomy note.

- [ ] **Step 2: Validate and render**

Run: `xmllint --noout docs/nvmolkit-launchable-architecture.svg`

Expected: exit code 0.

Run: `rsvg-convert -w 1920 -h 1080 docs/nvmolkit-launchable-architecture.svg -o docs/nvmolkit-launchable-architecture.png`

Expected: a 1920 x 1080 PNG.

- [ ] **Step 3: Inspect the export**

Open the PNG at original detail and correct any clipped, overlapping, or unreadable text.

- [ ] **Step 4: Verify scope**

Run: `git diff --check` and `git status --short`.

Expected: only the design, plan, SVG, and PNG are new.
