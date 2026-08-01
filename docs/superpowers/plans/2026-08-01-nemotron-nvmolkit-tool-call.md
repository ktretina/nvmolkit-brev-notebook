# Nemotron nvMolKit Tool Call Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the notebook's bounded Nemotron parameter request into one visible OpenAI-compatible function call that invokes the existing nvMolKit chemistry workflow and returns its compact result summary to Nemotron.

**Architecture:** Keep the notebook as the tool runtime: Nemotron emits one forced `analyze_molecule_library` call, Pydantic validates its five arguments, and a local function executes the existing deterministic nvMolKit workflow. The notebook sends only the compact summary back as a `role="tool"` message for a final conservative interpretation. No arbitrary code, autonomous loop, MCP server, Agent Toolkit runtime, LangChain, LangGraph, or new dependency is added.

**Tech Stack:** Python 3.12, OpenAI-compatible NVIDIA hosted API, Pydantic, Jupyter, nvMolKit, RDKit, PyTorch/CUDA, pytest.

---

### Task 1: One bounded nvMolKit scientific tool

**Files:**
- Modify: `demo_agent.py`
- Modify: `notebooks/nvmolkit_nemotron_demo.ipynb`
- Modify: `tests/test_demo_agent.py`
- Modify: `tests/test_notebook.py`
- Modify: `README.md`

- [x] **Step 1: Write failing tool-contract tests**

Add tests that require `request_tool_call()` to send exactly one OpenAI function named `analyze_molecule_library`, reuse the strict five-field `WorkflowPlan` schema, force that function through `tool_choice`, parse `message.tool_calls[0].function.arguments`, and preserve secret-safe 401/403 handling. Add tests for missing/malformed/wrong-name tool calls and invalid arguments; these must produce a labeled validated default decision for non-auth failures and must never execute scientific code.

- [x] **Step 2: Run focused tests and record RED**

Run:

```bash
/private/tmp/nvmolkit-review-venv/bin/python -m pytest tests/test_demo_agent.py -q
```

Expected: failures because the tool definition and tool-call parsing do not exist.

- [x] **Step 3: Implement the minimal tool-call contract**

In `demo_agent.py`, replace the free-form planning request with a forced OpenAI-compatible function call. Keep `WorkflowPlan`, `PlanDecision`, default fallback, hosted key validation, and authentication guidance. Store only the validated tool name, tool-call ID, and raw arguments needed for the follow-up; never store or display the key. Construct the final explanation request with the assistant tool-call message followed by a `role="tool"` message containing only `json.dumps(summary)`.

- [x] **Step 4: Run focused tests and record GREEN**

Run the focused agent tests and confirm all pass.

- [x] **Step 5: Write failing notebook-contract tests**

Require the notebook to display Nemotron's requested tool name and validated arguments, define exactly one local `analyze_molecule_library(mols, plan)` executor, invoke it only after validation, and use its compact summary for the tool-result round trip. Require the existing nvMolKit fingerprint, similarity, clustering, conformer embedding, and MMFF94 calls to remain inside that executor; keep outputs and execution counts clear in the committed notebook.

- [x] **Step 6: Run notebook tests and record RED**

Run:

```bash
/private/tmp/nvmolkit-review-venv/bin/python -m pytest tests/test_notebook.py -q
```

Expected: failures because the notebook still uses the old planning flow and top-level executor cells.

- [x] **Step 7: Refactor only the existing workflow behind the tool boundary**

Move the existing deterministic chemistry operations into `analyze_molecule_library(mols, plan)`. It may render the same heatmap and conformers, but it must return a JSON-safe summary for Nemotron. Replace the planning cell with `request_tool_call`, visibly display the requested tool and validated arguments, invoke the single allow-listed executor, and send its summary through the tool-result explanation call. Preserve the fixed dataset, CUDA preflight, invalid-SMILES handling, scientific boundaries, 401/403 behavior, deterministic non-auth fallback, and existing visual outputs.

- [x] **Step 8: Update concise documentation**

Update the notebook introduction and README to say the Agent Toolkit skill informs the tool contract while the notebook executes the function. Do not claim that the model itself executes Python or that the skill is dynamically loaded.

- [x] **Step 9: Verify and commit**

Run:

```bash
/private/tmp/nvmolkit-review-venv/bin/python -m pytest -q
git diff --check
```

Confirm the notebook is valid nbformat 4 with empty code-cell outputs and null execution counts. Commit only the five implementation files and this plan.
