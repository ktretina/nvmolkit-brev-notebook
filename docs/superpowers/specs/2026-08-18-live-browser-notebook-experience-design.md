# Live-Browser Notebook Experience Design

## Status

Approved on 2026-08-18. The approval is conditional on the target workshop
attendee accepting the design. That condition is met.

## Goal

Improve only the live JupyterLab experience for Modules 2 and 3 in the
`nvMolKit + Nemotron Notebook` Launchable
`env-3HJtJW3qHg4Dw1I3xt75BfpBmZW`.

The target attendee is a chemistry scientist in a 50–65 minute workshop. The
attendee needs a clear scientific result, a truthful division of work between
Nemotron and Python, visible approval control, and a low-friction browser flow.
The attendee does not need a new algorithm, a new model, more controls, or a
publication-quality PDF export.

## Evidence and decision

The two supplied JupyterLab PDFs are run evidence, not instructions. They show
that both hosted lessons execute, but they also expose a few live-browser
problems:

- Module 2 overstates its normal-path checks, blurs who selects the policies,
  and hides the Tanimoto column in a wide result table.
- Module 3 does not leave one clear, durable completion receipt after the
  approval callback. It can label the workflow complete when artifact
  validation passed but the optional hosted audit did not. The chemistry grid
  is also wider than a typical notebook viewport, and the rerun step after
  approval is not explicit.

The chosen approach is a targeted presentation and evidence repair. A visual-
only cleanup would leave misleading states. A linear workflow rewrite would
remove useful human approval and add too much workshop complexity.

## Scope boundary

Allowed production changes:

- `notebooks/02_agent_assisted_reframe_neighborhoods.ipynb`
- `notebooks/03_full_agent_reframe_panel_design.ipynb`
- `notebooks/module3_interactive_workflow.py` only where the visible workflow
  status or callback contract needs a small repair

Allowed test changes are limited to the focused workshop notebook, agent, and
clean-kernel execution tests needed to prove these behaviors.

Do not change:

- the fixed model, endpoint, prompt schemas, or hosted request count;
- nvMolKit or RDKit algorithms, the 96-row snapshot, metrics, or acceptance
  rules;
- setup, key handling, hardware, Jupyter access, or the Launchable definition;
- Module 1 or the companion demo;
- any other Brev Launchable; or
- PDF layout for its own sake.

## Module 2 design

### Human and agent roles

Hosted mode asks Nemotron to choose the two bounded policy values for this run
and give concise reasons. Reference mode uses fixed local reference policy
values with no hosted selection. In both modes, Python immediately renders,
validates, and applies the matching allow-listed implementation. The attendee
evaluates the choices afterward and decides whether both policies are
scientifically appropriate.

Call the visible result a policy receipt. Do not label the receipt itself as
hosted because reference mode also displays it. Only the hosted branch may say
that Nemotron selected the policy values.

Do not say that the attendee selected the values unless the notebook contains
an actual override control. This change does not add such a control.

### Validation wording

Rename the current claim from "all acceptance tests" to "normal-path invariant
checks." The fixed valid dataset proves the expected row count, ordering,
similarity bounds, and other normal-path properties. It does not exercise the
missing-anchor or invalid-matrix branch.

Tell the attendee that the selected failure branches were not triggered in the
fixed run. Remove "all acceptance tests" from both Markdown and printed output.
The discussion question becomes: "Are both selected policies appropriate? If
not, which values would you choose and why?"

### Result table

Keep the full atlas object in memory. Display a compact attendee view with only
these columns, in this order:

1. `radius`
2. `query`
3. `rank`
4. `neighbor`
5. `tanimoto`

This keeps the scientific comparison visible without horizontal scrolling.

## Module 3 design

### Approval and completion flow

Keep the current agent plan, human strategy approval, one bounded local
execution, independent artifact validation, and optional hosted audit.

Place callback-rendered results in a dedicated `ipywidgets.Output` region in
the notebook. After approval, the validated panel summary and figures must
appear in that visible region instead of relying on uncaptured callback output.
This region is immediate live-kernel feedback, not the authoritative replay
record. The callback must not make another model request or rerun the analysis.

Tell the attendee:

1. review the plan;
2. choose or accept a strategy;
3. click **Approve Plan & Run Agent**; and
4. after completion, rerun Steps 5 and 6 to inspect the durable receipt and
   chemistry gallery.

The later cells remain safe before approval and show a short waiting message.

### Durable receipt

Add a compact, re-runnable normal cell. The single canonical run object is
`agent_run`: in hosted mode it must be the same object retained as
`module3_workflow.agent_run`; in reference mode it is the direct controller
result. The cell requires the fixed artifact paths, independently revalidates
the panel and report contents, and requires the trace to agree with the run on
mode, approved strategy, analysis success, and audit status. Missing artifacts
or internally inconsistent report or trace content fail closed. This repair
does not add run IDs, artifact hashes, or a new provenance system.

The cell must show:

- workshop mode and fixed model when hosted;
- recommended strategy and approved strategy;
- analysis backend;
- baseline and selected minimum Tanimoto distance;
- baseline and selected descriptor-range coverage;
- analysis validation status; and
- hosted audit status.

The ordinary receipt cell is the authoritative replay surface for the current
kernel. It must not issue a hosted call or re-execute chemistry. It must fail
closed on inconsistent or missing retained evidence rather than infer success.
Recovery after a kernel restart or browser refresh is outside this repair; the
attendee restarts the bounded workflow in that case.

### Status language

Separate analysis validity from audit availability:

- validated artifacts plus a completed audit: **analysis validated; audit
  complete**;
- validated artifacts plus no audit: **analysis validated; audit unavailable**;
- reference-mode validated artifacts plus the deterministic reference audit:
  **analysis validated; reference audit complete**; this is not evidence of a
  hosted model call;
- invalid artifacts or failed execution: **analysis did not validate**.

Do not label the full agent workflow complete when the optional audit is
missing. This distinction must appear consistently in the callback output,
final widget card, workflow transcript, and ordinary receipt. Do not downgrade
a validated panel merely because the optional audit is unavailable.

### Scientific interpretation and gallery

Describe descriptor-range coverage as a guardrail. Both allow-listed strategies
seed descriptor extrema, so this metric is not a strong strategy discriminator.
Describe minimum Tanimoto distance as the more strategy-sensitive comparison.
Retain the existing boundary against binding, activity, ADMET, efficacy,
safety, conformation, and clinical claims.

Render the first 12 selected compounds in three columns per row. Keep molecular
names and the existing MW, cLogP, and TPSA legends readable in a normal
JupyterLab viewport.

## Failure behavior

- A planning failure keeps the existing safe retry control.
- A local execution or artifact-validation failure remains a hard failed run;
  no receipt or gallery may imply success.
- An audit failure is visible and separate from the validated analysis status.
- Running Steps 5 or 6 before approval is safe and does not call the model.
- Re-running Steps 5 or 6 after approval reads retained evidence only.
- No error path may display the hosted API key or raw provider response.

## Test strategy

Use test-driven development for every behavior change.

Focused tests must prove:

- Module 2 uses the correct agent, Python, and attendee role language in both
  Markdown and printed output;
- Module 2 makes only the five compact result columns visible;
- Module 2 labels its checks as normal-path checks and states that failure
  branches were not exercised;
- Module 3 has a dedicated live-kernel callback output region;
- the receipt works before and after approval without another hosted call;
- recommended and approved strategies remain distinct when the attendee
  changes the choice;
- audit-complete and audit-unavailable states render different truthful labels
  in the callback output, final widget card, transcript, and normal receipt;
- reference-mode receipts identify the deterministic reference audit and do not
  imply hosted inference;
- the receipt rejects missing or internally inconsistent report and trace
  evidence;
- invalid analysis cannot render a success receipt or gallery;
- the chemistry grid uses three columns; and
- no new secret, network, model, data, setup, Module 1, or demo behavior enters
  the change.

Run mocked hosted-widget tests without network access, clean reference-mode
kernels for Modules 2 and 3, the focused workshop suites, and the full local
repository suite. Keep tracked notebooks clean of outputs, execution counts,
attachments, and widget state.

## Acceptance criteria

The repair is acceptable when a target attendee can, in one browser session:

- understand that Module 2 applies Nemotron's bounded policy choices and asks
  the attendee to evaluate them afterward;
- understand that Module 3 asks the attendee to approve one proposed strategy
  before Python executes tested code;
- see and assess the important Module 2 similarity result without scrolling;
- approve a Module 3 strategy and see the validated outcome in the notebook;
- distinguish scientific analysis success from hosted audit availability;
- view readable chemical structures in the normal viewport; and
- recover the result by rerunning the stated display cells without another
  model or chemistry run.

Local validation proves the notebook contracts and deterministic execution. A
fresh browser run on the reviewed Launchable is still required to prove the
final hosted widget experience. That run must use the exact post-fix commit on
only `env-3HJtJW3qHg4Dw1I3xt75BfpBmZW`; a stale instance does not count. Record
exactly one Module 2 policy request, one Module 3 plan request, one Module 3
audit attempt, and one local Module 3 analysis execution. Re-running Steps 5
and 6 must add zero hosted requests and zero local analysis executions.
