# Interactive Stage Cards Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reliable stage-by-stage approval interface that lets the audience override only valid parameters and shows both the approved agent tool call and the scientific Python invocation actually executed.

**Architecture:** Extract the existing monolithic scientific loop into a reusable `BoundedWorkflowController` that preserves the exact state machine and hosted conversation. Add a separate `interactive_workflow.py` presentation layer using `ipywidgets`; it pauses between controller calls but never permits a different tool or stage order. Keep `run_workflow(...)` working by implementing it through the same controller.

**Tech Stack:** Python 3.12, Pydantic 2, ipywidgets 8, JupyterLab, OpenAI-compatible hosted Nemotron client, RDKit, nvMolKit, pytest, nbformat

---

## File Structure

- Modify `demo_agent.py`: expose the reusable stagewise controller and retain the synchronous compatibility path.
- Create `command_receipts.py`: deterministically render approved tool calls and RDKit/nvMolKit invocations from validated arguments.
- Create `interactive_workflow.py`: own widget construction, approval callbacks, immutable stage cards, retry behavior, and final rendering.
- Modify `notebooks/nvmolkit_nemotron_demo.ipynb`: replace the synchronous public call with one interactive launch call.
- Modify `requirements.txt`: pin `ipywidgets`.
- Modify `README.md`: describe the approval flow and acceptance checks.
- Modify `tests/test_demo_agent.py`: cover stagewise controller behavior and synchronous parity.
- Create `tests/test_command_receipts.py`: cover every displayed command template.
- Create `tests/test_interactive_workflow.py`: cover controls, overrides, duplicate-click protection, failures, and stage progression.
- Modify `tests/test_notebook.py`: enforce the compact interactive notebook surface.
- Modify `tests/test_gpu_acceptance.py`: exercise the controller with programmatic approvals over the real GPU executors.

### Task 1: Extract the bounded stagewise controller

**Files:**
- Modify: `demo_agent.py:585-679`
- Test: `tests/test_demo_agent.py`

- [ ] **Step 1: Write failing controller tests**

Add tests that create the controller with the existing fake client and executors, request the plan, and advance one stage at a time:

```python
def controller(responses=None, executor_calls=None):
    completions = FakeCompletions(responses or valid_responses())
    value = demo_agent.BoundedWorkflowController.create(
        "Analyze the library.",
        VALID_API_KEY,
        client=fake_client(completions),
        executors=fake_executors(executor_calls),
    )
    return value, completions


def test_stagewise_controller_waits_for_approval_before_execution():
    calls = []
    value, completions = controller(executor_calls=calls)

    plan = value.request_plan()
    proposal = value.request_next_stage()

    assert tuple(item.stage for item in plan.stages) == STAGES
    assert proposal.stage == "inspect_library"
    assert proposal.arguments == demo_agent.InspectionArgs()
    assert calls == []
    assert len(completions.calls) == 2

    result = value.execute_pending(proposal.arguments)

    assert result.stage == "inspect_library"
    assert calls == [("inspect_library", {})]
    assert value.session.state.phase is WorkflowPhase.INSPECTED


def test_stagewise_controller_records_user_override_and_executed_arguments():
    calls = []
    value, _ = controller(executor_calls=calls)
    value.request_plan()
    value.execute_pending(value.request_next_stage().arguments)
    proposed = value.request_next_stage()
    approved = demo_agent.FingerprintArgs(
        radius=3,
        size=2048,
        decision_basis=proposed.arguments.decision_basis,
    )

    value.execute_pending(approved)

    assert calls[-1] == (
        "generate_morgan_fingerprints",
        {"fingerprint_radius": 3, "fingerprint_size": 2048},
    )
    payload = json.loads(value.session.messages[-1]["content"])
    assert payload["user_override"] is True
    assert payload["proposed_arguments"]["radius"] == 2
    assert payload["executed_arguments"]["radius"] == 3


def test_stagewise_controller_rejects_wrong_model_without_execution():
    calls = []
    value, _ = controller(executor_calls=calls)
    value.request_plan()
    value.request_next_stage()

    with pytest.raises(demo_agent.ToolCallError, match="approved arguments"):
        value.execute_pending(
            demo_agent.FingerprintArgs(
                radius=2, size=1024, decision_basis="Wrong stage."
            )
        )

    assert calls == []
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```bash
python3 -m pytest tests/test_demo_agent.py -k stagewise_controller -v
```

Expected: FAIL because `BoundedWorkflowController` and `StageProposal` do not exist.

- [ ] **Step 3: Implement the minimal reusable controller**

In `demo_agent.py`, add the following public state objects after `_executor_arguments` and move the existing loop operations into their methods:

```python
@dataclass(frozen=True)
class StageProposal:
    stage: StageName
    arguments: BaseModel


@dataclass
class BoundedWorkflowController:
    session: AgentSession
    client: Any
    executors: dict[str, Any]
    plan: WorkflowPlan | None = None
    pending: StageProposal | None = None
    stage_results: list[StageResult] = field(default_factory=list)
    report: WorkflowReport | None = None

    @classmethod
    def create(
        cls,
        user_goal: str,
        api_key: str,
        *,
        client: Any = None,
        executors: dict[str, Any] | None = None,
        state: WorkflowState | None = None,
        skill: str | None = None,
    ) -> "BoundedWorkflowController":
        _validate_api_key(api_key)
        if not isinstance(user_goal, str) or not user_goal.strip():
            raise ValueError("A non-empty scientific goal is required.")
        active_executors = _default_executors() if executors is None else executors
        required = set(STAGES) | {"build_workflow_report"}
        if set(active_executors) != required or not all(
            callable(active_executors[key]) for key in required
        ):
            raise ValueError("Executors must match the fixed scientific workflow.")
        session = AgentSession(
            messages=[
                {"role": "system", "content": _system_grounding(skill)},
                {"role": "user", "content": user_goal.strip()},
            ],
            state=state or WorkflowState(),
        )
        return cls(session, client or _client(api_key), active_executors)

    def request_plan(self) -> WorkflowPlan:
        if self.plan is not None or self.session.turn_count != 0:
            raise ToolCallError("The workflow plan can be requested exactly once.")
        self.plan = _request_call(
            self.session, self.client, "submit_workflow_plan", WorkflowPlan, DEFAULT_MODEL
        )
        _append_tool_result(
            self.session,
            {
                "accepted": True,
                "stages": [item.model_dump(mode="json") for item in self.plan.stages],
            },
        )
        return self.plan

    def request_next_stage(self) -> StageProposal:
        if self.plan is None:
            raise ToolCallError("The workflow plan must be accepted first.")
        if self.pending is not None:
            raise ToolCallError("The pending stage must be approved before continuing.")
        stage = self.session.eligible_tool_name()
        if stage not in STAGES:
            raise ToolCallError("All scientific stages are already complete.")
        arguments = _request_call(
            self.session, self.client, stage, TOOL_ARGUMENT_MODELS[stage], DEFAULT_MODEL
        )
        self.pending = StageProposal(stage, arguments)
        return self.pending

    def execute_pending(self, approved: BaseModel) -> StageResult:
        if self.pending is None:
            raise ToolCallError("There is no pending stage to execute.")
        stage = self.pending.stage
        model = TOOL_ARGUMENT_MODELS[stage]
        if type(approved) is not model:
            raise ToolCallError("The approved arguments do not match the pending stage.")
        if self.session.eligible_tool_name() != stage:
            raise ToolCallError("The scientific workflow state is out of phase.")
        approved = model.model_validate(approved.model_dump(mode="python"))
        proposed_values = self.pending.arguments.model_dump(mode="json")
        executed_values = approved.model_dump(mode="json")
        proposed_executor = _executor_arguments(stage, self.pending.arguments)
        executed_executor = _executor_arguments(stage, approved)
        try:
            result = self.executors[stage](self.session.state, **executed_executor)
        except Exception:
            raise ToolCallError("The scientific executor failed.") from None
        if not isinstance(result, StageResult) or result.stage != stage:
            raise ToolCallError("The scientific executor returned an invalid stage result.")
        if self.session.state.phase is not POST_STAGE_PHASES[stage]:
            raise ToolCallError("The scientific executor left the workflow out of phase.")
        _append_tool_result(
            self.session,
            {
                "stage": stage,
                "decision_basis": getattr(approved, "decision_basis", None),
                "proposed_arguments": proposed_values,
                "executed_arguments": executed_values,
                "user_override": proposed_executor != executed_executor,
                "summary": result.summary,
            },
        )
        self.stage_results.append(result)
        self.pending = None
        return result

    def scientific_result(self) -> ScientificLoopResult:
        if self.session.state.phase is not WorkflowPhase.OPTIMIZED:
            raise ToolCallError("The scientific workflow is incomplete.")
        if self.session.turn_count != 7 or len(self.stage_results) != len(STAGES):
            raise ToolCallError("The scientific workflow did not complete exactly once.")
        if self.report is None:
            report = self.executors["build_workflow_report"](self.session.state)
            if not isinstance(report, WorkflowReport):
                raise ToolCallError("The scientific report was invalid.")
            self.report = report
        return ScientificLoopResult(
            tuple(self.session.messages), self.report, self.plan,
            tuple(self.stage_results), self.session.turn_count,
        )

    def request_synthesis(self) -> WorkflowResult:
        scientific = self.scientific_result()
        evidence = _serialize({
            "evidence": [item.__dict__ for item in scientific.report.evidence]
        })
        self.session.messages.append({
            "role": "user", "content": _SYNTHESIS_PROMPT + "\n" + evidence
        })
        conclusion = _request_call(
            self.session, self.client, "submit_synthesis",
            SubmitSynthesisArgs, DEFAULT_MODEL,
        )
        conclusion = validate_conclusion(conclusion, scientific.report)
        return WorkflowResult(
            tuple(self.session.messages), scientific.report, scientific.plan,
            conclusion, scientific.stage_results, self.session.turn_count,
        )
```

Import `field` from `dataclasses`. Update `run_workflow(...)` to preserve its current public behavior and exception rendering while using this same synthesis sequence; do not maintain a second validation path with different semantics.

- [ ] **Step 4: Reimplement `run_scientific_loop(...)` through the controller**

Replace its duplicated setup and loop with:

```python
controller = BoundedWorkflowController.create(
    user_goal, api_key, client=client, executors=executors,
    state=state, skill=skill,
)
try:
    plan = controller.request_plan()
    _emit_progress(progress_callback, "plan", plan)
    for stage in STAGES:
        proposal = controller.request_next_stage()
        if proposal.stage != stage:
            raise ToolCallError("The scientific workflow state is out of phase.")
        result = controller.execute_pending(proposal.arguments)
        _emit_progress(
            progress_callback,
            "stage",
            {"result": result, "arguments": proposal.arguments},
        )
    return controller.scientific_result()
except Exception as error:
    _emit_progress(progress_callback, "failure", str(error))
    raise
```

Preserve the current error messages and all existing `run_scientific_loop` and `run_workflow` behavior.

- [ ] **Step 5: Run controller and existing agent tests**

Run:

```bash
python3 -m pytest tests/test_demo_agent.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit the controller extraction**

```bash
git add demo_agent.py tests/test_demo_agent.py
git commit -m "refactor: expose bounded stagewise workflow controller"
```

### Task 2: Add deterministic command receipts

**Files:**
- Create: `command_receipts.py`
- Create: `tests/test_command_receipts.py`

- [ ] **Step 1: Write failing tests for all six stages**

Create parameterized tests that require exact output and verify secrets and runtime object representations cannot enter it:

```python
@pytest.mark.parametrize(
    ("stage", "arguments", "tool_fragment", "science_fragment", "label"),
    [
        ("inspect_library", InspectionArgs(), "inspect_library()", "inspect_library(state, DATA_PATH)", "RDKit inspection executed by Python"),
        ("generate_morgan_fingerprints", FingerprintArgs(radius=2, size=1024, decision_basis="brief"), "radius=2, size=1024", "MorganFingerprintGenerator(radius=2, fpSize=1024)", "nvMolKit invocation executed by Python"),
        ("measure_tanimoto_similarity", SimilarityArgs(), "measure_tanimoto_similarity()", "crossTanimotoSimilarity(fingerprints)", "nvMolKit invocation executed by Python"),
        ("discover_fused_butina_clusters", ClusterArgs(cutoff=0.5, decision_basis="brief"), "cutoff=0.5", "fused_butina(fingerprints.torch(), cutoff=0.5)", "nvMolKit invocation executed by Python"),
        ("embed_representative_conformers", EmbedArgs(representative_count=4, policy="include_singleton_if_available", conformers_per_representative=4, decision_basis="brief"), "representative_count=4", "EmbedMolecules(molecules, parameters, confsPerMolecule=4, maxIterations=-1)", "nvMolKit invocation executed by Python"),
        ("optimize_conformers_mmff94", OptimizationArgs(), "optimize_conformers_mmff94()", "MMFFOptimizeMoleculesConfs(molecules, maxIters=500, output=CoordinateOutput.DEVICE)", "nvMolKit invocation executed by Python"),
    ],
)
def test_command_receipts_match_validated_calls(
    stage, arguments, tool_fragment, science_fragment, label
):
    receipt = command_receipt(stage, arguments)
    assert tool_fragment in receipt.approved_tool_call
    assert science_fragment in receipt.scientific_invocation
    assert receipt.scientific_label == label
    assert "decision_basis" not in receipt.approved_tool_call
    assert "nvapi-" not in repr(receipt)
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
python3 -m pytest tests/test_command_receipts.py -v
```

Expected: FAIL because `command_receipts.py` does not exist.

- [ ] **Step 3: Implement a closed six-stage formatter**

Create `CommandReceipt` and `command_receipt(stage, arguments)`. Use a closed `if` chain, `repr` for validated scalar values, stable symbolic names for runtime objects, and raise `ValueError("Unsupported workflow stage.")` for any other stage. The returned object is:

```python
@dataclass(frozen=True)
class CommandReceipt:
    approved_tool_call: str
    scientific_label: str
    scientific_invocation: str
```

The templates must produce the exact fragments asserted above. For embedding, include the Python/RDKit representative-selection line before the nvMolKit call, but keep the nvMolKit call as the final line. Never accept raw display strings from Nemotron.

- [ ] **Step 4: Run formatter tests**

Run:

```bash
python3 -m pytest tests/test_command_receipts.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit command receipts**

```bash
git add command_receipts.py tests/test_command_receipts.py
git commit -m "feat: render validated scientific command receipts"
```

### Task 3: Build the guarded interactive stage cards

**Files:**
- Create: `interactive_workflow.py`
- Create: `tests/test_interactive_workflow.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Pin and install the widget dependency**

Append exactly:

```text
ipywidgets==8.1.7
```

Install the project requirements into the active development environment:

```bash
python3 -m pip install -r requirements.txt
```

Expected: `python3 -c "import ipywidgets; print(ipywidgets.__version__)"` prints `8.1.7`.

- [ ] **Step 2: Write failing control-construction tests**

Test the exact widget domains:

```python
def test_fingerprint_controls_are_closed_choices():
    proposal = StageProposal(
        "generate_morgan_fingerprints",
        FingerprintArgs(radius=2, size=1024, decision_basis="brief"),
    )
    controls = controls_for(proposal)
    assert tuple(controls["radius"].options) == (2, 3)
    assert tuple(controls["size"].options) == (1024, 2048)
    assert controls["radius"].value == 2
    assert controls["size"].value == 1024


def test_cluster_and_embedding_controls_match_approved_bounds():
    cluster = controls_for(StageProposal(
        "discover_fused_butina_clusters",
        ClusterArgs(cutoff=0.5, decision_basis="brief"),
    ))
    assert (cluster["cutoff"].min, cluster["cutoff"].max, cluster["cutoff"].step) == (0.4, 0.6, 0.05)
    embed = controls_for(StageProposal(
        "embed_representative_conformers",
        EmbedArgs(representative_count=4, policy="largest_clusters_first", conformers_per_representative=4, decision_basis="brief"),
    ))
    assert (embed["representative_count"].min, embed["representative_count"].max) == (3, 6)
    assert tuple(embed["policy"].options) == (
        "largest_clusters_first", "include_singleton_if_available"
    )
    assert (embed["conformers_per_representative"].min, embed["conformers_per_representative"].max) == (3, 8)
```

- [ ] **Step 3: Write failing interaction tests with a fake controller**

Use `unittest.mock.Mock` controllers with explicit return values and side effects. A shared helper returns the validated fixed plan:

```python
def fixed_plan():
    return WorkflowPlan.model_validate({
        "stages": [
            {"stage": stage, "rationale": f"Run {stage} after its prerequisite."}
            for stage in STAGES
        ]
    })
```

Verify:

```python
def test_approval_uses_override_and_click_executes_once():
    proposal = StageProposal(
        "generate_morgan_fingerprints",
        FingerprintArgs(radius=2, size=1024, decision_basis="brief"),
    )
    controller = Mock()
    controller.request_plan.return_value = fixed_plan()
    controller.request_next_stage.side_effect = [
        proposal, ToolCallError("stop after tested stage")
    ]
    controller.execute_pending.return_value = StageResult(
        proposal.stage, "nvMolKit MorganFingerprintGenerator", {"molecule_count": 2}
    )
    view = InteractiveWorkflow(controller)
    view.start()
    view.controls["radius"].value = 3

    original_button = view.approve_button
    original_button.click()
    original_button.click()

    assert controller.execute_pending.call_count == 1
    assert controller.execute_pending.call_args.args[0].radius == 3
    assert "radius=3" in view.transcript_text
    assert "MorganFingerprintGenerator(radius=3" in view.transcript_text


def test_proposal_failure_is_rendered_and_retryable_without_raising():
    proposal = StageProposal("inspect_library", InspectionArgs())
    controller = Mock()
    controller.request_plan.return_value = fixed_plan()
    controller.request_next_stage.side_effect = [
        ToolCallError("hosted request failed"), proposal
    ]
    view = InteractiveWorkflow(controller)

    view.start()

    assert view.status == "proposal_failed"
    assert view.retry_button.description == "Retry Proposal"
    view.retry_button.click()
    assert view.status == "awaiting_approval"


def test_executor_failure_stays_on_stage_and_reuses_approved_arguments():
    proposal = StageProposal("inspect_library", InspectionArgs())
    controller = Mock()
    controller.request_plan.return_value = fixed_plan()
    controller.request_next_stage.return_value = proposal
    controller.execute_pending.side_effect = [
        ToolCallError("The scientific executor failed."),
        StageResult("inspect_library", "RDKit input validation", {"valid_count": 2}),
    ]
    view = InteractiveWorkflow(controller)
    view.start()
    view.approve_button.click()

    assert view.status == "execution_failed"
    assert view.retry_button.description == "Retry Execution"
    view.retry_button.click()
    assert controller.execute_pending.call_count == 2
    first, second = controller.execute_pending.call_args_list
    assert first.args[0] == second.args[0] == InspectionArgs()


def test_unexpected_callback_error_is_generic_and_secret_safe():
    controller = Mock()
    controller.request_plan.side_effect = RuntimeError("nvapi-secret-must-not-render")
    view = InteractiveWorkflow(controller)

    view.start()

    assert view.status == "stopped"
    assert "local error" in view.transcript_text.lower()
    assert "nvapi-secret-must-not-render" not in view.transcript_text
```

- [ ] **Step 4: Run interaction tests and verify they fail**

Run:

```bash
python3 -m pytest tests/test_interactive_workflow.py -v
```

Expected: FAIL because the interactive module does not exist.

- [ ] **Step 5: Implement `controls_for(...)` and approved-argument reconstruction**

Create dropdowns for literal values, sliders for integer ranges, and a `FloatSlider(min=0.40, max=0.60, step=0.05, readout_format=".2f")` for the cutoff. Parameter-free stages return `{}`. Rebuild the same strict Pydantic model from the control values plus the original `decision_basis`; do not mutate the frozen proposal.

- [ ] **Step 6: Implement `InteractiveWorkflow`**

The class must:

- display a **Start Agent** button without making a hosted call during construction;
- expose `start()` as the guarded callback body and maintain `status`, `controls`, `approve_button`, and `retry_button` references for deterministic tests;
- maintain `transcript_text` as the plain-text equivalent of rendered cards so behavior can be asserted without a browser;
- request and display the fixed plan inside the guarded start callback;
- create one active stage card and append completed cards to a `VBox` transcript;
- show evidence summaries, concise decision basis, proposed call, and valid controls;
- disable the button and controls before executing;
- generate command blocks only from the approved Pydantic model;
- call `controller.execute_pending(...)` exactly once per approval;
- render `StageResult.figures` with the existing `_display_figure` helper;
- request the next proposal only after success;
- request and render the synthesis after stage six;
- catch `ToolCallError` and the known authentication guidance inside callbacks;
- render a generic local-error message for unexpected exceptions; and
- retain approved arguments for execution retry only while the workflow phase still matches the pending stage.

Expose:

```python
def launch_interactive_workflow(
    user_goal: str,
    api_key: str,
    *,
    skill: str | None = None,
    client: Any = None,
    executors: dict[str, Any] | None = None,
) -> InteractiveWorkflow:
    controller = BoundedWorkflowController.create(
        user_goal, api_key, skill=skill, client=client, executors=executors
    )
    workflow = InteractiveWorkflow(controller)
    workflow.display()
    return workflow
```

`launch_interactive_workflow(...)` must construct and display only; the hosted plan request begins when **Start Agent** is clicked.

- [ ] **Step 7: Run command and interaction tests**

Run:

```bash
python3 -m pytest tests/test_command_receipts.py tests/test_interactive_workflow.py -v
```

Expected: all tests PASS.

- [ ] **Step 8: Commit the interactive layer**

```bash
git add requirements.txt interactive_workflow.py tests/test_interactive_workflow.py
git commit -m "feat: add guarded interactive workflow stage cards"
```

### Task 4: Integrate the compact notebook presentation

**Files:**
- Modify: `notebooks/nvmolkit_nemotron_demo.ipynb`
- Modify: `tests/test_notebook.py`
- Modify: `README.md`

- [ ] **Step 1: Update notebook tests first**

Require the Preflight cell to import `notebook_preflight` and `launch_interactive_workflow`, and require exactly one public launch call:

```python
calls = [
    node for node in ast.walk(tree)
    if isinstance(node, ast.Call)
    and dotted_name(node.func) == "launch_interactive_workflow"
]
assert len(calls) == 1
assert "workflow = launch_interactive_workflow(" in notebook.cells[6].source
assert "skill=(PROJECT_ROOT / \"skills\" / \"nvmolkit\" / \"SKILL.md\").read_text(encoding=\"utf-8\")" in notebook.cells[6].source
assert "run_workflow(" not in code
```

Retain the eight-cell limit, the approved introduction, the explicit skill line, the scientific boundary, and the no-functions-or-classes rule in visible notebook code.

- [ ] **Step 2: Run notebook tests and verify they fail**

Run:

```bash
python3 -m pytest tests/test_notebook.py -v
```

Expected: FAIL because the notebook still calls `run_workflow(...)`.

- [ ] **Step 3: Change only the two necessary notebook code cells**

The Preflight cell imports:

```python
from demo_agent import notebook_preflight  # noqa: E402
from interactive_workflow import launch_interactive_workflow  # noqa: E402
```

The Agent run cell becomes:

```python
workflow = launch_interactive_workflow(
    USER_GOAL, api_key,
    skill=(PROJECT_ROOT / "skills" / "nvmolkit" / "SKILL.md").read_text(encoding="utf-8"),
)
```

Update the Agent run markdown to tell the presenter to click **Start Agent**, review each proposal, optionally change only the displayed bounded controls, and click **Approve & Run**. State that every completed card shows the approved agent call and the corresponding RDKit or nvMolKit invocation.

- [ ] **Step 4: Update README acceptance wording**

Replace the synchronous hosted acceptance description with the exact interactive sequence: one plan, six stage approvals, six command receipts, six completed result cards, and one schema-checked synthesis. Add one sentence that the notebook cell returns after displaying the interface and guarded button failures remain inside the active card.

- [ ] **Step 5: Run notebook and documentation tests**

Run:

```bash
python3 -m pytest tests/test_notebook.py tests/test_skill_snapshot.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit notebook integration**

```bash
git add notebooks/nvmolkit_nemotron_demo.ipynb tests/test_notebook.py README.md
git commit -m "feat: make the chemistry demo stage-interactive"
```

### Task 5: Exercise the controller on the real GPU path

**Files:**
- Modify: `tests/test_gpu_acceptance.py`

- [ ] **Step 1: Add programmatic-approval coverage around the controller**

Adapt the existing GPU test so its scientific operations are invoked through `BoundedWorkflowController`. Use a scripted fake hosted client for the fixed plan and six valid stage calls; do not require hosted inference for the GPU gate. For each stage:

```python
controller.request_plan()
for expected_stage in STAGES:
    proposal = controller.request_next_stage()
    assert proposal.stage == expected_stage
    result = controller.execute_pending(proposal.arguments)
    assert result.stage == expected_stage
scientific = controller.scientific_result()
```

Retain the current CUDA, L4, nvMolKit-version, similarity, clustering, conformer, convergence, coordinate, and provenance assertions. Inject the temporary input-library executor for `inspect_library`; use the real nvMolKit executors for all five GPU stages.

- [ ] **Step 2: Verify the test remains skipped locally**

Run:

```bash
python3 -m pytest tests/test_gpu_acceptance.py -v
```

Expected: one SKIPPED test because `RUN_GPU_TESTS` is not set.

- [ ] **Step 3: Run the full deterministic suite**

Run:

```bash
python3 -m pytest -v
```

Expected: all non-GPU tests PASS and only the explicit GPU acceptance test is SKIPPED.

- [ ] **Step 4: Commit GPU acceptance wiring**

```bash
git add tests/test_gpu_acceptance.py
git commit -m "test: route GPU acceptance through stage approvals"
```

### Task 6: Live Brev acceptance and handoff

**Files:**
- Modify only if acceptance exposes a defect: the smallest responsible source or test file

- [ ] **Step 1: Run the real GPU gate on the task-owned Brev instance**

From the repository root on the Brev VM, using the same CPython 3.12 environment as Jupyter:

```bash
RUN_GPU_TESTS=1 python3 -m pytest tests/test_gpu_acceptance.py -v
```

Expected: PASS on the intended NVIDIA L4 environment. Record the exact interpreter path, GPU name, nvMolKit version, test result, and commit SHA; do not infer GPU acceptance from local tests.

- [ ] **Step 2: Run hosted interactive acceptance in a fresh kernel**

Open the notebook through the Brev Secure Link, run Preflight, enter the hosted Developer API key through the hidden prompt, then verify:

1. the Agent run cell displays **Start Agent** and completes without a red cell error;
2. the fixed six-stage plan appears after the click;
3. every stage waits for approval;
4. fingerprint radius can be changed to another allowed dropdown value;
5. the completed card records proposed and approved values;
6. both command blocks reflect the approved value;
7. every figure appears below the corresponding completed card;
8. later stages receive the executed structured result; and
9. the final synthesis appears only after MMFF94 completes.

- [ ] **Step 3: Verify guarded recovery once**

Before the final clean run, use a test-only fake controller locally—not a deliberately invalid scientific command on the live GPU—to confirm **Retry Proposal** and **Retry Execution** remain on the same card and do not produce a failed notebook cell. Do not inject a live failure into the presentation notebook.

- [ ] **Step 4: Run final repository checks**

Run locally:

```bash
python3 -m pytest -v
git diff --check
git status --short
```

Expected: all deterministic tests PASS; only the documented GPU skip occurs locally; `git diff --check` is clean; no unintended files are staged or committed.

- [ ] **Step 5: Commit only acceptance-driven fixes, if any**

If no defect was found, do not create an empty commit. If a defect was found, add its regression test first, make the smallest fix, rerun the relevant focused test plus the full deterministic suite, and commit only those files with a message describing the defect.
