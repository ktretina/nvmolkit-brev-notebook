import inspect
from types import SimpleNamespace

import ipywidgets as widgets
import pytest

import demo_agent
from chemistry_workflow import StageResult
from demo_agent import (
    ClusterArgs, EmbedArgs, FingerprintArgs, InspectionArgs, OptimizationArgs,
    SimilarityArgs, StageProposal, ToolCallError,
)
from interactive_workflow import InteractiveWorkflow, controls_for


def proposals():
    return [
        StageProposal("inspect_library", InspectionArgs()),
        StageProposal("generate_morgan_fingerprints", FingerprintArgs(radius=2, size=1024, decision_basis="Choose standard fingerprint settings.")),
        StageProposal("measure_tanimoto_similarity", SimilarityArgs()),
        StageProposal("discover_fused_butina_clusters", ClusterArgs(cutoff=0.45, decision_basis="Separate close analogs.")),
        StageProposal("embed_representative_conformers", EmbedArgs(representative_count=4, policy="largest_clusters_first", conformers_per_representative=5, decision_basis="Sample the largest groups.")),
        StageProposal("optimize_conformers_mmff94", OptimizationArgs()),
    ]


class Session:
    def __init__(self, controller):
        self.controller = controller
        self.turn_count = 0
        self.state = SimpleNamespace(phase=demo_agent.WorkflowPhase.NEW)
    def eligible_tool_name(self): return demo_agent.STAGES[self.controller.index] if self.controller.index < 6 else "submit_synthesis"


class Controller:
    def __init__(self):
        self.calls = []
        self.plan_failures = []
        self.proposal_failures = []
        self.execution_failures = []
        self.synthesis_failures = []
        self.index = 0
        self.pending = None
        self.plan = None
        self.report = None
        self.synthesis_prompt_appended = False
        self.stage_results = []
        self.session = Session(self)

    def request_plan(self):
        self.calls.append("plan")
        if self.plan_failures: raise self.plan_failures.pop(0)
        self.plan = demo_agent.WorkflowPlan(stages=[{"stage": stage, "rationale": f"Plan {stage}."} for stage in demo_agent.STAGES])
        self.session.turn_count += 1
        return self.plan

    def request_next_stage(self):
        self.calls.append("proposal")
        if self.proposal_failures: raise self.proposal_failures.pop(0)
        self.pending = proposals()[self.index]
        self.session.turn_count += 1
        return self.pending

    def execute_pending(self, approved):
        self.calls.append(("execute", approved))
        if self.execution_failures: raise self.execution_failures.pop(0)
        stage = self.pending.stage
        summary = {
            "raw_count": 256, "valid_count": 255,
            "molecule_count": 255, "active_bits_median": 14.0,
            "q1": 0.12, "median": 0.22, "cluster_count": 17,
            "selected_representative_count": 4, "converged_conformer_count": 20,
        }
        result = StageResult(stage, f"label {stage}", summary)
        self.stage_results.append(result)
        self.pending = None
        self.index += 1
        self.session.state.phase = tuple(demo_agent.POST_STAGE_PHASES.values())[self.index - 1]
        return result

    def request_synthesis(self):
        self.calls.append("synthesis")
        self.report = SimpleNamespace(evidence=())
        self.synthesis_prompt_appended = True
        if self.synthesis_failures: raise self.synthesis_failures.pop(0)
        return SimpleNamespace(conclusion=SimpleNamespace())


def html_text(widget):
    values = []
    if isinstance(widget, widgets.HTML): values.append(widget.value)
    for child in getattr(widget, "children", ()): values.extend([html_text(child)])
    return " ".join(values)


def started(controller=None):
    controller = controller or Controller()
    workflow = InteractiveWorkflow(controller)
    workflow.start_button.click()
    return workflow, controller


def test_all_control_domains_and_parameter_free_stages():
    items = proposals()
    for index in (0, 2, 5): assert controls_for(items[index]) == {}
    fingerprint = controls_for(items[1])
    assert fingerprint["radius"].options == (2, 3)
    assert fingerprint["size"].options == (1024, 2048)
    cluster = controls_for(items[3])["cutoff"]
    assert (cluster.min, cluster.max, cluster.step, cluster.readout_format) == (0.4, 0.6, 0.05, ".2f")
    embed = controls_for(items[4])
    assert (embed["representative_count"].min, embed["representative_count"].max) == (3, 6)
    assert embed["policy"].options == ("largest_clusters_first", "include_singleton_if_available")
    assert (embed["conformers_per_representative"].min, embed["conformers_per_representative"].max) == (3, 8)
    with pytest.raises(ValueError): controls_for(StageProposal("unknown", InspectionArgs()))
    with pytest.raises(ValueError): controls_for(StageProposal("inspect_library", items[1].arguments))


def test_construction_display_no_calls_start_once_and_plan_retained(monkeypatch):
    controller = Controller(); rendered = []
    monkeypatch.setattr("interactive_workflow.ipython_display", rendered.append)
    workflow = InteractiveWorkflow(controller)
    workflow.display()
    assert controller.calls == [] and rendered == [workflow.root]
    workflow.start_button.click(); workflow.start_button.click()
    assert controller.calls == ["plan", "proposal"]
    assert "Fixed workflow plan" in html_text(workflow.root)
    assert len(workflow.plan_cards) == 1


def test_proposal_has_evidence_values_decision_and_proposed_call():
    workflow, controller = started()
    workflow.approve_button.click()
    text = html_text(workflow.active_card)
    assert "raw_count=256" in text and "valid_count=255" in text
    assert "Choose standard fingerprint settings." in text
    assert "generate_morgan_fingerprints(radius=2, size=1024)" in text


def test_override_exact_model_basis_receipt_and_completed_content():
    workflow, controller = started()
    workflow.approve_button.click()
    button = workflow.approve_button
    workflow.controls["radius"].value = 3
    workflow.controls["size"].value = 2048
    button.click(); button.click()
    approved = [call[1] for call in controller.calls if isinstance(call, tuple)][1]
    assert type(approved) is FingerprintArgs
    assert approved.decision_basis == "Choose standard fingerprint settings."
    card = workflow.completed_cards[1]
    text = html_text(card)
    assert "Decision summary" in text and "Proposed tool call" in text
    assert "radius=2" in text and "radius=3" in text and "fpSize=2048" in text
    assert button.disabled and all(control.disabled for control in card.children if hasattr(control, "disabled"))
    assert len([call for call in controller.calls if isinstance(call, tuple)]) == 2


@pytest.mark.parametrize("where", ["plan", "proposal"])
def test_known_hosted_failure_has_one_guarded_retry(where):
    controller = Controller()
    getattr(controller, f"{where}_failures").append(ToolCallError("safe hosted failure"))
    workflow = InteractiveWorkflow(controller); workflow.start_button.click()
    assert workflow.status == f"{where}_failed"
    stale = workflow.retry_button; stale.click(); stale.click()
    assert workflow.status == "awaiting_approval"
    expected = ["plan", "plan", "proposal"] if where == "plan" else ["plan", "proposal", "proposal"]
    assert controller.calls == expected


def test_retry_plan_rechecks_turn_phase_and_pending():
    controller = Controller(); controller.plan_failures.append(ToolCallError("safe hosted failure"))
    workflow = InteractiveWorkflow(controller); workflow.start_button.click()
    before = list(controller.calls)
    controller.session.turn_count = 1
    workflow.retry_button.click()
    assert workflow.status == "stopped" and controller.calls == before


@pytest.mark.parametrize("mutation", ["stage_results", "report", "prompt"])
def test_retry_plan_rejects_noncanonical_fresh_state(mutation):
    controller = Controller(); controller.plan_failures.append(ToolCallError("safe hosted failure"))
    workflow = InteractiveWorkflow(controller); workflow.start_button.click()
    if mutation == "stage_results": controller.stage_results.append(StageResult("inspect_library", "label", {}))
    elif mutation == "report": controller.report = SimpleNamespace()
    else: controller.synthesis_prompt_appended = True
    before = list(controller.calls)
    workflow.retry_button.click()
    assert workflow.status == "stopped" and workflow.retry_button is None
    assert controller.calls == before


def test_safe_execution_retry_uses_same_model_and_stale_retry_is_inert():
    controller = Controller(); controller.execution_failures.append(ToolCallError("executor failed"))
    workflow, _ = started(controller)
    workflow.approve_button.click()
    assert workflow.status == "execution_failed"
    first = [call[1] for call in controller.calls if isinstance(call, tuple)][0]
    retry = workflow.retry_button; retry.click(); retry.click()
    calls = [call[1] for call in controller.calls if isinstance(call, tuple)]
    assert calls == [first, first]
    assert workflow.status == "awaiting_approval" and retry.disabled


@pytest.mark.parametrize("mutation", ["pending", "phase"])
def test_retry_rechecks_pending_identity_and_phase_or_stops(mutation):
    controller = Controller(); controller.execution_failures.append(ToolCallError("executor failed"))
    workflow, _ = started(controller); workflow.approve_button.click()
    executions = len([x for x in controller.calls if isinstance(x, tuple)])
    if mutation == "pending": controller.pending = proposals()[0]
    else: controller.index = 1
    workflow.retry_button.click()
    assert workflow.status == "stopped" and workflow.retry_button is None
    assert len([x for x in controller.calls if isinstance(x, tuple)]) == executions


@pytest.mark.parametrize("where", ["plan", "proposal", "execution", "synthesis"])
def test_unexpected_failures_are_generic_stopped_and_secret_safe(monkeypatch, where):
    controller = Controller(); secret = "SECRET-raw-exception"
    getattr(controller, f"{where}_failures").append(RuntimeError(secret))
    monkeypatch.setattr(demo_agent, "_display_conclusion", lambda result: None)
    workflow = InteractiveWorkflow(controller); workflow.start_button.click()
    if where == "execution": workflow.approve_button.click()
    elif where == "synthesis":
        for _ in range(6): workflow.approve_button.click()
    assert workflow.status == "stopped" and workflow.retry_button is None
    assert secret not in workflow.transcript_text
    assert "local workflow error" in workflow.transcript_text.lower()


def test_six_stages_gate_synthesis_and_render_conclusion(monkeypatch):
    rendered = []; monkeypatch.setattr(demo_agent, "_display_conclusion", rendered.append)
    workflow, controller = started()
    for index in range(6):
        assert controller.calls.count("synthesis") == 0
        workflow.approve_button.click()
    assert controller.calls.count("synthesis") == 1
    assert workflow.status == "completed" and len(workflow.completed_cards) == 6
    assert len(rendered) == 1 and "Final synthesis" in html_text(workflow.active_card)


def test_known_synthesis_failure_has_guarded_retry(monkeypatch):
    monkeypatch.setattr(demo_agent, "_display_conclusion", lambda result: None)
    controller = Controller(); controller.synthesis_failures.append(ToolCallError("hosted synthesis failed"))
    workflow, _ = started(controller)
    for _ in range(6): workflow.approve_button.click()
    assert workflow.status == "synthesis_failed"
    retry = workflow.retry_button; retry.click(); retry.click()
    assert workflow.status == "completed" and retry.disabled
    assert controller.calls.count("synthesis") == 2


def test_retry_synthesis_rechecks_turn_phase_and_pending(monkeypatch):
    monkeypatch.setattr(demo_agent, "_display_conclusion", lambda result: None)
    controller = Controller(); controller.synthesis_failures.append(ToolCallError("hosted synthesis failed"))
    workflow, _ = started(controller)
    for _ in range(6): workflow.approve_button.click()
    before = controller.calls.count("synthesis")
    controller.session.state.phase = demo_agent.WorkflowPhase.EMBEDDED
    workflow.retry_button.click()
    assert workflow.status == "stopped"
    assert controller.calls.count("synthesis") == before


@pytest.mark.parametrize("mutation", ["pending", "plan", "missing_stage", "wrong_order", "report", "prompt"])
def test_retry_synthesis_rejects_noncanonical_state(monkeypatch, mutation):
    monkeypatch.setattr(demo_agent, "_display_conclusion", lambda result: None)
    controller = Controller(); controller.synthesis_failures.append(ToolCallError("hosted synthesis failed"))
    workflow, _ = started(controller)
    for _ in range(6): workflow.approve_button.click()
    if mutation == "pending": controller.pending = proposals()[5]
    elif mutation == "plan": controller.plan = None
    elif mutation == "missing_stage": controller.stage_results.pop()
    elif mutation == "wrong_order": controller.stage_results[0], controller.stage_results[1] = controller.stage_results[1], controller.stage_results[0]
    elif mutation == "report": controller.report = None
    else: controller.synthesis_prompt_appended = False
    before = controller.calls.count("synthesis")
    workflow.retry_button.click()
    assert workflow.status == "stopped" and workflow.retry_button is None
    assert controller.calls.count("synthesis") == before


def test_known_synthesis_error_that_mutates_state_never_offers_retry(monkeypatch):
    monkeypatch.setattr(demo_agent, "_display_conclusion", lambda result: None)
    controller = Controller()
    def unsafe_synthesis():
        controller.calls.append("synthesis")
        controller.report = None
        controller.synthesis_prompt_appended = True
        raise ToolCallError("known but state-mutating failure")
    controller.request_synthesis = unsafe_synthesis
    workflow, _ = started(controller)
    for _ in range(6): workflow.approve_button.click()
    assert workflow.status == "stopped" and workflow.retry_button is None


def test_control_observer_unexpected_error_is_secret_safe_fatal(monkeypatch):
    workflow, _ = started()
    workflow.approve_button.click()
    monkeypatch.setattr("interactive_workflow._approved_model", lambda *_: (_ for _ in ()).throw(RuntimeError("OBSERVER-SECRET")))
    workflow.controls["radius"].value = 3
    assert workflow.status == "stopped" and workflow.retry_button is None
    assert "OBSERVER-SECRET" not in workflow.transcript_text
    assert "local workflow error" in workflow.transcript_text.lower()


def test_launch_signature_create_display_and_no_hosted_call(monkeypatch):
    from interactive_workflow import launch_interactive_workflow

    controller = Controller(); created = []; displayed = []
    def create(*args, **kwargs):
        created.append((args, kwargs)); return controller
    monkeypatch.setattr(demo_agent.BoundedWorkflowController, "create", create)
    monkeypatch.setattr(InteractiveWorkflow, "display", lambda self: displayed.append(self))
    workflow = launch_interactive_workflow("goal", "nvapi-test", skill="s", client="c", executors={})
    assert created == [(("goal", "nvapi-test"), {"skill": "s", "client": "c", "executors": {}})]
    assert displayed == [workflow] and controller.calls == []
    assert str(inspect.signature(launch_interactive_workflow)).startswith("(user_goal: 'str', api_key: 'str', *")
    with pytest.raises(TypeError): launch_interactive_workflow("goal", "nvapi-test", "positional")
