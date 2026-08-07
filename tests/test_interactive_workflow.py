import inspect
import re
from io import BytesIO
from dataclasses import fields, replace
from html import escape
from types import SimpleNamespace

import httpx
import ipywidgets as widgets
import numpy as np
import pytest
from PIL import Image as PILImage
from openai import AuthenticationError

import demo_agent
from chemistry_workflow import StageResult
from demo_agent import (
    ClusterArgs, EmbedArgs, FingerprintArgs, InspectionArgs, OptimizationArgs,
    ObjectiveSelection, SimilarityArgs, StageProposal, ToolCallError,
)
from objective_challenge import (
    ObjectiveAttempt, ObjectiveCandidate, ObjectiveContext, ObjectiveSwap,
    TerminationReason,
    TARGET_FRACTION, accepted_maxima, build_action_menu, evaluate_selected_swap,
    build_objective_evidence, measure_panel, score_key, terminal_objective_run,
)
from objective_fixtures import (
    BOUNDARY_CASES, controlled_context_with_action_count,
    controlled_context_without_improving_swaps, two_revision_context,
    evidence_report, report_and_run,
)
from objective_findings import (
    FindingCatalog, build_evidence_snapshot, build_finding_catalog_from_snapshot,
)
from objective_receipts import objective_receipt
from interactive_workflow import InteractiveWorkflow, controls_for
from ipywidgets.embed import dependency_state, embed_minimal_html


def connect_error(label="objective transport failure"):
    return httpx.ConnectError(
        label,
        request=httpx.Request("POST", f"{demo_agent.NVIDIA_BASE_URL}/chat/completions"),
    )


def auth_error():
    response = httpx.Response(
        401,
        request=httpx.Request("POST", f"{demo_agent.NVIDIA_BASE_URL}/chat/completions"),
    )
    return AuthenticationError("objective authorization failure", response=response, body=None)


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
        self.objective_failures = []
        self.index = 0
        self.pending = None
        self.plan = None
        self.report = None
        self.synthesis_prompt_appended = False
        self.figures = ()
        self.stage_results = []
        self.objective_context = None
        self.pending_action_menu = None
        self.pending_objective_selection = None
        self.objective_attempts = []
        self.objective_suggestions = ()
        self.objective_rejection_count = 0
        self.objective_run = None
        self.objective_evidence = None
        self.objective_prompt_appended = False
        self._objective_transport_retry_pending = False
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
        result = StageResult(stage, f"label {stage}", summary, self.figures)
        self.stage_results.append(result)
        self.pending = None
        self.index += 1
        self.session.state.phase = tuple(demo_agent.POST_STAGE_PHASES.values())[self.index - 1]
        return result

    def begin_objective_challenge(self):
        self.calls.append("objective_begin")
        self.report = SimpleNamespace(evidence=())
        self.objective_prompt_appended = True
        self.objective_context = two_revision_context()
        baseline = measure_panel(
            self.objective_context, self.objective_context.baseline_ids
        )
        self.pending_action_menu = build_action_menu(
            self.objective_context, baseline, 0
        )
        return self.objective_context

    def request_objective_attempt(self):
        self.calls.append("objective_proposal")
        if self.objective_failures:
            error = self.objective_failures.pop(0)
            if isinstance(error, (httpx.TransportError, demo_agent.APIConnectionError)):
                self._objective_transport_retry_pending = True
                raise ToolCallError("hosted objective proposal failed") from None
            demo_agent._raise_request_error(error)
        self._objective_transport_retry_pending = False
        menu = self.pending_action_menu
        action = accepted_maxima(menu)[0]
        self.pending_objective_selection = ObjectiveSelection(
            state_id=menu.state_id,
            swap_id=action.swap_id,
            observed_limiting_pairs=[list(pair) for pair in menu.source.limiting_pairs],
            decision_rule="maximize_predicted_minimum_distance",
        )
        self.session.turn_count += 1
        return self.pending_objective_selection

    @property
    def objective_transport_retry_pending(self):
        return self._objective_transport_retry_pending

    def execute_objective_attempt(self, selection):
        self.calls.append(("objective_execute", selection))
        if selection is not self.pending_objective_selection:
            raise ToolCallError("The exact pending objective selection is required.")
        menu = self.pending_action_menu
        action = next(item for item in menu.actions if item.swap_id == selection.swap_id)
        number = len(self.objective_attempts) + 1
        attempt = evaluate_selected_swap(
            self.objective_context, menu, action, number
        )
        achieved = attempt.achieved
        self.objective_attempts.append(attempt)
        self.pending_objective_selection = None
        if achieved:
            self.objective_suggestions = ()
            self.pending_action_menu = None
            self.objective_run = SimpleNamespace(
                context=self.objective_context,
                attempts=tuple(self.objective_attempts),
                achieved=True,
                termination_reason="target_achieved",
                final_ids=attempt.selected_ids,
                final_score=attempt.score,
            )
            self.objective_evidence = SimpleNamespace(key="O01")
        else:
            self.pending_action_menu = build_action_menu(
                self.objective_context, attempt.measurement, number
            )
            self.objective_suggestions = self.pending_action_menu.actions
        return attempt

    def request_synthesis(self):
        self.calls.append("synthesis")
        self.report = SimpleNamespace(evidence=())
        self.synthesis_prompt_appended = True
        if self.synthesis_failures: raise self.synthesis_failures.pop(0)
        self.session.turn_count += 1
        return SimpleNamespace(
            conclusion=SimpleNamespace(
                headline="Measured objective conclusion",
                sections=(SimpleNamespace(
                    theme="objective_driven_selection",
                    prose="The measured panel reached the bounded target. O01",
                ),),
            ),
            objective_run=self.objective_run,
            objective_evidence=self.objective_evidence,
        )


class EvidenceConclusionController(Controller):
    def __init__(self, finding_selection_status="selected"):
        super().__init__()
        self.finding_selection_status = finding_selection_status

    def execute_objective_attempt(self, selection):
        attempt = super().execute_objective_attempt(selection)
        if attempt.achieved:
            self.objective_run = terminal_objective_run(
                self.objective_context,
                tuple(self.objective_attempts),
                TerminationReason.TARGET_ACHIEVED,
            )
            self.objective_evidence = build_objective_evidence(self.objective_run)
        return attempt

    def request_synthesis(self):
        self.calls.append("synthesis")
        self.report = evidence_report()
        self.synthesis_prompt_appended = True
        if self.synthesis_failures:
            raise self.synthesis_failures.pop(0)
        snapshot = build_evidence_snapshot(self.report, self.objective_run)
        current_catalog = build_finding_catalog_from_snapshot(snapshot)
        if self.finding_selection_status == "selected":
            findings = tuple(
                next(
                    item for item in reversed(current_catalog.findings)
                    if item.theme == theme
                )
                for theme in demo_agent.CONCLUSION_THEMES
            )
        else:
            findings = tuple(
                next(item for item in current_catalog.findings if item.theme == theme)
                for theme in demo_agent.CONCLUSION_THEMES
            )
        conclusion = demo_agent.EvidenceControlledConclusion(
            evidence_snapshot=snapshot,
            measured_summary=snapshot.summary,
            ordered_findings=findings,
            finding_selection_status=self.finding_selection_status,
        )
        self.session.turn_count += 1
        return demo_agent.WorkflowResult(
            messages=(),
            report=self.report,
            plan=self.plan,
            conclusion=conclusion,
            stage_results=tuple(self.stage_results),
            turn_count=self.session.turn_count,
            objective_run=self.objective_run,
            objective_evidence=self.objective_evidence,
        )


def html_text(widget):
    values = []
    if isinstance(widget, widgets.HTML): values.append(widget.value)
    for child in getattr(widget, "children", ()): values.extend([html_text(child)])
    return " ".join(values)


def combined_html(widget):
    if isinstance(widget, widgets.HTML):
        return widget.value
    return " ".join(combined_html(child) for child in getattr(widget, "children", ()))


def production_menu(action_count):
    context = controlled_context_with_action_count(action_count)
    source = measure_panel(context, context.baseline_ids)
    menu = build_action_menu(context, source, 0)
    assert len(menu.actions) == action_count
    assert menu == build_action_menu(context, source, 0)
    return context, menu


def started(controller=None):
    controller = controller or Controller()
    workflow = InteractiveWorkflow(controller)
    workflow.start_button.click()
    return workflow, controller


def ready_interactive_workflow():
    return InteractiveWorkflow(Controller())


def completed_objective_workflow(monkeypatch):
    monkeypatch.setattr("interactive_workflow.objective_figures", lambda run, state: ())
    workflow = ready_interactive_workflow()
    workflow._show_objective_challenge()
    workflow._continue_objective_challenge()
    return workflow, workflow.controller


def evaluation_failed_workflow(monkeypatch):
    workflow = ready_interactive_workflow()
    controller = workflow.controller

    def fail_evaluation(selection):
        controller.calls.append(("objective_execute", selection))
        controller.objective_run = SimpleNamespace(
            context=controller.objective_context,
            attempts=tuple(controller.objective_attempts),
            achieved=False,
            termination_reason=TerminationReason.EVALUATION_NOT_COMPLETED,
            final_ids=controller.objective_context.baseline_ids,
            final_score=controller.objective_context.baseline_score,
        )
        controller.objective_evidence = SimpleNamespace(key="O01")
        controller.objective_failure_reason = TerminationReason.EVALUATION_NOT_COMPLETED
        controller.pending_action_menu = None
        controller.pending_objective_selection = None
        raise demo_agent.ObjectiveEvaluationError(
            "The objective evaluation was not completed; no attempt was accepted."
        )

    controller.execute_objective_attempt = fail_evaluation
    monkeypatch.setattr("interactive_workflow.objective_figures", lambda run, state: ())
    workflow._show_objective_challenge()
    workflow._continue_objective_challenge()
    return workflow, controller


def complete_six_stages(workflow):
    for _ in range(6):
        workflow.approve_button.click()


def run_objective(workflow):
    complete_six_stages(workflow)
    workflow.objective_button.click()


def evaluated_objective_records(candidate_ids=tuple(f"mol-{index}" for index in range(8))):
    distance = np.full((8, 8), 0.90, dtype=float)
    np.fill_diagonal(distance, 0.0)
    distance[0, 1] = distance[1, 0] = 0.35
    distance[2, 3] = distance[3, 2] = 0.45
    context = ObjectiveContext(
        candidates=tuple(
            ObjectiveCandidate(molecule_id, index, index, index)
            for index, molecule_id in enumerate(candidate_ids)
        ),
        baseline_ids=candidate_ids[:4],
        baseline_score=0.35,
        benchmark_score=0.90,
        target_score=0.35 + TARGET_FRACTION * (0.90 - 0.35),
        distance_matrix=distance,
    )
    baseline = measure_panel(context, context.baseline_ids)
    first_menu = build_action_menu(context, baseline, 0)
    first = evaluate_selected_swap(context, first_menu, accepted_maxima(first_menu)[0], 1)
    second_menu = build_action_menu(context, first.measurement, 1)
    second = evaluate_selected_swap(context, second_menu, accepted_maxima(second_menu)[0], 2)
    return context, first, second


def two_revision_decisions():
    context = two_revision_context()
    source = measure_panel(context, context.baseline_ids)
    records = []
    for attempt_number in (1, 2):
        menu = build_action_menu(context, source, attempt_number - 1)
        action = accepted_maxima(menu)[0]
        selection = ObjectiveSelection(
            state_id=menu.state_id,
            swap_id=action.swap_id,
            observed_limiting_pairs=[list(pair) for pair in menu.source.limiting_pairs],
            decision_rule="maximize_predicted_minimum_distance",
        )
        attempt = evaluate_selected_swap(
            context, menu, action, attempt_number
        )
        records.append((menu, selection, attempt))
        source = attempt.measurement
    return context, tuple(records)


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


@pytest.mark.parametrize("where", ["plan", "proposal", "execution", "objective", "synthesis"])
def test_unexpected_failures_are_generic_stopped_and_secret_safe(monkeypatch, where):
    controller = Controller(); secret = "SECRET-raw-exception"
    getattr(controller, f"{where}_failures").append(RuntimeError(secret))
    monkeypatch.setattr(demo_agent, "_display_conclusion", lambda result: None)
    workflow = InteractiveWorkflow(controller); workflow.start_button.click()
    if where == "execution": workflow.approve_button.click()
    elif where in {"objective", "synthesis"}:
        complete_six_stages(workflow)
        workflow.objective_button.click()
    assert workflow.status == "stopped" and workflow.retry_button is None
    assert secret not in workflow.transcript_text
    assert "local workflow error" in workflow.transcript_text.lower()


def test_six_stages_open_objective_challenge_then_gate_conclusion(monkeypatch):
    rendered = []; monkeypatch.setattr(demo_agent, "_display_conclusion", rendered.append)
    monkeypatch.setattr("interactive_workflow.objective_figures", lambda run, state: ())
    workflow, controller = started()
    for index in range(6):
        assert controller.calls.count("synthesis") == 0
        workflow.approve_button.click()
    assert controller.calls.count("synthesis") == 0
    assert workflow.status == "objective_ready"
    assert workflow.objective_button.description == "Run Objective Challenge"
    workflow.objective_button.click()
    assert controller.calls.count("synthesis") == 1
    assert workflow.status == "completed" and len(workflow.completed_cards) == 6
    assert rendered == []
    assert "Evidence-Backed Conclusion" in html_text(workflow.active_card)


def test_ready_workflow_invokes_show_objective_challenge_exactly_once(monkeypatch):
    workflow = ready_interactive_workflow()
    controller = workflow.controller
    calls = []
    original = workflow._show_objective_challenge

    def show_once():
        calls.append("show")
        original()

    monkeypatch.setattr(workflow, "_show_objective_challenge", show_once)
    workflow._show_objective_challenge()

    assert calls == ["show"]
    assert controller.calls.count("objective_begin") == 1
    assert controller.objective_attempts == []


def test_baseline_step_zero_is_measured_before_any_attempt(monkeypatch):
    workflow = ready_interactive_workflow()
    controller = workflow.controller
    workflow._show_objective_challenge()

    summary = workflow.objective_summary.value
    assert "Step 0" in summary
    assert "Measured baseline" in summary
    assert "Attempt 1" not in summary
    assert f"score_key={score_key(controller.objective_context.baseline_score)}" in summary
    assert controller.objective_attempts == []


def test_attempt_card_persists_complete_decision_ladder_without_model_rationale(monkeypatch):
    workflow, _controller = completed_objective_workflow(monkeypatch)

    first = combined_html(workflow.objective_attempt_cards[0])
    for label in ("Observe", "Candidate actions", "Nemotron choice", "Execute", "Measure"):
        assert label in first
    assert "select_next_panel_swap" in first
    assert "decision_basis" not in first


@pytest.mark.parametrize("action_count", [0, 1, 2, 3])
def test_action_menu_renders_exact_ordered_cards_and_explicit_empty_state(action_count):
    context, menu = production_menu(action_count)

    rendered = InteractiveWorkflow._objective_action_menu_html(context, menu)

    assert rendered.count("aria-label='Candidate action'") == action_count
    positions = [rendered.index(escape(action.swap_id)) for action in menu.actions]
    assert positions == sorted(positions)
    assert ("No legal improving candidate actions." in rendered) is (action_count == 0)
    if action_count:
        assert "Deterministic score" in rendered
        assert "Delta" in rendered
        assert "Resulting co-limiting pairs" in rendered
        assert "Target status" in rendered


@pytest.mark.parametrize(("first", "second", "improving"), BOUNDARY_CASES)
def test_score_comparison_uses_complete_shared_boundary_table(
    first, second, improving
):
    left, right, status = InteractiveWorkflow._score_comparison(first, second)

    if score_key(first) == score_key(second):
        assert status == "tied at 1e-12 decision precision"
        assert left == right
    else:
        assert improving is True
        assert float(left) > float(right)
        assert left != right


def test_attempt_row_rejects_forged_selection_and_committed_attempt():
    _context, menu = production_menu(3)
    action = accepted_maxima(menu)[0]
    selection = ObjectiveSelection(
        state_id=menu.state_id,
        swap_id=action.swap_id,
        observed_limiting_pairs=[list(pair) for pair in menu.source.limiting_pairs],
        decision_rule="maximize_predicted_minimum_distance",
    )
    attempt = evaluate_selected_swap(_context, menu, action, 1)

    with pytest.raises(ValueError):
        InteractiveWorkflow._objective_attempt_row(
            menu, selection.model_copy(update={"state_id": "state-0000000000000000"}), attempt
        )
    with pytest.raises(ValueError):
        InteractiveWorkflow._objective_attempt_row(
            menu, selection, replace(attempt, score=attempt.score - 0.01)
        )


def test_attempt_rows_bind_initial_and_revision_to_exact_menu_provenance():
    _context, (first, second) = two_revision_decisions()

    first_row = InteractiveWorkflow._objective_attempt_row(*first)
    second_row = InteractiveWorkflow._objective_attempt_row(*second)

    assert "Attempt 1" in first_row and "Attempt 2" in second_row
    assert first[0].source.selected_ids != second[0].source.selected_ids
    assert second[0].source == first[2].measurement


def test_revision_row_rejects_wrong_attempt_number_or_missing_selected_swap():
    _context, (_first, second) = two_revision_decisions()
    menu, selection, attempt = second

    with pytest.raises(ValueError):
        InteractiveWorkflow._objective_attempt_row(
            menu, selection, replace(attempt, attempt_number=1)
        )
    with pytest.raises(ValueError):
        InteractiveWorkflow._objective_attempt_row(
            menu, selection, replace(attempt, selected_swap=None)
        )


def test_attempt_row_uses_neutral_amber_and_green_for_unmeasured_miss_and_success():
    _context, (first, second) = two_revision_decisions()

    assert "#6c757d" in InteractiveWorkflow._objective_attempt_row(
        first[0], first[1], None
    )
    assert "#D68A00" in InteractiveWorkflow._objective_attempt_row(*first)
    assert "#76B900" in InteractiveWorkflow._objective_attempt_row(*second)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda attempt: replace(attempt, attempt_number=3),
        lambda attempt: replace(attempt, selected_ids=attempt.selected_swap.resulting_ids[::-1]),
        lambda attempt: replace(attempt, score=attempt.score - 0.01),
        lambda attempt: replace(attempt, score_key=attempt.score_key - 1),
        lambda attempt: replace(
            attempt, limiting_pairs=(("candidate-0", "candidate-9"),)
        ),
    ],
)
def test_attempt_row_rejects_forged_committed_transition_fields(mutation):
    _context, (first, _second) = two_revision_decisions()

    with pytest.raises(ValueError):
        InteractiveWorkflow._objective_attempt_row(
            first[0], first[1], mutation(first[2])
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda attempt: replace(attempt, constraints_passed=False),
        lambda attempt: replace(attempt, achieved=not attempt.achieved),
        lambda attempt: replace(attempt, score=float("nan")),
    ],
)
def test_attempt_row_rejects_false_domain_truth(mutation):
    _context, (first, _second) = two_revision_decisions()

    with pytest.raises(ValueError):
        InteractiveWorkflow._objective_attempt_row(
            first[0], first[1], mutation(first[2])
        )


@pytest.mark.parametrize("delta", [0.0, -0.10])
def test_attempt_row_rejects_nonpositive_action_delta_even_if_records_agree(delta):
    _context, (first, _second) = two_revision_decisions()
    menu, selection, attempt = first
    selected = attempt.selected_swap
    forged_action = replace(
        selected,
        predicted_score=menu.source.score + delta,
        predicted_score_key=score_key(menu.source.score + delta),
        score_delta=delta,
    )
    forged_menu = replace(
        menu,
        actions=tuple(
            forged_action if action.swap_id == selected.swap_id else action
            for action in menu.actions
        ),
    )
    forged_attempt = replace(
        attempt,
        selected_swap=forged_action,
        score=forged_action.predicted_score,
        score_key=forged_action.predicted_score_key,
        achieved=False,
    )

    with pytest.raises(ValueError):
        InteractiveWorkflow._objective_attempt_row(
            forged_menu, selection, forged_attempt
        )


@pytest.mark.parametrize(
    "pair",
    [
        ("candidate-9", "candidate-0"),
        ("candidate-0", "candidate-0"),
        ("", "candidate-0"),
    ],
)
def test_attempt_row_rejects_invalid_source_limiting_pairs(pair):
    _context, (first, _second) = two_revision_decisions()
    menu, selection, attempt = first
    forged_source = replace(menu.source, limiting_pairs=(pair,))
    forged_menu = replace(menu, source=forged_source)
    forged_selection = selection.model_copy(
        update={"observed_limiting_pairs": [list(pair)]}
    )

    with pytest.raises(ValueError):
        InteractiveWorkflow._objective_attempt_row(
            forged_menu, forged_selection, attempt
        )


@pytest.mark.parametrize(
    "pair",
    [
        ("candidate-9", "candidate-2"),
        ("candidate-2", "candidate-2"),
        ("", "candidate-2"),
    ],
)
def test_attempt_row_rejects_invalid_resulting_limiting_pairs(pair):
    _context, (first, _second) = two_revision_decisions()
    menu, selection, attempt = first
    selected = attempt.selected_swap
    forged_action = replace(
        selected, limiting_pair=pair, limiting_pairs=(pair,)
    )
    forged_menu = replace(
        menu,
        actions=tuple(
            forged_action if action.swap_id == selected.swap_id else action
            for action in menu.actions
        ),
    )
    forged_attempt = replace(
        attempt,
        selected_swap=forged_action,
        limiting_pair=pair,
        limiting_pairs=(pair,),
    )

    with pytest.raises(ValueError):
        InteractiveWorkflow._objective_attempt_row(
            forged_menu, selection, forged_attempt
        )


def test_attempt_row_rejects_objective_swap_subclasses():
    _context, (first, _second) = two_revision_decisions()
    menu, selection, attempt = first

    class SwapSubclass(ObjectiveSwap):
        pass

    subclass = SwapSubclass(**attempt.selected_swap.__dict__)
    forged_menu = replace(
        menu,
        actions=tuple(
            subclass if action.swap_id == subclass.swap_id else action
            for action in menu.actions
        ),
    )

    with pytest.raises(ValueError):
        InteractiveWorkflow._objective_attempt_row(
            forged_menu,
            selection,
            replace(attempt, selected_swap=subclass),
        )


def test_attempt_row_rejects_nonmaximal_menu_choice():
    _context, menu = production_menu(3)
    maximum = accepted_maxima(menu)[0]
    canonical_attempt = evaluate_selected_swap(_context, menu, maximum, 1)
    lower = next(
        action for action in menu.actions if action not in accepted_maxima(menu)
    )
    selection = ObjectiveSelection(
        state_id=menu.state_id,
        swap_id=lower.swap_id,
        observed_limiting_pairs=[list(pair) for pair in menu.source.limiting_pairs],
        decision_rule="maximize_predicted_minimum_distance",
    )
    forged_attempt = replace(
        canonical_attempt,
        selected_swap=lower,
        selected_ids=lower.resulting_ids,
        score=lower.predicted_score,
        score_key=lower.predicted_score_key,
        limiting_pair=lower.limiting_pair,
        limiting_pairs=lower.limiting_pairs,
        achieved=lower.target_status == "meets_target",
    )

    with pytest.raises(ValueError):
        InteractiveWorkflow._objective_attempt_row(
            menu, selection, forged_attempt
        )


def test_score_comparison_preserves_near_boundary_order_without_rounding_it_away():
    left, right, status = InteractiveWorkflow._score_comparison(
        0.5000000000005, 0.5
    )

    assert left != right
    assert float(left) > float(right)
    assert status == "above at 1e-12 decision precision"


def test_score_comparison_collapses_raw_difference_inside_one_decision_key():
    left, right, status = InteractiveWorkflow._score_comparison(
        0.5000000000004, 0.5
    )

    assert left == right == "0.500000000000"
    assert status == "tied at 1e-12 decision precision"


def test_no_legal_step_zero_terminal_renders_empty_menu_and_truthful_reason(monkeypatch):
    controller = Controller()

    def begin_no_legal():
        controller.calls.append("objective_begin")
        controller.report = SimpleNamespace(evidence=())
        controller.objective_prompt_appended = True
        controller.objective_context = controlled_context_without_improving_swaps()
        source = measure_panel(
            controller.objective_context, controller.objective_context.baseline_ids
        )
        controller.pending_action_menu = build_action_menu(
            controller.objective_context, source, 0
        )
        controller.objective_run = SimpleNamespace(
            context=controller.objective_context,
            attempts=(),
            achieved=False,
            termination_reason=TerminationReason.NO_LEGAL_IMPROVING_SWAP,
            final_ids=source.selected_ids,
            final_score=source.score,
        )
        controller.objective_evidence = SimpleNamespace(key="O01")
        return controller.objective_context

    controller.begin_objective_challenge = begin_no_legal
    monkeypatch.setattr("interactive_workflow.objective_figures", lambda run, state: ())
    workflow = InteractiveWorkflow(controller)

    workflow._show_objective_challenge()

    summary = workflow.objective_summary.value
    assert "No legal improving swap" in summary
    assert "No legal improving candidate actions." in summary
    assert "attempt limit" not in summary.lower()


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        (TerminationReason.TARGET_ACHIEVED, "target achieved"),
        (TerminationReason.BASELINE_ALREADY_OPTIMAL, "Baseline already optimal"),
        (TerminationReason.ATTEMPT_LIMIT_REACHED, "Attempt limit reached"),
        (TerminationReason.NO_LEGAL_IMPROVING_SWAP, "No legal improving swap"),
        (TerminationReason.OBJECTIVE_CORRECTION_LIMIT, "correction limit reached"),
        (TerminationReason.OBJECTIVE_PROVIDER_FAILURE, "provider failure"),
        (TerminationReason.EVALUATION_NOT_COMPLETED, "Evaluation not completed"),
    ],
)
def test_every_terminal_reason_has_an_explicit_truthful_label(reason, expected):
    context = controlled_context_without_improving_swaps()
    run = SimpleNamespace(
        achieved=reason is TerminationReason.TARGET_ACHIEVED,
        termination_reason=reason,
    )

    summary = InteractiveWorkflow._objective_summary_html(context, (), run)

    assert expected in summary


def test_step_zero_uses_visible_baseline_target_boundary_precision():
    context = replace(
        controlled_context_without_improving_swaps(),
        baseline_score=0.5,
        target_score=0.500000000001,
    )

    summary = InteractiveWorkflow._objective_summary_html(context, ())

    target_headline = summary.split("<p><b>Target:</b>", 1)[1].split("</p>", 1)[0]
    baseline_strip = summary.split("<b>Baseline</b>", 1)[1].split("</span>", 1)[0]
    assert "0.500000000001" in target_headline
    assert "0.500000000000" in baseline_strip
    assert "0.500000000000" not in target_headline
    assert "0.500000000001" not in baseline_strip
    assert "below at 1e-12 decision precision" in summary


def test_summary_rebuilds_menu_and_rejects_self_consistent_forged_target_truth():
    context, (first, _second) = two_revision_decisions()
    menu, selection, attempt = first
    selected = attempt.selected_swap
    forged_action = replace(selected, target_status="meets_target")
    forged_menu = replace(
        menu,
        actions=tuple(
            forged_action if action.swap_id == selected.swap_id else action
            for action in menu.actions
        ),
    )
    forged_attempt = replace(
        attempt, selected_swap=forged_action, achieved=True
    )

    with pytest.raises(ValueError):
        InteractiveWorkflow._objective_summary_html(
            context, ((forged_menu, selection, forged_attempt),)
        )


def test_attempt_and_menu_rows_escape_every_molecule_and_state_value():
    context, first, _second = evaluated_objective_records(
        ("mol<0>", "mol-1", "mol-2", "mol-3", "mol&4", "mol-5", "mol-6", "mol-7")
    )
    menu = build_action_menu(context, measure_panel(context, context.baseline_ids), 0)
    selection = ObjectiveSelection(
        state_id=menu.state_id,
        swap_id=first.selected_swap.swap_id,
        observed_limiting_pairs=[list(pair) for pair in menu.source.limiting_pairs],
        decision_rule="maximize_predicted_minimum_distance",
    )

    rendered = (
        InteractiveWorkflow._objective_action_menu_html(context, menu)
        + InteractiveWorkflow._objective_attempt_row(menu, selection, first)
    )

    assert "mol<0>" not in rendered
    assert "mol&4" not in rendered
    assert "mol&lt;0&gt;" in rendered
    assert "mol&amp;4" in rendered
    assert escape(menu.state_id) in rendered


def test_evaluation_failure_persists_neutral_unmeasured_selection_and_never_success(monkeypatch):
    workflow, controller = evaluation_failed_workflow(monkeypatch)

    assert controller.objective_run.termination_reason is TerminationReason.EVALUATION_NOT_COMPLETED
    assert controller.pending_action_menu is None
    assert controller.pending_objective_selection is None
    rendered = combined_html(workflow.objective_card)
    assert "Evaluation not completed" in rendered
    assert "validated but unmeasured" in rendered
    assert "executed" not in rendered.lower()
    assert "success" not in rendered.lower()
    assert "Goal achieved" not in rendered
    assert controller.calls.count("objective_proposal") == 1
    assert len([call for call in controller.calls if isinstance(call, tuple) and call[0] == "objective_execute"]) == 1


def test_conclusion_factual_text_persists_in_html_value(monkeypatch):
    workflow, _controller = completed_objective_workflow(monkeypatch)

    assert "Schema-checked scientific conclusion" in combined_html(workflow.active_card)
    assert "Measured objective conclusion" in combined_html(workflow.active_card)
    assert not any(isinstance(child, widgets.Output) for child in workflow.active_card.children)


@pytest.mark.parametrize(
    "finding_selection_status",
    ["selected", "finding_selection_unavailable"],
)
def test_completed_real_workflow_root_survives_embed_with_ladder_conclusion_and_png(
    monkeypatch, tmp_path, finding_selection_status
):
    from matplotlib.figure import Figure

    controller = EvidenceConclusionController(finding_selection_status)
    matplotlib_figure = Figure(figsize=(1, 1))
    matplotlib_figure.subplots().plot([0, 1], [1, 0])
    monkeypatch.setattr(
        "interactive_workflow.objective_figures",
        lambda run, state: (
            PILImage.new("RGB", (4, 3), "green"),
            matplotlib_figure,
        ),
    )
    workflow, _ = started(controller)
    complete_six_stages(workflow)
    workflow.objective_button.click()

    assert workflow.status == "completed"
    assert isinstance(workflow.workflow_result, demo_agent.WorkflowResult)
    assert isinstance(workflow.workflow_result.conclusion, demo_agent.EvidenceControlledConclusion)
    target = tmp_path / "embedded.html"
    embed_minimal_html(
        target,
        views=[workflow.root],
        title="Evidence conclusion",
        state=dependency_state([workflow.root]),
    )
    rendered = target.read_text(encoding="utf-8")
    for text in (
        "Step 0",
        "Candidate actions",
        "Nemotron choice",
        "Evidence-Backed Conclusion",
    ):
        assert text in rendered
    assert rendered.count("iVBOR") == 2
    if finding_selection_status == "selected":
        assert rendered.count("agent-selected evidence emphasis") == 7
        assert "agent-selected emphasis unavailable" not in rendered
    else:
        assert "agent-selected emphasis unavailable" in rendered
        assert rendered.count("deterministic fallback finding") == 7
        assert "agent-selected evidence emphasis" not in rendered


def test_unavailable_conclusion_never_claims_agent_selection(monkeypatch):
    report, run = report_and_run()
    snapshot = build_evidence_snapshot(report, run)
    catalog = build_finding_catalog_from_snapshot(snapshot)
    findings = tuple(
        next(item for item in catalog.findings if item.theme == theme)
        for theme in demo_agent.CONCLUSION_THEMES
    )
    conclusion = demo_agent.EvidenceControlledConclusion(
        snapshot, snapshot.summary, findings, "finding_selection_unavailable"
    )
    workflow = ready_interactive_workflow()
    workflow._render_workflow_result(conclusion)

    rendered = combined_html(workflow.active_card)
    assert "agent-selected emphasis unavailable" in rendered
    assert rendered.count("deterministic fallback finding") == 7
    assert "agent-selected evidence emphasis" not in rendered


def test_selected_single_catalog_option_is_labeled_required_measured_finding(monkeypatch):
    report, run = report_and_run()
    snapshot = build_evidence_snapshot(report, run)
    complete_catalog = build_finding_catalog_from_snapshot(snapshot)
    single_theme = demo_agent.CONCLUSION_THEMES[0]
    retained = []
    for finding in complete_catalog.findings:
        if finding.theme != single_theme or not any(
            prior.theme == single_theme for prior in retained
        ):
            retained.append(finding)
    reduced_catalog = FindingCatalog(tuple(retained))
    findings = tuple(
        next(item for item in reduced_catalog.findings if item.theme == theme)
        for theme in demo_agent.CONCLUSION_THEMES
    )
    monkeypatch.setattr(
        demo_agent,
        "build_finding_catalog_from_snapshot",
        lambda supplied: reduced_catalog,
    )
    conclusion = demo_agent.EvidenceControlledConclusion(
        snapshot, snapshot.summary, findings, "selected"
    )

    workflow = ready_interactive_workflow()
    workflow._render_workflow_result(conclusion)
    rendered = combined_html(workflow.active_card)
    assert rendered.count("required measured finding") == 1
    assert rendered.count("agent-selected evidence emphasis") == 6


def test_measured_summary_table_is_complete_deterministic_and_precedes_findings():
    report, run = report_and_run()
    snapshot = build_evidence_snapshot(report, run)
    current_catalog = build_finding_catalog_from_snapshot(snapshot)
    canonical = tuple(
        next(item for item in current_catalog.findings if item.theme == theme)
        for theme in demo_agent.CONCLUSION_THEMES
    )
    alternate = tuple(
        next(
            item for item in reversed(current_catalog.findings)
            if item.theme == theme
        )
        for theme in demo_agent.CONCLUSION_THEMES
    )
    conclusions = (
        demo_agent.EvidenceControlledConclusion(
            snapshot, snapshot.summary, canonical, "finding_selection_unavailable"
        ),
        demo_agent.EvidenceControlledConclusion(
            snapshot, snapshot.summary, canonical, "selected"
        ),
        demo_agent.EvidenceControlledConclusion(
            snapshot, snapshot.summary, alternate, "selected"
        ),
    )
    tables = []
    for conclusion in conclusions:
        workflow = ready_interactive_workflow()
        workflow._render_workflow_result(conclusion)
        rendered = combined_html(workflow.active_card)
        table = re.search(
            r'<table class="measured-summary">.*?</table>', rendered, re.DOTALL
        ).group(0)
        tables.append(table)
        assert rendered.index(table) < rendered.index(conclusion.ordered_findings[0].text)

    assert tables[0] == tables[1] == tables[2]
    summary = snapshot.summary
    limiting = "; ".join(
        f"{first} / {second}: Tanimoto {similarity}"
        for (first, second), similarity in zip(
            summary.limiting_pairs, summary.limiting_similarities, strict=True
        )
    )
    for field in fields(summary):
        value = getattr(summary, field.name)
        row = re.search(
            rf'<tr data-field="{field.name}">.*?</tr>', tables[0], re.DOTALL
        )
        assert row is not None, field.name
        assert escape(str(value)) in row.group(0), field.name
    assert "within molecule among converged sampled conformers" in tables[0]
    assert limiting in tables[0]


def test_render_revalidates_the_entire_retained_conclusion_container():
    report, run = report_and_run()
    snapshot = build_evidence_snapshot(report, run)
    catalog = build_finding_catalog_from_snapshot(snapshot)
    findings = tuple(
        next(item for item in catalog.findings if item.theme == theme)
        for theme in demo_agent.CONCLUSION_THEMES
    )
    conclusion = demo_agent.EvidenceControlledConclusion(
        snapshot, snapshot.summary, findings, "finding_selection_unavailable"
    )
    object.__setattr__(
        conclusion,
        "measured_summary",
        replace(snapshot.summary, headline="Forged success headline"),
    )

    with pytest.raises(ValueError, match="snapshot"):
        ready_interactive_workflow()._render_workflow_result(conclusion)


def test_objective_card_retains_attempts_receipts_scores_and_limiting_pairs(monkeypatch):
    monkeypatch.setattr(demo_agent, "_display_conclusion", lambda result: None)
    monkeypatch.setattr("interactive_workflow.objective_figures", lambda run, state: ())
    workflow, controller = started()

    run_objective(workflow)

    text = " ".join(html_text(card) for card in workflow.objective_attempt_cards)
    assert len(workflow.objective_attempt_cards) == 2
    assert "Validated Nemotron selection" in text
    assert "Planned deterministic command" in text
    assert "Executed measurement" in text
    assert "Evaluation executed by Python" in text
    assert "select_next_panel_swap" in text
    assert "evaluate_selected_swap" in text
    assert "decision_basis" not in text and "Decision summary" not in text
    assert "D_min" in text and "co-limiting pairs" in text
    assert "Revise" in text and "Goal achieved" in text
    assert "Baseline" in workflow.objective_summary.value
    assert "Attempt 1" in workflow.objective_summary.value
    assert "Attempt 2" in workflow.objective_summary.value
    assert [attempt.score for attempt in controller.objective_attempts] == [0.40, 0.90]


def test_objective_attempts_render_as_observe_act_measure_decision_ladder(monkeypatch):
    monkeypatch.setattr(demo_agent, "_display_conclusion", lambda result: None)
    monkeypatch.setattr("interactive_workflow.objective_figures", lambda run, state: ())
    workflow, _controller = started()

    run_objective(workflow)

    summary = workflow.objective_summary.value
    details = " ".join(html_text(card) for card in workflow.objective_attempt_cards)
    assert "Observe" in summary
    assert "Candidate actions" in summary
    assert "Nemotron choice" in summary
    assert "Execute" in summary
    assert "Measure" in summary
    assert "Outcome" in summary
    assert "candidate-0-&gt;candidate-4" in summary
    assert "state-" in summary
    assert "<b>Outcome:</b> Revise" in summary
    assert "<b>Outcome:</b> Goal achieved" in summary
    assert "#D68A00" in summary
    assert "#76B900" in summary
    assert "Planned deterministic command" in details
    assert "select_next_panel_swap(" in details
    assert "Goal achieved" in details


def test_objective_receipt_render_failure_after_commit_never_reexecutes(monkeypatch):
    workflow, controller = started()
    complete_six_stages(workflow)
    reached = []

    real_receipt = objective_receipt

    def fail_receipt(*args, **kwargs):
        reached.append("render")
        if len(reached) == 1:
            raise RuntimeError("injected receipt render failure")
        return real_receipt(*args, **kwargs)

    monkeypatch.setattr("interactive_workflow.objective_receipt", fail_receipt)
    workflow.objective_button.click()
    workflow.objective_button.click()

    assert len(reached) == 4
    assert len(controller.objective_attempts) == 2
    assert len([
        call for call in controller.calls
        if isinstance(call, tuple) and call[0] == "objective_execute"
    ]) == 2
    assert workflow.status == "completed"


def test_known_objective_proposal_failure_after_measured_attempt_has_one_guarded_retry(monkeypatch):
    monkeypatch.setattr(demo_agent, "_display_conclusion", lambda result: None)
    monkeypatch.setattr("interactive_workflow.objective_figures", lambda run, state: ())
    controller = Controller()
    original_request = controller.request_objective_attempt
    failed_once = False

    def fail_second_request_once():
        nonlocal failed_once
        if len(controller.objective_attempts) == 1 and not failed_once:
            failed_once = True
            controller.objective_failures.append(connect_error())
            return original_request()
        return original_request()

    controller.request_objective_attempt = fail_second_request_once
    workflow, _ = started(controller)
    complete_six_stages(workflow)
    workflow.objective_button.click()

    assert workflow.status == "objective_failed"
    assert len(controller.objective_attempts) == 1
    assert len(workflow.objective_attempt_cards) == 1
    retry = workflow.retry_button
    assert retry.description == "Retry Objective Proposal"
    retry.click()
    retry.click()

    assert retry.disabled
    assert workflow.status == "completed"
    assert len(controller.objective_attempts) == 2
    assert controller.calls.count("objective_proposal") == 3


def test_objective_retry_invariant_counts_rejected_hosted_responses(monkeypatch):
    monkeypatch.setattr(demo_agent, "_display_conclusion", lambda result: None)
    monkeypatch.setattr("interactive_workflow.objective_figures", lambda run, state: ())
    controller = Controller()
    original_request = controller.request_objective_attempt
    failed_once = False

    def fail_second_request_after_one_correction():
        nonlocal failed_once
        if len(controller.objective_attempts) == 1 and not failed_once:
            failed_once = True
            controller.objective_rejection_count = 1
            controller.session.turn_count += 1
            controller.objective_failures.append(connect_error())
            return original_request()
        return original_request()

    controller.request_objective_attempt = fail_second_request_after_one_correction
    workflow, _ = started(controller)
    complete_six_stages(workflow)
    workflow.objective_button.click()

    assert workflow.status == "objective_failed"
    assert workflow.retry_button.description == "Retry Objective Proposal"


@pytest.mark.parametrize("accepted_attempt_count", [0, 1])
def test_objective_correction_exhaustion_stops_explicitly_without_fabricating_result(
    accepted_attempt_count,
):
    controller = Controller()
    original_request = controller.request_objective_attempt
    summary_before_stop = []

    def exhaust_corrections():
        if len(controller.objective_attempts) < accepted_attempt_count:
            return original_request()
        summary_before_stop.append(workflow.objective_summary.value)
        controller.objective_rejection_count = demo_agent.MAX_OBJECTIVE_CORRECTIONS
        controller.session.turn_count += demo_agent.MAX_OBJECTIVE_CORRECTIONS
        controller.calls.append("objective_proposal")
        raise demo_agent.ObjectiveCorrectionLimitError(
            "The objective correction limit was reached. <do-not-render-as-html>"
        )

    controller.request_objective_attempt = exhaust_corrections
    workflow, _ = started(controller)
    complete_six_stages(workflow)
    completed_cards = workflow.completed_cards
    stage_results = tuple(controller.stage_results)
    objective_card = workflow.objective_card
    workflow.objective_button.click()

    assert workflow.status == "objective_stopped"
    assert workflow.retry_button is None
    assert workflow.objective_button.disabled
    assert workflow.completed_cards == completed_cards
    assert tuple(controller.stage_results) == stage_results
    assert workflow.objective_card is objective_card
    assert workflow.objective_summary.value == summary_before_stop[0]
    assert len(workflow.objective_attempt_cards) == accepted_attempt_count
    assert len(controller.objective_attempts) == accepted_attempt_count
    assert controller.objective_run is None
    assert controller.objective_evidence is None
    assert controller.calls.count("synthesis") == 0
    expected_count = f"Accepted scientific attempts: {accepted_attempt_count}."
    explicit_stop = "No additional scientific attempt was executed."
    rendered = html_text(workflow.objective_card)
    assert expected_count in rendered and expected_count in workflow.transcript_text
    assert explicit_stop in rendered and explicit_stop in workflow.transcript_text
    assert "The objective correction limit was reached." in rendered
    assert "&lt;do-not-render-as-html&gt;" in rendered
    assert "<do-not-render-as-html>" not in rendered
    assert "local workflow error" not in workflow.transcript_text.lower()
    assert "O01" not in rendered and "O01" not in workflow.transcript_text


@pytest.mark.parametrize(
    "failure_message",
    [
        "The objective evaluator rejected the proposed panel.",
        "The legal improving objective swaps could not be ranked.",
        "The objective run could not be finalized safely.",
        "The objective protocol result could not be appended safely.",
    ],
)
def test_nonretryable_objective_tool_errors_remain_generic_without_false_receipt(
    failure_message,
):
    controller = Controller()

    def fail_objective_execution(proposal):
        controller.calls.append(("objective_execute", proposal))
        raise ToolCallError(failure_message)

    controller.execute_objective_attempt = fail_objective_execution
    workflow, _ = started(controller)
    complete_six_stages(workflow)
    workflow.objective_button.click()

    assert workflow.status == "stopped"
    assert workflow.retry_button is None
    assert controller.objective_attempts == []
    assert controller.objective_run is None
    assert controller.objective_evidence is None
    assert controller.calls.count("synthesis") == 0
    assert failure_message not in workflow.transcript_text
    assert "local workflow error" in workflow.transcript_text.lower()
    assert "Accepted scientific attempts:" not in workflow.transcript_text
    assert "No additional scientific attempt was executed." not in workflow.transcript_text


def test_retry_objective_rechecks_pending_state_and_stops(monkeypatch):
    controller = Controller()
    original_request = controller.request_objective_attempt
    failed_once = False

    def fail_second_request_once():
        nonlocal failed_once
        if len(controller.objective_attempts) == 1 and not failed_once:
            failed_once = True
            controller.objective_failures.append(connect_error())
            return original_request()
        return original_request()

    controller.request_objective_attempt = fail_second_request_once
    workflow, _ = started(controller)
    complete_six_stages(workflow)
    workflow.objective_button.click()
    before = controller.calls.count("objective_proposal")
    controller.pending_objective_selection = ObjectiveSelection(
        state_id=controller.pending_action_menu.state_id,
        swap_id=controller.pending_action_menu.actions[0].swap_id,
        observed_limiting_pairs=[
            list(pair) for pair in controller.pending_action_menu.source.limiting_pairs
        ],
        decision_rule="maximize_predicted_minimum_distance",
    )
    workflow.retry_button.click()

    assert workflow.status == "stopped"
    assert workflow.retry_button is None
    assert controller.calls.count("objective_proposal") == before


@pytest.mark.parametrize("mutation", ["invalid_menu", "missing_suggestions"])
def test_retry_objective_rejects_noncanonical_guidance_state(mutation):
    controller = Controller()
    original_request = controller.request_objective_attempt
    failed_once = False

    def fail_second_request_once():
        nonlocal failed_once
        if len(controller.objective_attempts) == 1 and not failed_once:
            failed_once = True
            controller.objective_failures.append(connect_error())
            return original_request()
        return original_request()

    controller.request_objective_attempt = fail_second_request_once
    workflow, _ = started(controller)
    complete_six_stages(workflow)
    workflow.objective_button.click()
    assert workflow.status == "objective_failed"
    before = controller.calls.count("objective_proposal")
    if mutation == "invalid_menu":
        controller.pending_action_menu = object()
    else:
        controller.objective_suggestions = ()

    workflow.retry_button.click()

    assert workflow.status == "stopped"
    assert workflow.retry_button is None
    assert controller.calls.count("objective_proposal") == before


def test_real_connect_error_offers_exactly_one_initial_objective_retry(monkeypatch):
    monkeypatch.setattr(demo_agent, "_display_conclusion", lambda result: None)
    monkeypatch.setattr("interactive_workflow.objective_figures", lambda run, state: ())
    controller = Controller()
    controller.objective_failures.append(connect_error())
    workflow, _ = started(controller)
    complete_six_stages(workflow)
    workflow.objective_button.click()

    assert controller.objective_attempts == []
    assert controller.objective_suggestions == ()
    assert workflow.status == "objective_failed"
    assert controller.objective_transport_retry_pending is True
    retry = workflow.retry_button
    assert retry.description == "Retry Objective Proposal"

    retry.click()
    retry.click()

    assert retry.disabled
    assert workflow.status == "completed"
    assert controller.calls.count("objective_proposal") == 3
    assert controller.objective_transport_retry_pending is False


@pytest.mark.parametrize(
    "failure",
    [
        auth_error(),
        ToolCallError("generic non-transport objective failure"),
    ],
)
def test_auth_and_nontransport_objective_failures_offer_no_retry(failure):
    controller = Controller()
    controller.objective_failures.append(failure)
    workflow, _ = started(controller)
    complete_six_stages(workflow)

    workflow.objective_button.click()

    assert workflow.status == "stopped"
    assert workflow.retry_button is None
    assert controller.calls.count("objective_proposal") == 1
    assert controller.objective_transport_retry_pending is False
    assert controller.objective_attempts == []
    assert controller.objective_run is None
    assert controller.objective_evidence is None


def test_known_synthesis_failure_has_guarded_retry(monkeypatch):
    monkeypatch.setattr(demo_agent, "_display_conclusion", lambda result: None)
    controller = Controller(); controller.synthesis_failures.append(ToolCallError("hosted synthesis failed"))
    workflow, _ = started(controller)
    complete_six_stages(workflow)
    workflow.objective_button.click()
    assert workflow.status == "synthesis_failed"
    retry = workflow.retry_button; retry.click(); retry.click()
    assert workflow.status == "completed" and retry.disabled
    assert controller.calls.count("synthesis") == 2


def test_synthesis_failure_at_extended_objective_turn_bound_remains_retryable(monkeypatch):
    monkeypatch.setattr(demo_agent, "_display_conclusion", lambda result: None)
    controller = Controller()
    original_synthesis = controller.request_synthesis
    failed_once = False

    def fail_once_at_turn_twelve():
        nonlocal failed_once
        if not failed_once:
            failed_once = True
            controller.calls.append("synthesis")
            controller.report = SimpleNamespace(evidence=())
            controller.synthesis_prompt_appended = True
            controller.session.turn_count = demo_agent.MAX_OBJECTIVE_HOSTED_TURNS
            raise ToolCallError("hosted synthesis failed")
        return original_synthesis()

    controller.request_synthesis = fail_once_at_turn_twelve
    workflow, _ = started(controller)
    complete_six_stages(workflow)
    workflow.objective_button.click()

    assert workflow.status == "synthesis_failed"
    assert workflow.retry_button.description == "Retry Synthesis"


def test_retry_synthesis_rechecks_turn_phase_and_pending(monkeypatch):
    monkeypatch.setattr(demo_agent, "_display_conclusion", lambda result: None)
    controller = Controller(); controller.synthesis_failures.append(ToolCallError("hosted synthesis failed"))
    workflow, _ = started(controller)
    complete_six_stages(workflow)
    workflow.objective_button.click()
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
    complete_six_stages(workflow)
    workflow.objective_button.click()
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
    complete_six_stages(workflow)
    workflow.objective_button.click()
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
    assert created == [(("goal", "nvapi-test"), {"skill": "s", "client": "c", "executors": {}, "objective_required": True})]
    assert displayed == [workflow] and controller.calls == []
    assert str(inspect.signature(launch_interactive_workflow)).startswith("(user_goal: 'str', api_key: 'str', *")
    with pytest.raises(TypeError): launch_interactive_workflow("goal", "nvapi-test", "positional")


def test_figure_render_failure_keeps_completed_result_and_advances(monkeypatch):
    controller = Controller(); controller.figures = (object(),)
    monkeypatch.setattr(demo_agent, "_display_figure", lambda figure: (_ for _ in ()).throw(RuntimeError("FIGURE-SECRET")))
    workflow, _ = started(controller)
    workflow.approve_button.click()
    assert len(workflow.completed_cards) == 1 and len(workflow.completed_results) == 1
    assert workflow.status == "awaiting_approval"
    assert controller.calls.count("proposal") == 2
    assert len([call for call in controller.calls if isinstance(call, tuple)]) == 1
    assert "Figure unavailable in this notebook frontend." in html_text(workflow.completed_cards[0])
    assert "FIGURE-SECRET" not in workflow.transcript_text


def test_conclusion_render_failure_retains_result_and_completion(monkeypatch):
    controller = Controller()
    original = InteractiveWorkflow._render_workflow_result
    monkeypatch.setattr(
        InteractiveWorkflow,
        "_render_workflow_result",
        lambda self, conclusion: (_ for _ in ()).throw(RuntimeError("CONCLUSION-SECRET")),
    )
    workflow, _ = started(controller)
    complete_six_stages(workflow)
    workflow.objective_button.click()
    assert workflow.status == "completed" and workflow.workflow_result is not None
    assert workflow.retry_button is None and controller.calls.count("synthesis") == 1
    assert "Conclusion rendering unavailable" in html_text(workflow.active_card)
    assert "CONCLUSION-SECRET" not in workflow.transcript_text
    monkeypatch.setattr(InteractiveWorkflow, "_render_workflow_result", original)
    rebuilt = workflow.reconstruct_completed_view()
    assert "Schema-checked scientific conclusion" in html_text(rebuilt)
    assert controller.calls.count("synthesis") == 1


def test_completed_control_observers_are_detached():
    workflow, _ = started()
    workflow.approve_button.click()
    old_controls = dict(workflow.controls)
    workflow.approve_button.click()
    old_card = workflow.completed_cards[1]
    before = (html_text(old_card), html_text(workflow.active_card), workflow.transcript_text)
    old_controls["radius"].value = 3
    assert (html_text(old_card), html_text(workflow.active_card), workflow.transcript_text) == before
