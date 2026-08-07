import inspect
from dataclasses import replace
from types import SimpleNamespace

import httpx
import ipywidgets as widgets
import numpy as np
import pytest
from openai import AuthenticationError

import demo_agent
from chemistry_workflow import StageResult
from demo_agent import (
    ClusterArgs, EmbedArgs, FingerprintArgs, InspectionArgs, OptimizationArgs,
    ObjectiveSelection, SimilarityArgs, StageProposal, ToolCallError,
)
from objective_challenge import (
    ObjectiveAttempt, ObjectiveCandidate, ObjectiveContext, ObjectiveSwap,
    TARGET_FRACTION, accepted_maxima, build_action_menu, evaluate_selected_swap,
    measure_panel,
)
from interactive_workflow import InteractiveWorkflow, controls_for


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
        self.pending_objective = None
        self.pending_objective_swap = None
        self.objective_attempts = []
        self.objective_suggestions = ()
        self.objective_rejection_count = 0
        self.objective_run = None
        self.objective_evidence = None
        self.objective_prompt_appended = False
        self._objective_transport_retry_pending = False
        self.objective_proposals = []
        self._objective_expected_attempts = []
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
        self.objective_context, first, second = evaluated_objective_records()
        self._objective_expected_attempts = [first, second]
        current = measure_panel(self.objective_context, self.objective_context.baseline_ids)
        self.objective_proposals = []
        for attempt in self._objective_expected_attempts:
            menu = build_action_menu(
                self.objective_context, current, len(self.objective_proposals)
            )
            self.objective_proposals.append(ObjectiveSelection(
                state_id=menu.state_id,
                swap_id=attempt.selected_swap.swap_id,
                observed_limiting_pairs=[list(pair) for pair in menu.source.limiting_pairs],
                decision_rule="maximize_predicted_minimum_distance",
            ))
            current = attempt.measurement
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
        self.pending_objective = self.objective_proposals[len(self.objective_attempts)]
        self.session.turn_count += 1
        return self.pending_objective

    @property
    def objective_transport_retry_pending(self):
        return self._objective_transport_retry_pending

    def execute_objective_attempt(self, proposal):
        self.calls.append(("objective_execute", proposal))
        number = len(self.objective_attempts) + 1
        attempt = self._objective_expected_attempts[number - 1]
        achieved = attempt.achieved
        self.objective_attempts.append(attempt)
        self.pending_objective = None
        if achieved:
            self.objective_suggestions = ()
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
            self.objective_suggestions = ("listed-revision",)
        return attempt

    def request_synthesis(self):
        self.calls.append("synthesis")
        self.report = SimpleNamespace(evidence=())
        self.synthesis_prompt_appended = True
        if self.synthesis_failures: raise self.synthesis_failures.pop(0)
        self.session.turn_count += 1
        return SimpleNamespace(
            conclusion=SimpleNamespace(),
            objective_run=self.objective_run,
            objective_evidence=self.objective_evidence,
        )


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
    assert len(rendered) == 1 and "Evidence-Backed Conclusion" in html_text(workflow.active_card)


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
    assert "D_min" in text and "Limiting pair" in text
    assert "Revise" in text and "Goal achieved" in text
    assert "Baseline" in workflow.objective_summary.value
    assert "Attempt 1" in workflow.objective_summary.value
    assert "Attempt 2" in workflow.objective_summary.value
    assert [attempt.score for attempt in controller.objective_attempts] == [0.45, 0.90]


def test_objective_attempts_render_as_observe_act_measure_decision_ladder(monkeypatch):
    monkeypatch.setattr(demo_agent, "_display_conclusion", lambda result: None)
    monkeypatch.setattr("interactive_workflow.objective_figures", lambda run, state: ())
    workflow, _controller = started()

    run_objective(workflow)

    summary = workflow.objective_summary.value
    details = " ".join(html_text(card) for card in workflow.objective_attempt_cards)
    assert "Observe" in summary
    assert "Agent action" in summary
    assert "Measure" in summary
    assert "Outcome" in summary
    assert "swap mol-0-&gt;mol-4" in summary
    assert "from state state-" in summary
    assert "+0.450" in summary
    assert "0.900 ≥ 0.790" in summary
    assert "<b>Outcome:</b> Revise" in summary
    assert "<b>Outcome:</b> Goal achieved" in summary
    assert "#D68A00" in summary
    assert "#76B900" in summary
    assert "Planned deterministic command" in details
    assert "if action.swap_id == &#x27;mol-0-&gt;mol-4&#x27;" in details
    assert "Goal achieved" in details


def test_objective_receipt_render_failure_after_commit_never_reexecutes(monkeypatch):
    workflow, controller = started()
    complete_six_stages(workflow)
    reached = []

    def fail_receipt(*args, **kwargs):
        reached.append("render")
        raise RuntimeError("injected receipt render failure")

    monkeypatch.setattr("interactive_workflow.objective_receipt", fail_receipt)
    workflow.objective_button.click()
    workflow.objective_button.click()

    assert reached == ["render"]
    assert len(controller.objective_attempts) == 1
    assert len([
        call for call in controller.calls
        if isinstance(call, tuple) and call[0] == "objective_execute"
    ]) == 1
    assert workflow.status == "stopped"


def test_objective_revision_without_prior_attempt_fails_closed():
    attempt = ObjectiveAttempt(
        attempt_number=2,
        selected_ids=("mol-0", "mol-5", "mol-2", "mol-3"),
        decision_basis="Revise the measured panel.",
        score=0.80,
        limiting_pair=("mol-0", "mol-2"),
        constraints_passed=True,
        achieved=True,
        state_id="state-0123456789abcdef",
        selected_swap=ObjectiveSwap(
            replace_id="mol-1",
            replacement_id="mol-5",
            resulting_ids=("mol-0", "mol-5", "mol-2", "mol-3"),
            predicted_score=0.80,
            score_delta=0.80 - 0.35,
            limiting_pair=("mol-0", "mol-2"),
        ),
    )

    with pytest.raises(ValueError, match="prior attempt"):
        InteractiveWorkflow._objective_attempt_row(
            SimpleNamespace(baseline_score=0.35, target_score=0.71),
            attempt,
        )


def test_objective_attempt_row_enforces_initial_and_revision_state_shapes():
    context = SimpleNamespace(baseline_score=0.35, target_score=0.71)
    swap = ObjectiveSwap(
        replace_id="mol-1",
        replacement_id="mol-5",
        resulting_ids=("mol-0", "mol-5", "mol-2", "mol-3"),
        predicted_score=0.80,
        score_delta=0.80 - 0.35,
        limiting_pair=("mol-0", "mol-2"),
    )
    initial = ObjectiveAttempt(
        attempt_number=1,
        selected_ids=("mol-0", "mol-1", "mol-2", "mol-3"),
        decision_basis="Measure the initial panel.",
        score=0.35,
        limiting_pair=("mol-0", "mol-1"),
        constraints_passed=True,
        achieved=False,
    )
    initial_with_swap = ObjectiveAttempt(
        attempt_number=1,
        selected_ids=swap.resulting_ids,
        decision_basis="Incorrect initial provenance.",
        score=0.80,
        limiting_pair=swap.limiting_pair,
        constraints_passed=True,
        achieved=True,
        state_id="",
        selected_swap=swap,
    )
    revision_without_swap = ObjectiveAttempt(
        attempt_number=2,
        selected_ids=swap.resulting_ids,
        decision_basis="Incorrect revision provenance.",
        score=0.80,
        limiting_pair=swap.limiting_pair,
        constraints_passed=True,
        achieved=True,
    )
    class SwapSubclass(ObjectiveSwap):
        pass

    revision_with_subclass = ObjectiveAttempt(
        attempt_number=2,
        selected_ids=swap.resulting_ids,
        decision_basis="Incorrect subclass provenance.",
        score=0.80,
        limiting_pair=swap.limiting_pair,
        constraints_passed=True,
        achieved=True,
        state_id="state-0123456789abcdef",
        selected_swap=SwapSubclass(**swap.__dict__),
    )

    with pytest.raises(ValueError, match="initial"):
        InteractiveWorkflow._objective_attempt_row(context, initial, initial)
    with pytest.raises(ValueError, match="state-bound"):
        InteractiveWorkflow._objective_attempt_row(context, initial_with_swap)
    with pytest.raises(ValueError, match="state-bound"):
        InteractiveWorkflow._objective_attempt_row(context, revision_without_swap, initial)
    with pytest.raises(ValueError, match="exact"):
        InteractiveWorkflow._objective_attempt_row(context, revision_with_subclass, initial)


def test_objective_attempt_row_uses_gray_amber_and_green_by_state():
    context = SimpleNamespace(baseline_score=0.35, target_score=0.71)
    initial_miss = ObjectiveAttempt(
        attempt_number=1,
        selected_ids=("mol-0", "mol-1", "mol-2", "mol-3"),
        decision_basis="Measure the initial panel.",
        score=0.35,
        limiting_pair=("mol-0", "mol-1"),
        constraints_passed=True,
        achieved=False,
    )
    swap = ObjectiveSwap(
        replace_id="mol-1",
        replacement_id="mol-5",
        resulting_ids=("mol-0", "mol-5", "mol-2", "mol-3"),
        predicted_score=0.50,
        score_delta=0.50 - 0.35,
        limiting_pair=("mol-0", "mol-2"),
    )
    revision_miss = ObjectiveAttempt(
        attempt_number=2,
        selected_ids=swap.resulting_ids,
        decision_basis="Use the measured replacement.",
        score=0.50,
        limiting_pair=swap.limiting_pair,
        constraints_passed=True,
        achieved=False,
        state_id="state-0123456789abcdef",
        selected_swap=swap,
    )
    initial_achieved = ObjectiveAttempt(
        attempt_number=1,
        selected_ids=("mol-0", "mol-1", "mol-2", "mol-3"),
        decision_basis="The initial panel meets target.",
        score=0.80,
        limiting_pair=("mol-0", "mol-1"),
        constraints_passed=True,
        achieved=True,
    )

    assert "#6c757d" in InteractiveWorkflow._objective_attempt_row(context, initial_miss)
    assert "#D68A00" in InteractiveWorkflow._objective_attempt_row(
        context, revision_miss, initial_miss
    )
    assert "#76B900" in InteractiveWorkflow._objective_attempt_row(
        context, initial_achieved
    )


def test_objective_attempt_row_binds_evaluated_records_and_escapes_action_content():
    context, first, second = evaluated_objective_records(
        (
            "mol<0>", "mol-1", "mol-2", "mol-3", "mol&4", "mol-5",
            "mol-6", "mol-7",
        )
    )
    adversarial = replace(second, decision_basis="<img src=x onerror=alert(1)>")

    initial_row = InteractiveWorkflow._objective_attempt_row(context, first)
    revision_row = InteractiveWorkflow._objective_attempt_row(context, adversarial, first)

    assert "baseline D_min 0.350" in initial_row
    assert "prior D_min 0.450" in revision_row
    assert "limiting pair" in revision_row
    assert "swap" in revision_row and "from state" in revision_row
    assert "mol<0>" not in initial_row
    assert "&lt;img src=x onerror=alert(1)&gt;" not in revision_row
    assert "<img src=x" not in revision_row
    assert "<b>Outcome:</b> Goal achieved" in revision_row


@pytest.mark.parametrize(
    "mutation",
    [
        lambda attempt: replace(attempt, attempt_number=4),
        lambda attempt: replace(attempt, attempt_number=3),
        lambda attempt: replace(attempt, selected_ids=("mol-0", "mol-1", "mol-2", "mol-3")),
        lambda attempt: replace(attempt, score=0.79),
        lambda attempt: replace(
            attempt,
            selected_swap=replace(attempt.selected_swap, score_delta=0.44),
        ),
    ],
)
def test_objective_attempt_row_rejects_forged_evaluated_transition_records(mutation):
    context, first, second = evaluated_objective_records()

    with pytest.raises(ValueError):
        InteractiveWorkflow._objective_attempt_row(context, mutation(second), first)


def test_objective_attempt_row_adapts_precision_without_erasing_near_threshold_delta():
    context = SimpleNamespace(baseline_score=0.7096, target_score=0.7104)
    prior = ObjectiveAttempt(
        attempt_number=1,
        selected_ids=("mol-0", "mol-1", "mol-2", "mol-3"),
        decision_basis="Measure the baseline.",
        score=0.7096,
        limiting_pair=("mol-0", "mol-1"),
        constraints_passed=True,
        achieved=False,
    )
    swap = ObjectiveSwap(
        replace_id="mol-1",
        replacement_id="mol-5",
        resulting_ids=("mol-0", "mol-5", "mol-2", "mol-3"),
        predicted_score=0.7100,
        score_delta=0.7100 - 0.7096,
        limiting_pair=("mol-0", "mol-2"),
    )
    attempt = ObjectiveAttempt(
        attempt_number=2,
        selected_ids=swap.resulting_ids,
        decision_basis="Use the narrow measured improvement.",
        score=0.7100,
        limiting_pair=swap.limiting_pair,
        constraints_passed=True,
        achieved=False,
        state_id="state-0123456789abcdef",
        selected_swap=swap,
    )

    row = InteractiveWorkflow._objective_attempt_row(context, attempt, prior)

    assert "0.7100 &lt; 0.7104" in row
    assert "Δ +0.0004" in row


@pytest.mark.parametrize(
    "attempt, prior, context",
    [
        (
            lambda first, second: replace(first, constraints_passed=False),
            lambda first, second: None,
            lambda context: context,
        ),
        (
            lambda first, second: replace(first, achieved=True),
            lambda first, second: None,
            lambda context: context,
        ),
        (
            lambda first, second: replace(second, achieved=False),
            lambda first, second: first,
            lambda context: context,
        ),
        (
            lambda first, second: second,
            lambda first, second: replace(first, achieved=True),
            lambda context: context,
        ),
        (
            lambda first, second: replace(first, score=float("nan")),
            lambda first, second: None,
            lambda context: context,
        ),
        (
            lambda first, second: first,
            lambda first, second: None,
            lambda context: SimpleNamespace(
                baseline_score=float("inf"), target_score=context.target_score,
            ),
        ),
    ],
)
def test_objective_attempt_row_rejects_false_domain_truth(attempt, prior, context):
    objective_context, first, second = evaluated_objective_records()

    with pytest.raises(ValueError):
        InteractiveWorkflow._objective_attempt_row(
            context(objective_context), attempt(first, second), prior(first, second)
        )


@pytest.mark.parametrize("delta", [0.0, -0.10])
def test_objective_attempt_row_rejects_nonpositive_revision_delta(delta):
    context, first, second = evaluated_objective_records()
    no_op_swap = replace(
        second.selected_swap,
        predicted_score=first.score + delta,
        score_delta=delta,
    )
    no_op = replace(
        second,
        score=first.score + delta,
        achieved=False,
        selected_swap=no_op_swap,
    )

    with pytest.raises(ValueError):
        InteractiveWorkflow._objective_attempt_row(context, no_op, first)


def test_objective_attempt_row_uses_scientific_notation_when_six_decimals_hide_truth():
    context = SimpleNamespace(baseline_score=0.7099996, target_score=0.7100004)
    prior = ObjectiveAttempt(
        attempt_number=1,
        selected_ids=("mol-0", "mol-1", "mol-2", "mol-3"),
        decision_basis="Measure the baseline.",
        score=0.7099996,
        limiting_pair=("mol-0", "mol-1"),
        constraints_passed=True,
        achieved=False,
    )
    swap = ObjectiveSwap(
        replace_id="mol-1",
        replacement_id="mol-5",
        resulting_ids=("mol-0", "mol-5", "mol-2", "mol-3"),
        predicted_score=0.7100000,
        score_delta=0.7100000 - 0.7099996,
        limiting_pair=("mol-0", "mol-2"),
    )
    attempt = ObjectiveAttempt(
        attempt_number=2,
        selected_ids=swap.resulting_ids,
        decision_basis="Keep the measurable narrow improvement.",
        score=0.7100000,
        limiting_pair=swap.limiting_pair,
        constraints_passed=True,
        achieved=False,
        state_id="state-0123456789abcdef",
        selected_swap=swap,
    )

    row = InteractiveWorkflow._objective_attempt_row(context, attempt, prior)

    assert "0.710000 &lt; 0.710000" not in row
    assert "Δ +0.000000" not in row
    assert "e-07" in row


def test_objective_attempt_row_distinguishes_exact_quantized_tie_and_miss_outcomes():
    target = 0.71

    def initial(score, achieved):
        return ObjectiveAttempt(
            attempt_number=1,
            selected_ids=("mol-0", "mol-1", "mol-2", "mol-3"),
            decision_basis="Measure the initial panel.",
            score=score,
            limiting_pair=("mol-0", "mol-1"),
            constraints_passed=True,
            achieved=achieved,
        )

    exact = InteractiveWorkflow._objective_attempt_row(
        SimpleNamespace(baseline_score=target, target_score=target),
        initial(target, True),
    )
    quantized_tie_score = float(np.nextafter(target, 0.0))
    quantized_tie = InteractiveWorkflow._objective_attempt_row(
        SimpleNamespace(
            baseline_score=quantized_tie_score,
            target_score=target,
        ),
        initial(quantized_tie_score, True),
    )
    miss = InteractiveWorkflow._objective_attempt_row(
        SimpleNamespace(baseline_score=0.709, target_score=target),
        initial(0.709, False),
    )

    assert "0.710 ≥ 0.710" in exact
    assert "meets the quantized target" in quantized_tie
    assert "<b>Outcome:</b> Goal achieved" in quantized_tie
    assert "0.709 &lt; 0.710" in miss


def test_objective_attempt_row_uses_score_keys_for_target_attainment():
    score = 0.5
    target = 0.50000000000075
    attempt = ObjectiveAttempt(
        attempt_number=1,
        selected_ids=("mol-0", "mol-1", "mol-2", "mol-3"),
        decision_basis="Measure the initial panel.",
        score=score,
        limiting_pair=("mol-0", "mol-1"),
        constraints_passed=True,
        achieved=False,
    )

    row = InteractiveWorkflow._objective_attempt_row(
        SimpleNamespace(baseline_score=score, target_score=target), attempt
    )

    assert "<b>Outcome:</b> Revise" in row


def test_objective_attempt_row_accepts_one_score_key_revision_below_old_tolerance():
    current = 0.5
    improved = 0.5000000000005
    delta = improved - current
    prior = ObjectiveAttempt(
        attempt_number=1,
        selected_ids=("mol-0", "mol-1", "mol-2", "mol-3"),
        decision_basis="Measure the initial panel.",
        score=current,
        limiting_pair=("mol-0", "mol-1"),
        constraints_passed=True,
        achieved=False,
    )
    swap = ObjectiveSwap(
        replace_id="mol-1",
        replacement_id="mol-5",
        resulting_ids=("mol-0", "mol-5", "mol-2", "mol-3"),
        predicted_score=improved,
        score_delta=delta,
        limiting_pair=("mol-0", "mol-2"),
    )
    attempt = ObjectiveAttempt(
        attempt_number=2,
        selected_ids=swap.resulting_ids,
        decision_basis="Use the one-key measured improvement.",
        score=improved,
        limiting_pair=swap.limiting_pair,
        constraints_passed=True,
        achieved=True,
        state_id="state-0123456789abcdef",
        selected_swap=swap,
    )

    row = InteractiveWorkflow._objective_attempt_row(
        SimpleNamespace(baseline_score=current, target_score=improved),
        attempt,
        prior,
    )

    assert "<b>Outcome:</b> Goal achieved" in row


@pytest.mark.parametrize(
    "pair",
    [
        ("mol-9", "mol-0"),
        ("mol-0", "mol-0"),
        ("", "mol-0"),
    ],
)
def test_objective_attempt_row_rejects_noncanonical_or_out_of_panel_limiting_pairs(pair):
    context, first, second = evaluated_objective_records()
    forged_swap = replace(second.selected_swap, limiting_pair=pair)
    forged_second = replace(second, limiting_pair=pair, selected_swap=forged_swap)

    with pytest.raises(ValueError):
        InteractiveWorkflow._objective_attempt_row(
            context, replace(first, limiting_pair=pair)
        )
    with pytest.raises(ValueError):
        InteractiveWorkflow._objective_attempt_row(context, second, replace(first, limiting_pair=pair))
    with pytest.raises(ValueError):
        InteractiveWorkflow._objective_attempt_row(context, forged_second, first)


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
    controller.pending_objective = controller.objective_proposals[1]
    workflow.retry_button.click()

    assert workflow.status == "stopped"
    assert workflow.retry_button is None
    assert controller.calls.count("objective_proposal") == before


@pytest.mark.parametrize("mutation", ["pending_swap", "missing_suggestions"])
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
    if mutation == "pending_swap":
        controller.pending_objective_swap = object()
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
    monkeypatch.setattr(demo_agent, "_display_conclusion", lambda result: (_ for _ in ()).throw(RuntimeError("CONCLUSION-SECRET")))
    workflow, _ = started(controller)
    complete_six_stages(workflow)
    workflow.objective_button.click()
    assert workflow.status == "completed" and workflow.workflow_result is not None
    assert workflow.retry_button is None and controller.calls.count("synthesis") == 1
    assert "Conclusion rendering unavailable in this notebook frontend." in html_text(workflow.active_card)
    assert "CONCLUSION-SECRET" not in workflow.transcript_text


def test_completed_control_observers_are_detached():
    workflow, _ = started()
    workflow.approve_button.click()
    old_controls = dict(workflow.controls)
    workflow.approve_button.click()
    old_card = workflow.completed_cards[1]
    before = (html_text(old_card), html_text(workflow.active_card), workflow.transcript_text)
    old_controls["radius"].value = 3
    assert (html_text(old_card), html_text(workflow.active_card), workflow.transcript_text) == before
