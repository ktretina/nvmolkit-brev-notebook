import copy
import json
from dataclasses import fields, replace
from types import SimpleNamespace

import httpx
import pytest

import demo_agent
from chemistry_workflow import StageResult, WorkflowPhase, WorkflowReport
from objective_fixtures import evidence_report, optimized_state


CANONICAL_REPORT_ROWS = (
    ("E01", "Library inspection", "RDKit input validation"),
    ("E02", "Morgan fingerprints", "MorganFingerprintGenerator"),
    ("E03", "Tanimoto similarity", "crossTanimotoSimilarity"),
    ("E04", "Fused Butina clusters", "fused_butina"),
    ("E05", "Representative embedding", "EmbedMolecules"),
    ("E06", "MMFF94 optimization", "MMFFOptimizeMoleculesConfs"),
)


def _make_prepared_snapshot():
    messages = [
        {"role": "system", "content": "grounding"},
        {"role": "user", "content": "goal"},
    ]
    for index, name in enumerate(("submit_workflow_plan", *demo_agent.STAGES)):
        call_id = f"prepared-{index}"
        messages.extend((
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": "{}"},
                }],
            },
            {"role": "tool", "tool_call_id": call_id, "content": "{}"},
        ))
    plan = demo_agent.WorkflowPlan(stages=[
        demo_agent.PlanStage(stage=stage, rationale=f"Run {stage} safely.")
        for stage in demo_agent.STAGES
    ])
    stages = tuple(
        StageResult(stage, stage, {"stage": stage}) for stage in demo_agent.STAGES
    )
    report = evidence_report()
    return demo_agent.PreparedScientificSnapshot(
        session=demo_agent.AgentSession(messages, optimized_state(), 7),
        plan=plan,
        stage_results=stages,
        report=report,
        executors={
            **{stage: (lambda state, **kwargs: None) for stage in demo_agent.STAGES},
            "build_workflow_report": lambda state: report,
        },
    )


@pytest.fixture
def prepared_snapshot():
    return _make_prepared_snapshot()


def test_prepared_snapshot_requires_exact_completed_scientific_boundary(prepared_snapshot):
    snapshot = prepared_snapshot

    assert snapshot.session.turn_count == 7
    assert len(snapshot.session.messages) == 16
    assert snapshot.session.state.phase.value == "optimized"
    assert tuple(item.stage for item in snapshot.plan.stages) == demo_agent.STAGES
    assert tuple(item.stage for item in snapshot.stage_results) == demo_agent.STAGES
    assert tuple(item.key for item in snapshot.report.evidence) == (
        "E01", "E02", "E03", "E04", "E05", "E06"
    )


@pytest.mark.parametrize("mutation", ("turn_count", "messages", "report"))
def test_prepared_snapshot_fails_closed_on_incomplete_or_noncanonical_input(
    prepared_snapshot, mutation
):
    valid = prepared_snapshot
    if mutation == "turn_count":
        valid.session.turn_count = 6
    elif mutation == "messages":
        valid.session.messages[-1]["tool_call_id"] = "unpaired"
    else:
        object.__setattr__(valid, "report", WorkflowReport(valid.report.evidence[:-1]))

    with pytest.raises((TypeError, ValueError), match="prepared|canonical|paired"):
        demo_agent.PreparedScientificSnapshot(
            session=valid.session,
            plan=valid.plan,
            stage_results=valid.stage_results,
            report=valid.report,
            executors=valid.executors,
        )


def test_prepared_snapshot_rejects_named_but_fabricated_evidence_payload(
    prepared_snapshot,
):
    valid = prepared_snapshot
    records = list(valid.report.evidence)
    records[0] = type(records[0])(
        records[0].key,
        records[0].label,
        '{"key":"E01"}',
        records[0].provenance,
    )

    with pytest.raises(ValueError, match="production-shaped"):
        demo_agent.PreparedScientificSnapshot(
            session=valid.session,
            plan=valid.plan,
            stage_results=valid.stage_results,
            report=WorkflowReport(tuple(records)),
            executors=valid.executors,
        )


def test_clone_prepared_controller_is_deep_isolated_and_objective_clean(
    prepared_snapshot,
):
    snapshot = prepared_snapshot
    first = demo_agent.clone_prepared_controller(snapshot, client=object())
    second = demo_agent.clone_prepared_controller(snapshot, client=object())

    assert first.session is not second.session is not snapshot.session
    assert first.session.state is not second.session.state is not snapshot.session.state
    assert first.plan is not second.plan is not snapshot.plan
    assert first.stage_results is not second.stage_results
    assert first.report is not second.report is not snapshot.report
    assert first.objective_required is True
    assert first.objective_context is None
    assert first.pending_action_menu is None
    assert first.pending_objective_selection is None
    assert first.objective_attempts == []
    assert first.accepted_attempt_count == first.rejected_selection_count == 0
    assert first.correction_prompts_sent == first.selection_response_count == 0
    assert first.provider_request_attempt_count == 0
    assert first.objective_run is first.objective_evidence is None
    assert first.objective_prompt_appended is False
    first.session.messages[0]["content"] = "mutated"
    first.session.state.records[0]["id"] = "mutated"
    first.stage_results[0].summary["mutated"] = True

    assert second.session.messages[0]["content"] == "grounding"
    assert snapshot.session.messages[0]["content"] == "grounding"
    assert second.session.state.records[0]["id"] == "mol-0"
    assert "mutated" not in second.stage_results[0].summary


class _CurrentMaximumCompletions:
    def __init__(self, fail_first=False):
        self.controller = None
        self.fail_first = fail_first
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_first:
            self.fail_first = False
            raise httpx.ConnectError("provider details must not leak")
        menu = self.controller.pending_action_menu
        action = demo_agent.accepted_maxima(menu)[0]
        arguments = {
            "state_id": menu.state_id,
            "swap_id": action.swap_id,
            "observed_limiting_pairs": [list(pair) for pair in menu.source.limiting_pairs],
            "decision_rule": "maximize_predicted_minimum_distance",
        }
        call = SimpleNamespace(
            id=f"objective-{len(self.calls)}",
            type="function",
            function=SimpleNamespace(
                name="select_next_panel_swap", arguments=json.dumps(arguments)
            ),
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=None, tool_calls=[call]
        ))])


class ScriptedControllerFactory:
    def __init__(self, snapshot=None, *, transport_failure_trial=None):
        self.snapshot = snapshot or _make_prepared_snapshot()
        self.transport_failure_trial = transport_failure_trial
        self.created = 0
        self.completions = []

    def __call__(self):
        self.created += 1
        completions = _CurrentMaximumCompletions(
            fail_first=self.created == self.transport_failure_trial
        )
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        controller = demo_agent.clone_prepared_controller(self.snapshot, client=client)
        completions.controller = controller
        self.completions.append(completions)
        return controller


def test_run_trials_uses_current_argmax_and_labels_one_transport_retry(prepared_snapshot):
    from scripts.run_objective_reliability import run_trials

    factory = ScriptedControllerFactory(
        prepared_snapshot, transport_failure_trial=2
    )
    trials = run_trials(factory, trials=2)

    assert len(trials) == 2
    assert all(item["completed"] for item in trials)
    assert all(item["argmax_success"] for item in trials)
    assert [item["retry_assisted"] for item in trials] == [False, True]
    assert all(item["termination_reason"] == "target_achieved" for item in trials)
    assert all(item["message_pairing_passed"] for item in trials)
    assert all(item["claim_safety_passed"] for item in trials)
    assert all(item["production_temperature_zero"] for item in trials)
    assert all(
        call["model"] == demo_agent.DEFAULT_MODEL
        and call["temperature"] == 0.0
        and call["extra_body"] == demo_agent.NEMOTRON_TOOL_EXTRA_BODY
        and call["tool_choice"]["function"]["name"] == "select_next_panel_swap"
        and call["tools"][0]["function"]["strict"] is True
        for completions in factory.completions
        for call in completions.calls
    )


def test_failed_trial_retains_sanitized_controller_counters_and_pairing(prepared_snapshot):
    from scripts.run_objective_reliability import run_trials

    factory = ScriptedControllerFactory(prepared_snapshot)
    controller = factory()
    completions = factory.completions[-1]
    real_create = completions.create

    def invalid_create(**kwargs):
        response = real_create(**kwargs)
        response.choices[0].message.tool_calls[0].function.arguments = json.dumps({
            "state_id": controller.pending_action_menu.state_id,
            "swap_id": "mol-0->not-in-menu",
            "observed_limiting_pairs": [
                list(pair) for pair in controller.pending_action_menu.source.limiting_pairs
            ],
            "decision_rule": "maximize_predicted_minimum_distance",
        })
        return response

    completions.create = invalid_create
    records = run_trials(lambda: controller, trials=1)

    assert records[0]["completed"] is False
    assert records[0]["argmax_success"] is False
    assert records[0]["rejected_selection_count"] == 2
    assert records[0]["correction_prompts_sent"] == 1
    assert records[0]["selection_response_count"] == 2
    assert records[0]["message_pairing_passed"] is True
    assert "provider details" not in json.dumps(records[0])


class _EndToEndCompletions(_CurrentMaximumCompletions):
    def create(self, **kwargs):
        name = kwargs["tool_choice"]["function"]["name"]
        if name == "select_next_panel_swap":
            return super().create(**kwargs)
        self.calls.append(kwargs)
        if name == "submit_workflow_plan":
            arguments = {
                "stages": [
                    {"stage": stage, "rationale": f"Run {stage} after prerequisites."}
                    for stage in demo_agent.STAGES
                ]
            }
        elif name == "select_evidence_findings":
            catalog = json.loads(kwargs["messages"][-1]["content"])["findings"]
            selected = []
            seen = set()
            for finding in catalog:
                if finding["theme"] not in seen:
                    selected.append(finding["finding_id"])
                    seen.add(finding["theme"])
            arguments = {"ordered_finding_ids": selected}
        else:
            arguments = {
                "inspect_library": {},
                "generate_morgan_fingerprints": {
                    "radius": 2, "size": 1024,
                    "decision_basis": "Use fixed qualification parameters.",
                },
                "measure_tanimoto_similarity": {},
                "discover_fused_butina_clusters": {
                    "cutoff": 0.4,
                    "decision_basis": "Use the fixed qualification cutoff.",
                },
                "embed_representative_conformers": {
                    "representative_count": 4,
                    "policy": "largest_clusters_first",
                    "conformers_per_representative": 5,
                    "decision_basis": "Use fixed representative sampling.",
                },
                "optimize_conformers_mmff94": {},
            }[name]
        call = SimpleNamespace(
            id=f"e2e-{len(self.calls)}", type="function",
            function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=None, tool_calls=[call]
        ))])


def _fresh_end_to_end_controller():
    phases = dict(zip(demo_agent.STAGES, (
        WorkflowPhase.INSPECTED, WorkflowPhase.FINGERPRINTED,
        WorkflowPhase.COMPARED, WorkflowPhase.CLUSTERED,
        WorkflowPhase.EMBEDDED, WorkflowPhase.OPTIMIZED,
    ), strict=True))
    report = evidence_report()

    def executor(stage):
        def execute(state, **kwargs):
            if stage == demo_agent.STAGES[-1]:
                prepared = optimized_state()
                state.__dict__.clear()
                state.__dict__.update(copy.deepcopy(prepared.__dict__))
            else:
                state.phase = phases[stage]
            return StageResult(stage, stage, {"stage": stage})
        return execute

    completions = _EndToEndCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    controller = demo_agent.BoundedWorkflowController.create(
        "Run the fresh qualification workflow.", "nvapi-test-key",
        client=client,
        objective_required=True,
        executors={
            **{stage: executor(stage) for stage in demo_agent.STAGES},
            "build_workflow_report": lambda state: report,
        },
    )
    completions.controller = controller
    return controller


def test_run_end_to_end_requires_fresh_state_and_reaches_evidence_controlled_finding():
    from scripts.run_objective_reliability import run_end_to_end

    records = run_end_to_end(_fresh_end_to_end_controller, runs=1)

    assert len(records) == 1
    assert records[0]["completed"] is True
    assert records[0]["argmax_success"] is True
    assert records[0]["message_pairing_passed"] is True
    assert records[0]["claim_safety_passed"] is True
    assert records[0]["production_temperature_zero"] is True
    assert records[0]["conclusion_status"] == "selected"


def test_end_to_end_missing_conclusion_is_incomplete_even_after_objective_success():
    from scripts.run_objective_reliability import run_end_to_end

    def factory():
        controller = _fresh_end_to_end_controller()

        def fail_conclusion():
            raise demo_agent.ToolCallError("raw provider prose must not leak")

        controller.request_synthesis = fail_conclusion
        return controller

    records = run_end_to_end(factory, runs=1)

    assert records[0]["completed"] is False
    assert records[0]["conclusion_status"] is None
    assert "raw provider prose" not in json.dumps(records[0])


def test_reliability_receipt_has_exact_frozen_fields_and_fail_closed_exit_code(monkeypatch):
    import scripts.run_objective_reliability as reliability

    assert [field.name for field in fields(reliability.ReliabilityReceipt)] == [
        "requested_trials", "completed_trials", "argmax_successes",
        "clean_first_request_trials", "retry_assisted_trials",
        "requested_end_to_end_runs", "completed_end_to_end_runs",
        "message_pairing_passes", "claim_safety_passes",
        "production_temperature_zero", "objective_trials", "end_to_end_runs",
        "failed_trials",
    ]
    monkeypatch.setattr(reliability, "run_end_to_end", lambda factory, *, runs: tuple(
        {
            "run": index + 1, "completed": True, "argmax_success": True,
            "message_pairing_passed": True, "claim_safety_passed": True,
            "production_temperature_zero": True,
            "conclusion_status": "selected",
        }
        for index in range(runs)
    ))
    receipt = reliability.run_qualification(
        ScriptedControllerFactory(), ScriptedControllerFactory(),
        trials=2, end_to_end_runs=1,
    )

    assert receipt.requested_trials == receipt.completed_trials == 2
    assert receipt.requested_end_to_end_runs == receipt.completed_end_to_end_runs == 1
    assert receipt.argmax_successes == 2
    assert receipt.message_pairing_passes == 3
    assert receipt.claim_safety_passes == 3
    assert reliability.qualification_exit_code(receipt) == 0
    unsafe = reliability.ReliabilityReceipt(
        **{**receipt.__dict__, "claim_safety_passes": 2}
    )
    assert reliability.qualification_exit_code(unsafe) != 0


def test_completed_but_nonqualifying_trial_is_listed_as_failed(monkeypatch):
    import scripts.run_objective_reliability as reliability

    objective = {
        "completed": True, "argmax_success": False, "retry_assisted": False,
        "message_pairing_passed": True, "claim_safety_passed": True,
        "production_temperature_zero": True,
    }
    end_to_end = {
        "completed": True, "argmax_success": True,
        "message_pairing_passed": True, "claim_safety_passed": True,
        "production_temperature_zero": True, "conclusion_status": "selected",
    }
    monkeypatch.setattr(
        reliability, "run_trials", lambda factory, *, trials: (objective,)
    )
    monkeypatch.setattr(
        reliability, "run_end_to_end", lambda factory, *, runs: (end_to_end,)
    )

    receipt = reliability.run_qualification(
        lambda: None, lambda: None, trials=1, end_to_end_runs=1
    )

    assert receipt.failed_trials == (objective,)


@pytest.mark.parametrize(
    "change",
    (
        {"completed_trials": 1},
        {"argmax_successes": 1},
        {"completed_end_to_end_runs": 0},
        {"message_pairing_passes": 2},
        {"claim_safety_passes": 2},
        {"production_temperature_zero": False},
    ),
)
def test_qualification_exit_code_directly_rejects_each_incomplete_gate(monkeypatch, change):
    import scripts.run_objective_reliability as reliability

    monkeypatch.setattr(reliability, "run_end_to_end", lambda factory, *, runs: ({
        "kind": "end_to_end", "index": 1, "completed": True,
        "argmax_success": True, "message_pairing_passed": True,
        "claim_safety_passed": True, "production_temperature_zero": True,
        "conclusion_status": "selected", "termination_reason": "target_achieved",
    },))
    receipt = reliability.run_qualification(
        ScriptedControllerFactory(), ScriptedControllerFactory(),
        trials=2, end_to_end_runs=1,
    )

    assert reliability.qualification_exit_code(replace(receipt, **change)) == 1


def test_cli_calls_run_qualification_and_writes_only_allowlisted_receipt(monkeypatch, tmp_path):
    import scripts.run_objective_reliability as reliability

    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-secret-must-not-appear")
    monkeypatch.setattr(reliability, "_build_prepared_snapshot", lambda api_key: object())
    monkeypatch.setattr(reliability.demo_agent, "_client", lambda api_key: object())
    monkeypatch.setattr(
        reliability.demo_agent.BoundedWorkflowController,
        "create",
        classmethod(lambda cls, *args, **kwargs: object()),
    )
    called = {}
    receipt = reliability.ReliabilityReceipt(
        1, 1, 1, 1, 0, 1, 1, 2, 2, True,
        ({"completed": True},), ({"completed": True},), (),
    )

    def fake_qualification(objective_factory, end_to_end_factory, *, trials, end_to_end_runs):
        called.update(
            objective_factory=objective_factory,
            end_to_end_factory=end_to_end_factory,
            trials=trials,
            end_to_end_runs=end_to_end_runs,
        )
        return receipt

    monkeypatch.setattr(reliability, "run_qualification", fake_qualification)
    monkeypatch.setattr(reliability, "qualification_exit_code", lambda value: 0)
    output = tmp_path / "receipt.json"

    assert reliability.main([
        "--trials", "1", "--end-to-end-runs", "1", "--output", str(output)
    ]) == 0
    payload = json.loads(output.read_text())
    assert called["trials"] == called["end_to_end_runs"] == 1
    assert payload["model"] == demo_agent.DEFAULT_MODEL
    assert payload["environment"] == "production_hosted_api"
    assert payload["requested_trials"] == 1
    assert "nvapi-secret" not in output.read_text()
    assert "metadata" not in output.read_text()
