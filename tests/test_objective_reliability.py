import copy
import inspect
import json
from dataclasses import fields, replace
from types import SimpleNamespace

import httpx
import pytest

import demo_agent
from chemistry_workflow import (
    StageResult,
    WorkflowPhase,
    WorkflowReport,
    build_workflow_report,
)
from objective_fixtures import evidence_report, optimized_state


CANONICAL_REPORT_ROWS = (
    ("E01", "Library inspection", "RDKit input validation"),
    ("E02", "Morgan fingerprints", "MorganFingerprintGenerator"),
    ("E03", "Tanimoto similarity", "crossTanimotoSimilarity"),
    ("E04", "Fused Butina clusters", "fused_butina"),
    ("E05", "Representative embedding", "EmbedMolecules"),
    ("E06", "MMFF94 optimization", "MMFFOptimizeMoleculesConfs"),
)


def prepared_snapshot():
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
    state = optimized_state()
    report = build_workflow_report(state)
    stages = tuple(
        StageResult(stage, record.label, json.loads(record.payload_json))
        for stage, record in zip(demo_agent.STAGES, report.evidence, strict=True)
    )
    return demo_agent.PreparedScientificSnapshot(
        messages=tuple(messages),
        state=state,
        plan=plan,
        stage_results=stages,
        report=report,
        turn_count=7,
    )


def _prepared_executors(report):
    return {
        **{stage: (lambda state, **kwargs: None) for stage in demo_agent.STAGES},
        "build_workflow_report": lambda state: report,
    }


def test_prepared_snapshot_requires_exact_completed_scientific_boundary():
    snapshot = prepared_snapshot()

    assert [field.name for field in fields(demo_agent.PreparedScientificSnapshot)] == [
        "messages", "state", "plan", "stage_results", "report", "turn_count"
    ]
    assert snapshot.turn_count == 7
    assert len(snapshot.messages) == 16
    assert snapshot.state.phase.value == "optimized"
    assert tuple(item.stage for item in snapshot.plan.stages) == demo_agent.STAGES
    assert tuple(item.stage for item in snapshot.stage_results) == demo_agent.STAGES
    assert tuple(item.key for item in snapshot.report.evidence) == (
        "E01", "E02", "E03", "E04", "E05", "E06"
    )


@pytest.mark.parametrize("mutation", ("turn_count", "messages", "report"))
def test_prepared_snapshot_fails_closed_on_incomplete_or_noncanonical_input(
    mutation,
):
    valid = prepared_snapshot()
    messages = copy.deepcopy(valid.messages)
    turn_count = valid.turn_count
    report = valid.report
    if mutation == "turn_count":
        turn_count = 6
    elif mutation == "messages":
        messages[-1]["tool_call_id"] = "unpaired"
    else:
        report = WorkflowReport(valid.report.evidence[:-1])

    with pytest.raises((TypeError, ValueError), match="prepared|canonical|paired"):
        demo_agent.PreparedScientificSnapshot(
            messages=tuple(messages),
            state=valid.state,
            plan=valid.plan,
            stage_results=valid.stage_results,
            report=report,
            turn_count=turn_count,
        )


def test_prepared_snapshot_rejects_named_but_fabricated_evidence_payload():
    valid = prepared_snapshot()
    records = list(valid.report.evidence)
    records[0] = type(records[0])(
        records[0].key,
        records[0].label,
        '{"key":"E01"}',
        records[0].provenance,
    )

    with pytest.raises(ValueError, match="production-shaped"):
        demo_agent.PreparedScientificSnapshot(
            messages=valid.messages,
            state=valid.state,
            plan=valid.plan,
            stage_results=valid.stage_results,
            report=WorkflowReport(tuple(records)),
            turn_count=valid.turn_count,
        )


def test_prepared_snapshot_requires_report_exactly_rebuilt_from_copied_state():
    valid = prepared_snapshot()
    records = list(valid.report.evidence)
    payload = json.loads(records[0].payload_json)
    payload["preview_count"] -= 1
    records[0] = type(records[0])(
        records[0].key,
        records[0].label,
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        records[0].provenance,
    )

    with pytest.raises(ValueError, match="inconsistent"):
        demo_agent.PreparedScientificSnapshot(
            messages=valid.messages,
            state=valid.state,
            plan=valid.plan,
            stage_results=valid.stage_results,
            report=WorkflowReport(tuple(records)),
            turn_count=valid.turn_count,
        )


def test_clone_prepared_controller_is_deep_isolated_and_objective_clean():
    snapshot = prepared_snapshot()
    executors = _prepared_executors(snapshot.report)
    assert tuple(inspect.signature(demo_agent.clone_prepared_controller).parameters) == (
        "snapshot", "client", "executors"
    )
    first = demo_agent.clone_prepared_controller(
        snapshot, client=object(), executors=executors
    )
    second = demo_agent.clone_prepared_controller(
        snapshot, client=object(), executors=executors
    )

    assert first.session is not second.session
    assert first.session.state is not second.session.state is not snapshot.state
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
    assert snapshot.messages[0]["content"] == "grounding"
    assert second.session.state.records[0]["id"] == "mol-0"
    assert "mutated" not in second.stage_results[0].summary


class _CurrentMaximumCompletions:
    def __init__(self, fail_first=False):
        self.controller = None
        self.fail_first = fail_first
        self.calls = []
        self.schema_digests = []

    def create(self, **kwargs):
        from scripts.run_objective_reliability import _canonical_tool_schema_digest

        self.calls.append(kwargs)
        self.schema_digests.append(_canonical_tool_schema_digest(kwargs["tools"]))
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
    def __init__(self, snapshot, retry_trial_index=None):
        self.snapshot = snapshot
        self.retry_trial_index = retry_trial_index
        self.created = 0
        self.completions = []

    def __call__(self):
        self.created += 1
        completions = _CurrentMaximumCompletions(
            fail_first=(self.created - 1) == self.retry_trial_index
        )
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        controller = demo_agent.clone_prepared_controller(
            self.snapshot,
            client=client,
            executors=_prepared_executors(self.snapshot.report),
        )
        completions.controller = controller
        self.completions.append(completions)
        return controller


def test_run_trials_uses_current_argmax_and_labels_one_transport_retry():
    from scripts.run_objective_reliability import run_trials

    factory = ScriptedControllerFactory(
        prepared_snapshot(), retry_trial_index=1
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


def test_successful_trial_fails_temperature_gate_when_request_settings_are_unobservable(
):
    from scripts.run_objective_reliability import run_trials

    factory = ScriptedControllerFactory(prepared_snapshot())
    controller = factory()
    recorded = factory.completions[-1]
    controller.client.chat.completions = SimpleNamespace(create=recorded.create)

    trials = run_trials(lambda: controller, trials=1)

    assert trials[0]["completed"] is True
    assert trials[0]["production_temperature_zero"] is False


@pytest.mark.parametrize("mutation", ("max_tokens", "stream", "schema", "phase"))
def test_request_audit_rejects_each_control_or_phase_mutation(mutation):
    import scripts.run_objective_reliability as reliability

    factory = ScriptedControllerFactory(prepared_snapshot())
    assert reliability.run_trials(factory, trials=1)[0]["completed"] is True
    completions = factory.completions[0]
    controller = completions.controller
    call = completions.calls[0]
    if mutation == "max_tokens":
        call["max_tokens"] = 999
    elif mutation == "stream":
        call["stream"] = True
    elif mutation == "schema":
        call["tools"][0]["function"]["parameters"]["additionalProperties"] = True
    else:
        call["tool_choice"]["function"]["name"] = "submit_workflow_plan"
        call["tools"][0]["function"]["name"] = "submit_workflow_plan"
    assert reliability._calls_use_production_contract(controller, "objective") is False


def test_failed_trial_retains_sanitized_controller_counters_and_pairing():
    from scripts.run_objective_reliability import run_trials

    factory = ScriptedControllerFactory(prepared_snapshot())
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
        from scripts.run_objective_reliability import _canonical_tool_schema_digest

        self.calls.append(kwargs)
        self.schema_digests.append(_canonical_tool_schema_digest(kwargs["tools"]))
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


def test_end_to_end_rejects_forged_truthy_conclusion_object():
    from scripts.run_objective_reliability import run_end_to_end

    def factory():
        controller = _fresh_end_to_end_controller()
        real = controller.request_synthesis

        def forged():
            return replace(real(), conclusion=SimpleNamespace(
                finding_selection_status="selected"
            ))

        controller.request_synthesis = forged
        return controller

    record = run_end_to_end(factory, runs=1)[0]
    assert record["completed"] is False
    assert record["claim_safety_passed"] is False
    assert record["conclusion_status"] is None


@pytest.mark.parametrize("mutation", ("orphan", "reordered", "bad_type", "duplicate"))
def test_pairing_validator_is_an_immediate_strict_state_machine(mutation):
    import scripts.run_objective_reliability as reliability

    messages = list(copy.deepcopy(prepared_snapshot().messages))
    if mutation == "orphan":
        messages.insert(2, {"role": "tool", "tool_call_id": "orphan", "content": "{}"})
    elif mutation == "reordered":
        messages[2], messages[3] = messages[3], messages[2]
    elif mutation == "bad_type":
        messages[2]["tool_calls"][0]["type"] = "not-function"
    else:
        messages[4]["tool_calls"][0]["id"] = messages[2]["tool_calls"][0]["id"]
        messages[5]["tool_call_id"] = messages[2]["tool_calls"][0]["id"]
    assert reliability._messages_are_paired(messages) is False


def test_clone_rejects_snapshot_nested_tampering_after_construction():
    snapshot = prepared_snapshot()
    snapshot.messages[0]["content"] = "tampered"
    with pytest.raises(ValueError, match="tampered"):
        demo_agent.clone_prepared_controller(
            snapshot,
            client=object(),
            executors=_prepared_executors(snapshot.report),
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "invalid_ids",
        "fingerprint_parameters",
        "fingerprints",
        "molecules",
        "cluster_cutoff",
        "representative_records",
        "conformer_molecules",
        "embedding_parameters",
        "summaries",
        "optimization_result",
    ),
)
def test_clone_rejects_every_omitted_authoritative_state_family(mutation):
    snapshot = prepared_snapshot()
    if mutation == "invalid_ids":
        snapshot.state.invalid_ids = ("new-invalid",)
    elif mutation == "fingerprint_parameters":
        snapshot.state.fingerprint_parameters = (3, 1024)
    elif mutation == "fingerprints":
        snapshot.state.fingerprints.tensor.values[0, 0] = 1
    elif mutation == "molecules":
        snapshot.state.molecules.pop()
    elif mutation == "cluster_cutoff":
        snapshot.state.cluster_cutoff = 0.5
    elif mutation == "representative_records":
        snapshot.state.representative_records[0]["source_row"] = 99
    elif mutation == "conformer_molecules":
        snapshot.state.conformer_molecules.pop()
    elif mutation == "embedding_parameters":
        snapshot.state.embedding_parameters = (2, "largest_clusters_first", 1)
    elif mutation == "summaries":
        snapshot.state.summaries["external"] = {"changed": True}
    else:
        snapshot.state.optimization_result.energies.values[0] += 1.0

    with pytest.raises(ValueError, match="tampered"):
        demo_agent.clone_prepared_controller(
            snapshot,
            client=object(),
            executors=_prepared_executors(snapshot.report),
        )


@pytest.mark.parametrize("phase", ("objective", "finding"))
@pytest.mark.parametrize("mutation", ("type", "enum", "items", "constraint"))
def test_request_audit_rejects_nested_schema_mutation(phase, mutation):
    import scripts.run_objective_reliability as reliability

    if phase == "objective":
        factory = ScriptedControllerFactory(prepared_snapshot())
        assert reliability.run_trials(factory, trials=1)[0]["completed"] is True
        controller = factory.completions[0].controller
        call = factory.completions[0].calls[0]
        properties = call["tools"][0]["function"]["parameters"]["properties"]
        if mutation == "type":
            properties["state_id"]["type"] = "integer"
        elif mutation == "enum":
            properties["decision_rule"]["enum"] = ["tampered"]
        elif mutation == "items":
            properties["observed_limiting_pairs"]["items"] = {"type": "string"}
        else:
            properties["observed_limiting_pairs"]["minItems"] = 0
        kind = "objective"
    else:
        controller = _fresh_end_to_end_controller()
        assert reliability.run_end_to_end(lambda: controller, runs=1)[0]["completed"] is True
        calls = controller.client.chat.completions.calls
        call = next(
            item for item in calls
            if item["tool_choice"]["function"]["name"] == "select_evidence_findings"
        )
        properties = call["tools"][0]["function"]["parameters"]["properties"]
        selected = properties["ordered_finding_ids"]
        if mutation == "type":
            selected["type"] = "string"
        elif mutation == "enum":
            selected["items"]["enum"] = ["F999"]
        elif mutation == "items":
            selected["items"] = {"type": "integer"}
        else:
            selected["minItems"] = 0
        kind = "end_to_end"

    assert reliability._calls_use_production_contract(controller, kind) is False


def test_reliability_receipt_has_exact_frozen_fields_and_fail_closed_exit_code():
    import scripts.run_objective_reliability as reliability

    assert [field.name for field in fields(reliability.ReliabilityReceipt)] == [
        "requested_trials", "completed_trials", "argmax_successes",
        "clean_first_request_trials", "retry_assisted_trials",
        "requested_end_to_end_runs", "completed_end_to_end_runs",
        "message_pairing_passes", "claim_safety_passes",
        "production_temperature_zero", "objective_trials", "end_to_end_runs",
        "failed_trials",
    ]
    receipt = reliability.run_qualification(
        ScriptedControllerFactory(prepared_snapshot()),
        _fresh_end_to_end_controller,
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


def test_run_qualification_covers_20_isolated_trials_and_3_fresh_runs(
):
    import scripts.run_objective_reliability as reliability

    objective_factory = ScriptedControllerFactory(
        prepared_snapshot(), retry_trial_index=0
    )
    receipt = reliability.run_qualification(
        objective_factory,
        _fresh_end_to_end_controller,
        trials=20,
        end_to_end_runs=3,
    )

    assert receipt.completed_trials == 20
    assert receipt.argmax_successes == 20
    assert receipt.retry_assisted_trials == 1
    assert receipt.clean_first_request_trials == 19
    assert receipt.completed_end_to_end_runs == 3
    assert receipt.message_pairing_passes == 23
    assert receipt.claim_safety_passes == 23
    assert receipt.production_temperature_zero is True
    assert receipt.failed_trials == ()
    assert reliability.qualification_exit_code(receipt) == 0


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("kind", "forged"), ("index", 999), ("model", "wrong/model"),
        ("environment", "wrong"), ("accepted_attempt_count", 0),
        ("accepted_attempt_count", 4), ("rejected_selection_count", 1),
        ("correction_prompts_sent", 1), ("provider_request_attempt_count", 999),
        ("retry_assisted", True), ("termination_reason", "attempt_limit_reached"),
    ),
)
def test_exit_code_rejects_forged_objective_record_contract(field, value):
    import scripts.run_objective_reliability as reliability

    receipt = reliability.run_qualification(
        ScriptedControllerFactory(prepared_snapshot()),
        _fresh_end_to_end_controller,
        trials=1, end_to_end_runs=1,
    )
    forged = {**receipt.objective_trials[0], field: value}
    assert reliability.qualification_exit_code(
        replace(receipt, objective_trials=(forged,))
    ) == 1


@pytest.mark.parametrize("value", (0, True, -1, 999))
def test_exit_code_rejects_invalid_or_inconsistent_requested_counts(value):
    import scripts.run_objective_reliability as reliability

    receipt = reliability.run_qualification(
        ScriptedControllerFactory(prepared_snapshot()),
        _fresh_end_to_end_controller,
        trials=1, end_to_end_runs=1,
    )
    assert reliability.qualification_exit_code(
        replace(receipt, requested_trials=value)
    ) == 1


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
        {"clean_first_request_trials": 999},
        {"retry_assisted_trials": 999},
    ),
)
def test_qualification_exit_code_directly_rejects_each_incomplete_gate(change):
    import scripts.run_objective_reliability as reliability

    receipt = reliability.run_qualification(
        ScriptedControllerFactory(prepared_snapshot()),
        _fresh_end_to_end_controller,
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
    trial = {
        "kind": "objective", "index": 1, "model": demo_agent.DEFAULT_MODEL,
        "environment": "production_hosted_api", "completed": True,
        "argmax_success": True, "accepted_attempt_count": 1,
        "rejected_selection_count": 0, "correction_prompts_sent": 0,
        "selection_response_count": 1, "provider_request_attempt_count": 1,
        "retry_assisted": False, "baseline_score": 0.35,
        "target_score": 0.71, "final_score": 0.8,
        "termination_reason": "target_achieved", "message_pairing_passed": True,
        "claim_safety_passed": True, "production_temperature_zero": True,
        "conclusion_status": None,
    }
    receipt = reliability.ReliabilityReceipt(
        1, 1, 1, 1, 0, 1, 1, 2, 2, True,
        (trial,), ({**trial, "kind": "end_to_end", "conclusion_status": "selected"},), (),
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


@pytest.fixture
def receipt_with_sensitive_trial_metadata():
    from scripts.run_objective_reliability import ReliabilityReceipt

    trial = {
        "kind": "objective",
        "index": 1,
        "model": demo_agent.DEFAULT_MODEL,
        "environment": "production_hosted_api",
        "completed": True,
        "argmax_success": True,
        "accepted_attempt_count": 1,
        "rejected_selection_count": 0,
        "correction_prompts_sent": 0,
        "selection_response_count": 1,
        "provider_request_attempt_count": 1,
        "retry_assisted": False,
        "baseline_score": 0.35,
        "target_score": 0.71,
        "final_score": 0.8,
        "termination_reason": "target_achieved",
        "message_pairing_passed": True,
        "claim_safety_passed": True,
        "production_temperature_zero": True,
        "conclusion_status": None,
        "api_key": "nvapi-SENSITIVE-SENTINEL",
        "raw_model_response": "RAW-RESPONSE-SENTINEL",
        "metadata": {"arbitrary": "ARBITRARY-METADATA-SENTINEL"},
    }
    return ReliabilityReceipt(
        requested_trials=1,
        completed_trials=1,
        argmax_successes=1,
        clean_first_request_trials=1,
        retry_assisted_trials=0,
        requested_end_to_end_runs=1,
        completed_end_to_end_runs=1,
        message_pairing_passes=2,
        claim_safety_passes=2,
        production_temperature_zero=True,
        objective_trials=(trial,),
        end_to_end_runs=({**trial, "kind": "end_to_end", "conclusion_status": "selected"},),
        failed_trials=(),
    )


def test_write_reliability_receipt_rebuilds_explicit_allowlisted_json(
    receipt_with_sensitive_trial_metadata, tmp_path
):
    import scripts.run_objective_reliability as reliability

    output = tmp_path / "allowlisted.json"
    reliability.write_reliability_receipt(output, receipt_with_sensitive_trial_metadata)
    raw = output.read_text()
    payload = json.loads(raw)

    assert set(payload) == {
        "model", "environment", "requested_trials", "completed_trials",
        "argmax_successes", "clean_first_request_trials", "retry_assisted_trials",
        "requested_end_to_end_runs", "completed_end_to_end_runs",
        "message_pairing_passes", "claim_safety_passes",
        "production_temperature_zero", "objective_trials", "end_to_end_runs",
        "failed_trials",
    }
    assert set(payload["objective_trials"][0]) == {
        "kind", "index", "model", "environment", "completed",
        "argmax_success", "accepted_attempt_count", "rejected_selection_count",
        "correction_prompts_sent", "selection_response_count",
        "provider_request_attempt_count", "retry_assisted", "baseline_score",
        "target_score", "final_score", "termination_reason",
        "message_pairing_passed", "claim_safety_passed",
        "production_temperature_zero", "conclusion_status",
    }
    for sentinel in (
        "nvapi-SENSITIVE-SENTINEL",
        "RAW-RESPONSE-SENTINEL",
        "ARBITRARY-METADATA-SENTINEL",
        "api_key",
        "raw_model_response",
        "metadata",
    ):
        assert sentinel not in raw


def test_writer_rejects_nested_values_in_every_allowlisted_trial_field_atomically(
    receipt_with_sensitive_trial_metadata, tmp_path
):
    import scripts.run_objective_reliability as reliability

    target = tmp_path / "existing.json"
    target.write_text("ORIGINAL")
    base = receipt_with_sensitive_trial_metadata.objective_trials[0]
    for field in reliability._TRIAL_RECEIPT_FIELDS:
        forged = {**base, field: {"api_key": f"SECRET-{field}"}}
        receipt = replace(
            receipt_with_sensitive_trial_metadata,
            objective_trials=(forged,),
        )
        with pytest.raises((TypeError, ValueError)):
            reliability.write_reliability_receipt(target, receipt)
        assert target.read_text() == "ORIGINAL"
        assert not list(tmp_path.glob(f".{target.name}.*.tmp"))


def test_atomic_writer_preserves_existing_directory_target_and_cleans_temp(
    receipt_with_sensitive_trial_metadata, tmp_path
):
    import scripts.run_objective_reliability as reliability

    target = tmp_path / "target-directory"
    target.mkdir()
    marker = target / "marker"
    marker.write_text("ORIGINAL")
    with pytest.raises(OSError):
        reliability.write_reliability_receipt(
            target, receipt_with_sensitive_trial_metadata
        )
    assert marker.read_text() == "ORIGINAL"
    assert not list(tmp_path.glob(f".{target.name}.*.tmp"))
