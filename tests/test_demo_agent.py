import copy
import builtins
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from openai import AuthenticationError, PermissionDeniedError
from pydantic import ValidationError

import demo_agent
from chemistry_workflow import (
    EvidenceRecord,
    StageResult,
    WorkflowPhase,
    WorkflowReport,
    WorkflowState,
)


VALID_API_KEY = "nvapi-test-key"
SECRET = "do-not-leak-this-secret"
STAGES = (
    "inspect_library",
    "generate_morgan_fingerprints",
    "measure_tanimoto_similarity",
    "discover_fused_butina_clusters",
    "embed_representative_conformers",
    "optimize_conformers_mmff94",
)
PHASES = (
    WorkflowPhase.INSPECTED,
    WorkflowPhase.FINGERPRINTED,
    WorkflowPhase.COMPARED,
    WorkflowPhase.CLUSTERED,
    WorkflowPhase.EMBEDDED,
    WorkflowPhase.OPTIMIZED,
)
VALID_ARGS = {
    "inspect_library": {},
    "generate_morgan_fingerprints": {
        "radius": 2,
        "size": 1024,
        "decision_basis": "Use a compact standard fingerprint.",
    },
    "measure_tanimoto_similarity": {},
    "discover_fused_butina_clusters": {
        "cutoff": 0.5,
        "decision_basis": "Use the observed similarity spread.",
    },
    "embed_representative_conformers": {
        "representative_count": 4,
        "policy": "include_singleton_if_available",
        "conformers_per_representative": 4,
        "decision_basis": "Sample diverse clusters and one singleton.",
    },
    "optimize_conformers_mmff94": {},
}


def call(name, arguments, call_id=None, call_type="function"):
    return SimpleNamespace(
        id=call_id or f"call-{name}",
        type=call_type,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def response(name, arguments, **kwargs):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=kwargs.get("content"),
                    tool_calls=[
                        call(
                            name,
                            arguments
                            if isinstance(arguments, str)
                            else json.dumps(arguments),
                            call_id=kwargs.get("call_id"),
                            call_type=kwargs.get("call_type", "function"),
                        )
                    ],
                )
            )
        ]
    )


def plan_arguments(stages=STAGES):
    return {
        "stages": [
            {
                "stage": stage,
                "rationale": f"Run {stage.replace('_', ' ')} after its prerequisite.",
            }
            for stage in stages
        ]
    }


def valid_responses():
    return [response("submit_workflow_plan", plan_arguments())] + [
        response(stage, VALID_ARGS[stage]) for stage in STAGES
    ]


THEMES = (
    "dataset_scope", "molecular_representation", "similarity_structure",
    "clustering", "conformational_sampling", "limitations_and_next_steps",
)
REQUIRED_KEYS = (("E01",), ("E02",), ("E03",), ("E04",), ("E05", "E06"), ("E01", "E06"))


def test_notebook_preflight_checks_cuda_and_exact_nvmolkit_capabilities(monkeypatch):
    imported = []
    capabilities = {
        "nvmolkit.fingerprints": "MorganFingerprintGenerator",
        "nvmolkit.similarity": "crossTanimotoSimilarity",
        "nvmolkit.clustering": "fused_butina",
        "nvmolkit.embedMolecules": "EmbedMolecules",
        "nvmolkit.mmffOptimization": "MMFFOptimizeMoleculesConfs",
    }

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            return SimpleNamespace(cuda=FakeCuda())
        if name in capabilities:
            imported.append(name)
            return SimpleNamespace(**{capabilities[name]: object()})
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setenv("NVIDIA_API_KEY", VALID_API_KEY)

    assert demo_agent.notebook_preflight() == VALID_API_KEY
    assert imported == list(capabilities)


def test_notebook_preflight_reads_protected_launch_key_without_prompt(
    monkeypatch, capsys, tmp_path
):
    class FakeCuda:
        @staticmethod
        def is_available():
            return True

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            return SimpleNamespace(cuda=FakeCuda())
        if name.startswith("nvmolkit."):
            entry_point = {
                "nvmolkit.fingerprints": "MorganFingerprintGenerator",
                "nvmolkit.similarity": "crossTanimotoSimilarity",
                "nvmolkit.clustering": "fused_butina",
                "nvmolkit.embedMolecules": "EmbedMolecules",
                "nvmolkit.mmffOptimization": "MMFFOptimizeMoleculesConfs",
            }[name]
            return SimpleNamespace(**{entry_point: object()})
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    key_path = tmp_path / "NVIDIA_API_KEY"
    key_path.write_text(VALID_API_KEY, encoding="utf-8")
    key_path.chmod(0o600)
    monkeypatch.setattr(demo_agent, "_NVIDIA_API_KEY_PATH", key_path, raising=False)
    monkeypatch.setattr(
        "getpass.getpass",
        lambda prompt: pytest.fail("notebook preflight must not prompt for a key"),
    )

    assert demo_agent.notebook_preflight() == VALID_API_KEY
    captured = capsys.readouterr()
    assert VALID_API_KEY not in captured.out + captured.err


def test_notebook_preflight_fails_closed_when_launch_key_is_missing(
    monkeypatch, tmp_path
):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setattr(
        demo_agent,
        "_NVIDIA_API_KEY_PATH",
        tmp_path / "missing" / "NVIDIA_API_KEY",
        raising=False,
    )
    monkeypatch.setattr(
        "getpass.getpass",
        lambda prompt: pytest.fail("notebook preflight must not prompt for a key"),
    )

    with pytest.raises(ValueError, match="Redeploy.*NVIDIA_API_KEY"):
        demo_agent.notebook_preflight()


def test_notebook_preflight_rejects_launch_key_with_unsafe_permissions(
    monkeypatch, tmp_path
):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    key_path = tmp_path / "NVIDIA_API_KEY"
    key_path.write_text(VALID_API_KEY, encoding="utf-8")
    key_path.chmod(0o644)
    monkeypatch.setattr(demo_agent, "_NVIDIA_API_KEY_PATH", key_path, raising=False)

    with pytest.raises(ValueError, match="mode 0600"):
        demo_agent.notebook_preflight()


def test_notebook_preflight_rejects_non_cpython_312(monkeypatch):
    monkeypatch.setattr(
        sys, "implementation", SimpleNamespace(name="pypy")
    )
    with pytest.raises(AssertionError, match="CPython 3.12"):
        demo_agent.notebook_preflight()


def test_notebook_preflight_rejects_non_cuda_runtime(monkeypatch):
    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    real_import = builtins.__import__
    monkeypatch.setattr(
        builtins,
        "__import__",
        lambda name, *args, **kwargs: (
            SimpleNamespace(cuda=FakeCuda())
            if name == "torch"
            else real_import(name, *args, **kwargs)
        ),
    )
    with pytest.raises(AssertionError, match="CUDA"):
        demo_agent.notebook_preflight()


def full_report():
    return WorkflowReport(tuple(EvidenceRecord(f"E0{i}", f"Evidence {i}", "{}", "test") for i in range(1, 7)))


def presentation_report():
    payloads = (
        {"raw_count": 26, "valid_count": 24, "invalid_count": 2, "invalid_ids": ["bad-a", "bad-b"], "preview_count": 24},
        {"fingerprint_radius": 2, "fingerprint_size_bits": 1024, "molecule_count": 24, "active_bits_min": 7, "active_bits_median": 12.5, "active_bits_max": 21},
        {"q1": 0.08, "median": 0.14, "q3": 0.27, "p90": 0.41, "max_off_diagonal": 0.82, "most_similar_pair": {"molecule_ids": ["mol-a", "mol-b"], "similarity": 0.82}},
        {"cutoff": 0.5, "cluster_count": 8, "singleton_count": 3, "largest_cluster_sizes": [7, 4, 3]},
        {"requested_representative_count": 4, "selected_representative_count": 4, "selection_shortfall": 0, "representative_policy": "include_singleton_if_available", "representatives": [{"molecule_id": "mol-a", "cluster_id": 0}, {"molecule_id": "mol-c", "cluster_id": 1}], "generated_conformer_count": 16, "partial_embedding_ids": [], "zero_embedding_ids": []},
        {"attempted_conformer_count": 16, "converged_conformer_count": 15, "unconverged_conformer_count": 1, "per_conformer_records": [{"molecule_id": "must-not-render"}], "selected_conformer_records": [{"molecule_id": "mol-a", "conformer_index": 2, "energy_kcal_mol": 11.25}, {"molecule_id": "mol-c", "conformer_index": 0, "energy_kcal_mol": 4.5}]},
    )
    return WorkflowReport(tuple(
        EvidenceRecord(f"E0{i}", f"Evidence {i}", json.dumps(payload), "test")
        for i, payload in enumerate(payloads, 1)
    ))


def synthesis_arguments():
    return {
        "headline": "A coherent chemical library with bounded structural diversity",
        "sections": [
            {"theme": theme, "prose": f"Qualitative interpretation for {theme.replace('_', ' ')}.", "evidence_keys": list(keys)}
            for theme, keys in zip(THEMES, REQUIRED_KEYS)
        ],
    }


def workflow_responses(synthesis=None):
    return valid_responses() + [response("submit_synthesis", synthesis or synthesis_arguments())]


class FakeCompletions:
    def __init__(self, responses=None, error=None):
        self.responses = list(responses or [])
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        if self.error is not None:
            raise self.error
        if not self.responses:
            return SimpleNamespace(choices=[])
        return self.responses.pop(0)


def fake_client(completions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def fake_executors(record=None):
    calls = record if record is not None else []
    executors = {}
    for stage, next_phase in zip(STAGES, PHASES):
        def execute(state, _stage=stage, _phase=next_phase, **kwargs):
            calls.append((_stage, kwargs))
            summary = {"stage": _stage, "sequence": len(calls)}
            state.summaries[_stage] = summary
            state.phase = _phase
            return StageResult(_stage, _stage, summary, (f"figure-{_stage}",))

        executors[stage] = execute
    executors["build_workflow_report"] = lambda state: WorkflowReport(
        evidence=(EvidenceRecord("E01", "test", "{}", "test"),)
    )
    return executors


def run(responses=None, **kwargs):
    completions = FakeCompletions(responses or valid_responses())
    result = demo_agent.run_scientific_loop(
        "Analyze the bundled molecular library.",
        VALID_API_KEY,
        client=fake_client(completions),
        executors=fake_executors(),
        **kwargs,
    )
    return result, completions


def controller(responses=None, *, executors=None, state=None):
    completions = FakeCompletions(responses or valid_responses())
    instance = demo_agent.BoundedWorkflowController.create(
        "Analyze the bundled molecular library.",
        VALID_API_KEY,
        client=fake_client(completions),
        executors=executors or fake_executors(),
        state=state,
    )
    return instance, completions


def request_error(status, error_type):
    http_response = httpx.Response(
        status,
        request=httpx.Request("POST", f"{demo_agent.NVIDIA_BASE_URL}/chat/completions"),
    )
    return error_type(SECRET, response=http_response, body=None)


def test_stateful_loop_uses_one_history_and_exactly_seven_hosted_turns():
    result, completions = run()

    assert result.turn_count == 7 == len(completions.calls)
    assert result.report.evidence[0].key == "E01"
    roles = [message["role"] for message in result.messages]
    assert roles[:2] == ["system", "user"]
    assert roles.count("assistant") == 7
    assert roles.count("tool") == 7
    for index, hosted_call in enumerate(completions.calls):
        assert hosted_call["messages"] == list(result.messages[: 2 + 2 * index])


def test_controller_waits_for_approval_before_executing_stage():
    calls = []
    instance, completions = controller(executors=fake_executors(calls))

    instance.request_plan()
    proposal = instance.request_next_stage()

    assert proposal == demo_agent.StageProposal(
        "inspect_library", demo_agent.InspectionArgs()
    )
    assert calls == []
    assert instance.session.state.phase is WorkflowPhase.NEW
    assert len(completions.calls) == 2

    result = instance.execute_pending(proposal.arguments)

    assert result.stage == "inspect_library"
    assert calls == [("inspect_library", {})]
    assert instance.pending is None


def test_controller_records_user_override_and_executed_arguments():
    calls = []
    instance, _ = controller(executors=fake_executors(calls))
    instance.request_plan()
    instance.request_next_stage()
    instance.execute_pending(demo_agent.InspectionArgs())
    proposed = instance.request_next_stage()
    approved = demo_agent.FingerprintArgs(
        radius=3,
        size=2048,
        decision_basis=proposed.arguments.decision_basis,
    )

    instance.execute_pending(approved)

    history = json.loads(instance.session.messages[-1]["content"])
    assert history["proposed_arguments"]["radius"] == 2
    assert history["proposed_arguments"]["size"] == 1024
    assert history["executed_arguments"]["radius"] == 3
    assert history["executed_arguments"]["size"] == 2048
    assert history["user_override"] is True
    assert calls[-1] == (
        "generate_morgan_fingerprints",
        {"fingerprint_radius": 3, "fingerprint_size": 2048},
    )


def test_controller_rejects_wrong_approval_model_before_execution():
    calls = []
    instance, _ = controller(executors=fake_executors(calls))
    instance.request_plan()
    instance.request_next_stage()

    with pytest.raises(demo_agent.ToolCallError, match="approved arguments"):
        instance.execute_pending(demo_agent.SimilarityArgs())

    assert calls == []
    assert instance.pending is not None


def test_controller_rejects_decision_basis_edit_before_execution():
    calls = []
    instance, _ = controller(executors=fake_executors(calls))
    instance.request_plan()
    inspection = instance.request_next_stage()
    instance.execute_pending(inspection.arguments)
    proposal = instance.request_next_stage()
    edited = demo_agent.FingerprintArgs(
        radius=proposal.arguments.radius,
        size=proposal.arguments.size,
        decision_basis="Human-authored replacement rationale.",
    )

    with pytest.raises(demo_agent.ToolCallError, match="decision summary") as error:
        instance.execute_pending(edited)

    assert SECRET not in str(error.value)
    assert calls == [("inspect_library", {})]
    assert instance.pending is proposal


def test_controller_plan_can_be_requested_exactly_once():
    instance, completions = controller()

    instance.request_plan()
    with pytest.raises(demo_agent.ToolCallError, match="exactly once"):
        instance.request_plan()

    assert len(completions.calls) == 1


def test_controller_pending_proposal_prevents_another_hosted_call():
    instance, completions = controller()
    instance.request_plan()
    instance.request_next_stage()

    with pytest.raises(demo_agent.ToolCallError, match="pending"):
        instance.request_next_stage()

    assert len(completions.calls) == 2


def test_controller_executor_failure_retains_pending_and_blocks_unsafe_retry():
    calls = []
    executors = fake_executors(calls)

    def fail_after_advancing(state):
        calls.append(("inspect_library", {}))
        state.phase = WorkflowPhase.INSPECTED
        raise RuntimeError(SECRET)

    executors["inspect_library"] = fail_after_advancing
    instance, _ = controller(executors=executors)
    instance.request_plan()
    proposal = instance.request_next_stage()

    with pytest.raises(demo_agent.ToolCallError, match="executor failed"):
        instance.execute_pending(proposal.arguments)

    assert instance.pending is proposal
    with pytest.raises(demo_agent.ToolCallError, match="out of phase"):
        instance.execute_pending(proposal.arguments)
    assert calls == [("inspect_library", {})]


def test_controller_retries_retained_pending_after_nonmutating_executor_failure():
    calls = []
    executors = fake_executors(calls)

    def fail_once_then_advance(state):
        calls.append(("inspect_library", {}))
        if len(calls) == 1:
            raise RuntimeError(SECRET)
        state.phase = WorkflowPhase.INSPECTED
        return StageResult(
            "inspect_library",
            "inspect_library",
            {"stage": "inspect_library", "sequence": len(calls)},
        )

    executors["inspect_library"] = fail_once_then_advance
    instance, _ = controller(executors=executors)
    instance.request_plan()
    proposal = instance.request_next_stage()
    messages_before_execution = len(instance.session.messages)

    with pytest.raises(demo_agent.ToolCallError, match="executor failed"):
        instance.execute_pending(proposal.arguments)

    assert instance.session.state.phase is WorkflowPhase.NEW
    assert instance.pending is proposal
    assert len(instance.session.messages) == messages_before_execution
    assert instance.stage_results == []

    result = instance.execute_pending(proposal.arguments)

    assert result.stage == "inspect_library"
    assert calls == [("inspect_library", {}), ("inspect_library", {})]
    assert instance.pending is None
    assert instance.session.state.phase is WorkflowPhase.INSPECTED
    assert instance.stage_results == [result]
    assert len(instance.session.messages) == messages_before_execution + 1
    assert instance.session.messages[-1]["role"] == "tool"


def test_controller_scientific_and_synthesis_paths_use_seven_then_eight_turns():
    completions = FakeCompletions(workflow_responses())
    instance = demo_agent.BoundedWorkflowController.create(
        "Analyze the library.",
        VALID_API_KEY,
        client=fake_client(completions),
        executors={
            **fake_executors(),
            "build_workflow_report": lambda state: full_report(),
        },
    )
    instance.request_plan()
    for _stage in STAGES:
        proposal = instance.request_next_stage()
        instance.execute_pending(proposal.arguments)

    scientific = instance.scientific_result()
    assert scientific.turn_count == 7

    result = instance.request_synthesis()
    assert result.turn_count == 8 == len(completions.calls)
    assert result.stage_results == scientific.stage_results


def test_controller_synthesis_retry_reuses_one_evidence_prompt():
    empty = SimpleNamespace(choices=[])
    completions = FakeCompletions(
        valid_responses()
        + [empty, response("submit_synthesis", synthesis_arguments())]
    )
    instance = demo_agent.BoundedWorkflowController.create(
        "Analyze the library.",
        VALID_API_KEY,
        client=fake_client(completions),
        executors={
            **fake_executors(),
            "build_workflow_report": lambda state: full_report(),
        },
    )
    instance.request_plan()
    for _stage in STAGES:
        proposal = instance.request_next_stage()
        instance.execute_pending(proposal.arguments)

    with pytest.raises(demo_agent.ToolCallError, match="strict validation"):
        instance.request_synthesis()

    assert instance.session.turn_count == 7
    assert sum(
        message["role"] == "user"
        and message["content"].startswith(demo_agent._SYNTHESIS_PROMPT)
        for message in instance.session.messages
    ) == 1

    result = instance.request_synthesis()

    assert result.turn_count == 8
    assert sum(
        message["role"] == "user"
        and message["content"].startswith(demo_agent._SYNTHESIS_PROMPT)
        for message in result.messages
    ) == 1


def test_every_assistant_call_has_a_matching_canonical_tool_result():
    result, _ = run()

    for index, message in enumerate(result.messages):
        if message["role"] != "assistant":
            continue
        tool_message = result.messages[index + 1]
        tool_call = message["tool_calls"][0]
        assert tool_message["role"] == "tool"
        assert tool_message["tool_call_id"] == tool_call["id"]
        decoded = json.loads(tool_message["content"])
        assert tool_message["content"] == json.dumps(
            decoded, sort_keys=True, separators=(",", ":"), allow_nan=False
        )


def test_skill_is_exact_initial_grounding_not_an_artificial_tool():
    result, completions = run()
    skill = Path("skills/nvmolkit/SKILL.md").read_text()
    system = result.messages[0]["content"]

    assert "skills/nvmolkit/SKILL.md" in system
    assert skill in system
    assert "RDKit input validation" in system
    assert "nvMolKit GPU" in system
    assert "concise decision summaries" in system
    assert "hidden chain-of-thought" in system
    assert "read_nvmolkit_skill" not in system
    exposed_names = {
        hosted["tools"][0]["function"]["name"] for hosted in completions.calls
    }
    assert exposed_names == {"submit_workflow_plan", *STAGES}
    assert "read_nvmolkit_skill" not in exposed_names


def test_explicit_bundled_skill_is_supplied_as_initial_grounding():
    skill = Path("skills/nvmolkit/SKILL.md").read_text(encoding="utf-8")

    result, _ = run(skill=skill)

    assert skill in result.messages[0]["content"]


def test_each_turn_exposes_and_forces_only_the_phase_eligible_schema():
    _, completions = run()

    expected = ("submit_workflow_plan", *STAGES)
    for hosted, tool_name in zip(completions.calls, expected):
        assert len(hosted["tools"]) == 1
        function = hosted["tools"][0]["function"]
        assert function["name"] == tool_name
        assert function["strict"] is True
        assert function["parameters"]["additionalProperties"] is False
        assert hosted["tool_choice"] == {
            "type": "function",
            "function": {"name": tool_name},
        }


def test_plan_requires_exact_dependency_order_and_nonempty_rationales():
    parsed = demo_agent.WorkflowPlan.model_validate(plan_arguments())
    assert tuple(stage.stage for stage in parsed.stages) == STAGES
    with pytest.raises(ValidationError):
        demo_agent.WorkflowPlan.model_validate(plan_arguments(reversed(STAGES)))
    bad = plan_arguments()
    bad["stages"][2]["rationale"] = ""
    with pytest.raises(ValidationError):
        demo_agent.WorkflowPlan.model_validate(bad)


@pytest.mark.parametrize(
    "value",
    ["", "valid rationale\ncontinued", "use `code` here", "x" * 241],
)
def test_decision_basis_rejects_unpresentable_or_unbounded_text(value):
    with pytest.raises(ValidationError):
        demo_agent.FingerprintArgs.model_validate(
            {"radius": 2, "size": 1024, "decision_basis": value}
        )


def test_decision_basis_accepts_brief_nonempty_text():
    parsed = demo_agent.FingerprintArgs.model_validate(
        {"radius": 2, "size": 1024, "decision_basis": "brief"}
    )

    assert parsed.decision_basis == "brief"


@pytest.mark.parametrize(
    "value",
    [
        "selected_ids",
        "decision_basis",
        "Selected IDs / selected-ids",
        "decision basis decision_basis",
        "selected_ids and decision_basis",
        "The selected IDs decision basis",
        "selectedIds",
        "selectedids",
        "decisionBasis",
        "decisionbasis",
        "decisionbases",
        "selectedIds decisionBasis",
        "SELECTEDIDS + DECISIONBASES",
        "selectedidsdecisionbasis",
        "selectedIdsDecisionBasis",
        "decisionbasisselectedids",
        "decisionBasisSelectedIds",
        "selected_idsdecision_basis",
        "decision_basisselected_ids",
    ],
)
def test_objective_selection_rejects_schema_field_rationale_values(value):
    with pytest.raises(ValidationError):
        demo_agent.ObjectiveSelection.model_validate({
            "state_id": "state-0123456789abcdef",
            "swap_id": "mol-0->mol-4",
            "observed_limiting_pairs": [["mol-0", "mol-1"]],
            "decision_rule": "maximize_predicted_minimum_distance",
            "decision_basis": value,
        })


@pytest.mark.parametrize(
    "value",
    ["0.80 exceeds 0.71.", "Score 0.8 > 0.7."],
)
def test_objective_selection_rejects_even_short_quantitative_model_reasons(value):
    with pytest.raises(ValidationError):
        demo_agent.ObjectiveSelection.model_validate({
            "state_id": "state-0123456789abcdef",
            "swap_id": "mol-0->mol-4",
            "observed_limiting_pairs": [["mol-0", "mol-1"]],
            "decision_rule": "maximize_predicted_minimum_distance",
            "rationale": value,
        })


@pytest.mark.parametrize(
    "value",
    [
        "Decisionmaking improves score to 0.8.",
        "selectedidsdecisionbasis improves score to 0.8.",
        "selectedidsdecisionbasisish is unrelated prose.",
    ],
)
def test_objective_selection_rejects_arbitrary_model_prose(value):
    with pytest.raises(ValidationError):
        demo_agent.ObjectiveSelection.model_validate({
            "state_id": "state-0123456789abcdef",
            "swap_id": "mol-0->mol-4",
            "observed_limiting_pairs": [["mol-0", "mol-1"]],
            "decision_rule": "maximize_predicted_minimum_distance",
            "model_prose": value,
        })


def test_stage_decision_basis_does_not_inherit_objective_rationale_floor():
    parsed = demo_agent.FingerprintArgs.model_validate(
        {"radius": 2, "size": 1024, "decision_basis": "selected_ids"}
    )

    assert parsed.decision_basis == "selected_ids"


def test_objective_selection_schema_is_exact_bounded_and_contains_no_rationale():
    assert list(demo_agent.ObjectiveSelection.model_fields) == [
        "state_id",
        "swap_id",
        "observed_limiting_pairs",
        "decision_rule",
    ]
    assert "ObjectiveDecisionBasis" not in vars(demo_agent)
    parsed = demo_agent.ObjectiveSelection.model_validate({
        "state_id": "state-0123456789abcdef",
        "swap_id": "mol-0->mol-4",
        "observed_limiting_pairs": [["mol-0", "mol-1"]],
        "decision_rule": "maximize_predicted_minimum_distance",
    })
    assert parsed.swap_id == "mol-0->mol-4"
    for bad in ("a->b->c", "a ->b", "a-> b", "a`->b"):
        with pytest.raises(ValidationError):
            demo_agent.ObjectiveSelection.model_validate({
                **parsed.model_dump(), "swap_id": bad,
            })


@pytest.mark.parametrize(
    ("model", "arguments"),
    [
        ("InspectionArgs", {"extra": True}),
        ("SimilarityArgs", {"extra": True}),
        ("OptimizationArgs", {"extra": True}),
        ("FingerprintArgs", {"radius": 1, "size": 1024, "decision_basis": "A valid scientific reason."}),
        ("FingerprintArgs", {"radius": 2, "size": "1024", "decision_basis": "A valid scientific reason."}),
        ("ClusterArgs", {"cutoff": 0.39, "decision_basis": "A valid scientific reason."}),
        ("ClusterArgs", {"cutoff": "0.5", "decision_basis": "A valid scientific reason."}),
        ("EmbedArgs", {"representative_count": 7, "policy": "largest_clusters_first", "conformers_per_representative": 4, "decision_basis": "A valid scientific reason."}),
        ("EmbedArgs", {"representative_count": 4, "policy": "unknown", "conformers_per_representative": 4, "decision_basis": "A valid scientific reason."}),
        ("EmbedArgs", {"representative_count": 4, "policy": "largest_clusters_first", "conformers_per_representative": 2, "decision_basis": "A valid scientific reason."}),
    ],
)
def test_argument_models_are_strict_bounded_frozen_and_forbid_extras(model, arguments):
    model = getattr(demo_agent, model)
    assert model.model_config["strict"] is True
    assert model.model_config["frozen"] is True
    assert model.model_config["extra"] == "forbid"
    with pytest.raises(ValidationError):
        model.model_validate(arguments)


def test_adaptive_choices_receive_prerequisite_summaries_in_same_history():
    _, completions = run()
    cluster_history = completions.calls[4]["messages"]
    embed_history = completions.calls[5]["messages"]

    assert any(
        message["role"] == "tool"
        and "measure_tanimoto_similarity" in message["content"]
        for message in cluster_history
    )
    assert any(
        message["role"] == "tool"
        and "discover_fused_butina_clusters" in message["content"]
        for message in embed_history
    )


def test_decision_basis_is_retained_in_history_but_not_passed_to_executors():
    recorded = []
    completions = FakeCompletions(valid_responses())
    demo_agent.run_scientific_loop(
        "Analyze the library.",
        VALID_API_KEY,
        client=fake_client(completions),
        executors=fake_executors(recorded),
    )

    fingerprint_call = next(
        message
        for message in completions.calls[3]["messages"]
        if message["role"] == "assistant"
        and message["tool_calls"][0]["function"]["name"]
        == "generate_morgan_fingerprints"
    )
    assert json.loads(
        fingerprint_call["tool_calls"][0]["function"]["arguments"]
    )["decision_basis"]
    assert recorded == [
        ("inspect_library", {}),
        ("generate_morgan_fingerprints", {"fingerprint_radius": 2, "fingerprint_size": 1024}),
        ("measure_tanimoto_similarity", {}),
        ("discover_fused_butina_clusters", {"cluster_cutoff": 0.5}),
        ("embed_representative_conformers", {"representative_count": 4, "representative_policy": "include_singleton_if_available", "conformers_per_representative": 4}),
        ("optimize_conformers_mmff94", {}),
    ]


@pytest.mark.parametrize(
    "bad_tool_calls",
    [
        None,
        [],
        [call("inspect_library", "{}"), call("inspect_library", "{}")],
        [call("wrong_stage", "{}")],
        [call("inspect_library", "{}", call_id=" ")],
        [call("inspect_library", "{}", call_type="custom")],
        [call("inspect_library", "not-json")],
        [call("inspect_library", "[]")],
        [call("inspect_library", '{"extra":true}')],
    ],
)
def test_invalid_scientific_calls_fail_before_any_executor(bad_tool_calls):
    plan = response("submit_workflow_plan", plan_arguments())
    bad = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="early text", tool_calls=bad_tool_calls))]
    )
    completions = FakeCompletions([plan, bad])
    executor_calls = []

    with pytest.raises(demo_agent.ToolCallError):
        demo_agent.run_scientific_loop(
            "Analyze the library.",
            VALID_API_KEY,
            client=fake_client(completions),
            executors=fake_executors(executor_calls),
        )

    assert executor_calls == []
    assert len(completions.calls) == (
        3 if bad_tool_calls is None or bad_tool_calls == [] else 2
    )


@pytest.mark.parametrize(
    "content",
    [[{"type": "text", "text": "unexpected"}], {"text": "unexpected"}, 1],
)
def test_nonstring_assistant_content_stops_before_executor(content):
    plan = response("submit_workflow_plan", plan_arguments())
    invalid = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                    tool_calls=[call("inspect_library", "{}")],
                )
            )
        ]
    )
    completions = FakeCompletions([plan, invalid])
    executor_calls = []

    with pytest.raises(demo_agent.ToolCallError, match="content"):
        demo_agent.run_scientific_loop(
            "Analyze the library.",
            VALID_API_KEY,
            client=fake_client(completions),
            executors=fake_executors(executor_calls),
        )

    assert executor_calls == []
    assert len(completions.calls) == 2


def test_string_assistant_content_is_discarded_before_valid_tool_call_is_retained():
    session = demo_agent.AgentSession(
        messages=[{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
        state=WorkflowState(),
    )
    completions = FakeCompletions(
        [response("inspect_library", {}, content="I will inspect the library.")]
    )

    parsed = demo_agent._request_call(
        session,
        fake_client(completions),
        "inspect_library",
        demo_agent.InspectionArgs,
        demo_agent.DEFAULT_MODEL,
    )

    assert parsed == demo_agent.InspectionArgs()
    assert session.messages[-1]["content"] is None
    assert session.messages[-1]["tool_calls"][0]["function"]["name"] == "inspect_library"


def test_leading_json_content_is_validated_as_the_forced_tool_call():
    session = demo_agent.AgentSession(
        messages=[{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
        state=WorkflowState(),
    )
    arguments = {
        "stage": "discover_fused_butina_clusters",
        **VALID_ARGS["discover_fused_butina_clusters"],
    }
    content_only = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(arguments) + "\nI selected this bounded cutoff.",
                    tool_calls=None,
                )
            )
        ]
    )
    completions = FakeCompletions([content_only])

    parsed = demo_agent._request_call(
        session,
        fake_client(completions),
        "discover_fused_butina_clusters",
        demo_agent.ClusterArgs,
        demo_agent.DEFAULT_MODEL,
    )

    assert parsed == demo_agent.ClusterArgs.model_validate(
        VALID_ARGS["discover_fused_butina_clusters"]
    )
    retained = session.messages[-1]["tool_calls"][0]
    assert retained["id"].startswith("compat-")
    assert retained["function"]["name"] == "discover_fused_butina_clusters"
    assert json.loads(retained["function"]["arguments"]) == VALID_ARGS[
        "discover_fused_butina_clusters"
    ]


def test_hosted_numeric_string_cutoff_is_normalized_before_strict_validation():
    session = demo_agent.AgentSession(
        messages=[{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
        state=WorkflowState(),
    )
    hosted_arguments = {
        **VALID_ARGS["discover_fused_butina_clusters"],
        "cutoff": "0.5",
    }
    completions = FakeCompletions(
        [response("discover_fused_butina_clusters", hosted_arguments)]
    )

    parsed = demo_agent._request_call(
        session,
        fake_client(completions),
        "discover_fused_butina_clusters",
        demo_agent.ClusterArgs,
        demo_agent.DEFAULT_MODEL,
    )

    assert parsed.cutoff == 0.5
    retained = json.loads(
        session.messages[-1]["tool_calls"][0]["function"]["arguments"]
    )
    assert retained["cutoff"] == 0.5


def test_text_only_response_gets_two_bounded_retries_before_any_executor():
    text_only = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="I will call the tool.", tool_calls=[])
            )
        ]
    )
    session = demo_agent.AgentSession(
        messages=[{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
        state=WorkflowState(),
    )
    completions = FakeCompletions(
        [text_only, text_only, response("inspect_library", {})]
    )

    parsed = demo_agent._request_call(
        session,
        fake_client(completions),
        "inspect_library",
        demo_agent.InspectionArgs,
        demo_agent.DEFAULT_MODEL,
    )

    assert parsed == demo_agent.InspectionArgs()
    assert len(completions.calls) == 3
    assert session.turn_count == 1


def test_text_only_response_retries_are_limited_to_two():
    text_only = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="I will call the tool.", tool_calls=[])
            )
        ]
    )
    session = demo_agent.AgentSession(
        messages=[{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
        state=WorkflowState(),
    )
    completions = FakeCompletions([text_only, text_only, text_only])

    with pytest.raises(demo_agent.ToolCallError, match="exactly one"):
        demo_agent._request_call(
            session,
            fake_client(completions),
            "inspect_library",
            demo_agent.InspectionArgs,
            demo_agent.DEFAULT_MODEL,
        )

    assert len(completions.calls) == 3
    assert session.turn_count == 0


def test_decision_basis_is_compacted_before_strict_validation():
    session = demo_agent.AgentSession(
        messages=[{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
        state=WorkflowState(),
    )
    arguments = {
        "cutoff": 0.5,
        "decision_basis": "Use the observed spread.\n" + ("detail " * 80) + "`note`",
    }
    completions = FakeCompletions(
        [response("discover_fused_butina_clusters", arguments)]
    )

    parsed = demo_agent._request_call(
        session,
        fake_client(completions),
        "discover_fused_butina_clusters",
        demo_agent.ClusterArgs,
        demo_agent.DEFAULT_MODEL,
    )

    assert 1 <= len(parsed.decision_basis) <= 240
    assert "\n" not in parsed.decision_basis
    assert "`" not in parsed.decision_basis
    assert parsed.decision_basis.endswith("...")


@pytest.mark.parametrize(
    ("arguments", "expected_fields"),
    [
        (
            {
                "fingerprint_radius": 2,
                "fingerprint_size": 1024,
                "decision_basis": "Use a compact molecular representation.",
            },
            {"radius", "size"},
        ),
    ],
)
def test_fingerprint_validation_error_exposes_only_safe_signature(
    arguments, expected_fields
):
    responses = [
        response("submit_workflow_plan", plan_arguments()),
        response("inspect_library", {}),
        response("generate_morgan_fingerprints", arguments),
    ]
    completions = FakeCompletions(responses)
    executor_calls = []

    with pytest.raises(demo_agent._HostedArgumentsValidationError) as captured:
        demo_agent.run_scientific_loop(
            "Analyze the library.",
            VALID_API_KEY,
            client=fake_client(completions),
            executors=fake_executors(executor_calls),
        )

    error = captured.value
    assert error.stage == "generate_morgan_fingerprints"
    assert {field for field, _error_type in error.issues} == expected_fields
    assert all(error_type for _field, error_type in error.issues)
    assert executor_calls == [("inspect_library", {})]
    rendered = str(error)
    assert "1024" not in rendered
    assert "compact molecular representation" not in rendered
    assert SECRET not in rendered


def test_redundant_matching_stage_metadata_is_removed_before_strict_validation():
    session = demo_agent.AgentSession(
        messages=[{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
        state=WorkflowState(),
    )
    arguments = {
        "stage": "discover_fused_butina_clusters",
        "summary": "Cluster at the selected bounded cutoff.",
        **VALID_ARGS["discover_fused_butina_clusters"],
    }
    completions = FakeCompletions(
        [response("discover_fused_butina_clusters", arguments)]
    )

    parsed = demo_agent._request_call(
        session,
        fake_client(completions),
        "discover_fused_butina_clusters",
        demo_agent.ClusterArgs,
        demo_agent.DEFAULT_MODEL,
    )

    assert parsed == demo_agent.ClusterArgs.model_validate(
        VALID_ARGS["discover_fused_butina_clusters"]
    )


def test_forced_call_uses_function_name_and_removes_mismatched_stage_metadata():
    session = demo_agent.AgentSession(
        messages=[{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
        state=WorkflowState(),
    )
    arguments = {
        "stage": "embed_representative_conformers",
        **VALID_ARGS["discover_fused_butina_clusters"],
    }
    completions = FakeCompletions(
        [response("discover_fused_butina_clusters", arguments)]
    )

    parsed = demo_agent._request_call(
        session,
        fake_client(completions),
        "discover_fused_butina_clusters",
        demo_agent.ClusterArgs,
        demo_agent.DEFAULT_MODEL,
    )

    assert parsed == demo_agent.ClusterArgs.model_validate(
        VALID_ARGS["discover_fused_butina_clusters"]
    )
    retained = json.loads(
        session.messages[-1]["tool_calls"][0]["function"]["arguments"]
    )
    assert retained == VALID_ARGS["discover_fused_butina_clusters"]


def test_text_json_fallback_mismatched_stage_metadata_fails_closed():
    session = demo_agent.AgentSession(
        messages=[{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
        state=WorkflowState(),
    )
    content_only = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(
                        {
                            "stage": "embed_representative_conformers",
                            **VALID_ARGS["discover_fused_butina_clusters"],
                        }
                    ),
                    tool_calls=None,
                )
            )
        ]
    )

    with pytest.raises(demo_agent.ToolCallError, match="out of phase"):
        demo_agent._request_call(
            session,
            fake_client(FakeCompletions([content_only])),
            "discover_fused_butina_clusters",
            demo_agent.ClusterArgs,
            demo_agent.DEFAULT_MODEL,
        )


def test_repeated_or_out_of_phase_call_stops_before_wrong_executor():
    responses = valid_responses()
    responses[2] = response("inspect_library", {})
    calls = []
    completions = FakeCompletions(responses)

    with pytest.raises(demo_agent.ToolCallError):
        demo_agent.run_scientific_loop(
            "Analyze.", VALID_API_KEY, client=fake_client(completions), executors=fake_executors(calls)
        )

    assert calls == [("inspect_library", {})]


def test_executor_failure_is_secret_safe_and_stops_later_calls():
    executors = fake_executors()
    executors["inspect_library"] = lambda state: (_ for _ in ()).throw(RuntimeError(SECRET))
    completions = FakeCompletions(valid_responses())

    with pytest.raises(demo_agent.ToolCallError) as error:
        demo_agent.run_scientific_loop(
            "Analyze.", VALID_API_KEY, client=fake_client(completions), executors=executors
        )

    assert SECRET not in str(error.value)
    assert len(completions.calls) == 2


def test_executor_cannot_report_success_without_exact_phase_advance():
    class SerializationProbe:
        calls = 0

        def tolist(self):
            self.calls += 1
            return ["serialized"]

    probe = SerializationProbe()
    executors = fake_executors()
    executors["inspect_library"] = lambda state: StageResult(
        "inspect_library", "inspect", {"probe": probe}
    )
    completions = FakeCompletions(valid_responses())

    with pytest.raises(demo_agent.ToolCallError, match="phase"):
        demo_agent.run_scientific_loop(
            "Analyze.", VALID_API_KEY, client=fake_client(completions), executors=executors
        )

    assert probe.calls == 0
    assert len(completions.calls) == 2


def test_nonfinite_tool_summary_is_rejected_before_next_hosted_turn():
    executors = fake_executors()
    def nonfinite(state):
        state.phase = WorkflowPhase.INSPECTED
        return StageResult("inspect_library", "inspect", {"value": float("nan")})
    executors["inspect_library"] = nonfinite
    completions = FakeCompletions(valid_responses())

    with pytest.raises(demo_agent.ToolCallError, match="serialized safely"):
        demo_agent.run_scientific_loop(
            "Analyze.", VALID_API_KEY, client=fake_client(completions), executors=executors
        )
    assert len(completions.calls) == 2


def test_hosted_configuration_disables_retries_and_thinking(monkeypatch):
    constructor_calls = []
    monkeypatch.setattr(demo_agent, "OpenAI", lambda **kwargs: constructor_calls.append(kwargs) or object())
    assert demo_agent._client(VALID_API_KEY) is not None
    assert constructor_calls == [{"base_url": demo_agent.NVIDIA_BASE_URL, "api_key": VALID_API_KEY, "max_retries": 0}]

    _, completions = run()
    for call_record in completions.calls:
        assert call_record["model"] == demo_agent.DEFAULT_MODEL
        assert call_record["temperature"] == 0.0
        assert call_record["stream"] is False
        assert call_record["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


@pytest.mark.parametrize("api_key", ["", "ngc-key", f"nvapi-{SECRET} "])
def test_invalid_key_is_secret_safe_and_stops_before_network(api_key):
    completions = FakeCompletions(valid_responses())
    with pytest.raises(ValueError) as error:
        demo_agent.run_scientific_loop(
            "Analyze.", api_key, client=fake_client(completions), executors=fake_executors()
        )
    assert api_key not in str(error.value) if api_key else True
    assert completions.calls == []


@pytest.mark.parametrize(
    ("hosted_error", "exception_type"),
    [
        (request_error(401, AuthenticationError), ValueError),
        (request_error(403, PermissionDeniedError), ValueError),
        (RuntimeError(SECRET), demo_agent.ToolCallError),
    ],
)
def test_hosted_errors_are_secret_safe(hosted_error, exception_type):
    completions = FakeCompletions(error=hosted_error)
    with pytest.raises(exception_type) as error:
        demo_agent.run_scientific_loop(
            "Analyze.", VALID_API_KEY, client=fake_client(completions), executors=fake_executors()
        )
    assert SECRET not in str(error.value)
    assert VALID_API_KEY not in str(error.value)
    assert len(completions.calls) == 1


def test_empty_hosted_response_fails_closed_without_retry():
    completions = FakeCompletions([])
    with pytest.raises(demo_agent.ToolCallError):
        demo_agent.run_scientific_loop(
            "Analyze.", VALID_API_KEY, client=fake_client(completions), executors=fake_executors()
        )
    assert len(completions.calls) == 1


def test_default_dispatcher_calls_only_fixed_chemistry_facade(monkeypatch):
    calls = []
    for stage, phase in zip(STAGES, PHASES):
        def facade(state, *args, _stage=stage, _phase=phase, **kwargs):
            calls.append((_stage, kwargs))
            state.phase = _phase
            return StageResult(_stage, _stage, {"stage": _stage})
        monkeypatch.setattr(demo_agent, stage, facade)
    report = WorkflowReport((EvidenceRecord("E01", "test", "{}", "test"),))
    monkeypatch.setattr(demo_agent, "build_workflow_report", lambda state: report)
    completions = FakeCompletions(valid_responses())

    result = demo_agent.run_scientific_loop(
        "Analyze.", VALID_API_KEY, client=fake_client(completions)
    )

    assert result.report is report
    assert [name for name, _ in calls] == list(STAGES)


def test_empty_executor_registry_is_rejected_without_using_production_tools(monkeypatch):
    default_calls = []
    monkeypatch.setattr(
        demo_agent,
        "_default_executors",
        lambda: default_calls.append("called") or fake_executors(),
    )
    completions = FakeCompletions(valid_responses())

    with pytest.raises(ValueError, match="fixed scientific workflow"):
        demo_agent.run_scientific_loop(
            "Analyze.", VALID_API_KEY, client=fake_client(completions), executors={}
        )

    assert default_calls == []
    assert completions.calls == []


def test_session_turn_limit_fails_closed():
    session = demo_agent.AgentSession(
        messages=[{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
        state=WorkflowState(),
        turn_count=7,
    )
    completions = FakeCompletions([response("submit_workflow_plan", plan_arguments())])
    with pytest.raises(demo_agent.ToolCallError, match="turn limit"):
        demo_agent._request_call(
            session,
            fake_client(completions),
            "submit_workflow_plan",
            demo_agent.WorkflowPlan,
            demo_agent.DEFAULT_MODEL,
        )
    assert completions.calls == []


def test_valid_conclusion_is_strict_frozen_and_evidence_grounded():
    conclusion = demo_agent.SubmitSynthesisArgs.model_validate(synthesis_arguments())

    assert demo_agent.validate_conclusion(conclusion, full_report()) is conclusion
    assert tuple(section.theme for section in conclusion.sections) == THEMES
    assert conclusion.model_config["strict"] is True
    with pytest.raises(ValidationError):
        conclusion.headline = "changed"


def test_conclusion_accepts_numeric_prose_and_global_evidence_coverage():
    arguments = synthesis_arguments()
    arguments["headline"] = "Two numerical regimes in one bounded library"
    arguments["sections"][4]["prose"] = "The sampled conformers include 2 regimes."
    arguments["sections"][4]["evidence_keys"] = ["E05"]
    arguments["sections"][5]["evidence_keys"] = ["E06"]
    conclusion = demo_agent.SubmitSynthesisArgs.model_validate(arguments)

    assert demo_agent.validate_conclusion(conclusion, full_report()) is conclusion


def test_conclusion_validation_keeps_structured_grounding_when_model_prose_cites_it():
    arguments = synthesis_arguments()
    arguments["sections"][0]["prose"] = "The dataset is described by E01."
    conclusion = demo_agent.SubmitSynthesisArgs.model_validate(arguments)

    assert demo_agent.validate_conclusion(conclusion, full_report()) is conclusion


def test_presentation_text_removes_model_generated_evidence_citations():
    prose = (
        "The dataset contains 256 molecules (E01). Similarity is broad (E02-E03). "
        "The optimized panel improves diversity (O01). Evidence: E01, E02, E03, O01."
    )

    rendered = demo_agent._presentation_text(prose)

    assert rendered == (
        "The dataset contains 256 molecules. Similarity is broad. "
        "The optimized panel improves diversity."
    )
    assert all(key not in rendered for key in ("E01", "E02", "E03", "O01"))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["sections"].__setitem__(1, copy.deepcopy(value["sections"][0])),
        lambda value: value["sections"].pop(),
        lambda value: value["sections"][0].__setitem__("evidence_keys", []),
        lambda value: value["sections"][0].__setitem__("evidence_keys", ["E09"]),
        lambda value: value["sections"][2].__setitem__("evidence_keys", ["E04"]),
        lambda value: (
            value["sections"][4].__setitem__("evidence_keys", ["E05"]),
            value["sections"][5].__setitem__("evidence_keys", ["E01"]),
        ),
    ],
    ids=("duplicate", "missing", "empty", "unknown", "global-missing", "all-global-missing"),
)
def test_invalid_conclusion_fails_closed_without_retaining_prose(mutation):
    arguments = synthesis_arguments()
    mutation(arguments)

    try:
        conclusion = demo_agent.SubmitSynthesisArgs.model_validate(arguments)
        with pytest.raises(demo_agent.ConclusionValidationError) as error:
            demo_agent.validate_conclusion(conclusion, full_report())
    except ValidationError:
        return

    assert error.value.report == full_report()
    assert error.value.rejected_prose is None
    assert arguments["headline"] not in str(error.value)


def test_run_workflow_adds_one_schema_checked_eighth_turn_and_retains_stage_results():
    completions = FakeCompletions(workflow_responses())
    result = demo_agent.run_workflow(
        "Analyze the library.", VALID_API_KEY, display_events=False,
        client=fake_client(completions), executors={**fake_executors(), "build_workflow_report": lambda state: full_report()},
    )

    assert result.turn_count == 8 == len(completions.calls)
    assert tuple(item.stage for item in result.plan.stages) == STAGES
    assert result.plan.stages[0].rationale == plan_arguments()["stages"][0]["rationale"]
    assert len(result.stage_results) == 6
    assert tuple(item.stage for item in result.stage_results) == STAGES
    assert result.messages[-1]["role"] == "assistant"
    assert result.messages[-1]["tool_calls"][0]["function"]["name"] == "submit_synthesis"
    assert not any(message["role"] == "tool" for message in result.messages[-1:])
    assert completions.calls[-1]["messages"] == list(result.messages[:-1])
    assert json.loads(result.messages[-2]["content"].split("\n", 1)[1]) == {
        "evidence": [record.__dict__ for record in full_report().evidence]
    }


def test_invalid_final_synthesis_stops_without_rendering_evidence_dump_or_rejected_prose(monkeypatch):
    shown = []
    monkeypatch.setattr("IPython.display.display", lambda *items: shown.extend(items))
    invalid = synthesis_arguments()
    invalid["headline"] = "Rejected synthesis"
    invalid["sections"][0]["evidence_keys"] = []
    completions = FakeCompletions(workflow_responses(invalid))

    with pytest.raises(demo_agent.ConclusionValidationError) as error:
        demo_agent.run_workflow(
            "Analyze.", VALID_API_KEY, display_events=True,
            client=fake_client(completions), executors={**fake_executors(), "build_workflow_report": lambda state: presentation_report()},
        )

    assert len(completions.calls) == 8
    assert error.value.report == presentation_report()
    assert error.value.rejected_prose is None
    assert "Rejected" not in str(error.value)
    rendered = "\n".join(getattr(item, "data", str(item)) for item in shown)
    assert "Workflow stopped" in rendered
    assert "Canonical evidence" not in rendered
    assert all(key not in rendered for key in ("E01", "E02", "E03", "E04", "E05", "E06"))
    assert "Rejected synthesis" not in rendered
    assert "must-not-render" not in rendered
    assert "per_conformer_records" not in rendered


def test_empty_final_hosted_response_keeps_protocol_error_and_prior_progress(monkeypatch):
    shown = []
    monkeypatch.setattr("IPython.display.display", lambda *items: shown.extend(items))
    completions = FakeCompletions(valid_responses() + [SimpleNamespace(choices=[])])

    with pytest.raises(demo_agent.ToolCallError, match="strict validation") as error:
        demo_agent.run_workflow(
            "Analyze.", VALID_API_KEY, display_events=True,
            client=fake_client(completions), executors={**fake_executors(), "build_workflow_report": lambda state: presentation_report()},
        )

    assert not isinstance(error.value, demo_agent.ConclusionValidationError)
    assert len(completions.calls) == 8
    rendered = "\n".join(getattr(item, "data", str(item)) for item in shown)
    assert "Nemotron plan" in rendered
    assert "figure-optimize_conformers_mmff94" in rendered
    assert "Workflow stopped" in rendered


def test_workflow_streams_plan_six_stages_then_compact_schema_checked_conclusion(monkeypatch):
    shown = []
    monkeypatch.setattr("IPython.display.display", lambda *items: shown.extend(items))
    completions = FakeCompletions(workflow_responses())
    result = demo_agent.run_workflow(
        "Analyze.", VALID_API_KEY, display_events=True,
        client=fake_client(completions), executors={**fake_executors(), "build_workflow_report": lambda state: presentation_report()},
    )

    rendered = "\n".join(getattr(item, "data", str(item)) for item in shown)
    assert rendered.count("## Nemotron plan") == 1
    heading = "## Schema-checked scientific conclusion"
    assert rendered.count(heading) == 1
    assert "Nemotron's qualitative interpretation is not automatically fact-verified" in rendered
    assert "Python checks the response structure before rendering" in rendered
    assert all(f"Nemotron → {stage}" in rendered for stage in STAGES)
    assert all(item.rationale in rendered for item in result.plan.stages)
    assert "radius" in rendered and "1024" in rendered
    assert "decision_basis" not in rendered
    assert VALID_ARGS["generate_morgan_fingerprints"]["decision_basis"] in rendered
    assert all(f"figure-{stage}" in rendered for stage in STAGES)
    assert result.conclusion.headline in rendered
    assert rendered.index("## Nemotron plan") < rendered.index("Nemotron → inspect_library")
    assert rendered.index("Nemotron → optimize_conformers_mmff94") < rendered.index(heading)
    assert "must-not-render" not in rendered
    assert "per_conformer_records" not in rendered
    assert "representative_eligibility" not in rendered
    assert '"raw_count":26' not in rendered
    assert "Canonical evidence" not in rendered
    assert "*Evidence:" not in rendered
    assert all(key not in rendered for key in ("E01", "E02", "E03", "E04", "E05", "E06"))


def test_progress_display_serializes_matplotlib_figure_as_png(monkeypatch):
    from IPython.display import Image
    from matplotlib.figure import Figure

    shown = []
    monkeypatch.setattr("IPython.display.display", lambda *items: shown.extend(items))
    figure = Figure(figsize=(2, 1))
    figure.subplots().plot([0, 1], [0, 1])
    result = StageResult(
        "generate_morgan_fingerprints",
        "nvMolKit MorganFingerprintGenerator",
        {"molecule_count": 2},
        (figure,),
    )
    arguments = demo_agent.FingerprintArgs(radius=2, size=1024, decision_basis="compact")

    demo_agent._display_progress_event("stage", {"result": result, "arguments": arguments})

    assert isinstance(shown[-1], Image)
    assert shown[-1].data.startswith(b"\x89PNG\r\n\x1a\n")


def test_mid_loop_executor_failure_keeps_prior_events_and_marks_failure(monkeypatch):
    shown = []
    monkeypatch.setattr("IPython.display.display", lambda *items: shown.extend(items))
    executors = fake_executors()
    executors["measure_tanimoto_similarity"] = lambda state: (_ for _ in ()).throw(RuntimeError("secret"))

    with pytest.raises(demo_agent.ToolCallError, match="executor failed"):
        demo_agent.run_workflow(
            "Analyze.", VALID_API_KEY, display_events=True,
            client=fake_client(FakeCompletions(workflow_responses())), executors=executors,
        )

    rendered = "\n".join(getattr(item, "data", str(item)) for item in shown)
    assert "Nemotron plan" in rendered
    assert "figure-generate_morgan_fingerprints" in rendered
    assert "figure-measure_tanimoto_similarity" not in rendered
    assert "Workflow stopped" in rendered


def test_display_events_false_performs_no_display_calls(monkeypatch):
    shown = []
    monkeypatch.setattr("IPython.display.display", lambda *items: shown.extend(items))

    demo_agent.run_workflow(
        "Analyze.", VALID_API_KEY, display_events=False,
        client=fake_client(FakeCompletions(workflow_responses())),
        executors={**fake_executors(), "build_workflow_report": lambda state: presentation_report()},
    )

    assert shown == []


def test_progress_callback_failure_stops_before_the_next_hosted_turn():
    completions = FakeCompletions(valid_responses())

    def fail_on_first_stage(event, payload):
        if event == "stage":
            raise RuntimeError("display backend unavailable")

    with pytest.raises(demo_agent.ToolCallError, match="Local progress display failed"):
        demo_agent.run_scientific_loop(
            "Analyze.", VALID_API_KEY, client=fake_client(completions),
            executors=fake_executors(), progress_callback=fail_on_first_stage,
        )

    assert len(completions.calls) == 2
