#!/usr/bin/env python3
"""Fail-closed hosted objective and end-to-end reliability qualification."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import demo_agent
from chemistry_workflow import WorkflowPhase
from objective_challenge import (
    accepted_maxima,
    build_action_menu,
    build_objective_evidence,
    measure_panel,
    target_is_achieved,
)


ControllerFactory = Callable[[], demo_agent.BoundedWorkflowController]
QUALIFICATION_GOAL = (
    "Qualify the fixed nvMolKit workflow and bounded molecular-diversity objective."
)


class _AuditedCompletions:
    """Record only hosted request contract fields, never messages or responses."""

    def __init__(self, delegate: object) -> None:
        self._delegate = delegate
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append({
            "model": kwargs.get("model"),
            "temperature": kwargs.get("temperature"),
            "extra_body": deepcopy(kwargs.get("extra_body")),
            "tool_choice": deepcopy(kwargs.get("tool_choice")),
            "tools": deepcopy(kwargs.get("tools")),
        })
        return self._delegate.create(**kwargs)


def _audited_client(api_key: str) -> object:
    client = demo_agent._client(api_key)
    completions = _AuditedCompletions(client.chat.completions)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


class _DeterministicScientificCompletions:
    """Produce the fixed scientific preparation transcript without hosted inference."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.names = ("submit_workflow_plan", *demo_agent.STAGES)
        self.arguments = ({
            "stages": [
                {
                    "stage": stage,
                    "rationale": f"Run {stage.replace('_', ' ')} after its prerequisite.",
                }
                for stage in demo_agent.STAGES
            ]
        }, {
        }, {
            "radius": 2,
            "size": 1024,
            "decision_basis": "Use the fixed qualification fingerprint parameters.",
        }, {
        }, {
            "cutoff": 0.4,
            "decision_basis": "Use the fixed qualification clustering cutoff.",
        }, {
            "representative_count": 4,
            "policy": "largest_clusters_first",
            "conformers_per_representative": 5,
            "decision_basis": "Use the fixed qualification conformer sample.",
        }, {})

    def create(self, **kwargs: object) -> object:
        index = len(self.calls)
        if index >= len(self.names):
            raise RuntimeError("Deterministic preparation exceeded the fixed stage count.")
        name = self.names[index]
        if (
            kwargs.get("model") != demo_agent.DEFAULT_MODEL
            or kwargs.get("temperature") != 0.0
            or kwargs.get("extra_body") != demo_agent.NEMOTRON_TOOL_EXTRA_BODY
            or kwargs.get("tool_choice") != {
                "type": "function", "function": {"name": name}
            }
        ):
            raise RuntimeError("Deterministic preparation request contract changed.")
        tools = kwargs.get("tools")
        if (
            type(tools) is not list
            or len(tools) != 1
            or tools[0]["function"].get("name") != name
            or tools[0]["function"].get("strict") is not True
        ):
            raise RuntimeError("Deterministic preparation tool contract changed.")
        self.calls.append(dict(kwargs))
        call = SimpleNamespace(
            id=f"prepared-{index}",
            type="function",
            function=SimpleNamespace(
                name=name,
                arguments=json.dumps(self.arguments[index], separators=(",", ":")),
            ),
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=None, tool_calls=[call]
        ))])


def _build_prepared_snapshot(api_key: str) -> demo_agent.PreparedScientificSnapshot:
    completions = _DeterministicScientificCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    controller = demo_agent.BoundedWorkflowController.create(
        QUALIFICATION_GOAL,
        api_key,
        client=client,
        objective_required=False,
    )
    result = demo_agent._complete_scientific_loop(controller, None)
    if len(completions.calls) != 7:
        raise RuntimeError("Scientific preparation did not execute exactly seven calls.")
    return demo_agent.PreparedScientificSnapshot(
        messages=result.messages,
        state=controller.session.state,
        plan=result.plan,
        stage_results=result.stage_results,
        report=result.report,
        turn_count=result.turn_count,
    )


@dataclass(frozen=True)
class ReliabilityReceipt:
    requested_trials: int
    completed_trials: int
    argmax_successes: int
    clean_first_request_trials: int
    retry_assisted_trials: int
    requested_end_to_end_runs: int
    completed_end_to_end_runs: int
    message_pairing_passes: int
    claim_safety_passes: int
    production_temperature_zero: bool
    objective_trials: tuple[dict[str, object], ...]
    end_to_end_runs: tuple[dict[str, object], ...]
    failed_trials: tuple[dict[str, object], ...]


_TRIAL_RECEIPT_FIELDS = (
    "kind",
    "index",
    "model",
    "environment",
    "completed",
    "argmax_success",
    "accepted_attempt_count",
    "rejected_selection_count",
    "correction_prompts_sent",
    "selection_response_count",
    "provider_request_attempt_count",
    "retry_assisted",
    "baseline_score",
    "target_score",
    "final_score",
    "termination_reason",
    "message_pairing_passed",
    "claim_safety_passed",
    "production_temperature_zero",
    "conclusion_status",
)


def _allowlisted_trial(trial: dict[str, object]) -> dict[str, object]:
    if type(trial) is not dict:
        raise TypeError("Reliability trial records must be dictionaries.")
    return {field: trial.get(field) for field in _TRIAL_RECEIPT_FIELDS}


def write_reliability_receipt(
    path: Path | str,
    receipt: ReliabilityReceipt,
) -> None:
    """Write canonical receipt JSON rebuilt exclusively from explicit allowlists."""
    if type(receipt) is not ReliabilityReceipt:
        raise TypeError("An exact reliability receipt is required.")
    payload = {
        "model": demo_agent.DEFAULT_MODEL,
        "environment": "production_hosted_api",
        "requested_trials": receipt.requested_trials,
        "completed_trials": receipt.completed_trials,
        "argmax_successes": receipt.argmax_successes,
        "clean_first_request_trials": receipt.clean_first_request_trials,
        "retry_assisted_trials": receipt.retry_assisted_trials,
        "requested_end_to_end_runs": receipt.requested_end_to_end_runs,
        "completed_end_to_end_runs": receipt.completed_end_to_end_runs,
        "message_pairing_passes": receipt.message_pairing_passes,
        "claim_safety_passes": receipt.claim_safety_passes,
        "production_temperature_zero": receipt.production_temperature_zero,
        "objective_trials": [
            _allowlisted_trial(item) for item in receipt.objective_trials
        ],
        "end_to_end_runs": [
            _allowlisted_trial(item) for item in receipt.end_to_end_runs
        ],
        "failed_trials": [
            _allowlisted_trial(item) for item in receipt.failed_trials
        ],
    }
    Path(path).write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _positive_count(value: int, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer.")
    return value


def _messages_are_paired(messages: object) -> bool:
    if type(messages) is not list:
        return False
    assistant_ids: list[str] = []
    tool_ids: list[str] = []
    for message in messages:
        if type(message) is not dict:
            return False
        if message.get("role") == "assistant":
            calls = message.get("tool_calls")
            if type(calls) is not list or len(calls) != 1:
                return False
            call_id = calls[0].get("id") if type(calls[0]) is dict else None
            if type(call_id) is not str or not call_id:
                return False
            assistant_ids.append(call_id)
        elif message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            if type(call_id) is not str or not call_id:
                return False
            tool_ids.append(call_id)
    return assistant_ids == tool_ids and len(set(assistant_ids)) == len(assistant_ids)


def _calls_use_production_contract(controller: demo_agent.BoundedWorkflowController) -> bool:
    completions = getattr(getattr(controller.client, "chat", None), "completions", None)
    calls = getattr(completions, "calls", None)
    if type(calls) is not list or not calls:
        return False
    for call in calls:
        try:
            tools = call["tools"]
            choice = call["tool_choice"]
            extra_body = call["extra_body"]
        except (KeyError, IndexError, TypeError):
            return False
        if type(tools) is not list or len(tools) != 1:
            return False
        try:
            tool = tools[0]["function"]
            choice_name = choice["function"]["name"]
            thinking = extra_body["chat_template_kwargs"]["enable_thinking"]
        except (KeyError, IndexError, TypeError):
            return False
        if (
            call.get("model") != demo_agent.DEFAULT_MODEL
            or type(call.get("temperature")) is not float
            or call["temperature"] != 0.0
            or call.get("extra_body") != demo_agent.NEMOTRON_TOOL_EXTRA_BODY
            or thinking is not False
            or choice.get("type") != "function"
            or tool.get("strict") is not True
            or tool.get("name") != choice_name
        ):
            return False
    return True


def _argmax_trace_passes(controller: demo_agent.BoundedWorkflowController) -> bool:
    context = controller.objective_context
    if context is None or controller.rejected_selection_count != 0:
        return False
    current = measure_panel(context, context.baseline_ids)
    for index, attempt in enumerate(controller.objective_attempts):
        menu = build_action_menu(context, current, index)
        maxima = accepted_maxima(menu)
        selected = attempt.selected_swap
        if (
            selected is None
            or attempt.state_id != menu.state_id
            or selected.swap_id not in {item.swap_id for item in maxima}
        ):
            return False
        current = attempt.measurement
    return True


def _deterministic_claims_are_safe(controller: demo_agent.BoundedWorkflowController) -> bool:
    run = controller.objective_run
    evidence = controller.objective_evidence
    if run is None or evidence is None or evidence.key != "O01":
        return False
    try:
        expected = build_objective_evidence(run)
        payload = json.loads(evidence.payload_json)
    except Exception:
        return False
    if evidence != expected or type(payload) is not dict:
        return False
    forbidden = {"decision_basis", "rationale", "prose", "response", "metadata", "secret"}

    def keys(value: object) -> set[str]:
        if type(value) is dict:
            return {str(key).lower() for key in value} | set().union(
                *(keys(item) for item in value.values()), set()
            )
        if type(value) is list:
            return set().union(*(keys(item) for item in value), set())
        return set()

    return not (keys(payload) & forbidden)


def _drive_objective(controller: demo_agent.BoundedWorkflowController) -> None:
    controller.begin_objective_challenge()
    while controller.objective_run is None:
        try:
            selection = controller.request_objective_attempt()
        except demo_agent.ToolCallError:
            if controller.objective_transport_retry_pending:
                continue
            raise
        controller.execute_objective_attempt(selection)


def _trial_record(
    controller: demo_agent.BoundedWorkflowController,
    *,
    index: int,
    kind: str,
    conclusion_status: str | None = None,
) -> dict[str, object]:
    run = controller.objective_run
    context = controller.objective_context
    completed = bool(
        run is not None
        and context is not None
        and run.termination_reason.value == "target_achieved"
        and target_is_achieved(run.final_score, context.target_score)
    )
    record: dict[str, object] = {
        "kind": kind,
        "index": index,
        "model": demo_agent.DEFAULT_MODEL,
        "environment": "production_hosted_api",
        "completed": completed,
        "argmax_success": _argmax_trace_passes(controller),
        "accepted_attempt_count": controller.accepted_attempt_count,
        "rejected_selection_count": controller.rejected_selection_count,
        "correction_prompts_sent": controller.correction_prompts_sent,
        "selection_response_count": controller.selection_response_count,
        "provider_request_attempt_count": controller.provider_request_attempt_count,
        "retry_assisted": controller.objective_transport_retry_used,
        "baseline_score": None if context is None else context.baseline_score,
        "target_score": None if context is None else context.target_score,
        "final_score": None if run is None else run.final_score,
        "termination_reason": None if run is None else run.termination_reason.value,
        "message_pairing_passed": _messages_are_paired(controller.session.messages),
        "claim_safety_passed": _deterministic_claims_are_safe(controller),
        "production_temperature_zero": _calls_use_production_contract(controller),
        "conclusion_status": conclusion_status,
    }
    return record


def _failure_record(index: int, kind: str, reason: str) -> dict[str, object]:
    return {
        "kind": kind,
        "index": index,
        "model": demo_agent.DEFAULT_MODEL,
        "environment": "production_hosted_api",
        "completed": False,
        "argmax_success": False,
        "accepted_attempt_count": 0,
        "rejected_selection_count": 0,
        "correction_prompts_sent": 0,
        "selection_response_count": 0,
        "provider_request_attempt_count": 0,
        "retry_assisted": False,
        "baseline_score": None,
        "target_score": None,
        "final_score": None,
        "termination_reason": reason,
        "message_pairing_passed": False,
        "claim_safety_passed": False,
        "production_temperature_zero": False,
        "conclusion_status": None,
    }


def run_trials(
    controller_factory: ControllerFactory,
    *,
    trials: int,
) -> tuple[dict[str, object], ...]:
    """Run isolated objective trials, retaining only canonical receipt fields."""
    _positive_count(trials, "trials")
    records: list[dict[str, object]] = []
    for index in range(1, trials + 1):
        controller = None
        try:
            controller = controller_factory()
            if type(controller) is not demo_agent.BoundedWorkflowController:
                raise TypeError
            _drive_objective(controller)
            records.append(_trial_record(controller, index=index, kind="objective"))
        except Exception:
            records.append(
                _trial_record(controller, index=index, kind="objective")
                if type(controller) is demo_agent.BoundedWorkflowController
                else _failure_record(index, "objective", "trial_failed")
            )
    return tuple(records)


def run_end_to_end(
    controller_factory: ControllerFactory,
    *,
    runs: int,
) -> tuple[dict[str, object], ...]:
    """Run fresh plan-to-finding workflows with no retained scientific snapshot."""
    _positive_count(runs, "runs")
    records: list[dict[str, object]] = []
    for index in range(1, runs + 1):
        controller = None
        try:
            controller = controller_factory()
            if (
                type(controller) is not demo_agent.BoundedWorkflowController
                or controller.session.turn_count != 0
                or controller.session.state.phase is not WorkflowPhase.NEW
            ):
                raise ValueError
            demo_agent._complete_scientific_loop(controller, None)
            _drive_objective(controller)
            result = controller.request_synthesis()
            conclusion_status = getattr(result.conclusion, "finding_selection_status", None)
            record = _trial_record(
                controller,
                index=index,
                kind="end_to_end",
                conclusion_status=conclusion_status,
            )
            record["completed"] = bool(record["completed"] and conclusion_status)
            records.append(record)
        except Exception:
            if type(controller) is demo_agent.BoundedWorkflowController:
                record = _trial_record(controller, index=index, kind="end_to_end")
                record["completed"] = False
                records.append(record)
            else:
                records.append(_failure_record(index, "end_to_end", "end_to_end_failed"))
    return tuple(records)


def run_qualification(
    objective_factory: ControllerFactory,
    end_to_end_factory: ControllerFactory,
    *,
    trials: int,
    end_to_end_runs: int,
) -> ReliabilityReceipt:
    objective = run_trials(objective_factory, trials=trials)
    end_to_end = run_end_to_end(end_to_end_factory, runs=end_to_end_runs)
    combined = (*objective, *end_to_end)

    def qualifies(item: dict[str, object], *, conclusion_required: bool) -> bool:
        valid = (
            item.get("completed") is True
            and item.get("argmax_success") is True
            and item.get("message_pairing_passed") is True
            and item.get("claim_safety_passed") is True
            and item.get("production_temperature_zero") is True
        )
        if conclusion_required:
            valid &= item.get("conclusion_status") in {
                "selected", "finding_selection_unavailable"
            }
        return bool(valid)

    failed = (
        *(item for item in objective if not qualifies(item, conclusion_required=False)),
        *(item for item in end_to_end if not qualifies(item, conclusion_required=True)),
    )
    return ReliabilityReceipt(
        requested_trials=trials,
        completed_trials=sum(item["completed"] is True for item in objective),
        argmax_successes=sum(item["argmax_success"] is True for item in objective),
        clean_first_request_trials=sum(
            qualifies(item, conclusion_required=False)
            and item["retry_assisted"] is False
            for item in objective
        ),
        retry_assisted_trials=sum(
            qualifies(item, conclusion_required=False)
            and item["retry_assisted"] is True
            for item in objective
        ),
        requested_end_to_end_runs=end_to_end_runs,
        completed_end_to_end_runs=sum(item["completed"] is True for item in end_to_end),
        message_pairing_passes=sum(item["message_pairing_passed"] is True for item in combined),
        claim_safety_passes=sum(item["claim_safety_passed"] is True for item in combined),
        production_temperature_zero=all(
            item["production_temperature_zero"] is True for item in combined
        ),
        objective_trials=tuple(objective),
        end_to_end_runs=tuple(end_to_end),
        failed_trials=failed,
    )


def qualification_exit_code(receipt: ReliabilityReceipt) -> int:
    if type(receipt) is not ReliabilityReceipt:
        return 1
    expected_total = receipt.requested_trials + receipt.requested_end_to_end_runs
    expected_clean = sum(
        item.get("completed") is True
        and item.get("argmax_success") is True
        and item.get("message_pairing_passed") is True
        and item.get("claim_safety_passed") is True
        and item.get("production_temperature_zero") is True
        and item.get("retry_assisted") is False
        for item in receipt.objective_trials
    )
    expected_retry = sum(
        item.get("completed") is True
        and item.get("argmax_success") is True
        and item.get("message_pairing_passed") is True
        and item.get("claim_safety_passed") is True
        and item.get("production_temperature_zero") is True
        and item.get("retry_assisted") is True
        for item in receipt.objective_trials
    )
    valid = (
        receipt.completed_trials == receipt.requested_trials
        and receipt.argmax_successes == receipt.requested_trials
        and receipt.completed_end_to_end_runs == receipt.requested_end_to_end_runs
        and receipt.message_pairing_passes == expected_total
        and receipt.claim_safety_passes == expected_total
        and receipt.production_temperature_zero is True
        and not receipt.failed_trials
        and len(receipt.objective_trials) == receipt.requested_trials
        and len(receipt.end_to_end_runs) == receipt.requested_end_to_end_runs
        and receipt.clean_first_request_trials == expected_clean
        and receipt.retry_assisted_trials == expected_retry
        and expected_clean + expected_retry == receipt.requested_trials
    )
    valid &= all(
        item.get("completed") is True
        and item.get("argmax_success") is True
        and item.get("message_pairing_passed") is True
        and item.get("claim_safety_passed") is True
        and item.get("production_temperature_zero") is True
        and item.get("termination_reason") == "target_achieved"
        and type(item.get("accepted_attempt_count")) is int
        and 1 <= item["accepted_attempt_count"] <= 3
        and item.get("rejected_selection_count") == 0
        for item in receipt.objective_trials
    )
    valid &= all(
        item.get("completed") is True
        and item.get("argmax_success") is True
        and item.get("message_pairing_passed") is True
        and item.get("claim_safety_passed") is True
        and item.get("production_temperature_zero") is True
        and item.get("conclusion_status") in {"selected", "finding_selection_unavailable"}
        for item in receipt.end_to_end_runs
    )
    return 0 if valid else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--end-to-end-runs", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    _positive_count(args.trials, "trials")
    _positive_count(args.end_to_end_runs, "end-to-end-runs")
    api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    demo_agent._validate_api_key(api_key)
    snapshot = _build_prepared_snapshot(api_key)

    def objective_factory() -> demo_agent.BoundedWorkflowController:
        return demo_agent.clone_prepared_controller(
            snapshot,
            client=_audited_client(api_key),
            executors=demo_agent._default_executors(),
        )

    def end_to_end_factory() -> demo_agent.BoundedWorkflowController:
        return demo_agent.BoundedWorkflowController.create(
            QUALIFICATION_GOAL,
            api_key,
            client=_audited_client(api_key),
            objective_required=True,
        )

    receipt = run_qualification(
        objective_factory,
        end_to_end_factory,
        trials=args.trials,
        end_to_end_runs=args.end_to_end_runs,
    )
    write_reliability_receipt(args.output, receipt)
    return qualification_exit_code(receipt)


if __name__ == "__main__":
    raise SystemExit(main())
