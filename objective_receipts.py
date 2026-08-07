"""Deterministic display receipts for state-bound objective selections."""

from dataclasses import dataclass

from demo_agent import ObjectiveSelection
from objective_challenge import (
    ObjectiveActionMenu,
    ObjectiveAttempt,
    ObjectiveContext,
    ObjectiveSwap,
    evaluate_selected_swap,
    resolve_menu_action,
)


@dataclass(frozen=True)
class ObjectiveReceipt:
    status: str
    validated_selection: str
    planned_command: str
    python_evaluation: str
    executed_measurement: str | None = None


def objective_receipt(
    context: ObjectiveContext,
    selection: ObjectiveSelection,
    menu: ObjectiveActionMenu,
    action: ObjectiveSwap,
    result: ObjectiveAttempt | None = None,
) -> ObjectiveReceipt:
    """Render an exact menu-bound selection without model-authored prose."""
    if type(context) is not ObjectiveContext:
        raise ValueError("Receipt context requires the exact objective domain record.")
    if type(selection) is not ObjectiveSelection:
        raise ValueError("Selection does not match the objective schema.")
    if type(menu) is not ObjectiveActionMenu or type(action) is not ObjectiveSwap:
        raise ValueError("Receipt provenance requires exact objective domain records.")
    try:
        resolved = resolve_menu_action(
            context,
            menu,
            state_id=selection.state_id,
            swap_id=selection.swap_id,
            observed_limiting_pairs=tuple(
                tuple(pair) for pair in selection.observed_limiting_pairs
            ),
            decision_rule=selection.decision_rule,
        )
    except ValueError:
        raise ValueError("Selection does not match the exact deterministic action menu.")
    if resolved != action or action not in menu.actions:
        raise ValueError("Selection does not match the exact deterministic action menu.")
    executed_measurement = None
    status = "validated"
    if result is not None:
        if type(result) is not ObjectiveAttempt:
            raise ValueError("Executed objective result must use the exact domain type.")
        expected_result = evaluate_selected_swap(
            context, menu, action, menu.accepted_attempt_count + 1
        )
        if result != expected_result:
            raise ValueError("Executed objective result does not match the exact menu action.")
        status = "executed"
        executed_measurement = (
            "measurement = PanelMeasurement("
            f"selected_ids={result.selected_ids!r}, score={result.score!r}, "
            f"score_key={result.score_key!r}, limiting_pairs={result.limiting_pairs!r}, "
            f"achieved={result.achieved!r})"
        )
    return ObjectiveReceipt(
        status=status,
        validated_selection=(
            "select_next_panel_swap("
            f"state_id={selection.state_id!r}, swap_id={selection.swap_id!r}, "
            "decision_rule='maximize_predicted_minimum_distance')"
        ),
        planned_command=(
            "selected_action = next(action for action in pending_action_menu.actions "
            f"if action.swap_id == {action.swap_id!r})"
        ),
        python_evaluation=(
            "result = evaluate_selected_swap(\n"
            "    context,\n"
            "    pending_action_menu,\n"
            "    selected_action,\n"
            "    accepted_attempt_count + 1,\n"
            ")"
        ),
        executed_measurement=executed_measurement,
    )
