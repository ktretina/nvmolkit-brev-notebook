"""Deterministic display receipts for state-bound objective selections."""

from dataclasses import dataclass

from demo_agent import ObjectiveSelection
from objective_challenge import ObjectiveActionMenu, ObjectiveSwap, accepted_maxima


@dataclass(frozen=True)
class ObjectiveReceipt:
    validated_proposal: str
    validated_intervention: str
    python_evaluation: str


def objective_receipt(
    selection: ObjectiveSelection,
    menu: ObjectiveActionMenu,
    action: ObjectiveSwap,
) -> ObjectiveReceipt:
    """Render an exact menu-bound selection without model-authored prose."""
    if type(selection) is not ObjectiveSelection:
        raise ValueError("Selection does not match the objective schema.")
    if type(menu) is not ObjectiveActionMenu or type(action) is not ObjectiveSwap:
        raise ValueError("Receipt provenance requires exact objective domain records.")
    expected_pairs = [list(pair) for pair in menu.source.limiting_pairs]
    if (
        selection.state_id != menu.state_id
        or selection.swap_id != action.swap_id
        or selection.observed_limiting_pairs != expected_pairs
        or selection.decision_rule != "maximize_predicted_minimum_distance"
        or action not in menu.actions
        or action not in accepted_maxima(menu)
        or not all(action.replace_id in pair for pair in menu.source.limiting_pairs)
    ):
        raise ValueError("Selection does not match the exact deterministic action menu.")
    return ObjectiveReceipt(
        validated_proposal=(
            "select_next_panel_swap("
            f"state_id={selection.state_id!r}, swap_id={selection.swap_id!r}, "
            "decision_rule='maximize_predicted_minimum_distance')"
        ),
        validated_intervention=(
            f"replace molecule_id={action.replace_id!r} "
            f"with molecule_id={action.replacement_id!r}"
        ),
        python_evaluation=(
            "result = evaluate_selected_swap(\n"
            "    context,\n"
            "    pending_action_menu,\n"
            "    selected_action,\n"
            "    accepted_attempt_count + 1,\n"
            ")"
        ),
    )
