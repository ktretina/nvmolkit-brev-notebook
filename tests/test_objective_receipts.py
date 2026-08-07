import inspect
from dataclasses import FrozenInstanceError, replace

import pytest

import demo_agent
from objective_challenge import (
    ObjectiveActionMenu,
    ObjectiveSwap,
    accepted_maxima,
    build_action_menu,
    evaluate_selected_swap,
    measure_panel,
)
from objective_fixtures import controlled_context_with_ranked_swaps
from objective_receipts import ObjectiveReceipt, objective_receipt


def records():
    context = controlled_context_with_ranked_swaps()
    source = measure_panel(context, context.baseline_ids)
    menu = build_action_menu(context, source, 0)
    action = accepted_maxima(menu)[0]
    selection = demo_agent.ObjectiveSelection(
        state_id=menu.state_id,
        swap_id=action.swap_id,
        observed_limiting_pairs=[list(pair) for pair in menu.source.limiting_pairs],
        decision_rule="maximize_predicted_minimum_distance",
    )
    return context, menu, action, selection


def test_objective_receipt_is_state_bound_rationale_free_and_deterministic():
    _context, menu, action, selection = records()
    first = objective_receipt(_context, selection, menu, action)
    second = objective_receipt(_context, selection, menu, action)

    assert first == second == ObjectiveReceipt(
        status="validated_selection",
        validated_selection=(
            "ObjectiveSelection("
            f"state_id={selection.state_id!r}, swap_id={selection.swap_id!r}, "
            f"observed_limiting_pairs={menu.source.limiting_pairs!r}, "
            f"decision_rule={selection.decision_rule!r})"
        ),
        planned_command=(
            "select_next_panel_swap("
            f"state_id={menu.state_id!r}, swap_id={action.swap_id!r}, "
            "decision_rule='maximize_predicted_minimum_distance')"
        ),
        python_evaluation=(
            "result = evaluate_selected_swap(\n"
            "    context,\n"
            "    pending_action_menu,\n"
            "    selected_action,\n"
            "    accepted_attempt_count + 1,\n"
            ")"
        ),
        executed_measurement=None,
    )
    rendered = repr(first)
    assert "decision_basis" not in rendered
    assert "rationale" not in rendered
    assert "selected_ids" not in rendered


def test_objective_receipt_validates_and_renders_exact_executed_measurement():
    context, menu, action, selection = records()
    result = evaluate_selected_swap(context, menu, action, 1)

    receipt = objective_receipt(context, selection, menu, action, result)

    assert receipt.status == "measured"
    assert receipt.executed_measurement == (
        "measurement = PanelMeasurement("
        f"selected_ids={result.selected_ids!r}, score={result.score!r}, "
        f"score_key={result.score_key!r}, limiting_pairs={result.limiting_pairs!r}, "
        f"achieved={result.achieved!r})"
    )
    with pytest.raises(ValueError, match="result"):
        objective_receipt(context, selection, menu, action, replace(result, score=0.123))


def test_objective_receipt_evaluation_matches_domain_signature_and_executes():
    context, menu, action, selection = records()
    receipt = objective_receipt(context, selection, menu, action)
    signature = inspect.signature(evaluate_selected_swap)
    assert list(signature.parameters) == ["context", "menu", "action", "attempt_number"]
    namespace = {
        "context": context,
        "pending_action_menu": menu,
        "selected_action": action,
        "accepted_attempt_count": 0,
        "evaluate_selected_swap": evaluate_selected_swap,
    }
    exec(receipt.python_evaluation, namespace)
    assert namespace["result"] == evaluate_selected_swap(context, menu, action, 1)


@pytest.mark.parametrize(
    "mutator, match",
    [
        (lambda s, m, a: (s.model_copy(update={"state_id": "state-0000000000000000"}), m, a), "menu"),
        (lambda s, m, a: (s.model_copy(update={"swap_id": "mol-0->mol-7"}), m, a), "menu"),
        (lambda s, m, a: (s.model_copy(update={"observed_limiting_pairs": [["mol-2", "mol-3"]]}), m, a), "menu"),
        (lambda s, m, a: (s, replace(m, state_id="state-0000000000000000"), a), "menu"),
        (lambda s, m, a: (s, m, replace(a, swap_id="mol-0->mol-7")), "menu"),
    ],
)
def test_objective_receipt_rejects_stale_or_forged_provenance(mutator, match):
    _context, menu, action, selection = records()
    forged = mutator(selection, menu, action)
    with pytest.raises(ValueError, match=match):
        objective_receipt(_context, *forged)


def test_objective_receipt_rejects_nonmax_displayed_action():
    _context, menu, _action, _selection = records()
    lower = next(action for action in menu.actions if action not in accepted_maxima(menu))
    selection = demo_agent.ObjectiveSelection(
        state_id=menu.state_id,
        swap_id=lower.swap_id,
        observed_limiting_pairs=[list(pair) for pair in menu.source.limiting_pairs],
        decision_rule="maximize_predicted_minimum_distance",
    )
    with pytest.raises(ValueError, match="menu"):
        objective_receipt(_context, selection, menu, lower)


def test_objective_receipt_requires_exact_types_and_rejects_legacy_shape():
    _context, menu, action, selection = records()

    class SelectionSubclass(demo_agent.ObjectiveSelection):
        pass

    class MenuSubclass(ObjectiveActionMenu):
        pass

    class SwapSubclass(ObjectiveSwap):
        pass

    with pytest.raises(ValueError, match="schema"):
        objective_receipt(_context, SelectionSubclass(**selection.model_dump()), menu, action)
    with pytest.raises(ValueError, match="domain"):
        objective_receipt(_context, selection, MenuSubclass(**menu.__dict__), action)
    with pytest.raises(ValueError, match="domain"):
        objective_receipt(_context, selection, menu, SwapSubclass(**action.__dict__))
    with pytest.raises((TypeError, ValueError)):
        objective_receipt(_context, {"selected_ids": list(action.resulting_ids), "decision_basis": "model prose"}, menu, action)


def test_objective_receipt_is_frozen():
    _context, menu, action, selection = records()
    receipt = objective_receipt(_context, selection, menu, action)
    with pytest.raises(FrozenInstanceError):
        receipt.validated_proposal = "changed"
