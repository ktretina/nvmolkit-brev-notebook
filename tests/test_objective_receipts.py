import inspect
from dataclasses import FrozenInstanceError, replace

import pytest

from demo_agent import ObjectiveProposal
from objective_challenge import ObjectiveAttempt, ObjectiveSwap, evaluate_diverse_panel
from objective_receipts import ObjectiveReceipt, objective_receipt


def proposal() -> ObjectiveProposal:
    return ObjectiveProposal(
        selected_ids=["mol-0", "mol-2", "mol-5", "mol-7"],
        decision_basis="Replace the limiting analogue.",
    )


def selected_swap() -> ObjectiveSwap:
    return ObjectiveSwap(
        replace_id="mol-1",
        replacement_id="mol-5",
        resulting_ids=("mol-0", "mol-5", "mol-2", "mol-3"),
        predicted_score=0.80,
        score_delta=0.45,
        limiting_pair=("mol-0", "mol-2"),
    )


def swapped_proposal() -> ObjectiveProposal:
    return ObjectiveProposal(
        selected_ids=["mol-0", "mol-5", "mol-2", "mol-3"],
        decision_basis="Replace the member of the measured limiting pair.",
    )


def prior_attempt() -> ObjectiveAttempt:
    return ObjectiveAttempt(
        attempt_number=1,
        selected_ids=("mol-0", "mol-1", "mol-2", "mol-3"),
        decision_basis="Measure the initial panel.",
        score=0.35,
        limiting_pair=("mol-0", "mol-1"),
        constraints_passed=True,
        achieved=False,
    )


def test_objective_receipt_displays_validated_ids_and_fixed_executor():
    receipt = objective_receipt(proposal())

    assert receipt == ObjectiveReceipt(
        validated_proposal=(
            "select_diverse_panel(selected_ids=['mol-0', 'mol-2', 'mol-5', 'mol-7'])"
        ),
        validated_intervention=None,
        python_evaluation=(
            "result = evaluate_diverse_panel(\n"
            "    context,\n"
            "    tuple(proposal.selected_ids),\n"
            "    attempt_number=1,\n"
            "    decision_basis=proposal.decision_basis,\n"
            "    selected_swap=None,\n"
            ")"
        ),
    )


def test_objective_receipt_renders_exact_selected_intervention_without_scores_or_model_text():
    receipt = objective_receipt(swapped_proposal(), selected_swap(), prior_attempt())

    assert receipt.validated_intervention == (
        "replace molecule_id='mol-1' with molecule_id='mol-5'"
    )
    assert "0.80" not in receipt.validated_intervention
    assert "0.45" not in receipt.validated_intervention
    assert "limiting" not in receipt.validated_intervention
    assert "measured limiting pair" not in receipt.validated_intervention


def test_objective_receipt_python_evaluation_matches_controller_evaluator_signature():
    receipt = objective_receipt(proposal())
    signature = inspect.signature(evaluate_diverse_panel)
    keyword_names = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    }

    assert "    context," in receipt.python_evaluation
    assert "    tuple(proposal.selected_ids)," in receipt.python_evaluation
    assert {"attempt_number", "decision_basis", "selected_swap"} == keyword_names
    assert all(f"    {name}=" in receipt.python_evaluation for name in keyword_names)
    assert "candidate_pool" not in receipt.python_evaluation
    assert "similarity_matrix" not in receipt.python_evaluation


def test_objective_receipt_rejects_wrong_exact_swap_type():
    class SwapSubclass(ObjectiveSwap):
        pass

    with pytest.raises(ValueError, match="exact swap"):
        objective_receipt(
            swapped_proposal(), SwapSubclass(**selected_swap().__dict__), prior_attempt()
        )


def test_objective_receipt_requires_a_canonical_prior_for_revisions():
    class AttemptSubclass(ObjectiveAttempt):
        pass

    with pytest.raises(ValueError, match="prior"):
        objective_receipt(swapped_proposal(), selected_swap())
    with pytest.raises(ValueError, match="exact"):
        objective_receipt(
            swapped_proposal(), selected_swap(), AttemptSubclass(**prior_attempt().__dict__)
        )
    with pytest.raises(ValueError, match="prior"):
        objective_receipt(
            swapped_proposal(), selected_swap(), replace(prior_attempt(), achieved=True)
        )


@pytest.mark.parametrize(
    "swap",
    [
        ObjectiveSwap(
            replace_id="mol-1",
            replacement_id="mol-5",
            resulting_ids=("mol-0", "mol-5", "mol-2", "mol-7"),
            predicted_score=0.80,
            score_delta=0.45,
            limiting_pair=("mol-0", "mol-2"),
        ),
        ObjectiveSwap(
            replace_id="mol-1",
            replacement_id="mol-5",
            resulting_ids=("mol-0", "mol-1", "mol-5", "mol-2"),
            predicted_score=0.80,
            score_delta=0.45,
            limiting_pair=("mol-0", "mol-2"),
        ),
    ],
)
def test_objective_receipt_rejects_false_swap_panel_or_id_provenance(swap):
    with pytest.raises(ValueError, match="provenance"):
        objective_receipt(swapped_proposal(), swap, prior_attempt())


@pytest.mark.parametrize(
    "swap, prior",
    [
        (
            selected_swap(),
            replace(
                prior_attempt(),
                selected_ids=("mol-0", "mol-4", "mol-2", "mol-3"),
            ),
        ),
        (replace(selected_swap(), score_delta=0.40), prior_attempt()),
    ],
)
def test_objective_receipt_rejects_forged_predecessor_or_score_delta(swap, prior):
    with pytest.raises(ValueError, match="provenance"):
        objective_receipt(swapped_proposal(), swap, prior)


def test_objective_receipt_excludes_decision_text_and_is_deterministic():
    first = objective_receipt(proposal())
    second = objective_receipt(proposal())

    assert first == second
    assert "Replace the limiting analogue" not in repr(first)


def test_objective_receipt_rejects_wrong_exact_type():
    class ProposalSubclass(ObjectiveProposal):
        pass

    with pytest.raises(ValueError, match="objective schema"):
        objective_receipt(ProposalSubclass(**proposal().model_dump()))


def test_objective_receipt_is_frozen():
    receipt = objective_receipt(proposal())
    with pytest.raises(FrozenInstanceError):
        receipt.validated_proposal = "changed"
