from dataclasses import FrozenInstanceError

import pytest

from demo_agent import ObjectiveProposal
from objective_challenge import ObjectiveSwap
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


def test_objective_receipt_displays_validated_ids_and_fixed_executor():
    receipt = objective_receipt(proposal())

    assert receipt == ObjectiveReceipt(
        validated_proposal=(
            "select_diverse_panel(selected_ids=['mol-0', 'mol-2', 'mol-5', 'mol-7'])"
        ),
        validated_intervention=None,
        python_evaluation=(
            "result = evaluate_diverse_panel(\n"
            "    selected_ids=proposal.selected_ids,\n"
            "    candidate_pool=candidate_pool,\n"
            "    similarity_matrix=similarity_matrix,\n"
            ")"
        ),
    )


def test_objective_receipt_renders_exact_selected_intervention_without_scores_or_model_text():
    receipt = objective_receipt(swapped_proposal(), selected_swap())

    assert receipt.validated_intervention == (
        "replace molecule_id='mol-1' with molecule_id='mol-5'"
    )
    assert "0.80" not in receipt.validated_intervention
    assert "0.45" not in receipt.validated_intervention
    assert "limiting" not in receipt.validated_intervention
    assert "measured limiting pair" not in receipt.validated_intervention


def test_objective_receipt_rejects_wrong_exact_swap_type():
    class SwapSubclass(ObjectiveSwap):
        pass

    with pytest.raises(ValueError, match="swap schema"):
        objective_receipt(swapped_proposal(), SwapSubclass(**selected_swap().__dict__))


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
        objective_receipt(swapped_proposal(), swap)


def test_objective_receipt_excludes_decision_text_and_is_deterministic():
    first = objective_receipt(proposal())
    second = objective_receipt(proposal())

    assert first == second
    assert "decision_basis" not in repr(first)
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
