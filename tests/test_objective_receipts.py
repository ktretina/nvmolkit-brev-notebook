from dataclasses import FrozenInstanceError

import pytest

from demo_agent import ObjectiveProposal
from objective_receipts import ObjectiveReceipt, objective_receipt


def proposal() -> ObjectiveProposal:
    return ObjectiveProposal(
        selected_ids=["mol-0", "mol-2", "mol-5", "mol-7"],
        decision_basis="Replace the limiting analogue.",
    )


def test_objective_receipt_displays_validated_ids_and_fixed_executor():
    receipt = objective_receipt(proposal())

    assert receipt == ObjectiveReceipt(
        validated_proposal=(
            "select_diverse_panel(selected_ids=['mol-0', 'mol-2', 'mol-5', 'mol-7'])"
        ),
        python_evaluation=(
            "result = evaluate_diverse_panel(\n"
            "    selected_ids=proposal.selected_ids,\n"
            "    candidate_pool=candidate_pool,\n"
            "    similarity_matrix=similarity_matrix,\n"
            ")"
        ),
    )


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
