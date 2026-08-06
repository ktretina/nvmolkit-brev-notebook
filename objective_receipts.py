"""Deterministic display receipts for validated objective proposals."""

from dataclasses import dataclass

from demo_agent import ObjectiveProposal


@dataclass(frozen=True)
class ObjectiveReceipt:
    validated_proposal: str
    python_evaluation: str


def objective_receipt(proposal: ObjectiveProposal) -> ObjectiveReceipt:
    """Render the hosted proposal and the fixed evaluator without model prose."""
    if type(proposal) is not ObjectiveProposal:
        raise ValueError("Proposal does not match the objective schema.")
    selected_ids = repr(list(proposal.selected_ids))
    return ObjectiveReceipt(
        validated_proposal=f"select_diverse_panel(selected_ids={selected_ids})",
        python_evaluation=(
            "result = evaluate_diverse_panel(\n"
            "    selected_ids=proposal.selected_ids,\n"
            "    candidate_pool=candidate_pool,\n"
            "    similarity_matrix=similarity_matrix,\n"
            ")"
        ),
    )
