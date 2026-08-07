"""Deterministic display receipts for validated objective proposals."""

from dataclasses import dataclass

from demo_agent import ObjectiveProposal
from objective_challenge import ObjectiveSwap


@dataclass(frozen=True)
class ObjectiveReceipt:
    validated_proposal: str
    validated_intervention: str | None
    python_evaluation: str


def objective_receipt(
    proposal: ObjectiveProposal,
    selected_swap: ObjectiveSwap | None = None,
) -> ObjectiveReceipt:
    """Render the hosted proposal and the fixed evaluator without model prose."""
    if type(proposal) is not ObjectiveProposal:
        raise ValueError("Proposal does not match the objective schema.")
    if selected_swap is not None and type(selected_swap) is not ObjectiveSwap:
        raise ValueError("Intervention does not match the objective swap schema.")
    selected_ids = repr(list(proposal.selected_ids))
    intervention = None
    if selected_swap is not None:
        if (
            type(selected_swap.replace_id) is not str
            or type(selected_swap.replacement_id) is not str
            or not selected_swap.replace_id
            or not selected_swap.replacement_id
            or selected_swap.replace_id == selected_swap.replacement_id
            or type(selected_swap.resulting_ids) is not tuple
            or any(type(molecule_id) is not str for molecule_id in selected_swap.resulting_ids)
        ):
            raise ValueError("Intervention provenance has invalid canonical IDs.")
        resulting_ids = selected_swap.resulting_ids
        proposal_ids = tuple(proposal.selected_ids)
        if (
            len(resulting_ids) != len(proposal_ids)
            or len(set(resulting_ids)) != len(resulting_ids)
            or set(resulting_ids) != set(proposal_ids)
            or selected_swap.replacement_id not in resulting_ids
            or selected_swap.replace_id in resulting_ids
        ):
            raise ValueError("Intervention provenance does not match the selected panel.")
        intervention = (
            f"replace molecule_id={selected_swap.replace_id!r} "
            f"with molecule_id={selected_swap.replacement_id!r}"
        )
    return ObjectiveReceipt(
        validated_proposal=f"select_diverse_panel(selected_ids={selected_ids})",
        validated_intervention=intervention,
        python_evaluation=(
            "result = evaluate_diverse_panel(\n"
            "    selected_ids=proposal.selected_ids,\n"
            "    candidate_pool=candidate_pool,\n"
            "    similarity_matrix=similarity_matrix,\n"
            ")"
        ),
    )
