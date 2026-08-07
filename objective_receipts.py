"""Deterministic display receipts for validated objective proposals."""

from dataclasses import dataclass
from math import isfinite

from demo_agent import ObjectiveProposal, ObjectiveSelection
from objective_challenge import ObjectiveAttempt, ObjectiveSwap, is_strict_improvement


@dataclass(frozen=True)
class ObjectiveReceipt:
    validated_proposal: str
    validated_intervention: str | None
    python_evaluation: str


def objective_receipt(
    proposal: ObjectiveProposal | ObjectiveSelection,
    selected_swap: ObjectiveSwap | None = None,
    prior_attempt: ObjectiveAttempt | None = None,
) -> ObjectiveReceipt:
    """Render the hosted proposal and the fixed evaluator without model prose."""
    if type(proposal) is ObjectiveSelection:
        if type(selected_swap) is not ObjectiveSwap or proposal.swap_id != selected_swap.swap_id:
            raise ValueError("Selection does not match the deterministic objective action.")
        return ObjectiveReceipt(
            validated_proposal=(
                "select_next_panel_swap("
                f"state_id={proposal.state_id!r}, swap_id={proposal.swap_id!r})"
            ),
            validated_intervention=(
                f"replace molecule_id={selected_swap.replace_id!r} "
                f"with molecule_id={selected_swap.replacement_id!r}"
            ),
            python_evaluation=(
                "result = evaluate_selected_swap(\n"
                "    context,\n"
                "    self.pending_action_menu,\n"
                "    self.pending_objective_swap,\n"
                "    self.accepted_attempt_count + 1,\n"
                ")"
            ),
        )
    if type(proposal) is not ObjectiveProposal:
        raise ValueError("Proposal does not match the objective schema.")
    selected_ids = repr(list(proposal.selected_ids))
    intervention = None
    if selected_swap is None and prior_attempt is None:
        pass
    elif type(selected_swap) is not ObjectiveSwap or type(prior_attempt) is not ObjectiveAttempt:
        raise ValueError("Intervention provenance requires exact swap and prior attempt records.")
    else:
        if (
            prior_attempt.constraints_passed is not True
            or prior_attempt.achieved is not False
        ):
            raise ValueError("Intervention provenance requires a prior attempt below target.")
        if not _is_canonical_panel(prior_attempt.selected_ids):
            raise ValueError("Intervention provenance has an invalid prior panel.")
        if not _is_finite_float(prior_attempt.score):
            raise ValueError("Intervention provenance has an invalid prior score.")
        if (
            type(selected_swap.replace_id) is not str
            or type(selected_swap.replacement_id) is not str
            or not selected_swap.replace_id
            or not selected_swap.replacement_id
            or selected_swap.replace_id == selected_swap.replacement_id
            or not _is_canonical_panel(selected_swap.resulting_ids)
            or not _is_finite_float(selected_swap.predicted_score)
            or not _is_finite_float(selected_swap.score_delta)
            or not is_strict_improvement(
                selected_swap.predicted_score, prior_attempt.score
            )
        ):
            raise ValueError("Intervention provenance has invalid canonical values.")
        resulting_ids = selected_swap.resulting_ids
        proposal_ids = tuple(proposal.selected_ids)
        if (
            not _same_panel(resulting_ids, proposal_ids)
            or selected_swap.replacement_id not in resulting_ids
            or selected_swap.replace_id in resulting_ids
        ):
            raise ValueError("Intervention provenance does not match the selected panel.")
        replacement_position = resulting_ids.index(selected_swap.replacement_id)
        predecessor = (
            resulting_ids[:replacement_position]
            + (selected_swap.replace_id,)
            + resulting_ids[replacement_position + 1 :]
        )
        if (
            not _same_panel(predecessor, prior_attempt.selected_ids)
            or selected_swap.score_delta
            != selected_swap.predicted_score - prior_attempt.score
        ):
            raise ValueError("Intervention provenance does not match the prior measurement.")
        intervention = (
            f"replace molecule_id={selected_swap.replace_id!r} "
            f"with molecule_id={selected_swap.replacement_id!r}"
        )
    return ObjectiveReceipt(
        validated_proposal=f"select_diverse_panel(selected_ids={selected_ids})",
        validated_intervention=intervention,
        python_evaluation=(
            "result = evaluate_diverse_panel(\n"
            "    context,\n"
            "    tuple(proposal.selected_ids),\n"
            "    attempt_number=len(self.objective_attempts) + 1,\n"
            "    decision_basis=proposal.decision_basis,\n"
            "    selected_swap=self.pending_objective_swap,\n"
            ")"
        ),
    )


def _is_finite_float(value: object) -> bool:
    return type(value) is float and isfinite(value)


def _is_canonical_panel(value: object) -> bool:
    return (
        type(value) is tuple
        and len(value) == 4
        and len(set(value)) == len(value)
        and all(type(molecule_id) is str for molecule_id in value)
    )


def _same_panel(first: tuple[str, ...], second: tuple[str, ...]) -> bool:
    return len(first) == len(second) and set(first) == set(second)
